"""
MPDOK — Python interface to the Fortran GMRES-IR kernel.

Drop-in replacement for scipy.sparse.linalg.gmres / cp.linalg.solve
for dense problems where FP64 accuracy at TF32 speed is needed.

Usage:
    from MPDOK.mpdok_ops import MPDOKSolver
    solver = MPDOKSolver()
    x = solver.solve(A, b)                     # default tol=1e-11, restart=50
    x = solver.solve(A, b, tol=1e-8)
    x = solver.solve(A, b, restart=100)        # wider subspace for harder problems

Memory:
    Device memory is used by default.  If VRAM is insufficient the solver
    automatically retries with CUDA unified memory (host+device paging),
    which removes the VRAM size ceiling at a modest bandwidth cost.
"""

import contextlib
import ctypes
import os
import warnings

import cupy as cp
import numpy as np


class _NullContext:
    """Context manager that does nothing — preserves CuPy's default allocator."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


class LUIRSolver:
    """FP32/TF32 LU factorization with FP64 iterative refinement.

    Uses cuSOLVER Sgetrf (tensor cores on Ampere+) for the inner factorization
    and cuBLAS Dgemv for FP64 residual computation.  Two outer iterations
    typically reach FP64 machine precision (~1e-14 relative residual).

    Compared to GMRES-IR:
      - Faster per-solve for well-conditioned systems (direct factor, no Krylov)
      - Memory: O(N^2) for LU factors + O(N^2) for original A = 2x matrix storage
      - No restart parameter; inner solve is exact given the FP32 factorization
      - Less robust on severely ill-conditioned problems (GMRES-IR is more flexible)
    """

    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'lu_ir.so'
            )
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"lu_ir.so not found at {lib_path}. "
                f"Run 'make' inside the MPDOK directory first."
            )
        self._lib = ctypes.CDLL(lib_path)
        self._setup_signatures()

    def _setup_signatures(self):
        self._lib.py_lu_ir.argtypes = [
            ctypes.c_void_p,                   # A  — FP64 device ptr (N×N, Fortran order)
            ctypes.c_void_p,                   # b  — FP64 device ptr (N,)
            ctypes.c_void_p,                   # x  — FP64 device ptr (N,), output
            ctypes.c_int,                      # N
            ctypes.c_double,                   # tol
            ctypes.c_int,                      # maxiter_outer
            ctypes.POINTER(ctypes.c_int),      # converged_out
        ]
        self._lib.py_lu_ir.restype = None

    def solve(self, A, b, tol=1e-11, maxiter_outer=3):
        """Solve Ax = b using LU-IR (FP32 LU + FP64 iterative refinement).

        Args:
            A:             (N, N) FP64 NumPy or CuPy array.
            b:             (N,) FP64 NumPy or CuPy array.
            tol:           Convergence on relative FP64 residual ||b-Ax||/||b||.
            maxiter_outer: Max refinement steps. Default 3; 2 almost always enough.

        Returns:
            x: FP64 CuPy (N,) solution array.

        Raises:
            RuntimeWarning: if the solver did not converge within maxiter_outer.
        """
        A_gpu = cp.asfortranarray(cp.asarray(A, dtype=cp.float64))
        b_gpu = cp.ascontiguousarray(cp.asarray(b, dtype=cp.float64))
        N = int(A_gpu.shape[0])

        if A_gpu.shape != (N, N):
            raise ValueError(f"A must be square (N×N), got {A_gpu.shape}")
        if b_gpu.shape != (N,):
            raise ValueError(f"b must be (N,), got {b_gpu.shape}")

        x_gpu = cp.zeros(N, dtype=cp.float64)
        converged = ctypes.c_int(0)

        self._lib.py_lu_ir(
            ctypes.c_void_p(A_gpu.data.ptr),
            ctypes.c_void_p(b_gpu.data.ptr),
            ctypes.c_void_p(x_gpu.data.ptr),
            ctypes.c_int(N),
            ctypes.c_double(tol),
            ctypes.c_int(maxiter_outer),
            ctypes.byref(converged),
        )
        cp.cuda.Stream.null.synchronize()

        if not converged.value:
            warnings.warn(
                f"LU-IR did not converge to tol={tol:.1e} within "
                f"{maxiter_outer} outer iterations.",
                RuntimeWarning, stacklevel=2,
            )

        return x_gpu


class MPDOKSolver:
    """Fortran GMRES-IR kernel wrapped as a SciPy-style solver.

    Loads mpdok.so once at construction; subsequent solve() calls have
    no Python-level overhead beyond pointer passing.

    The kernel runs entirely on-device:
      - A is uploaded to device (if not already there) and converted to FP32
      - All Krylov vectors live on device in FP32
      - FP64 residual computed on-device each outer iteration
      - Solution returned as FP64 CuPy array

    Accuracy:
      Typically converges to ~1e-13 relative residual in 2 outer iterations.
      Set tol down to ~1e-13 safely; below that you are near FP64 machine eps.

    Memory fallback:
      On OutOfMemoryError the solver retries transparently with CUDA unified
      memory, allowing matrices that exceed VRAM to be solved via host paging.
    """

    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'mpdok.so'
            )
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"mpdok.so not found at {lib_path}. "
                f"Run 'make' inside the MPDOK directory first."
            )
        self._lib = ctypes.CDLL(lib_path)
        self._setup_signatures()

    def _setup_signatures(self):
        self._lib.py_gmres_ir.argtypes = [
            ctypes.c_void_p,                   # A  — FP64 device ptr (N×N, Fortran order)
            ctypes.c_void_p,                   # b  — FP64 device ptr (N,)
            ctypes.c_void_p,                   # x  — FP64 device ptr (N,), output
            ctypes.c_int,                      # N
            ctypes.c_double,                   # tol
            ctypes.c_int,                      # maxiter_outer
            ctypes.c_int,                      # restart
            ctypes.POINTER(ctypes.c_int),      # converged_out
        ]
        self._lib.py_gmres_ir.restype = None

    # ----------------------------------------------------------------
    def solve(self, A, b, tol=1e-11, maxiter_outer=5, restart=50):
        """Solve Ax = b using GMRES-IR.

        Args:
            A:             (N, N) FP64 NumPy or CuPy array.
            b:             (N,) FP64 NumPy or CuPy array.
            tol:           Convergence on relative FP64 residual ||b-Ax||/||b||.
                           Practical floor ~1e-13 (FP64 machine epsilon).
            maxiter_outer: Max refinement steps. Default 5; 2 almost always enough.
                           Increase to 10 for ill-conditioned problems.
            restart:       Inner GMRES subspace size. Default 50.
                           Increase to 100+ for harder (worse-conditioned) problems.

        Returns:
            x: FP64 CuPy (N,) solution array.
               Backed by regular device memory normally, or CUDA unified memory
               if VRAM was insufficient (transparent to the caller).

        Raises:
            RuntimeWarning: if the solver did not converge within maxiter_outer.
        """
        try:
            return self._solve(A, b, tol, maxiter_outer, restart, unified=False)
        except cp.cuda.memory.OutOfMemoryError:
            N = int(np.asarray(A).shape[0]) if not isinstance(A, cp.ndarray) else A.shape[0]
            needed_gb = N * N * 12 / 1e9   # FP64 + FP32 copies
            warnings.warn(
                f"VRAM insufficient for N={N} ({needed_gb:.1f} GB needed); "
                "retrying with CUDA unified memory (host+device paging). "
                "Performance may be lower for data that does not fit in VRAM.",
                ResourceWarning, stacklevel=2
            )
            cp.get_default_memory_pool().free_all_blocks()
            return self._solve(A, b, tol, maxiter_outer, restart, unified=True)

    # ----------------------------------------------------------------
    def _solve(self, A, b, tol, maxiter_outer, restart, unified):
        """Internal solve — allocates in device or unified memory."""
        def _run(alloc_ctx):
            with alloc_ctx:
                A_gpu = cp.asfortranarray(cp.asarray(A, dtype=cp.float64))
                b_gpu = cp.ascontiguousarray(cp.asarray(b, dtype=cp.float64))
                N = int(A_gpu.shape[0])

                if A_gpu.shape != (N, N):
                    raise ValueError(f"A must be square (N×N), got {A_gpu.shape}")
                if b_gpu.shape != (N,):
                    raise ValueError(f"b must be (N,), got {b_gpu.shape}")

                x_gpu = cp.zeros(N, dtype=cp.float64)
                converged = ctypes.c_int(0)

                self._lib.py_gmres_ir(
                    ctypes.c_void_p(A_gpu.data.ptr),
                    ctypes.c_void_p(b_gpu.data.ptr),
                    ctypes.c_void_p(x_gpu.data.ptr),
                    ctypes.c_int(N),
                    ctypes.c_double(tol),
                    ctypes.c_int(maxiter_outer),
                    ctypes.c_int(restart),
                    ctypes.byref(converged),
                )
                cp.cuda.Stream.null.synchronize()
            return x_gpu, converged, N

        if unified:
            _pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            ctx = cp.cuda.using_allocator(_pool.malloc)
        else:
            ctx = _NullContext()   # no-op — use CuPy's default pool

        x_gpu, converged, N = _run(ctx)

        if not converged.value:
            warnings.warn(
                f"GMRES-IR did not converge to tol={tol:.1e} within "
                f"{maxiter_outer} outer iterations. "
                f"Try increasing maxiter_outer or restart.",
                RuntimeWarning, stacklevel=3,
            )

        return x_gpu
