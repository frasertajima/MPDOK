"""
MPDOK — Python interface to the Fortran GMRES-IR and LU-IR kernels.

Drop-in replacement for scipy.sparse.linalg.gmres / cp.linalg.solve
for dense problems where FP64 accuracy at TF32 speed is needed.

Usage — standard (matrix fits in VRAM):
    from MPDOK.mpdok_ops import MPDOKSolver
    solver = MPDOKSolver()
    x = solver.solve(A, b)              # A: FP64 NumPy or CuPy (N×N)

Usage — large matrix (exceeds VRAM), construct entirely on-device:
    A = solver.alloc_managed(N)         # (N,N) FP64 CuPy array in managed memory
    A[:] = ...                          # fill on GPU — no host transfer
    x = solver.solve(A, b)
    solver.free_managed()               # release when done

Managed memory path:
    Uses Fortran's 'managed' attribute (cudaMallocManaged) rather than
    CuPy's allocator-context approach, which does not reliably intercept
    cp.asarray().  The managed pointer is wrapped as a cp.cuda.UnownedMemory
    CuPy array so GPU kernels can write into it before the Fortran solve.
"""

import contextlib
import ctypes
import os
import warnings

import cupy as cp
import numpy as np


class _ManagedArray:
    """CuPy array backed by Fortran-managed memory.

    Lifetime is controlled by the owning solver's free_managed() call,
    not by CuPy's garbage collector (UnownedMemory, owner=None).
    """

    def __init__(self, cupy_array, free_fn):
        self._arr = cupy_array
        self._free_fn = free_fn
        self._freed = False

    # Delegate array protocol so callers can treat this like a cp.ndarray.
    def __getattr__(self, name):
        return getattr(self._arr, name)

    def __setitem__(self, key, value):
        self._arr[key] = value

    def __getitem__(self, key):
        return self._arr[key]

    @property
    def array(self):
        return self._arr

    def free(self):
        if not self._freed:
            self._free_fn()
            self._freed = True

    def __del__(self):
        self.free()


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

    Managed memory:
      solver.alloc_managed(N) returns a (N,N) FP64 CuPy array in CUDA managed
      memory — fill it on-device, pass to solve(), call free_managed() when done.
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
        self._managed = None
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

        self._lib.py_lu_alloc_managed.argtypes = [
            ctypes.c_int,                      # N
            ctypes.POINTER(ctypes.c_void_p),   # ptr_out
        ]
        self._lib.py_lu_alloc_managed.restype = None

        self._lib.py_lu_free_managed.argtypes = []
        self._lib.py_lu_free_managed.restype = None

        self._lib.py_lu_factor.argtypes = [
            ctypes.c_void_p,                   # A  — FP64 device ptr (N×N)
            ctypes.c_int,                      # N
            ctypes.POINTER(ctypes.c_int),      # info_out (0 = ok)
        ]
        self._lib.py_lu_factor.restype = None

        self._lib.py_lu_solve_factored.argtypes = [
            ctypes.c_void_p,                   # A  — FP64 device ptr (N×N, for Dgemv)
            ctypes.c_void_p,                   # b  — FP64 device ptr (N,)
            ctypes.c_void_p,                   # x  — FP64 device ptr (N,), output
            ctypes.c_int,                      # N
            ctypes.c_double,                   # tol
            ctypes.c_int,                      # maxiter_outer
            ctypes.POINTER(ctypes.c_int),      # converged_out
        ]
        self._lib.py_lu_solve_factored.restype = None

        self._lib.py_lu_free_factored.argtypes = []
        self._lib.py_lu_free_factored.restype = None

    # ----------------------------------------------------------------
    # Managed memory API
    # ----------------------------------------------------------------

    def alloc_managed(self, N):
        """Allocate an (N, N) FP64 matrix in CUDA managed memory.

        Uses Fortran's 'managed' attribute (cudaMallocManaged) — reliable
        for matrices that exceed VRAM.  Fill the returned array on the GPU
        using CuPy kernels, then pass it to solve().

        Returns a _ManagedArray whose .array property is a (N,N) Fortran-order
        FP64 CuPy ndarray backed by the managed allocation.
        """
        ptr = ctypes.c_void_p()
        self._lib.py_lu_alloc_managed(ctypes.c_int(N), ctypes.byref(ptr))
        if not ptr.value:
            raise MemoryError(f"py_lu_alloc_managed failed for N={N}")
        mem = cp.cuda.UnownedMemory(ptr.value, N * N * 8, owner=None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        arr = cp.ndarray((N, N), dtype=cp.float64, memptr=memptr, order='F')
        managed = _ManagedArray(arr, self._lib.py_lu_free_managed)
        self._managed = managed
        return managed

    def free_managed(self):
        """Release the managed matrix allocation explicitly."""
        if self._managed is not None:
            self._managed.free()
            self._managed = None
        else:
            self._lib.py_lu_free_managed()

    # ----------------------------------------------------------------
    # Pre-factored API — factor once, solve many times
    # ----------------------------------------------------------------

    def factor(self, A):
        """Factor A using FP32/TF32 LU (tensor cores). Stores LU on device.

        Call once when A is fixed; then call solve_factored(b) for each
        right-hand side.  O(N^3) cost paid here; subsequent solves are O(N^2).

        Args:
            A: (N, N) FP64 NumPy, CuPy, or _ManagedArray.
        """
        A_arr = A.array if isinstance(A, _ManagedArray) else A
        A_gpu = cp.asfortranarray(cp.asarray(A_arr, dtype=cp.float64))
        N = int(A_gpu.shape[0])
        self._factored_A = A_gpu   # keep reference for Dgemv in refinement
        self._factored_N = N
        info = ctypes.c_int(0)
        self._lib.py_lu_factor(
            ctypes.c_void_p(A_gpu.data.ptr),
            ctypes.c_int(N),
            ctypes.byref(info),
        )
        cp.cuda.Stream.null.synchronize()
        if info.value != 0:
            raise RuntimeError(f"py_lu_factor failed: info={info.value}")

    def solve_factored(self, b, tol=1e-11, maxiter_outer=3):
        """Solve Ax=b using the LU factors from the last factor() call.

        O(N^2) triangular solve + FP64 iterative refinement.  Does NOT
        re-factor A — call factor() first, then call this for each b.

        Args:
            b:             (N,) FP64 NumPy or CuPy array.
            tol:           Convergence on relative FP64 residual.
            maxiter_outer: Max refinement steps.

        Returns:
            x: FP64 CuPy (N,) solution array.
        """
        if not hasattr(self, '_factored_A'):
            raise RuntimeError("Call factor(A) before solve_factored(b).")
        A_gpu = self._factored_A
        b_gpu = cp.ascontiguousarray(cp.asarray(b, dtype=cp.float64))
        N = int(A_gpu.shape[0])
        x_gpu = cp.zeros(N, dtype=cp.float64)
        converged = ctypes.c_int(0)
        self._lib.py_lu_solve_factored(
            ctypes.c_void_p(A_gpu.data.ptr),
            ctypes.c_void_p(b_gpu.data.ptr),
            ctypes.c_void_p(x_gpu.data.ptr),
            ctypes.c_int(N),
            ctypes.c_double(tol),
            ctypes.c_int(maxiter_outer),
            ctypes.byref(converged),
        )
        cp.cuda.Stream.null.synchronize()
        if converged.value == -1:
            raise RuntimeError("solve_factored: no factorization available — call factor(A) first.")
        if not converged.value:
            warnings.warn(
                f"LU-IR solve_factored did not converge to tol={tol:.1e} "
                f"within {maxiter_outer} outer iterations.",
                RuntimeWarning, stacklevel=2,
            )
        return x_gpu

    def free_factored(self):
        """Release saved LU factors and cuBLAS/cuSOLVER handles."""
        self._lib.py_lu_free_factored()
        if hasattr(self, '_factored_A'):
            del self._factored_A
            del self._factored_N

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------

    def solve(self, A, b, tol=1e-11, maxiter_outer=3):
        """Solve Ax = b using LU-IR (FP32 LU + FP64 iterative refinement).

        Args:
            A:             (N, N) FP64 NumPy, CuPy, or _ManagedArray.
                           For matrices exceeding VRAM, pass a _ManagedArray
                           obtained from alloc_managed() and filled on-device.
            b:             (N,) FP64 NumPy or CuPy array.
            tol:           Convergence on relative FP64 residual ||b-Ax||/||b||.
            maxiter_outer: Max refinement steps. Default 3; 2 almost always enough.

        Returns:
            x: FP64 CuPy (N,) solution array.
        """
        try:
            return self._solve(A, b, tol, maxiter_outer)
        except cp.cuda.memory.OutOfMemoryError:
            A_arr = A.array if isinstance(A, _ManagedArray) else A
            N = A_arr.shape[0]
            needed_gb = N * N * 12 / 1e9
            warnings.warn(
                f"VRAM insufficient for N={N} ({needed_gb:.1f} GB needed). "
                "Use solver.alloc_managed(N) to build the matrix in CUDA managed "
                "memory from the start — no host-to-device transfer required.",
                ResourceWarning, stacklevel=2,
            )
            raise

    def _solve(self, A, b, tol, maxiter_outer):
        is_managed = isinstance(A, _ManagedArray)
        A_arr = A.array if is_managed else A
        if is_managed:
            A_gpu = A_arr
        else:
            A_gpu = cp.asfortranarray(cp.asarray(A_arr, dtype=cp.float64))
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

    Managed memory (for N where A > VRAM):
      A = solver.alloc_managed(N)    # allocates via cudaMallocManaged
      A[:] = build_rbf_kernel(...)   # fill on GPU — stays in managed memory
      x = solver.solve(A, b)
      solver.free_managed()
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
        self._managed = None  # track live _ManagedArray so free_managed() can mark it freed
        self._fp32_buf = None  # Python-owned FP32 scratch; Fortran receives pointer only
        self._setup_signatures()

    def _setup_signatures(self):
        self._lib.py_gmres_ir.argtypes = [
            ctypes.c_void_p,                   # A       — FP64 device ptr (N×N, Fortran order)
            ctypes.c_void_p,                   # b       — FP64 device ptr (N,)
            ctypes.c_void_p,                   # x       — FP64 device ptr (N,), output
            ctypes.c_void_p,                   # A_fp32  — FP32 device ptr (N×N), Python-owned scratch
            ctypes.c_int,                      # N
            ctypes.c_double,                   # tol
            ctypes.c_int,                      # maxiter_outer
            ctypes.c_int,                      # restart
            ctypes.POINTER(ctypes.c_int),      # converged_out
        ]
        self._lib.py_gmres_ir.restype = None

        self._lib.py_alloc_managed.argtypes = [
            ctypes.c_int,                      # N
            ctypes.POINTER(ctypes.c_void_p),   # ptr_out
        ]
        self._lib.py_alloc_managed.restype = None

        self._lib.py_free_managed.argtypes = []
        self._lib.py_free_managed.restype = None

    # ----------------------------------------------------------------
    # Managed memory API
    # ----------------------------------------------------------------

    def alloc_managed(self, N):
        """Allocate an (N, N) FP64 matrix in CUDA managed memory.

        Calls Fortran py_alloc_managed which uses cudaMallocManaged directly —
        bypassing CuPy's allocator context (which cp.asarray ignores).

        The returned _ManagedArray behaves like a CuPy array for indexing and
        arithmetic.  Fill it on-device (e.g. via rbf_kernel.build_rbf_kernel),
        pass to solve(), then call free_managed() or let it go out of scope.

        For N = 40,000: allocates 12.8 GB in managed memory.  CUDA pages it
        between host RAM and VRAM automatically as the solver accesses it.
        """
        ptr = ctypes.c_void_p()
        self._lib.py_alloc_managed(ctypes.c_int(N), ctypes.byref(ptr))
        if not ptr.value:
            raise MemoryError(f"py_alloc_managed failed for N={N}")
        mem = cp.cuda.UnownedMemory(ptr.value, N * N * 8, owner=None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        arr = cp.ndarray((N, N), dtype=cp.float64, memptr=memptr, order='F')
        managed = _ManagedArray(arr, self._lib.py_free_managed)
        self._managed = managed  # keep reference so free_managed() can mark it freed
        return managed

    def free_managed(self):
        """Release the managed matrix allocation explicitly."""
        if self._managed is not None:
            self._managed.free()  # sets _freed=True → __del__ becomes a no-op
            self._managed = None
        else:
            self._lib.py_free_managed()

    def free_fp32_buf(self):
        """Release the Python-owned FP32 scratch buffer and flush it from the CuPy pool.

        Call after each solve (or batch of same-N solves) to immediately return
        device memory.  Safe to call even if no buffer is allocated.
        """
        self._fp32_buf = None
        cp.get_default_memory_pool().free_all_blocks()

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------

    def solve(self, A, b, tol=1e-11, maxiter_outer=5, restart=50):
        """Solve Ax = b using GMRES-IR.

        Args:
            A:             (N, N) FP64 NumPy, CuPy, or _ManagedArray.
                           For matrices exceeding VRAM, pass a _ManagedArray
                           from alloc_managed() filled on-device — no host
                           transfer occurs.
            b:             (N,) FP64 NumPy or CuPy array.
            tol:           Convergence on relative FP64 residual. Floor ~1e-13.
            maxiter_outer: Max refinement steps. Default 5; 2 almost always enough.
            restart:       Inner GMRES subspace size. Default 50.

        Returns:
            x: FP64 CuPy (N,) solution array.
        """
        try:
            return self._solve(A, b, tol, maxiter_outer, restart)
        except cp.cuda.memory.OutOfMemoryError:
            A_arr = A.array if isinstance(A, _ManagedArray) else A
            N = int(np.asarray(A_arr).shape[0]) if not isinstance(A_arr, cp.ndarray) else A_arr.shape[0]
            needed_gb = N * N * 12 / 1e9
            warnings.warn(
                f"VRAM insufficient for N={N} ({needed_gb:.1f} GB needed). "
                "Use solver.alloc_managed(N) to build the matrix in CUDA managed "
                "memory from the start — no host-to-device transfer required.",
                ResourceWarning, stacklevel=2,
            )
            raise

    def _solve(self, A, b, tol, maxiter_outer, restart):
        is_managed = isinstance(A, _ManagedArray)
        A_arr = A.array if is_managed else A
        if is_managed:
            A_gpu = A_arr
        else:
            A_gpu = cp.asfortranarray(cp.asarray(A_arr, dtype=cp.float64))
        b_gpu = cp.ascontiguousarray(cp.asarray(b, dtype=cp.float64))
        N = int(A_gpu.shape[0])

        if A_gpu.shape != (N, N):
            raise ValueError(f"A must be square (N×N), got {A_gpu.shape}")
        if b_gpu.shape != (N,):
            raise ValueError(f"b must be (N,), got {b_gpu.shape}")

        # FP32 scratch buffer — allocated by Python/CuPy so CuPy owns the memory
        # and free_fp32_buf() can return it cleanly via pool flush.
        # Reuse if shape matches (common in multi-RHS loops); reallocate if N changed.
        if self._fp32_buf is None or self._fp32_buf.shape != (N, N):
            self._fp32_buf = cp.empty((N, N), dtype=cp.float32, order='F')

        x_gpu = cp.zeros(N, dtype=cp.float64)
        converged = ctypes.c_int(0)

        self._lib.py_gmres_ir(
            ctypes.c_void_p(A_gpu.data.ptr),
            ctypes.c_void_p(b_gpu.data.ptr),
            ctypes.c_void_p(x_gpu.data.ptr),
            ctypes.c_void_p(self._fp32_buf.data.ptr),
            ctypes.c_int(N),
            ctypes.c_double(tol),
            ctypes.c_int(maxiter_outer),
            ctypes.c_int(restart),
            ctypes.byref(converged),
        )
        cp.cuda.Stream.null.synchronize()

        if not converged.value:
            warnings.warn(
                f"GMRES-IR did not converge to tol={tol:.1e} within "
                f"{maxiter_outer} outer iterations. "
                f"Try increasing maxiter_outer or restart.",
                RuntimeWarning, stacklevel=2,
            )

        return x_gpu
