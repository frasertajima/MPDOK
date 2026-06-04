"""
bem_assembly_ops.py — Python ctypes interface to bem_assembly.so (Fortran CUDA).

Provides the same public API as bem_gpu.py but backed by nvfortran-compiled
CUDA kernels instead of CuPy RawKernel.  Both backends produce identical
numerical results; the Fortran backend eliminates NVRTC JIT compilation overhead
and runs inside the same NVHPC toolchain as mpdok_solver.cuf.

Public API
----------
    BEMAssembler.build_c64(nodes, lengths, k)   → cupy complex64  (N,N)
    BEMAssembler.build_c128(nodes, lengths, k)  → cupy complex128 (N,N)
    BEMAssembler.residual_c128(A, b, x, r)      → float  (‖b−Ax‖/‖b‖)

    build_bem_matrix_fortran(nodes, lengths, k)       → cupy complex64
    build_bem_matrix_fortran_c128(nodes, lengths, k)  → cupy complex128
    is_available()                                     → bool

Usage
-----
    from bem_assembly_ops import build_bem_matrix_fortran
    A = build_bem_matrix_fortran(nodes, lengths, k=8.0)
    # A: cupy.complex64, (N,N), already in VRAM — drop in for build_bem_matrix_gpu
"""

import ctypes
import os
import time
import warnings

import cupy as cp
import numpy as np

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, 'bem_assembly.so')


class BEMAssembler:
    """Fortran CUDA backend for 2D Helmholtz BEM matrix assembly.

    Loads bem_assembly.so on first instantiation and keeps the handle alive.
    Thread-safe for concurrent Python threads (each kernel launch is independent).
    """

    _instance  = None   # module-level singleton
    _lib       = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        if not os.path.exists(_SO_PATH):
            raise FileNotFoundError(
                f'bem_assembly.so not found at {_SO_PATH}. '
                "Run: cd MPDOK && make radar_scattering/bem_assembly.so"
            )
        self._lib = ctypes.CDLL(_SO_PATH)

        # py_build_bem_c64(A_ptr, nx_ptr, ny_ptr, dl_ptr, N, k)
        self._lib.py_build_bem_c64.restype  = None
        self._lib.py_build_bem_c64.argtypes = [
            ctypes.c_void_p,   # A   — complex64 device ptr (N*N*8 bytes)
            ctypes.c_void_p,   # nx  — float64  device ptr (N*8 bytes)
            ctypes.c_void_p,   # ny
            ctypes.c_void_p,   # dl
            ctypes.c_int,      # N
            ctypes.c_double,   # k
        ]

        # py_build_bem_c128(A_ptr, nx_ptr, ny_ptr, dl_ptr, N, k)
        self._lib.py_build_bem_c128.restype  = None
        self._lib.py_build_bem_c128.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_double,
        ]

        # py_bem_residual_c128(A_ptr, b_ptr, x_ptr, r_ptr, N, rel_res_out)
        self._lib.py_bem_residual_c128.restype  = None
        self._lib.py_bem_residual_c128.argtypes = [
            ctypes.c_void_p,                   # A   — complex128 (N,N)
            ctypes.c_void_p,                   # b   — complex128 (N,)
            ctypes.c_void_p,                   # x   — complex128 (N,)
            ctypes.c_void_p,                   # r   — complex128 (N,) output
            ctypes.c_int,                      # N
            ctypes.POINTER(ctypes.c_double),   # rel_res_out
        ]

        # py_bem_solve_ir(nx,ny,dl,b,x, A_c64,A_c128, N,k, restart,tol,
        #                 maxiter_ir, converged_out, rel_res_out)
        self._lib.py_bem_solve_ir.restype  = None
        self._lib.py_bem_solve_ir.argtypes = [
            ctypes.c_void_p,                   # nx  — float64 device (N,)
            ctypes.c_void_p,                   # ny
            ctypes.c_void_p,                   # dl
            ctypes.c_void_p,                   # b   — complex128 device (N,)
            ctypes.c_void_p,                   # x   — complex128 device (N,) output
            ctypes.c_void_p,                   # A_c64  — complex64 workspace (N*N)
            ctypes.c_void_p,                   # A_c128 — complex128 workspace or NULL
            ctypes.c_int,                      # N
            ctypes.c_double,                   # k
            ctypes.c_int,                      # restart
            ctypes.c_double,                   # tol
            ctypes.c_int,                      # maxiter_ir (0 = GMRES only)
            ctypes.POINTER(ctypes.c_int),      # converged_out
            ctypes.POINTER(ctypes.c_double),   # rel_res_out
        ]

        # py_bem_solve_multi_rhs(nx,ny,dl, B,X, A_c64, N,M,k, restart,tol, n_conv)
        self._lib.py_bem_solve_multi_rhs.restype  = None
        self._lib.py_bem_solve_multi_rhs.argtypes = [
            ctypes.c_void_p,               # nx  — float64 device (N,)
            ctypes.c_void_p,               # ny
            ctypes.c_void_p,               # dl
            ctypes.c_void_p,               # B   — complex128 device (N, M) col-major
            ctypes.c_void_p,               # X   — complex128 device (N, M) col-major output
            ctypes.c_void_p,               # A_c64 — complex64 workspace (N*N)
            ctypes.c_int,                  # N
            ctypes.c_int,                  # M  (number of RHS)
            ctypes.c_double,               # k
            ctypes.c_int,                  # restart
            ctypes.c_double,               # tol
            ctypes.POINTER(ctypes.c_int),  # n_converged output
        ]

        self._loaded = True

    # ── Build functions ────────────────────────────────────────────────────

    def build_c64(self, nodes: np.ndarray, lengths: np.ndarray,
                  k: float) -> cp.ndarray:
        """Assemble Helmholtz BEM matrix as complex64 in GPU VRAM.

        Args:
            nodes:   (N, 2) float64 panel centroids (NumPy).
            lengths: (N,)   float64 panel arc lengths (NumPy).
            k:       Wavenumber.

        Returns:
            (N, N) cupy.complex64 on device (row-major, Python convention).
        """
        self._load()
        N = nodes.shape[0]

        free_b, _ = cp.cuda.runtime.memGetInfo()
        need_b    = N * N * 8
        if need_b > free_b * 0.90:
            raise RuntimeError(
                f'N={N}: need {need_b/1e6:.0f} MB complex64 '
                f'but only {free_b/1e6:.0f} MB free'
            )

        nx_d = cp.asarray(nodes[:, 0], dtype=cp.float64)
        ny_d = cp.asarray(nodes[:, 1], dtype=cp.float64)
        dl_d = cp.asarray(lengths,     dtype=cp.float64)
        A_d  = cp.empty(N * N, dtype=cp.complex64)

        self._lib.py_build_bem_c64(
            ctypes.c_void_p(int(A_d.data.ptr)),
            ctypes.c_void_p(int(nx_d.data.ptr)),
            ctypes.c_void_p(int(ny_d.data.ptr)),
            ctypes.c_void_p(int(dl_d.data.ptr)),
            ctypes.c_int(N),
            ctypes.c_double(k),
        )
        return A_d.reshape(N, N)

    def build_c128(self, nodes: np.ndarray, lengths: np.ndarray,
                   k: float) -> cp.ndarray:
        """Assemble Helmholtz BEM matrix as complex128 in GPU VRAM.

        Returns:
            (N, N) cupy.complex128 on device.
        """
        self._load()
        N = nodes.shape[0]

        free_b, _ = cp.cuda.runtime.memGetInfo()
        need_b    = N * N * 16
        if need_b > free_b * 0.90:
            raise RuntimeError(
                f'N={N}: need {need_b/1e6:.0f} MB complex128 '
                f'but only {free_b/1e6:.0f} MB free'
            )

        nx_d = cp.asarray(nodes[:, 0], dtype=cp.float64)
        ny_d = cp.asarray(nodes[:, 1], dtype=cp.float64)
        dl_d = cp.asarray(lengths,     dtype=cp.float64)
        A_d  = cp.empty(N * N, dtype=cp.complex128)

        self._lib.py_build_bem_c128(
            ctypes.c_void_p(int(A_d.data.ptr)),
            ctypes.c_void_p(int(nx_d.data.ptr)),
            ctypes.c_void_p(int(ny_d.data.ptr)),
            ctypes.c_void_p(int(dl_d.data.ptr)),
            ctypes.c_int(N),
            ctypes.c_double(k),
        )
        return A_d.reshape(N, N)

    def residual_c128(self, A: cp.ndarray, b: np.ndarray,
                      x: cp.ndarray, r: cp.ndarray) -> float:
        """Compute r = b - A @ x in complex128; return ‖r‖/‖b‖.

        Args:
            A: (N,N) cupy.complex128 on device.
            b: (N,)  complex128 NumPy or CuPy.
            x: (N,)  cupy.complex128 solution estimate.
            r: (N,)  cupy.complex128 output buffer (written in-place).

        Returns:
            Relative residual ‖b − Ax‖₂ / ‖b‖₂ as Python float.
        """
        self._load()
        N = A.shape[0]

        b_d = cp.asarray(b, dtype=cp.complex128)
        rel_res = ctypes.c_double(0.0)

        self._lib.py_bem_residual_c128(
            ctypes.c_void_p(int(A.data.ptr)),
            ctypes.c_void_p(int(b_d.data.ptr)),
            ctypes.c_void_p(int(x.data.ptr)),
            ctypes.c_void_p(int(r.data.ptr)),
            ctypes.c_int(N),
            ctypes.byref(rel_res),
        )
        return rel_res.value

    def solve_ir(self, nodes: np.ndarray, lengths: np.ndarray,
                 k: float, phi_inc: float,
                 restart: int = 50, tol: float = 1e-6,
                 maxiter_ir: int = 2,
                 verbose: bool = False) -> tuple:
        """Full Fortran pipeline: GPU BEM build → complex GMRES → IR refinement.

        All heavy lifting (matrix assembly, GMRES iterations, IR residual
        computation) happens inside the Fortran .so — zero Python overhead
        per GMRES iteration.

        Args:
            nodes:      (N, 2) float64 panel centroids (NumPy).
            lengths:    (N,)   float64 panel arc lengths (NumPy).
            k:          Wavenumber.
            phi_inc:    Incident plane-wave angle (radians).
            restart:    GMRES restart (default 50).
            tol:        Convergence tolerance for GMRES and IR (default 1e-6).
            maxiter_ir: IR steps after initial GMRES (0 = GMRES only, 2 = full IR).
            verbose:    Print timing and convergence info.

        Returns:
            (sigma, info) where sigma is (N,) complex128 NumPy and info is a
            dict with keys: t_build_c64, t_gmres_ir, converged, rel_res, backend.
        """
        self._load()
        N = nodes.shape[0]
        d = np.array([np.cos(phi_inc), np.sin(phi_inc)])
        b = -np.exp(1j * k * (nodes @ d)).astype(np.complex128)

        # Allocate Python-owned GPU buffers (Fortran never calls cudaFree for these)
        nx_d = cp.asarray(nodes[:, 0], dtype=cp.float64)
        ny_d = cp.asarray(nodes[:, 1], dtype=cp.float64)
        dl_d = cp.asarray(lengths,     dtype=cp.float64)
        b_d  = cp.asarray(b,           dtype=cp.complex128)
        x_d  = cp.zeros(N,             dtype=cp.complex128)

        free_b, _ = cp.cuda.runtime.memGetInfo()
        A_c64  = cp.empty(N * N, dtype=cp.complex64)

        use_ir = (maxiter_ir > 0) and (N * N * 16 < free_b * 0.45)
        A_c128 = cp.empty(N * N, dtype=cp.complex128) if use_ir else None
        a128_ptr = ctypes.c_void_p(int(A_c128.data.ptr)) if use_ir else ctypes.c_void_p(0)

        converged_out = ctypes.c_int(0)
        rel_res_out   = ctypes.c_double(1.0)

        t0 = time.perf_counter()
        self._lib.py_bem_solve_ir(
            ctypes.c_void_p(int(nx_d.data.ptr)),
            ctypes.c_void_p(int(ny_d.data.ptr)),
            ctypes.c_void_p(int(dl_d.data.ptr)),
            ctypes.c_void_p(int(b_d.data.ptr)),
            ctypes.c_void_p(int(x_d.data.ptr)),
            ctypes.c_void_p(int(A_c64.data.ptr)),
            a128_ptr,
            ctypes.c_int(N),
            ctypes.c_double(k),
            ctypes.c_int(restart),
            ctypes.c_double(tol),
            ctypes.c_int(maxiter_ir if use_ir else 0),
            ctypes.byref(converged_out),
            ctypes.byref(rel_res_out),
        )
        cp.cuda.Stream.null.synchronize()
        t_total = time.perf_counter() - t0

        sigma = cp.asnumpy(x_d).astype(np.complex128)
        info  = dict(
            t_total=t_total,
            converged=bool(converged_out.value),
            rel_res=rel_res_out.value,
            backend='fortran',
            maxiter_ir_used=maxiter_ir if use_ir else 0,
        )

        if verbose:
            print(f'[Fortran IR] N={N}  k={k:.0f}  '
                  f't={t_total:.3f}s  conv={info["converged"]}  '
                  f'rel_res={info["rel_res"]:.2e}  '
                  f'IR={info["maxiter_ir_used"]} steps')
        return sigma, info

    def solve_multi_rhs(self, nodes: np.ndarray, lengths: np.ndarray,
                        k: float, b_matrix: np.ndarray,
                        restart: int = 50, tol: float = 1e-6,
                        verbose: bool = False) -> tuple:
        """1 GPU build + M sequential GMRES solves (Stage 7 multi-RHS).

        Builds A exactly once, then solves for each column of b_matrix.
        For M=90 incident angles this eliminates 89 redundant GPU builds.

        Args:
            nodes:    (N, 2) float64 panel centroids (NumPy).
            lengths:  (N,)   float64 panel arc lengths (NumPy).
            k:        Wavenumber.
            b_matrix: (N, M) complex128 — column j is the RHS for incident j.
                      Must be Fortran-ordered (column-major) for zero-copy.
            restart:  GMRES restart.
            tol:      Per-solve tolerance.
            verbose:  Print timing and convergence count.

        Returns:
            (X, info) where X is (N, M) complex128 NumPy (column j = solution j)
            and info is dict with t_build, t_solve, n_converged, M.
        """
        self._load()
        N, M = nodes.shape[0], b_matrix.shape[1]

        nx_d  = cp.asarray(nodes[:, 0],  dtype=cp.float64)
        ny_d  = cp.asarray(nodes[:, 1],  dtype=cp.float64)
        dl_d  = cp.asarray(lengths,      dtype=cp.float64)

        # Column-major layout so Fortran B_d(:,j) is contiguous
        B_d   = cp.asfortranarray(cp.asarray(b_matrix, dtype=cp.complex128))
        X_d   = cp.asfortranarray(cp.zeros((N, M),     dtype=cp.complex128))
        A_c64 = cp.empty(N * N, dtype=cp.complex64)

        n_conv_out = ctypes.c_int(0)

        t0 = time.perf_counter()
        self._lib.py_bem_solve_multi_rhs(
            ctypes.c_void_p(int(nx_d.data.ptr)),
            ctypes.c_void_p(int(ny_d.data.ptr)),
            ctypes.c_void_p(int(dl_d.data.ptr)),
            ctypes.c_void_p(int(B_d.data.ptr)),
            ctypes.c_void_p(int(X_d.data.ptr)),
            ctypes.c_void_p(int(A_c64.data.ptr)),
            ctypes.c_int(N),
            ctypes.c_int(M),
            ctypes.c_double(k),
            ctypes.c_int(restart),
            ctypes.c_double(tol),
            ctypes.byref(n_conv_out),
        )
        cp.cuda.Stream.null.synchronize()
        t_total = time.perf_counter() - t0

        X = np.asfortranarray(cp.asnumpy(X_d))   # (N, M) column-major NumPy
        info = dict(
            t_total=t_total,
            t_per_solve=t_total / M,
            n_converged=n_conv_out.value,
            M=M,
            backend='fortran-multi-rhs',
        )
        if verbose:
            print(f'[multi-RHS] N={N} M={M} k={k:.0f}  '
                  f't={t_total:.2f}s ({t_total/M:.3f}s/solve)  '
                  f'conv={n_conv_out.value}/{M}')
        return X, info


# ── Module-level convenience functions ────────────────────────────────────────

def build_bem_matrix_fortran(nodes: np.ndarray, lengths: np.ndarray,
                              k: float) -> cp.ndarray:
    """Drop-in replacement for bem_gpu.build_bem_matrix_gpu using Fortran kernel."""
    return BEMAssembler().build_c64(nodes, lengths, k)


def build_bem_matrix_fortran_c128(nodes: np.ndarray, lengths: np.ndarray,
                                   k: float) -> cp.ndarray:
    """Drop-in replacement for bem_gpu.build_bem_matrix_gpu_c128 using Fortran kernel."""
    return BEMAssembler().build_c128(nodes, lengths, k)


def is_available() -> bool:
    """Return True if bem_assembly.so is compiled and loadable."""
    try:
        BEMAssembler()._load()
        return True
    except Exception:
        return False


# ── Accuracy and speed benchmark ──────────────────────────────────────────────

def benchmark(N: int = 4096, k: float = 3.0) -> dict:
    """Compare Fortran kernel vs CuPy RawKernel vs CPU for BEM assembly.

    Returns dict with timing and accuracy metrics for all three backends.
    """
    import sys
    _MPDOK = os.path.dirname(_HERE)
    for p in [_MPDOK, _HERE]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from radar_scattering.bem_gpu import build_bem_matrix_gpu, build_bem_matrix_gpu_c128
    from acoustic_scattering.bem_helmholtz import build_bem_matrix_helmholtz
    from radar_scattering.geometry import circle_panels

    nodes, _, lengths = circle_panels(N, R=1.0)

    print(f'N={N}  k={k}')

    # CPU baseline
    t0 = time.perf_counter()
    A_cpu = build_bem_matrix_helmholtz(nodes, lengths, k)
    t_cpu = time.perf_counter() - t0
    print(f'  CPU (scipy Hankel):       {t_cpu:.3f}s  '
          f'({N*N*16/1e6:.0f} MB complex128)')

    # CuPy RawKernel
    t0 = time.perf_counter()
    A_cupy = build_bem_matrix_gpu(nodes, lengths, k)
    cp.cuda.Stream.null.synchronize()
    t_cupy = time.perf_counter() - t0
    print(f'  CuPy RawKernel (c64):     {t_cupy:.3f}s  '
          f'({N*N*8/1e6:.0f} MB complex64)')

    # Fortran kernel
    t0 = time.perf_counter()
    A_fort = build_bem_matrix_fortran(nodes, lengths, k)
    cp.cuda.Stream.null.synchronize()
    t_fort = time.perf_counter() - t0
    print(f'  Fortran kernel (c64):     {t_fort:.3f}s  '
          f'({N*N*8/1e6:.0f} MB complex64)')

    # Accuracy: Fortran vs CPU (on 500 random off-diagonal elements)
    rng = np.random.default_rng(42)
    ii  = rng.integers(0, N, 600); jj = rng.integers(0, N, 600)
    m   = ii != jj; ii, jj = ii[m], jj[m]

    cpu_s  = A_cpu[ii, jj].astype(np.complex64)
    cupy_s = cp.asnumpy(A_cupy[ii, jj])
    fort_s = cp.asnumpy(A_fort[ii, jj])

    err_fort  = (np.abs(fort_s  - cpu_s) / (np.abs(cpu_s) + 1e-30)).max()
    err_cupy  = (np.abs(cupy_s  - cpu_s) / (np.abs(cpu_s) + 1e-30)).max()

    print(f'  Max rel error vs CPU (c64):')
    print(f'    CuPy kernel: {err_cupy:.2e}')
    print(f'    Fortran:     {err_fort:.2e}')

    # Agreement between CuPy and Fortran
    err_cf = (np.abs(fort_s - cupy_s) / (np.abs(cupy_s) + 1e-30)).max()
    print(f'  CuPy vs Fortran max rel diff: {err_cf:.2e}')
    print(f'  Speedup Fortran vs CPU: {t_cpu/t_fort:.0f}×')
    print(f'  Fortran vs CuPy ratio:  {t_fort/t_cupy:.2f}×  '
          f'({"faster" if t_fort < t_cupy else "slower"})')

    return dict(
        t_cpu=t_cpu, t_cupy=t_cupy, t_fort=t_fort,
        err_fort_vs_cpu=err_fort, err_cupy_vs_cpu=err_cupy,
        err_cupy_vs_fort=err_cf,
        speedup_vs_cpu=t_cpu / t_fort,
    )


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--N',  type=int,   default=4096)
    p.add_argument('--N2', type=int,   default=8192)
    p.add_argument('--k',  type=float, default=3.0)
    args = p.parse_args()

    print(f'=== BEM Assembly: Fortran vs CuPy vs CPU ===\n')
    print(f'Fortran backend available: {is_available()}\n')
    benchmark(args.N,  args.k)
    print()
    benchmark(args.N2, args.k)
