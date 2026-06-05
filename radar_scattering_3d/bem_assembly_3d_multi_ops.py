"""
bem_assembly_3d_multi_ops.py — ctypes interface to bem_assembly_3d_multi.so.

Superset of bem_assembly_3d_ops.py: exposes assembly + multi-RHS + IR.

Public API
----------
    BEMAssembler3DMulti.build_matrix(nodes, areas, k, precision='c64')
        → cupy (N, N)

    BEMAssembler3DMulti.solve_multi_rhs(nodes, areas, k, B,
                                         restart=50, tol=1e-6)
        → (X_np, n_converged)
        X_np : (N, M) complex128 NumPy, one solution per column
        n_converged : int

    BEMAssembler3DMulti.solve_ir(nodes, areas, k, b,
                                  restart=50, tol=1e-6, maxiter_ir=2)
        → (x_np, converged, rel_res)
        x_np : (N,) complex128 NumPy
"""

import ctypes
import os
import numpy as np
import cupy as cp

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, 'bem_assembly_3d_multi.so')


def is_available():
    return os.path.exists(_SO_PATH)


class BEMAssembler3DMulti:
    """Fortran CUDA backend: 3D BEM assembly + GMRES + multi-RHS + IR."""

    _instance = None
    _lib      = None

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
                f'bem_assembly_3d_multi.so not found at {_SO_PATH}. '
                'Run: make'
            )
        lib = ctypes.CDLL(_SO_PATH)

        # py_build_bem3d_c64/c128(A, nx, ny, nz, da, N, k)
        for name in ('py_build_bem3d_c64', 'py_build_bem3d_c128'):
            fn = getattr(lib, name)
            fn.restype  = None
            fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_double]

        # py_bem_solve_multi_rhs_3d(nx,ny,nz,da, B,X, A_c64, N,M,k, restart,tol, n_conv)
        lib.py_bem_solve_multi_rhs_3d.restype  = None
        lib.py_bem_solve_multi_rhs_3d.argtypes = [
            ctypes.c_void_p,               # nx
            ctypes.c_void_p,               # ny
            ctypes.c_void_p,               # nz
            ctypes.c_void_p,               # da
            ctypes.c_void_p,               # B   complex128 (N,M) col-major
            ctypes.c_void_p,               # X   complex128 (N,M) col-major
            ctypes.c_void_p,               # A_c64 flat complex64
            ctypes.c_int,                  # N
            ctypes.c_int,                  # M
            ctypes.c_double,               # k
            ctypes.c_int,                  # restart
            ctypes.c_double,               # tol
            ctypes.POINTER(ctypes.c_int),  # n_converged (output)
        ]

        # py_bem_solve_ir_3d(nx,ny,nz,da, b,x, A_c64,A_c128, N,k, restart,tol,
        #                     maxiter_ir, converged_out, rel_res_out)
        lib.py_bem_solve_ir_3d.restype  = None
        lib.py_bem_solve_ir_3d.argtypes = [
            ctypes.c_void_p,                  # nx
            ctypes.c_void_p,                  # ny
            ctypes.c_void_p,                  # nz
            ctypes.c_void_p,                  # da
            ctypes.c_void_p,                  # b  complex128 (N,)
            ctypes.c_void_p,                  # x  complex128 (N,) output
            ctypes.c_void_p,                  # A_c64
            ctypes.c_void_p,                  # A_c128 (or None → c_null_ptr)
            ctypes.c_int,                     # N
            ctypes.c_double,                  # k
            ctypes.c_int,                     # restart
            ctypes.c_double,                  # tol
            ctypes.c_int,                     # maxiter_ir
            ctypes.POINTER(ctypes.c_int),     # converged_out
            ctypes.POINTER(ctypes.c_double),  # rel_res_out
        ]

        self._lib    = lib
        self._loaded = True

    # ── geometry helpers ────────────────────────────────────────────────────

    def _push_geometry(self, nodes, areas):
        """Transfer centroid coordinates and panel areas to GPU."""
        nodes = np.asarray(nodes, dtype=np.float64)
        areas = np.asarray(areas, dtype=np.float64)
        nx_d  = cp.asarray(np.ascontiguousarray(nodes[:, 0]))
        ny_d  = cp.asarray(np.ascontiguousarray(nodes[:, 1]))
        nz_d  = cp.asarray(np.ascontiguousarray(nodes[:, 2]))
        da_d  = cp.asarray(np.ascontiguousarray(areas))
        return nx_d, ny_d, nz_d, da_d

    # ── public methods ──────────────────────────────────────────────────────

    def build_matrix(self, nodes, areas, k, precision='c64'):
        """Build (N, N) BEM matrix on GPU."""
        self._load()
        N = nodes.shape[0]
        nx_d, ny_d, nz_d, da_d = self._push_geometry(nodes, areas)

        if precision == 'c64':
            A_d = cp.zeros(N * N, dtype=cp.complex64)
            self._lib.py_build_bem3d_c64(
                ctypes.c_void_p(A_d.data.ptr),
                ctypes.c_void_p(nx_d.data.ptr),
                ctypes.c_void_p(ny_d.data.ptr),
                ctypes.c_void_p(nz_d.data.ptr),
                ctypes.c_void_p(da_d.data.ptr),
                ctypes.c_int(N), ctypes.c_double(float(k)),
            )
        else:
            A_d = cp.zeros(N * N, dtype=cp.complex128)
            self._lib.py_build_bem3d_c128(
                ctypes.c_void_p(A_d.data.ptr),
                ctypes.c_void_p(nx_d.data.ptr),
                ctypes.c_void_p(ny_d.data.ptr),
                ctypes.c_void_p(nz_d.data.ptr),
                ctypes.c_void_p(da_d.data.ptr),
                ctypes.c_int(N), ctypes.c_double(float(k)),
            )
        return A_d.reshape(N, N)

    def solve_multi_rhs(self, nodes, areas, k, B,
                        restart=50, tol=1e-6):
        """1 GPU build + M sequential GMRES solves.

        Args:
            nodes:   (N, 3) float64 — panel centroids.
            areas:   (N,)   float64 — panel areas.
            k:       float  — wavenumber.
            B:       (N, M) complex128 NumPy, Fortran column-major.
            restart: int    — GMRES restart dimension.
            tol:     float  — per-solve convergence tolerance.

        Returns:
            X          : (N, M) complex128 NumPy solutions.
            n_converged: int — number of columns that met tol.
        """
        self._load()
        N, M = B.shape
        nx_d, ny_d, nz_d, da_d = self._push_geometry(nodes, areas)

        B_f  = np.asfortranarray(B.astype(np.complex128))
        B_d  = cp.asarray(B_f)
        X_d  = cp.zeros((N, M), dtype=cp.complex128, order='F')
        A_d  = cp.zeros(N * N,  dtype=cp.complex64)

        n_conv = ctypes.c_int(0)

        self._lib.py_bem_solve_multi_rhs_3d(
            ctypes.c_void_p(nx_d.data.ptr),
            ctypes.c_void_p(ny_d.data.ptr),
            ctypes.c_void_p(nz_d.data.ptr),
            ctypes.c_void_p(da_d.data.ptr),
            ctypes.c_void_p(B_d.data.ptr),
            ctypes.c_void_p(X_d.data.ptr),
            ctypes.c_void_p(A_d.data.ptr),
            ctypes.c_int(N),
            ctypes.c_int(M),
            ctypes.c_double(float(k)),
            ctypes.c_int(restart),
            ctypes.c_double(float(tol)),
            ctypes.byref(n_conv),
        )

        X_np = cp.asnumpy(X_d)
        return X_np, int(n_conv.value)

    def solve_ir(self, nodes, areas, k, b,
                 restart=50, tol=1e-6, maxiter_ir=2):
        """GPU build → GMRES → iterative refinement.

        Args:
            nodes:      (N, 3) float64.
            areas:      (N,)   float64.
            k:          float  — wavenumber.
            b:          (N,)   complex128 NumPy — RHS.
            restart:    int.
            tol:        float.
            maxiter_ir: int  — 0 = GMRES only, 2 = full IR.

        Returns:
            x_np      : (N,) complex128 NumPy solution.
            converged : bool.
            rel_res   : float — final ‖r‖/‖b‖.
        """
        self._load()
        N = nodes.shape[0]
        nx_d, ny_d, nz_d, da_d = self._push_geometry(nodes, areas)

        b_d      = cp.asarray(np.ascontiguousarray(b, dtype=np.complex128))
        x_d      = cp.zeros(N, dtype=cp.complex128)
        A_c64_d  = cp.zeros(N * N, dtype=cp.complex64)
        A_c128_d = cp.zeros(N * N, dtype=cp.complex128) if maxiter_ir > 0 else None

        converged_out = ctypes.c_int(0)
        rel_res_out   = ctypes.c_double(1.0)

        self._lib.py_bem_solve_ir_3d(
            ctypes.c_void_p(nx_d.data.ptr),
            ctypes.c_void_p(ny_d.data.ptr),
            ctypes.c_void_p(nz_d.data.ptr),
            ctypes.c_void_p(da_d.data.ptr),
            ctypes.c_void_p(b_d.data.ptr),
            ctypes.c_void_p(x_d.data.ptr),
            ctypes.c_void_p(A_c64_d.data.ptr),
            ctypes.c_void_p(A_c128_d.data.ptr) if A_c128_d is not None
                else ctypes.c_void_p(0),
            ctypes.c_int(N),
            ctypes.c_double(float(k)),
            ctypes.c_int(restart),
            ctypes.c_double(float(tol)),
            ctypes.c_int(maxiter_ir),
            ctypes.byref(converged_out),
            ctypes.byref(rel_res_out),
        )

        return (cp.asnumpy(x_d),
                bool(converged_out.value),
                float(rel_res_out.value))
