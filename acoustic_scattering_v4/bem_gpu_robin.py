"""
bem_gpu_robin.py — Python ctypes wrapper for bem_assembly_robin.so

Exposes:
    build_matrix_robin_gpu(nodes, normals, lengths, k, alpha, dtype=np.complex128)
        → (N, N) complex array on CPU (NumPy), assembled on GPU.
        Returns None if GPU kernel is unavailable.

    HAS_ROBIN_GPU : bool — True when bem_assembly_robin.so loaded successfully

The kernel computes A_robin = A_neumann − iα·A_dirichlet in one GPU pass,
using J₀,Y₀,J₁,Y₁ device intrinsics (NVHPC).  Tikhonov regularisation is
NOT applied here — call _tikhonov(A) after retrieval.

All intermediate arithmetic is float64; the c64 variant casts to float32 only
at write time (used by the CuPy GMRES path).
"""

import ctypes
import os
from pathlib import Path

import numpy as np

# ── Load shared library ───────────────────────────────────────────────────────

_LIB_PATH = Path(__file__).parent / "bem_assembly_robin.so"
_lib = None
HAS_ROBIN_GPU = False

try:
    _lib = ctypes.CDLL(str(_LIB_PATH))

    # py_build_robin_c128(A, cx, cy, nnx, nny, dl, N, k, alpha)
    _lib.py_build_robin_c128.restype  = None
    _lib.py_build_robin_c128.argtypes = [
        ctypes.c_void_p,   # A_ptr   (complex128 device)
        ctypes.c_void_p,   # cx_ptr  (float64 device)
        ctypes.c_void_p,   # cy_ptr  (float64 device)
        ctypes.c_void_p,   # nnx_ptr (float64 device)
        ctypes.c_void_p,   # nny_ptr (float64 device)
        ctypes.c_void_p,   # dl_ptr  (float64 device)
        ctypes.c_int,      # N
        ctypes.c_double,   # k
        ctypes.c_double,   # alpha
    ]

    # py_build_robin_c64(A, cx, cy, nnx, nny, dl, N, k, alpha)
    _lib.py_build_robin_c64.restype  = None
    _lib.py_build_robin_c64.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_double, ctypes.c_double,
    ]

    HAS_ROBIN_GPU = True
except (OSError, AttributeError):
    pass


# ── CuPy availability ─────────────────────────────────────────────────────────

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


# ── Public API ────────────────────────────────────────────────────────────────

def build_matrix_robin_gpu(nodes, normals, lengths, k, alpha,
                            dtype=np.complex128):
    """Assemble A_robin = A_neumann − iα·A_dirichlet on GPU.

    Parameters
    ----------
    nodes   : (N, 2) float64  panel midpoints
    normals : (N, 2) float64  outward unit normals
    lengths : (N,)   float64  panel arc-lengths
    k       : float  wavenumber
    alpha   : float  Robin parameter α = k/ζ
    dtype   : np.complex128 (default) or np.complex64

    Returns
    -------
    A : (N, N) NumPy complex array on CPU, or None if GPU unavailable.
    """
    if not (HAS_ROBIN_GPU and _HAS_CUPY):
        return None

    N = len(nodes)
    cx = np.ascontiguousarray(nodes[:, 0], dtype=np.float64)
    cy = np.ascontiguousarray(nodes[:, 1], dtype=np.float64)
    nx = np.ascontiguousarray(normals[:, 0], dtype=np.float64)
    ny = np.ascontiguousarray(normals[:, 1], dtype=np.float64)
    dl = np.ascontiguousarray(lengths,       dtype=np.float64)

    # Upload geometry to GPU
    cx_d  = cp.asarray(cx)
    cy_d  = cp.asarray(cy)
    nnx_d = cp.asarray(nx)
    nny_d = cp.asarray(ny)
    dl_d  = cp.asarray(dl)

    if dtype == np.complex64:
        A_d = cp.zeros(N * N, dtype=np.complex64)
        _lib.py_build_robin_c64(
            ctypes.c_void_p(A_d.data.ptr),
            ctypes.c_void_p(cx_d.data.ptr),
            ctypes.c_void_p(cy_d.data.ptr),
            ctypes.c_void_p(nnx_d.data.ptr),
            ctypes.c_void_p(nny_d.data.ptr),
            ctypes.c_void_p(dl_d.data.ptr),
            ctypes.c_int(N),
            ctypes.c_double(float(k)),
            ctypes.c_double(float(alpha)),
        )
    else:
        A_d = cp.zeros(N * N, dtype=np.complex128)
        _lib.py_build_robin_c128(
            ctypes.c_void_p(A_d.data.ptr),
            ctypes.c_void_p(cx_d.data.ptr),
            ctypes.c_void_p(cy_d.data.ptr),
            ctypes.c_void_p(nnx_d.data.ptr),
            ctypes.c_void_p(nny_d.data.ptr),
            ctypes.c_void_p(dl_d.data.ptr),
            ctypes.c_int(N),
            ctypes.c_double(float(k)),
            ctypes.c_double(float(alpha)),
        )

    cp.cuda.Stream.null.synchronize()
    return A_d.reshape(N, N).get()
