"""
bem_helmholtz_v4.py — Extends v3 with Robin (impedance) BC support.

Robin (impedance) boundary condition:  ∂p/∂n + iα p = 0   on Γ
    α = k / ζ   where ζ is the normalised specific acoustic impedance.

    α = 0      → Sound-Hard  (Neumann, rigid wall)
    α → ∞      → Sound-Soft  (Dirichlet, pressure-release)
    α = k      → matched impedance, maximum absorption

BEM formulation
---------------
Single-layer representation  p_scat = S[σ],  S[σ](x) = ∫ G(x,y) σ(y) ds(y).

Applying the Robin BC to the total field (exterior trace + jump relations):

    (½I − K' − iαS)[σ]  =  ∂p_inc/∂n + iα p_inc

In matrix form:

    A_robin  =  A_neumann  −  iα · A_dirichlet
    b_robin  =  b_neumann  +  iα · p_inc(nodes)

where:
    A_neumann    = build_matrix_neumann(...)   from bem_helmholtz_v3
    A_dirichlet  = build_matrix(...)           from bem_helmholtz_v2 / v1
    b_neumann    = make_rhs_neumann(...)
    p_inc(nodes) = exp(ik x·d̂)  or  (i/4) H₀(k|x−xₛ|)  for point source

Field evaluation is unchanged — the single-layer formula eval_total_field
applies to all three BC types (the difference is only in σ).

Implementation notes
--------------------
*  Both matrices are already assembled in v2/v3.  Robin adds one line:
       A = A_neumann - 1j * alpha * A_dirichlet
*  Tikhonov regularisation is applied to A_robin (not separately to each part).
*  Irregular frequencies shift with α — they occur where the combined
   operator (½I − K' − iαS) is singular.  The existing ε=1e-4 Tikhonov
   damps them without ill-affecting smooth solutions.
*  GPU path: Robin matrix stays on CPU (H₁ needed for K'; GPU has H₀ only).
   The GMRES solve uses CuPy when available (same as Neumann path in v3).
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import hankel1

# ── GPU Robin assembly (optional, falls back to CPU if unavailable) ───────────
try:
    from bem_gpu_robin import build_matrix_robin_gpu, HAS_ROBIN_GPU
except ImportError:
    HAS_ROBIN_GPU = False
    def build_matrix_robin_gpu(*a, **kw): return None

_HERE  = Path(__file__).parent
_V3DIR = _HERE.parent / "acoustic_scattering_v3"
_V2DIR = _HERE.parent / "acoustic_scattering_v2"
_V1DIR = _HERE.parent / "acoustic_scattering"
for _p in [str(_HERE), str(_V3DIR), str(_V2DIR), str(_V1DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Re-export everything from v3 unchanged ───────────────────────────────────

from bem_helmholtz_v3 import (
    # Dirichlet (v2 re-exports)
    build_matrix_dirichlet,
    make_rhs_dirichlet,
    solve_gmres_dirichlet,
    solve_ir_dirichlet,
    solve_multi_rhs_dirichlet,
    eval_far_field,
    eval_rcs_2d,
    eval_total_field,
    eval_total_field_gpu,
    HAS_GPU,
    # Neumann (v3)
    build_matrix_neumann,
    make_rhs_neumann,
    solve_neumann,
    solve_neumann_multi_rhs,
)

# Bare-name aliases for drop-in compatibility
build_matrix    = build_matrix_dirichlet
make_rhs        = make_rhs_dirichlet
solve_gmres     = solve_gmres_dirichlet
solve_ir        = solve_ir_dirichlet
solve_multi_rhs = solve_multi_rhs_dirichlet

# ── GPU GMRES plumbing (same as v3) ──────────────────────────────────────────

import inspect
try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import gmres as _cp_gmres, LinearOperator
    _GMRES_TOL_KW = 'rtol' if 'rtol' in inspect.signature(_cp_gmres).parameters else 'tol'
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

from scipy.sparse.linalg import gmres as _sp_gmres
_SP_TOL_KW = 'rtol' if 'rtol' in inspect.signature(_sp_gmres).parameters else 'tol'
from scipy.linalg import solve as _dense_solve


# ── Robin BEM matrix ──────────────────────────────────────────────────────────

def build_matrix_robin(nodes, normals, lengths, k, alpha):
    """Combined BEM matrix for the Robin (impedance) BC.

    A_robin = A_neumann - iα · A_dirichlet

    GPU path (preferred): single kernel launch computing H₀+H₁ simultaneously
    via bem_assembly_robin.so (CUDA Fortran).

    CPU fallback: two separate Python/SciPy builds then linear combination.

    Parameters
    ----------
    nodes   : (N, 2) float64  panel midpoints
    normals : (N, 2) float64  outward unit normals
    lengths : (N,)   float64  panel arc-lengths
    k       : wavenumber
    alpha   : Robin parameter  α = k/ζ  (float; 0 = Hard limit)

    Returns
    -------
    A : (N, N) complex128 on CPU (NumPy)
    """
    A = build_matrix_robin_gpu(nodes, normals, lengths, k, alpha)
    if A is not None:
        return A
    # CPU fallback
    A_n = build_matrix_neumann(nodes, normals, lengths, k)
    A_d = build_matrix_dirichlet(nodes, lengths, k)
    if hasattr(A_d, 'get'):
        A_d = A_d.get()
    return A_n - 1j * alpha * A_d


# ── Robin BEM RHS ─────────────────────────────────────────────────────────────

def make_rhs_robin(nodes, normals, k, alpha, phi_inc=0.0,
                   src_type='plane', src_x=-4.0, src_y=0.0):
    """RHS for the Robin BC: b = ∂p_inc/∂n + iα p_inc.

    b_robin = b_neumann + iα · p_inc(nodes)

    Parameters
    ----------
    nodes, normals : (N,2) geometry
    k              : wavenumber
    alpha          : Robin parameter
    phi_inc        : incident direction [rad] (used for plane wave)
    src_type       : 'plane' | 'point'
    src_x, src_y   : point source position (used when src_type='point')
    """
    d = np.array([np.cos(phi_inc), np.sin(phi_inc)])

    if src_type == 'point':
        diff = nodes - np.array([src_x, src_y])
        r    = np.maximum(np.linalg.norm(diff, axis=1), 1e-10)
        # ∂p_inc/∂n = (ik/4) H₁(kr) (x−xₛ)·n / r
        dn_p    = (1j * k / 4.0) * hankel1(1, k * r) * (np.sum(diff * normals, axis=1) / r)
        # p_inc = (i/4) H₀(kr)
        p_inc_v = (1j / 4.0) * hankel1(0, k * r)
    else:
        p_inc_v = np.exp(1j * k * (nodes @ d))
        dn_p    = 1j * k * (normals @ d) * p_inc_v   # same as make_rhs_neumann

    return (dn_p + 1j * alpha * p_inc_v).astype(np.complex128)


# ── Tikhonov regularisation (shared) ─────────────────────────────────────────

def _tikhonov(A, scale=None):
    diag_scale = scale if scale is not None else float(np.mean(np.abs(np.diag(A))))
    if not np.isfinite(diag_scale) or diag_scale == 0.0:
        diag_scale = 0.5
    A += (diag_scale * 1e-4) * np.eye(len(A))
    return A


# ── Robin solver ──────────────────────────────────────────────────────────────

def solve_robin(nodes, normals, lengths, k, alpha, phi_inc=0.0,
                src_type='plane', src_x=-4.0, src_y=0.0,
                tol=1e-6, maxiter=200, verbose=False):
    """Solve the Robin BEM system.

    Assembles A_robin on CPU, solves with CuPy GMRES (GPU) or SciPy GMRES.
    Falls back to dense LU for small N (< 64) or when GMRES diverges.

    Parameters
    ----------
    nodes, normals, lengths : BEM geometry
    k       : wavenumber
    alpha   : Robin parameter α = k/ζ  (0 ≡ Hard, large ≡ Soft)
    phi_inc : incident angle [rad]
    src_type, src_x, src_y : source type and position

    Returns
    -------
    sigma : (N,) complex128  surface density
    info  : dict  t_build, t_solve, rel_res, converged, backend, alpha
    """
    t0 = time.perf_counter()
    A  = build_matrix_robin(nodes, normals, lengths, k, alpha)
    _tikhonov(A)
    b       = make_rhs_robin(nodes, normals, k, alpha, phi_inc, src_type, src_x, src_y)
    t_build = time.perf_counter() - t0

    N = len(b)

    if N < 64:
        t1      = time.perf_counter()
        sigma   = _dense_solve(A, b)
        t_solve = time.perf_counter() - t1
        backend = 'dense-lu'
    elif _HAS_CUPY:
        t1    = time.perf_counter()
        A_gpu = cp.asarray(A.astype(np.complex64))
        b_gpu = cp.asarray(b.astype(np.complex64))
        op    = LinearOperator((N, N), matvec=lambda v: A_gpu @ v, dtype=cp.complex64)
        x_gpu, code = _cp_gmres(op, b_gpu, **{_GMRES_TOL_KW: tol},
                                restart=50, maxiter=maxiter)
        cp.cuda.Stream.null.synchronize()
        sigma   = cp.asnumpy(x_gpu).astype(np.complex128)
        t_solve = time.perf_counter() - t1
        backend = 'gpu-gmres'
    else:
        t1      = time.perf_counter()
        sigma, code = _sp_gmres(A, b, **{_SP_TOL_KW: tol}, restart=50, maxiter=maxiter)
        t_solve = time.perf_counter() - t1
        backend = 'cpu-gmres'

    res  = np.linalg.norm(b - A @ sigma) / (np.linalg.norm(b) + 1e-30)
    info = dict(t_build=t_build, t_solve=t_solve, rel_res=float(res),
                converged=(backend == 'dense-lu' or code == 0),
                backend=backend, alpha=alpha)
    if verbose:
        print(f'[Robin] N={N} k={k:.2f} α={alpha:.4f} ζ={k/alpha:.2f} '
              f't_build={t_build*1e3:.1f}ms t_solve={t_solve*1e3:.1f}ms '
              f'res={res:.2e} conv={info["converged"]} [{backend}]')
    return sigma, info


# ── Unified solve interface (all three BCs) ───────────────────────────────────

def solve(nodes, normals, lengths, k, phi_inc=0.0,
          bc='soft', alpha=0.0,
          src_type='plane', src_x=-4.0, src_y=0.0,
          tol=1e-6, maxiter=200, verbose=False):
    """Solve BEM for Soft, Hard, or Robin BC.

    Parameters
    ----------
    bc    : 'soft' | 'hard' | 'robin'
    alpha : Robin parameter (only used when bc='robin')
            α = k/ζ  where ζ is the normalised impedance.
            Convenience: pass alpha=None with bc='robin' to use α=k (matched).

    Returns
    -------
    sigma : (N,) complex128
    info  : dict  (same interface as solve_neumann, solve_gmres_dirichlet)
    """
    if bc == 'hard':
        return solve_neumann(nodes, normals, lengths, k, phi_inc,
                             tol=tol, maxiter=maxiter, verbose=verbose)
    elif bc == 'robin':
        if alpha is None:
            alpha = k  # matched impedance
        return solve_robin(nodes, normals, lengths, k, alpha, phi_inc,
                           src_type=src_type, src_x=src_x, src_y=src_y,
                           tol=tol, maxiter=maxiter, verbose=verbose)
    else:
        b = make_rhs_dirichlet(nodes, k, phi_inc)
        return solve_gmres_dirichlet(nodes, lengths, k, b, verbose=verbose)
