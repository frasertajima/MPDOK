"""
bem_helmholtz_v2.py — GPU-accelerated 2D Helmholtz BEM for acoustic scattering.

Key improvement over bem_helmholtz.py (v1):

  1. Matrix assembly: CUDA Fortran kernel (bem_assembly.so) instead of
     CPU scipy.special.hankel1 — 150–400× faster at N=2k–8k.

  2. Solve: CuPy GMRES on complex N×N instead of block-real (2N)×(2N) LU.
     Half the problem size; iterative so it handles N>8k where LU goes OOM.

  3. Iterative refinement: Fortran py_bem_solve_ir gives GMRES c64 floor
     (~1e-5) improved to c128 floor (~1e-12) in one extra GPU pass.

  4. Multi-RHS efficiency: py_bem_solve_multi_rhs builds A once and solves
     M incident directions — used by Stage 7 bistatic sweep.

The Green's function and all physics are unchanged:
    G(x,y) = (i/4) H₀⁽¹⁾(k|x−y|)  (identical to v1 and to radar_scattering)

The 2D acoustic Helmholtz BEM and the 2D radar EM BEM are mathematically
identical — the same Fortran CUDA kernel serves both domains.

Public API
----------
    build_matrix(nodes, lengths, k, precision='c64')   → cupy (N,N)
    make_rhs(nodes, k, phi_inc)                        → np (N,) complex128
    solve_gmres(nodes, lengths, k, b, ...)             → (sigma, info)
    solve_ir(nodes, lengths, k, phi_inc, ...)          → (sigma, info)
    solve_multi_rhs(nodes, lengths, k, phi_arr, ...)   → (X, info)
    eval_far_field(nodes, lengths, sigma, k, phi_obs)  → np (M,) complex128
    eval_total_field(nodes, lengths, sigma, grid_pts, k, phi_inc)
    eval_total_field_gpu(...)   — GPU asymptotic (kr > 2), fast
"""

import sys
import os
import time

import numpy as np
from scipy.special import hankel1

_HERE  = os.path.dirname(os.path.abspath(__file__))
_MPDOK = os.path.join(_HERE, '..')
_RADAR = os.path.join(_MPDOK, 'radar_scattering')

for _p in [_RADAR, _MPDOK]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bem_assembly_ops import BEMAssembler

try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import gmres as cp_gmres, LinearOperator
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# ── RHS ───────────────────────────────────────────────────────────────────────

def make_rhs(nodes, k, phi_inc=0.0):
    """Incident plane-wave RHS for sound-soft Dirichlet BC.

    p_inc(x) = exp(ik x·d),  d = (cos φ, sin φ)
    b_i = −p_inc(x_i)  [so that p_total = 0 on Γ]

    Returns complex128 (N,) array.
    """
    d = np.array([np.cos(phi_inc), np.sin(phi_inc)])
    return -np.exp(1j * k * (nodes @ d)).astype(np.complex128)


# ── Matrix assembly ───────────────────────────────────────────────────────────

def build_matrix(nodes, lengths, k, precision='c64'):
    """GPU BEM matrix assembly via Fortran CUDA kernel.

    Args:
        nodes:     (N, 2) float64 panel centroids.
        lengths:   (N,)   float64 panel arc lengths.
        k:         Wavenumber.
        precision: 'c64' or 'c128'.

    Returns:
        A: cupy (N,N) complex64 or complex128, already in VRAM.
    """
    asm = BEMAssembler()
    if precision == 'c128':
        return asm.build_c128(nodes, lengths, k)
    return asm.build_c64(nodes, lengths, k)


# ── Solvers ───────────────────────────────────────────────────────────────────

def solve_gmres(nodes, lengths, k, b,
                restart=50, tol=1e-6, maxiter=20,
                A_gpu=None, verbose=False):
    """Solve A σ = b via CuPy GMRES on the complex N×N system.

    Builds A on GPU if not supplied. Uses the GPU-resident A directly
    without extracting to CPU.

    Args:
        nodes, lengths: geometry.
        k:       Wavenumber.
        b:       (N,) complex128 RHS (NumPy).
        restart: GMRES restart (default 50).
        tol:     Convergence tolerance.
        maxiter: Max outer restarts.
        A_gpu:   Pre-built cupy (N,N) complex64 — skip assembly if provided.
        verbose: Print timing.

    Returns:
        sigma: (N,) complex128 NumPy surface density.
        info:  dict with t_build, t_solve, rel_res, backend.
    """
    if not HAS_GPU:
        raise RuntimeError('CuPy not available — cannot use solve_gmres')

    N = len(b)
    t0_build = time.perf_counter()
    if A_gpu is None:
        A_gpu = build_matrix(nodes, lengths, k, precision='c64')
    t_build = time.perf_counter() - t0_build

    b_d = cp.asarray(b, dtype=cp.complex64)
    def mv(v): return A_gpu @ v
    op = LinearOperator((N, N), matvec=mv, dtype=cp.complex64)

    t0_solve = time.perf_counter()
    x_d, code = cp_gmres(op, b_d, tol=tol, restart=restart, maxiter=maxiter)
    cp.cuda.Stream.null.synchronize()
    t_solve = time.perf_counter() - t0_solve

    sigma = cp.asnumpy(x_d).astype(np.complex128)
    res = np.linalg.norm(b - (cp.asnumpy(A_gpu) @ sigma.astype(np.complex64)).astype(np.complex128)) / np.linalg.norm(b)

    info = dict(t_build=t_build, t_solve=t_solve, rel_res=float(res),
                converged=(code == 0), backend='gpu-gmres')
    if verbose:
        print(f'[GMRES] N={N} k={k:.1f} t_build={t_build:.3f}s '
              f't_solve={t_solve:.3f}s res={res:.2e} conv={code==0}')
    return sigma, info


def solve_ir(nodes, lengths, k, phi_inc=0.0,
             restart=50, tol=1e-6, maxiter_ir=2, verbose=False):
    """Fortran pipeline: GPU build → GMRES → iterative refinement.

    Calls py_bem_solve_ir inside bem_assembly.so — the matrix build,
    GMRES iterations, and IR residual computation all stay on GPU.

    Returns:
        sigma: (N,) complex128 NumPy.
        info:  dict with t_total, rel_res, converged, backend.
    """
    return BEMAssembler().solve_ir(
        nodes, lengths, k, phi_inc,
        restart=restart, tol=tol, maxiter_ir=maxiter_ir, verbose=verbose)


def solve_multi_rhs(nodes, lengths, k, phi_arr,
                    restart=50, tol=1e-6, verbose=False):
    """1 GPU build + M incident-direction solves (for bistatic sweep).

    Args:
        phi_arr: (M,) incident angles [rad].

    Returns:
        X:    (N, M) complex128 — column j = σ for incident j.
        info: dict with t_total, t_per_solve, n_converged.
    """
    N = nodes.shape[0]
    M = len(phi_arr)
    B = np.zeros((N, M), dtype=np.complex128, order='F')
    for j, phi in enumerate(phi_arr):
        B[:, j] = make_rhs(nodes, k, phi)
    return BEMAssembler().solve_multi_rhs(
        nodes, lengths, k, B,
        restart=restart, tol=tol, verbose=verbose)


# ── Far-field and total field ─────────────────────────────────────────────────

def eval_far_field(nodes, lengths, sigma, k, phi_obs):
    """Far-field amplitude f(φ_obs) for the scattered field.

    p_scat ~ f(φ) √(2/(πkr)) exp(i(kr − π/4))  as  r → ∞

    f(φ) = (i/4) Σ_j exp(−ik x_j · r̂(φ)) σ_j Δl_j   (panel sum)

    Args:
        nodes:   (N, 2) panel centroids.
        lengths: (N,)   arc lengths.
        sigma:   (N,)   complex128 surface current.
        k:       Wavenumber.
        phi_obs: scalar or (M,) observation angles [rad].

    Returns:
        f: complex128 scalar or (M,) array.
    """
    phi_obs = np.asarray(phi_obs, dtype=float)
    scalar  = phi_obs.ndim == 0
    phi_obs = np.atleast_1d(phi_obs)

    r_hat = np.stack([np.cos(phi_obs), np.sin(phi_obs)], axis=1)  # (M,2)
    phase  = np.exp(-1j * k * (r_hat @ nodes.T))                   # (M,N)
    weights = sigma * lengths                                        # (N,)
    f = (1j / 4.0) * (phase @ weights)                             # (M,)
    return complex(f[0]) if scalar else f


def eval_rcs_2d(nodes, lengths, sigma, k, phi_obs):
    """2D RCS (scattering width) from BEM surface current [metres]."""
    f = eval_far_field(nodes, lengths, sigma, k, phi_obs)
    return (4.0 / k) * np.abs(f) ** 2


def eval_total_field(nodes, lengths, sigma, grid_pts, k, phi_inc=0.0, chunk=4096):
    """Exact total field at grid_pts (CPU, exact Hankel).

    Identical to v1 but with cleaner interface. Use for near-field validation.
    For visualisation at large grids (>64k pts) prefer eval_total_field_gpu.
    """
    d     = np.array([np.cos(phi_inc), np.sin(phi_inc)])
    p_inc = np.exp(1j * k * (grid_pts @ d))
    w     = (1j / 4.0) * sigma * lengths

    sq_n = np.sum(nodes**2, axis=1)
    sq_g = np.sum(grid_pts**2, axis=1)
    M    = grid_pts.shape[0]
    p_sc = np.zeros(M, dtype=complex)

    for i in range(0, M, chunk):
        j  = min(i + chunk, M)
        D2 = sq_g[i:j, None] + sq_n[None, :] - 2.0 * (grid_pts[i:j] @ nodes.T)
        np.maximum(D2, 1e-30, out=D2)
        R  = np.sqrt(D2)
        p_sc[i:j] = hankel1(0, k * R) @ w

    return p_inc + p_sc


def eval_total_field_gpu(nodes, lengths, sigma, grid_pts, k,
                          phi_inc=0.0, chunk=8192):
    """GPU field evaluation — asymptotic H₀ approximation (kr > 2).

    Accurate to ~1% for kr > 5. Fast for large visualisation grids.
    Identical algorithm to v1 eval_total_field_gpu.
    """
    if not HAS_GPU:
        return eval_total_field(nodes, lengths, sigma, grid_pts, k, phi_inc, chunk)

    d     = np.array([np.cos(phi_inc), np.sin(phi_inc)])
    p_inc = np.exp(1j * k * (grid_pts @ d))

    w_full = (1j / 4.0) * sigma * lengths
    wr_g   = cp.asarray(w_full.real, dtype=cp.float64)
    wi_g   = cp.asarray(w_full.imag, dtype=cp.float64)
    nodes_g = cp.asarray(nodes, dtype=cp.float64)
    grid_g  = cp.asarray(grid_pts, dtype=cp.float64)
    sq_n    = cp.sum(nodes_g**2, axis=1)
    sq_g    = cp.sum(grid_g**2,  axis=1)
    M       = grid_g.shape[0]
    pr_g    = cp.zeros(M, dtype=cp.float64)
    pi_g    = cp.zeros(M, dtype=cp.float64)

    for i in range(0, M, chunk):
        j   = min(i + chunk, M)
        D2  = sq_g[i:j, None] + sq_n[None, :] - 2.0 * (grid_g[i:j] @ nodes_g.T)
        cp.maximum(D2, 1e-30, out=D2)
        z   = k * cp.sqrt(D2)
        amp = cp.sqrt(2.0 / (cp.pi * z))
        ph  = z - cp.pi / 4.0
        H0r = amp * cp.cos(ph)
        H0i = amp * cp.sin(ph)
        pr_g[i:j] += H0r @ wr_g - H0i @ wi_g
        pi_g[i:j] += H0r @ wi_g + H0i @ wr_g

    cp.cuda.Stream.null.synchronize()
    return p_inc + cp.asnumpy(pr_g) + 1j * cp.asnumpy(pi_g)
