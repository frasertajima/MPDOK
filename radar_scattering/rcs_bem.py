"""
rcs_bem.py — Radar cross-section (RCS) from 2D TM BEM surface currents.

The 2D TM EFIE for a PEC target uses the same Helmholtz Green's function as
acoustic scattering:

    G(x, y) = (i/4) H₀⁽¹⁾(k |x − y|)

so bem_helmholtz.py is reused verbatim.  This module adds:

  far_field_amplitude()   Complex far-field amplitude S(φ) from surface current σ
  far_field_sweep()       Vectorised S(φ) for many observation angles
  rcs_2d()               Bistatic RCS in metres
  rcs_2d_sweep()         Vectorised bistatic RCS
  rcs_2d_db()            Bistatic RCS in dB·m
  solve_bem_scipy()      Build + solve via scipy LU (small N, Stage 1 baseline)
  solve_bem_mpdok()      Build + solve via MPDOK GMRES (large N, Stage 2)
  monostatic_sweep()     Full monostatic RCS pattern (360 incident angles)
  bistatic_pattern()     Full bistatic pattern at a fixed incident angle

Physics
-------
Far-field of the single-layer potential for large |x|:

    p_scat(x) ~ √(2 / πkr) · exp(i(kr − π/4)) · S(φ)

where S(φ) = (i/4) Σ_j exp(−ik x_j · r̂(φ)) σ_j Δl_j

The 2D bistatic RCS (scattering width) is:

    σ_2D(φ) = lim_{r→∞} [ 2πr · |p_scat|² / |p_inc|² ]
             = (4/k) |S(φ)|²          [metres]
"""

import sys, os
import numpy as np
from scipy.linalg import solve as sp_solve

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from acoustic_scattering.bem_helmholtz import (
    build_bem_matrix_helmholtz,
    to_block_real,
    rhs_to_real,
    sigma_from_real,
    make_rhs_helmholtz,
)


# ── Far-field computation ─────────────────────────────────────────────────────

def far_field_amplitude(nodes, lengths, sigma, k, phi_obs):
    """Complex far-field amplitude S(φ_obs) from surface current σ.

    S(φ) = (i/4) Σ_j exp(−ik x_j · r̂(φ)) σ_j Δl_j

    Args:
        nodes:    (N, 2) panel midpoints [m].
        lengths:  (N,)   panel arc lengths [m].
        sigma:    (N,)   complex surface current density.
        k:        Wavenumber [1/m].
        phi_obs:  Observation angle [rad]. Scalar.

    Returns:
        S: complex scalar.
    """
    r_hat = np.array([np.cos(phi_obs), np.sin(phi_obs)])
    phase = np.exp(-1j * k * (nodes @ r_hat))          # (N,) complex
    return (1j / 4.0) * np.dot(sigma * lengths, phase)


def far_field_sweep(nodes, lengths, sigma, k, phi_arr):
    """Vectorised far-field amplitude for many observation angles.

    Args:
        phi_arr: (M,) observation angles [rad].

    Returns:
        S: (M,) complex array.
    """
    phi_arr = np.asarray(phi_arr, dtype=float)
    r_hats  = np.stack([np.cos(phi_arr), np.sin(phi_arr)], axis=1)  # (M, 2)
    # phases[m, j] = exp(−ik x_j · r̂_m)
    phases  = np.exp(-1j * k * (r_hats @ nodes.T))                   # (M, N)
    weights = (sigma * lengths).astype(complex)                        # (N,)
    return (1j / 4.0) * (phases @ weights)                            # (M,)


# ── RCS from far-field amplitude ──────────────────────────────────────────────

def rcs_2d(nodes, lengths, sigma, k, phi_obs):
    """Bistatic 2D RCS at observation angle φ_obs [metres].

    σ_2D(φ_obs) = (4/k) |S(φ_obs)|²
    """
    S = far_field_amplitude(nodes, lengths, sigma, k, phi_obs)
    return (4.0 / k) * abs(S) ** 2


def rcs_2d_sweep(nodes, lengths, sigma, k, phi_arr):
    """Vectorised bistatic 2D RCS for a vector of observation angles [metres]."""
    S = far_field_sweep(nodes, lengths, sigma, k, phi_arr)
    return (4.0 / k) * np.abs(S) ** 2


def rcs_2d_db(nodes, lengths, sigma, k, phi_obs, eps=1e-20):
    """Bistatic 2D RCS in dB·m  (10 log₁₀ σ_2D)."""
    return 10.0 * np.log10(max(rcs_2d(nodes, lengths, sigma, k, phi_obs), eps))


def rcs_2d_sweep_db(nodes, lengths, sigma, k, phi_arr, eps=1e-20):
    """Vectorised bistatic 2D RCS in dB·m."""
    rcs = rcs_2d_sweep(nodes, lengths, sigma, k, phi_arr)
    return 10.0 * np.log10(np.maximum(rcs, eps))


# ── BEM solvers ───────────────────────────────────────────────────────────────

def solve_bem_scipy(nodes, lengths, k, phi_inc=0.0):
    """Build and solve the 2D TM BEM system with scipy LU.

    Suitable for N ≤ ~8k.  Exact FP64 LU.

    Args:
        nodes, lengths: panel geometry.
        k:              wavenumber.
        phi_inc:        incident angle [rad].

    Returns:
        sigma: (N,) complex128 surface current density.
        cond:  estimated condition number of A (via np.linalg.cond).
    """
    A     = build_bem_matrix_helmholtz(nodes, lengths, k)
    b, _  = make_rhs_helmholtz(nodes, k, phi_inc)
    sigma = sp_solve(A, b)
    return sigma


def solve_bem_mpdok(nodes, lengths, k, phi_inc=0.0,
                    tol=1e-7, restart=50, verbose=False):
    """Build and solve the 2D TM BEM system with MPDOK GMRES (TF32 matvec).

    Block-real split: complex N×N → real (2N)×(2N).  The TF32 tensor-core
    matvec provides ~20× speedup over CPU FP64 for the dominant Krylov cost.

    Args:
        nodes, lengths: panel geometry.
        k:              wavenumber.
        phi_inc:        incident angle [rad].
        tol:            GMRES convergence tolerance on relative residual.
        restart:        Krylov restart parameter m (default 50).
        verbose:        Print GMRES residual at each restart.

    Returns:
        sigma:     (N,) complex128 surface current density.
        converged: bool — True if GMRES met tol within maxiter.
        history:   list of (iters, rel_res) tuples from dense_krylov.gmres.
    """
    import cupy as cp
    from MPDOK.dense_krylov import DenseLinearOperator, gmres as mpdok_gmres

    A_c   = build_bem_matrix_helmholtz(nodes, lengths, k)
    A_r   = to_block_real(A_c)                     # (2N, 2N) real float64
    b_c, b_r = make_rhs_helmholtz(nodes, k, phi_inc)

    op    = DenseLinearOperator(A_r)
    x_gpu, hist, conv = mpdok_gmres(op, b_r, tol=tol, restart=restart,
                                    verbose=verbose)
    x_np  = cp.asnumpy(x_gpu)
    sigma = sigma_from_real(x_np)
    return sigma, conv, hist


# ── Pattern sweeps ────────────────────────────────────────────────────────────

def bistatic_pattern(nodes, lengths, k, phi_inc=0.0,
                     N_phi=720, solver='scipy'):
    """Full bistatic RCS pattern at a fixed incident angle.

    Solves once for σ at phi_inc, then evaluates far-field at N_phi angles.

    Args:
        nodes, lengths: geometry.
        k:              wavenumber.
        phi_inc:        incident angle [rad].
        N_phi:          number of observation angles.
        solver:         'scipy' or 'mpdok'.

    Returns:
        phi_arr: (N_phi,) observation angles.
        rcs:     (N_phi,) bistatic RCS [metres].
        rcs_db:  (N_phi,) bistatic RCS [dB·m].
        sigma:   (N,)     complex surface current.
    """
    if solver == 'mpdok':
        sigma, _, _ = solve_bem_mpdok(nodes, lengths, k, phi_inc)
    else:
        sigma = solve_bem_scipy(nodes, lengths, k, phi_inc)

    phi_arr = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    rcs     = rcs_2d_sweep(nodes, lengths, sigma, k, phi_arr)
    rcs_db  = 10.0 * np.log10(np.maximum(rcs, 1e-20))
    return phi_arr, rcs, rcs_db, sigma


def monostatic_sweep(nodes, lengths, k, phi_inc_arr, solver='scipy'):
    """Monostatic RCS pattern: solve BEM for each incident angle.

    Observation angle = incident angle + π (backscatter toward transmitter).
    Each angle requires a separate BEM solve (different RHS).

    Args:
        nodes, lengths: geometry.
        k:              wavenumber.
        phi_inc_arr:    (M,) incident angles [rad].
        solver:         'scipy' or 'mpdok'.

    Returns:
        rcs_arr: (M,) monostatic RCS [metres].
    """
    phi_inc_arr = np.asarray(phi_inc_arr, dtype=float)
    rcs_arr     = np.zeros(len(phi_inc_arr))

    for i, phi_inc in enumerate(phi_inc_arr):
        if solver == 'mpdok':
            sigma, _, _ = solve_bem_mpdok(nodes, lengths, k, float(phi_inc))
        else:
            sigma = solve_bem_scipy(nodes, lengths, k, float(phi_inc))

        phi_obs    = float(phi_inc) + np.pi
        rcs_arr[i] = rcs_2d(nodes, lengths, sigma, k, phi_obs)

    return rcs_arr


def condition_number(nodes, lengths, k):
    """Compute condition number of the BEM matrix (for diagnostics).

    Expensive O(N³) — only for small N validation.
    """
    A = build_bem_matrix_helmholtz(nodes, lengths, k)
    return np.linalg.cond(A)
