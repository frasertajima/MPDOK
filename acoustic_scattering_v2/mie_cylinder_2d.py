"""
mie_cylinder_2d.py — Exact Mie series for 2D acoustic scattering.

Sound-soft (Dirichlet) boundary condition: p_total = 0 on the cylinder
surface.  This is the acoustic analogue of PEC in EM, and the Mie series
is mathematically identical to the electromagnetic TM case in radar_scattering.

Re-exports the validated functions from radar_scattering/mie_series.py under
acoustic naming conventions, and adds helper functions for BEM comparison.
"""

import sys
import os

import numpy as np
from scipy.special import jv, hankel1

_HERE  = os.path.dirname(os.path.abspath(__file__))
_RADAR = os.path.join(_HERE, '..', 'radar_scattering')
if _RADAR not in sys.path:
    sys.path.insert(0, _RADAR)

from mie_series import (
    mie_far_field        as _mie_far_field,
    mie_rcs_2d           as _mie_rcs_2d,
    mie_rcs_2d_db        as _mie_rcs_2d_db,
    mie_bistatic_pattern as _mie_bistatic_pattern,
    mie_frequency_sweep  as _mie_frequency_sweep,
)

# ── Re-exports with acoustic naming ──────────────────────────────────────────

def acoustic_far_field(k, R, phi_obs, phi_inc=0.0, N_terms=None):
    """Far-field amplitude S(φ_obs) for sound-soft scattering from a cylinder.

    p_scat(r, φ) ~ S(φ) √(2/(πkr)) exp(i(kr − π/4))  as r → ∞

    S(φ_obs, φ_inc) = −Σ_n exp(in(φ_obs − φ_inc)) J_n(kR) / H_n⁽¹⁾(kR)

    This is identical to the EM TM Mie series — the same Dirichlet BC governs
    both sound-soft acoustic and PEC electromagnetic scattering.

    Args:
        k:        Wavenumber (ω/c) > 0.
        R:        Cylinder radius > 0.
        phi_obs:  Observation angle(s) [rad].
        phi_inc:  Incident angle [rad] (default 0 → +x direction).
        N_terms:  Series truncation (default ceil(kR) + 15).

    Returns:
        S: complex scalar or (M,) array.
    """
    return _mie_far_field(k, R, phi_obs, phi_inc, N_terms)


def acoustic_rcs_2d(k, R, phi_obs, phi_inc=0.0, N_terms=None):
    """2D scattering cross-section (scattering width) [metres].

    σ_2D(φ_obs) = (4/k) |S(φ_obs, φ_inc)|²
    """
    return _mie_rcs_2d(k, R, phi_obs, phi_inc, N_terms)


def acoustic_rcs_db(k, R, phi_obs, phi_inc=0.0, N_terms=None):
    """2D scattering cross-section in dBm (10 log₁₀ σ_2D)."""
    return _mie_rcs_2d_db(k, R, phi_obs, phi_inc, N_terms)


def bistatic_pattern(k, R, phi_inc=0.0, N_phi=720, N_terms=None):
    """Full bistatic RCS pattern over the observation sphere.

    Returns:
        phi_arr: (N_phi,) observation angles [rad].
        rcs:     (N_phi,) scattering cross-section [metres].
        rcs_db:  (N_phi,) in dBm.
    """
    return _mie_bistatic_pattern(k, R, phi_inc, N_phi, N_terms)


def frequency_sweep(R, ka_arr, phi_obs=np.pi, phi_inc=0.0):
    """Scattering cross-section vs electrical size kR."""
    return _mie_frequency_sweep(R, ka_arr, phi_obs, phi_inc)


# ── BEM comparison helpers ────────────────────────────────────────────────────

def compare_bem_mie(nodes, lengths, sigma, k, R,
                    phi_inc=0.0, N_phi=360):
    """Compare BEM far-field against exact Mie series.

    Args:
        nodes, lengths: (N,2) and (N,) BEM geometry.
        sigma:          (N,) complex128 BEM surface current.
        k:              Wavenumber.
        R:              Cylinder radius (for Mie series).
        phi_inc:        Incident angle [rad].
        N_phi:          Number of observation angles.

    Returns:
        dict with keys:
            phi_arr:   (N_phi,) observation angles.
            rcs_bem:   (N_phi,) BEM scattering cross-section [m].
            rcs_mie:   (N_phi,) Mie scattering cross-section [m].
            rcs_db_bem, rcs_db_mie: same in dBm.
            max_err_db: max |BEM_db − Mie_db| over phi_arr.
            rms_err_db: RMS |BEM_db − Mie_db|.
    """
    from bem_helmholtz_v2 import eval_rcs_2d

    phi_arr  = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    rcs_bem  = eval_rcs_2d(nodes, lengths, sigma, k, phi_arr)
    rcs_mie  = acoustic_rcs_2d(k, R, phi_arr, phi_inc)

    rcs_db_bem = 10.0 * np.log10(np.maximum(rcs_bem, 1e-20))
    rcs_db_mie = 10.0 * np.log10(np.maximum(rcs_mie, 1e-20))

    diff       = np.abs(rcs_db_bem - rcs_db_mie)
    return dict(
        phi_arr=phi_arr,
        rcs_bem=rcs_bem, rcs_mie=rcs_mie,
        rcs_db_bem=rcs_db_bem, rcs_db_mie=rcs_db_mie,
        max_err_db=float(diff.max()),
        rms_err_db=float(np.sqrt(np.mean(diff**2))),
    )


def mie_error_table(k_vals, N_vals, R=1.0, phi_inc=0.0,
                    solver_fn=None, verbose=True):
    """Compute BEM vs Mie error for a grid of (k, N) values.

    Args:
        k_vals:    list of wavenumbers.
        N_vals:    list of panel counts.
        R:         Cylinder radius.
        phi_inc:   Incident angle.
        solver_fn: callable(nodes, lengths, k, phi_inc) → sigma.
                   Defaults to solve_ir with maxiter_ir=0 (GMRES only).
        verbose:   Print table as computed.

    Returns:
        rows: list of dicts with k, N, max_err_db, rms_err_db, t_solve.
    """
    import sys, os
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, '..', 'acoustic_scattering'))
    from geometry import circle_panels

    from bem_helmholtz_v2 import solve_ir
    import time

    if solver_fn is None:
        def solver_fn(nodes, lengths, k, phi_inc):
            sigma, _ = solve_ir(nodes, lengths, k, phi_inc, maxiter_ir=0)
            return sigma

    if verbose:
        print(f'{"k":>5}  {"N":>6}  {"max|Δ| [dB]":>12}  {"rms|Δ| [dB]":>12}  {"t [s]":>8}')
        print('─' * 50)

    rows = []
    for k in k_vals:
        for N in N_vals:
            nodes, _, lengths = circle_panels(N, R=R)
            t0    = time.perf_counter()
            sigma = solver_fn(nodes, lengths, k, phi_inc)
            t     = time.perf_counter() - t0
            res   = compare_bem_mie(nodes, lengths, sigma, k, R, phi_inc)
            row   = dict(k=k, N=N,
                         max_err_db=res['max_err_db'],
                         rms_err_db=res['rms_err_db'],
                         t_solve=t)
            rows.append(row)
            if verbose:
                print(f'{k:>5.1f}  {N:>6d}  {res["max_err_db"]:>12.4f}  '
                      f'{res["rms_err_db"]:>12.4f}  {t:>8.3f}')
    return rows
