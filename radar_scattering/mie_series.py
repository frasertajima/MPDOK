"""
mie_series.py — Exact 2D Mie series for TM scattering from a PEC cylinder.

Physics:  An incident TM plane wave (E_z polarised) at wavenumber k scatters
off a perfectly electrically conducting (PEC) cylinder of radius R.  The exact
scattered field is a Mie series — a sum over cylindrical harmonics.

2D bistatic RCS (scattering width), units metres:

    σ_2D(φ_obs) = (4/k) |S(φ_obs, φ_inc)|²

where the far-field amplitude is:

    S(φ_obs, φ_inc) = −Σ_n exp(in(φ_obs − φ_inc)) J_n(kR) / H_n⁽¹⁾(kR)

Reference: Harrington, "Time-Harmonic Electromagnetic Fields", §6-4.
           Bohren & Huffman, "Absorption and Scattering of Light", §4.4 (2D).

Used exclusively for BEM validation.  Not needed in production.
"""

import numpy as np
from scipy.special import jv, hankel1


def mie_far_field(k, R, phi_obs, phi_inc=0.0, N_terms=None):
    """Far-field amplitude S(φ_obs) for TM scattering from a PEC cylinder.

    Args:
        k:        Wavenumber (ω/c) > 0.
        R:        Cylinder radius > 0.
        phi_obs:  Observation angle(s) [rad]. Scalar or (M,) array.
        phi_inc:  Incident plane-wave angle [rad] (default 0 → +x direction).
        N_terms:  Series truncation order. Default: ceil(kR) + 15.

    Returns:
        S: complex scalar or (M,) complex array — far-field amplitude.
    """
    kR = k * R
    if N_terms is None:
        N_terms = int(np.ceil(kR)) + 15

    n   = np.arange(-N_terms, N_terms + 1)  # (2P+1,)
    Jn  = jv(n, kR)                          # J_n(kR)
    Hn  = hankel1(n, kR)                     # H_n^(1)(kR)
    cn  = -Jn / Hn                           # Mie coefficients (n-vector)

    phi_obs = np.asarray(phi_obs, dtype=float)
    scalar  = phi_obs.ndim == 0
    phi_obs = np.atleast_1d(phi_obs)

    # S(φ) = Σ_n cn · exp(in(φ_obs − φ_inc))
    delta   = phi_obs[:, None] - phi_inc         # (M, 1) broadcast with n
    phase   = np.exp(1j * delta * n[None, :])    # (M, 2P+1)
    S       = phase @ cn                         # (M,)

    return complex(S[0]) if scalar else S


def mie_rcs_2d(k, R, phi_obs, phi_inc=0.0, N_terms=None):
    """2D bistatic RCS (scattering width) for a PEC cylinder [metres].

    σ_2D(φ_obs) = (4/k) |S(φ_obs, φ_inc)|²

    Args:
        k, R, phi_obs, phi_inc, N_terms: see mie_far_field().

    Returns:
        rcs: float or (M,) float array — RCS in metres.
    """
    S = mie_far_field(k, R, phi_obs, phi_inc, N_terms)
    return (4.0 / k) * np.abs(S) ** 2


def mie_rcs_2d_db(k, R, phi_obs, phi_inc=0.0, N_terms=None, eps=1e-20):
    """2D RCS in dB·m (10 log₁₀ σ_2D), clipped to avoid −inf."""
    rcs = mie_rcs_2d(k, R, phi_obs, phi_inc, N_terms)
    return 10.0 * np.log10(np.maximum(rcs, eps))


def mie_monostatic(k, R, phi_inc_arr, N_terms=None):
    """Monostatic 2D RCS: observation angle = incident angle + π (backscatter).

    Args:
        k, R:          Wavenumber and radius.
        phi_inc_arr:   (M,) incident angles [rad].
        N_terms:       Series truncation (default ceil(kR)+15).

    Returns:
        rcs: (M,) float array — monostatic RCS in metres.
    """
    phi_inc_arr = np.asarray(phi_inc_arr, dtype=float)
    # For a circular cylinder the monostatic RCS is independent of phi_inc
    # (azimuthal symmetry), but we compute it correctly for completeness.
    phi_obs_arr = phi_inc_arr + np.pi
    return np.array([
        mie_rcs_2d(k, R, float(phi_obs), float(phi_inc), N_terms)
        for phi_inc, phi_obs in zip(phi_inc_arr, phi_obs_arr)
    ])


def mie_bistatic_pattern(k, R, phi_inc=0.0, N_phi=720, N_terms=None):
    """Full bistatic RCS pattern as a function of observation angle.

    Args:
        k, R:      Wavenumber and radius.
        phi_inc:   Incident angle [rad].
        N_phi:     Number of observation angles (default 720 → 0.5° resolution).
        N_terms:   Series truncation.

    Returns:
        phi_arr: (N_phi,) observation angles [rad].
        rcs:     (N_phi,) RCS values [metres].
        rcs_db:  (N_phi,) RCS in dB·m.
    """
    phi_arr = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    rcs     = mie_rcs_2d(k, R, phi_arr, phi_inc, N_terms)
    rcs_db  = 10.0 * np.log10(np.maximum(rcs, 1e-20))
    return phi_arr, rcs, rcs_db


def mie_frequency_sweep(R, ka_arr, phi_obs=np.pi, phi_inc=0.0):
    """Monostatic RCS as electrical size kR varies (frequency sweep).

    Useful for showing Mie resonances (oscillatory RCS vs frequency).

    Args:
        R:        Cylinder radius.
        ka_arr:   (M,) array of kR values.
        phi_obs:  Observation angle (default π = backscatter).
        phi_inc:  Incident angle (default 0).

    Returns:
        rcs: (M,) RCS in metres.
    """
    ka_arr = np.asarray(ka_arr, dtype=float)
    return np.array([
        mie_rcs_2d(ka / R, R, phi_obs, phi_inc)
        for ka in ka_arr
    ])
