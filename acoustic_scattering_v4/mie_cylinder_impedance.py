"""
mie_cylinder_impedance.py — Exact 2D Mie series for an impedance (Robin BC) cylinder.

Boundary condition:  ∂p/∂r + iα p = 0  at r = R

    α = k / ζ   where ζ is the normalised specific acoustic impedance.

Limiting cases
--------------
    α → 0   (ζ → ∞, rigid wall):  recovers Sound-Hard (Neumann) Mie series
    α → ∞   (ζ → 0, open surface): recovers Sound-Soft (Dirichlet) Mie series
    ζ = 1   (matched impedance):   maximum absorption, zero far-field scatter

Physical interpretation
-----------------------
For a locally-reacting surface obeying  p = ζ ρc v_n  (Ingard normalisation),
the Robin parameter is  α = k / ζ.

    ζ ≈ 25   glass / concrete  →  α = k/25   ~4% absorption per reflection
    ζ ≈ 8    plasterboard      →  α = k/8    ~12%
    ζ ≈ 3    carpet / earth    →  α = k/3    ~33%
    ζ ≈ 1.5  acoustic foam     →  α = k/1.5  ~67%
    ζ → 0    open surface      →  Soft BC

Scattered field coefficients
-----------------------------
Expanding  p_inc = exp(ikr cos(φ−φ_inc)) = Σₙ εₙ iⁿ Jₙ(kr) cos(n(φ−φ_inc))
and        p_scat = Σₙ εₙ iⁿ cₙ Hₙ⁽¹⁾(kr) cos(n(φ−φ_inc))

Applying ∂p_total/∂r + iα p_total = 0 at r = R:

    cₙ = − [ k Jₙ'(kR) + iα Jₙ(kR) ]
           ─────────────────────────────
           [ k Hₙ⁽¹⁾'(kR) + iα Hₙ⁽¹⁾(kR) ]

Far-field amplitude:
    F(φ) = Σₙ εₙ cₙ cos(n(φ − φ_inc))

2D bistatic RCS (scattering width):
    σ_2D(φ) = (4/k) |F(φ)|²

This normalisation matches bem_helmholtz_v2.eval_rcs_2d exactly.

References
----------
Morse & Ingard, "Theoretical Acoustics" §7.2
Colton & Kress, "Inverse Acoustic and Electromagnetic Scattering Theory" §3.4
"""

import numpy as np
from scipy.special import jv, hankel1


# ── Derivative helpers (reused from mie_cylinder_2d_hard) ────────────────────

def _jnp(n, z):
    """Jₙ'(z) using recurrence: −J₁(z) for n=0, else (Jₙ₋₁ − Jₙ₊₁)/2."""
    if n == 0:
        return -jv(1, z)
    return (jv(n - 1, z) - jv(n + 1, z)) / 2.0


def _h1np(n, z):
    """Hₙ⁽¹⁾'(z) using recurrence: −H₁⁽¹⁾(z) for n=0, else (Hₙ₋₁ − Hₙ₊₁)/2."""
    if n == 0:
        return -hankel1(1, z)
    return (hankel1(n - 1, z) - hankel1(n + 1, z)) / 2.0


# ── Series truncation ─────────────────────────────────────────────────────────

def n_terms_auto(k, R):
    """Truncation order: ceil(2kR) + 25 (conservative, always converges)."""
    return max(int(np.ceil(2 * k * R)) + 25, 20)


# ── Core: scattering coefficients ────────────────────────────────────────────

def mie_coeffs_impedance(k, R, alpha, n_terms=None):
    """Scattering coefficients cₙ for the impedance cylinder.

    cₙ = −[k Jₙ'(kR) + iα Jₙ(kR)] / [k Hₙ'(kR) + iα Hₙ(kR)]

    Parameters
    ----------
    k       : wavenumber > 0
    R       : cylinder radius > 0
    alpha   : Robin parameter  α = k/ζ  (0 = Hard, ∞ → Soft)
    n_terms : series truncation (default: auto)

    Returns
    -------
    coeffs : (n_terms,) complex128
    """
    n_terms = n_terms or n_terms_auto(k, R)
    kR = k * R
    coeffs = np.empty(n_terms, dtype=np.complex128)
    for n in range(n_terms):
        num = k * _jnp(n, kR) + 1j * alpha * jv(n, kR)
        den = k * _h1np(n, kR) + 1j * alpha * hankel1(n, kR)
        coeffs[n] = -num / den
    return coeffs


# ── Far-field and RCS ─────────────────────────────────────────────────────────

def mie_far_field_impedance(phi_obs, k, R, alpha, phi_inc=0.0, n_terms=None):
    """Far-field amplitude F(φ) for the impedance cylinder.

    F(φ) = Σₙ εₙ cₙ cos(n(φ − φ_inc))

    p_scat(r,φ) → F(φ) √(2/(πkr)) exp(i(kr − π/4))   as r → ∞
    """
    phi_obs = np.asarray(phi_obs, dtype=float)
    n_terms = n_terms or n_terms_auto(k, R)
    coeffs  = mie_coeffs_impedance(k, R, alpha, n_terms)

    F = np.zeros_like(phi_obs, dtype=np.complex128)
    for n, cn in enumerate(coeffs):
        eps_n = 1.0 if n == 0 else 2.0
        F += eps_n * cn * np.cos(n * (phi_obs - phi_inc))
    return F


def mie_rcs_2d_impedance(phi_obs, k, R, alpha, phi_inc=0.0, n_terms=None):
    """2D bistatic RCS σ_2D(φ) = (4/k)|F(φ)|² [metres].

    Normalisation matches bem_helmholtz_v2.eval_rcs_2d.
    """
    F = mie_far_field_impedance(phi_obs, k, R, alpha, phi_inc, n_terms)
    return (4.0 / k) * np.abs(F) ** 2


# ── Near-field (full domain) ──────────────────────────────────────────────────

def mie_scattered_field_impedance(x, y, k, R, alpha, phi_inc=0.0, n_terms=None):
    """Scattered pressure p_scat(x, y) for the impedance cylinder.

    Returns complex128 array; points inside the cylinder are set to 0.
    """
    x, y    = np.asarray(x, float), np.asarray(y, float)
    r       = np.sqrt(x**2 + y**2)
    phi     = np.arctan2(y, x)
    n_terms = n_terms or n_terms_auto(k, R)
    coeffs  = mie_coeffs_impedance(k, R, alpha, n_terms)

    p = np.zeros_like(r, dtype=np.complex128)
    outside = r > R
    r_out   = r[outside]
    phi_out = phi[outside]

    for n, cn in enumerate(coeffs):
        eps_n = 1.0 if n == 0 else 2.0
        p[outside] += (eps_n * (1j ** n) * cn
                       * hankel1(n, k * r_out)
                       * np.cos(n * (phi_out - phi_inc)))
    return p


def mie_total_field_impedance(x, y, k, R, alpha, phi_inc=0.0, n_terms=None):
    """Total field p_total = p_inc + p_scat for the impedance cylinder.

    Interior points (r < R) are NaN.
    """
    x, y  = np.asarray(x, float), np.asarray(y, float)
    d     = np.array([np.cos(phi_inc), np.sin(phi_inc)])
    p_inc = np.exp(1j * k * (x * d[0] + y * d[1]))
    p_scat = mie_scattered_field_impedance(x, y, k, R, alpha, phi_inc, n_terms)
    total  = p_inc + p_scat
    total[np.sqrt(x**2 + y**2) < R] = np.nan
    return total


# ── Absorption coefficient ────────────────────────────────────────────────────

def absorption_coefficient(alpha, k):
    """Normal-incidence energy absorption coefficient  A = 1 − |R|²

    For a flat rigid surface with Robin BC ∂p/∂n + iαp = 0 (outward normal
    pointing into the fluid), the plane-wave reflection coefficient at
    normal incidence is:

        R = (k − α) / (k + α)   →   A = 1 − R²  =  4kα / (k + α)²

    Limiting cases:
        α = 0  (Hard):   R = 1,   A = 0   (perfect reflection, in-phase)
        α = k  (matched ζ=1): R = 0,   A = 1   (total absorption)
        α → ∞  (Soft):   R = −1,  A = 0   (total reflection, phase-reversed)

    Note: this formula holds for real α and normal incidence on a flat surface.
    For a cylinder the angle-averaged absorption is lower; the Mie series
    gives the exact scattered power.
    """
    if alpha == 0:
        return 0.0
    R = (k - alpha) / (k + alpha)
    return float(1.0 - R ** 2)


# ── BEM comparison helper ─────────────────────────────────────────────────────

def compare_bem_mie_impedance(nodes, lengths, sigma, k, R, alpha,
                               phi_inc=0.0, N_phi=360):
    """Compare Robin BEM far-field against Mie series.

    Parameters
    ----------
    nodes   : (N, 2) float64  panel midpoints
    lengths : (N,)   float64  panel arc-lengths
    sigma   : (N,)   complex128  BEM surface density
    k       : wavenumber
    R       : cylinder radius
    alpha   : Robin parameter
    phi_inc : incident angle [rad]
    N_phi   : observation angles

    Returns
    -------
    dict with phi_arr, rcs_bem, rcs_mie, rcs_db_bem, rcs_db_mie,
              max_err_db, rms_err_db
    """
    import sys, os
    _here = os.path.dirname(os.path.abspath(__file__))
    for _p in [os.path.join(_here, '..', 'acoustic_scattering_v2'),
               os.path.join(_here, '..', 'acoustic_scattering')]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from bem_helmholtz_v2 import eval_rcs_2d

    phi_arr    = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    rcs_bem    = eval_rcs_2d(nodes, lengths, sigma, k, phi_arr)
    rcs_mie    = mie_rcs_2d_impedance(phi_arr, k, R, alpha, phi_inc)
    rcs_db_bem = 10.0 * np.log10(np.maximum(rcs_bem, 1e-20))
    rcs_db_mie = 10.0 * np.log10(np.maximum(rcs_mie, 1e-20))
    diff       = np.abs(rcs_db_bem - rcs_db_mie)

    return dict(
        phi_arr=phi_arr,
        rcs_bem=rcs_bem,    rcs_mie=rcs_mie,
        rcs_db_bem=rcs_db_bem, rcs_db_mie=rcs_db_mie,
        max_err_db=float(diff.max()),
        rms_err_db=float(np.sqrt(np.mean(diff**2))),
    )
