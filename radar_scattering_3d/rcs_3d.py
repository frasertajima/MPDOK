"""
rcs_3d.py — RCS computation layer for radar_scattering_3d.

Re-exports the core far-field/RCS functions from bem_cobol/bem_3d.py and
adds three new functions specific to the Stage 2–7 pipeline:

  incident_grid(n_theta, n_phi)
      Uniform sphere grid of incident directions (bin-midpoint spacing,
      avoids north/south poles).  Default 6×12 = 72 directions.

  obs_grid(n_theta, n_phi)
      Uniform sphere grid for observation directions.
      Default 18×36 = 648 directions.

  bistatic_sphere_sweep(nodes, areas, sigma, k, theta_arr, phi_arr)
      GPU-accelerated RCS on the full (n_theta × n_phi) observation grid.
      Returns (n_theta, n_phi) float64 array [m²].

  build_rhs_matrix(nodes, k, inc_dirs)
      Build (N, M) complex128 RHS matrix in Fortran column-major layout for
      all M incident directions — the B argument to solve_multi_rhs.
"""

import sys
import os
import numpy as np

_BEM_COBOL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'bem_cobol')
if _BEM_COBOL not in sys.path:
    sys.path.insert(0, _BEM_COBOL)

from bem_3d import (
    far_field_3d,
    rcs_3d,
    rcs_3d_db,
    make_rhs_3d,
    sphere_obs_grid,
    backscatter_direction,
)

__all__ = [
    'far_field_3d', 'rcs_3d', 'rcs_3d_db',
    'make_rhs_3d', 'backscatter_direction',
    'incident_grid', 'obs_grid',
    'bistatic_sphere_sweep', 'bistatic_sphere_sweep_batch', 'build_rhs_matrix',
]


# ── Angle grids ────────────────────────────────────────────────────────────

def incident_grid(n_theta=6, n_phi=12):
    """Uniform incident-direction grid on the unit sphere.

    Uses bin-midpoint elevations: theta = π/n_theta * (i + 0.5) for
    i = 0..n_theta-1.  This avoids the north/south poles (which are
    degenerate in phi) and distributes elevation bands uniformly in [0, π].

    Args:
        n_theta: Number of elevation bands   (default 6  → 30° spacing).
        n_phi:   Number of azimuth steps     (default 12 → 30° spacing).

    Returns:
        dirs:   (n_theta*n_phi, 3) float64 unit vectors.
        theta:  (n_theta,) polar angles [rad].
        phi:    (n_phi,)   azimuthal angles [rad].
    """
    theta = np.pi / n_theta * (np.arange(n_theta) + 0.5)
    phi   = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    dirs = np.stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH),
    ], axis=-1).reshape(-1, 3)
    return dirs, theta, phi


def obs_grid(n_theta=18, n_phi=36):
    """Uniform observation-direction grid on the unit sphere.

    Same bin-midpoint spacing as incident_grid.  Default 18×36 = 648 dirs.

    Returns:
        dirs:   (n_theta*n_phi, 3) float64 unit vectors.
        theta:  (n_theta,) polar angles [rad].
        phi:    (n_phi,)   azimuthal angles [rad].
    """
    return incident_grid(n_theta=n_theta, n_phi=n_phi)


# ── GPU-accelerated full bistatic sphere sweep ─────────────────────────────

def bistatic_sphere_sweep(nodes, areas, sigma, k, theta_arr, phi_arr):
    """Compute bistatic RCS over a full (n_theta × n_phi) observation sphere.

    Builds the (M, N) phase matrix on GPU and returns the 2D RCS grid.
    Significantly faster than looping over observation directions in Python.

    Args:
        nodes:     (N, 3) float64  — panel centroids.
        areas:     (N,)   float64  — panel areas.
        sigma:     (N,)   complex  — BEM surface current (CPU array).
        k:         float  — wavenumber.
        theta_arr: (n_theta,) polar observation angles [rad].
        phi_arr:   (n_phi,)   azimuthal observation angles [rad].

    Returns:
        rcs_grid:  (n_theta, n_phi) float64 [m²] on CPU.
    """
    import cupy as cp

    n_t = len(theta_arr)
    n_p = len(phi_arr)

    TH, PH = np.meshgrid(theta_arr, phi_arr, indexing='ij')
    obs_dirs = np.stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH),
    ], axis=-1).reshape(-1, 3)          # (M, 3)

    dirs_d    = cp.asarray(obs_dirs)
    nodes_d   = cp.asarray(nodes)
    weights_d = cp.asarray((sigma * areas).astype(np.complex128))

    # phases: (M, N)
    phases_d = cp.exp((-1j * k) * (dirs_d @ nodes_d.T))
    f_d      = (1.0 / (4.0 * np.pi)) * (phases_d @ weights_d)  # (M,)
    rcs_d    = 4.0 * np.pi * cp.abs(f_d) ** 2                  # (M,)

    return cp.asnumpy(rcs_d).reshape(n_t, n_p)


# ── RHS matrix builder ─────────────────────────────────────────────────────

def build_rhs_matrix(nodes, k, inc_dirs):
    """Build (N, M) Fortran column-major RHS matrix for M incident directions.

    Each column j is -exp(ik d_j · x_i), the incident pressure at panel i
    for direction d_j.

    Args:
        nodes:    (N, 3) float64 — panel centroids.
        k:        float  — wavenumber.
        inc_dirs: (M, 3) float64 — unit incident direction vectors.

    Returns:
        B: (N, M) complex128 NumPy, Fortran column-major (order='F').
    """
    N = nodes.shape[0]
    M = len(inc_dirs)
    B = np.zeros((N, M), dtype=np.complex128, order='F')
    for j, d in enumerate(inc_dirs):
        d = d / np.linalg.norm(d)
        B[:, j] = make_rhs_3d(nodes, k, d)
    return B


# ── Batched bistatic sweep — all M incident dirs at once ──────────────────

def bistatic_sphere_sweep_batch(nodes, areas, sigma_matrix, k, theta_arr, phi_arr):
    """GPU-accelerated RCS for all M incident directions simultaneously.

    Replaces M sequential bistatic_sphere_sweep calls with a single
    (n_obs, N) @ (N, M) GPU matmul — typically 50-100x faster.

    Args:
        nodes:        (N, 3) float64  — panel centroids.
        areas:        (N,)   float64  — panel areas.
        sigma_matrix: (N, M) complex128 — column j = surface current for
                      incident direction j.
        k:            float  — wavenumber.
        theta_arr:    (n_theta,) observation polar angles [rad].
        phi_arr:      (n_phi,)   observation azimuthal angles [rad].

    Returns:
        rcs_cube: (M, n_theta, n_phi) float64 [m²] on CPU.
    """
    import cupy as cp

    n_t = len(theta_arr)
    n_p = len(phi_arr)
    M   = sigma_matrix.shape[1]

    TH, PH = np.meshgrid(theta_arr, phi_arr, indexing='ij')
    obs_dirs = np.stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH),
    ], axis=-1).reshape(-1, 3)                             # (n_obs, 3)

    nodes_d   = cp.asarray(nodes)
    obs_d     = cp.asarray(obs_dirs)
    weights_d = cp.asarray(sigma_matrix * areas[:, None])  # (N, M)

    phases_d  = cp.exp((-1j * k) * (obs_d @ nodes_d.T))   # (n_obs, N)
    f_d = (1.0 / (4.0 * np.pi)) * (phases_d @ weights_d)  # (n_obs, M)

    rcs_flat = cp.asnumpy(4.0 * np.pi * cp.abs(f_d) ** 2) # (n_obs, M)
    return rcs_flat.T.reshape(M, n_t, n_p)                 # (M, n_theta, n_phi)
