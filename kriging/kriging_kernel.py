"""
Kriging covariance matrix builder — GPU-accelerated via GEMM trick.

Supports three standard variogram models:
  'exponential'  : C[i,j] = σ² exp(-d/l)
  'gaussian'     : C[i,j] = σ² exp(-d²/2l²)   (same as RBF)
  'matern32'     : C[i,j] = σ²(1 + √3 d/l) exp(-√3 d/l)

Distance uses the GEMM identity:
    ||a-b||² = ||a||² + ||b||² - 2 a·b

so the only O(N²·D) op is one DGEMM — no N²×D tensor ever materialised.

Usage:
    from MPDOK.kriging.kriging_kernel import build_kriging_cov, synthetic_field
    coords, z = synthetic_field(N=50_000)
    C, length_scale = build_kriging_cov(coords)
"""

import ctypes
import cupy as cp
import numpy as np


# ── length-scale estimation ───────────────────────────────────────────────────

def estimate_length_scale(coords, max_sample=512):
    """Estimate l as the median pairwise distance on a subsample.

    Uses median(sqrt(D²)) = median(D) — identical to the CPU path in
    build_kriging_cov_cpu so both backends get the same length scale.
    """
    n = min(max_sample, coords.shape[0])
    c = coords[:n]
    sq = cp.sum(c ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (c @ c.T)
    cp.maximum(D2, 0.0, out=D2)
    D = cp.sqrt(D2)
    median_d = float(cp.median(D[D > 1e-15]))
    return median_d if median_d > 1e-15 else 1.0


# ── kernel functions (applied element-wise to a distance chunk) ───────────────

def _apply_kernel(D2, model, l, sigma2):
    """Return kernel values for a (rows, N) D² chunk.  In-place where possible."""
    if model == 'gaussian':
        K = cp.exp((-1.0 / (2.0 * l * l)) * D2)
    elif model == 'exponential':
        D = cp.sqrt(D2)
        K = cp.exp(-D / l)
    elif model == 'matern32':
        D = cp.sqrt(D2) * (1.7320508075688772 / l)   # √3/l * d
        K = (1.0 + D) * cp.exp(-D)
    else:
        raise ValueError(f"Unknown model '{model}'. Use gaussian/exponential/matern32.")
    if sigma2 != 1.0:
        K *= sigma2
    return K


# ── main builder ──────────────────────────────────────────────────────────────

def build_kriging_cov(coords, model='matern32', length_scale=None, sigma2=1.0,
                      nugget=1e-6, out=None, chunk=1024):
    """Build the N×N kriging covariance matrix on the GPU.

    Args:
        coords:       (N, D) FP64 CuPy array of observation coordinates.
        model:        Variogram model: 'gaussian' | 'exponential' | 'matern32'.
        length_scale: Correlation length l.  Auto-estimated if None.
        sigma2:       Variance scale (default 1.0).
        nugget:       Diagonal regularisation for numerical SPD (default 1e-6).
        out:          (N, N) FP64 CuPy array to fill.  Fresh allocation if None.
                      Pass solver.alloc_managed(N) for out-of-core.
        chunk:        Rows per GPU pass (tune for VRAM budget).

    Returns:
        (C, length_scale): C is the covariance matrix, l is the bandwidth used.
    """
    coords = cp.asarray(coords, dtype=cp.float64)
    N = coords.shape[0]

    if length_scale is None:
        length_scale = estimate_length_scale(coords)

    sq = cp.sum(coords ** 2, axis=1)   # (N,) reused every chunk

    if out is None:
        C = cp.empty((N, N), dtype=cp.float64, order='F')
    else:
        C = out.array if hasattr(out, 'array') else out

    for i in range(0, N, chunk):
        rows = min(chunk, N - i)
        c_chunk = coords[i:i + rows]
        sq_chunk = sq[i:i + rows]
        D2 = sq_chunk[:, None] + sq[None, :] - 2.0 * (c_chunk @ coords.T)
        cp.maximum(D2, 0.0, out=D2)
        C[i:i + rows, :] = _apply_kernel(D2, model, length_scale, sigma2)

    idx = cp.arange(N)
    C[idx, idx] += nugget

    cp.cuda.Stream.null.synchronize()
    return C, length_scale


def build_kriging_cov_cpu(coords_np, model='matern32', length_scale=None,
                           sigma2=1.0, nugget=1e-6, chunk=512):
    """CPU fallback — returns a NumPy F-order array.

    Used for the scipy baseline and for filling managed memory from host.
    """
    N = coords_np.shape[0]
    row_sq = np.sum(coords_np ** 2, axis=1)

    if length_scale is None:
        n = min(512, N)
        c = coords_np[:n]
        sq = row_sq[:n]
        D2 = sq[:, None] + sq[None, :] - 2.0 * (c @ c.T)
        np.maximum(D2, 0.0, out=D2)
        med = np.median(np.sqrt(D2[D2 > 1e-30]))
        length_scale = float(med) if med > 1e-30 else 1.0

    C = np.empty((N, N), dtype=np.float64, order='F')

    for j in range(0, N, chunk):
        end = min(j + chunk, N)
        gram = coords_np @ coords_np[j:end].T
        D2 = row_sq[:, None] + row_sq[None, j:end] - 2.0 * gram
        np.maximum(D2, 0.0, out=D2)

        if model == 'gaussian':
            K = np.exp((-1.0 / (2.0 * length_scale ** 2)) * D2)
        elif model == 'exponential':
            K = np.exp(-np.sqrt(D2) / length_scale)
        elif model == 'matern32':
            D = np.sqrt(D2) * (1.7320508075688772 / length_scale)
            K = (1.0 + D) * np.exp(-D)
        else:
            raise ValueError(f"Unknown model '{model}'.")

        if sigma2 != 1.0:
            K *= sigma2
        C[:, j:end] = K

    np.fill_diagonal(C, C.diagonal() + nugget)
    return C, length_scale


# ── synthetic benchmark field ─────────────────────────────────────────────────

def synthetic_coords(N, D=2, seed=42, domain=100.0):
    """Generate N random coordinates uniformly in [0, domain]^D."""
    rng = cp.random.default_rng(seed)
    return rng.uniform(0.0, domain, size=(N, D)).astype(cp.float64)


def synthetic_field(N, D=2, seed=42, domain=100.0, noise=0.02):
    """Generate N random sensor locations + smooth ground-truth field values.

    The field is a sum of Gaussian bumps — smooth, visually verifiable,
    and stress-tests the solver across the full spatial frequency range.

    Returns:
        coords: (N, 2) FP64 CuPy array
        z:      (N,)  FP64 CuPy array of noisy observations
    """
    rng = cp.random.default_rng(seed)
    coords = rng.uniform(0.0, domain, size=(N, D)).astype(cp.float64)

    # Ground truth: 5 Gaussian bumps
    n_bumps = 5
    bump_rng = np.random.default_rng(seed + 1)
    centers = bump_rng.uniform(10.0, domain - 10.0, size=(n_bumps, D))
    widths  = bump_rng.uniform(8.0, 20.0, size=n_bumps)
    amps    = bump_rng.uniform(0.5, 2.0, size=n_bumps)

    z = cp.zeros(N, dtype=cp.float64)
    for k in range(n_bumps):
        c = cp.asarray(centers[k])
        d2 = cp.sum((coords - c) ** 2, axis=1)
        z += float(amps[k]) * cp.exp(-d2 / (2.0 * float(widths[k]) ** 2))

    if noise > 0:
        z += rng.standard_normal(N, dtype=cp.float64) * noise

    return coords, z.astype(cp.float64)


def prediction_grid(domain=100.0, resolution=200):
    """Return a (resolution², 2) array of grid points for interpolation."""
    xs = cp.linspace(0.0, domain, resolution)
    ys = cp.linspace(0.0, domain, resolution)
    xx, yy = cp.meshgrid(xs, ys)
    grid = cp.stack([xx.ravel(), yy.ravel()], axis=1).astype(cp.float64)
    return grid, resolution
