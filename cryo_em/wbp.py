"""
Weighted Back-Projection (WBP) reconstruction.

Standard filtered backprojection using the sparse A matrix already built for
PnP-CTF-in-A.  No CTF correction — this is the baseline that PnP-CTF-in-A
should beat.

Algorithm
---------
1. Apply 2D ramp filter to each projection:
       b_filt_i = IFFT2( |k| × FFT2(b_i) )
   where |k| = sqrt(kx² + ky²) / (P/2)  (normalised Nyquist = 1).
   The ramp compensates for the over-sampling of low frequencies in the
   backprojection sum (same as standard filtered backprojection / FBP).

2. Backproject all filtered projections:
       x_wbp = Aᵀ b_filt    (using the same sparse A as PnP-CTF-in-A)

3. Normalise by the number of tilts so units match the PnP reconstruction.
"""

import numpy as np
import time


def ramp_filter_2d(proj: np.ndarray) -> np.ndarray:
    """
    Apply a 2D ramp filter to a single projection (P×P float32/64).

    Returns a real-valued filtered projection of the same shape.
    """
    P = proj.shape[0]
    F = np.fft.fftfreq(P)                     # [-0.5, 0.5)
    kx, ky = np.meshgrid(F, F, indexing='ij')
    ramp = np.sqrt(kx**2 + ky**2)             # |k|, peak = sqrt(2)/2 ≈ 0.707
    ramp /= ramp.max() + 1e-12                 # normalise to [0, 1]
    spec = np.fft.fft2(proj.astype(np.float64))
    return np.real(np.fft.ifft2(spec * ramp)).astype(np.float32)


def reconstruct_wbp(A_csr, b_raw: np.ndarray, vol_size: int,
                    proj_size: int, n_tilts: int,
                    verbose: bool = True) -> np.ndarray:
    """
    Weighted backprojection reconstruction.

    Parameters
    ----------
    A_csr    : scipy CSR sparse matrix  (n_tilts×P² × N³)
    b_raw    : (n_tilts × P²,) float32  — raw projections (flattened)
    vol_size : int  N — reconstruction is N³
    proj_size: int  P
    n_tilts  : int
    verbose  : bool

    Returns
    -------
    vol : (N, N, N) float32
    """
    import scipy.sparse as sp

    P2 = proj_size * proj_size
    t0 = time.time()

    # ── 1. Apply ramp filter to each tilt ────────────────────────────────────
    if verbose:
        print(f"  [WBP] Ramp-filtering {n_tilts} projections ({proj_size}²)…",
              end=" ", flush=True)
    b_filt = np.empty_like(b_raw)
    for i in range(n_tilts):
        proj = b_raw[i * P2:(i + 1) * P2].reshape(proj_size, proj_size)
        b_filt[i * P2:(i + 1) * P2] = ramp_filter_2d(proj).ravel()
    if verbose:
        print(f"{time.time()-t0:.1f}s")

    # ── 2. Backproject ───────────────────────────────────────────────────────
    if verbose:
        print(f"  [WBP] Backprojecting…", end=" ", flush=True)
    t1 = time.time()
    x = A_csr.T @ b_filt.astype(np.float32)
    x = x.astype(np.float64) / n_tilts
    if verbose:
        print(f"{time.time()-t1:.1f}s")

    return x.reshape(vol_size, vol_size, vol_size).astype(np.float32)
