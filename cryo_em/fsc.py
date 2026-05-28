"""
Fourier Shell Correlation (FSC) for cryo-ET resolution estimation.

Gold-standard FSC
-----------------
Split the tilt series into two independent half-datasets (odd/even tilt indices).
Reconstruct each half independently.  Compute FSC between the two half-volumes.

    FSC(r) = Σ_{|k|=r} F1(k) × F2*(k)
             ─────────────────────────────────────────
             sqrt( Σ|F1(k)|² × Σ|F2(k)|² )

Resolution is quoted at FSC = 0.143 (gold-standard; Rosenthal & Henderson 2003).
The 0.5 threshold is more conservative and commonly used in cryo-ET.

Half-dataset split
------------------
Splitting by odd/even tilt index gives two independent datasets that together
cover the same angular range.  This differs from the strict "independent
half-datasets" used in SPA (where particles are split randomly), but is standard
for single-tomogram FSC in cryo-ET.
"""

import numpy as np


def fsc(vol1: np.ndarray, vol2: np.ndarray,
        n_shells: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FSC between two volumes of identical shape.

    Parameters
    ----------
    vol1, vol2 : (N, N, N) float32/64
    n_shells   : int  number of radial shells  [N//2]

    Returns
    -------
    freq   : (n_shells,) float  — spatial frequency in units of Nyquist (0→1)
    fsc_v  : (n_shells,) float  — FSC value per shell
    """
    N = vol1.shape[0]
    if n_shells is None:
        n_shells = N // 2

    F1 = np.fft.fftn(vol1.astype(np.float64))
    F2 = np.fft.fftn(vol2.astype(np.float64))

    # Radial coordinate for each voxel in Fourier space
    idx  = np.fft.fftfreq(N) * N          # integer frequency coords [-N/2, N/2)
    z, y, x = np.meshgrid(idx, idx, idx, indexing='ij')
    r = np.sqrt(x**2 + y**2 + z**2)

    fsc_vals = np.zeros(n_shells)
    freq     = np.zeros(n_shells)

    for s in range(n_shells):
        r_lo = s
        r_hi = s + 1
        mask = (r >= r_lo) & (r < r_hi)
        if not mask.any():
            continue
        f1 = F1[mask]
        f2 = F2[mask]
        num  = np.real(np.sum(f1 * np.conj(f2)))
        denom = np.sqrt(np.sum(np.abs(f1)**2) * np.sum(np.abs(f2)**2)) + 1e-12
        fsc_vals[s] = num / denom
        freq[s]     = (s + 0.5) / (N / 2)   # normalised to Nyquist

    return freq, fsc_vals


def resolution_at_threshold(freq: np.ndarray, fsc_vals: np.ndarray,
                             pixel_size: float,
                             threshold: float = 0.143) -> float:
    """
    Return resolution in Å at the first crossing of `threshold`.

    Parameters
    ----------
    freq       : normalised spatial frequency (0→1 = DC→Nyquist)
    fsc_vals   : FSC per shell
    pixel_size : Å/voxel
    threshold  : FSC threshold  [0.143]

    Returns
    -------
    float — resolution in Å, or Nyquist (2×pixel_size) if never crossed
    """
    nyquist = 2.0 * pixel_size
    for i in range(1, len(fsc_vals)):
        if fsc_vals[i] < threshold and fsc_vals[i-1] >= threshold:
            # Linear interpolation
            f = fsc_vals[i-1] + (threshold - fsc_vals[i-1]) * \
                (freq[i] - freq[i-1]) / (fsc_vals[i] - fsc_vals[i-1] + 1e-12)
            f = max(f, 1e-6)
            return (pixel_size * 2) / f   # Å
    return nyquist   # never crossed — resolution = Nyquist


def half_indices(n_tilts: int) -> tuple[np.ndarray, np.ndarray]:
    """Split n_tilts into odd/even index halves."""
    all_idx = np.arange(n_tilts)
    return all_idx[::2], all_idx[1::2]


def plot_fsc(results: list[dict], pixel_size: float, ax=None):
    """
    Plot FSC curves for multiple reconstructions on one axis.

    Parameters
    ----------
    results : list of dicts with keys 'label', 'freq', 'fsc'
    pixel_size : Å/voxel (used for x-axis in Å)
    ax      : matplotlib Axes  [creates one if None]
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    nyquist = 2.0 * pixel_size

    for r, color in zip(results, colors):
        freq, fsc_v = r['freq'], r['fsc']
        # Convert normalised freq → Å (avoid div/0)
        res_a = np.where(freq > 0, (nyquist) / freq, np.inf)
        res143 = resolution_at_threshold(freq, fsc_v, pixel_size, 0.143)
        res05  = resolution_at_threshold(freq, fsc_v, pixel_size, 0.500)
        label  = (f"{r['label']}  "
                  f"(0.143→{res143:.1f}Å, 0.5→{res05:.1f}Å)")
        ax.plot(res_a, fsc_v, '-', color=color, lw=2, label=label)

    ax.axhline(0.143, color='black', ls='--', lw=1, alpha=0.6, label='FSC=0.143')
    ax.axhline(0.500, color='gray',  ls=':',  lw=1, alpha=0.6, label='FSC=0.5')
    ax.set_xlim(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else nyquist * 30, nyquist)
    ax.set_xlabel('Resolution (Å)')
    ax.set_ylabel('FSC')
    ax.set_title('Fourier Shell Correlation  (gold-standard half-dataset)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax
