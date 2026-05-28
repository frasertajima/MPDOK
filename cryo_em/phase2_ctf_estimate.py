"""
Phase 2 — Per-tilt CTF estimation.

Estimates defocus for each tilt in a cryo-ET series by fitting the CTF model
to the incoherent average power spectrum of non-overlapping patches.

Algorithm (CTFFIND4-style, simplified)
---------------------------------------
For each tilt image:
  1. Divide into 512×512 patches (ignoring borders)
  2. Compute 2D power spectrum of each patch
  3. Average all patch power spectra → incoherent average (suppresses sample
     structure, which is incoherent across patches; preserves CTF, which is
     coherent across the whole image)
  4. Subtract a smooth radial background (boxcar average in k)
  5. Fit defocus: scan Δf in [0.3, 8.0] µm, maximise cross-correlation between
     background-subtracted PS and theoretical CTF²(k, Δf)

Output
------
Writes a CTFFIND4-style text file with one line per tilt:
    z_index  tilt_angle  defocus_um  (fit_score)

Usage
-----
    python -m v28e_cryo_em.production2.phase2_ctf_estimate \\
        --mrc  /path/to/run20.mrc \\
        --tlt  /path/to/run20_portal.tlt \\
        --out  /path/to/run20_defocus.txt \\
        --voltage-kv 300 --cs-mm 2.7 --amp-contrast 0.07
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

HERE       = os.path.dirname(os.path.abspath(__file__))
MPDOK_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, MPDOK_ROOT)


# ── CTF physics ───────────────────────────────────────────────────────────────

def electron_wavelength(voltage_kV: float) -> float:
    """Relativistic electron wavelength in Å."""
    V = voltage_kV * 1e3
    return 12.2643 / np.sqrt(V * (1.0 + V / 1.021e6))


def ctf_1d(k: np.ndarray, defocus_um: float,
           lam: float, Cs_mm: float, amp_contrast: float) -> np.ndarray:
    """
    CTF amplitude on a 1D spatial frequency array k (1/Å).

    CTF(k) = -sqrt(1-Q²) sin(χ) - Q cos(χ)
    χ(k)   = π λ Δf k²  -  π/2 λ³ Cs k⁴

    Parameters
    ----------
    k            : spatial frequency (1/Å)
    defocus_um   : defocus in µm (positive = underfocus)
    lam          : electron wavelength (Å)
    Cs_mm        : spherical aberration (mm)
    amp_contrast : amplitude contrast fraction
    """
    Cs  = Cs_mm * 1e7                       # mm → Å
    df  = defocus_um * 1e4                   # µm → Å
    Q   = amp_contrast
    chi = np.pi * lam * df * k**2 - 0.5 * np.pi * lam**3 * Cs * k**4
    return -np.sqrt(1 - Q**2) * np.sin(chi) - Q * np.cos(chi)


def ctf2_1d(k, defocus_um, lam, Cs_mm, amp_contrast):
    return ctf_1d(k, defocus_um, lam, Cs_mm, amp_contrast) ** 2


# ── Power spectrum estimation ─────────────────────────────────────────────────

def incoherent_avg_ps(image: np.ndarray, patch_size: int = 512) -> np.ndarray:
    """
    Incoherent average power spectrum from non-overlapping patches.

    Averaging suppresses sample structure (incoherent across patches) and
    preserves CTF (coherent).  Returns a (patch_size × patch_size) float64
    array of mean |FFT|².
    """
    H, W = image.shape
    ps_sum   = np.zeros((patch_size, patch_size), dtype=np.float64)
    n_patches = 0
    window    = np.outer(np.hanning(patch_size), np.hanning(patch_size))

    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size].astype(np.float64)
            patch -= patch.mean()
            F = np.fft.fft2(patch * window)
            ps_sum += np.abs(np.fft.fftshift(F))**2
            n_patches += 1

    return ps_sum / max(n_patches, 1)


def radial_average(ps2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially average a 2D power spectrum.  Returns (r_px, mean_ps)."""
    N  = ps2d.shape[0]
    cy, cx = N // 2, N // 2
    y, x   = np.ogrid[:N, :N]
    r      = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    r_max  = min(cy, cx)
    mean_ps = np.array([ps2d[r == i].mean() if (r == i).any() else 0.0
                        for i in range(r_max)])
    return np.arange(r_max, dtype=float), mean_ps


def subtract_background(ps1d: np.ndarray, window: int = 20) -> np.ndarray:
    """Subtract smooth background via boxcar (rolling) average."""
    from numpy.lib.stride_tricks import sliding_window_view
    bg = np.convolve(ps1d, np.ones(window)/window, mode='same')
    return ps1d - bg


# ── Per-tilt defocus fit ──────────────────────────────────────────────────────

def fit_defocus(ps2d: np.ndarray, pixel_size: float,
                lam: float, Cs_mm: float, amp_contrast: float,
                df_min_um: float = 0.3, df_max_um: float = 8.0,
                k_min_inv_A: float = 0.02, k_max_inv_A: float = 0.20,
                ) -> tuple[float, float]:
    """
    Fit defocus to a 2D power spectrum by 1D cross-correlation.

    Returns (defocus_um, fit_score).
    """
    r_px, ps1d = radial_average(ps2d)
    ps_bg = subtract_background(ps1d, window=15)

    # Convert pixel radius → spatial frequency (1/Å)
    N     = ps2d.shape[0]
    k_arr = r_px / (N * pixel_size)           # 1/Å

    # Restrict to informative frequency range
    mask  = (k_arr >= k_min_inv_A) & (k_arr <= k_max_inv_A) & (r_px > 2)
    if mask.sum() < 10:
        return 3.0, 0.0   # fallback

    k_fit  = k_arr[mask]
    ps_fit = ps_bg[mask]
    # Normalise to unit variance
    ps_fit = ps_fit / (np.std(ps_fit) + 1e-12)

    def neg_xcorr(df):
        th = ctf2_1d(k_fit, df, lam, Cs_mm, amp_contrast)
        th -= th.mean()
        th /= (np.std(th) + 1e-12)
        return -float(np.dot(ps_fit, th))

    result = minimize_scalar(neg_xcorr, bounds=(df_min_um, df_max_um),
                             method='bounded',
                             options={'xatol': 0.05, 'maxiter': 50})
    return float(result.x), -float(result.fun)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        prog='phase2_ctf_estimate',
        description='Per-tilt CTF defocus estimation (CTFFIND4-style)',
    )
    p.add_argument('--mrc',          required=True)
    p.add_argument('--tlt',          required=True)
    p.add_argument('--out',          required=True, help='Output defocus .txt')
    p.add_argument('--voltage-kv',   type=float, default=300.0)
    p.add_argument('--cs-mm',        type=float, default=2.7)
    p.add_argument('--amp-contrast', type=float, default=0.07)
    p.add_argument('--patch-size',   type=int,   default=512,
                   help='Patch size for incoherent PS averaging  [512]')
    p.add_argument('--pixel-size',   type=float, default=None)
    args = p.parse_args()

    import mrcfile
    from v28e_cryo_em.workflow_demo.ctf_model import load_tlt_file

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Phase 2: CTF estimation  |  {os.path.basename(args.mrc)}")
    print(f"{'='*60}")

    with mrcfile.open(args.mrc, permissive=True) as f:
        stack     = f.data            # (n_tilts, H, W) — memory-mapped, not loaded
        px_header = float(f.voxel_size.x)

    pixel_size = args.pixel_size or px_header
    lam        = electron_wavelength(args.voltage_kv)
    n_tilts    = stack.shape[0]

    print(f"  {n_tilts} tilts  pixel={pixel_size:.3f}Å  λ={lam:.4f}Å  "
          f"Cs={args.cs_mm}mm  Q={args.amp_contrast}")

    tilt_angles = load_tlt_file(args.tlt)
    assert len(tilt_angles) == n_tilts

    rows = []
    for i in range(n_tilts):
        t1 = time.time()
        img   = stack[i].astype(np.float32)
        ps2d  = incoherent_avg_ps(img, patch_size=args.patch_size)
        df_um, score = fit_defocus(ps2d, pixel_size, lam,
                                   args.cs_mm, args.amp_contrast)
        rows.append((i, tilt_angles[i], df_um, score))
        print(f"  tilt {i:2d}  {tilt_angles[i]:+7.2f}°  "
              f"defocus={df_um:.3f}µm  score={score:.3f}  "
              f"({time.time()-t1:.1f}s)")

    # Write CTFFIND4-compatible output
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write("# z_index  tilt_angle  defocus_um  fit_score\n")
        fh.write(f"# voltage={args.voltage_kv}kV  Cs={args.cs_mm}mm  "
                 f"amp_contrast={args.amp_contrast}  pixel={pixel_size}A\n")
        for z, angle, df, score in rows:
            fh.write(f"{z:3d}  {angle:+8.4f}  {df:.4f}  {score:.4f}\n")

    defoci = np.array([r[2] for r in rows])
    print(f"\n  Defocus range: {defoci.min():.3f}–{defoci.max():.3f} µm  "
          f"(mean={defoci.mean():.3f} µm)")
    print(f"  Saved → {args.out}")
    print(f"  Total: {time.time()-t0:.0f}s")

    # Quick diagnostic plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    angles = [r[1] for r in rows]
    scores = [r[3] for r in rows]

    axes[0].plot(angles, defoci, 'o-', color='steelblue', ms=5)
    axes[0].set_xlabel('Tilt angle (°)'); axes[0].set_ylabel('Defocus (µm)')
    axes[0].set_title('Per-tilt defocus'); axes[0].grid(alpha=0.3)

    axes[1].plot(angles, scores, 's-', color='tomato', ms=5)
    axes[1].set_xlabel('Tilt angle (°)'); axes[1].set_ylabel('Fit score')
    axes[1].set_title('CTF fit quality'); axes[1].grid(alpha=0.3)

    fig.suptitle(f'CTF estimation  |  {os.path.basename(args.mrc)}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    plot_path = args.out.replace('.txt', '_plot.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot  → {plot_path}")


if __name__ == '__main__':
    main()
