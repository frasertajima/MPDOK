"""
Phase 1 — WBP baseline + FSC comparison.

Runs four reconstructions on the same crop region:
    WBP half1,  WBP half2      → FSC_wbp
    PnP half1,  PnP half2      → FSC_pnp

Then plots both FSC curves on one axis so we can quote:
    "PnP-CTF-in-A resolves X Å vs WBP Y Å  (FSC=0.143, gold-standard halves)"

Usage
-----
    python -m v28e_cryo_em.production2.phase1_baseline \\
        --mrc  /path/to/run20.mrc \\
        --tlt  /path/to/run20.rawtlt \\
        --out  /path/to/phase1/

With the Chlamydomonas dataset at crop-256 (our best single-tomogram result):
    python -m v28e_cryo_em.production2.phase1_baseline \\
        --mrc  /var/home/fraser/cryo_em_v2/data/czii_10009_run20/run20.mrc \\
        --tlt  /var/home/fraser/cryo_em_v2/data/czii_10009_run20/run20.rawtlt \\
        --out  /var/home/fraser/cryo_em_v2/data/czii_10009_run20/phase1/ \\
        --crop-size 256 --defocus 3.0
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE       = os.path.dirname(os.path.abspath(__file__))
MPDOK_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, MPDOK_ROOT)

from v28e_cryo_em.production2.wbp import reconstruct_wbp
from v28e_cryo_em.production2.fsc import fsc, resolution_at_threshold, half_indices, plot_fsc
from v28e_cryo_em.production.pnp_ctf_reconstruct import build_A
from v28e_cryo_em.production.autotune import heuristic_sigma, heuristic_rho, heuristic_n_admm


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(args):
    """Load MRC, apply crop, resample, return projections + geometry."""
    from v28e_cryo_em.workflow_demo.ctf_model import (
        load_tilt_series_mrc, load_tlt_file,
    )
    from scipy.ndimage import zoom

    proj_raw, px_header = load_tilt_series_mrc(args.mrc)
    n_tilts, H, W = proj_raw.shape
    pixel_size = args.pixel_size or px_header

    if args.crop_size:
        cs = args.crop_size
        cx = W // 2; cy = H // 2
        x0, y0 = cx - cs // 2, cy - cs // 2
        proj_raw = proj_raw[:, y0:y0+cs, x0:x0+cs]
        H = W = cs
        print(f"  Cropped to {cs}×{cs}")

    tilt_angles = load_tlt_file(args.tlt)
    assert len(tilt_angles) == n_tilts, \
        f"Tilt count mismatch: {len(tilt_angles)} angles vs {n_tilts} MRC frames"

    vol_size  = args.vol_size  or (1 << int(np.log2(min(H, W) // 2)))
    proj_size = args.proj_size or int(vol_size * 1.5)

    effective_px = pixel_size * (min(H, W) / proj_size)
    print(f"  vol={vol_size}³  proj={proj_size}²  "
          f"pixel: {pixel_size:.3f}→{effective_px:.3f} Å  "
          f"Nyquist={2*effective_px:.2f} Å")

    if H != proj_size or W != proj_size:
        scale = proj_size / min(H, W)
        proj_raw = np.stack([
            zoom(proj_raw[i], scale, order=1)[:proj_size, :proj_size]
            for i in range(n_tilts)
        ]).astype(np.float32)

    return proj_raw, tilt_angles, vol_size, proj_size, effective_px


def build_half_A(tilt_angles, half_idx, vol_size, proj_size, cache_dir):
    """Build or load A for a subset of tilt angles."""
    angles_half = tilt_angles[half_idx]
    key = f"{vol_size}_{proj_size}_" + ",".join(f"{a:.4f}" for a in angles_half)
    cid = hashlib.md5(key.encode()).hexdigest()[:10]
    return build_A(angles_half, vol_size, proj_size, verbose=True,
                   cache_dir=cache_dir)


def load_defocus_file(path: str) -> np.ndarray:
    """
    Load per-tilt defocus values from a CTFFIND4-style text file.

    Returns (n_tilts,) float64 array of defocus values in µm, sorted by z_index.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            rows.append((int(parts[0]), float(parts[2])))   # z_index, defocus_um
    rows.sort()
    return np.array([r[1] for r in rows], dtype=np.float64)


def run_pnp(A_csr, b_raw, tilt_angles, vol_size, proj_size, pixel_size,
            defocus_arr, verbose=True):
    """Run PnP-CTF-in-A with full autotune on a (possibly half) dataset."""
    from v28e_cryo_em.workflow_demo.ctf_projector import CTFProjector, PnPCTFSolver
    from v28e_cryo_em.workflow_demo.denoisers import GaussianSpatial

    ctf_proj = CTFProjector(A_csr, tilt_angles, vol_size, proj_size, pixel_size,
                            defocus_per_tilt=defocus_arr)

    sigma   = heuristic_sigma(pixel_size)
    rho     = heuristic_rho(ctf_proj.ctf_power, sigma=sigma)
    n_admm  = heuristic_n_admm(rho, sigma=sigma)

    if verbose:
        print(f"  [PnP] σ={sigma:.2f}  ρ={rho:.3f}  n_admm={n_admm}")

    denoiser = GaussianSpatial(sigma=sigma)
    solver   = PnPCTFSolver(ctf_proj, vol_size, denoiser=denoiser)
    vol, hist = solver.solve(b_raw.ravel(), rho=rho,
                             n_admm=n_admm, n_cg=25, conv_tol=0.25,
                             verbose=verbose)
    return vol, hist


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        prog='phase1_baseline',
        description='Phase 1: WBP baseline + FSC comparison vs PnP-CTF-in-A',
    )
    p.add_argument('--mrc',        required=True)
    p.add_argument('--tlt',        required=True)
    p.add_argument('--out',        required=True, help='Output directory')
    p.add_argument('--defocus',    type=float, default=3.0,
                   help='Constant defocus µm  [3.0]  (ignored if --ctffind given)')
    p.add_argument('--ctffind',    default=None,
                   help='Per-tilt CTFFIND4-style defocus file (overrides --defocus)')
    p.add_argument('--crop-size',  type=int,   default=None)
    p.add_argument('--vol-size',   type=int,   default=None)
    p.add_argument('--proj-size',  type=int,   default=None)
    p.add_argument('--pixel-size', type=float, default=None)
    p.add_argument('--no-pnp',     action='store_true',
                   help='Skip PnP halves (WBP only — much faster)')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Phase 1: WBP baseline + FSC  |  {os.path.basename(args.mrc)}")
    print(f"{'='*60}")

    # ── Load ─────────────────────────────────────────────────────────────────
    proj, tilt_angles, vol_size, proj_size, pixel_size = load_data(args)
    n_tilts = len(tilt_angles)
    P2 = proj_size ** 2

    # ── Defocus array ─────────────────────────────────────────────────────────
    if args.ctffind:
        defocus_all = load_defocus_file(args.ctffind)
        assert len(defocus_all) == n_tilts, \
            f"ctffind file has {len(defocus_all)} rows but {n_tilts} tilts"
        print(f"  Per-tilt defocus from {os.path.basename(args.ctffind)}: "
              f"{defocus_all.min():.3f}–{defocus_all.max():.3f} µm "
              f"(mean={defocus_all.mean():.3f})")
    else:
        defocus_all = np.full(n_tilts, args.defocus)
        print(f"  Constant defocus: {args.defocus:.3f} µm")

    half1_idx, half2_idx = half_indices(n_tilts)
    print(f"  Half-datasets: half1={len(half1_idx)} tilts, half2={len(half2_idx)} tilts")

    # ── Build A matrices ─────────────────────────────────────────────────────
    print("\n  Building A matrices…")
    A_full = build_A(tilt_angles,        vol_size, proj_size, True, cache_dir=args.out)
    A_h1   = build_A(tilt_angles[half1_idx], vol_size, proj_size, True, cache_dir=args.out)
    A_h2   = build_A(tilt_angles[half2_idx], vol_size, proj_size, True, cache_dir=args.out)

    b_full = proj.reshape(-1)
    b_h1   = proj[half1_idx].reshape(-1)
    b_h2   = proj[half2_idx].reshape(-1)

    results_fsc = []

    # ── WBP ──────────────────────────────────────────────────────────────────
    print("\n  --- WBP ---")
    import mrcfile as mf

    def save(vol, name):
        path = os.path.join(args.out, name)
        with mf.new(path, overwrite=True) as m:
            m.set_data(vol.astype(np.float32))
            m.voxel_size = pixel_size
        return path

    t1 = time.time()
    wbp_full = reconstruct_wbp(A_full, b_full, vol_size, proj_size, n_tilts)
    wbp_h1   = reconstruct_wbp(A_h1,   b_h1,   vol_size, proj_size, len(half1_idx), verbose=False)
    wbp_h2   = reconstruct_wbp(A_h2,   b_h2,   vol_size, proj_size, len(half2_idx), verbose=False)
    print(f"  WBP done in {time.time()-t1:.1f}s")

    save(wbp_full, 'wbp_full.mrc')
    save(wbp_h1,   'wbp_half1.mrc')
    save(wbp_h2,   'wbp_half2.mrc')

    freq_wbp, fsc_wbp = fsc(wbp_h1, wbp_h2)
    res_wbp_143 = resolution_at_threshold(freq_wbp, fsc_wbp, pixel_size, 0.143)
    res_wbp_05  = resolution_at_threshold(freq_wbp, fsc_wbp, pixel_size, 0.500)
    print(f"  WBP FSC:  0.143→{res_wbp_143:.1f}Å   0.5→{res_wbp_05:.1f}Å")
    results_fsc.append({'label': 'WBP (no CTF)', 'freq': freq_wbp, 'fsc': fsc_wbp})

    # ── PnP-CTF-in-A ─────────────────────────────────────────────────────────
    if not args.no_pnp:
        print("\n  --- PnP-CTF-in-A ---")
        t2 = time.time()
        pnp_h1, hist1 = run_pnp(A_h1, proj[half1_idx], tilt_angles[half1_idx],
                                 vol_size, proj_size, pixel_size,
                                 defocus_all[half1_idx])
        pnp_h2, hist2 = run_pnp(A_h2, proj[half2_idx], tilt_angles[half2_idx],
                                 vol_size, proj_size, pixel_size,
                                 defocus_all[half2_idx])

        # Full reconstruction (for display)
        pnp_full, _ = run_pnp(A_full, proj, tilt_angles,
                               vol_size, proj_size, pixel_size, defocus_all)
        print(f"  PnP done in {time.time()-t2:.1f}s")

        save(pnp_full, 'pnp_full.mrc')
        save(pnp_h1,   'pnp_half1.mrc')
        save(pnp_h2,   'pnp_half2.mrc')

        freq_pnp, fsc_pnp = fsc(pnp_h1, pnp_h2)
        res_pnp_143 = resolution_at_threshold(freq_pnp, fsc_pnp, pixel_size, 0.143)
        res_pnp_05  = resolution_at_threshold(freq_pnp, fsc_pnp, pixel_size, 0.500)
        print(f"  PnP FSC:  0.143→{res_pnp_143:.1f}Å   0.5→{res_pnp_05:.1f}Å")
        results_fsc.append({'label': 'PnP-CTF-in-A', 'freq': freq_pnp, 'fsc': fsc_pnp})

    # ── Summary figure ────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # FSC plot
    plot_fsc(results_fsc, pixel_size, ax=axes[0])

    # Side-by-side XZ slices: WBP full vs PnP full
    mid = vol_size // 2
    vols_to_show = [('WBP full', wbp_full)]
    if not args.no_pnp:
        vols_to_show.append(('PnP-CTF full', pnp_full))

    n_show = len(vols_to_show)
    if n_show == 1:
        axes[1].set_visible(False)
        ax_img = axes[0] if False else fig.add_subplot(1, 2, 2)
    else:
        ax_img = axes[1]

    # split axes[1] manually
    fig.delaxes(axes[1])
    for k, (lbl, vol) in enumerate(vols_to_show):
        ax = fig.add_subplot(1, 2 + n_show - 1, 2 + k)
        sl = vol[:, mid, :]
        lo, hi = np.percentile(sl, 2), np.percentile(sl, 98)
        ext = [0, sl.shape[1]*pixel_size/10, 0, sl.shape[0]*pixel_size/10]
        ax.imshow(sl, cmap='gray', vmin=lo, vmax=hi,
                  origin='lower', extent=ext, aspect='equal')
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel('nm', fontsize=8)
        if k == 0: ax.set_ylabel('nm', fontsize=8)

    fig.suptitle(
        f'Phase 1: WBP vs PnP-CTF-in-A  |  {pixel_size:.2f}Å/px  '
        f'Nyquist={2*pixel_size:.1f}Å',
        fontsize=11, fontweight='bold',
    )
    fig.tight_layout()
    fig_path = os.path.join(args.out, 'phase1_fsc.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved figure → {fig_path}")

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary = {
        'pixel_size': pixel_size,
        'vol_size':   vol_size,
        'n_tilts':    n_tilts,
        'wbp': {'res_143': res_wbp_143, 'res_05': res_wbp_05},
    }
    if not args.no_pnp:
        summary['pnp'] = {'res_143': res_pnp_143, 'res_05': res_pnp_05}
        summary['improvement_143_A'] = res_wbp_143 - res_pnp_143
    with open(os.path.join(args.out, 'phase1_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done  ({time.time()-t0:.0f}s total)")
    if not args.no_pnp:
        print(f"  WBP resolution (FSC=0.143): {res_wbp_143:.1f} Å")
        print(f"  PnP resolution (FSC=0.143): {res_pnp_143:.1f} Å")
        print(f"  Improvement:                {res_wbp_143 - res_pnp_143:+.1f} Å")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
