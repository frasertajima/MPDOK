"""
Phase 3 — Sub-tomogram averaging (STA).

Runs four non-overlapping 128³ crops (256-px offsets from detector centre)
to cover a ~262 nm combined FOV, giving enough particles (~20–40 ribosomes)
for meaningful gold-standard STA FSC comparison against the Phase 2 result.

Algorithm
---------
1.  Reconstruct four 128³ crops with PnP-CTF-in-A (each ~5 min)
2.  Template match each volume — spherical Gaussian blob (ribosome ~25 nm Ø)
3.  Pool all particle sub-volumes, assign to even/odd gold-standard halves
4.  Iterative translational alignment (5 rounds, reference = running mean)
5.  Gold-standard FSC: even-half average vs odd-half average
6.  Compare with single-tomogram Phase 2 FSC (9.1 Å baseline)

Usage
-----
    # Build 4 crops from scratch and run STA:
    python -m v28e_cryo_em.production2.phase3_sta \\
        --mrc     /path/to/run20.mrc \\
        --tlt     /path/to/run20_portal.tlt \\
        --ctffind /path/to/run20_defocus_smoothed.txt \\
        --out     /path/to/phase3/

    # Use pre-built volumes (one per crop):
    python -m v28e_cryo_em.production2.phase3_sta \\
        --pnp-mrcs vol1.mrc vol2.mrc vol3.mrc vol4.mrc \\
        --wbp-mrcs wbp1.mrc wbp2.mrc wbp3.mrc wbp4.mrc \\
        --out /path/to/phase3/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

HERE       = os.path.dirname(os.path.abspath(__file__))
MPDOK_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, MPDOK_ROOT)

from v28e_cryo_em.production2.fsc import fsc, resolution_at_threshold


# ── Crop offsets: four 256×256 px tiles in a 2×2 grid around detector centre ─
# Each is (dy, dx) in raw detector pixels from the image centre.
QUAD_OFFSETS = [(-256, -256), (-256, +256), (+256, -256), (+256, +256)]


# ── Template ─────────────────────────────────────────────────────────────────

def make_blob_template(box: int, sigma_vox: float) -> np.ndarray:
    c = box // 2
    z, y, x = np.ogrid[:box, :box, :box]
    r2 = (x - c)**2 + (y - c)**2 + (z - c)**2
    t = np.exp(-0.5 * r2 / sigma_vox**2).astype(np.float32)
    t -= t.mean()
    return t


# ── Template matching ─────────────────────────────────────────────────────────

def template_match_3d(vol: np.ndarray, template: np.ndarray) -> np.ndarray:
    N  = vol.shape[0]
    s  = template.shape[0]
    pad = np.zeros((N, N, N), dtype=np.float64)
    lo  = N // 2 - s // 2
    pad[lo:lo+s, lo:lo+s, lo:lo+s] = template.astype(np.float64)
    cc  = np.real(np.fft.ifftn(np.fft.fftn(vol.astype(np.float64)) *
                               np.conj(np.fft.fftn(pad))))
    cc  = np.fft.fftshift(cc)
    cc /= np.std(vol) * np.std(template) * vol.size + 1e-12
    return cc.astype(np.float32)


def nms_3d(cc: np.ndarray, min_dist: int, n_max: int,
           border: int = 36) -> tuple[np.ndarray, np.ndarray]:
    N   = cc.shape[0]
    cw  = cc.copy()
    cw[:border], cw[-border:] = -1e9, -1e9
    cw[:, :border], cw[:, -border:] = -1e9, -1e9
    cw[:, :, :border], cw[:, :, -border:] = -1e9, -1e9
    peaks, scores = [], []
    for _ in range(n_max):
        idx = np.argmax(cw)
        v   = cw.flat[idx]
        if v < 0:
            break
        z, y, x = np.unravel_index(idx, cw.shape)
        peaks.append((int(z), int(y), int(x)))
        scores.append(float(v))
        r = min_dist
        cw[max(0,z-r):min(N,z+r+1),
           max(0,y-r):min(N,y+r+1),
           max(0,x-r):min(N,x+r+1)] = -1e9
    return np.array(peaks, dtype=int), np.array(scores)


# ── Sub-volume extraction ─────────────────────────────────────────────────────

def extract_subvols(vol: np.ndarray, centres: np.ndarray,
                    box: int) -> tuple[np.ndarray, list[int]]:
    N, half = vol.shape[0], box // 2
    svs, ok = [], []
    for i, (z, y, x) in enumerate(centres):
        z0, z1 = z - half, z - half + box
        y0, y1 = y - half, y - half + box
        x0, x1 = x - half, x - half + box
        if z0 < 0 or y0 < 0 or x0 < 0 or z1 > N or y1 > N or x1 > N:
            continue
        svs.append(vol[z0:z1, y0:y1, x0:x1].copy().astype(np.float32))
        ok.append(i)
    return np.array(svs) if svs else np.empty((0, box, box, box)), ok


# ── Translational alignment ───────────────────────────────────────────────────

def xcorr_align(sv: np.ndarray, ref: np.ndarray, max_shift: int) -> np.ndarray:
    box = sv.shape[0]
    F1  = np.fft.fftn(sv.astype(np.float64))
    F2  = np.fft.fftn(ref.astype(np.float64))
    cc  = np.fft.fftshift(np.real(np.fft.ifftn(F1 * np.conj(F2))))
    c   = box // 2
    lo, hi = c - max_shift, c + max_shift + 1
    sub = cc[lo:hi, lo:hi, lo:hi]
    dz, dy, dx = np.unravel_index(np.argmax(sub), sub.shape)
    dz -= max_shift; dy -= max_shift; dx -= max_shift
    freq  = np.fft.fftfreq(box)
    fz, fy, fx = np.meshgrid(freq, freq, freq, indexing='ij')
    phase = np.exp(-2j * np.pi * (dz * fz + dy * fy + dx * fx))
    return np.real(np.fft.ifftn(F1 * phase)).astype(np.float32)


def iterative_average(subvols: np.ndarray, n_iter: int,
                      max_shift: int, verbose: bool = True) -> np.ndarray:
    aligned = subvols.copy()
    for it in range(n_iter):
        ref = gaussian_filter(aligned.mean(axis=0), sigma=1.5)
        aligned = np.array([xcorr_align(sv, ref, max_shift) for sv in aligned])
        if verbose:
            d = float(np.mean(np.abs(aligned - subvols)))
            print(f"    iter {it+1}/{n_iter}  mean|Δ|={d:.4f}")
    return aligned


# ── Single-crop reconstruction ────────────────────────────────────────────────

def reconstruct_one_crop(args, crop_idx: int, dy: int, dx: int,
                         out_dir: str) -> tuple[str, str, float]:
    """
    Build WBP + PnP 128³ reconstruction for a 256×256 crop offset (dy, dx)
    from the detector centre.  Returns (pnp_path, wbp_path, pixel_size).
    """
    import mrcfile
    from scipy.ndimage import zoom

    from v28e_cryo_em.workflow_demo.ctf_model import (
        load_tilt_series_mrc, load_tlt_file,
    )
    from v28e_cryo_em.production.pnp_ctf_reconstruct import build_A
    from v28e_cryo_em.production.autotune import (
        heuristic_sigma, heuristic_rho, heuristic_n_admm,
    )
    from v28e_cryo_em.production2.wbp import reconstruct_wbp
    from v28e_cryo_em.production2.phase1_baseline import load_defocus_file
    from v28e_cryo_em.workflow_demo.ctf_projector import CTFProjector, PnPCTFSolver
    from v28e_cryo_em.workflow_demo.denoisers import GaussianSpatial

    crop_size = 256
    pnp_path = os.path.join(out_dir, f'crop{crop_idx}_pnp.mrc')
    wbp_path = os.path.join(out_dir, f'crop{crop_idx}_wbp.mrc')

    if os.path.isfile(pnp_path):
        with mrcfile.open(pnp_path, permissive=True) as m:
            px = float(m.voxel_size.x)
        print(f"  crop{crop_idx}: loaded from cache  (pixel={px:.3f}Å)")
        return pnp_path, wbp_path, px

    proj_raw, px_hdr = load_tilt_series_mrc(args.mrc)
    n_tilts, H, W    = proj_raw.shape
    pixel_size       = args.pixel_size or px_hdr

    # Crop centred at (cy+dy, cx+dx)
    cy, cx   = H // 2 + dy, W // 2 + dx
    half     = crop_size // 2
    proj_raw = proj_raw[:, cy-half:cy+half, cx-half:cx+half]

    tilt_angles = load_tlt_file(args.tlt)
    assert len(tilt_angles) == n_tilts

    vol_size  = 1 << int(np.log2(crop_size // 2))   # 128
    proj_size = int(vol_size * 1.5)                  # 192
    eff_px    = pixel_size * (crop_size / proj_size)

    # Resample to proj_size
    scale = proj_size / crop_size
    proj_raw = np.stack([
        zoom(proj_raw[i], scale, order=1)[:proj_size, :proj_size]
        for i in range(n_tilts)
    ]).astype(np.float32)

    A = build_A(tilt_angles, vol_size, proj_size, verbose=True,
                cache_dir=out_dir)
    b = proj_raw.reshape(-1)

    # WBP
    wbp = reconstruct_wbp(A, b, vol_size, proj_size, n_tilts, verbose=False)
    with mrcfile.new(wbp_path, overwrite=True) as m:
        m.set_data(wbp); m.voxel_size = eff_px

    # Per-tilt defocus
    if args.ctffind:
        defocus_all = load_defocus_file(args.ctffind)
        assert len(defocus_all) == n_tilts
    else:
        defocus_all = np.full(n_tilts, args.defocus)

    # PnP
    ctf_proj = CTFProjector(A, tilt_angles, vol_size, proj_size, eff_px,
                            defocus_per_tilt=defocus_all)
    sigma  = heuristic_sigma(eff_px)
    rho    = heuristic_rho(ctf_proj.ctf_power, sigma=sigma)
    n_admm = heuristic_n_admm(rho, sigma=sigma)
    print(f"  crop{crop_idx}: σ={sigma:.2f}  ρ={rho:.3f}  n_admm={n_admm}")

    solver = PnPCTFSolver(ctf_proj, vol_size, denoiser=GaussianSpatial(sigma=sigma))
    pnp_flat, _ = solver.solve(b, rho=rho, n_admm=n_admm, n_cg=25,
                               conv_tol=0.25, verbose=True)

    pnp_vol = pnp_flat.reshape(vol_size, vol_size, vol_size).astype(np.float32)
    with mrcfile.new(pnp_path, overwrite=True) as m:
        m.set_data(pnp_vol); m.voxel_size = eff_px

    return pnp_path, wbp_path, eff_px


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(prog='phase3_sta')
    # Reconstruction inputs
    p.add_argument('--mrc',      default=None)
    p.add_argument('--tlt',      default=None)
    p.add_argument('--ctffind',  default=None)
    p.add_argument('--defocus',  type=float, default=5.408)
    p.add_argument('--pixel-size', type=float, default=None)
    # Pre-built volumes (skip reconstruction)
    p.add_argument('--pnp-mrcs', nargs='+', default=None,
                   help='Pre-built PnP .mrc files (one per crop)')
    p.add_argument('--wbp-mrcs', nargs='+', default=None,
                   help='Pre-built WBP .mrc files (one per crop)')
    # STA parameters
    p.add_argument('--out',              required=True)
    p.add_argument('--box',              type=int,   default=48,
                   help='Sub-volume box in voxels [48]')
    p.add_argument('--particle-radius-a', type=float, default=100.0,
                   help='Particle radius Å for template σ + NMS [100]')
    p.add_argument('--n-particles',      type=int,   default=20,
                   help='Max particles per crop [20]')
    p.add_argument('--n-iter',           type=int,   default=5)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Phase 3: Sub-tomogram averaging  (4-crop tiling)")
    print(f"{'='*60}")

    import mrcfile

    # ── Step 1: reconstructions ───────────────────────────────────────────────
    if args.pnp_mrcs:
        pnp_paths = args.pnp_mrcs
        wbp_paths = args.wbp_mrcs or [None] * len(pnp_paths)
        with mrcfile.open(pnp_paths[0], permissive=True) as m:
            pixel_size = float(m.voxel_size.x) or (args.pixel_size or 4.56)
    else:
        if not args.mrc or not args.tlt:
            p.error("Either --pnp-mrcs or --mrc + --tlt required")
        pnp_paths, wbp_paths = [], []
        pixel_size = None
        for i, (dy, dx) in enumerate(QUAD_OFFSETS):
            print(f"\n--- Crop {i+1}/4  offset=({dy:+d},{dx:+d}) px ---")
            pp, wp, eff_px = reconstruct_one_crop(args, i+1, dy, dx, args.out)
            pnp_paths.append(pp)
            wbp_paths.append(wp)
            if pixel_size is None:
                pixel_size = eff_px
        print(f"\n  All 4 crops done  ({time.time()-t0:.0f}s)")

    if args.pixel_size:
        pixel_size = args.pixel_size

    radius_vox = args.particle_radius_a / pixel_size
    nms_dist   = int(radius_vox * 1.8)
    box        = args.box
    border     = box // 2 + 2
    max_shift  = max(2, int(radius_vox * 0.3))

    print(f"\n  pixel={pixel_size:.3f}Å  Nyquist={2*pixel_size:.2f}Å")
    print(f"  Particle radius: {args.particle_radius_a:.0f}Å = {radius_vox:.1f} vox")
    print(f"  Box: {box}³  NMS dist: {nms_dist} vox  max_shift: {max_shift} vox")
    print(f"  Combined FOV: ~{int(len(pnp_paths)**0.5 * 256 * pixel_size / 10 + 0.5)} nm²")

    template = make_blob_template(box, sigma_vox=radius_vox / 2.5)

    # ── Step 2: template matching on all crops ────────────────────────────────
    print(f"\n  Template matching across {len(pnp_paths)} crops…")
    all_svs_pnp, all_svs_wbp = [], []

    for i, (pp, wp) in enumerate(zip(pnp_paths, wbp_paths)):
        with mrcfile.open(pp, permissive=True) as m:
            vol = m.data.copy().astype(np.float32)

        cc           = template_match_3d(vol, template)
        centres, sc  = nms_3d(cc, min_dist=nms_dist, n_max=args.n_particles,
                              border=border)
        svs, ok_idx  = extract_subvols(vol, centres, box)
        n_found      = len(svs)
        print(f"  crop{i+1}: {len(centres)} candidates → {n_found} extracted  "
              f"(scores {sc[:3].round(4) if len(sc) else '—'})")
        all_svs_pnp.append(svs)

        if wp and os.path.isfile(wp):
            with mrcfile.open(wp, permissive=True) as m:
                wvol = m.data.copy().astype(np.float32)
            wsvs, _ = extract_subvols(wvol, centres[ok_idx], box)
            all_svs_wbp.append(wsvs)

    subvols_pnp = np.concatenate([s for s in all_svs_pnp if len(s)], axis=0)
    n_total     = len(subvols_pnp)
    print(f"\n  Total particles pooled: {n_total}")

    if n_total < 4:
        print("  Too few particles for FSC — try --particle-radius-a smaller value")
        return

    subvols_wbp = None
    if all_svs_wbp:
        wbp_cat = [s for s in all_svs_wbp if len(s)]
        if wbp_cat:
            subvols_wbp = np.concatenate(wbp_cat, axis=0)

    # ── Step 3: iterative alignment ───────────────────────────────────────────
    print(f"\n  Aligning PnP particles ({n_total} × {box}³)…")
    t2 = time.time()
    aligned_pnp = iterative_average(subvols_pnp, n_iter=args.n_iter,
                                    max_shift=max_shift, verbose=True)
    print(f"  Alignment done  ({time.time()-t2:.1f}s)")

    if subvols_wbp is not None and len(subvols_wbp) >= 4:
        print(f"  Aligning WBP particles…")
        aligned_wbp = iterative_average(subvols_wbp, n_iter=args.n_iter,
                                        max_shift=max_shift, verbose=False)
    else:
        aligned_wbp = None

    # ── Step 4: averages ─────────────────────────────────────────────────────
    avg_pnp = aligned_pnp.mean(axis=0)
    with mrcfile.new(os.path.join(args.out, 'sta_avg_pnp.mrc'), overwrite=True) as m:
        m.set_data(avg_pnp); m.voxel_size = pixel_size

    if aligned_wbp is not None:
        avg_wbp = aligned_wbp.mean(axis=0)
        with mrcfile.new(os.path.join(args.out, 'sta_avg_wbp.mrc'), overwrite=True) as m:
            m.set_data(avg_wbp); m.voxel_size = pixel_size

    # ── Step 5: gold-standard FSC ─────────────────────────────────────────────
    print(f"\n  Gold-standard FSC…")

    def gs_fsc(aligned, label):
        n = len(aligned)
        if n < 4:
            print(f"  {label}: only {n} — skipping"); return None, None, None, None
        h1 = aligned[::2].mean(axis=0)
        h2 = aligned[1::2].mean(axis=0)
        freq, fsc_v = fsc(h1, h2)
        r143 = resolution_at_threshold(freq, fsc_v, pixel_size, 0.143)
        r05  = resolution_at_threshold(freq, fsc_v, pixel_size, 0.500)
        print(f"  {label} STA FSC: 0.143→{r143:.1f}Å  0.5→{r05:.1f}Å  (N={n})")
        return freq, fsc_v, r143, r05

    freq_pnp, fsc_pnp, r143_pnp, r05_pnp = gs_fsc(aligned_pnp, 'PnP')
    if aligned_wbp is not None:
        freq_wbp, fsc_wbp, r143_wbp, r05_wbp = gs_fsc(aligned_wbp, 'WBP')
    else:
        freq_wbp = fsc_wbp = r143_wbp = r05_wbp = None

    # ── Step 6: figure ────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nyquist = 2.0 * pixel_size
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # FSC comparison
    ax = axes[0]
    results = []
    if freq_pnp is not None:
        results.append(('PnP STA', freq_pnp, fsc_pnp, '#1f77b4'))
    if freq_wbp is not None:
        results.append(('WBP STA', freq_wbp, fsc_wbp, '#ff7f0e'))

    for label, freq, fsc_v, col in results:
        res_a = np.where(freq > 0, nyquist / freq, np.inf)
        r143  = resolution_at_threshold(freq, fsc_v, pixel_size, 0.143)
        ax.plot(res_a, fsc_v, '-', color=col, lw=2,
                label=f'{label}  N={n_total}  0.143→{r143:.1f}Å')

    ax.axhline(0.143, color='k',    ls='--', lw=1, alpha=0.7, label='FSC=0.143')
    ax.axhline(0.500, color='gray', ls=':',  lw=1, alpha=0.7, label='FSC=0.5')
    # Phase 2 single-tomo reference line
    ax.axvline(9.1, color='steelblue', ls=':', lw=1.5, alpha=0.5,
               label='Single-tomo 9.1Å (Phase 2)')
    ax.set_xlim(left=nyquist * 20, right=nyquist)
    ax.set_xlabel('Resolution (Å)'); ax.set_ylabel('FSC')
    ax.set_title(f'STA gold-standard FSC  (N={n_total} particles)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # PnP average central slices
    mid = box // 2
    for k, (sl, title) in enumerate([
            (avg_pnp[mid],     'PnP STA avg — XY (central)'),
            (avg_pnp[:, mid, :], 'PnP STA avg — XZ (central)')]):
        lo, hi = np.percentile(sl, 2), np.percentile(sl, 98)
        ext = [0, sl.shape[1]*pixel_size/10, 0, sl.shape[0]*pixel_size/10]
        axes[k+1].imshow(sl, cmap='gray', vmin=lo, vmax=hi,
                         origin='lower', extent=ext, aspect='equal')
        axes[k+1].set_title(title, fontsize=9)
        axes[k+1].set_xlabel('nm')
        if k == 0: axes[k+1].set_ylabel('nm')

    fig.suptitle(
        f'Phase 3: STA  |  {len(pnp_paths)} crops  N={n_total} particles  '
        f'box={box}³  {pixel_size:.2f}Å/px',
        fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(args.out, 'phase3_sta.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure → {fig_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        'n_crops':    len(pnp_paths),
        'n_particles': n_total,
        'box':         box,
        'pixel_size':  pixel_size,
        'single_tomo_phase2_fsc143_a': 9.1,
    }
    if r143_pnp is not None:
        summary['pnp_sta'] = {'res_143': r143_pnp, 'res_05': r05_pnp,
                              'improvement_a': 9.1 - r143_pnp}
    if r143_wbp is not None:
        summary['wbp_sta'] = {'res_143': r143_wbp, 'res_05': r05_wbp}

    with open(os.path.join(args.out, 'phase3_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 3 done  ({time.time()-t0:.0f}s total)")
    if r143_pnp is not None:
        print(f"  Single-tomogram FSC=0.143:  9.1 Å  (Phase 2)")
        print(f"  PnP STA FSC=0.143:          {r143_pnp:.1f} Å  (N={n_total})")
        impr = 9.1 - r143_pnp
        tag  = '✓' if impr > 0 else '(limited by particle count / alignment)'
        print(f"  Improvement:               {impr:+.1f} Å  {tag}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
