"""
generate_stage4_data_3d.py — Stage 4 MC data for radar_scattering_3d.

Fixed nose-on incidence (d = [1, 0, 0], θ=90°, φ=0°).
Bistatic observation: full 18×36 sphere grid (648 directions).
Online Welford aggregation — no per-seed storage on disk.

Configuration:
  5 targets × 4 roughness × 5 wavenumbers × N_SEEDS = total groups

Per-group output (.npz):
  mean     (18, 36) float32  — Welford mean RCS [dBsm]
  std      (18, 36) float32  — Welford std       [dB]
  p_detect (18, 36) float32  — P(RCS > THRESHOLD_DBSM)
  n_seeds  int

Resume support: existing .npz files are skipped.
"""

import sys, os, time, argparse

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / '..' / 'bem_cobol'))

from geometry_3d_targets import TARGETS, ROUGHNESS_FRACS, perturb_mesh_3d
from bem_assembly_3d_multi_ops import BEMAssembler3DMulti
from rcs_3d import make_rhs_3d, bistatic_sphere_sweep, obs_grid

# ── Configuration ─────────────────────────────────────────────────────────────

N_PANELS        = 2560
N_SEEDS         = 50
WAVENUMBERS     = [3.0, 5.0, 8.0, 12.0, 16.0]
INC_DIR         = np.array([1., 0., 0.])   # nose-on: θ=90°, φ=0°
THRESHOLD_DBSM  = -10.0                    # detection threshold [dBsm]
GMRES_RESTART   = 50
GMRES_TOL       = 1e-6

_OBS_DIRS, OBS_THETA, OBS_PHI = obs_grid(18, 36)   # 648 observation directions

DATA_ROOT = _HERE / 'stage4_data_3d' / 'groups'


# ── Single group: N_SEEDS solves + online Welford ─────────────────────────────

def aggregate_group(tgt, eps_idx, freq_idx, asm, verbose=False):
    """Solve N_SEEDS BEM problems and return aggregated RCS statistics.

    Returns dict with mean, std, p_detect (18,36), n_seeds — or None if
    the output file already exists.
    """
    t       = tgt['id']
    eps     = ROUGHNESS_FRACS[eps_idx]
    k       = WAVENUMBERS[freq_idx]
    outdir  = DATA_ROOT / f'R{eps_idx}_F{freq_idx}'
    outfile = outdir / f'rcs3d_s4_T{t:02d}.npz'

    if outfile.exists():
        return None

    outdir.mkdir(parents=True, exist_ok=True)

    base_n, base_nm, base_a = tgt['geom_fn'](N_PANELS, k)

    # Welford accumulators for (18,36) RCS grid
    count    = 0
    mean_rcs = np.zeros((18, 36), dtype=np.float64)
    M2_rcs   = np.zeros((18, 36), dtype=np.float64)
    p_detect = np.zeros((18, 36), dtype=np.float64)  # fraction > threshold

    t0_group = time.perf_counter()

    for seed in range(N_SEEDS):
        nodes, normals, areas = perturb_mesh_3d(
            base_n, base_nm, base_a, eps, tgt['char_size'], seed)

        b = make_rhs_3d(nodes, k, INC_DIR)
        sigma, converged, rel_res = asm.solve_ir(
            nodes, areas, k, b,
            restart=GMRES_RESTART, tol=GMRES_TOL, maxiter_ir=0)

        rcs_grid = bistatic_sphere_sweep(nodes, areas, sigma, k,
                                         OBS_THETA, OBS_PHI)   # (18,36) m²
        rcs_db   = 10.0 * np.log10(np.maximum(rcs_grid, 1e-20))

        # Online Welford
        count += 1
        delta       = rcs_db - mean_rcs
        mean_rcs   += delta / count
        delta2      = rcs_db - mean_rcs
        M2_rcs     += delta * delta2
        p_detect   += (rcs_db > THRESHOLD_DBSM).astype(np.float64)

    std_rcs   = np.sqrt(M2_rcs / max(count - 1, 1))
    p_detect /= count

    np.savez_compressed(outfile,
                        mean=mean_rcs.astype(np.float32),
                        std=std_rcs.astype(np.float32),
                        p_detect=p_detect.astype(np.float32),
                        n_seeds=count)

    t_group = time.perf_counter() - t0_group
    if verbose:
        print(f'    {tgt["name"]:8s} seed 0..{N_SEEDS-1}  '
              f'{t_group:.1f}s  '
              f'peak_mean={mean_rcs.max():.1f}dBsm  '
              f'max_pdet={p_detect.max():.0%}')

    return dict(mean=mean_rcs.astype(np.float32),
                std=std_rcs.astype(np.float32),
                p_detect=p_detect.astype(np.float32),
                n_seeds=count)


# ── Main sweep ────────────────────────────────────────────────────────────────

def generate_all(verbose=True):
    total   = len(TARGETS) * len(ROUGHNESS_FRACS) * len(WAVENUMBERS)
    done    = 0
    skipped = 0
    t0_all  = time.perf_counter()

    asm = BEMAssembler3DMulti()

    print(f'Stage 4 3D generation:  {len(TARGETS)} tgts × '
          f'{len(ROUGHNESS_FRACS)} roughness × {len(WAVENUMBERS)} freq = '
          f'{total} groups × {N_SEEDS} seeds = {total*N_SEEDS} solves')
    print(f'  N={N_PANELS}  inc=(1,0,0)  obs=18×36 sphere  '
          f'threshold={THRESHOLD_DBSM} dBsm')
    print()

    for r_idx, eps in enumerate(ROUGHNESS_FRACS):
        for f_idx, k in enumerate(WAVENUMBERS):
            if verbose:
                print(f'  ε={eps:.0%}  k={k:.0f}')
            for tgt in TARGETS:
                result = aggregate_group(tgt, r_idx, f_idx, asm,
                                         verbose=verbose)
                done += 1
                if result is None:
                    skipped += 1
                    if verbose:
                        print(f'    {tgt["name"]:8s} SKIP (exists)')

    elapsed = time.perf_counter() - t0_all
    n_written = done - skipped
    print(f'\nDone: {n_written} groups written, {skipped} skipped, '
          f'{elapsed:.1f}s total')
    return dict(n_written=n_written, n_skipped=skipped, elapsed=elapsed)


def count_existing():
    return sum(1 for _ in DATA_ROOT.rglob('rcs3d_s4_T*.npz'))


def load_group(t_idx, r_idx, f_idx):
    """Load one aggregated group. Returns None if not yet generated."""
    path = DATA_ROOT / f'R{r_idx}_F{f_idx}' / f'rcs3d_s4_T{t_idx:02d}.npz'
    if not path.exists():
        return None
    return dict(np.load(path))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    existing = count_existing()
    total    = len(TARGETS) * len(ROUGHNESS_FRACS) * len(WAVENUMBERS)
    print(f'Existing: {existing}/{total} groups')
    generate_all(verbose=not args.quiet)
