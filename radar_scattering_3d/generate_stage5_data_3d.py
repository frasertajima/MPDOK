"""
generate_stage5_data_3d.py — Stage 5 resolution audit at N=5120.

Identical physics to Stage 4 (same incident direction, obs grid, targets,
roughness, wavenumbers) with two changes:
  N_PANELS = 5120   (vs 2560)
  N_SEEDS  = 20     (vs 50, fewer seeds — audit not full MC)

Output layout mirrors Stage 4 under stage5_data_3d/.
Comparison: Stage 4 mean/p_detect at N=2560 vs Stage 5 at N=5120.
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
from generate_stage4_data_3d import (
    WAVENUMBERS, INC_DIR, THRESHOLD_DBSM, GMRES_RESTART, GMRES_TOL,
    OBS_THETA, OBS_PHI,
)

# ── Configuration ─────────────────────────────────────────────────────────────

N_PANELS = 5120   # 4× Stage 4 panel count
N_SEEDS  = 20     # 50→20: resolution audit needs fewer seeds

DATA_ROOT = _HERE / 'stage5_data_3d' / 'groups'


# ── Single group ──────────────────────────────────────────────────────────────

def aggregate_group(tgt, eps_idx, freq_idx, asm, verbose=False):
    """Identical to Stage 4 aggregate_group with N=5120."""
    t       = tgt['id']
    eps     = ROUGHNESS_FRACS[eps_idx]
    k       = WAVENUMBERS[freq_idx]
    outdir  = DATA_ROOT / f'R{eps_idx}_F{freq_idx}'
    outfile = outdir / f'rcs3d_s5_T{t:02d}.npz'

    if outfile.exists():
        return None

    outdir.mkdir(parents=True, exist_ok=True)

    base_n, base_nm, base_a = tgt['geom_fn'](N_PANELS, k)

    count    = 0
    mean_rcs = np.zeros((18, 36), dtype=np.float64)
    M2_rcs   = np.zeros((18, 36), dtype=np.float64)
    p_detect = np.zeros((18, 36), dtype=np.float64)

    t0 = time.perf_counter()
    for seed in range(N_SEEDS):
        nodes, normals, areas = perturb_mesh_3d(
            base_n, base_nm, base_a, eps, tgt['char_size'], seed)

        b     = make_rhs_3d(nodes, k, INC_DIR)
        sigma, _, _ = asm.solve_ir(nodes, areas, k, b,
                                    restart=GMRES_RESTART, tol=GMRES_TOL,
                                    maxiter_ir=0)
        rcs_grid = bistatic_sphere_sweep(nodes, areas, sigma, k,
                                          OBS_THETA, OBS_PHI)
        rcs_db   = 10.0 * np.log10(np.maximum(rcs_grid, 1e-20))

        count  += 1
        delta   = rcs_db - mean_rcs
        mean_rcs += delta / count
        delta2   = rcs_db - mean_rcs
        M2_rcs  += delta * delta2
        p_detect += (rcs_db > THRESHOLD_DBSM).astype(np.float64)

    std_rcs   = np.sqrt(M2_rcs / max(count - 1, 1))
    p_detect /= count

    np.savez_compressed(outfile,
                        mean=mean_rcs.astype(np.float32),
                        std=std_rcs.astype(np.float32),
                        p_detect=p_detect.astype(np.float32),
                        n_seeds=count)

    if verbose:
        t_grp = time.perf_counter() - t0
        print(f'    {tgt["name"]:8s}  {t_grp:.1f}s  '
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

    print(f'Stage 5 3D resolution audit:  {total} groups × {N_SEEDS} seeds  '
          f'N={N_PANELS}')
    print()

    for r_idx, eps in enumerate(ROUGHNESS_FRACS):
        for f_idx, k in enumerate(WAVENUMBERS):
            if verbose:
                print(f'  ε={eps:.0%}  k={k:.0f}')
            for tgt in TARGETS:
                result = aggregate_group(tgt, r_idx, f_idx, asm, verbose=verbose)
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
    return sum(1 for _ in DATA_ROOT.rglob('rcs3d_s5_T*.npz'))


def load_group(t_idx, r_idx, f_idx):
    path = DATA_ROOT / f'R{r_idx}_F{f_idx}' / f'rcs3d_s5_T{t_idx:02d}.npz'
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
