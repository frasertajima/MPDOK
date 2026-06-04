"""
aggregate_stage7.py — Welford ensemble aggregation for Stage 7 bistatic matrices.

Computes mean and std of the 90×90 bistatic RCS matrix across seeds,
for every (target, roughness, frequency) group.

Output: stage7_data/ensembles/R{r}_F{f}/ens7_T{t:02d}.npz
  Keys: mean  (90,90) float32  — ensemble mean RCS [dBm]
        std   (90,90) float32  — ensemble std  [dB]
        n     int              — seeds used
"""

import sys
from pathlib import Path
import time
import numpy as np

_HERE = Path(__file__).parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from generate_stage7_data import ROUGHNESS_FRACS, WAVENUMBERS, N_SEEDS, TARGETS, N_ANGLES

DATA_ROOT = _HERE / 'stage7_data' / 'groups'
ENS_ROOT  = _HERE / 'stage7_data' / 'ensembles'


def aggregate_group(chk_dir: Path, t_idx: int, n_expected: int) -> dict:
    """Welford online mean+std over all seed matrices for one target."""
    count  = 0
    mean   = np.zeros((N_ANGLES, N_ANGLES), dtype=np.float64)
    M2     = np.zeros((N_ANGLES, N_ANGLES), dtype=np.float64)

    for fpath in sorted(chk_dir.glob(f'rcs7_T{t_idx:02d}_S*.npy')):
        x = np.load(fpath).astype(np.float64)
        count += 1
        delta  = x - mean
        mean  += delta / count
        delta2 = x - mean
        M2    += delta * delta2

    if count < 2:
        std = np.zeros_like(mean)
    else:
        std = np.sqrt(M2 / (count - 1))

    return dict(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        n=count,
    )


def aggregate_all(force=False, verbose=True):
    ENS_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    n_total = len(ROUGHNESS_FRACS) * len(WAVENUMBERS) * len(TARGETS)
    n_done  = 0
    t0_all  = time.perf_counter()

    for r_idx, eps in enumerate(ROUGHNESS_FRACS):
        for f_idx, k in enumerate(WAVENUMBERS):
            chk_dir  = DATA_ROOT / f'R{r_idx}_F{f_idx}'
            ens_dir  = ENS_ROOT  / f'R{r_idx}_F{f_idx}'
            ens_dir.mkdir(parents=True, exist_ok=True)

            for tgt in TARGETS:
                t_idx   = tgt['id']
                out_path = ens_dir / f'ens7_T{t_idx:02d}.npz'
                n_done  += 1

                if out_path.exists() and not force:
                    d = dict(np.load(out_path))
                    results[(r_idx, f_idx, t_idx)] = d
                    if verbose:
                        print(f'  [{n_done:4d}/{n_total}] ε={eps:.0%} k={k:.0f} '
                              f'{tgt["name"]:8s}  loaded  (n={d["n"]})')
                    continue

                if not chk_dir.exists():
                    if verbose:
                        print(f'  [{n_done:4d}/{n_total}] ε={eps:.0%} k={k:.0f} '
                              f'{tgt["name"]:8s}  SKIP (no data)')
                    continue

                t0 = time.perf_counter()
                d  = aggregate_group(chk_dir, t_idx, N_SEEDS)
                dt = time.perf_counter() - t0

                if d['n'] == 0:
                    if verbose:
                        print(f'  [{n_done:4d}/{n_total}] ε={eps:.0%} k={k:.0f} '
                              f'{tgt["name"]:8s}  SKIP (no .npy files)')
                    continue

                np.savez_compressed(out_path, **d)
                results[(r_idx, f_idx, t_idx)] = d

                if verbose:
                    print(f'  [{n_done:4d}/{n_total}] ε={eps:.0%} k={k:.0f} '
                          f'{tgt["name"]:8s}  n={d["n"]:2d}  '
                          f'max_std={d["std"].max():.3f}dB  {dt*1e3:.0f}ms')

    total_dt = time.perf_counter() - t0_all
    n_agg = sum(1 for d in results.values() if d['n'] > 0)
    if verbose:
        print(f'\nAggregation complete: {n_agg} ensembles in {total_dt:.1f}s')
    return results


def load_all():
    return aggregate_all(force=False, verbose=False)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    print('Stage 7 ensemble aggregation')
    print(f'  {len(ROUGHNESS_FRACS)} roughness × {len(WAVENUMBERS)} frequencies '
          f'× {len(TARGETS)} targets = '
          f'{len(ROUGHNESS_FRACS)*len(WAVENUMBERS)*len(TARGETS)} ensemble files')
    print()
    aggregate_all(force=args.force)
