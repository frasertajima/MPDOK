"""
aggregate_stage4.py — COBOL Welford aggregation for Stage 4 ensemble data.

Runs one COBOL aggregator call per (roughness, frequency) combination.
Each call processes 5 targets × 50 seeds = 250 checkpoint records and
writes one ensemble .stls file with mean ± std RCS for all 5 targets.

Output structure:
  stage4_data/ensembles/R{r}_F{f}_ensemble.stls

Reading the output from Python:
  from rcs_bridge import read_ensemble
  ens = read_ensemble('stage4_data/ensembles/R1_F2_ensemble.stls')
  # ens[i].target_id, ens[i].mean_rcs_dbm, ens[i].std_rcs_dbm
"""

import sys
from pathlib import Path
import time

_HERE = Path(__file__).parent
if str(_HERE / 'cobol_rcs') not in sys.path:
    sys.path.insert(0, str(_HERE / 'cobol_rcs'))

from rcs_bridge import aggregate_rcs
from generate_stage4_data import ROUGHNESS_FRACS, WAVENUMBERS, N_SEEDS

DATA_ROOT  = _HERE / 'stage4_data' / 'checkpoints'
ENS_ROOT   = _HERE / 'stage4_data' / 'ensembles'


def aggregate_all(force=False, verbose=True):
    """Run COBOL aggregation for every (roughness, frequency) pair.

    Args:
        force:   Re-run even if ensemble file already exists.
        verbose: Print progress.

    Returns:
        dict mapping (r_idx, f_idx) -> list[RCSEnsemble]
    """
    ENS_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    n_total = len(ROUGHNESS_FRACS) * len(WAVENUMBERS)
    n_done  = 0
    t0_all  = time.perf_counter()

    for r_idx, eps in enumerate(ROUGHNESS_FRACS):
        for f_idx, k in enumerate(WAVENUMBERS):
            ens_path = ENS_ROOT / f'R{r_idx}_F{f_idx}_ensemble.stls'
            chk_dir  = DATA_ROOT / f'R{r_idx}_F{f_idx}'

            n_done += 1

            if ens_path.exists() and not force:
                from rcs_bridge import read_ensemble
                results[(r_idx, f_idx)] = read_ensemble(ens_path)
                if verbose:
                    print(f'  [{n_done:2d}/{n_total}] ε={eps:.0%} k={k:.0f}'
                          f'  loaded from cache')
                continue

            if not chk_dir.exists() or not list(chk_dir.glob('rcs_T*_S*.bin')):
                if verbose:
                    print(f'  [{n_done:2d}/{n_total}] ε={eps:.0%} k={k:.0f}'
                          f'  SKIP — no checkpoint files (run generate_stage4_data.py first)')
                continue

            t0 = time.perf_counter()
            ens = aggregate_rcs(chk_dir, ens_path, n_expected=N_SEEDS)
            dt  = time.perf_counter() - t0

            results[(r_idx, f_idx)] = ens
            if verbose:
                tgt_ids = [r.target_id for r in ens]
                print(f'  [{n_done:2d}/{n_total}] ε={eps:.0%} k={k:.0f}'
                      f'  {len(ens)} targets  {dt*1e3:.0f}ms'
                      f'  targets={tgt_ids}')

    total_dt = time.perf_counter() - t0_all
    if verbose:
        print(f'\nAggregation complete: {len(results)} groups in {total_dt:.1f}s')
    return results


def load_all():
    """Load all pre-aggregated ensemble results without re-running COBOL."""
    return aggregate_all(force=False, verbose=False)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true',
                   help='Re-aggregate even if output files exist')
    args = p.parse_args()

    print('Stage 4 COBOL aggregation')
    print(f'  {len(ROUGHNESS_FRACS)} roughness × {len(WAVENUMBERS)} frequencies'
          f' = {len(ROUGHNESS_FRACS)*len(WAVENUMBERS)} aggregator calls')
    print()
    aggregate_all(force=args.force)
