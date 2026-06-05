"""
generate_test_data_3d.py — Synthetic 3D checkpoint generator for Phase 3 tests.

Writes N_TARGETS × N_SEEDS rcs3d_T{t:02d}_S{s:04d}.bin files using known
RCS values so the COBOL aggregator output can be verified analytically.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path
from rcs_bridge_3d import write_checkpoint_3d, N_MONO_ANGLES

N_TARGETS = 5
N_SEEDS   = 20
FREQ_GHZ  = 9.0
KA        = 8.0
N_PANELS  = 5120


def synthetic_rcs(target_id: int, seed: int) -> np.ndarray:
    """Deterministic RCS pattern: target_id sets base level, seed adds noise.

    rcs[i] = -20 + 2*target_id + sin(2π i/72)*5 + N(0, 0.5)*seed_scale

    The mean over seeds converges to the noise-free pattern; std → 0.5*seed_scale.
    """
    rng = np.random.default_rng(seed * 100 + target_id)
    base     = -20.0 + 2.0 * target_id
    pattern  = base + 5.0 * np.sin(2 * np.pi * np.arange(N_MONO_ANGLES) / N_MONO_ANGLES)
    noise    = rng.normal(0.0, 0.5, N_MONO_ANGLES)
    return pattern + noise


def generate(output_dir: Path, n_targets: int = N_TARGETS,
             n_seeds: int = N_SEEDS, verbose: bool = True) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total  = n_targets * n_seeds
    done   = 0

    for t in range(n_targets):
        for s in range(n_seeds):
            rcs  = synthetic_rcs(t, s)
            path = output_dir / f'rcs3d_T{t:02d}_S{s:04d}.bin'
            write_checkpoint_3d(
                path, target_id=t, seed=s,
                freq_ghz=FREQ_GHZ, ka=KA,
                mono_rcs_dbm=rcs, n_panels=N_PANELS,
            )
            done += 1
            if verbose and done % 20 == 0:
                print(f'  {done}/{total} checkpoints written')

    if verbose:
        print(f'Done: {done} checkpoint files in {output_dir}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='test_data_3d')
    args = p.parse_args()
    generate(args.outdir)
