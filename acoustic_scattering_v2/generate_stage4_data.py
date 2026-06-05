"""
generate_stage4_data.py — Monte Carlo roughness study for acoustic_scattering_v2.

For each (shape, roughness, k, seed):
  1. Perturb panels via normal-direction Gaussian displacement
  2. Solve BEM (GPU GMRES, N=1024)
  3. Compute RCS at N_PHI=90 observer angles
  4. Online Welford accumulation on LINEAR RCS — mean, M2, p_detect
     (converted to dBm / dB-relative std at save time)

Output: stage4_data/R{r_idx}_F{f_idx}/{shape}.npz
Each file: mean(N_PHI) [dBm], std(N_PHI) [dB, delta-method], p_detect(N_PHI), n_seeds

Total groups: 4 shapes × 4 roughness × 5 k = 80
Seeds/group: 50
Estimated runtime: ~25 s (N=1024, GPU)
"""

import sys, os, time, argparse
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path

_HERE  = Path(__file__).parent
_MPDOK = _HERE.parent
for _p in [str(_HERE), str(_MPDOK/'acoustic_scattering'), str(_MPDOK/'radar_scattering')]:
    if _p not in sys.path: sys.path.insert(0, _p)

from geometry    import circle_panels, ellipse_panels, joukowski_panels, submarine_panels
from geometry_v2 import perturb_panels, ROUGHNESS_FRACS
from bem_helmholtz_v2 import solve_ir, eval_rcs_2d

N_PANELS     = 1024
N_SEEDS      = 50
N_PHI        = 90
WAVENUMBERS  = [2.0, 4.0, 6.0, 10.0, 16.0]
THRESHOLD_DB = -5.0          # p_detect threshold [dBm]
PHI_OBS      = np.linspace(0, 2*np.pi, N_PHI, endpoint=False)

SHAPES = {
    'circle':    (lambda: circle_panels(N_PANELS),    1.0),
    'ellipse':   (lambda: ellipse_panels(N_PANELS),   2.0),
    'joukowski': (lambda: joukowski_panels(N_PANELS), 2.2),
    'submarine': (lambda: submarine_panels(N_PANELS), 1.0),
}

DATA_ROOT = _HERE / 'stage4_data'


def aggregate_group(shape_name, r_idx, f_idx, verbose=False):
    gen_fn, char_size = SHAPES[shape_name]
    eps    = ROUGHNESS_FRACS[r_idx]
    k      = WAVENUMBERS[f_idx]
    outdir = DATA_ROOT / f'R{r_idx}_F{f_idx}'
    outfile = outdir / f'{shape_name}.npz'
    if outfile.exists():
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    base_nodes, base_normals, base_lengths = gen_fn()

    count    = 0
    mean_lin = np.zeros(N_PHI, dtype=np.float64)   # Welford accumulator: linear RCS [m]
    M2_lin   = np.zeros(N_PHI, dtype=np.float64)
    p_detect = np.zeros(N_PHI, dtype=np.float64)

    t0 = time.perf_counter()
    for seed in range(N_SEEDS):
        nodes, normals, lengths = perturb_panels(
            base_nodes, base_normals, base_lengths, eps, char_size, seed)

        sigma, _ = solve_ir(nodes, lengths, k, phi_inc=0.0, maxiter_ir=0)
        rcs_lin  = eval_rcs_2d(nodes, lengths, sigma, k, PHI_OBS)
        rcs_db   = 10.0 * np.log10(np.maximum(rcs_lin, 1e-20))

        count   += 1
        delta    = rcs_lin - mean_lin
        mean_lin += delta / count
        delta2   = rcs_lin - mean_lin
        M2_lin  += delta * delta2
        p_detect += (rcs_db > THRESHOLD_DB).astype(np.float64)

    std_lin  = np.sqrt(M2_lin / max(count - 1, 1))
    p_detect /= count

    # Convert to dB at save time: arithmetic mean of linear RCS → dBm
    mean_db = 10.0 * np.log10(np.maximum(mean_lin, 1e-20))
    # Delta-method dB std: σ_dB ≈ (10/ln10) * σ_lin / μ_lin
    std_db  = (10.0 / np.log(10.0)) * std_lin / np.maximum(mean_lin, 1e-20)

    np.savez_compressed(outfile,
        mean=mean_db.astype(np.float32),
        std=std_db.astype(np.float32),
        p_detect=p_detect.astype(np.float32),
        phi_obs=PHI_OBS.astype(np.float32),
        n_seeds=count)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f'  {shape_name:10s}  {elapsed:.1f}s  '
              f'mean_back={mean_db[N_PHI//2]:.1f}dBm  '
              f'max_pdet={p_detect.max():.0%}')
    return dict(mean=mean_db, std=std_db, p_detect=p_detect)


def generate_all(verbose=True):
    shape_names = list(SHAPES.keys())
    total   = len(shape_names) * len(ROUGHNESS_FRACS) * len(WAVENUMBERS)
    done    = skipped = 0
    t0_all  = time.perf_counter()

    print(f'Stage 4 acoustic: {len(shape_names)} shapes × '
          f'{len(ROUGHNESS_FRACS)} roughness × {len(WAVENUMBERS)} k × '
          f'{N_SEEDS} seeds = {total} groups  N={N_PANELS}')

    for r_idx, eps in enumerate(ROUGHNESS_FRACS):
        for f_idx, k in enumerate(WAVENUMBERS):
            if verbose:
                print(f'  ε={eps:.0%}  k={k:.0f}')
            for name in shape_names:
                result = aggregate_group(name, r_idx, f_idx, verbose=verbose)
                done += 1
                if result is None:
                    skipped += 1
                    if verbose:
                        print(f'    {name:10s}  SKIP (exists)')

    elapsed = time.perf_counter() - t0_all
    print(f'\nDone: {done-skipped} written, {skipped} skipped, '
          f'{elapsed:.1f}s total')


def count_existing():
    return sum(1 for _ in DATA_ROOT.rglob('*.npz'))


def load_group(shape_name, r_idx, f_idx):
    path = DATA_ROOT / f'R{r_idx}_F{f_idx}' / f'{shape_name}.npz'
    if not path.exists():
        return None
    return dict(np.load(path))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    print(f'Existing: {count_existing()}/80 groups')
    generate_all(verbose=not args.quiet)
