"""
generate_stage7_data.py — Full bistatic acoustic scattering tensor.

For each (shape, roughness, k, seed):
  1. Perturb panels
  2. Build BEM matrix A once on GPU (N=2048)
  3. Solve 72 incident directions via py_bem_solve_multi_rhs (1 build × 72 solves)
  4. Compute bistatic RCS for 72 observer directions — one GPU matmul
  5. Online Welford accumulation on LINEAR RCS over 20 seeds
     (converted to dBm / dB-relative std at save time)

Output: stage7_data/R{r_idx}_F{f_idx}/{shape}.npz
Each file: mean(72,72) [dBm], std(72,72) [dB, delta-method], p_detect(72,72), n_seeds

Total groups: 4 shapes × 4 roughness × 5 k = 80
Seeds/group: 20
Estimated runtime: ~12 min (N=2048, GPU multi-RHS)
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
from bem_helmholtz_v2 import make_rhs, solve_multi_rhs
import cupy as cp

N_PANELS     = 2048
N_SEEDS      = 20
N_INC        = 72
N_OBS        = 72
WAVENUMBERS  = [2.0, 4.0, 6.0, 10.0, 16.0]
THRESHOLD_DB = -5.0

INC_PHI = np.linspace(0, 2*np.pi, N_INC, endpoint=False)
OBS_PHI = np.linspace(0, 2*np.pi, N_OBS, endpoint=False)

SHAPES = {
    'circle':    (lambda: circle_panels(N_PANELS),    1.0),
    'ellipse':   (lambda: ellipse_panels(N_PANELS),   2.0),
    'joukowski': (lambda: joukowski_panels(N_PANELS), 2.2),
    'submarine': (lambda: submarine_panels(N_PANELS), 1.0),
}

DATA_ROOT = _HERE / 'stage7_data'


def bistatic_tensor(nodes, lengths, sigma_matrix, k):
    """Compute bistatic RCS matrix (N_INC, N_OBS) from multi-RHS solution.

    sigma_matrix: (N_panels, N_INC) complex128 — col j = surface current for inc j
    Returns: (N_INC, N_OBS) float32 [m]
    """
    N = nodes.shape[0]
    # weights: (N, N_INC)
    weights = sigma_matrix * lengths[:, None]  # (N, N_INC)
    # obs phase matrix: (N_OBS, N)
    r_obs   = np.stack([np.cos(OBS_PHI), np.sin(OBS_PHI)], axis=1)  # (N_OBS, 2)
    phase_d = cp.asarray(np.exp(-1j * k * (r_obs @ nodes.T)), dtype=cp.complex128)  # (N_OBS, N)
    w_d     = cp.asarray(weights, dtype=cp.complex128)                               # (N, N_INC)
    # f_matrix: (N_OBS, N_INC) — batched far-field integral
    f_mat   = (1j / 4.0) * (phase_d @ w_d)    # (N_OBS, N_INC)
    rcs_mat = (4.0 / k) * cp.abs(f_mat) ** 2  # (N_OBS, N_INC) [m]
    # transpose to (N_INC, N_OBS) for intuitive indexing
    return cp.asnumpy(rcs_mat.T).astype(np.float32)


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
    mean_lin = np.zeros((N_INC, N_OBS), dtype=np.float64)   # Welford accumulator: linear RCS [m]
    M2_lin   = np.zeros((N_INC, N_OBS), dtype=np.float64)
    p_detect = np.zeros((N_INC, N_OBS), dtype=np.float64)

    t0 = time.perf_counter()
    for seed in range(N_SEEDS):
        nodes, normals, lengths = perturb_panels(
            base_nodes, base_normals, base_lengths, eps, char_size, seed)

        # 1 GPU build + 72 GMRES solves
        sigma_matrix, _ = solve_multi_rhs(nodes, lengths, k, INC_PHI)
        # (N_INC, N_OBS) bistatic RCS
        rcs_lin = bistatic_tensor(nodes, lengths, sigma_matrix, k).astype(np.float64)
        rcs_db  = 10.0 * np.log10(np.maximum(rcs_lin, 1e-20))

        count   += 1
        delta    = rcs_lin - mean_lin
        mean_lin += delta / count
        delta2   = rcs_lin - mean_lin
        M2_lin  += delta * delta2
        p_detect += (rcs_db > THRESHOLD_DB).astype(np.float64)

        # Free GPU memory between seeds
        cp.get_default_memory_pool().free_all_blocks()

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
        inc_phi=INC_PHI.astype(np.float32),
        obs_phi=OBS_PHI.astype(np.float32),
        n_seeds=count)

    elapsed = time.perf_counter() - t0
    if verbose:
        mono_mean = np.mean([mean_db[j, j] for j in range(min(N_INC, N_OBS))])
        print(f'  {shape_name:10s}  {elapsed:.1f}s  '
              f'mean_mono={mono_mean:.1f}dBm  '
              f'max_pdet={p_detect.max():.0%}')
    return dict(mean=mean_db, std=std_db, p_detect=p_detect)


def generate_all(verbose=True):
    shape_names = list(SHAPES.keys())
    total   = len(shape_names) * len(ROUGHNESS_FRACS) * len(WAVENUMBERS)
    done    = skipped = 0
    t0_all  = time.perf_counter()

    print(f'Stage 7 acoustic bistatic tensor:')
    print(f'  {len(shape_names)} shapes × {len(ROUGHNESS_FRACS)} roughness × '
          f'{len(WAVENUMBERS)} k × {N_SEEDS} seeds = {total} groups')
    print(f'  N={N_PANELS}  N_inc={N_INC}  N_obs={N_OBS}')
    print()

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
                        print(f'    {name:10s}  SKIP')

    elapsed = time.perf_counter() - t0_all
    print(f'\nDone: {done-skipped} written, {skipped} skipped, '
          f'{elapsed/60:.1f} min total')


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
