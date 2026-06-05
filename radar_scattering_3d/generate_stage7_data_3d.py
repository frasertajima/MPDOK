"""
generate_stage7_data_3d.py — Full bistatic scattering tensor, N=5120.

For each (target, roughness, k, seed):
  1. Build BEM matrix A once on GPU  (complex64, ~4 ms)
  2. Solve 72 incident directions     (CuPy GMRES, reusing A)
  3. Compute full bistatic sphere     (batched GPU matmul, 18×36 obs)
  4. Online Welford accumulation      (no per-seed storage)
  5. Save one .npz per group          (mean, std, p_detect — 186 KB each)

Architecture: build A once per seed, solve M=72 RHS via CuPy GMRES reusing
the same GPU-resident matrix.  This achieves the "1-build × M-solves"
efficiency without the NVHPC device-allocatable overhead that affects the
Fortran py_bem_solve_multi_rhs_3d kernel at N=5120.

Total groups: 5 × 4 × 5 × 20 = 2,000   Estimated runtime: ~1.5 h
Data per group:  (72, 18, 36) float32 × 2 arrays  ≈  186 KB
Total on disk:   ~18 MB
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
from rcs_3d import (make_rhs_3d, bistatic_sphere_sweep_batch,
                    incident_grid, obs_grid)

import cupy as cp
from cupyx.scipy.sparse.linalg import gmres as cp_gmres, LinearOperator

# ── Configuration ─────────────────────────────────────────────────────────────

N_PANELS        = 5120
N_SEEDS         = 20
WAVENUMBERS     = [3.0, 5.0, 8.0, 12.0, 16.0]
THRESHOLD_DBSM  = -10.0
GMRES_TOL       = 1e-6
GMRES_RESTART   = 50
GMRES_MAXITER   = 20     # max inner iterations; accepts partial convergence

INC_DIRS, INC_THETA, INC_PHI = incident_grid(6, 12)    # 72 incident directions
OBS_DIRS, OBS_THETA, OBS_PHI = obs_grid(18, 36)         # 648 observation dirs
M_INC = len(INC_DIRS)                                   # 72

DATA_ROOT = _HERE / 'stage7_data_3d' / 'groups'


# ── Single group: N_SEEDS solves + online Welford ─────────────────────────────

def aggregate_group(tgt, eps_idx, freq_idx, asm, verbose=False):
    """Full bistatic tensor for one (target, roughness, k) group.

    Builds A once per seed, solves 72 incident directions via CuPy GMRES
    (reusing the cached A), computes the full (18×36) bistatic sphere via
    batched GPU matmul, and accumulates online Welford statistics.

    Returns dict with mean/std/p_detect (72,18,36) — or None if file exists.
    """
    t       = tgt['id']
    eps     = ROUGHNESS_FRACS[eps_idx]
    k       = WAVENUMBERS[freq_idx]
    outdir  = DATA_ROOT / f'R{eps_idx}_F{freq_idx}'
    outfile = outdir / f'rcs3d_s7_T{t:02d}.npz'

    if outfile.exists():
        return None

    outdir.mkdir(parents=True, exist_ok=True)

    base_n, base_nm, base_a = tgt['geom_fn'](N_PANELS, k)

    # Welford accumulators for (72, 18, 36) tensor
    count    = 0
    mean_rcs = np.zeros((M_INC, 18, 36), dtype=np.float64)
    M2_rcs   = np.zeros((M_INC, 18, 36), dtype=np.float64)
    p_detect = np.zeros((M_INC, 18, 36), dtype=np.float64)

    t0_group = time.perf_counter()

    for seed in range(N_SEEDS):
        nodes, normals, areas = perturb_mesh_3d(
            base_n, base_nm, base_a, eps, tgt['char_size'], seed)
        N = len(nodes)

        # Build A once for this seed
        A_d = asm.build_matrix(nodes, areas, k, precision='c64')

        # Solve 72 RHS using CuPy GMRES — A stays on GPU between solves
        sigma_cols = np.zeros((N, M_INC), dtype=np.complex128, order='F')
        for j, d in enumerate(INC_DIRS):
            b_d = cp.asarray(make_rhs_3d(nodes, k, d).astype(np.complex64))
            def mv(v, A=A_d): return A @ v
            op  = LinearOperator((N, N), matvec=mv, dtype=cp.complex64)
            x_d, _ = cp_gmres(op, b_d, tol=GMRES_TOL,
                               restart=GMRES_RESTART, maxiter=GMRES_MAXITER)
            sigma_cols[:, j] = cp.asnumpy(x_d).astype(np.complex128)

        # Batched bistatic sweep — (72, 18, 36) in one GPU matmul
        rcs_cube = bistatic_sphere_sweep_batch(
            nodes, areas, sigma_cols, k, OBS_THETA, OBS_PHI)  # (72,18,36) m²
        rcs_db   = 10.0 * np.log10(np.maximum(rcs_cube, 1e-20))

        # Online Welford update
        count  += 1
        delta   = rcs_db - mean_rcs
        mean_rcs += delta / count
        delta2   = rcs_db - mean_rcs
        M2_rcs  += delta * delta2
        p_detect += (rcs_db > THRESHOLD_DBSM).astype(np.float64)

        # Free GPU matrix to avoid VRAM accumulation across seeds
        del A_d
        cp.get_default_memory_pool().free_all_blocks()

    std_rcs   = np.sqrt(M2_rcs / max(count - 1, 1))
    p_detect /= count

    np.savez_compressed(outfile,
                        mean=mean_rcs.astype(np.float32),
                        std=std_rcs.astype(np.float32),
                        p_detect=p_detect.astype(np.float32),
                        n_seeds=count)

    t_group = time.perf_counter() - t0_group
    if verbose:
        back_mean = mean_rcs.mean()
        max_pdet  = p_detect.max()
        print(f'    {tgt["name"]:8s}  {t_group:.1f}s  '
              f'mean_rcs={back_mean:.1f}dBsm  max_pdet={max_pdet:.0%}')

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

    print(f'Stage 7 3D full bistatic tensor:')
    print(f'  {len(TARGETS)} tgts × {len(ROUGHNESS_FRACS)} roughness × '
          f'{len(WAVENUMBERS)} freq × {N_SEEDS} seeds = {total} groups')
    print(f'  N={N_PANELS}  M_inc={M_INC}  obs=18×36  '
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
          f'{elapsed/60:.1f} min total')
    return dict(n_written=n_written, n_skipped=skipped, elapsed=elapsed)


def count_existing():
    return sum(1 for _ in DATA_ROOT.rglob('rcs3d_s7_T*.npz'))


def load_group(t_idx, r_idx, f_idx):
    path = DATA_ROOT / f'R{r_idx}_F{f_idx}' / f'rcs3d_s7_T{t_idx:02d}.npz'
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
