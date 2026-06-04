"""
generate_stage5_data.py — High-fidelity Monte Carlo BEM at N=8,192.

Same parametric grid as Stage 4:
  5 targets × 4 roughness levels × 5 wavenumbers × 50 seeds = 5,000 solves

Stage 4 used N=512 with scipy LU (~0.2 s/solve, 17 min total).
Stage 5 uses N=8,192 with GPU kernel build + MPDOK GMRES:
  - GPU build:    ~0.05 s  (bem_gpu RawKernel, direct VRAM)
  - MPDOK solve:  ~0.3–1.0 s  (k-dependent; harder at high k)
  - Per-solve:    ~0.4–1.1 s  →  5,000 solves ≈ 40 min

Resume support: existing checkpoint files are skipped.
Data stored in stage5_data/checkpoints/ to keep Stage 4 data intact.

Physical setup: identical to Stage 4 (same ROUGHNESS_FRACS, WAVENUMBERS, TARGETS,
THREAT_SECTOR, THRESHOLD_DBM) so results are directly comparable.
"""

import sys, os, time, argparse

# Single-threaded LAPACK for panel geometry only (no LU at this N)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np
import cupy as cp
from pathlib import Path

_HERE  = Path(__file__).parent
_MPDOK = _HERE.parent
for p in [str(_MPDOK), str(_HERE / 'cobol_rcs')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from radar_scattering.geometry import (
    circle_panels, square_panels, diamond_panels,
    corner_reflector_panels, stealth_panels,
)
from radar_scattering.bem_gpu import build_bem_matrix_gpu
from radar_scattering.gmres_complex import (
    ComplexDenseOperator, gmres_complex, diagonal_preconditioner,
)
from rcs_bridge import write_checkpoint, N_ANGLES, ANGLES_DEG
from radar_scattering.rcs_bem import rcs_2d_sweep

# ── Configuration (identical to Stage 4) ──────────────────────────────────────
N_PANELS        = 8192
N_SEEDS         = 50
PHI_INC         = 0.0
PHI_OBS         = np.deg2rad(ANGLES_DEG)

ROUGHNESS_FRACS = [0.00, 0.05, 0.10, 0.20]
WAVENUMBERS     = [3.0,  5.0,  8.0, 12.0, 16.0]

THREAT_SECTOR   = np.arange(37, 53)
THRESHOLD_DBM   = -5.0

GMRES_RESTART   = 50
GMRES_TOL       = 1e-6
GMRES_MAXITER   = 400        # 8 restarts; accept partial convergence at hard k


def _circle(N, k):    return circle_panels(N, R=1.0)
def _square(N, k):    return square_panels(N, L=2.0)
def _diamond(N, k):   return diamond_panels(N, a=1.5, b=1.0)
def _corner(N, k):    return corner_reflector_panels(N, arm_length=2.0)
def _stealth(N, k):   return stealth_panels(N, length=4.0, half_width=0.4)

TARGETS = [
    dict(id=0, name='Circle',  geom_fn=_circle,  char_size=1.0),
    dict(id=1, name='Square',  geom_fn=_square,  char_size=1.0),
    dict(id=2, name='Diamond', geom_fn=_diamond, char_size=1.3),
    dict(id=3, name='Corner',  geom_fn=_corner,  char_size=0.4),
    dict(id=4, name='Stealth', geom_fn=_stealth, char_size=0.4),
]

DATA_ROOT = _HERE / 'stage5_data' / 'checkpoints'


# ── Roughness perturbation ─────────────────────────────────────────────────────

def perturb(base_nodes, base_normals, base_lengths, eps, char_size, seed):
    if eps == 0.0:
        return base_nodes, base_normals, base_lengths
    rng = np.random.default_rng(seed)
    dr  = rng.normal(0, eps * char_size, len(base_nodes))
    return (base_nodes + dr[:, None] * base_normals,
            base_normals.copy(),
            base_lengths.copy())


# ── Single GPU solve ───────────────────────────────────────────────────────────

def solve_one(tgt, eps_idx, freq_idx, seed, data_dir):
    """GPU-build + MPDOK solve for one (target, roughness, frequency, seed).

    Returns (rcs_dbm, t_build, t_solve, converged) or None if file exists.
    """
    t = tgt['id']
    eps  = ROUGHNESS_FRACS[eps_idx]
    k    = WAVENUMBERS[freq_idx]
    path = data_dir / f'rcs_T{t:02d}_S{seed:04d}.bin'

    if path.exists():
        return None

    base_n, base_nm, base_l = tgt['geom_fn'](N_PANELS, k)
    nodes, normals, lengths  = perturb(base_n, base_nm, base_l,
                                       eps, tgt['char_size'], seed)

    # ── GPU build ──────────────────────────────────────────────────────────
    t0_build = time.perf_counter()
    A_gpu    = build_bem_matrix_gpu(nodes, lengths, k)
    cp.cuda.Stream.null.synchronize()
    t_build  = time.perf_counter() - t0_build

    # ── MPDOK GMRES ────────────────────────────────────────────────────────
    d = np.array([np.cos(PHI_INC), np.sin(PHI_INC)])
    b = -np.exp(1j * k * (nodes @ d)).astype(np.complex128)

    op    = ComplexDenseOperator(A_gpu); del A_gpu
    M_inv = diagonal_preconditioner(op)

    t0_solve = time.perf_counter()
    x_gpu, hist, conv = gmres_complex(op, b,
                                      tol=GMRES_TOL,
                                      restart=GMRES_RESTART,
                                      M_inv=M_inv,
                                      maxiter=GMRES_MAXITER)
    cp.cuda.Stream.null.synchronize()
    t_solve = time.perf_counter() - t0_solve

    sigma   = cp.asnumpy(x_gpu)
    op.free()
    cp.get_default_memory_pool().free_all_blocks()

    rcs_lin = rcs_2d_sweep(nodes, lengths, sigma, k, PHI_OBS)
    rcs_dbm = 10.0 * np.log10(np.maximum(rcs_lin, 1e-20))

    write_checkpoint(path, target_id=t, seed=seed,
                     freq_ghz=k, ka=k * tgt['char_size'],
                     rcs_dbm=rcs_dbm, n_panels=N_PANELS, complete=True)

    return rcs_dbm, t_build, t_solve, conv


# ── Main loop ──────────────────────────────────────────────────────────────────

def generate_all(verbose=True, dry_run=False):
    total_jobs   = len(TARGETS) * len(ROUGHNESS_FRACS) * len(WAVENUMBERS) * N_SEEDS
    jobs_done    = 0
    jobs_skipped = 0
    n_diverged   = 0
    t_total      = time.perf_counter()

    for eps_idx, eps in enumerate(ROUGHNESS_FRACS):
        for f_idx, k in enumerate(WAVENUMBERS):
            subdir = DATA_ROOT / f'R{eps_idx}_F{f_idx}'
            subdir.mkdir(parents=True, exist_ok=True)

            for tgt in TARGETS:
                for seed in range(N_SEEDS):
                    if dry_run:
                        jobs_done += 1
                        if (DATA_ROOT / f'R{eps_idx}_F{f_idx}' /
                                f'rcs_T{tgt["id"]:02d}_S{seed:04d}.bin').exists():
                            jobs_skipped += 1
                        continue

                    result = solve_one(tgt, eps_idx, f_idx, seed, subdir)
                    jobs_done += 1

                    if result is None:
                        jobs_skipped += 1
                    else:
                        rcs_dbm, t_build, t_solve, conv = result
                        if not conv:
                            n_diverged += 1
                        if verbose:
                            pct     = 100 * jobs_done / total_jobs
                            elapsed = time.perf_counter() - t_total
                            eta     = elapsed / jobs_done * (total_jobs - jobs_done)
                            threat  = rcs_dbm[THREAT_SECTOR].max()
                            print(f'  [{pct:5.1f}%]  ε={eps:.0%} k={k:.0f} '
                                  f'{tgt["name"]:8s} seed {seed:02d}  '
                                  f'build={t_build:.3f}s solve={t_solve:.3f}s  '
                                  f'conv={conv}  threat={threat:+5.1f}dBm  '
                                  f'ETA {eta/60:.1f}min')

    elapsed = time.perf_counter() - t_total
    n_written = jobs_done - jobs_skipped
    print(f'\nDone: {n_written} new files, {jobs_skipped} skipped, '
          f'{n_diverged} non-converged, {elapsed/60:.1f} min total')
    return dict(n_written=n_written, n_skipped=jobs_skipped,
                n_diverged=n_diverged, elapsed=elapsed)


def count_existing():
    return sum(1 for _ in DATA_ROOT.rglob('rcs_T*_S*.bin'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--quiet',   action='store_true')
    args = p.parse_args()

    existing = count_existing()
    total    = len(TARGETS) * len(ROUGHNESS_FRACS) * len(WAVENUMBERS) * N_SEEDS
    print(f'Stage 5 data generation  (N={N_PANELS}, GPU build + MPDOK GMRES)')
    print(f'  Config: {len(TARGETS)} targets × {len(ROUGHNESS_FRACS)} roughness × '
          f'{len(WAVENUMBERS)} frequencies × {N_SEEDS} seeds = {total} solves')
    print(f'  Existing: {existing}/{total} files')
    print(f'  Remaining: {total-existing} solves × ~0.5s ≈ '
          f'{(total-existing)*0.5/60:.0f} min')
    print()

    if args.dry_run or existing == total:
        if existing == total:
            print('All files present — nothing to generate.')
        generate_all(verbose=not args.quiet, dry_run=True)
    else:
        generate_all(verbose=not args.quiet)
