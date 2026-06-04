"""
generate_stage7_data.py — Full bistatic scattering matrix, N=4096.

For each (target, roughness, k, seed):
  1. Build BEM matrix A once on GPU    (~0.015 s, N=4096)
  2. Solve for 90 incident angles      (Fortran py_bem_solve_multi_rhs)
  3. Compute 90×90 RCS matrix          (rcs_2d_sweep per incident)
  4. Save as .npy float32              (90×90 × 4 bytes = 32 KB)

Total: 5 targets × 4 roughness × 5 wavenumbers × 20 seeds = 2,000 groups
       × 90 solves = 180,000 BEM solves, only 2,000 GPU builds.

Efficiency vs naive (90 full pipelines per group):
  Naive:   2,000 × 90 × 0.015 s build = 2,700 s wasted builds alone
  Stage 7: 2,000 × 0.015 s builds     =    30 s
"""

import sys, os, time, argparse

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np
import cupy as cp
from pathlib import Path

_HERE  = Path(__file__).parent
_MPDOK = _HERE.parent
for p in [str(_MPDOK), str(_HERE), str(_HERE / 'cobol_rcs')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from radar_scattering.geometry import (
    circle_panels, square_panels, diamond_panels,
    corner_reflector_panels, stealth_panels,
)
from radar_scattering.bem_assembly_ops import BEMAssembler
from radar_scattering.rcs_bem import rcs_2d_sweep
from rcs_bridge import N_ANGLES, ANGLES_DEG

# ── Configuration ──────────────────────────────────────────────────────────────
N_PANELS        = 4096
N_SEEDS         = 20
PHI_INC_DEG     = ANGLES_DEG                       # 90 incident angles
PHI_OBS_RAD     = np.deg2rad(ANGLES_DEG)           # 90 observation angles
PHI_INC_RAD     = PHI_OBS_RAD

ROUGHNESS_FRACS = [0.00, 0.05, 0.10, 0.20]
WAVENUMBERS     = [3.0,  5.0,  8.0, 12.0, 16.0]

GMRES_RESTART   = 50
GMRES_TOL       = 1e-6


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

DATA_ROOT = _HERE / 'stage7_data' / 'groups'


def perturb(base_nodes, base_normals, base_lengths, eps, char_size, seed):
    if eps == 0.0:
        return base_nodes, base_normals, base_lengths
    rng = np.random.default_rng(seed)
    dr  = rng.normal(0, eps * char_size, len(base_nodes))
    return (base_nodes + dr[:, None] * base_normals,
            base_normals.copy(), base_lengths.copy())


def build_b_matrix(nodes, k):
    """Build N×90 complex128 RHS matrix (Fortran column-major)."""
    N = nodes.shape[0]
    B = np.zeros((N, N_ANGLES), dtype=np.complex128, order='F')
    for j, phi in enumerate(PHI_INC_RAD):
        d = np.array([np.cos(phi), np.sin(phi)])
        B[:, j] = -np.exp(1j * k * (nodes @ d))
    return B


def solve_one_group(tgt, eps_idx, freq_idx, seed, subdir, asm):
    """
    Solve full bistatic matrix for one (target, roughness, k, seed).
    Returns (rcs_matrix, t_build, t_solve, n_conv) or None if file exists.
    """
    t = tgt['id']
    eps = ROUGHNESS_FRACS[eps_idx]
    k   = WAVENUMBERS[freq_idx]
    path = subdir / f'rcs7_T{t:02d}_S{seed:04d}.npy'
    if path.exists():
        return None

    base_n, base_nm, base_l = tgt['geom_fn'](N_PANELS, k)
    nodes, normals, lengths  = perturb(base_n, base_nm, base_l,
                                       eps, tgt['char_size'], seed)

    B = build_b_matrix(nodes, k)

    t0_build = time.perf_counter()
    X, info = asm.solve_multi_rhs(nodes, lengths, k, B,
                                   restart=GMRES_RESTART, tol=GMRES_TOL)
    t_total = time.perf_counter() - t0_build

    # Compute RCS for each incident angle (X column j → σ_j → rcs row j)
    rcs_matrix = np.zeros((N_ANGLES, N_ANGLES), dtype=np.float32)
    for j in range(N_ANGLES):
        rcs_lin = rcs_2d_sweep(nodes, lengths, X[:, j], k, PHI_OBS_RAD)
        rcs_matrix[j] = (10.0 * np.log10(np.maximum(rcs_lin, 1e-20))).astype(np.float32)

    np.save(path, rcs_matrix)
    return rcs_matrix, t_total, info['n_converged']


def count_existing():
    return sum(1 for _ in DATA_ROOT.rglob('rcs7_T*_S*.npy'))


def generate_all(verbose=True, dry_run=False):
    total_jobs   = len(TARGETS) * len(ROUGHNESS_FRACS) * len(WAVENUMBERS) * N_SEEDS
    jobs_done    = 0
    jobs_skipped = 0
    t_total_run  = time.perf_counter()

    asm = BEMAssembler()

    for eps_idx, eps in enumerate(ROUGHNESS_FRACS):
        for f_idx, k in enumerate(WAVENUMBERS):
            subdir = DATA_ROOT / f'R{eps_idx}_F{f_idx}'
            subdir.mkdir(parents=True, exist_ok=True)

            for tgt in TARGETS:
                for seed in range(N_SEEDS):
                    if dry_run:
                        jobs_done += 1
                        path = subdir / f'rcs7_T{tgt["id"]:02d}_S{seed:04d}.npy'
                        if path.exists():
                            jobs_skipped += 1
                        continue

                    result = solve_one_group(tgt, eps_idx, f_idx, seed, subdir, asm)
                    jobs_done += 1

                    if result is None:
                        jobs_skipped += 1
                    else:
                        rcs_mat, t_grp, n_conv = result
                        if verbose:
                            pct     = 100 * jobs_done / total_jobs
                            elapsed = time.perf_counter() - t_total_run
                            eta     = elapsed / jobs_done * (total_jobs - jobs_done)
                            # peak from stealth monostatic (diag element)
                            diag_max = max(rcs_mat[i, (i + 45) % N_ANGLES] for i in range(N_ANGLES))
                            print(f'  [{pct:5.1f}%]  ε={eps:.0%} k={k:.0f} '
                                  f'{tgt["name"]:8s} seed {seed:02d}  '
                                  f'{t_grp:.1f}s  conv={n_conv}/{N_ANGLES}  '
                                  f'diag_peak={diag_max:+.1f}dBm  '
                                  f'ETA {eta/60:.1f}min')

    elapsed = time.perf_counter() - t_total_run
    n_written = jobs_done - jobs_skipped
    print(f'\nDone: {n_written} groups, {jobs_skipped} skipped, '
          f'{elapsed/60:.1f} min total')
    return dict(n_written=n_written, n_skipped=jobs_skipped, elapsed=elapsed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--quiet',   action='store_true')
    args = p.parse_args()

    existing = count_existing()
    total    = len(TARGETS) * len(ROUGHNESS_FRACS) * len(WAVENUMBERS) * N_SEEDS
    print(f'Stage 7 bistatic matrix generation')
    print(f'  Config: {len(TARGETS)} targets × {len(ROUGHNESS_FRACS)} roughness × '
          f'{len(WAVENUMBERS)} frequencies × {N_SEEDS} seeds = {total} groups')
    print(f'  Each group: 1 GPU build + {N_ANGLES} GMRES solves → (90,90) RCS matrix')
    print(f'  Existing: {existing}/{total} groups')
    print(f'  Remaining: {total-existing} groups × ~9s ≈ '
          f'{(total-existing)*9/3600:.1f} h')
    print()

    if args.dry_run or existing == total:
        if existing == total:
            print('All files present — nothing to generate.')
        generate_all(verbose=not args.quiet, dry_run=True)
    else:
        generate_all(verbose=not args.quiet)
