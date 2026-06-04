"""
generate_stage4_data.py — Monte Carlo BEM data for Stage 4 detectability study.

5 targets × 4 roughness levels × 5 wavenumbers × 50 seeds = 5,000 bistatic
RCS solves.  Each solve uses N=512 panels (scipy LU, ~0.2s including build).

Checkpoint files are written to:
  stage4_data/checkpoints/R{r}_F{f}/rcs_T{t:02d}_S{s:04d}.bin

Resume support: files that already exist are skipped.

Expected runtime (fresh run): ~17 min on a modern CPU.
Subsequent runs (all files present): <1s.

Physical setup
--------------
Bistatic RCS: radar illuminates from phi_inc=0° (+x direction).  The backscatter
at phi_obs=180° is the relevant monostatic-equivalent.  We report the maximum RCS
in a ±30° arc around backscatter (THREAT_SECTOR) which is what a radar would
integrate over its beam.

Detection threshold: -5 dBm (chosen so smooth stealth body is undetectable at
k≥5, and is detected at k=3 — demonstrating frequency-dependent stealth).
"""

import sys, os, time, argparse

# Force single-threaded LAPACK/OpenBLAS — threading overhead dominates at N=512
# and makes each solve 7× slower than single-threaded.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'BLAS_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np
from pathlib import Path

_HERE  = Path(__file__).parent
_MPDOK = _HERE.parent
for p in [str(_MPDOK), str(_HERE / 'cobol_rcs')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from acoustic_scattering.bem_helmholtz import build_bem_matrix_helmholtz
from radar_scattering.geometry import (
    circle_panels, square_panels, diamond_panels,
    corner_reflector_panels, stealth_panels,
)
from radar_scattering.rcs_bem import solve_bem_scipy, rcs_2d_sweep
from rcs_bridge import write_checkpoint, N_ANGLES, ANGLES_DEG

# ── Configuration ──────────────────────────────────────────────────────────────
N_PANELS       = 512
N_SEEDS        = 50
PHI_INC        = 0.0                          # fixed incident angle
PHI_OBS        = np.deg2rad(ANGLES_DEG)       # 90 observation angles

ROUGHNESS_FRACS = [0.00, 0.05, 0.10, 0.20]   # ε: nominal / mild / moderate / severe
WAVENUMBERS     = [3.0,  5.0,  8.0, 12.0, 16.0]

# Threat sector: ±30° around backscatter (148°–208°)
THREAT_SECTOR   = np.arange(37, 53)           # phi_obs indices
THRESHOLD_DBM   = -5.0                        # detection threshold

# ── Target definitions ─────────────────────────────────────────────────────────
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

DATA_ROOT = _HERE / 'stage4_data' / 'checkpoints'


# ── Roughness perturbation ─────────────────────────────────────────────────────

def perturb(base_nodes, base_normals, base_lengths, eps, char_size, seed):
    """Displace panel midpoints along outward normal by N(0, eps*char_size)."""
    if eps == 0.0:
        return base_nodes, base_normals, base_lengths
    rng = np.random.default_rng(seed)
    dr  = rng.normal(0, eps * char_size, len(base_nodes))
    return (base_nodes + dr[:, None] * base_normals,
            base_normals.copy(),
            base_lengths.copy())


# ── Single solve ───────────────────────────────────────────────────────────────

def solve_one(tgt, eps_idx, freq_idx, seed, data_dir):
    """Solve BEM for one (target, roughness, frequency, seed) combination.

    Returns (rcs_dbm, t_solve) or None if file already exists.
    """
    t = tgt['id']
    eps  = ROUGHNESS_FRACS[eps_idx]
    k    = WAVENUMBERS[freq_idx]
    path = data_dir / f'rcs_T{t:02d}_S{seed:04d}.bin'

    if path.exists():
        return None  # already computed

    # Geometry
    base_n, base_nm, base_l = tgt['geom_fn'](N_PANELS, k)
    nodes, normals, lengths  = perturb(base_n, base_nm, base_l,
                                       eps, tgt['char_size'], seed)
    t0 = time.perf_counter()
    sigma   = solve_bem_scipy(nodes, lengths, k, PHI_INC)
    rcs_lin = rcs_2d_sweep(nodes, lengths, sigma, k, PHI_OBS)
    rcs_dbm = 10.0 * np.log10(np.maximum(rcs_lin, 1e-20))
    dt = time.perf_counter() - t0

    write_checkpoint(path, target_id=t, seed=seed,
                     freq_ghz=k, ka=k * tgt['char_size'],
                     rcs_dbm=rcs_dbm, n_panels=N_PANELS, complete=True)
    return rcs_dbm, dt


# ── Main generation loop ───────────────────────────────────────────────────────

def generate_all(verbose=True, dry_run=False):
    """Generate all 5,000 checkpoint files.

    Returns a summary dict with total time and files written.
    """
    total_jobs    = len(TARGETS) * len(ROUGHNESS_FRACS) * len(WAVENUMBERS) * N_SEEDS
    jobs_done     = 0
    jobs_skipped  = 0
    t_total_start = time.perf_counter()

    for eps_idx, eps in enumerate(ROUGHNESS_FRACS):
        for f_idx, k in enumerate(WAVENUMBERS):
            subdir = DATA_ROOT / f'R{eps_idx}_F{f_idx}'
            subdir.mkdir(parents=True, exist_ok=True)

            for tgt in TARGETS:
                for seed in range(N_SEEDS):
                    if dry_run:
                        path = subdir / f'rcs_T{tgt["id"]:02d}_S{seed:04d}.bin'
                        if path.exists():
                            jobs_skipped += 1
                        jobs_done += 1
                        continue

                    result = solve_one(tgt, eps_idx, f_idx, seed, subdir)
                    jobs_done += 1

                    if result is None:
                        jobs_skipped += 1
                    else:
                        rcs_dbm, dt = result
                        threat_max  = rcs_dbm[THREAT_SECTOR].max()
                        if verbose:
                            pct = 100 * jobs_done / total_jobs
                            elapsed = time.perf_counter() - t_total_start
                            eta = elapsed / jobs_done * (total_jobs - jobs_done)
                            print(f'  [{pct:5.1f}%]  ε={eps:.0%} k={k:.0f} '
                                  f'{tgt["name"]:8s} seed {seed:02d}  '
                                  f'{dt:.2f}s  threat={threat_max:+5.1f}dBm  '
                                  f'ETA {eta/60:.1f}min')

    elapsed = time.perf_counter() - t_total_start
    n_written = jobs_done - jobs_skipped
    print(f'\nDone: {n_written} new files, {jobs_skipped} skipped, '
          f'{elapsed/60:.1f} min total')
    return dict(n_written=n_written, n_skipped=jobs_skipped, elapsed=elapsed)


def count_existing():
    """Return number of existing checkpoint files."""
    return sum(1 for _ in DATA_ROOT.rglob('rcs_T*_S*.bin'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true',
                   help='Count existing files without running solves')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    existing = count_existing()
    total    = len(TARGETS) * len(ROUGHNESS_FRACS) * len(WAVENUMBERS) * N_SEEDS
    print(f'Stage 4 data generation')
    print(f'  Config: {len(TARGETS)} targets × {len(ROUGHNESS_FRACS)} roughness × '
          f'{len(WAVENUMBERS)} frequencies × {N_SEEDS} seeds = {total} solves')
    print(f'  Existing: {existing}/{total} files')
    print(f'  Remaining: {total - existing} solves × ~0.2s ≈ '
          f'{(total-existing)*0.2/60:.1f} min')
    print()

    if args.dry_run or existing == total:
        if existing == total:
            print('All files present — nothing to generate.')
        generate_all(verbose=not args.quiet, dry_run=True)
    else:
        generate_all(verbose=not args.quiet)
