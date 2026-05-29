"""
NTK / Hessian demo benchmarks.

Two sweeps:
  1. Lanczos eigenspectrum: SciPy eigsh (CPU) vs GPU Lanczos (MPDOK).
  2. NTK solve scaling:     SciPy (OOM at N~12k) vs MPDOK LU-IR.

Usage:
    conda run -n py314 python benchmark.py
    conda run -n py314 python benchmark.py --skip-lanczos --ntk-max 20000
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from MPDOK.ntk_hessian.models import get_model, load_mnist
from MPDOK.ntk_hessian.hvp    import make_hvp_fn
from MPDOK.ntk_hessian.lanczos import GPULanczos, scipy_eigsh_cpu
from MPDOK.ntk_hessian.ntk_solver import time_ntk_solve, _gpu_memory_reset


# ── Lanczos sweep ─────────────────────────────────────────────────────────────

def run_lanczos_benchmark(device='cuda', k=50, batch_size=512,
                          save_json=True, verbose=True):
    """Compare SciPy eigsh (CPU) vs GPU Lanczos for Hessian eigenvalues."""
    print('\n' + '='*60)
    print('Lanczos Benchmark — Hessian Eigenspectrum')
    print('='*60)

    model_gpu = get_model(hidden=(512, 256), device=device, verbose=verbose)
    P = model_gpu.num_params
    print(f'  MnistMLP: P = {P:,} parameters')
    print(f'  Hessian size: {P}×{P} = {P**2 * 8 / 1e12:.2f} TB  (uncomputable)')

    X_batch, y_batch = load_mnist('train', max_n=batch_size, device=device)

    # ── GPU Lanczos ───────────────────────────────────────────────────────────
    print('\n  [1/2] GPU Lanczos (MPDOK)')
    hvp_gpu = make_hvp_fn(model_gpu, X_batch, y_batch, device=device)
    solver  = GPULanczos(hvp_gpu, P, device=device)
    evals_gpu, _, info_gpu = solver.run(k=k, verbose=verbose)
    t_gpu = info_gpu['elapsed']

    # ── SciPy eigsh (CPU) ─────────────────────────────────────────────────────
    print('\n  [2/2] SciPy eigsh (CPU)')
    model_cpu = get_model(hidden=(512, 256), device='cpu', verbose=False)
    X_cpu, y_cpu = load_mnist('train', max_n=min(batch_size, 256), device='cpu')
    hvp_cpu = make_hvp_fn(model_cpu, X_cpu, y_cpu, device='cpu')
    evals_cpu, t_cpu, n_calls = scipy_eigsh_cpu(hvp_cpu, P, k=k, verbose=verbose)

    speedup = t_cpu / t_gpu
    print(f'\n  Speedup: GPU Lanczos {speedup:.1f}× faster than SciPy eigsh')

    results = {
        'P': P,
        'k': k,
        't_gpu_lanczos': t_gpu,
        't_scipy_eigsh': t_cpu,
        'speedup': speedup,
        'n_scipy_calls': n_calls,
        'evals_gpu': evals_gpu.tolist(),
        'evals_cpu': evals_cpu.tolist(),
        'info': info_gpu,
    }

    if save_json:
        path = os.path.join(HERE, 'lanczos_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Saved to {path}')

    return results


# ── NTK solve sweep ───────────────────────────────────────────────────────────

def run_ntk_benchmark(device='cuda', nugget=1e-4,
                      N_scipy=None, N_mpdok=None,
                      save_json=True, verbose=False):
    """NTK solve scaling: SciPy vs MPDOK across increasing N."""
    print('\n' + '='*60)
    print('NTK Solve Benchmark — Scaling Comparison')
    print('='*60)

    N_scipy = N_scipy or [2000, 5000, 8000, 10000, 12000, 15000, 18000]
    N_mpdok = N_mpdok or [2000, 5000, 8000, 10000, 12000, 15000, 18000]

    model = get_model(hidden=(512, 256), device=device, verbose=False)
    X_all, y_all = load_mnist('train', device='cpu')
    X_all_np = X_all.numpy()
    y_all_np = y_all.numpy()

    results = {'scipy': [], 'mpdok': []}

    # ── SciPy ─────────────────────────────────────────────────────────────────
    print('\n  SciPy solve (CPU Cholesky):')
    for N in N_scipy:
        X_np = X_all_np[:N]; y_np = y_all_np[:N]
        print(f'    N={N:>6,}  K={N*N*8/1e9:.2f} GB …', end=' ', flush=True)
        try:
            _, elapsed = time_ntk_solve(model, X_np, y_np, backend='scipy',
                                         nugget=nugget, device=device, verbose=False)
            if elapsed is None:
                print('OOM')
                results['scipy'].append({'N': N, 'success': False, 'error': 'OOM'})
                break
            print(f'{elapsed:.2f}s')
            results['scipy'].append({'N': N, 'success': True, 'time': elapsed})
        except Exception as e:
            print(f'ERROR: {e}')
            results['scipy'].append({'N': N, 'success': False, 'error': str(e)})
            break
        _gpu_memory_reset()

    # ── MPDOK ─────────────────────────────────────────────────────────────────
    print('\n  MPDOK LU-IR (GPU tensor cores):')
    for N in N_mpdok:
        X_np = X_all_np[:N]; y_np = y_all_np[:N]
        print(f'    N={N:>6,}  K={N*N*8/1e9:.2f} GB …', end=' ', flush=True)
        try:
            _, elapsed = time_ntk_solve(model, X_np, y_np, backend='mpdok',
                                         nugget=nugget, device=device, verbose=False)
            if elapsed is None:
                print('OOM')
                results['mpdok'].append({'N': N, 'success': False, 'error': 'OOM'})
            else:
                print(f'{elapsed:.2f}s')
                results['mpdok'].append({'N': N, 'success': True, 'time': elapsed})
        except Exception as e:
            print(f'ERROR: {e}')
            results['mpdok'].append({'N': N, 'success': False, 'error': str(e)})
        _gpu_memory_reset()

    if save_json:
        path = os.path.join(HERE, 'ntk_benchmark_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\n  Saved to {path}')

    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(lanczos_res, ntk_res, out_dir=None):
    out_dir = out_dir or HERE
    BG = '#0d1117'; FG = 'white'; GRID = '#374151'
    C_SCIPY = '#f97316'; C_MPDOK = '#22d3ee'; C_OOM = '#ef4444'

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=FG); ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG); ax.title.set_color(FG)

    # ── panel 1: eigenspectrum ────────────────────────────────────────────────
    ax = axes[0]
    if lanczos_res:
        evals = np.array(lanczos_res['evals_gpu'])
        ax.semilogy(range(1, len(evals) + 1), np.abs(evals),
                    'o-', color=C_MPDOK, lw=1.5, ms=4, label='GPU Lanczos')
        if 'evals_cpu' in lanczos_res:
            evals_cpu = np.array(lanczos_res['evals_cpu'])
            ax.semilogy(range(1, len(evals_cpu) + 1), np.abs(evals_cpu),
                        's--', color=C_SCIPY, lw=1, ms=3, alpha=0.7, label='SciPy eigsh')
        ax.set_xlabel('Eigenvalue rank'); ax.set_ylabel('|λ|')
        ax.set_title('Hessian Eigenspectrum\n(bulk + outliers)')
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
        ax.grid(color=GRID, alpha=0.4, lw=0.5)

    # ── panel 2: NTK scaling ──────────────────────────────────────────────────
    ax = axes[1]
    if ntk_res:
        for backend, color, label in [('scipy', C_SCIPY, 'SciPy (CPU Cholesky)'),
                                       ('mpdok', C_MPDOK, 'MPDOK LU-IR (GPU)')]:
            rows = ntk_res.get(backend, [])
            N_ok = [r['N'] for r in rows if r.get('success')]
            t_ok = [r['time'] for r in rows if r.get('success')]
            N_oom = [r['N'] for r in rows if not r.get('success')]
            if N_ok:
                ax.plot(N_ok, t_ok, 'o-', color=color, label=label, lw=2, ms=5)
            for N_o in N_oom:
                ax.axvline(N_o, color=C_OOM, lw=1, ls='--', alpha=0.7)
                ax.text(N_o, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 1,
                        'OOM', color=C_OOM, fontsize=8, ha='center', va='bottom',
                        rotation=90)
        ax.set_xlabel('N (training samples)'); ax.set_ylabel('Time (s)')
        ax.set_title('NTK Solve Scaling\n(K is N×N dense SPD)')
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
        ax.grid(color=GRID, alpha=0.4, lw=0.5)

    # ── panel 3: speedup bar ──────────────────────────────────────────────────
    ax = axes[2]
    bars = []
    if lanczos_res:
        bars.append(('Lanczos\neigsh', lanczos_res.get('speedup', 0)))
    if ntk_res:
        # Find the largest N where both succeed
        scipy_ok = {r['N']: r['time'] for r in ntk_res.get('scipy', [])
                    if r.get('success')}
        mpdok_ok = {r['N']: r['time'] for r in ntk_res.get('mpdok', [])
                    if r.get('success')}
        common = set(scipy_ok) & set(mpdok_ok)
        if common:
            N_max = max(common)
            su = scipy_ok[N_max] / mpdok_ok[N_max]
            bars.append((f'NTK solve\nN={N_max:,}', su))
    if bars:
        labels, vals = zip(*bars)
        bar_x = np.arange(len(bars))
        rects = ax.bar(bar_x, vals, color=[C_MPDOK] * len(bars),
                       alpha=0.85, width=0.5, edgecolor=GRID, linewidth=0.5)
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.3,
                    f'{v:.1f}×', ha='center', va='bottom', color=FG, fontsize=12,
                    fontweight='bold')
        ax.set_xticks(bar_x); ax.set_xticklabels(labels, color=FG, fontsize=10)
        ax.set_ylabel('Speedup vs SciPy (CPU)'); ax.set_title('MPDOK Speedup')
        ax.axhline(1, color=FG, lw=0.8, ls='--', alpha=0.5)
        ax.grid(color=GRID, alpha=0.4, lw=0.5, axis='y')

    fig.suptitle('MPDOK Tensor-Core Engine — Deep Learning Mathematical Foundations',
                 color=FG, fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'ntk_benchmark.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'  Figure saved to {path}')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-lanczos', action='store_true')
    parser.add_argument('--skip-ntk',     action='store_true')
    parser.add_argument('--k',     type=int, default=50)
    parser.add_argument('--ntk-max', type=int, default=20000)
    args = parser.parse_args()

    lanczos_res = ntk_res = None

    if not args.skip_lanczos:
        lanczos_res = run_lanczos_benchmark(k=args.k)

    if not args.skip_ntk:
        # Cap MPDOK at N=18k — NVHPC device-alloc leaks accumulate in one process;
        # 4 benchmark points (5k,10k,15k,18k) leak ~2.7 GB which fits in 8 GB VRAM.
        max_scipy = min(args.ntk_max, 18000)
        N_scipy = [n for n in [2000, 5000, 8000, 10000, 12000, 15000, 18000]
                   if n <= max_scipy]
        N_mpdok = [n for n in [2000, 5000, 8000, 10000, 12000, 15000, 18000]
                   if n <= min(args.ntk_max, 18000)]
        ntk_res = run_ntk_benchmark(N_scipy=N_scipy, N_mpdok=N_mpdok)

    if lanczos_res or ntk_res:
        plot_results(lanczos_res, ntk_res)
