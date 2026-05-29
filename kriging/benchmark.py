"""
Kriging scaling benchmark: SciPy vs CuPy vs MPDOK.

The key story:
  - SciPy (CPU) runs out of RAM / time at n ≈ 20k (4 GB for FP64 alone)
  - CuPy alone OOMs VRAM at moderate n (no managed memory)
  - MPDOK solves up to n = 100k+ via LU-IR on tensor cores

Run:
    conda run -n py314 python -m MPDOK.kriging.benchmark
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

# Append project root so MPDOK is importable when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cupy as cp
from MPDOK.kriging.kriging_solver import run_trial, _gpu_memory_reset


# ── default sweep ─────────────────────────────────────────────────────────────

# N ceilings are empirical — extend to find real OOM boundaries
N_VALUES_SCIPY = [1_000, 2_000, 5_000, 10_000, 15_000, 20_000]
N_VALUES_CUPY  = [1_000, 2_000, 5_000, 10_000, 20_000, 22_000, 24_000]
N_VALUES_MPDOK = [1_000, 2_000, 5_000, 10_000, 15_000, 20_000, 22_000, 25_000, 30_000]


def run_sweep(backends=None, save_json=True, out_dir=None):
    if backends is None:
        backends = ['scipy', 'cupy', 'mpdok']
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    results = []

    for backend in backends:
        if backend == 'scipy':
            ns = N_VALUES_SCIPY
        elif backend == 'cupy':
            ns = N_VALUES_CUPY
        else:
            ns = N_VALUES_MPDOK

        # Flush GPU memory pool before each backend to prevent cross-contamination
        _gpu_memory_reset()

        print(f"\n{'='*60}")
        print(f"Backend: {backend.upper()}")
        print(f"{'='*60}")

        for N in ns:
            print(f"  N={N:>7,} ... ", end='', flush=True)
            r = run_trial(N, backend=backend)
            results.append(r)

            if r['success']:
                print(f"{r['fit_time']:.2f}s")
            else:
                print(f"FAILED — {r.get('error', '?')}")

    if save_json:
        out_path = os.path.join(out_dir, 'benchmark_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, out_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    backends = ['scipy', 'cupy', 'mpdok']
    colors   = {'scipy': '#e74c3c', 'cupy': '#f39c12', 'mpdok': '#2ecc71'}
    labels   = {'scipy': 'SciPy (CPU)', 'cupy': 'CuPy (GPU, no IR)',
                 'mpdok': 'MPDOK (GPU, LU-IR)'}
    markers  = {'scipy': 'o', 'cupy': 's', 'mpdok': '^'}

    fig, ax = plt.subplots(figsize=(10, 6))

    for backend in backends:
        rows = [r for r in results if r['backend'] == backend and r['success']]
        if not rows:
            continue
        ns = [r['N']        for r in rows]
        ts = [r['fit_time'] for r in rows]
        ax.plot(ns, ts, color=colors[backend], marker=markers[backend],
                linewidth=2.0, markersize=7, label=labels[backend])

        # Mark failures
        fail_rows = [r for r in results if r['backend'] == backend and not r['success']]
        for r in fail_rows:
            ax.axvline(r['N'], color=colors[backend], linestyle=':', alpha=0.5)
            ax.text(r['N'], ax.get_ylim()[1] * 0.95,
                    f"← {backend} fails\nn={r['N']:,}",
                    color=colors[backend], fontsize=8, ha='center', va='top')

    # Annotate actual OOM boundaries derived from results
    for backend in backends:
        fail_rows = [r for r in results if r['backend'] == backend and not r['success']]
        for r in fail_rows:
            ylim = ax.get_ylim()
            ax.text(r['N'] * 1.02, ylim[0] * 3,
                    f"{backend}\nOOM\n{r['N']//1000}k",
                    color=colors[backend], fontsize=8, va='bottom', ha='left')

    ax.set_xlabel('Number of observation points  (N)', fontsize=12)
    ax.set_ylabel('Fit time  (seconds)', fontsize=12)
    ax.set_title('Kriging Scaling Benchmark\nSciPy vs CuPy vs MPDOK  (Matérn-3/2 covariance)',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Memory line — FP64 matrix bytes
    ax2 = ax.twiny()
    ns_all = np.logspace(3, 5, 100)
    gb = ns_all ** 2 * 8 / 1e9
    ax2.plot(ns_all, [1] * len(ns_all), alpha=0)   # invisible — just sets limits
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('FP64 covariance matrix size  (GB)', fontsize=10, color='grey')
    xticks = [1000, 3000, 10000, 30000, 100000]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f'{x**2*8/1e9:.1f}' for x in xticks], color='grey', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'benchmark_scaling.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    plt.close(fig)


# ── visual interpolation map ──────────────────────────────────────────────────

def plot_kriging_map(N=5000, backend='mpdok', out_dir=None):
    """Run kriging on a synthetic field and plot observed vs predicted."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cupy as cp
    from MPDOK.kriging.kriging_kernel import synthetic_field, prediction_grid
    from MPDOK.kriging.kriging_solver import OrdinaryKriging

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\nGenerating kriging map (N={N:,}, backend={backend}) ...", flush=True)

    coords, z = synthetic_field(N, seed=42)
    ok = OrdinaryKriging(model='matern32', backend=backend)
    ok.fit(coords, z)

    grid, res = prediction_grid(domain=100.0, resolution=200)
    z_pred = ok.predict(grid)

    # To numpy for plotting
    coords_np = cp.asnumpy(coords)
    z_np      = cp.asnumpy(z)
    z_pred_np = cp.asnumpy(z_pred).reshape(res, res)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Observed scatter
    sc = axes[0].scatter(coords_np[:, 0], coords_np[:, 1], c=z_np,
                         cmap='RdYlBu_r', s=4, vmin=z_np.min(), vmax=z_np.max())
    axes[0].set_title(f'Observations  (N={N:,})', fontsize=12)
    axes[0].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0])

    # Kriged surface
    im = axes[1].imshow(z_pred_np, origin='lower', extent=[0, 100, 0, 100],
                         cmap='RdYlBu_r', vmin=z_np.min(), vmax=z_np.max())
    axes[1].set_title(f'Kriged surface  ({backend.upper()}, {res}×{res} grid)', fontsize=12)
    axes[1].set_aspect('equal')
    plt.colorbar(im, ax=axes[1])

    fig.suptitle('Ordinary Kriging — Matérn-3/2  |  MPDOK tensor-core backend',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'kriging_map.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Map saved to {out_path}")
    plt.close(fig)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kriging scaling benchmark')
    parser.add_argument('--backends', nargs='+', default=['scipy', 'cupy', 'mpdok'],
                        choices=['scipy', 'cupy', 'mpdok'])
    parser.add_argument('--map-only', action='store_true',
                        help='Skip benchmark, only generate the kriging map')
    parser.add_argument('--map-n', type=int, default=5000)
    parser.add_argument('--map-backend', default='mpdok')
    args = parser.parse_args()

    if not args.map_only:
        results = run_sweep(backends=args.backends)
        plot_results(results)

    plot_kriging_map(N=args.map_n, backend=args.map_backend)
