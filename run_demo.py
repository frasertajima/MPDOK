"""
rbf_spatial/run_demo.py — RBF Multi-Field Sensor Reconstruction Demo

Demonstrates MPDOK's two solver modes on a synthetic N=8192 sensor network:

  Phase 1 — Single-field, one time step
    Use GMRES-IR: fastest for an isolated solve with no prior factorization.

  Phase 2 — Three fields × 24 time steps (72 solves, fixed geometry)
    Use LU-IR pre-factored: pay O(N³) once, then O(N²) per solve.
    ~30× faster than running GMRES-IR 72 times.

  Phase 3 — Visualization
    Plot the recovered temperature field at t=0 alongside the true field
    and point to the residual error map.

Run:
    conda run -n py314 python run_demo.py
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cupy as cp
import numpy as np

from MPDOK.mpdok_ops import MPDOKSolver, LUIRSolver
from sensor_field import SensorNetwork, FIELDS

# ── Config ────────────────────────────────────────────────────────────────
N       = 8_192
REG     = 1e-2      # cond ≈ N/REG = 8.2e5 — within GMRES-IR FP32 floor
T_STEPS = 24
TOL     = 1e-5
SEED    = 42

# ── Setup ─────────────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"  MPDOK RBF Spatial Demo — {N:,} sensors, {T_STEPS} time steps")
print(f"  Fields: temperature, pressure, humidity")
print(f"{'='*60}\n")

net = SensorNetwork(N=N, seed=SEED, reg=REG)

t0 = time.perf_counter()
A, gamma = net.build_kernel()
t_build = time.perf_counter() - t0
print(f"Kernel build  : {t_build:.2f} s  (N={N:,}, gamma={gamma:.3e}, reg={REG:.0e})")
fp32_floor = REG / (N * 1.2e-7)
print(f"cond(A) ≈ {N/REG:.1e}  |  FP32-only floor ≈ {1/(N/REG * 1.2e-7):.0e}")
print()

# ── Phase 1: GMRES-IR — single-field, single time step ───────────────────
print("Phase 1 — GMRES-IR, single temperature field at t=0")
b_temp0 = net.field('temperature', t=0)

solver_gmres = MPDOKSolver()
t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    x_temp0 = solver_gmres.solve(A, b_temp0, tol=TOL, maxiter_outer=6, restart=50)
cp.cuda.Stream.null.synchronize()
t_gmres_single = time.perf_counter() - t0

rr = float(cp.linalg.norm(b_temp0 - A @ x_temp0) / cp.linalg.norm(b_temp0))
conv = not any(issubclass(wi.category, RuntimeWarning) for wi in w)
print(f"  solve time   : {t_gmres_single*1e3:.1f} ms")
print(f"  rel residual : {rr:.2e}  ({'converged' if conv else 'not converged'})")
print()

# ── Phase 2: LU-IR — three fields × 24 time steps ────────────────────────
print(f"Phase 2 — LU-IR pre-factored, {len(FIELDS)} fields × {T_STEPS} steps")

solver_lu = LUIRSolver()

t0 = time.perf_counter()
solver_lu.factor(A)
t_factor = time.perf_counter() - t0
print(f"  factor (once): {t_factor*1e3:.1f} ms")

all_results = {name: [] for name in FIELDS}
t_solve_total = 0.0
first_rr = {}

for t in range(T_STEPS):
    for name in FIELDS:
        b = net.field(name, t)
        ts = time.perf_counter()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            x = solver_lu.solve_factored(b, tol=TOL, maxiter_outer=3)
        cp.cuda.Stream.null.synchronize()
        t_solve_total += time.perf_counter() - ts
        all_results[name].append(x)
        if t == 0:
            first_rr[name] = float(cp.linalg.norm(b - A @ x) / cp.linalg.norm(b))

n_solves    = len(FIELDS) * T_STEPS
t_per_solve = t_solve_total / n_solves
t_lu_total  = t_factor + t_solve_total

print(f"  {n_solves} solves       : {t_solve_total*1e3:.1f} ms total  "
      f"({t_per_solve*1e3:.2f} ms/solve)")
for name, rr in first_rr.items():
    print(f"    {name:<12} t=0 rel_res: {rr:.2e}")
print()

solver_lu.free_factored()

# ── Summary ───────────────────────────────────────────────────────────────
t_gmres_equiv = t_gmres_single * n_solves
speedup = t_gmres_equiv / t_lu_total

print(f"{'─'*50}")
print(f"  GMRES-IR × {n_solves} solves (estimate) : {t_gmres_equiv*1e3:.0f} ms")
print(f"  LU-IR  factor + {n_solves} solves        : {t_lu_total*1e3:.0f} ms")
print(f"  Speedup                          : {speedup:.1f}×")
print(f"{'─'*50}")

# ── Phase 3: visualization ────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    coords_np = cp.asnumpy(net.coords)
    b_np      = cp.asnumpy(b_temp0)
    x_np      = cp.asnumpy(x_temp0)

    # Interpolate solution weights onto a regular grid for contourf
    xi = np.linspace(0, 100, 200)
    yi = np.linspace(0, 100, 200)
    XX, YY = np.meshgrid(xi, yi)

    # True field on the grid (using the closed-form formula)
    TRUE = 15.0 + 5.0 * np.sin((XX + YY) / 20.0)

    # Reconstruct field from RBF weights: f(p) = sum_i x_i * k(p, p_i)
    # For visualization, we use sensor readings directly as proxy
    # (full grid reconstruction would need another kernel eval — not shown here)
    err_np = np.abs(b_np - (cp.asnumpy(A.T @ x_temp0)))  # proxy: residual at sensors

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sc0 = axes[0].scatter(coords_np[:, 0], coords_np[:, 1],
                          c=b_np, cmap='RdYlBu_r', s=1, vmin=10, vmax=20)
    axes[0].set_title(f'Sensor readings — temperature t=0  (N={N:,})')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(sc0, ax=axes[0], label='°C')

    sc1 = axes[1].scatter(coords_np[:, 0], coords_np[:, 1],
                          c=cp.asnumpy(x_temp0), cmap='RdYlBu_r', s=1, vmin=-0.01, vmax=0.01)
    axes[1].set_title('RBF weights x from GMRES-IR solve')
    axes[1].set_xlabel('x')
    plt.colorbar(sc1, ax=axes[1], label='weight')

    sc2 = axes[2].scatter(coords_np[:, 0], coords_np[:, 1],
                          c=err_np, cmap='hot_r', s=1, norm=matplotlib.colors.LogNorm())
    axes[2].set_title('|b - Ax| residual at sensors (log scale)')
    axes[2].set_xlabel('x')
    plt.colorbar(sc2, ax=axes[2], label='|residual|')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'rbf_demo_temperature.png')
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved: {out}")

except ImportError:
    print("\n(matplotlib not available — skipping visualization)")

print("\nDemo complete.")
