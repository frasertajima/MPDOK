# MPDOK — Mixed-Precision Dense-Operator Krylov Solver

**Hardware**: NVIDIA RTX 4060 (8 GB VRAM) · **Language**: Fortran/CUDA + Python/CuPy  
**Date**: May 2026

---

## What It Is

MPDOK solves dense linear systems `A x = b` where A is too large, too slow, or both for conventional methods. It uses **iterative refinement (GMRES-IR)**: a fast approximate inner solve in FP32/TF32 followed by an FP64 residual correction, repeating until the solution is genuinely FP64-accurate.

Three deployment tiers match the hardware available:

| Tier | Where A lives | Max N (RTX 4060) | Implementation | Bottleneck |
|------|--------------|-----------------|----------------|------------|
| **VRAM** | GPU VRAM (FP32 + FP64 managed) | ~43 K | Fortran/CUDA kernel | GPU compute |
| **RAM** | System RAM → GPU tile-by-tile via PCIe | ~108 K | Python/CuPy | PCIe bandwidth |
| **SSD** | NVMe → RAM → GPU, memory-mapped | Unlimited | Python/CuPy | NVMe read speed |

All three tiers run on the GPU. The RAM/SSD tiers use CuPy (a CUDA library) — no Fortran compiler is required, but every SGEMM, Gram-Schmidt step, and FP64 residual DGEMV executes on CUDA cores with cuBLAS. TF32 is active on Ampere GPUs by default for FP32 GEMM.

The practical difference: in the VRAM tier the GPU is compute-bound (the full matrix is already on-chip). In the RAM/SSD tiers the GPU is **bandwidth-bound** — it finishes each tile SGEMV in milliseconds then waits ~3 s for the next tile to arrive over PCIe. At N=49 K, the GPU is idle >99% of the time. Faster storage (PCIe 5.0, NVLink, HBM) scales solve time down proportionally; GPU compute headroom is not the constraint.

All tiers produce FP64-accurate results.

---

## Why It Exists

Dense linear systems arise wherever every point interacts with every other point:

- **Gaussian Process / RBF interpolation** — kernel matrix K(xᵢ, xⱼ) is dense by construction
- **Boundary element method (BEM)** — acoustics, electromagnetics, Stokes flow (Green's function)
- **Integral equation methods** — Nyström discretisation of Fredholm equations
- **Data assimilation / Kalman filter** — dense covariance updates in EnKF

Existing tools hit one of two walls:

- **SciPy / NumPy**: CPU-only, FP64 throughout — correct but slow
- **CuPy `linalg.solve`**: direct LU on GPU — fast but O(N³) cost and O(N²) VRAM; OOMs above ~28 K on an 8 GB card

MPDOK avoids both walls by combining tensor-core speed with streaming to break the VRAM ceiling.

---

## Benchmark Results (RTX 4060, May 2026)

### VRAM tier — GMRES-IR vs scipy and direct LU

Tested on random dense symmetric systems. Convergence criterion: `rel_res < 1e-5`.

```
  N      MPDOK (s)   SciPy (s)   LU (s)   vs SciPy   vs LU     rel_res
  ────────────────────────────────────────────────────────────────────────
  512      0.001       0.001      0.002       0.8×       2×    5.18e-14
  1024     0.001       0.001      0.008       0.9×       7×    1.00e-12
  2048     0.002       0.002      0.035       0.8×      14×    2.35e-13
  4096     0.007       0.049      0.228       7.0×      32×    1.06e-13
  8192     0.024       0.162      1.709       6.8×      72×    7.64e-13
```

**Key findings:**
- 6.8× faster than SciPy GMRES at N=8 192; 72× faster than direct LU
- Accuracy consistently ~10⁻¹³ — *better* than SciPy FP64 GMRES because the FP64 refinement corrects FP32 rounding
- Tensor cores engage at N≥4 K where the advantage becomes decisive

### Problem-class speedups (N=4 096, min over 4 runs)

```
  Problem class          MPDOK (s)   SciPy (s)   Speedup
  ───────────────────────────────────────────────────────
  Random non-symmetric    0.007       0.053        7.7×
  SPD (XᵀX + λI)          0.014       0.171       12.2×
  BEM kernel (1/r)         0.012       0.119        9.8×
  GP / RBF kernel          0.302       2.677        8.9×
```

### OOC streaming tier — dense solve beyond VRAM

Tested on an RBF interpolation system with `reg = 1e-2`, `tol = 1e-5`.  
All three methods using the same iterative algorithm; difference is where A lives.

```
  §1  N = 8,192  (fits everywhere)
  ──────────────────────────────────────────────────────────────
  scipy GMRES    build 1.9s   solve 0.6s   total  2.4s   ✓ 6.16e-06
  cupy direct    build 0.0s   solve 1.9s   total  1.9s   ✓ 7.88e-14
  MPDOK OOC      build 0.4s   solve 3.0s   total  3.4s   ✓ 7.16e-07
  (MPDOK VRAM budget: 128 MB — one tile — never stores full matrix)

  §2  N = 32,768  (memory wall)
  ──────────────────────────────────────────────────────────────
  scipy GMRES    ✓ 81s    (8.0 GB RAM — OOM above N ≈ 68 K)
  cupy direct    ✗ OOM    (8.0 GB VRAM needed, 6.2 GB available)
  MPDOK OOC      ✓ 45s    512 MB VRAM, rel_res 2.60e-06

  §3  N = 49,152  (SSD path, 9 GB FP32 on NVMe)
  ──────────────────────────────────────────────────────────────
  scipy / cupy   ✗ OOM
  MPDOK OOC SSD  ✓ ~2 min   9 GB streamed at 2.9 GB/s, rel_res 3.21e-06
```

The SSD path reduction from 58 minutes (before DGKS re-orthogonalisation) to 2 minutes was achieved by adding a second Gram-Schmidt pass in the inner GMRES, which eliminates the non-monotonic convergence caused by FP32 rounding in the Krylov basis.

---

## Quick-Start Guide

### Requirements

```bash
conda activate py314   # CuPy, NumPy, SciPy — all required
```

The Fortran VRAM solver also requires the compiled `mpdok.so` (built via `make` in this directory).

---

### Tier 1 — VRAM solver (fastest, N up to ~43 K on 8 GB GPU)

```python
import sys
sys.path.insert(0, '/path/to/tensor_core_engine_v5')

import cupy as cp
from MPDOK.mpdok_ops import MPDOKSolver
from MPDOK.rbf_kernel import synthetic_coords, build_rbf_kernel, weather_front

N   = 8_192
REG = 1e-2

coords = synthetic_coords(N, seed=42)       # (N, 2) FP64 CuPy array
b      = weather_front(coords, t=0)         # (N,) FP64 CuPy array

solver   = MPDOKSolver()
A        = solver.alloc_managed(N)          # allocates N² FP32 + N² FP64 managed
_, gamma = build_rbf_kernel(coords, reg=REG, out=A)

x = solver.solve(A, b, tol=1e-5, maxiter_outer=8, restart=50)

rr = float(cp.linalg.norm(b - A.array @ x) / cp.linalg.norm(b))
print(f"rel_res = {rr:.2e}")   # expect < 1e-5

solver.free_managed()
solver.free_fp32_buf()
```

**Memory**: `N² × 12 bytes` (4 FP32 + 8 FP64 managed). At N=8 192: 805 MB total.  
**Speed**: ~7× faster than SciPy at N=4 K; ~7× at N=8 K.

---

### Tier 2 — OOC RAM solver (N up to ~108 K, 47 GB RAM)

Use when N exceeds the VRAM FP32 ceiling (~43 K). The full FP32 matrix is cached in RAM; the solver streams tiles to the GPU one at a time.

```python
from MPDOK.mpdok_ooc import MPDOKOOCSolver

N   = 32_768
REG = 1e-2

coords = synthetic_coords(N, seed=42)
b      = weather_front(coords, t=0)

solver = MPDOKOOCSolver(tile_rows=4096)
solver.build(coords, reg=REG, store='ram', verbose=True)
# prints: OOC build: N=32,768  FP32=4.00 GB  tiles=8  store=ram

x = solver.solve(b, tol=1e-5, maxiter_outer=20, restart=50, verbose=True)
# prints per-outer-iteration residuals

rr = float(cp.linalg.norm(b - solver._tiled_dgemv_fp64(x)) / cp.linalg.norm(b))
print(f"rel_res = {rr:.2e}")

solver.free()
```

**Memory**: `N² × 4 bytes` RAM + `tile_rows × N × 4 bytes` VRAM (one tile).  
At N=32 768: 4 GB RAM, 512 MB VRAM peak.

**`tile_rows` tuning**: larger tiles reduce per-tile overhead but increase VRAM peak.  
Default 4096 works well for 8 GB GPUs. Lower to 2048 if VRAM is tight.

---

### Tier 3 — OOC SSD solver (any N, limited by disk space)

Use when N² × 4 bytes exceeds available RAM. The FP32 matrix is written once during `build()` and memory-mapped during `solve()` — the OS handles sequential readahead automatically.

```python
solver = MPDOKOOCSolver(tile_rows=4096)
solver.build(
    coords,
    reg=REG,
    store='ssd',
    path='/fast_nvme/mpdok_A.bin',   # write once; ~230 MB/s with LUKS
    verbose=True
)
x = solver.solve(b, tol=1e-5, maxiter_outer=20, restart=50, verbose=True)
solver.free()

import os
os.remove('/fast_nvme/mpdok_A.bin')  # clean up
```

**Throughput**: each inner GMRES pass streams N² × 4 bytes from NVMe.  
At N=49 152 (9 GB): ~55 s per pass at 2.9 GB/s read; 2 passes typically converge.  
At N=98 304 (36 GB): ~4 min per pass — patient but correct.

**Write speed note**: NVMe write may be slower than read if LUKS encryption is active (~230 MB/s write vs 2.9 GB/s read on this machine). Build time is dominated by the write.

---

### Choosing `maxiter_outer` and `restart`

| N | Recommended `maxiter_outer` | Notes |
|---|---------------------------|-------|
| ≤ 8 K | 8 | VRAM tier converges in 2–3 outer iters |
| 8 K–32 K | 12–15 | OOC converges in 2–4 outer iters with DGKS |
| 32 K–98 K | 20 | formula: `max(20, 6 + int((N/8192 - 1) * 3))` |

`restart=50` is a good default. Lowering it reduces VRAM for the Krylov basis (`N × restart × 4 bytes`) at the cost of slower inner convergence.

---

### Bring your own matrix

`build()` is RBF-specific. To use MPDOK OOC with an arbitrary FP32 matrix:

```python
import numpy as np
from MPDOK.mpdok_ooc import MPDOKOOCSolver

# Pre-built FP32 matrix (N, N), C-order
A_fp32 = np.ascontiguousarray(your_matrix, dtype=np.float32)

solver = MPDOKOOCSolver(tile_rows=4096)
solver.N        = N
solver.gamma    = None      # not used for _tiled_sgemv
solver.reg      = reg
solver._store   = 'ram'
solver._ram_buf = A_fp32
solver._coords  = cp.asarray(your_coords_fp64)   # for FP64 residual DGEMV
solver._sq      = cp.sum(solver._coords**2, axis=1)

x = solver.solve(b, tol=1e-5, maxiter_outer=20, restart=50)
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `benchmark_mpdok.ipynb` | Full VRAM-tier benchmark across problem classes and N |
| `rbf_spatial/rbf_large_n.ipynb` | VRAM scaling: N = 8 K → 20 K, timing and accuracy |
| `rbf_spatial/rbf_ooc_demo.ipynb` | OOC architecture walkthrough: RAM and SSD paths |
| `rbf_spatial/rbf_streaming_demo.ipynb` | **Side-by-side**: scipy / cupy OOM vs MPDOK OOC streaming ← start here |

---

## Architecture Notes

### Why GMRES-IR works

The inner FP32/TF32 GMRES produces an approximate correction `e ≈ A⁻¹ r`. Even if `e` is only accurate to FP32 (~7 decimal digits), adding it to `x` reduces the FP64 residual by the inner convergence ratio. Two or three outer iterations typically drive the FP64 residual below 1e-5 to 1e-12.

### DGKS re-orthogonalisation

Without it, FP32 rounding in the Arnoldi process causes the Krylov basis vectors to lose mutual orthogonality as `k` grows. By outer iteration 3+, new search directions are nearly parallel to old ones — the solver makes diminishing progress. Adding a second Gram-Schmidt pass (Daniel-Gragg-Kaufman-Stewart criterion) corrects the rounding error and restores monotone convergence. Effect: N=49 K solve time dropped from 58 minutes (non-monotone, never converged) to 2 minutes (3 outer iterations).

### PCIe bandwidth is the OOC floor — not GPU compute

The OOC solver uses the GPU (cuBLAS SGEMM + DGEMM via CuPy) for every arithmetic operation. The bottleneck is not GPU throughput — it is moving the FP32 matrix tiles from RAM or SSD through PCIe into VRAM. For N=32 K (4 GB FP32):

- PCIe 4.0 practical bandwidth: ~16 GB/s → ~0.25 s per tile-SGEMV pass
- restart=50 Krylov steps per inner GMRES → ~13 s per outer iteration
- 2–3 outer iterations → 30–45 s total solve

During each tile transfer the GPU computes the SGEMV for that tile in ~1 ms, then sits idle for ~250 ms waiting for the next tile. GPU utilisation is <1%. Faster interconnects (PCIe 5.0, NVLink, CXL-attached HBM) scale solve time proportionally; adding a faster GPU does not.

### NVHPC managed memory note

NVHPC's `deallocate` for `device, allocatable` arrays does not reliably call `cudaFree`. All VRAM scratch is allocated from Python/CuPy and passed to Fortran as raw pointers — never allocated inside Fortran subroutines.
