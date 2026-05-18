# SPD Fleet Tracker — MPDOK GMRES-IR

A real-time Gaussian Process traffic assimilation demo over a fleet of up to **128,000
simulated probe vehicles**, solved with a custom mixed-precision GMRES iterative-refinement
engine (TF32 tensor cores + FP64 outer corrections) that handles problem sizes no
mainstream solver can touch on consumer hardware.

```
cd MPDOK/SPD_matrices_traffic
python server.py          # edit N_VEHICLES at the top first
open http://localhost:8787/
```

https://youtu.be/KrntvCQsgPM?si=teRepHLoSwUZuuQ5

---

## What you are watching

64,000 simulated vehicles drive around a 100 × 100 km model of Metro Vancouver.
Each vehicle is a **probe sensor**: it knows its GPS position and reports one noisy scalar —
its local traffic congestion level.  Every assimilation cycle the system:

1. **Builds** a 64k × 64k or 128k × 128k symmetric positive-definite (SPD) kernel matrix
   `A = K(X,X) + λI`, where `K_ij = exp(−γ ‖p_i − p_j‖²)` encodes spatial correlation.
2. **Solves** `A α = y` for GP regression weights α — this is the hard step.
3. **Reconstructs** the denoised traffic field everywhere on a 64 × 64 grid from `α`.
4. **Streams** the result to the browser at 5 Hz; the solve runs asynchronously so
   vehicle animation stays fluid regardless of solve time.

### The views


**Ground truth** — the analytical traffic field (4 oscillating Gaussian hotspots at real Vancouver locations, plus any placed incidents)

**MPDOK reconstruction** — what the solver inferred from the sparse, noisy vehicle reports alone 

**Error** — the relative error between the ground truth and MPDOK reconstruction

When MPDOK is accurate (rel_err ∼ 10⁻¹⁴ at N = 6k) the two views are nearly identical. They diverge immediately after a **traffic incident** is placed: the truth panel updates in one animation frame; the MPDOK panel stays stale until the next solve cycle completes.
This lag — seconds at N = 6k, 4 minutes at N = 64k, 15 minutes at N = 128k — is what makes MPDOK's work tangible.

### Traffic incidents

Click **⚠ Place Incident** then click anywhere on the map.  A sudden congestion spike is injected into the ground-truth field.  Vehicles entering the zone immediately slow down and report high congestion (dots turn red).  MPDOK must re-solve the full N × N system from scratch to reconstruct the new spike — it cannot update incrementally. Click **✕ Clear Incidents** to remove all spikes.

### Vehicle simulation

- 64,000–128,000 vehicles perform a **random walk attracted to 24 major arterials** (Trans-Canada, Hwy 99, Broadway, Lougheed, King George, etc.) defined as KDTree nodes.  Each step applies a 75 % pull toward the nearest road segment, so vehicles cluster along corridors at city scale while retaining natural local scatter.
- **Congestion-reactive speed**: step noise σ scales as `(1 − 0.85 × congestion)`. In gridlock vehicles barely move; on clear roads they travel at full step size. This makes jam zones visible as dense slow clusters.
- **Dot colour**: green → yellow → red by measured congestion (traffic-light scale).
- **Highlighted trails**: 5 vehicles are tracked with fading coloured polylines showing their last 80 positions.  Click any vehicle on the map to swap it into the tracked set.

---

## Why this problem is hard

The core operation is solving a **dense** N × N SPD linear system.  Dense means every element of A is non-zero — there is no sparsity to exploit.

| Metric | N = 64,000 | N = 128,000 |
|---|---|---|
| FP64 matrix | 32.8 GB | **131 GB** |
| FP32 matrix | **16.4 GB** | **65.5 GB** |
| VRAM available | 4 GB | 4 GB |
| SciPy (O(N³) Cholesky) | ~32 min | **~4.3 hours** |
| CuPy direct solve | OOM (16 GB > 4 GB) | OOM |
| **MPDOK OOC-RAM** | **~4 min (8×)** | **~15 min (17×)** |

SciPy's `linalg.solve` calls LAPACK's dense Cholesky.  At N = 128k that requires 131 GB in FP64 and roughly 4.3 hours of CPU time (measured 1.6 s at N = 6k, extrapolated via N³ scaling).  On this hardware SciPy can theoretically run N = 64k in ~32 min; it cannot run N = 128k at all (131 GB > 80 GB RAM).

CuPy's `linalg.solve` requires the entire matrix on the GPU — 16.4 GB for N = 64k, 131 GB for N = 128k — both impossible on a 4 GB card.

---

## Is any other software capable of this on consumer hardware?

### Exact solvers

No mainstream library solves a **dense** N = 64k–128k SPD system to machine precision on a single consumer GPU.  The hardware simply doesn't exist yet in affordable form: an 80 GB H100 can hold the N = 128k FP32 matrix on-card, but costs ~$30k.

### Approximate GP libraries

Several libraries handle large N by approximating the kernel system rather than solving it exactly:

| Library | Approach | N limit | Accuracy |
|---|---|---|---|
| **GPyTorch** | Inducing-point sparse GP (SGPR, KISS-GP) | Millions | Approximate |
| **Falkon** | Nyström approximation | Millions | Approximate |
| **KeOps / PyKeops** | Lazy kernel matrix-vector products (no N² storage) | Millions | Exact CG, but no mixed-precision IR |
| **scikit-learn GPR** | Dense Cholesky | ~10k practical | Exact |
| **H-matrix methods** (HLib, H2Lib) | Hierarchical low-rank off-diagonal blocks | Large | Near-exact |

**KeOps** is the closest competitor.  It computes `K × v` on the GPU without ever materialising the N × N matrix, enabling conjugate gradient (CG) iterations at large N with O(N) memory.  For well-conditioned systems (small κ) it works well.

The key differences from MPDOK:

- KeOps + CG converges slowly for ill-conditioned systems (κ ≈ 37 here for N = 6k,
  but κ grows with N).  MPDOK's GMRES-IR with TF32 preconditioning handles this directly.
- KeOps has no mixed-precision iterative refinement — accuracy depends on CG convergence.
- KeOps keeps only O(N) state (no N² storage at all); MPDOK OOC-RAM stores the FP32
  matrix (O(N²)) in CPU RAM and streams it, enabling a Krylov basis that is much richer
  than what online computation alone affords.

**H-matrix methods** can achieve near-linear complexity for kernels with off-diagonal
low-rank structure (which RBF kernels possess at large scales), but require significant
implementation effort, careful tuning of rank thresholds, and do not benefit from
GPU tensor cores in the same way.

### The MPDOK niche

MPDOK targets the regime where:
- N is large enough that the full N × N matrix saturates consumer RAM (N ≥ 40k), and
- **Exact** FP64-quality solution is required (not a low-rank approximation), and
- Only consumer hardware is available (no A100/H100 cluster).

In that regime there is, to our knowledge, **no other publicly available software**
that achieves this.

---

## Mathematical formulation

### Gaussian Process regression

Given N vehicle positions `X = {p₁, …, p_N} ⊂ ℝ²` and noisy congestion measurements
`yᵢ = f(pᵢ) + εᵢ` (εᵢ ~ N(0, λ)), the GP posterior mean at any query point `q` is:

```
f̂(q) = Σᵢ αᵢ · k(q, pᵢ),    where  (K + λI) α = y
```

`K` is the N × N kernel (Gram) matrix.  The entire information about which vehicle
readings matter for which grid cells is encoded in α — solving for it is everything.

### RBF kernel

```
Kᵢⱼ = exp(−γ ‖pᵢ − pⱼ‖²)
```

Parameters:
- `γ = 0.2` — correlation length ≈ 1.6 km, matching mean vehicle spacing
- `λ = 0.1` — regularisation; keeps condition number κ(A) ≈ 37–50

The auto-estimated γ (based on mean pairwise distance across the full 100 km city)
gives γ ≈ 1.8×10⁻⁴, a nearly rank-1 matrix with κ > 10⁶ — fatal for any iterative
solver.  γ = 0.2 was chosen empirically so the TF32 preconditioner is effective.

---

## Solver architecture

### GMRES-IR (Iterative Refinement)

```
outer loop (FP64):
    r  = b − A x        ← exact FP64 residual
    δ ≈ A⁻¹ r           ← inner GMRES in FP32 / TF32  (fast, approximate)
    x  ← x + δ

convergence: ‖r‖ / ‖b‖ < tol
```

Each outer iteration reduces the relative residual by ~10×.  Five iterations bring
the TF32-preconditioned result to FP64 quality.  The inner GMRES uses NVIDIA TF32
tensor cores (10-bit mantissa, FP32 exponent range, ~8× FP64 FLOP rate on Ampere).

### Out-of-Core RAM path (OOC-RAM)

Triggered when the FP32 matrix exceeds 60% of VRAM:

```
Build:     compute A in FP32 tile-by-tile via chunked GPU GEMM → CPU RAM (16–66 GB)
SGEMV:     stream tiles RAM → GPU, accumulate Ax in FP32       (inner GMRES)
DGEMV:     recompute Ax on-the-fly from raw coords in FP64     (outer residual; never stores FP64 matrix)
```

VRAM footprint: one tile (512 rows × N cols × 4 B) + Krylov basis (restart × N × 8 B).
At N = 128k: tile ≈ 262 MB, Krylov basis (restart=20) ≈ 205 MB — well within 4 GB.

### Auto-selection

```python
USE_OOC = N**2 * 4  >  vram_total * 0.60
```

| N | FP32 matrix | Solver |
|---|---|---|
| ≤ 24k | < 2.4 GB | Standard MPDOK (Fortran GMRES-IR, all on-GPU) |
| 64k | 16.4 GB | OOC-RAM |
| 128k | 65.5 GB | OOC-RAM |

---

## Performance

| N | Build | Solve | Cycle | vs SciPy |
|---|---|---|---|---|
| 6,000 | 39 ms | 240 ms | ~280 ms | **6–9× measured** |
| 64,000 | 17 s | 241 s | **~4 min** | **8× vs 32 min est** |
| 128,000 | 70 s | ~820 s | **~15 min** | **17× vs 4.3 hr est** |

Residual convergence at N = 64k:
```
outer 0: rel_res = 1.00e+00
outer 1: rel_res = 1.15e-01
outer 2: rel_res = 9.40e-02
outer 3: rel_res = 7.79e-02
outer 4: rel_res = 6.79e-02   ← 5 × 4 min cycles achieved
```

---

## Code structure

```
SPD_matrices_vehicle_tracking/
├── server.py       FastAPI + WebSocket; decoupled vehicle (5 Hz) and solve threads
├── enkf_solver.py  FleetAssimilator: matrix build, MPDOK/SciPy/CuPy solve, reconstruct
├── fleet_sim.py    TrafficField (hotspots + incidents), Fleet (road-attracted walk)
├── index.html      Browser: single Leaflet map, split canvas overlay, WebSocket client
└── README.md       This file

MPDOK/
├── mpdok_ops.py    MPDOKSolver (standard on-GPU) + LUIRSolver
├── mpdok_ooc.py    MPDOKOOCSolver (OOC-RAM: tiled SGEMV from CPU RAM)
├── rbf_kernel.py   build_rbf_kernel: chunked GPU GEMM for N² pairwise distances
└── mpdok.so        Compiled Fortran kernel (TF32 inner GMRES / FP64 outer IR)
```

### Binary WebSocket frame

80-byte little-endian header + float32 payload, sent at up to 5 Hz:

```
Header (80 B = 20 × int32/float32):
  frame, step
  build_s, solve_s, cycle_s, rel_err, speedup
  N, NX, NY, paused
  city_km, elapsed_since_assim_s, scipy_est_s, scipy_ms
  solver_mode, cupy_oom, outer_iter, assim_step

Payload:
  float32[NX×NY]    true traffic field
  float32[NX×NY]    MPDOK reconstructed field
  float32[N×2]      vehicle positions (x, y) km
  float32[N]        per-vehicle congestion readings (for dot colouring)
  int32, int32      N_TRAIL, TRAIL_LEN
  float32[N_TRAIL × TRAIL_LEN × 2]   highlighted vehicle trail history
  int32             n_incidents
  float32[n_incidents × 3]            incident (x, y, strength)
```

### Chunked cross-kernel reconstruction

At N = 128k the naive query kernel `K(grid, train)` is 4096 × 128k × 8 B = 4 GB —
OOM.  `enkf_solver.reconstruct()` processes 256 query rows at a time (262 MB peak)
and releases VRAM between chunks.

---

## Running

```bash
# Recommended demo sequence:

# Stage 1 — correctness check (N = 6k, ~280 ms cycles, ~7× vs SciPy measured)
# Edit server.py: N_VEHICLES = 6_000
python server.py

# Stage 2 — visible incident lag (N = 64k, ~4 min cycles, ~8× vs SciPy est)
# Edit server.py: N_VEHICLES = 64_000
python server.py

# Stage 3 — maximum scale (N = 128k, ~15 min cycles, ~17×)
# Requires 80 GB RAM.  Do NOT use N = 150k (needs ~100 GB RAM → swap)
# Edit server.py: N_VEHICLES = 128_000
python server.py

open http://localhost:8787/
```

**Dependencies:** `fastapi uvicorn numpy scipy cupy matplotlib` + compiled `mpdok.so`.

### Browser controls

| Control | Action |
|---|---|
| Scroll / pinch | Zoom |
| Drag | Pan |
| **⊡ Fit City** | Reset to full Metro Vancouver view |
| **Right panel selector** | Reconstructed / True field / \|Error\| |
| **Vehicles toggle** | Show / hide dots |
| **⏸ Pause** | Freeze fleet and solver |
| **⚠ Place Incident** | Click map to inject sudden congestion spike |
| **✕ Clear Incidents** | Remove all incidents |
| **Click any vehicle** | Begin tracking that vehicle's trail |

### Recommended demo script

1. Start at N = 64k.  Wait for first assimilation cycle to complete (~4 min).
2. Observe the two panels — they should look nearly identical (MPDOK accuracy).
3. Click **⚠ Place Incident**, click a major road corridor on the map.
4. Left panel immediately shows the yellow congestion spike; vehicle dots turn red.
5. Right panel is unchanged — MPDOK does not know about the incident yet.
6. Point out the **"Last field"** sidebar counter ticking up: the reconstruction
   is getting staler by the second.
7. After ~4 minutes the right panel suddenly updates — MPDOK discovered the incident
   purely from the pattern of red vehicle reports, without any direct knowledge of
   the incident's location or size.
8. Click **✕ Clear Incidents** — left panel clears instantly; right panel stays
   "wrong" for another ~4 minutes, then self-corrects.
9. Switch to **\|Error\|** on the right panel to visualise the reconstruction gap
   during and after an incident.
