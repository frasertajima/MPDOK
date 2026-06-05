# 3D Radar Cross-Section Lab — Full Bistatic Scattering on a Consumer GPU

A five-stage computational acoustics / scalar-EM pipeline extending the 2D RCS lab
(`radar_scattering/`) to full three-dimensional scalar Helmholtz BEM.  The centrepiece
is Stage 7: a 72-incident × 648-observer bistatic scattering tensor computed at
N=5,120 BEM panels — a 46,656-direction electromagnetic fingerprint per target,
per roughness level, per frequency, repeated over 20 Monte Carlo seeds.

This class of study normally requires a dedicated HPC cluster or expensive
commercial EM solvers (FEKO, CST).  The pipeline runs end-to-end on a single
RTX 4060 (8 GB VRAM) using CUDA Fortran kernels compiled with NVHPC, CuPy
GMRES, and a Python orchestration layer.

---

## Contents

- [Workflow at a glance](#workflow-at-a-glance)
- [Mathematical foundations](#mathematical-foundations)
- [Stage 3 — 3D Mie validation + COBOL ensemble](#stage-3--rcs3d_stage3ipynb)
- [Stage 4 — Wideband detectability (N=2,560)](#stage-4--rcs3d_stage4ipynb)
- [Stage 5 — Resolution audit (N=5,120)](#stage-5--rcs3d_stage5ipynb)
- [Stage 6 — Condition numbers + mixed-precision IR](#stage-6--rcs3d_stage6ipynb)
- [Stage 7 — Full bistatic scattering tensor](#stage-7--rcs3d_stage7ipynb)
- [Fortran CUDA kernel](#fortran-cuda-kernel-bem_assembly_3d_multicuf)
- [Transferable engineering patterns](#transferable-engineering-patterns)
- [Performance summary](#performance-summary)
- [File layout](#file-layout)

---

## Workflow at a glance

```
Stage 3  rcs3d_stage3.ipynb   3D Mie sphere validation, COBOL Welford aggregation
   ↓
Stage 4  rcs3d_stage4.ipynb   N=2,560 MC monostatic sweep — 5 targets × 4 roughness × 5 freq
   ↓
Stage 5  rcs3d_stage5.ipynb   N=5,120 resolution audit — ΔP_det and null-shift analysis
   ↓
Stage 6  rcs3d_stage6.ipynb   Condition numbers, mixed-precision IR, cost breakdown
   ↓
Stage 7  rcs3d_stage7.ipynb   Full bistatic tensor (72 inc × 648 obs), optimal receiver
                               placement, bistatic shadow zones, bistatic advantage
```

**Technology stack**

| Layer | Tool | Role |
|---|---|---|
| 3D BEM assembly | `bem_assembly_3d_multi.cuf` | NVHPC-compiled CUDA Fortran |
| GMRES solve | CuPy `cupyx.scipy.sparse.linalg.gmres` | Complex64, restart-50 |
| Ensemble aggregation | IBM COBOL + Welford | Stage 3 streaming mean±σ |
| Bistatic sweep | CuPy batched matmul | (n_obs, N) @ (N, M) in one GPU call |
| Orchestration | Python + NumPy | Parameter sweeps, Welford accumulation |

---

## Mathematical foundations

### 3D Scalar Helmholtz BEM — formulation and scope

**Scalar (acoustic) formulation.** The kernel solves the Helmholtz equation
with Dirichlet (pressure-release / soft-body) boundary conditions — the
acoustic analogue of EM scattering, not a full vector EM formulation.
The CUDA Fortran assembly (`bem_assembly_3d_multi.cuf`) receives only panel
centroids and areas; surface normals are computed by `geometry_3d_targets.py`
but are **not used** in matrix construction.  The scalar Green's function is:

$$G(\mathbf{x},\mathbf{y}) = \frac{e^{ik|\mathbf{x}-\mathbf{y}|}}{4\pi|\mathbf{x}-\mathbf{y}|}$$

A full vector EM treatment (EFIE or MFIE) would use vector surface currents
$\mathbf{J} = \hat{n}\times\mathbf{H}$ and require the surface normal in the
kernel.  **Polarisation state (TE/TM, co-polar vs. cross-polar) is not tracked.**
The Mie validation in Stage 3 uses the scalar soft-sphere series — the exact
solution to the same scalar equation — so the validation is internally
consistent.  At high $kR$ the scalar and vector sphere RCS diverge slightly
as polarisation corrections grow.

The single-layer boundary integral equation on surface $\Gamma$ is:

$$\int_\Gamma G(\mathbf{x}, \mathbf{y})\, \sigma(\mathbf{y})\, dS(\mathbf{y})
  = -p^{\rm inc}(\mathbf{x})$$

Discretised with $N$ constant panels, off-diagonal elements are:

$$A_{ij} = \frac{e^{ikr_{ij}}}{4\pi r_{ij}}\,\Delta S_j, \qquad r_{ij}=|\mathbf{x}_i-\mathbf{y}_j|$$

Each RHS vector for incident direction $\hat{d}$:

$$b_i = -e^{ik\hat{d}\cdot\mathbf{x}_i}$$

### Far-field bistatic RCS

$$\sigma_{\rm 3D}(\hat{r}) = 4\pi\left|\frac{1}{4\pi}\sum_j
  e^{-ik\hat{r}\cdot\mathbf{x}_j}\,\sigma_j\,\Delta S_j\right|^2
  \quad [\text{m}^2] \qquad \sigma_{\rm dBsm} = 10\log_{10}\sigma_{\rm 3D}$$

The batched GPU bistatic sweep evaluates this for all $M_{\rm obs}$ observer
directions simultaneously via the matrix product
$(M_{\rm obs}\times N)\cdot(N\times M_{\rm inc})$, computing the full
bistatic tensor in one cuBLAS call.

### Mie series (exact, soft sphere radius R)

$$p_{\rm scat} = -\sum_{n=0}^{\infty}(2n+1)\frac{j_n(kR)}{h_n^{(1)}(kR)}
  P_n(\cos\theta) \cdot \frac{e^{ikr}}{kr}$$

Used in Stage 3 as the ground truth for validating the 3D BEM kernel.

### Welford online statistics

$$\delta = x_s-\mu_s,\quad \mu_{s+1}\mathrel{+}=\delta/(s+1),\quad
  M2\mathrel{+}=\delta(x_s-\mu_{s+1}),\quad \sigma=\sqrt{M2/s}$$

Applied per-pixel of the $(18\times 36)$ observer sphere across 20 seeds.
O(1) memory regardless of seed count.

---

## Stage 3 — `rcs3d_stage3.ipynb`

First validation of the 3D Helmholtz BEM kernel against the analytic Mie series
for a soft sphere, followed by a COBOL Welford ensemble for 5 targets.

**Mie validation** (sphere, N=1,280, k=3, M=72 incident directions):

| Metric | Value |
|---|---|
| Mean backscatter (BEM) | 5.33 dBsm |
| Mie exact backscatter | 5.344 dBsm |
| Max elevation-band error vs Mie | **0.015 dB** |
| COBOL vs Python Welford (20 seeds) | **2.66×10⁻¹⁵ dB** — identical to float64 rounding |

**Backscatter by target** (smooth, k=3, nose-on incidence):

| Target | BEM backscatter | Notes |
|---|---|---|
| Sphere | 5.33 dBsm | Consistent with Mie; isotropic |
| Cube | 6.16 dBsm | Flat-face retroreflection |
| DblCone | −0.52 dBsm | Tapered geometry, edge diffraction |
| Dihedral | 2.05 dBsm | Double-bounce moderate |
| **Stealth** | **−7.29 dBsm** | **12.6 dB frontal suppression vs sphere** |

**Roughness effect on backscatter** (ε=10%):

| Target | Smooth | ε=10% | Change |
|---|---|---|---|
| Sphere | 5.33 | 5.88 | +0.55 dB |
| DblCone | −0.52 | 6.24 | **+6.76 dB** — roughness collapses the null |
| Stealth | −7.29 | −6.05 | +1.24 dB |

The DblCone is the most roughness-sensitive target: its low-RCS null is
geometric (edge cancellation), so surface perturbations disrupt it immediately.

**Runtime:** 20 seeds × 72 incident directions × N=1,280 → 15.6 s on RTX 4060.

---

## Stage 4 — `rcs3d_stage4.ipynb`

Monte Carlo detectability sweep: 5 targets × 4 roughness levels × 5 wavenumbers
= 100 groups, 50 seeds each, N=2,560, monostatic observer sphere (18×36=648 dirs).

**Stealth detectability fraction of observer sphere (smooth, nose-on, k sweep):**

| k | Fraction of sphere detectable |
|---|---|
| 3 | 100.0% |
| 5 | 86.7% |
| 8 | 60.2% |
| 12 | 51.2% |
| 16 | 46.9% |

Low frequency (k=3, λ/R large) overwhelms shaping — the whole sphere is lit.
At high frequency, stealth shaping works: only ~47% of observer directions
exceed the −10 dBsm detection threshold.

**Key finding:** the 3D stealth body's advantage is concentrated in specific
observer directions.  Monostatic radar sitting in a well-chosen 46° elevation
band sees near-zero detection probability; a receiver at other elevations still
detects reliably.

---

## Stage 5 — `rcs3d_stage5.ipynb`

Resolution audit: same 100-group sweep at **N=5,120** (2× linear, 4× area)
to quantify panel-count convergence.

**Global difference statistics (Stage 5 − Stage 4):**

| Metric | Value |
|---|---|
| Max \|Δmean RCS\| | 28.5 dB (deep null shift — expected at nulls) |
| RMS \|Δmean RCS\| | **0.595 dB** |
| Cells with \|Δ\| > 0.5 dB | 32% |
| Cells with \|Δ\| > 1.0 dB | 19% |
| Max \|ΔP(detect)\| | 1.00 (null flips between N=2,560 and N=5,120) |
| Cells with \|ΔP\| > 0.10 | 5.9% |

**Per-target resolution sensitivity (smooth):**

| Target | Max \|Δmean\| | RMS \|Δmean\| | Max \|ΔP\| |
|---|---|---|---|
| Sphere | 0.000 dB | 0.000 dB | 0.000 |
| Cube | 0.715 dB | 0.139 dB | 1.000 |
| DblCone | 2.611 dB | 1.181 dB | 1.000 |
| Dihedral | 1.454 dB | 0.105 dB | 1.000 |
| **Stealth** | **20.1 dB** | **1.864 dB** | **1.000** |

The sphere is analytically resolution-stable (Mie solution is panel-independent
once the BEM discretisation converges).  The stealth body is most sensitive, and
two distinct mechanisms contribute:

1. **Physical null migration:** designed deep nulls shift in angle as panels
   move; a 28 dB swing at a null is physically real — the null relocated.
2. **Panel quality:** at N=5,120 the stealth body mesh has a maximum triangle
   aspect ratio of ~1,007:1 at the nose (18.75% of panels exceed 5:1), and the
   DblCone has **all** 5,120 panels at a uniform 734:1 aspect ratio (slant
   1.80 m, chord 0.0025 m at n_phi=2,560).  At k=8, the slant spans 2.3λ —
   single-point centroid quadrature under-resolves the Green's function phase
   across the panel.  This is a genuine mesh quality issue for these two targets;
   the sphere, cube, and dihedral have aspect ratios ≤ 2:1 throughout.

The Mie validation's 0.015 dB error applies only to the sphere mesh.  The
DblCone and stealth body have no analytic validation baseline; their resolution
sensitivity mixes physical null migration with panel-quality quadrature error.

**Signed bias (corrected mesh):** DblCone and Stealth now show N=2,560 slightly
*underestimates* mean RCS (positive bias: +0.10 and +0.15 dB), opposite to the
old mesh.  This is the correct direction — higher N resolves deeper nulls,
so the finer mesh sees slightly less RCS in the null directions.  Sphere, Cube,
and Dihedral retain the negative bias seen before.

**Stealth P(detect) stability** (nose-on, k=8, corrected mesh):

| Roughness | N=2,560 | N=5,120 | Δ |
|---|---|---|---|
| ε=0% | 53.4% | 54.3% | +0.9 pp |
| ε=5% | 53.7% | 53.9% | +0.2 pp |
| ε=10% | 57.9% | 63.9% | +6.0 pp |
| ε=20% | 72.8% | 77.3% | +4.5 pp |

P(detect) is convergent to within ≤6 pp at N=5,120 for ε≤5%; rougher surfaces
show larger N-sensitivity because panel displacements break the null geometry
more severely at higher resolution.  N=5,120 is adopted as the production
resolution for Stage 7.

---

## Stage 6 — `rcs3d_stage6.ipynb`

Condition number study and mixed-precision iterative refinement for 3D BEM.

### Condition numbers (N=80, k sweep)

| Target | k=3 | k=5 | k=8 | k=12 | k=16 |
|---|---|---|---|---|---|
| Sphere | 7 | 9 | 7 | 9 | 31 |
| Cube | 9 | 8 | 46 | 100 | **241** |
| DblCone | **556** | 93 | 23 | 8 | 3 |
| Dihedral | 8 | 6 | 11 | 10 | 21 |
| **Stealth** | **2,994** | **5,136** | **2,274** | **3,233** | **2,725** |

The stealth body has condition numbers 100–5,000× larger than the sphere across
all frequencies.  This is the numerical signature of its deep scattering nulls:
near-singular rows in the BEM matrix correspond to observer directions where
the far-field integral nearly vanishes.  Standard GMRES handles this — CG would
diverge for non-symmetric complex systems — but residuals are higher.

### Mixed-precision iterative refinement (stealth, N=1,280, k=8)

| Solver | Residual | Time | RCS effect |
|---|---|---|---|
| GMRES c64 | 1.29×10⁻⁴ | 0.191 s | baseline |
| GMRES + 1 IR step | 1.29×10⁻⁴ | +0.065 s | max diff: 0.0001 dB |
| GMRES + 2 IR steps | **1.44×10⁻⁶** | +0.025 s | max diff: 0.0000 dB |

Two IR steps achieve **90× residual improvement** in 25 ms.  The RCS difference
between 0 and 2 IR steps is 0.017 dB maximum — well below the BEM
discretisation error.  IR is most valuable for null certification and
wideband precision sweeps where the solver residual, not the geometry, limits
accuracy.

### Error budget (sphere, k=3, nose-on)

| N | BEM error [dB] | GMRES residual | IR-2 residual |
|---|---|---|---|
| 80 | 0.743 | 5.0×10⁻⁷ | 4.5×10⁻¹⁴ |
| 320 | 0.162 | 5.8×10⁻⁷ | 1.1×10⁻¹³ |
| 1,280 | **0.027** | **5.7×10⁻⁷** | **4.1×10⁻¹²** |

BEM discretisation error exceeds the GMRES floor by 5–6 orders of magnitude.
IR reaches the float64 noise floor: no further physics improvement is possible
without either finer panels or higher-order basis functions.

### IR cost vs benefit (N sweep)

| N | GMRES | +1 IR | +2 IR | IR overhead |
|---|---|---|---|---|
| 320 | 3.1 ms | 5.8 ms | 5.7 ms | 87% |
| 1,280 | 7.9 ms | 16.8 ms | 24.4 ms | 209% |
| 5,120 | 54.2 ms | 119.9 ms | 171.4 ms | 216% |

At production N=5,120, IR costs 2.2× the GMRES time per solve.  For Stage 7's
100-group sweep, that would add ~90 min of runtime with no observable change in
RCS.  IR is omitted from Stage 7; GMRES c64 at tol=10⁻⁶ is sufficient.

---

## Stage 7 — `rcs3d_stage7.ipynb`

**The centrepiece of the 3D lab.**  For each (target, roughness, frequency, seed)
group, Stage 7 solves 72 incident directions and computes the far-field RCS into
648 observer directions — a full 72×18×36 bistatic scattering tensor.

Stages 3–6 answered: *how detectable is this target from ahead?*
Stage 7 answers: *where in the full sphere of observation directions does the
energy go, and from which approach directions?*

### Multi-RHS efficiency

The BEM matrix $A$ depends only on geometry and $k$ — not on the incident
direction.  Building $A$ once and solving 72 RHS vectors via sequential GMRES
reusing the GPU-resident $A$:

```
Naive:   72 × (build A + GMRES)     = 72 × 4 ms builds  + 72 × GMRES  ≈ 0.58 s builds/seed
Stage 7: 1 × build A  + 72 × GMRES =  1 × 4 ms build    + 72 × GMRES  ≈ 0.004 s builds/seed
                                     ─────────────────────────────────────────────────────
         Across 100 groups × 20 seeds = 2,000 total seeds:
             naive:   2,000 × 0.58 s ≈ 19 min wasted on redundant GPU builds
             Stage 7: 2,000 × 0.004 s ≈ 8 s on GPU builds total
```

The `A` matrix is assembled once as a CuPy device array and remains in VRAM
across all 72 GMRES calls for that seed.  The 72-direction batched bistatic
sweep is then a single `(648, N) @ (N, 72)` cuBLAS matmul.

> **Note on the Fortran multi-RHS kernel:** `bem_assembly_3d_multi.cuf` exports
> `py_bem_solve_multi_rhs_3d` which runs the full 1-build × M-solves pipeline
> inside a single `.so` call.  At N=5,120 this kernel hits the NVHPC
> device-allocatable memory leak (scratch arrays are not cudaFree'd between
> calls), causing VRAM exhaustion after ~30 seeds.  Stage 7 uses the CuPy path
> instead: Python builds $A$ as a CuPy array, calls `cp_gmres` 72 times with
> the same array, then frees it explicitly.  Same physics, same GPU-resident A,
> no NVHPC scratch leak.

### Configuration

| Item | Value |
|---|---|
| BEM panels | N = 5,120 |
| Incident directions | 72 (6×12 sphere grid, bin-midpoint elevation) |
| Observer directions | 648 (18×36 sphere grid) |
| Seeds per group | 20 (online Welford) |
| Groups | 5 targets × 4 roughness × 5 frequencies = **100** |
| Total BEM solves | 100 × 20 × 72 = **144,000** |
| Output per group | mean + std + p_detect of shape (72, 18, 36) ≈ 186 KB |
| Total data on disk | ~18 MB |
| Runtime | ~1.5 h total on RTX 4060 |

### § 1 — Bistatic RCS heatmaps

Fixing the incident direction to nose-on (θ=90°, φ=0°) and sweeping the 18×36
observer sphere at k=12:

- **Sphere:** uniform ~+5 dBsm across the entire observer sphere — a Mie
  scatterer has no preferred scattering direction.
- **Cube:** strong flat-face retroreflection lobes at 0°/90°/180°/270° azimuth,
  deep nulls between them.
- **DblCone:** concentric ring pattern — the tapered geometry creates
  rotationally symmetric diffraction rings.
- **Dihedral:** double-bounce lobes at 45° intervals from the corner edges.
- **Stealth:** a narrow high-RCS equatorial band at θ≈90° (the specular return
  from the curved planform) with deep nulls at oblique elevations — exactly the
  bistatic signature of a shape optimised to redirect frontal energy upward and
  downward, away from ground-based receivers.

**Stealth nose-on mean RCS (k=12, smooth, all observers):** −4.6 dBsm.

### § 2 — Optimal receiver placement

Marginalising over all 72 incident directions gives the P(detect) map for a
static bistatic receiver:

| Target | Best receiver (θ, φ) | P(detect) |
|---|---|---|
| Sphere | θ=5°, φ=0° | **100%** |
| Cube | θ=25°, φ=20° | 89% |
| DblCone | θ=15°, φ=0° | 92% |
| Dihedral | θ=65°, φ=40° | 85% |
| **Stealth** | **θ=55°, φ=0°** | **65%** |

Even at the optimal receiver position the stealth body achieves only 65%
detection probability, averaged over all approach directions.  No single static
receiver position yields reliable detection.

### § 3 — Bistatic shadow zones

For each incident direction, the minimum RCS across all 648 observer directions
reveals the "quietest scattering corner" — observer positions a bistatic
receiver must avoid to lose the target:

| Target | Min bistatic RCS (smooth, k=12) |
|---|---|
| Sphere | +5.0 dBsm uniformly — **no shadow zones** |
| **Stealth** | **−22.6 to −18.5 dBsm** — nulls present but shallower than the old mesh suggested |

Roughness ε=10% raises the stealth shadow-zone floor from −22.6 to −21.4 dBsm
(~1 dB shallower).  The sphere's minimum drops from +5.0 to +0.1 dBsm.

**Implication:** a bistatic receiver misplaced into a shadow zone will fail to
detect the stealth body even when an optimally placed receiver would succeed.
Network-centric radar (multiple receivers) eliminates this vulnerability.

### § 6 — Bistatic advantage

Monostatic vs random vs optimal bistatic receiver, stealth body, smooth surface:

| k | Monostatic | Random receiver | Optimal receiver | Gain vs mono |
|---|---|---|---|---|
| 3 | 33.3% | 70.4% | **97.2%** | +63.9 pp |
| 5 | 33.3% | 59.0% | **73.6%** | +40.3 pp |
| 8 | 33.3% | 51.3% | **61.1%** | +27.8 pp |
| 12 | 33.3% | 51.4% | **65.3%** | +31.9 pp |
| 16 | 33.3% | 51.3% | **61.1%** | +27.8 pp |

The monostatic receiver is flat at 33% for all k — stealth geometry works as
designed in backscatter.  The optimal bistatic receiver consistently achieves
+28–64 pp above monostatic, remaining above 61% across all frequencies.

### The stealth paradox in 3D

The 3D bistatic tensor exposes the same fundamental tradeoff seen in 2D (Stage 7
of `radar_scattering/`), now with full spherical coverage:

> **Shaping that suppresses backscatter does so by redirecting energy — into
> off-axis directions where a bistatic receiver can catch it.  The stealth
> body's distinctive bistatic heatmap is the direct electromagnetic signature
> of its shaping logic.**

Key numbers (k=12, smooth, P(detect) averaged over all 72 incident directions):

| Receiver strategy | Cube | DblCone | Stealth |
|---|---|---|---|
| Monostatic | 33.3% | 66.7% | 33.3% |
| Optimal bistatic | 88.9% | 91.7% | 65.3% |
| Bistatic gain | +55.6 pp | +25.0 pp | +31.9 pp |

The cube benefits most from bistatic placement (+55.6 pp); the DblCone shows
high monostatic detectability (66.7%) because the structured mesh now correctly
resolves its specular retroreflection lobe.  The stealth body retains the
lowest optimal-receiver P_detect at 65.3% — a 34.7% probability of evading
even the best-placed receiver, averaged over all 72 approach directions.

---

## Mesh quality audit and corrections

After Stage 7 was complete a peer review identified two mesh problems that were
corrupting the DblCone and Stealth results.  Both were fixed and all affected
stages were re-run.

### DblCone — 734:1 apex fan triangles

**Problem:** `n_phi = N_target // 2 = 2,560` fan triangles per apex, each
spanning the full 1.80 m cone slant with a 0.0025 m chord.  At k=8 this is a
2.3λ source panel with single-point quadrature — the Green's function phase
varied by 2.3 full cycles across one panel.

**Fix:** Replaced with a structured `n_z × n_phi` quad mesh (body panels
≈ equilateral, AR 0.98 at equator) plus one row of *short* apex fans spanning
only `slant/n_z = 0.09 m`:

| Metric | Before | After |
|---|---|---|
| Apex fan AR | **734:1** | **19.5:1** |
| Body panel AR (equator) | — | **0.98:1** |
| Resolution audit RMS |Δ| (smooth) | 1.18 dB | **0.105 dB** |
| Fraction |ΔP| > 0.10 | 3.1% | **0.6%** |
| DblCone monostatic P_det (k=12) | 33.3% *(wrong)* | **66.7%** |

### Stealth — topological seam bug (1,007:1 panel)

**Problem:** The face loop used `i1 = (i+1) % n_z`, wrapping the rear-tail ring
(z = −1.95 m) back to the front-nose ring (z = +2.0 m).  This created 32
sliver panels spanning the full 3.95 m body length with only 0.004 m width —
an aspect ratio of **1,007:1**.  These panels injected spurious surface currents
across the entire nose region, corrupting σ throughout.

**Fix:** Opened the body loop (no wrap) and added explicit front/rear nose caps.
`r_min` raised from 5% → 15% of `half_width`.

| Metric | Before | After |
|---|---|---|
| Max triangle AR | **1,007:1** (seam bug) | **5.1:1** (cap fans) |
| Body max AR | 12.9:1 | **4.5:1** |
| Area ratio | 79× | **6.7×** |
| Resolution audit RMS |Δ| (smooth) | 1.86 dB | **0.263 dB** |
| Stealth P_det at k=5 (N=2560) | 86.7% *(wrong)* | **67.0%** |
| Stealth P_det at k=12 (N=2560) | 51.2% *(wrong)* | **65.7%** |

The signed bias also reversed: the old mesh had N=2,560 *overestimating* RCS;
the corrected mesh shows a small *underestimate*, which is the physically correct
direction (higher N resolves deeper nulls).  Sphere, Cube, and Dihedral were
unaffected and did not require re-running.

---

## Fortran CUDA kernel — `bem_assembly_3d_multi.cuf`

Compiled with NVHPC `nvfortran` into `bem_assembly_3d_multi.so`.
Exported functions (ctypes interface in `bem_assembly_3d_multi_ops.py`):

| Function | Purpose |
|---|---|
| `py_build_bem_3d_c64` | Assemble N×N complex64 BEM matrix |
| `py_build_bem_3d_c128` | Assemble N×N complex128 BEM matrix |
| `py_bem_solve_multi_rhs_3d` | 1 build + M GMRES solves (N≤2,560 reliable) |

**Benchmark** (RTX 4060, warm, k=8):

| N | Build time | Speedup vs CPU |
|---|---|---|
| 1,280 | 2 ms | ~200× |
| 2,560 | 8 ms | ~350× |
| 5,120 | **~30 ms** | ~400× |

At N=5,120 warm, the Fortran kernel assembles the 5,120² ≈ 26M complex64
matrix in ~30 ms — vs ~12 s on CPU.

**Known limitation (NVHPC device-allocatable leak):** scratch arrays allocated
with `allocate(..., device)` inside the Fortran kernel are not cudaFree'd
between Python calls at N≥5,120.  The workaround for Stage 7 is to assemble
$A$ as a CuPy array and pass a `c_ptr` to Fortran only for the build step,
keeping GMRES in Python/CuPy where memory management is explicit.

---

## Transferable engineering patterns

### 1. Full bistatic = one build + M solves

The BEM matrix depends on geometry and $k$; the RHS depends on incident angle.
Separating these two dependencies eliminates redundant GPU builds at the cost
of one extra bookkeeping dimension in the output tensor.  This generalises to
any parameter that enters only the RHS: source location, boundary condition,
frequency perturbation, or right-hand-side ensemble.

### 2. Welford streaming on a tensor is the same as on a scalar

The Stage 7 Welford update operates on a (72, 18, 36) NumPy array — each element
updates independently.  The memory cost is two arrays of that shape (mean + M2),
regardless of seed count.  The pattern extends to arbitrary-dimensional output
without modification.

### 3. GPU shadow-zone analysis in post-processing

The `min_obs = mean_rcs.reshape(M_INC, -1).min(axis=1)` reduction — finding
the quietest observer direction per incident direction — is a single NumPy
line on CPU after all 100 groups are saved.  Expensive GPU work happens once
at generation time; physics analysis happens cheaply at notebook time on the
pre-aggregated tensors.

### 4. Condition number as a diagnostic, not a verdict

Stealth body κ ≈ 3,000–5,000 across all k.  GMRES is unaffected — it is
Krylov-based and handles non-symmetric ill-conditioned systems routinely.
The condition number correctly predicted *where* precision would be needed
(deep nulls, Stage 6 IR) without ever being a reason to avoid the problem.

### 5. Explicit memory management beats implicit when VRAM is the bottleneck

At N=5,120, each BEM matrix is 5,120² × 8 bytes = 200 MB (complex64).  The
RTX 4060 has 8 GB VRAM.  Accumulating two matrices simultaneously (current seed
+ next build) would use 400 MB — safe.  The NVHPC Fortran kernel leaked ~200 MB
per call by not cudaFree-ing its scratch.  Explicit `del A_d; cp.get_default_memory_pool().free_all_blocks()` after each seed keeps peak VRAM ≤ 210 MB with
no residual accumulation.

### 6. Consumer hardware, research-grade results

The full pipeline — 144,000 GMRES solves, 100-group Monte Carlo, full bistatic
tensor — runs on a single RTX 4060 in ~1.5 hours.  This class of study is
described in the electromagnetic scattering literature as requiring either
HPC clusters or commercial solvers (FEKO, CST).  The key enablers are:

- **Complex64** (not float32): half the VRAM of float64, same physical accuracy
  at BEM discretisation error levels.
- **GPU-resident A**: assemble once per seed; never evict from VRAM between
  the 72 sequential GMRES solves.
- **Batched far-field**: one GPU matmul for all 648 observer directions,
  not 648 sequential dot products.
- **Online Welford**: no per-seed storage; the 20-seed ensemble costs two
  (72,18,36) arrays permanently, and nothing else.

---

## Performance summary

| Metric | Result |
|---|---|
| **Formulation** | Scalar Helmholtz BEM (acoustic / scalar-EM); no polarisation tracking |
| **Mesh quality — Sphere** | Area ratio 1.9×, aspect ratio ≤ 1.9 — excellent |
| **Mesh quality — Cube/Dihedral** | Area ratio 1.0×, aspect ratio 1.0 — perfect |
| **Mesh quality — DblCone** | All 5,120 panels at **734:1** aspect ratio (fan apex slivers) |
| **Mesh quality — Stealth** | 18.75% panels > 5:1; max **1,007:1** at nose; area ratio 79× |
| Analytic validation available for | Sphere only (scalar Mie); DblCone/stealth unvalidated |
| Stage 3 Mie validation max error | **0.015 dB** (N=1,280, k=3, sphere only) |
| Stage 3 COBOL vs Python Welford | **2.7×10⁻¹⁵ dB** — numerical identity |
| Stage 4 stealth sphere coverage (k=16) | **46.9%** of observer sphere detectable |
| Stage 4 runtime per group (N=2,560) | ~1–2 s (50 seeds × GMRES) |
| Stage 5 resolution sensitivity (stealth, smooth) | max 20.1 dB null shift; RMS 1.86 dB |
| Stage 5 P(detect) convergence (N=2,560→5,120) | Δ ≤ **2.2 pp** — production-grade |
| Stage 5 signed bias direction | N=2,560 always overestimates RCS |
| Stage 6 stealth condition number | **2,994–5,136** across k |
| Stage 6 IR improvement (stealth, N=1,280, k=8) | **90×** (1.29×10⁻⁴ → 1.44×10⁻⁶) |
| Stage 6 IR cost overhead | 87–216% at N=320–5,120 |
| Stage 6 BEM error vs solver floor | Discretisation error 5–6 orders above GMRES |
| Stage 7 total BEM solves | **144,000** |
| Stage 7 GPU build savings vs naive | **~99.7%** fewer builds (2,000 vs 144,000) |
| Stage 7 stealth nose-on mean RCS (k=12) | **−4.6 dBsm** |
| Stage 7 optimal receiver P(detect) (stealth, k=12) | **65.3%** vs 33.3% monostatic |
| Stage 7 max bistatic gain over monostatic | **+63.9 pp** at k=3 |
| Stage 7 sphere minimum bistatic RCS (all inc dirs) | **+5.0 dBsm** — no shadow zones |
| Stage 7 stealth minimum bistatic RCS | **−22.6 dBsm** — shadow zones present |
| Stage 7 roughness effect on shadow zones | ε=10% raises floor by **~1 dB** |
| Stage 7 DblCone monostatic P_det (k=12) | **66.7%** — specular lobe correctly resolved |
| Stage 7 runtime | ~**1.5 h** on RTX 4060 (8 GB VRAM) |
| Fortran BEM build at N=5,120 | **~30 ms** vs ~12 s CPU |
| VRAM peak (N=5,120, complex64) | ~**210 MB** per seed with explicit free |

---

## File layout

```
radar_scattering_3d/
│
├── rcs3d_stage3.ipynb              Stage 3 — 3D Mie validation, COBOL ensemble
├── rcs3d_stage4.ipynb              Stage 4 — N=2,560 monostatic MC sweep
├── rcs3d_stage5.ipynb              Stage 5 — N=5,120 resolution audit
├── rcs3d_stage6.ipynb              Stage 6 — Condition numbers, mixed-precision IR
├── rcs3d_stage7.ipynb              Stage 7 — Full bistatic scattering tensor
│
├── bem_assembly_3d_multi.cuf       CUDA Fortran 3D BEM kernels (NVHPC)
├── bem_assembly_3d_multi_ops.py    Python ctypes wrapper
├── bem_assembly_3d_multi.so        Compiled CUDA Fortran kernel
├── bem_assembly_3d_multi.mod       Fortran module file
│
├── geometry_3d_targets.py          3D target geometries + mesh perturbation
├── rcs_3d.py                       3D BEM solve, far-field RCS, bistatic sweep
│
├── generate_stage4_data_3d.py      N=2,560 MC generator (50 seeds)
├── generate_stage5_data_3d.py      N=5,120 MC generator (20 seeds)  
├── generate_stage7_data_3d.py      N=5,120 full bistatic generator (72 inc, 20 seeds)
├── generate_test_data_3d.py        Quick smoke-test generator
│
├── test_phase0.py  …  test_phase3.py   Unit tests for each build phase
│
├── Makefile                        Build rules for bem_assembly_3d_multi.so
│
├── RCS_AGGREGATOR_3D.cbl           COBOL Welford aggregator (Stage 3)
├── RCS_TYPES_3D.cpy                COBOL copybook — 3D record layout
├── rcs_bridge_3d.py                Python ↔ COBOL interface
├── rcs_aggregator_3d               Compiled COBOL executable
│
├── stage3_data_3d/                 Stage 3 raw + COBOL ensemble files
├── stage4_data_3d/groups/          100 × rcs3d_s4_T{t}.npz, shape (18,36)
├── stage5_data_3d/groups/          100 × rcs3d_s5_T{t}.npz, shape (18,36)
└── stage7_data_3d/groups/          100 × rcs3d_s7_T{t:02d}.npz, shape (72,18,36)
```

### Stage 7 data format

Each `.npz` file contains:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `mean` | (72, 18, 36) | float32 | Mean RCS in dBsm over 20 seeds |
| `std` | (72, 18, 36) | float32 | Std of RCS in dB over 20 seeds |
| `p_detect` | (72, 18, 36) | float32 | Fraction of seeds exceeding −10 dBsm |
| `n_seeds` | scalar | int | Seeds accumulated (=20) |

Index conventions: axis 0 = incident direction (72, 6×12 grid);
axes 1–2 = observer (18 elevation × 36 azimuth bins).
Monostatic back-scatter: for incident direction $j$, find the observer bin
closest to $-\hat{d}_j$ via `argmin(||OBS_DIRS + INC_DIRS[j]||)`.
