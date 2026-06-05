# Acoustic Scattering v2 — GPU-Accelerated BEM Built on Radar Scattering Learnings

A ground-up rebuild of `acoustic_scattering/` using every architectural
advancement developed during the `radar_scattering` and `radar_scattering_3d`
projects.  The original lab is kept untouched for comparison.

**Central insight:** the 2D acoustic Helmholtz BEM and the 2D radar EM BEM share
the *same* Green's function `G = (i/4)H₀⁽¹⁾(kr)Δl`.  The Fortran CUDA kernel
compiled for radar (`bem_assembly.so`) is imported directly — zero new Fortran
written for this lab.

---

## What changed and why

| Layer | v1 (original) | v2 (this lab) | Came from |
|---|---|---|---|
| Matrix assembly | CPU `scipy.special.hankel1` | CUDA Fortran `bem_assembly.so` | radar_scattering Stage 2 |
| Solve | Block-real `(2N)×(2N)` scipy LU | Complex `N×N` CuPy GMRES | radar_scattering Stages 4/5 |
| Iterative refinement | None | c64 GMRES → c128 IR | radar_scattering Stage 6 |
| Multi-RHS | M separate builds+solves | 1 build + M solves (Fortran) | radar_scattering Stage 7 |
| Monte Carlo roughness | None | Welford streaming (50 seeds) | radar_scattering Stages 4/5 |
| Bistatic tensor | None | 72×72 scattering matrix | radar_scattering_3d Stage 7 |
| Max feasible N (8 GB) | ~4,096 (LU memory wall) | **~30,000** (GMRES iterative) | — |

---

## Benchmark results (RTX 4060, warm)

| N | Assembly speedup | Solve speedup | Notes |
|---|---|---|---|
| 512 | 228× | 3× | GMRES startup overhead at small N |
| 1,024 | 268× | 38× | v2 dominates above N~800 |
| 2,048 | 298× | 51× | |
| 4,096 | 282× | 58× | |
| 8,192 | **326×** | — | v1 OOM; v2 runs fine |

**Accuracy:** v1 and v2 agree to 6.65×10⁻⁶ dB — float32 cast floor only.  
**Mie validation:** max |BEM−Mie| = 0.001 dB at N=2,048, k=8.

---

## Contents

### Stage 0 — `stage0_comparison.ipynb`
Head-to-head benchmark: v1 CPU stack vs v2 GPU stack.  Self-contained,
runs in ~5 minutes, records the architectural progress from radar_scattering.

### Stage 1 — `stage1_validation.ipynb`
Mie series validation before any performance work. Error convergence table,
condition number survey (max κ=683 at k≈16 — GMRES sufficient throughout),
frequency sweep (63 k values × 4 shapes in 0.1 s).

### Stage 4 — `stage4_roughness_mc.ipynb`
Monte Carlo surface roughness: 4 shapes × 4 ε levels × 5 k × 50 seeds = 400
groups. Welford streaming statistics. Key finding: roughness raises the RCS null
floor; high-k patterns are most sensitive (shorter λ). Consistent with radar
Stage 3 observation that Welford σ peaks at null locations.

### Stage 5 — `stage5_frequency_sweep.ipynb`
79 frequencies × 4 shapes = 316 solves in **10 seconds** on GPU. Shows:
- TSCS resonance fringes converging to 2× geometric at high k
- Bistatic pattern waterfall — forward-scatter ridge dominates all shapes
- Submarine: three qualitatively distinct field regimes (Rayleigh, resonance,
  geometric optics)
- Joukowski: directivity lobe rotates cleanly with incident angle

### Stage 7 — `stage7_bistatic.ipynb`
Full 72×72 bistatic scattering tensor: 4 shapes × 4 ε × 5 k × 20 seeds = 80
groups, N=2,048. Multi-RHS: 1 GPU build × 72 GMRES solves per seed.

**Detection summary (k=6, smooth, threshold=−5 dBm):**

| Shape | Monostatic P_det | Optimal P_det | Bistatic gain | Shadow dirs |
|---|---|---|---|---|
| Circle | 100% | 100% | 0 pp | 0/72 — no shadow zones |
| Ellipse | 100% | 100% | 0 pp | 0/72 |
| **Joukowski** | **100%** | **100%** | **+0.3 pp** | **39/72** — genuine shadow zones |
| Submarine | 100% | 100% | 0 pp | 0/72 |

The Joukowski airfoil is the acoustic analogue of the radar stealth body:
its geometry creates genuine bistatic shadow zones (min_obs < −5 dBm for 54%
of incident directions at k=6), where even an optimally placed bistatic receiver
fails.  At k=10 the shadow depth reaches −10 dBm — growing deeper as the
sharp trailing edge produces narrower, more coherent nulls.  Circle, ellipse
and submarine are convex and scatter strongly in every direction; no bistatic
shadow zone exists for any incident angle.

Bistatic advantage (Joukowski only, smooth): +0.21 pp at k=4, growing to
+0.38 pp at k=10 as shadow zones deepen.  Convex shapes show zero bistatic
advantage — they are uniformly detectable from any receiver position.

Forward-scatter ridge (φ_obs ≈ φ_inc) is bright in all four bistatic matrices,
confirming Babinet's theorem across all target geometries.

---

## Rich examples

### `rich_joukowski.ipynb` — Aeroacoustic fingerprint
- 12 angle-of-attack × 403 (k, α) combinations: **11 seconds** total
- Backscatter rises 21 dB from head-on to broadside
- RCS fingerprint heatmap T(k, α) reveals chord resonance bands at kc≈nπ
- Thickness sweep: max RCS grows 3.4 dB as eps increases from 0.02 to 0.30
- Near-field: bow-on shows trailing-edge wake diffraction; broadside shows
  clean shadow cone

### `rich_submarine.ipynb` — Sonar signature
- Aspect sweep: broadside TS = **20.4 dBm** vs bow-on 5.0 dBm
- Flat-hull geometry adds up to **+6.6 dB** over equivalent ellipse at oblique angles
- Hull fouling (30 seeds): smooth σ=0.00 dB, ε=10% σ=0.40 dB — significant
  uncertainty growth from biofouling
- 213 broadband solves in 5.5 s

### `rich_multi_obstacle.ipynb` — Coupled BEM
Multiple scatterers handled exactly: concatenate all panels into one N_total×N_total
system; cross-obstacle blocks provide exact multiple-scattering coupling.

- **Two cylinders at d=1λ:** median multiple-scattering correction **3.86 dB**
  (independent single-scattering approximation off by median 4 dB)
- **Deep-null filling:** independent model predicts ~200 dB nulls from
  destructive interference; coupled BEM fills them to ~−15 dBm
- **Triangle array:** +10.1 dB peak RCS over single cylinder
- **Linear array:** grating lobes appear exactly at d=nλ spacing
- GPU handles N_total=5×192=960 in **20 ms** (assembly + GMRES)

---

## Known limitations

### 1. Sound-soft (Dirichlet) boundary condition throughout

`bem_helmholtz_v2.make_rhs` enforces `p_total = 0 on Γ` — the pressure-release
(sound-soft) boundary condition, directly inherited from the PEC radar kernel.
This is physically correct for a water–air interface (e.g. a bubble or a thin
membrane) but **incorrect for rigid solid obstacles** such as a submarine hull
or a metal airfoil.

Rigid bodies obey the Neumann (sound-hard) condition `∂p/∂n = 0`, which
requires a double-layer BEM kernel `∂G/∂n · σ` and a modified RHS
`b_i = −∂p_inc/∂n`. All comparative shape rankings in this lab are internally
consistent, but the physical interpretation of "submarine" and "Joukowski" as
solid structures is inaccurate — they model hollow pressure-release shells.
Extending to sound-hard boundaries would require a new RHS function and a
second Fortran kernel for the hypersingular operator.

### 2. Internal resonances — surveyed, not suppressed

Stage 1 maps condition numbers across k=0.5–20 and identifies the irregular
frequencies (zeros of J_n(kR)) as vertical markers. Max κ=683 at k≈16;
GMRES (restart=50) converges throughout the tested range. No Burton–Miller
or CFIE regularisation was applied.

For higher k or larger N, condition-number spikes at interior eigenfrequencies
may cause GMRES stagnation. The fix is Burton–Miller (α-weighted combination
of single- and double-layer operators), which eliminates non-uniqueness at all
frequencies.

---

## Transferable patterns

### 1. Same kernel, different domain
`bem_assembly.so` from radar_scattering assembles the acoustic BEM matrix
unchanged. Any domain governed by the 2D Helmholtz equation (acoustics, EM,
shallow-water waves, quantum scattering) can use the same infrastructure.

### 2. Block-real solve is never the right answer at scale
The v1 block-real (2N)×(2N) LU was convenient but pays 4× the memory price
and O(8N³) compute.  Complex GMRES on the N×N system is always superior above
N~800 and removes the memory wall entirely at large N.

### 3. Multi-RHS pays for itself immediately
At 72 incident directions, naive (72 builds × 72 solves) wastes 71 GPU builds
per seed. The 1-build × 72-solves pattern from radar Stage 7 transfers exactly
to the bistatic acoustic tensor.

### 4. Monte Carlo Welford needs only two extra arrays
The 50-seed roughness study stores exactly two (N_PHI,) float64 arrays
(mean_lin and M2_lin) per group, regardless of seed count. Welford runs on
linear RCS; mean and std are converted to dBm / dB-relative at save time via
`mean_db = 10 log₁₀(mean_lin)` and `std_db = (10/ln10) · std_lin/mean_lin`
(delta method). Same pattern as radar_scattering Stage 4.

### 5. Multiple scattering matters at d < λ
The 3.86 dB median coupling correction confirms that independent-scatterer
models fail at typical sonar operating distances. The full coupled BEM is
the correct tool whenever obstacle separations are sub-wavelength.

---

## File layout

```
acoustic_scattering_v2/
│
├── bem_helmholtz_v2.py          GPU wrapper (imports bem_assembly.so from radar_scattering/)
├── mie_cylinder_2d.py           Acoustic Mie series (wraps radar_scattering/mie_series.py)
├── geometry_v2.py               Original geometry + perturb_panels() roughness
│
├── stage0_comparison.ipynb      v1 vs v2 benchmark — the progress record
├── stage1_validation.ipynb      Mie error table, condition numbers, frequency sweep
├── stage4_roughness_mc.ipynb    Monte Carlo roughness (80 groups, 50 seeds each)
├── stage5_frequency_sweep.ipynb 79 k values × 4 shapes (316 solves, 10 s)
├── stage7_bistatic.ipynb        Full 72×72 bistatic tensor (80 groups, 20 seeds)
│
├── rich_joukowski.ipynb         Aeroacoustic fingerprint (403 solves in 11 s)
├── rich_submarine.ipynb         Sonar signature + hull fouling MC
├── rich_multi_obstacle.ipynb    Coupled BEM: 2/3/5 obstacles, coupling correction
│
├── generate_stage4_data.py      Generator: 80 groups × 50 seeds, N=1024
├── generate_stage7_data.py      Generator: 80 groups × 20 seeds × 72 inc, N=2048
├── stage4_data/                 Precomputed: mean/std/p_detect (4, 4, 5, 90)
└── stage7_data/                 Precomputed: mean/std/p_detect (4, 4, 5, 72, 72)
```

## Performance numbers at a glance

| Task | Solves | Time | Notes |
|---|---|---|---|
| Mie error table (Stage 1) | 20 | 0.1 s | N=128–2048, 4 k values |
| Stage 1 frequency sweep | 63 | 0.1 s | N=512 |
| Stage 4 generator | 400×50=20,000 | ~25 s | N=1024 |
| Stage 5 sweep | 316 | 10 s | N=512 |
| Stage 7 generator | 80×20×72=115,200 | ~12 min | N=2048, multi-RHS |
| Joukowski fingerprint | 403 | 11 s | N=1024 |
| Submarine broadband | 213 | 5.5 s | N=1024 |
| Submarine fouling MC | 120 | 3.4 s | N=1024, 30 seeds |
| Multi-obstacle (5 cylinders) | 1 | 20 ms | N=960 total |
