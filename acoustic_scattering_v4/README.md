# acoustic_scattering_v4 — Robin (Impedance) BC

Extends the BEM solver with a **Robin boundary condition** that models energy-dissipating surfaces. A Dirichlet surface absorbs no energy (soft); a Neumann surface reflects everything (hard). Robin gives every point in between — and frequency-selective behaviour when the impedance is made dispersive.

---

## Theory

### Boundary condition

```
∂p/∂n + iα p = 0    on Γ
```

where `α = k/ζ` and `ζ` is the (possibly complex) **surface impedance ratio** (normalised to ρc).

| ζ | α | Behaviour |
|---|---|-----------|
| → ∞ | 0 | Hard (Neumann) limit |
| 1 | k | Matched — 100% absorption |
| → 0 | → ∞ | Soft (Dirichlet) limit |

Absorption coefficient (real ζ): `A = 4ζ / (1+ζ)²`

### BEM formulation

Starting from the single-layer representation `p = ∫ G σ ds`, applying the Robin BC on the boundary trace, and using the standard jump relations gives:

```
A_robin σ = b_robin

A_robin  =  A_neumann  −  iα · A_dirichlet
b_robin  =  b_neumann  +  iα · p_inc|_Γ
```

`A_neumann` and `A_dirichlet` are the Neumann and Dirichlet BEM matrices already assembled in v3/v2. The Robin matrix is a **linear combination** — no new integral kernels are needed for CPU assembly.

### Sign convention

The codebase uses the **e^{−iωt} / outgoing-wave** convention throughout:

```
G(x,y) = (i/4) H₀⁽¹⁾(k|x−y|)
```

H₀⁽¹⁾ → e^{ikr}/√r for large r (outgoing). The Robin BC `∂p/∂n + iα p = 0` and the combined matrix `A_n − iα·A_d` are both consistent with this convention. `alpha = k / zeta` is the correct definition; no sign inversion is needed.

---

## Files

| File | Description |
|------|-------------|
| `bem_helmholtz_v4.py` | Robin BEM: `build_matrix_robin`, `make_rhs_robin`, `solve_robin` |
| `bem_assembly_robin.cuf` | CUDA Fortran kernel — assembles H₀+H₁ in a single GPU pass |
| `bem_assembly_robin.so` | Compiled shared library (cc86, RTX 4060) |
| `bem_gpu_robin.py` | ctypes wrapper for the GPU kernel |
| `mie_cylinder_impedance.py` | Exact Mie series for an impedance cylinder (validation reference) |
| `validate_robin.ipynb` | 6-section validation notebook — 11 tests, all pass |
| `zeta_sweep.ipynb` | Three metrics / three optima experiment |
| `exp13_helmholtz_resonator.ipynb` | Exp 13 — frequency-dispersive Helmholtz resonator array |

---

## GPU kernel — `bem_assembly_robin.cuf`

Computes the Robin BEM matrix in a single kernel launch, avoiding the two-pass CPU approach.

**Off-diagonal elements** (row ≠ col):
```
Re = (−k Y₁(kr) cosθ + α J₀(kr)) / 4 · Δl
Im = ( k J₁(kr) cosθ + α Y₀(kr)) / 4 · Δl
```
where `cosθ = (r⃗ij · n̂i) / r`, using NVHPC device intrinsics `bessel_j0/y0/j1/y1`.

**Diagonal elements** (self-panel, kΔl → 0 limit):
```
Re = 0.5 + α·Δl/4
Im = −α·Δl/(2π) · (1 − γ − ln(k·Δl/4))
```

Exports `py_build_robin_c128` (float64) and `py_build_robin_c64` (float32).

> **Limitation:** the GPU kernel accepts only **real** α. For complex α (frequency-dispersive
> impedance) the CPU fallback `A_n − iα·A_d` is used automatically.

**Recompile:**
```bash
cd acoustic_scattering_v4
nvfortran -cuda -gpu=cc86 -shared -fPIC -o bem_assembly_robin.so bem_assembly_robin.cuf
```

**Speedup** (RTX 4060, N panels, CPU baseline = scipy dense LU):

| N | Speedup |
|---|---------|
| 100 | 18× |
| 400 | 45× |
| 800 | 97× |

---

## Validation — `validate_robin.ipynb`

Compares BEM solution against the exact Mie series for a unit-radius impedance cylinder.

**Key results (k = 4, N = 300 panels):**

| Test | Error |
|------|-------|
| Hard limit (ζ → ∞) | 0.13 dB |
| ζ = 2 | 0.09 dB |
| ζ = 1 (matched) | 0.05 % linear |
| ζ = 0.5 | 0.04 dB |
| Convergence ζ = 2, N = 300 | < 0.5 dB ✓ |

**Note on k = 5:** avoided in primary tests — the Dirichlet irregular frequency for R = 1 lies at j₂,₁ = 5.136, inflating BEM errors artificially. Tests use k = 4 (safely between j₁,₁ = 3.832 and j₂,₁ = 5.136).

**Note on ζ = 1 dB errors:** when far-field amplitude nearly vanishes (matched impedance, maximum absorption), the dB comparison amplifies rounding noise. `_smart_err()` falls back to linear relative error (%) when signal is below 1% of peak — gives the correct 0.05%.

---

## Experiment: ζ sweep — `zeta_sweep.ipynb`

Motivated by the observation that Robin BC thins the phononic crystal quiet zone.

**Three metrics, three different optima** (k = 6, ka = 6, single cylinder):

| Metric | What it measures | Optimal ζ | Reason |
|--------|-----------------|-----------|--------|
| Geometric shadow depth | min\|p\| behind cylinder | ζ → 0 (absorptive) | Absorption drains field everywhere |
| Total scattered power | ∫\|f(φ)\|² dφ | ζ ≈ 1.6 | Absorption + edge cancellation balance |
| Backscattering \|f(π)\| | Crystal bandgap driver | ζ → ∞ (hard) | Maximum coherent reflection |

**Why Robin thins the crystal quiet zone:** the bandgap mechanism requires strong **backscattering** from each cylinder. Robin absorption reduces backscatter amplitude, degrading inter-cylinder coherence. More absorptive material = weaker bandgap. Shadow and bandgap are opposite effects.

The `(k, ζ)` heatmap (Fig 4) shows total-scatter minimum drifts slightly upward with k.

---

## Experiment 13 — `exp13_helmholtz_resonator.ipynb`

A Helmholtz resonator has a frequency-selective surface impedance (driven-oscillator model):

```
ζ(k) = r + iQ (k/k_res − k_res/k)
```

Parameters used: `k_res = 5.0`, `r = 1.0`, `Q = 5.0`

| k vs k_res | \|ζ(k)\| | Behaviour |
|-----------|---------|-----------|
| k ≪ k_res | ~5–11 | Hard-like (capacitive reactive) |
| k = k_res | 1.0 | Matched, A = 100% |
| k ≫ k_res | ~5–11 | Hard-like (inductive reactive) |

**Key finding — the dead zone:** the ring array (6 cylinders, R = 0.35, ring radius = 1.5) produces the *opposite* of what "trapping" implies:

- **Off resonance** (hard-like): inter-cylinder reflections set up internal standing waves → field inside ring is **elevated** (⟨|p|²⟩ > 1, accidental cavity)
- **At resonance** (A = 100%): all internal reflections absorbed → standing waves collapse → **resonant dead zone** (⟨|p|²⟩ ≪ 1)

Simultaneously at resonance: backscattering drops to near zero and total scattered power reaches its minimum — the ring is acoustically near-transparent from the outside.

True sub-λ trapping (energy concentration) would require the ring cavity resonance to coincide with k_res — a separate geometric design problem.

**Complex α note:** `solve_resonator` always uses the CPU path (`A_n − iα·A_d` with complex α). The GPU kernel is bypassed because it only handles real α. For the typical 480-panel ring (6 × 80), CPU solve time is ~0.5 s per k value.

---

## Dependencies

```bash
conda activate py314
# All dependencies inherited from v1–v3:
# scipy, numpy, matplotlib, cupy-cuda12x, nvfortran (for kernel recompile)
```

This module is consumed by `acoustic_lab/acoustic_solver.py` via:

```python
from bem_helmholtz_v4 import solve_robin, HAS_ROBIN_GPU
```

---

## Future work

- **UI resonator mode:** wire `ζ(k)` dispersive model into the interactive lab — requires a `resonator` BC type in `server.py`, per-frame complex α computation, and `k_res`/Q sliders in `index.html`
- **Intensity vectors:** `∇p` at field points via the H₁ Green's function kernel (acoustic energy flow arrows)
- **True sub-λ trapping geometry:** design ring radius so cavity eigenfrequency coincides with `k_res`
