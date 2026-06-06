# Acoustic Scattering Lab — v4

Interactive 2D Helmholtz BEM solver with a browser-based UI and a 13-experiment Jupyter notebook.
v4 adds **Robin (impedance) boundary conditions** — energy-dissipating surfaces, material presets,
and a ζ slider — alongside two new research notebooks.

---

## Quick start

```bash
cd MPDOK/acoustic_lab
conda run -n py314 python server.py
# Open http://localhost:8766/
```

Run the experiment notebook:

```bash
conda run -n py314 jupyter notebook acoustic_lab_experiments.ipynb
```

---

## What it does

Solves the exterior 2D Helmholtz equation `(∇² + k²)p = 0` for an arbitrary collection of
obstacles using the **single-layer BEM representation**:

```
p_scat(x) = ∫_Γ G(x,y) σ(y) ds(y)
```

where `G(x,y) = (i/4) H₀⁽¹⁾(k|x−y|)` is the free-space Green's function (e^{−iωt}, outgoing).

### Boundary conditions

| BC | Name | Condition | Physics |
|---|---|---|---|
| Soft | Dirichlet | `p = 0` on Γ | Pressure-release (air bubble, open end) |
| Hard | Neumann | `∂p/∂n = 0` on Γ | Rigid wall (concrete, steel, rock) |
| Robin | Impedance | `∂p/∂n + iαp = 0` on Γ | Dissipative surface (foam, carpet, plaster) |

Robin uses `α = k/ζ` where ζ is the **surface impedance ratio** (normalised to ρc).
Absorption coefficient (real ζ): `A = 4ζ / (1+ζ)²`, with maximum A = 100% at ζ = 1.

### Incident field types

| Type | Formula | Description |
|---|---|---|
| Plane wave (default) | `exp(ik x·d̂)` | Infinite-distance source, direction `α` |
| Point / line source | `(i/4) H₀(k\|x−xₛ\|)` | Cylindrical line source at `(xₛ, yₛ)` |

---

## Solver backends

Three backends are auto-selected in priority order:

| Backend | Key | Description |
|---|---|---|
| GPU-BEM Robin | `gpu_bem` | CUDA Fortran H₀+H₁ assembly (`bem_assembly_robin.so`) + CuPy GMRES |
| GPU-BEM Soft | `gpu_bem` | CUDA Fortran H₀ assembly (`bem_assembly.so`) + CuPy GMRES |
| Fortran LU-IR | `fortran` | TF32 LU + FP64 iterative refinement via `LUIRSolver` |
| SciPy | `scipy` | Pure-Python fallback, always available |

### Fortran kernels

**`acoustic_scattering_v4/bem_assembly_robin.cuf`** — CUDA Fortran, compiled to `bem_assembly_robin.so`
- Assembles the Robin BEM matrix `A_robin = A_neumann − iα·A_dirichlet` in a single GPU pass
- Off-diagonal: uses Bessel J₀/Y₀/J₁/Y₁ device intrinsics for both Neumann (H₁) and Dirichlet (H₀) contributions
- Diagonal self-integrals: `Re = 0.5 + α·Δl/4`, `Im = −α·Δl/(2π)(1−γ−ln(kΔl/4))`
- Exports `py_build_robin_c128` (float64) and `py_build_robin_c64` (float32)
- Accepts **real α only**; complex α (dispersive impedance) falls back to CPU

**`radar_scattering/bem_assembly.cuf`** — original Soft-only kernel, still used for Dirichlet path.

**`mpdok_solver.cuf`** (via `mpdok_ops.py`) — TF32 LU + FP64 iterative refinement (`fortran` path).

---

## Architecture

```
acoustic_lab/
├── server.py               WebSocket server (FastAPI + uvicorn), simulation loop
├── acoustic_solver.py      Python solver facade — dispatches to v1/v2/v3/v4 BEM kernels
├── index.html              Single-page browser UI
└── acoustic_lab_experiments.ipynb   13-experiment guide notebook

../acoustic_scattering/     v1 kernel: bem_helmholtz.py, geometry.py
../acoustic_scattering_v2/  v2 kernel: GPU-BEM (CUDA Fortran + CuPy GMRES)
../acoustic_scattering_v3/  v3 kernel: bem_helmholtz_v3.py — Neumann BC
../acoustic_scattering_v4/  v4 kernel: bem_helmholtz_v4.py, bem_assembly_robin.cuf — Robin BC
../radar_scattering/        bem_assembly.cuf source → bem_assembly.so shared lib
```

### Binary frame format (64 bytes, 16 × 4-byte fields)

```
frame(i32) solve_ms(f32) k(f32) alpha(f32) n_shapes(i32)
n_rec(i32) flags(i32) field_power(f32) scat_power(f32)
opt_iter(i32) opt_best(f32) opt_mode(i32) src_flag(i32) src_x(f32) src_y(f32)
impedance_zeta(f32)                                          ← new in v4
```

Followed by: `p_re | p_im | mask | far_field | shape_meta | boundary_pts`

---

## UI features

| Feature | Description |
|---|---|
| BC: Soft / Hard / Robin toggle | Switches boundary condition; Robin enables ζ slider |
| ζ slider (Robin mode) | Impedance ratio — 0.1 (near-Soft) → 10.0 (near-Hard) |
| Material presets | Matched (ζ=1), Foam (ζ=0.5), Carpet (ζ=2), Plaster (ζ=5) |
| Src: Plane / Point toggle | Switches incident field; point source is draggable (yellow ★) |
| Display selector | Re(p) / \|p\|² intensity / Phase (HSV) |
| Panels/λ safety rail | Amber warning when panels/wavelength < recommended threshold |
| Far-field snapshot | 📷 button freezes a reference polar curve (dashed amber overlay) |
| ▶ k / ▶ α sweeps | Continuous wavenumber or angle animation |
| Nelder-Mead optimiser | Minimises field power (dampen) or scatter power (cloak) |
| Presets | 4×4 Crystal, Double Slit, Waveguide, Lloyd's Mirror, … |

---

## Experiments (notebook)

| # | Experiment | Key physics |
|---|---|---|
| 1 | Single cylinder: Soft vs Hard | Fundamental BC difference |
| 2 | 4×4 phononic crystal — band gap | Bragg reflection at k = π/d |
| 3 | 4×4 crystal — trapped cavity mode | Resonance inside lattice |
| 4 | Waveguide channeling | Guided acoustic mode |
| 5 | Double slit interference | Young's fringes |
| 6 | Noise barrier row — optimizer | Soft vs hard energy budget |
| 7 | Joukowski wing — asymmetric stealth | Shape-dependent scattering |
| 8 | Parabolic reflector — focal point | Geometric acoustic focusing |
| 9 | Crystal point defect — localised mode | Impurity state in band gap |
| 10 | Sonic crystal L-waveguide | 90° bend guided by phononic band gap |
| 11 | Schroeder diffuser — hemispherical scatter | QR sequence phase randomisation |
| 12 | Lloyd's Mirror — hyperbolic fringes | Point source + rigid wall interference |
| 13 | Helmholtz resonator ring | Frequency-selective dead zone — see `exp13_helmholtz_resonator.ipynb` |

---

## Robin BC research notebooks (`acoustic_scattering_v4/`)

### `zeta_sweep.ipynb`

Three scattering metrics have three *different* optimal ζ values (k = 6, single cylinder):

| Metric | Optimal ζ |
|--------|-----------|
| Shadow depth min\|p\| | ζ → 0 (maximum absorption) |
| Total scattered power | ζ ≈ 1.6 (absorption + cancellation) |
| Backscattering \|f(π)\| | ζ → ∞ (hard — maximum coherent reflection) |

Key insight: Robin BC **thins the phononic crystal quiet zone** because the bandgap mechanism relies on strong backscattering from each cylinder. Any absorption degrades inter-cylinder coherence.

### `exp13_helmholtz_resonator.ipynb`

Frequency-dispersive impedance: `ζ(k) = r + iQ(k/k_res − k_res/k)`, Q = 5, k_res = 5.

Ring of 6 cylinders (R = 0.35, ring radius = 1.5, 480 panels total):

- **Off resonance**: hard-like → accidental cavity → internal field **elevated** (\|p\|² > 1)
- **At resonance (k = k_res)**: A = 100% → reflections absorbed → **resonant dead zone** (\|p\|² ≪ 1)

Resonance *drains* the ring interior; it does not trap energy. The ring is near-transparent from outside at k_res.

Full theory and results in [`acoustic_scattering_v4/README.md`](../acoustic_scattering_v4/README.md).

---

## Performance (RTX 4060, N = 200 panels, k = 8)

| Backend | BC | Solve time |
|---|---|---|
| GPU-BEM Robin | Robin | ~4 ms |
| GPU-BEM Soft | Soft | ~4 ms |
| CPU assembly + CuPy GMRES | Hard (Neumann) | ~8 ms |
| Fortran LU-IR (TF32 + FP64) | Soft | ~15 ms |
| SciPy (CPU, dense LU) | any | ~35 ms |

GPU-BEM is the default when CuPy is available.

---

## Dependencies

```bash
conda activate py314
pip install fastapi uvicorn cupy-cuda12x scipy numpy matplotlib
```

The Fortran kernels are pre-compiled. To recompile:

```bash
# Original Soft kernel (Dirichlet)
cd MPDOK/radar_scattering
nvfortran -cuda -gpu=cc86 -shared -fPIC -o ../acoustic_scattering_v2/bem_assembly.so bem_assembly.cuf

# v4 Robin kernel
cd MPDOK/acoustic_scattering_v4
nvfortran -cuda -gpu=cc86 -shared -fPIC -o bem_assembly_robin.so bem_assembly_robin.cuf
```

---

## v5 roadmap

- **UI resonator mode:** `ζ(k)` dispersive impedance in the live lab — needs `resonator` BC type, `k_res`/Q sliders, complex α CPU path wired into `server.py`
- **Intensity vectors:** `∇p` at field points via the H₁ Green's function kernel; visualises acoustic energy flow arrows
- **True sub-λ trapping geometry:** design ring radius so cavity eigenfrequency coincides with `k_res`
