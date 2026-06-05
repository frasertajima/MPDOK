# Acoustic Scattering Lab — v3

Interactive 2D Helmholtz BEM solver with a browser-based UI and a 12-experiment Jupyter notebook.

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

where `G(x,y) = (i/4) H₀⁽¹⁾(k|x−y|)` is the free-space Green's function.

### Boundary conditions

| BC | Name | Condition | Physics |
|---|---|---|---|
| Soft | Dirichlet | `p = 0` on surface | Pressure-release (foam, water, vegetation) |
| Hard | Neumann | `∂p/∂n = 0` on surface | Rigid wall (concrete, steel, rock) |

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
| GPU-BEM | `gpu_bem` | CUDA Fortran H₀ assembly (`bem_assembly.so`) + CuPy GMRES |
| Fortran LU-IR | `fortran` | TF32 LU + FP64 iterative refinement via `LUIRSolver` |
| SciPy | `scipy` | Pure-Python fallback, always available |

### Fortran kernels (unchanged from radar lab)

**`radar_scattering/bem_assembly.cuf`** — CUDA Fortran, compiled to `bem_assembly.so`
- Assembles the (N×N) complex BEM matrix on GPU using `bessel_j0` / `bessel_y0` device intrinsics
- Diagonal self-integrals computed analytically: `Re = dl/(2π)(1 − γ − ln(k·dl/4))`, `Im = dl/4`
- Dirichlet (Soft) only; Neumann (Hard) matrix stays on CPU (H₁ kernel, no Fortran equivalent)

**`mpdok_solver.cuf`** (via `mpdok_ops.py`) — CUDA Fortran, `LUIRSolver`
- Splits the N×N complex system into a real (2N×2N) block form
- First solve: TF32 tensor-core LU factorisation on GPU
- Refinement: FP64 residual → FP64 correction, iterate to machine precision
- Used for the `fortran` solver path

> **No new Fortran kernels were added in v3.** The Neumann BC (`build_matrix_neumann` in
> `bem_helmholtz_v3.py`) uses `scipy.special.hankel1` for CPU assembly and CuPy GMRES for the solve.

---

## Architecture

```
acoustic_lab/
├── server.py               WebSocket server (FastAPI + uvicorn), simulation loop
├── acoustic_solver.py      Python solver facade — dispatches to v1/v2/v3 BEM kernels
├── index.html              Single-page browser UI
└── acoustic_lab_experiments.ipynb   12-experiment guide notebook

../acoustic_scattering/     v1 kernel: bem_helmholtz.py, geometry.py
../acoustic_scattering_v2/  v2 kernel: GPU-BEM (CUDA Fortran + CuPy GMRES)
../acoustic_scattering_v3/  v3 kernel: bem_helmholtz_v3.py — adds Neumann BC
../radar_scattering/        bem_assembly.cuf source → bem_assembly.so shared lib
```

### Binary frame format (60 bytes, 15 × 4-byte fields)

```
frame(i32) solve_ms(f32) k(f32) alpha(f32) n_shapes(i32)
n_rec(i32) flags(i32) field_power(f32) scat_power(f32)
opt_iter(i32) opt_best(f32) opt_mode(i32) src_flag(i32) src_x(f32) src_y(f32)
```

Followed by: `p_re | p_im | mask | far_field | shape_meta | boundary_pts`

---

## UI features

| Feature | Description |
|---|---|
| BC: Soft / Hard toggle | Switches between Dirichlet and Neumann BC |
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

---

## Performance (RTX 3090, N=200 panels, k=8)

| Backend | Solve time |
|---|---|
| GPU-BEM (CUDA Fortran + CuPy GMRES) | ~4 ms |
| Fortran LU-IR (TF32 + FP64 refinement) | ~12 ms |
| SciPy (CPU, dense LU) | ~35 ms |

GPU-BEM is the default when CuPy is available. For the Neumann (Hard) path the matrix is
assembled on CPU (~20 ms for N=200) and solved with CuPy GMRES.

---

## Dependencies

```
conda activate py314
pip install fastapi uvicorn cupy-cuda12x scipy numpy matplotlib
```

The Fortran kernels (`bem_assembly.so`) are pre-compiled. To recompile from source:

```bash
cd MPDOK/radar_scattering
nvfortran -cuda -gpu=cc86 -shared -fPIC -o ../acoustic_scattering_v2/bem_assembly.so bem_assembly.cuf
```

---

## v4 roadmap

- **Robin BC** (impedance boundary): adds dissipation via `∂p/∂n + iαp = 0`; requires a new combined-layer BEM kernel
- **Intensity vector field**: computes `∇p` at field points using `∂G/∂x` (H₁ kernel); visualises acoustic energy flow arrows
- **Experiment 13 — Helmholtz resonators**: sub-λ trapping requires Robin BC for the neck impedance
