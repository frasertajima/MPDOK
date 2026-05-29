MNIST v8 vs v9 (with batched_matmul_bias_relu:

https://github.com/user-attachments/assets/3dcc3e77-eaa6-4e19-9e79-60a9aa1d9658


# MPDOK · Deep Learning Mathematical Foundations

**Neural Tangent Kernel & Hessian Eigenspectrum on MNIST**

Demonstrates MPDOK on two problems at the mathematical heart of deep learning —
both are dense, computationally impossible with standard tools, and exactly the
kind of work MPDOK was built for.

---

## The Problem

To understand *why* a neural network optimises the way it does, researchers analyse:

| Object | Size for a 535k-param MLP | Standard approach |
|--------|--------------------------|-------------------|
| **Hessian** ∇²L | 535k × 535k = **2.3 TB** | Impossible to store |
| **NTK** for N=15k samples | 15k × 15k = **1.8 GB** | SciPy: 13 s |
| **NTK** for N=30k samples | 30k × 30k = **7.2 GB** | SciPy: 64 s |

MPDOK solves both without ever forming the full matrix.

---

## What the Notebook Does (`ntk_demo.ipynb`)

### §1–2 — Setup & Model
Trains a 784→512→256→10 MLP on MNIST to 97.9 % accuracy in ~2 s.
Weights are cached so subsequent runs are instant.

### §3 — Loss Landscape
Slices the loss surface along a random direction in parameter space,
visualising the curvature that the Hessian encodes.

### §4 — Hessian Eigenspectrum via GPU Lanczos
The Hessian is 2.3 TB — it is never formed.
MPDOK's **GPU Lanczos** uses only Hessian-vector products (HVPs) as an oracle:

```
H·v  =  ∇(∇L · v)     computed by two backward passes through the network
```

The Krylov basis lives in VRAM; reorthogonalisation uses cuBLAS tensor-core GEMV.

| Method | Time | Speedup |
|--------|------|---------|
| SciPy `eigsh` (CPU HVPs) | ~15 s | 1× |
| **MPDOK GPU Lanczos** | **~1 s** | **14×** |

### §5 — NTK Kernel Regression
The **Neural Tangent Kernel** describes how a network evolves during training.
We use the feature kernel (penultimate-layer activations):

```
K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ) / D
```

Solving **(K + λI)α = Y** is the bottleneck — MPDOK uses two paths:

**In-VRAM LU-IR** (N ≤ ~18k):
K is built explicitly in VRAM and factored with TF32 LU (tensor cores) + FP64
iterative refinement.

| N | K size | SciPy | MPDOK LU-IR | Speedup |
|---|--------|-------|-------------|---------|
| 10,000 | 800 MB | 5.2 s | 0.73 s | **7×** |
| 15,000 | 1.8 GB | 13.7 s | 1.7 s | **8×** |
| 18,000 | 2.6 GB | 19.7 s | 2.4 s | **8×** |

**OOC matrix-free** (N > ~18k, automatic):
The N×N matrix is **never formed**. Every matrix-vector product uses the factored identity:

```
K·v  =  Φ(Φᵀv) + λv       O(ND) per matvec, D = 256
```

Φ (N×256 = 25 MB at N=30k) lives in VRAM. GMRES-IR converges in 2–3 outer iterations.

| N | K *would* be | SciPy | MPDOK OOC | Speedup |
|---|-------------|-------|-----------|---------|
| 25,000 | 5.0 GB | 42.6 s | 2.4 s | **18×** |
| 30,000 | 7.2 GB | 64.2 s | 2.3 s | **28×** |
| 50,000 | 20 GB | 238 s | 3.5 s | **68×** |

### §6 — Scaling Benchmark
Sweeps both backends across N = 2k → 18k, plots timing curves and speedup bars.

### §7 — Results Summary
Full tables for both problems; explains the three MPDOK mechanisms.

---

## Files

| File | Role |
|------|------|
| `ntk_demo.ipynb` | 20-cell demo notebook |
| `models.py` | MnistMLP (535k params), data loading, training |
| `hvp.py` | Hessian-vector products via double backprop |
| `lanczos.py` | GPU Lanczos + SciPy CPU baseline |
| `ntk_builder.py` | Cosine feature kernel (GPU chunks + CPU path) |
| `ntk_solver.py` | `NTKSolver` — auto-switches LU-IR ↔ OOC |
| `ntk_ooc.py` | `NTKOOCSolver` — matrix-free GMRES-IR, VRAM/RAM/SSD |
| `benchmark.py` | Full scaling sweep + figure generation |
| `weights/` | Cached MLP weights (97.9 % val accuracy) |

---

## Quick Start

```bash
# From the MPDOK directory
conda run -n py314 jupyter lab ntk_hessian/ntk_demo.ipynb

# Or run the full benchmark headless
conda run -n py314 python ntk_hessian/benchmark.py
```

---

## Key Implementation Notes

- **Cosine kernel required**: L2-normalise features before building K — raw
  features give cond ~10⁷ (LU-IR diverges); normalised gives cond ~N/10.
- **nugget = 1e-2**: larger than typical KRR regularisation, needed for
  LU-IR convergence at this conditioning.
- **OOC threshold N ≈ 18k** (RTX 4060, 8 GB): K (N²×8) + LU buffer (N²×4)
  = N²×12 bytes; beyond this the solver switches automatically to matrix-free.
- **LU-IR VRAM leak fixed**: `LUIRSolver` now allocates FP32 buffers via CuPy
  (proper `cudaFree` on delete) rather than Fortran device allocatables.
- **SciPy does not OOM here**: 46 GB system RAM holds K up to N ≈ 70k —
  the comparison is speed, not memory limits.
