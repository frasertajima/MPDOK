# MPDOK v28e — cryo-EM Orientation Classification & 3D Reconstruction

**Dataset**: EMPIAR-10025  T20S Proteasome  
**Tasks**:  
1. Predict orientational class (0–7) from noisy 64×64 particle images using MPDOK KRR  
2. 3D back-projection reconstruction from class averages using MPDOK GMRES-IR

## Background

Cryo-EM reconstructs 3D protein structure from thousands of noisy 2D projection images
photographed at unknown orientations. Assigning each particle image to an orientational
class is normally done by exhaustive template matching — expensive at scale. MPDOK offers
a kernel-based alternative: once trained on labelled examples, orientation prediction
is a single matrix-vector multiply per new particle.

The 8 class averages represent distinct 2D views of the T20S barrel (different tilt and
azimuthal orientations), computed by the prior `cryo_em_3d_reconstruction` project.

## Data pipeline

```
67 motion-corrected micrographs (7676×7420, 15 GB)
    ↓  prep_particles.py  (CC picking + NMS + downsample)
N particles × 64×64  +  class labels 0-7
    ↓  exp1 or exp2
PCA-128 or CNN-32 features  →  MPDOK KRR  →  predicted class
    ↓  exp3
8 class averages × 64×64  →  MPDOK GMRES-IR  →  32³ volume
```

## Experiments

| Exp | Task | Method | Key Result |
|-----|------|--------|-----------|
| 1 | Orientation classification | PCA-128 → MPDOK KRR | 10.7% acc (random=12.5%) |
| 2 | Orientation classification | CNN-32 → MPDOK KRR | — |
| 3 | 3D back-projection | MPDOK GMRES-IR (matrix-free) | **Vol CC 0.9110** (backproj: 0.8055), 2 outer iters, 0.2s |

**Exp 3 highlights**:
- Forward projector: sparse A (32,768×32,768, 0.043% density, 458,752 nnz), build 0.03s
- Matrix-free matvec: `(AᵀA + λI)v` = two sparse CuPy SpMVs, 10.8 ms FP32 / 1.0 ms FP64
- GMRES-IR converged in **2 outer iterations** (0.2s total) with λ=0.01
- All 8 synthetic views re-projected with CC > 0.997 — perfect data consistency
- Real data: low CC confirms angular assignment uncertainty without STAR file


