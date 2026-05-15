# MPDOK — Terrain Survey: Where Can the Tensor Core Engine Add Value in Iterative Solving?

**Date**: May 2026  
**Context**: Following completion of tensor_core_engine_v5 (cuBLAS-lt epilogue fusion, benchmarked).  

**Question posed**: Can the tensor core engine accelerate iterative solvers in a way that creates value for users who are not GPU experts?

---
Bottom line: faster than SciPy but with more precision, not less:

<img width="2231" height="1313" alt="fig5_summary_dashboard" src="https://github.com/user-attachments/assets/94515d57-39c4-41a5-a8ed-72b6d6b36245" />

---

## The Core Argument

Iterative solvers (CG, GMRES, BiCGSTAB) dominate large-scale scientific computing because direct methods (LU factorisation) become infeasible for large systems. The bottleneck in every iterative solver is the **matrix-vector product** (or preconditioner apply). Accelerating this with tensor cores is the enabling hypothesis.

The question is not whether the tensor core engine *could* help, but where it helps in a way that is:
1. Technically defensible (not already handled by existing libraries)
2. Differentiated (not just slower at something PETSc does better)
3. Useful to a real user who has a real problem

---

## The Competitive Landscape (What Already Exists)

| Library | Strength | What It Handles Well |
|---|---|---|
| PETSc | Gold standard, decades of development | Sparse iterative solvers for FEM/CFD |
| AMGX | NVIDIA's GPU sparse solver | Sparse systems, AMG preconditioners |
| Trilinos | Comprehensive package | FEM, sparse, multi-physics |
| cuSPARSE | GPU sparse primitives | SpMV (sparse matrix-vector product) |
| SciPy | Python-accessible | General Krylov solvers in FP64 |

**The gap**: None of these libraries cleanly expose mixed-precision iterative solving where the precision at each stage (inner solve, Krylov iteration, residual check) is user-controlled and tensor-core-backed.

---

## Evaluation Table: All Use Cases Considered

| Use Case | Matrix Type | Where TC Engine Helps | Competition Strength | TC Engine Differentiation | Verdict |
|---|---|---|---|---|---|
| **Dense-operator Krylov** (BEM, GP regression, integral equations, N-body) | Dense | GEMV at full tensor core throughput; Ozaki for accuracy | Weak — PETSc/AMGX optimised for sparse, not dense operators | Direct and clean: speed + accuracy on the dominant operation | **Stage 1. High value. Underserved.** |
| **Mixed-precision iterative refinement** (GMRES-IR) | Any | FP16 inner factorisation, FP32 Krylov body, FP64 residual — all on tensor cores | Weak at user-facing level; active research area but no packaged library | Precision control across three stages is unique; this is the MPDOK architecture in full | **Stage 2. Highest differentiation.** |
| **Mixed-precision preconditioning** (FP16/TF32 approx. preconditioner in FP64 outer Krylov) | Sparse or dense | FP16/TF32 approximate factorisation as preconditioner, fast on tensor cores | Not packaged anywhere; research papers only | Approximation-as-feature: fast inner, certified outer; extends the engine to sparse problems | **Stage 3. Extends to sparse world.** |
| **Block-ILU / Block-Jacobi** (CFD multi-physics, vector FEM) | Sparse, block-structured | Inversion of dense diagonal blocks (≥ 32×32) | Partial — some GPU codes exist but not mixed-precision | Block size regime matters: CFD with ≥5 variables per cell fits; small blocks (4×4) do not | **Later. Medium value. Domain-specific.** |
| **Domain decomposition subdomain solves** (Additive Schwarz, FETI) | Sparse globally, dense locally | Direct LU on each subdomain (dense subproblem) | PETSc supports but CPU-biased for subdomain solves | Subdomain LU on GPU with tensor cores is a real gap | **Later. Medium value. Specialist audience.** |
| **AMG coarse-level solve** | Small dense | Exact dense LU at coarsest level of multigrid hierarchy | AMG libraries handle this | Benefit real but coarse problem is tiny — not the bottleneck | **Low value. Skip.** |
| **Standard scalar ILU** | Sparse | **None** — sparse triangular solves are memory-bandwidth-bound, not compute-bound | cuSPARSE, AMGX | Tensor cores cannot help; do not claim otherwise | **Do not pursue.** |

---

## The Three High-Value Use Cases Form One Product

The three Stage 1–3 items are not separate products. They are one architectural idea with increasing scope:

```
Stage 1: Dense-operator Krylov
    — TC engine runs GEMV; Python runs Krylov loop; user brings dense A and b
    — Proof of concept: faster convergence AND FP64 accuracy for BEM / GP regression problems

Stage 2: GMRES-IR (Mixed-Precision Iterative Refinement)
    — FP16 LU preconditioner (tensor cores) + FP32 Krylov body + FP64 residual check
    — Precision at each stage is architecturally controlled, not accidental
    — Achieves FP64 accuracy at near-FP16 throughput

Stage 3: Mixed-Precision Preconditioning for Sparse Systems
    — FP16/TF32 approximate factorisation used as preconditioner in FP64 outer solver
    — Extends the mixed-precision idea to sparse problems without competing with SpMV libraries
    — Approximation is a feature: the preconditioner only needs to cluster eigenvalues
```

The single positioning statement: **FP64 accuracy at FP16 throughput, for problems where existing libraries make you choose one or the other.**

---

## The Name

**MPDOK** — Mixed-Precision Dense-Operator Krylov with accuracy guarantees.

The name captures all three stages: mixed precision is the mechanism, dense-operator is the primary domain, Krylov is the solver family, accuracy guarantees is the promise that differentiates this from naive low-precision computation.

---

## What NOT to Build (Decision Record)

- **Standard sparse iterative solvers** — PETSc/AMGX already exist and are mature. Competing is futile.
- **SpMV acceleration** — Sparse matrix-vector product for general CSR matrices is memory-bandwidth-bound; tensor cores do not help and cuSPARSE already handles it.
- **Scalar ILU** — Inherently sparse triangular factorisation; cannot benefit from tensor core tiling.
- **AMG coarse-level solves** — The coarse problem is too small for tensor core throughput to matter.
- **FP8 / MXFP8** — Not available on A1000/RTX 4060 (Blackwell-only hardware). Revisit for future hardware.

---

## References

- `PLAN_V5.md` — v5 engine: cuBLAS-lt epilogue fusion, benchmark results (completed May 2026)
- Benchmark results in `PLAN_V5.md §Completed` — crossover at ~128³, fc1/fc2/fc3 shapes validated
- Prior memory: Ozaki 5×FP32 gives ~1e-7 precision; expm squaring safe only for orthogonal matrices
