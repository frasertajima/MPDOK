"""
Krylov-Lanczos matrix exponential for Hermitian H.

Approximates  |ψ(t)⟩ = exp(-i·H·t)|ψ₀⟩  without forming exp(H).

Algorithm (Saad 1992 / Hochbruck-Lubich):
  1. Build m-step Lanczos basis {v₁, …, vₘ} via H·v products (GPU cuBLAS ZGEMV)
  2. Compress: T_m is m×m real-symmetric tridiagonal
  3. Compute exp(-i·t·T_m)·e₁ in full (m×m, cheap on CPU)
  4. Project back: |ψ(t)⟩ ≈ ‖ψ₀‖ · V_m · (exp(-i·t·T_m)·e₁)

Cost: m dense GEMV (bottleneck) + O(m³) on CPU (negligible).
GPU throughput for N=16384, m=80: ~80 × (2×16384²) complex flops → <1s on RTX 4060.

H may be a CuPy array (GPU path) or a numpy array (CPU/OOC path).
The array module (xp) is detected automatically so the same code runs either way —
this is the hook point for wiring in the MPDOK out-of-core kernel as a drop-in.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import cupy as cp
import numpy as np
import scipy.linalg


@dataclass
class KrylovResult:
    psi_t:      object               # (N,) complex128 — CuPy or numpy depending on H
    error_est:  float               # Saad error bound
    steps:      int                 # actual Lanczos steps taken
    wall_time:  float               # seconds
    gemv_times: list = field(default_factory=list)


def krylov_expm(
    H,               # (N,N) array (CuPy/numpy) OR callable matvec(psi) → H|ψ⟩
    psi0:  cp.ndarray,
    t:     float,
    m:     int   = 80,
    tol:   float = 1e-10,
    reorth: bool = True,
    N:     int   = None,   # required when H is callable
) -> KrylovResult:
    """Approximate exp(-i·H·t)|ψ₀⟩ via m-step Lanczos.

    H    : (N,N) CuPy/numpy Hermitian matrix, OR a callable matvec(psi) → H|ψ⟩.
           Pass N explicitly when H is callable (matrix-free path).
    psi0 : (N,) complex128 state vector
    t    : evolution time
    m    : Krylov dimension
    """
    t0 = time.perf_counter()

    if callable(H):
        matvec = H
        assert N is not None, "Pass N= when H is callable"
        xp = cp.get_array_module(psi0)
    else:
        matvec = lambda v: H @ v
        N  = H.shape[0]
        xp = cp.get_array_module(H)

    m = min(m, N)

    norm0 = float(xp.linalg.norm(psi0))
    # Fortran order: each column V[:,j] is contiguous → V[:,j].T is C-contiguous
    # avoids cuBLAS making a contiguous copy of the slice during reorthogonalisation
    V     = xp.zeros((N, m + 1), dtype=xp.complex128, order='F')
    alpha = np.zeros(m, dtype=np.float64)
    beta  = np.zeros(m + 1, dtype=np.float64)
    V[:, 0] = psi0 / norm0

    gemv_times = []
    j_max = m

    for j in range(m):
        gt = time.perf_counter()
        w  = matvec(V[:, j])       # cuBLAS ZGEMV, numpy GEMV, or matrix-free
        gemv_times.append(time.perf_counter() - gt)

        alpha[j] = float(xp.real(xp.dot(V[:, j].conj(), w)))
        w -= alpha[j] * V[:, j]
        if j > 0:
            w -= beta[j] * V[:, j - 1]

        if reorth and j >= 0:
            # V^H w = conj(V^T conj(w)) — conjugate the N-vector, not the N×j matrix
            projs = (V[:, :j + 1].T @ w.conj()).conj()
            w    -= V[:, :j + 1] @ projs

        beta[j + 1] = float(xp.linalg.norm(w))
        if beta[j + 1] < tol:
            j_max = j + 1
            break
        V[:, j + 1] = w / beta[j + 1]
    else:
        j_max = m

    # ── small tridiagonal expm on CPU (always numpy — it's m×m, trivial) ─────
    T = (np.diag(alpha[:j_max])
         + np.diag(beta[1:j_max], 1)
         + np.diag(beta[1:j_max], -1))

    # exp(-i·t·T)·e₁  via eigendecomposition (stable for symmetric T)
    lam, Q  = np.linalg.eigh(T)
    phase   = np.exp(-1j * t * lam)
    e1_proj = Q[0, :].conj()
    coeff   = Q @ (phase * e1_proj)

    # Error estimate: β_{m+1} |coeff[-1]| (Saad bound)
    error_est = float(beta[j_max]) * abs(coeff[-1])

    # ── project back to full space ────────────────────────────────────────────
    coeff_gpu = xp.array(coeff, dtype=xp.complex128)
    psi_t     = norm0 * (V[:, :j_max] @ coeff_gpu)

    return KrylovResult(
        psi_t     = psi_t,
        error_est = error_est,
        steps     = j_max,
        wall_time = time.perf_counter() - t0,
        gemv_times = gemv_times,
    )


def evolve_trajectory(
    H,                   # array or callable matvec
    psi0:    cp.ndarray,
    times:   np.ndarray,
    m:       int   = 80,
    verbose: bool  = True,
    N:       int   = None,   # required when H is callable
    dt:      float = None,   # if set, use cumulative short-step Krylov restart
) -> list[KrylovResult]:
    """Evolve |ψ₀⟩ to each time in `times`.

    dt=None : single-shot Krylov from psi0 for each t (fast for small t).
    dt=float: cumulative restart — step by dt from the running state, saving
              results at each requested time. Maintains accuracy for large t
              where a single shot would require an impractically large m.
              Accumulated error ≈ ceil(t/dt) × (per-step error) ≈ 10⁻¹³.
    """
    results   = []
    t_wall    = time.perf_counter()

    if dt is None:
        # ── independent single-shot path (original) ───────────────────────────
        for i, t in enumerate(times):
            r = krylov_expm(H, psi0, float(t), m=m, N=N)
            results.append(r)
            if verbose and (i % max(1, len(times) // 10) == 0
                            or i == len(times) - 1):
                print(f'  t={t:.2f}  steps={r.steps}  err={r.error_est:.2e}'
                      f'  ({r.wall_time*1000:.0f} ms)', flush=True)
    else:
        # ── cumulative short-step restart path ────────────────────────────────
        xp        = cp.get_array_module(psi0)
        psi       = psi0.copy()
        t_now     = 0.0
        t_sorted  = np.sort(np.asarray(times, dtype=float))
        total_err = 0.0

        for t_target in t_sorted:
            t_step_wall = time.perf_counter()
            step_count  = 0
            while t_now < t_target - 1e-12:
                step  = min(dt, t_target - t_now)
                r     = krylov_expm(H, psi, step, m=m, N=N)
                psi   = r.psi_t
                psi  /= xp.linalg.norm(psi)   # keep normalised
                t_now     += step
                total_err += r.error_est
                step_count += 1

            # Wrap current state as a KrylovResult for compatibility
            result = KrylovResult(
                psi_t      = psi.copy(),
                error_est  = total_err,
                steps      = r.steps if step_count else 0,
                wall_time  = time.perf_counter() - t_step_wall,
                gemv_times = [],
            )
            results.append(result)
            if verbose:
                print(f'  t={t_target:.2f}  krylov_steps={result.steps}'
                      f'  cum_err={total_err:.2e}'
                      f'  ({result.wall_time*1000:.0f} ms)', flush=True)

    if verbose:
        print(f'  Trajectory done: {len(times)} points in '
              f'{time.perf_counter()-t_wall:.2f}s')
    return results
