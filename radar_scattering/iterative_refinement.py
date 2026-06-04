"""
iterative_refinement.py — Mixed-precision GMRES iterative refinement for BEM.

Drives MPDOK GMRES solutions from the complex64 floor (~1e-6) to near
double-precision accuracy (~1e-12) using a two-kernel GPU strategy:

    Step 1  build_bem_matrix_gpu      complex64   0.05 s  →  x₀, ‖r‖ ≈ 1e-6
    Step 2  build_bem_matrix_gpu_c128 complex128  0.10 s  →  true residual r₀
    Step 3  MPDOK GMRES on δx         complex64   ~0.3 s  →  δx, ‖r₁‖ ≈ 1e-12
    Step 4  x₁ = x₀ + δx

The complex128 GPU kernel (bem_gpu.build_bem_matrix_gpu_c128) provides the
high-precision residual without the 29-second CPU Hankel build.  IR converges
in 1–2 steps for well-conditioned systems.

Public API
----------
    refine(op_c64, A_c128_gpu, b, x0, tol=1e-10, max_steps=3)
        Apply iterative refinement to an existing GMRES solution.

    solve_bem_ir(nodes, lengths, k, phi_inc, tol_gmres=1e-6,
                 tol_ir=1e-10, ir_steps=2, verbose=False)
        Full pipeline: GPU build (c64) → GMRES → GPU build (c128) → IR.
        Returns (sigma, info) where info contains per-step residuals and timings.

    residual_history(nodes, lengths, k, phi_inc, n_ir_steps=3)
        Returns array of shape (n_ir_steps+1,) with ‖r‖/‖b‖ at each step.
        Useful for convergence plots.
"""

import time
import numpy as np
import cupy as cp
from pathlib import Path
import sys, os

_HERE  = Path(__file__).parent
_MPDOK = _HERE.parent
for p in [str(_MPDOK), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from radar_scattering.bem_gpu import build_bem_matrix_gpu, build_bem_matrix_gpu_c128
from radar_scattering.gmres_complex import (
    ComplexDenseOperator, gmres_complex, diagonal_preconditioner,
)


# ── Core refinement step ───────────────────────────────────────────────────────

def refine(op_c64, A_c128_gpu, b, x0,
           tol=1e-10, max_steps=3, restart=50, verbose=False):
    """Apply iterative refinement to an existing complex64 GMRES solution.

    Args:
        op_c64:      ComplexDenseOperator wrapping the complex64 BEM matrix.
        A_c128_gpu:  (N, N) cupy.complex128 — the same matrix in full precision,
                     used only for computing true residuals.
        b:           (N,) complex128 NumPy/CuPy RHS.
        x0:          (N,) complex64 or complex128 initial solution (e.g. from GMRES).
        tol:         Target relative residual (default 1e-10, well below c64 floor).
        max_steps:   Maximum IR iterations (1–2 typically sufficient).
        restart:     GMRES restart for each correction solve.
        verbose:     Print residual at each step.

    Returns:
        x:       (N,) cupy.complex128 refined solution.
        history: list of (step, rel_residual) — step 0 is the input x0.
    """
    b_c128 = cp.asarray(b, dtype=cp.complex128)
    b_norm = float(cp.linalg.norm(b_c128))

    x = cp.asarray(x0, dtype=cp.complex128)

    # Preconditioner from the c64 operator (reused across steps)
    M_inv = diagonal_preconditioner(op_c64)

    def rel_res_c128(xv):
        r = b_c128 - A_c128_gpu @ xv
        return float(cp.linalg.norm(r)) / b_norm

    history = [(0, rel_res_c128(x))]
    if verbose:
        print(f'  IR step 0 (input):  rel_res = {history[0][1]:.3e}')

    for step in range(1, max_steps + 1):
        # Compute true residual in complex128
        r = b_c128 - A_c128_gpu @ x

        # Solve correction in complex64 GMRES (r is small → fast convergence)
        r_c128_norm = float(cp.linalg.norm(r))
        if r_c128_norm / b_norm < tol:
            break

        dx_c64, _, _ = gmres_complex(
            op_c64,
            cp.asnumpy(r),
            tol=tol * b_norm / r_c128_norm,   # scale tol relative to correction size
            restart=restart,
            M_inv=M_inv,
            maxiter=restart * 4,
        )

        x = x + cp.asarray(dx_c64, dtype=cp.complex128)
        rr = rel_res_c128(x)
        history.append((step, rr))

        if verbose:
            print(f'  IR step {step}:          rel_res = {rr:.3e}')

        if rr < tol:
            break

    return x, history


# ── Full pipeline ──────────────────────────────────────────────────────────────

def solve_bem_ir(nodes, lengths, k, phi_inc,
                 tol_gmres=1e-6, tol_ir=1e-10, ir_steps=2,
                 restart=50, verbose=False):
    """Complete pipeline: GPU c64 build → MPDOK GMRES → GPU c128 build → IR.

    Args:
        nodes:     (N, 2) float64 panel midpoints.
        lengths:   (N,)   float64 panel arc lengths.
        k:         Wavenumber.
        phi_inc:   Incident angle (radians).
        tol_gmres: GMRES tolerance for base solve (default 1e-6).
        tol_ir:    IR target relative residual (default 1e-10).
        ir_steps:  Maximum IR steps (default 2).
        restart:   GMRES restart parameter.
        verbose:   Print timing and convergence.

    Returns:
        sigma:  (N,) complex128 NumPy — surface current density.
        info:   dict with keys:
                  t_build_c64, t_gmres, t_build_c128, t_ir  (seconds)
                  gmres_converged, ir_history  (list of (step, rel_res))
                  rel_res_gmres, rel_res_final
    """
    N = nodes.shape[0]
    d = np.array([np.cos(phi_inc), np.sin(phi_inc)])
    b = -np.exp(1j * k * (nodes @ d)).astype(np.complex128)

    # ── Step 1: complex64 build + GMRES ──────────────────────────────────────
    if verbose:
        print(f'[IR] Step 1: GPU complex64 build  (N={N}, k={k:.0f})')
    t0 = time.perf_counter()
    A_c64 = build_bem_matrix_gpu(nodes, lengths, k)
    cp.cuda.Stream.null.synchronize()
    t_build_c64 = time.perf_counter() - t0

    op    = ComplexDenseOperator(A_c64); del A_c64
    M_inv = diagonal_preconditioner(op)

    if verbose:
        print(f'[IR] Step 2: MPDOK GMRES (tol={tol_gmres:.0e})')
    t0 = time.perf_counter()
    x0_gpu, hist0, conv = gmres_complex(
        op, b, tol=tol_gmres, restart=restart,
        M_inv=M_inv, maxiter=restart * 8, verbose=verbose,
    )
    cp.cuda.Stream.null.synchronize()
    t_gmres = time.perf_counter() - t0
    rel_gmres = hist0[-1][1] if hist0 else float('nan')

    # ── Step 2: complex128 build for accurate residuals ───────────────────────
    if verbose:
        print(f'[IR] Step 3: GPU complex128 build')
    t0 = time.perf_counter()
    A_c128 = build_bem_matrix_gpu_c128(nodes, lengths, k)
    cp.cuda.Stream.null.synchronize()
    t_build_c128 = time.perf_counter() - t0

    # ── Step 3: iterative refinement ─────────────────────────────────────────
    if verbose:
        print(f'[IR] Step 4: iterative refinement (tol={tol_ir:.0e}, max {ir_steps} steps)')
    t0 = time.perf_counter()
    x_refined, ir_history = refine(
        op, A_c128, b, x0_gpu,
        tol=tol_ir, max_steps=ir_steps,
        restart=restart, verbose=verbose,
    )
    t_ir = time.perf_counter() - t0

    # Clean up VRAM
    op.free()
    del A_c128
    cp.get_default_memory_pool().free_all_blocks()

    sigma = cp.asnumpy(x_refined).astype(np.complex128)
    rel_final = ir_history[-1][1]

    info = dict(
        t_build_c64=t_build_c64,
        t_gmres=t_gmres,
        t_build_c128=t_build_c128,
        t_ir=t_ir,
        t_total=t_build_c64 + t_gmres + t_build_c128 + t_ir,
        gmres_converged=conv,
        rel_res_gmres=rel_gmres,
        rel_res_final=rel_final,
        ir_history=ir_history,
    )

    if verbose:
        print(f'\n[IR] Summary:')
        print(f'  GPU c64 build:   {t_build_c64:.3f}s')
        print(f'  MPDOK GMRES:     {t_gmres:.3f}s  rel={rel_gmres:.2e}')
        print(f'  GPU c128 build:  {t_build_c128:.3f}s')
        print(f'  IR ({len(ir_history)-1} steps):  {t_ir:.3f}s  rel={rel_final:.2e}')
        print(f'  Total:           {info["t_total"]:.3f}s')
        improvement = rel_gmres / rel_final
        print(f'  Residual improvement: {improvement:.0e}×')

    return sigma, info


# ── Convergence diagnostic ────────────────────────────────────────────────────

def residual_history(nodes, lengths, k, phi_inc, n_ir_steps=3, restart=50):
    """Return relative residual at each stage for convergence plots.

    Returns:
        steps:    list of labels ['GMRES', 'IR-1', 'IR-2', ...]
        residuals: list of float rel_res values
        timings:  list of cumulative wall-clock seconds
    """
    _, info = solve_bem_ir(
        nodes, lengths, k, phi_inc,
        tol_gmres=1e-6, tol_ir=1e-13,
        ir_steps=n_ir_steps, restart=restart, verbose=False,
    )

    steps     = ['GMRES'] + [f'IR-{s}' for s, _ in info['ir_history'][1:]]
    residuals = [info['rel_res_gmres']] + [r for _, r in info['ir_history'][1:]]
    t_cum     = info['t_build_c64'] + info['t_gmres']
    timings   = [t_cum]
    t_cum    += info['t_build_c128']
    for dt in [info['t_ir'] / max(len(info['ir_history']) - 1, 1)] * (len(steps) - 1):
        t_cum += dt
        timings.append(t_cum)

    return steps, residuals, timings


if __name__ == '__main__':
    import argparse
    _S4 = Path(__file__).parent
    for p in [str(_S4), str(_S4.parent), str(_S4 / 'cobol_rcs')]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from radar_scattering.geometry import stealth_panels, circle_panels

    parser = argparse.ArgumentParser(description='BEM iterative refinement demo')
    parser.add_argument('--N',     type=int,   default=4096)
    parser.add_argument('--k',     type=float, default=8.0)
    parser.add_argument('--shape', choices=['stealth','circle'], default='stealth')
    parser.add_argument('--steps', type=int,   default=2)
    args = parser.parse_args()

    print(f'=== Iterative Refinement Demo  N={args.N}  k={args.k}  shape={args.shape} ===\n')

    if args.shape == 'stealth':
        nodes, _, lengths = stealth_panels(args.N, length=4.0, half_width=0.4)
    else:
        nodes, _, lengths = circle_panels(args.N, R=1.0)

    sigma, info = solve_bem_ir(
        nodes, lengths, args.k, phi_inc=0.0,
        tol_gmres=1e-6, tol_ir=1e-11,
        ir_steps=args.steps, verbose=True,
    )
    print(f'\nFinal sigma norm: {np.linalg.norm(sigma):.6f}')
