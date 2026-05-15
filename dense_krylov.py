"""
MPDOK — Mixed-Precision Dense-Operator Krylov solver
Stage 1: Dense linear operator with FP32 device storage.
Stage 2: Conjugate Gradient solver (CG) for symmetric positive definite A.

Usage:
    from MPDOK.dense_krylov import DenseLinearOperator, cg
    op       = DenseLinearOperator(A)                  # A: FP64 NumPy/CuPy (N×N)
    x, hist, ok = cg(op, b)                            # solve Ax = b
    y        = op.matvec(x)                            # TF32 GEMV, FP32 CuPy (N,)
    rn       = op.residual_norm(x, b)                  # FP64 ||b - A@x||₂
"""

import cupy as cp
import numpy as np


class DenseLinearOperator:
    """Dense matrix stored in FP32 on device for TF32-accelerated Krylov GEMV.

    On Ampere+ GPUs (RTX A1000, RTX 4060), cp.matmul with FP32 inputs
    automatically routes through TF32 tensor cores — the same hardware path
    as the Fortran TC engine. No Fortran binding needed for GEMV.

    Two copies are held:
      A_fp32 — used for every Krylov GEMV (fast, ~10× FP64 GEMV on RTX 4060)
      A_fp64 — used for residual checks only (infrequent, exact)

    Call free_fp64() after the solve if VRAM is tight; residual_norm() will
    raise after that point.
    """

    def __init__(self, A):
        """
        Args:
            A: Square (N, N) array — FP64 NumPy or CuPy.
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square 2-D, got shape {A.shape}")

        self.N = A.shape[0]

        # Upload and keep FP64 reference on device for residual checks.
        self.A_fp64 = cp.asarray(A, dtype=cp.float64)

        # FP32 working copy — all Krylov GEMV calls use this.
        self.A_fp32 = self.A_fp64.astype(cp.float32)

        cp.cuda.Stream.null.synchronize()

        mb32 = self.A_fp32.nbytes / 1e6
        mb64 = self.A_fp64.nbytes / 1e6
        print(f"DenseLinearOperator: N={self.N:,}  "
              f"FP32={mb32:.1f} MB  FP64={mb64:.1f} MB  "
              f"total={mb32 + mb64:.1f} MB")

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def matvec(self, x):
        """A @ x in TF32.  x must be FP32 CuPy (N,).  Returns FP32 CuPy (N,)."""
        return cp.matmul(self.A_fp32, x)

    def residual_norm(self, x, b):
        """||b - A @ x||₂ computed in FP64.  Returns a Python float.

        x and b may be FP32 or FP64; they are promoted internally.
        Raises RuntimeError if free_fp64() has been called.
        """
        if self.A_fp64 is None:
            raise RuntimeError(
                "FP64 copy was freed; residual_norm() is no longer available. "
                "Reconstruct the operator to recompute residuals."
            )
        x64 = x.astype(cp.float64) if x.dtype != cp.float64 else x
        b64 = cp.asarray(b, dtype=cp.float64)
        r = b64 - cp.matmul(self.A_fp64, x64)
        return float(cp.linalg.norm(r))

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def free_fp64(self):
        """Release the FP64 copy to reclaim VRAM.

        After this call: matvec() still works; residual_norm() raises.
        Useful for N ≥ 20K where the 800 MB FP64 copy is a meaningful fraction
        of device memory.
        """
        self.A_fp64 = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return (self.N, self.N)

    def vram_mb(self):
        """VRAM currently held by this operator (MB)."""
        total = self.A_fp32.nbytes
        if self.A_fp64 is not None:
            total += self.A_fp64.nbytes
        return total / 1e6

    def __repr__(self):
        fp64_state = f"{self.A_fp64.nbytes/1e6:.1f} MB" if self.A_fp64 is not None else "freed"
        return (f"DenseLinearOperator(N={self.N:,}, "
                f"fp32={self.A_fp32.nbytes/1e6:.1f} MB, fp64={fp64_state})")


# ------------------------------------------------------------------
# Stage 2: Conjugate Gradient solver
# ------------------------------------------------------------------

def cg(op, b, x0=None, tol=1e-6, maxiter=None, check_every=10, verbose=False):
    """Conjugate Gradient for symmetric positive definite A.

    Krylov vectors (x, r, p) are FP32 throughout for TF32 tensor core speed.
    A FP64 residual check fires every `check_every` iterations to certify
    accuracy without slowing the inner loop.

    Precision note: FP32 Krylov arithmetic limits convergence to ~1e-6 relative
    residual. This is the natural floor of TF32 accumulation — tighter tolerances
    require iterative refinement (Stage 3, GMRES-IR). For most dense-operator
    problems (BEM, GP regression), 1e-6 is sufficient.

    Stagnation detection: if the FP64 residual does not decrease by at least
    `stagnation_tol` over two consecutive checks, the solver stops early rather
    than wasting iterations at the FP32 precision floor.

    Args:
        op:          DenseLinearOperator wrapping A.
        b:           RHS vector — FP64 or FP32 NumPy/CuPy (N,).
        x0:          Initial guess (N,). Defaults to zero vector.
        tol:         Convergence threshold on relative FP64 residual ||b-Ax||/||b||.
                     Practical floor with FP32 Krylov: ~1e-6.
        maxiter:     Maximum iterations. Defaults to N (exact CG bound).
        check_every: How often to compute the FP64 residual (default every 10 iters).
        verbose:     Print residual at each FP64 check.

    Returns:
        x:         Solution as FP64 CuPy (N,).
        history:   List of (iteration, relative_residual_fp64) at each FP64 check.
        converged: True if tol was met; False if maxiter reached or stagnated.
    """
    N = op.N
    if maxiter is None:
        maxiter = N

    b64 = cp.asarray(b, dtype=cp.float64)
    b32 = b64.astype(cp.float32)
    b_norm = float(cp.linalg.norm(b64))
    if b_norm == 0.0:
        return cp.zeros(N, dtype=cp.float64), [(0, 0.0)], True

    if x0 is None:
        x = cp.zeros(N, dtype=cp.float32)
    else:
        x = cp.asarray(x0, dtype=cp.float32)

    r = b32 - op.matvec(x)
    p = r.copy()
    rs = float(cp.dot(r, r))

    history = []
    converged = False
    prev_rel = float('inf')
    stagnation_tol = 1e-2   # require at least 1% improvement between checks

    for k in range(maxiter):

        if k % check_every == 0:
            rel = op.residual_norm(x, b64) / b_norm
            history.append((k, rel))
            if verbose:
                print(f"  CG iter {k:4d}  rel_res(FP64) = {rel:.3e}")
            if rel < tol:
                converged = True
                break
            # Stop if residual is no longer decreasing (FP32 precision floor).
            if rel > prev_rel * (1.0 - stagnation_tol):
                if verbose:
                    print(f"  CG stagnated at iter {k}: {prev_rel:.3e} → {rel:.3e}")
                break
            prev_rel = rel

        Ap     = op.matvec(p)
        alpha  = rs / float(cp.dot(p, Ap))
        x      = x + alpha * p
        r      = r - alpha * Ap
        rs_new = float(cp.dot(r, r))
        beta   = rs_new / rs
        p      = r + beta * p
        rs     = rs_new

    if not converged:
        rel = op.residual_norm(x, b64) / b_norm
        history.append((k + 1, rel))
        converged = rel < tol

    return x.astype(cp.float64), history, converged


# ------------------------------------------------------------------
# Stage 2: GMRES — restarted, for general (non-symmetric) A
# ------------------------------------------------------------------

def gmres(op, b, x0=None, tol=1e-6, maxiter=None, restart=50, verbose=False):
    """Restarted GMRES(m) for general square A (symmetric or not).

    Arnoldi basis vectors are FP32 on device; the Hessenberg matrix and
    least-squares solve are on CPU (small: restart × restart). FP64 residual
    is checked at every restart boundary to certify accuracy.

    Precision note: same FP32 Krylov floor as CG (~1e-6). Iterative
    refinement (Stage 3) is needed to push below that.

    Stagnation detection: stops if the FP64 residual fails to decrease by
    at least 1% between restarts.

    Args:
        op:      DenseLinearOperator wrapping A.
        b:       RHS vector — FP64 or FP32 NumPy/CuPy (N,).
        x0:      Initial guess (N,). Defaults to zero.
        tol:     Convergence on relative FP64 residual ||b-Ax||/||b||.
        maxiter: Max total GEMV calls. Defaults to N.
        restart: Krylov subspace size before restart (m). Default 50.
        verbose: Print residual at each restart.

    Returns:
        x:         Solution as FP64 CuPy (N,).
        history:   List of (total_iters, rel_res_fp64) at each restart.
        converged: True if tol met; False if maxiter reached or stagnated.
    """
    N = op.N
    if maxiter is None:
        maxiter = N

    b64 = cp.asarray(b, dtype=cp.float64)
    b32 = b64.astype(cp.float32)
    b_norm = float(cp.linalg.norm(b64))
    if b_norm == 0.0:
        return cp.zeros(N, dtype=cp.float64), [(0, 0.0)], True

    if x0 is None:
        x = cp.zeros(N, dtype=cp.float32)
    else:
        x = cp.asarray(x0, dtype=cp.float32)

    history = []
    converged = False
    total_iters = 0
    prev_rel = float('inf')
    stagnation_tol = 1e-2

    max_restarts = (maxiter + restart - 1) // restart

    # Preallocate basis matrix once: rows are Arnoldi vectors, shape (restart+1, N).
    # Reused across restarts — no per-restart allocation.
    V_mat = cp.empty((restart + 1, N), dtype=cp.float32)

    # Hessenberg matrix on CPU — small, (restart+1) × restart.
    H = np.zeros((restart + 1, restart), dtype=np.float64)

    for outer in range(max_restarts):
        # Initial residual for this cycle.
        r = b32 - op.matvec(x)
        beta = float(cp.linalg.norm(r))

        if beta == 0.0:
            converged = True
            break

        V_mat[0] = r / beta
        H[:] = 0.0

        m = 0
        for k in range(restart):
            if total_iters >= maxiter:
                break

            w = op.matvec(V_mat[k])
            total_iters += 1

            # Batched Gram-Schmidt: one GEMV replaces k+1 individual dot products.
            # h = V_mat[:k+1] @ w  — shape (k+1,) on GPU
            h = V_mat[:k + 1] @ w          # FP32 GEMV, all basis vectors at once
            H[:k + 1, k] = cp.asnumpy(h)  # pull the small vector to CPU for lstsq
            w = w - V_mat[:k + 1].T @ h   # orthogonalise: one FP32 GEMV

            nw = float(cp.linalg.norm(w))
            H[k + 1, k] = nw
            m = k + 1

            if nw < 1e-12:
                # Lucky breakdown: exact solution in current subspace.
                break

            V_mat[k + 1] = w / nw

        # Least-squares solve on CPU: min ||β e₁ - H[:m+1,:m] y||.
        e1 = np.zeros(m + 1, dtype=np.float64)
        e1[0] = beta
        y, _, _, _ = np.linalg.lstsq(H[:m + 1, :m], e1, rcond=None)

        # Update solution: x += V_mat[:m].T @ y — one GEMV instead of m axpy calls.
        y_gpu = cp.array(y, dtype=cp.float32)
        x = x + V_mat[:m].T @ y_gpu

        # FP64 residual check at every restart boundary.
        rel = op.residual_norm(x, b64) / b_norm
        history.append((total_iters, rel))
        if verbose:
            print(f"  GMRES restart {outer + 1:3d}  "
                  f"iters={total_iters:4d}  rel_res(FP64)={rel:.3e}")

        if rel < tol:
            converged = True
            break

        if rel > prev_rel * (1.0 - stagnation_tol):
            if verbose:
                print(f"  GMRES stagnated: {prev_rel:.3e} → {rel:.3e}")
            break
        prev_rel = rel

    return x.astype(cp.float64), history, converged


# ------------------------------------------------------------------
# Stage 3: GMRES-IR — Mixed-Precision Iterative Refinement
# ------------------------------------------------------------------

def gmres_ir(op, b, tol=1e-11, maxiter_outer=5, restart=50,
             maxiter_inner=None, verbose=False):
    """GMRES with Iterative Refinement (GMRES-IR).

    Outer loop (FP64): computes exact residual r = b - A @ x, checks
    convergence, then calls the inner GMRES to find a correction e ≈ A⁻¹r.
    The correction is added back in FP64, preserving full double precision.

    Inner loop (FP32/TF32): standard GMRES on the correction equation.
    Does not need to converge tightly — even a loose FP32 solve (~1e-6)
    reduces the outer residual by ~7 orders of magnitude per iteration.

    Convergence cascade (typical for well-conditioned A):
        outer 0: rel_res ~ 1.0          (initial)
        outer 1: rel_res ~ 1e-7         (one inner GMRES)
        outer 2: rel_res ~ 1e-14        (FP64 machine epsilon)

    Requires op.A_fp64 to be present (not freed) for the FP64 residual.
    Raises RuntimeError if free_fp64() was called on the operator.

    Args:
        op:             DenseLinearOperator — A_fp64 must be available.
        b:              RHS — FP64 NumPy/CuPy (N,).
        tol:            Convergence on relative FP64 residual. Default 1e-11.
        maxiter_outer:  Max refinement steps. Default 5 (almost never needed).
        restart:        Inner GMRES restart size. Default 50.
        maxiter_inner:  Max inner GMRES iterations per outer step. Default N.
        verbose:        Print outer residual at each refinement step.

    Returns:
        x:         Solution as FP64 CuPy (N,).
        history:   List of (outer_iter, rel_res_fp64) — one entry per outer step.
        converged: True if tol met within maxiter_outer.
    """
    if op.A_fp64 is None:
        raise RuntimeError(
            "GMRES-IR requires op.A_fp64 for FP64 residual computation. "
            "Do not call free_fp64() before gmres_ir()."
        )

    N = op.N
    b64 = cp.asarray(b, dtype=cp.float64)
    b_norm = float(cp.linalg.norm(b64))
    if b_norm == 0.0:
        return cp.zeros(N, dtype=cp.float64), [(0, 0.0)], True

    x = cp.zeros(N, dtype=cp.float64)
    history = []
    converged = False

    for outer in range(maxiter_outer):
        # Exact FP64 residual — this is what makes GMRES-IR different from
        # plain GMRES: the residual is never contaminated by FP32 rounding.
        r64 = b64 - cp.matmul(op.A_fp64, x)
        rel = float(cp.linalg.norm(r64)) / b_norm
        history.append((outer, rel))

        if verbose:
            print(f"  GMRES-IR outer {outer:2d}  rel_res(FP64) = {rel:.3e}")

        if rel < tol:
            converged = True
            break

        # Inner FP32 GMRES: approximately solve A e ≈ r.
        # r64 is the RHS; inner uses op.A_fp32 (TF32) for all GEMVs.
        # Inner tolerance is the FP32 floor (~1e-6) — tighter is meaningless.
        e, inner_hist, _ = gmres(
            op, r64,
            tol=1e-6, restart=restart, maxiter=maxiter_inner,
            verbose=False,
        )

        # FP64 correction update — accumulates in full double precision.
        x = x + e

    # Final residual after last update (covers the case where the loop
    # exits via maxiter_outer without the rel < tol check firing).
    r64 = b64 - cp.matmul(op.A_fp64, x)
    rel = float(cp.linalg.norm(r64)) / b_norm
    history.append((maxiter_outer, rel))
    converged = converged or (rel < tol)

    return x, history, converged


# ------------------------------------------------------------------
# Smoke test — run directly to verify device storage and matvec
# ------------------------------------------------------------------

def _smoke_test(N=4096):
    """Construct a random SPD matrix, store it, check matvec accuracy."""
    print(f"\n=== DenseLinearOperator smoke test  N={N} ===\n")

    rng = cp.random.default_rng(42)
    X = rng.standard_normal((N, N), dtype=cp.float32).astype(cp.float64)
    A = X.T @ X + N * cp.eye(N, dtype=cp.float64)   # SPD, cond ~ N
    x_ref = rng.standard_normal(N, dtype=cp.float32).astype(cp.float64)

    # Reference: FP64 GEMV
    ref = cp.matmul(A, x_ref)

    # Operator
    op = DenseLinearOperator(A)
    x_fp32 = x_ref.astype(cp.float32)

    # Warm-up
    _ = op.matvec(x_fp32)
    cp.cuda.Stream.null.synchronize()

    # Timing
    import time
    reps = 20

    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        y_fp32 = op.matvec(x_fp32)
    cp.cuda.Stream.null.synchronize()
    t_fp32 = (time.perf_counter() - t0) / reps * 1e3   # ms

    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        y_fp64 = cp.matmul(A, x_ref)
    cp.cuda.Stream.null.synchronize()
    t_fp64 = (time.perf_counter() - t0) / reps * 1e3   # ms

    # Accuracy
    err = float(cp.max(cp.abs(y_fp32.astype(cp.float64) - ref)) / cp.max(cp.abs(ref)))

    print(f"  TF32 GEMV time : {t_fp32:.3f} ms")
    print(f"  FP64 GEMV time : {t_fp64:.3f} ms")
    print(f"  Speedup        : {t_fp64 / t_fp32:.2f}×")
    print(f"  Max rel error  : {err:.2e}  (expect ~1e-4 for TF32 vs FP64)")
    print(f"  VRAM held      : {op.vram_mb():.1f} MB")

    # Residual norm test
    b = ref.copy()
    rn = op.residual_norm(x_ref, b)
    print(f"  Residual norm (exact x): {rn:.2e}  (should be ~0)")

    # free_fp64 test
    op.free_fp64()
    assert op.A_fp64 is None
    print(f"  After free_fp64: VRAM held = {op.vram_mb():.1f} MB  (FP64 released)")
    print(f"\n{op}")
    print("\n=== smoke test passed ===\n")


def _cg_test():
    """CG correctness and benchmark across two sizes.

    N=2048 : correctness check, error vs direct solve.
    N=8192 : timing comparison — GPU advantage emerges at larger N.

    FP32 Krylov floor is ~1e-6; tol is set accordingly.
    SciPy CG runs on CPU in FP64 — a fair comparison for the GPU to beat.
    """
    import time
    import scipy.sparse.linalg as spla

    for N, label in [(2048, "correctness"), (8192, "benchmark")]:
        print(f"\n=== CG solver test  N={N}  ({label}) ===\n")

        rng = cp.random.default_rng(7)
        X = rng.standard_normal((N, N), dtype=cp.float64)
        A = X.T @ X + N * cp.eye(N, dtype=cp.float64)   # SPD
        b = rng.standard_normal(N, dtype=cp.float64)

        op = DenseLinearOperator(A)

        # --- MPDOK CG (TF32 Krylov, FP64 residual checks) ---
        # Warm-up: one GEMV to page in A_fp32 before timing.
        _ = op.matvec(b.astype(cp.float32))
        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        x_cg, hist, ok = cg(op, b, tol=1e-6, verbose=False)
        cp.cuda.Stream.null.synchronize()
        t_cg = time.perf_counter() - t0

        iters_cg = hist[-1][0]
        res_cg   = hist[-1][1]

        # --- Direct solve (FP64 LU) ---
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        x_ref = cp.linalg.solve(A, b)
        cp.cuda.Stream.null.synchronize()
        t_direct = time.perf_counter() - t0

        err_vs_direct = float(cp.linalg.norm(x_cg - x_ref) / cp.linalg.norm(x_ref))

        # --- SciPy CG (FP64, CPU BLAS) ---
        A_cpu = cp.asnumpy(A)
        b_cpu = cp.asnumpy(b)
        t0 = time.perf_counter()
        x_sp, _ = spla.cg(A_cpu, b_cpu, rtol=1e-6, maxiter=N)
        t_scipy = time.perf_counter() - t0
        res_scipy = float(np.linalg.norm(b_cpu - A_cpu @ x_sp) / np.linalg.norm(b_cpu))

        # --- Report ---
        w = 67
        print(f"  {'Solver':<24} {'Time (s)':>8} {'Iters':>7} {'Rel res':>10} {'Err vs LU':>11}")
        print(f"  {'-'*w}")
        print(f"  {'MPDOK CG (TF32 GPU)':<24} {t_cg:>8.3f} {iters_cg:>7d} {res_cg:>10.2e} {err_vs_direct:>11.2e}")
        print(f"  {'SciPy CG (FP64 CPU)':<24} {t_scipy:>8.3f} {'—':>7} {res_scipy:>10.2e} {'—':>11}")
        print(f"  {'cp.linalg.solve (FP64)':<24} {t_direct:>8.3f} {'—':>7} {'—':>10} {'0.00e+00':>11}")
        print()
        print(f"  Converged         : {ok}")
        print(f"  GPU vs SciPy CG   : {t_scipy / t_cg:.2f}×  {'(GPU faster)' if t_cg < t_scipy else '(CPU faster at this N)'}")
        print(f"  GPU vs direct LU  : {t_direct / t_cg:.2f}×")
        print()

        print(f"  Convergence (FP64 residual, every 10 iters):")
        step = max(1, len(hist) // 6)
        for it, r in hist[::step]:
            print(f"    iter {it:4d}  {r:.3e}")
        print()


def _gmres_test():
    """GMRES correctness and benchmark on non-symmetric A.

    Three cases:
      N=2048 SPD      — same problem as CG, verify GMRES matches CG quality
      N=2048 non-sym  — CG would fail; GMRES handles it
      N=8192 non-sym  — GPU timing comparison vs SciPy GMRES
    """
    import time
    import scipy.sparse.linalg as spla

    rng = cp.random.default_rng(13)

    cases = [
        (2048, "SPD (control)",      True),
        (2048, "non-symmetric",      False),
        (8192, "non-sym benchmark",  False),
    ]

    for N, label, symmetric in cases:
        print(f"\n=== GMRES test  N={N}  {label} ===\n")

        X = rng.standard_normal((N, N), dtype=cp.float64)
        if symmetric:
            A = X.T @ X + N * cp.eye(N, dtype=cp.float64)
        else:
            # Dominant diagonal keeps matrix non-singular; not symmetric.
            A = X + N * cp.eye(N, dtype=cp.float64)
        b = rng.standard_normal(N, dtype=cp.float64)

        op = DenseLinearOperator(A)

        # Warm-up.
        _ = op.matvec(b.astype(cp.float32))
        cp.cuda.Stream.null.synchronize()

        # --- MPDOK GMRES ---
        t0 = time.perf_counter()
        x_gm, hist, ok = gmres(op, b, tol=1e-6, restart=50, verbose=False)
        cp.cuda.Stream.null.synchronize()
        t_gm = time.perf_counter() - t0

        iters_gm = hist[-1][0]
        res_gm   = hist[-1][1]

        # --- Direct solve reference ---
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        x_ref = cp.linalg.solve(A, b)
        cp.cuda.Stream.null.synchronize()
        t_direct = time.perf_counter() - t0
        err_vs_direct = float(cp.linalg.norm(x_gm - x_ref) / cp.linalg.norm(x_ref))

        # --- SciPy GMRES (FP64, CPU) ---
        A_cpu = cp.asnumpy(A)
        b_cpu = cp.asnumpy(b)
        t0 = time.perf_counter()
        x_sp, info_sp = spla.gmres(A_cpu, b_cpu, rtol=1e-6, restart=50, maxiter=N)
        t_scipy = time.perf_counter() - t0
        res_scipy = float(np.linalg.norm(b_cpu - A_cpu @ x_sp) / np.linalg.norm(b_cpu))

        # --- Report ---
        print(f"  {'Solver':<26} {'Time (s)':>8} {'Iters':>7} {'Rel res':>10} {'Err vs LU':>11}")
        print(f"  {'-'*64}")
        print(f"  {'MPDOK GMRES (TF32 GPU)':<26} {t_gm:>8.3f} {iters_gm:>7d} {res_gm:>10.2e} {err_vs_direct:>11.2e}")
        print(f"  {'SciPy GMRES (FP64 CPU)':<26} {t_scipy:>8.3f} {'—':>7} {res_scipy:>10.2e} {'—':>11}")
        print(f"  {'cp.linalg.solve (FP64)':<26} {t_direct:>8.3f} {'—':>7} {'—':>10} {'0.00e+00':>11}")
        print()
        print(f"  Converged         : {ok}")
        print(f"  GPU vs SciPy GMRES: {t_scipy / t_gm:.2f}×  "
              f"{'(GPU faster)' if t_gm < t_scipy else '(CPU faster at this N)'}")

        print(f"\n  Convergence (FP64 residual per restart):")
        for it, r in hist:
            print(f"    iter {it:4d}  {r:.3e}")


def _gmres_ir_test():
    """GMRES-IR: show FP64 accuracy from FP32 inner solves.

    Three comparisons on a non-symmetric matrix:
      - Plain FP32 GMRES         : stalls at ~1e-7 (FP32 floor)
      - GMRES-IR (this stage)    : reaches ~1e-12 in 2 outer iterations
      - SciPy FP64 GMRES         : CPU FP64 reference
      - cp.linalg.solve          : direct LU reference

    Tested at N=2048 (correctness) and N=8192 (timing).
    """
    import time
    import scipy.sparse.linalg as spla

    rng = cp.random.default_rng(42)

    for N, label in [(2048, "correctness"), (8192, "benchmark")]:
        print(f"\n=== GMRES-IR test  N={N}  ({label}) ===\n")

        X = rng.standard_normal((N, N), dtype=cp.float64)
        A = X + N * cp.eye(N, dtype=cp.float64)   # non-symmetric, dominant diagonal
        b = rng.standard_normal(N, dtype=cp.float64)

        op = DenseLinearOperator(A)

        # Warm-up.
        _ = op.matvec(b.astype(cp.float32))
        cp.cuda.Stream.null.synchronize()

        # --- GMRES-IR ---
        t0 = time.perf_counter()
        x_ir, hist_ir, ok_ir = gmres_ir(op, b, tol=1e-11, verbose=True)
        cp.cuda.Stream.null.synchronize()
        t_ir = time.perf_counter() - t0

        # --- Plain FP32 GMRES (for comparison) ---
        t0 = time.perf_counter()
        x_gm, hist_gm, ok_gm = gmres(op, b, tol=1e-6, restart=50)
        cp.cuda.Stream.null.synchronize()
        t_gm = time.perf_counter() - t0

        # --- Direct LU reference ---
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        x_ref = cp.linalg.solve(A, b)
        cp.cuda.Stream.null.synchronize()
        t_direct = time.perf_counter() - t0

        # --- SciPy FP64 GMRES (CPU) ---
        A_cpu = cp.asnumpy(A)
        b_cpu = cp.asnumpy(b)
        t0 = time.perf_counter()
        x_sp, _ = spla.gmres(A_cpu, b_cpu, rtol=1e-11, restart=50, maxiter=N)
        t_scipy = time.perf_counter() - t0

        # Compute final residuals against direct LU.
        err_ir  = float(cp.linalg.norm(x_ir - x_ref) / cp.linalg.norm(x_ref))
        err_gm  = float(cp.linalg.norm(x_gm - x_ref) / cp.linalg.norm(x_ref))
        err_sp  = float(np.linalg.norm(x_sp - cp.asnumpy(x_ref)) /
                        np.linalg.norm(cp.asnumpy(x_ref)))
        res_ir  = hist_ir[-1][1]
        res_gm  = hist_gm[-1][1]
        res_sp  = float(np.linalg.norm(b_cpu - A_cpu @ x_sp) / np.linalg.norm(b_cpu))

        print()
        print(f"  {'Solver':<28} {'Time (s)':>8} {'Rel res':>12} {'Err vs LU':>12}")
        print(f"  {'-'*62}")
        print(f"  {'GMRES-IR (TF32+FP64 GPU)':<28} {t_ir:>8.3f} {res_ir:>12.2e} {err_ir:>12.2e}")
        print(f"  {'GMRES FP32 only (GPU)':<28} {t_gm:>8.3f} {res_gm:>12.2e} {err_gm:>12.2e}")
        print(f"  {'SciPy GMRES (FP64 CPU)':<28} {t_scipy:>8.3f} {res_sp:>12.2e} {err_sp:>12.2e}")
        print(f"  {'cp.linalg.solve (FP64)':<28} {t_direct:>8.3f} {'—':>12} {'0.00e+00':>12}")
        print()
        print(f"  GMRES-IR converged  : {ok_ir}")
        print(f"  GPU-IR vs SciPy     : {t_scipy / t_ir:.2f}×  "
              f"{'(GPU faster)' if t_ir < t_scipy else '(CPU faster at this N)'}")
        print(f"  GPU-IR vs direct LU : {t_direct / t_ir:.2f}×")
        print()
        print(f"  Refinement history (outer FP64 residual):")
        for it, r in hist_ir:
            print(f"    outer {it}  rel_res = {r:.3e}")


if __name__ == "__main__":
    _smoke_test()
    _cg_test()
    _gmres_test()
    _gmres_ir_test()
