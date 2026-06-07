"""
Genomic Best Linear Unbiased Prediction (G-BLUP) solver.

G-BLUP solves the mixed model equation:
    (G + λI) α = y
where G is the Genomic Relationship Matrix (N×N, dense SPD), λ = σ²_e / σ²_g
is the noise-to-signal ratio, and α are the dual coefficients (BLUP solutions).

Predictions on new individuals with cross-kernel G_new:
    ŷ_new = G_new @ α

This is structurally identical to Kernel Ridge Regression — the quantum kernel
regression framework from quantum_ml/ applies directly here, with G playing the
role of the kernel matrix.

Backends:
  - 'mpdok'       : GPU TF32 LU factorisation + float64 iterative refinement.
                    Fastest for N ≤ ~14,000 on an 8 GB GPU.
  - 'mpdok_ooc_z' : Z-based OOC GMRES-IR (preferred large-N backend).
                    Stores X_raw (SNP dosages, FP32) on GPU; computes G@v on-the-fly
                    as Z@(Z.T@v)/scale = X@(X.T@v - 2p*sum(v))/scale - 2*(p@Ztv).
                    All matvecs are GPU HBM bandwidth-bound (~272 GB/s); no PCIe.
                    Requires X_raw, p, scale kwargs. Auto-selected on VRAM overflow
                    when these are provided via gblup_solve().
  - 'mpdok_ooc'   : Tile-stream OOC GMRES-IR (fallback when X not available).
                    Stores (G+λI) as FP32 in RAM; streams tiles via PCIe (~16 GB/s).
                    PCIe-bandwidth-bound — slower than numpy at large N on PCIe GPUs.
  - 'numpy'       : numpy.linalg.solve (LAPACK dgesv, CPU float64)
  - 'scipy'       : scipy.linalg.solve with assume_a='pos' (Cholesky, CPU float64)
"""

import time
import warnings
import numpy as np
import sys, os

# MPDOK solver path
_MPDOK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MPDOK_ROOT not in sys.path:
    sys.path.insert(0, _MPDOK_ROOT)


# ---------------------------------------------------------------------------
# 0. OOC GMRES-IR internals
# ---------------------------------------------------------------------------

def _inner_gmres(rhs_gpu, sgemv_fn, restart: int):
    """FP32 GMRES inner solve.

    Args:
        rhs_gpu:  (N,) CuPy FP64 residual
        sgemv_fn: callable v_fp32 → y_fp32 (tiled A_fp32 @ v on GPU)
        restart:  Krylov dimension

    Returns:
        e: (N,) CuPy FP32 correction vector
    """
    import cupy as cp
    N = len(rhs_gpu)
    rhs = rhs_gpu.astype(cp.float32)
    beta = float(cp.linalg.norm(rhs))
    if beta < 1e-30:
        return cp.zeros(N, dtype=cp.float32)

    # Krylov basis: column-major so V[:, k] accesses are contiguous
    V  = cp.empty((N, restart + 1), dtype=cp.float32, order='F')
    V[:, 0] = rhs / beta

    H  = np.zeros((restart + 1, restart), dtype=np.float64)
    cs = np.zeros(restart, dtype=np.float64)
    sn = np.zeros(restart, dtype=np.float64)
    g  = np.zeros(restart + 1, dtype=np.float64)
    g[0] = beta

    m = restart
    for k in range(restart):
        w = sgemv_fn(V[:, k])

        # Arnoldi (modified Gram-Schmidt with re-orthogonalisation)
        h1 = V[:, :k + 1].T @ w
        H[:k + 1, k] = cp.asnumpy(h1).astype(np.float64)
        w -= V[:, :k + 1] @ h1
        h2 = V[:, :k + 1].T @ w   # DGKS correction
        H[:k + 1, k] += cp.asnumpy(h2).astype(np.float64)
        w -= V[:, :k + 1] @ h2

        nrm = float(cp.linalg.norm(w))
        H[k + 1, k] = nrm
        if nrm > 1e-12:
            V[:, k + 1] = w / nrm
        else:
            m = k + 1
            break

        # Apply previous Givens rotations
        for j in range(k):
            tmp        =  cs[j] * H[j, k] + sn[j] * H[j + 1, k]
            H[j + 1, k] = -sn[j] * H[j, k] + cs[j] * H[j + 1, k]
            H[j,     k] =  tmp

        rho    = np.hypot(H[k, k], H[k + 1, k])
        cs[k]  = H[k,     k] / rho
        sn[k]  = H[k + 1, k] / rho
        H[k,     k] = rho
        H[k + 1, k] = 0.0
        g[k + 1] = -sn[k] * g[k]
        g[k]     =  cs[k] * g[k]

        if abs(g[k + 1]) / (beta + 1e-30) < 1e-6:
            m = k + 1
            break

    # Back-substitution
    y_ls = np.zeros(m, dtype=np.float64)
    y_ls[m - 1] = g[m - 1] / H[m - 1, m - 1]
    for j in range(m - 2, -1, -1):
        y_ls[j] = (g[j] - np.dot(H[j, j + 1:m], y_ls[j + 1:m])) / H[j, j]

    return V[:, :m] @ cp.array(y_ls, dtype=cp.float32)


def gblup_solve_ooc(G: np.ndarray, y: np.ndarray, lam: float,
                    tile_rows: int = 4096,
                    tol: float = 1e-9,
                    restart: int = 200,
                    maxiter_outer: int = 20,
                    verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Out-of-Core GMRES-IR for G-BLUP at arbitrarily large N.

    Stores (G + λI) as FP32 in RAM and streams tiles (tile_rows × N × 4 B)
    to GPU during inner GMRES.  The outer FP64 residual uses the original G
    directly — no second copy of G is required.

    VRAM footprint (constant in N):
        one FP32 tile: tile_rows × N × 4 B   (e.g., 4096×20000×4 = 320 MB)
        Krylov basis:  N × restart × 4 B      (e.g., 20000×50×4  =   4 MB)

    RAM footprint:
        G (FP64, caller's):  N² × 8 B         (e.g., 20000² × 8  = 3.2 GB)
        A_fp32 (new):        N² × 4 B         (e.g., 20000² × 4  = 1.6 GB)

    Args:
        G:            (N, N) float64 GRM
        y:            (N,) phenotype
        lam:          regularisation
        tile_rows:    rows per GPU tile (tune to fill ~80% of free VRAM)
        tol:          outer relative residual tolerance
        restart:      inner GMRES Krylov dimension
        maxiter_outer: max outer GMRES-IR iterations
        verbose:      print outer-loop convergence

    Returns:
        alpha: (N,) dual coefficients
        stats: dict with time_s, residual, backend, outer_iters
    """
    import cupy as cp

    t0 = time.perf_counter()
    N = len(y)

    # Build FP32 regularised matrix in RAM (half the memory of FP64 copy)
    A_fp32 = G.astype(np.float32, copy=True)
    A_fp32.flat[::N + 1] += lam          # add lam to diagonal in-place

    # --- tiled matvecs ---
    def sgemv(v_fp32: 'cp.ndarray') -> 'cp.ndarray':
        """A_fp32 @ v  streaming tiles from RAM → VRAM."""
        out = cp.zeros(N, dtype=cp.float32)
        for i in range(0, N, tile_rows):
            rows = min(tile_rows, N - i)
            out[i:i + rows] = cp.asarray(A_fp32[i:i + rows]) @ v_fp32
        return out

    def dgemv(v_fp64: 'cp.ndarray') -> 'cp.ndarray':
        """(G + λI) @ v  — uses original G, adds λv on GPU (no FP64 copy)."""
        out = cp.zeros(N, dtype=cp.float64)
        for i in range(0, N, tile_rows):
            rows = min(tile_rows, N - i)
            out[i:i + rows] = cp.asarray(G[i:i + rows]) @ v_fp64
        out += lam * v_fp64
        return out

    b_gpu = cp.asarray(y, dtype=cp.float64)
    b_norm = float(cp.linalg.norm(b_gpu)) + 1e-30
    x = cp.zeros(N, dtype=cp.float64)
    rel = float("inf")
    outer_count = 0

    for outer in range(maxiter_outer):
        outer_count = outer + 1
        r   = b_gpu - dgemv(x)
        rel = float(cp.linalg.norm(r)) / b_norm
        if verbose:
            print(f"  OOC outer {outer}: rel={rel:.2e}")
        if rel < tol:
            break
        e = _inner_gmres(r, sgemv, restart)
        x = x + e.astype(cp.float64)

    alpha   = cp.asnumpy(x)
    elapsed = time.perf_counter() - t0

    # Final residual uses FP64 dgemv for accuracy
    r_final = cp.asnumpy(dgemv(cp.asarray(alpha, dtype=cp.float64))
                          - b_gpu)
    resid = float(np.linalg.norm(r_final) / (np.linalg.norm(y) + 1e-30))

    return alpha, {"time_s": elapsed, "residual": resid,
                   "backend": "mpdok_ooc", "N": N, "lam": lam,
                   "outer_iters": outer_count, "final_rel": rel}


def gblup_solve_ooc_z(X_raw: np.ndarray, p: np.ndarray, scale: float,
                       y: np.ndarray, lam: float,
                       tile_rows: int = 1024,
                       tol: float = 1e-9,
                       restart: int = 200,
                       maxiter_outer: int = 20,
                       verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Z-based OOC GMRES-IR for G-BLUP at arbitrarily large N.

    Avoids storing G entirely.  Computes the GRM matvec on-the-fly:

        (G + λI) v = Z (Z.T v) / scale + λv

    where Z_ij = X_ij − 2p_j, decomposed into two GPU GEMVs using the identity:

        Z.T v  = X.T v − 2p·sum(v)          [no Z ever materialised]
        Z w    = X w − 2(p·w)·1_N            [same trick]

    X_raw (FP32) is uploaded to GPU once at the start; all inner matvecs are
    GPU HBM-bandwidth-bound (~272 GB/s on RTX 4060) — no PCIe transfers during
    the solve.  The outer FP64 residual casts X tiles to FP64 in-place on GPU.

    VRAM footprint:
        X_fp32  : N × M × 4 bytes   (e.g., 20k × 38k × 4 ≈ 2.9 GB)
        V basis : N × restart × 4   (restart=50, N=20k → 4 MB)
        FP64 tile: tile_rows × M × 8 (peak ≈ 1.2 GB at M=38k, tile_rows=4096)

    Contrast with mpdok_ooc (tile-stream):
        1000 matvecs × 1.6 GB PCIe transfers = 1,600 GB → 100 s at 16 GB/s
    vs mpdok_ooc_z:
        1000 matvecs × 2 × 2.9 GB HBM reads = 5,800 GB → ~21 s at 272 GB/s
        (outer residual: 5 × 2 × tiled-FP64 reads → ~0.2 s)

    Args:
        X_raw:  (N, M) float32 SNP dosage matrix (MAF-filtered; Z = X − 2p)
        p:      (M,) float64 allele frequencies for X_raw columns
        scale:  float; 2·Σ p_j(1−p_j) — VanRaden denominator
        y:      (N,) phenotype vector
        lam:    regularisation = σ²_e / σ²_g
        tile_rows: rows per FP64 cast tile (outer residual only)
        tol:    outer relative residual tolerance
        restart: inner GMRES Krylov dimension (recommend 50; was 200 in ooc)
        maxiter_outer: max outer GMRES-IR iterations
        verbose: print outer-loop convergence

    Returns:
        alpha: (N,) dual coefficients
        stats: dict with time_s, residual, backend, outer_iters
    """
    import cupy as cp

    t0 = time.perf_counter()
    N, M = X_raw.shape

    # Upload X once — all inner matvecs read from GPU HBM, no PCIe
    X_gpu    = cp.asarray(X_raw, dtype=cp.float32)   # (N, M)
    p_fp32   = cp.asarray(p,     dtype=cp.float32)   # (M,)
    p_fp64   = cp.asarray(p,     dtype=cp.float64)   # (M,)
    lam_f32  = cp.float32(lam)
    scale_f32 = cp.float32(scale)

    # ── Inner FP32 matvec: (G + λI) @ v, fully on GPU ───────────────────────
    def sgemv(v_fp32: 'cp.ndarray') -> 'cp.ndarray':
        v_sum   = float(v_fp32.sum())
        Xtv     = X_gpu.T @ v_fp32                     # (M,)  X.T @ v
        Ztv     = Xtv - 2.0 * p_fp32 * v_sum           # (M,)  Z.T @ v
        out     = X_gpu @ Ztv                           # (N,)  X @ Ztv
        out    -= 2.0 * float(p_fp32 @ Ztv)            # Z @ Ztv = X@Ztv − 2(p·Ztv)
        out    /= scale_f32
        out    += lam_f32 * v_fp32
        return out

    # ── Outer FP64 residual: (G + λI) @ v, tiled cast of X on GPU ───────────
    def dgemv(v_fp64: 'cp.ndarray') -> 'cp.ndarray':
        v_sum = float(v_fp64.sum())

        # Pass 1: X.T @ v in FP64 — tile X_fp32→fp64 to avoid 5.8 GB peak
        Xtv = cp.zeros(M, dtype=cp.float64)
        for i in range(0, N, tile_rows):
            rows = min(tile_rows, N - i)
            tile = X_gpu[i:i + rows].astype(cp.float64)   # (tile_rows, M)
            Xtv += tile.T @ v_fp64[i:i + rows]
            del tile
        Ztv = Xtv - 2.0 * p_fp64 * v_sum                 # (M,)

        # Pass 2: X @ Ztv in FP64 — tile again
        out = cp.zeros(N, dtype=cp.float64)
        p_dot_Ztv = float(p_fp64 @ Ztv)
        for i in range(0, N, tile_rows):
            rows = min(tile_rows, N - i)
            tile = X_gpu[i:i + rows].astype(cp.float64)
            out[i:i + rows] = tile @ Ztv
            del tile
        out -= 2.0 * p_dot_Ztv
        out /= scale
        out += lam * v_fp64
        return out

    b_gpu  = cp.asarray(y, dtype=cp.float64)
    b_norm = float(cp.linalg.norm(b_gpu)) + 1e-30
    x      = cp.zeros(N, dtype=cp.float64)
    rel    = float("inf")
    outer_count = 0

    for outer in range(maxiter_outer):
        outer_count = outer + 1
        r   = b_gpu - dgemv(x)
        rel = float(cp.linalg.norm(r)) / b_norm
        if verbose:
            print(f"  OOC-Z outer {outer}: rel={rel:.2e}")
        if rel < tol:
            break
        e = _inner_gmres(r, sgemv, restart)
        x = x + e.astype(cp.float64)

    alpha   = cp.asnumpy(x)
    elapsed = time.perf_counter() - t0

    r_final = cp.asnumpy(dgemv(cp.asarray(alpha, dtype=cp.float64)) - b_gpu)
    resid   = float(np.linalg.norm(r_final) / (np.linalg.norm(y) + 1e-30))

    return alpha, {"time_s": elapsed, "residual": resid,
                   "backend": "mpdok_ooc_z", "N": N, "lam": lam,
                   "outer_iters": outer_count, "final_rel": rel}


# ---------------------------------------------------------------------------
# 1. Core solve
# ---------------------------------------------------------------------------

def gblup_solve(G: np.ndarray, y: np.ndarray, lam: float,
                backend: str = "mpdok",
                X_raw: np.ndarray | None = None,
                p: np.ndarray | None = None,
                scale: float | None = None) -> tuple[np.ndarray, dict]:
    """Solve (G + λI) α = y for the BLUP dual coefficients.

    Args:
        G:       (N, N) float64 GRM — dense SPD
        y:       (N,) phenotype vector (mean-centred recommended)
        lam:     regularisation = σ²_e / σ²_g; tune via CV
        backend: 'mpdok' | 'mpdok_ooc_z' | 'mpdok_ooc' | 'numpy' | 'scipy'
        X_raw:   (N, M) float32 MAF-filtered SNP dosage matrix — enables ooc_z
                 (from info["X_filtered"] returned by compute_grm / bootstrap_grm)
        p:       (M,) allele frequencies matching X_raw columns
        scale:   VanRaden denominator = 2·Σ p_j(1−p_j)

    Returns:
        alpha:  (N,) dual coefficients
        stats:  dict with time_s, residual, backend
    """
    N = len(y)
    A = G.copy()
    np.fill_diagonal(A, np.diag(A) + lam)

    t0 = time.perf_counter()

    if backend in ("mpdok", "mpdok_ooc_z"):
        import cupy as cp
        try:
            free_mem, _ = cp.cuda.Device().mem_info
        except Exception:
            free_mem = 0

    if backend == "mpdok_ooc_z":
        if X_raw is None or p is None or scale is None:
            raise ValueError("backend='mpdok_ooc_z' requires X_raw, p, and scale kwargs")
        N_x, M = X_raw.shape
        ooc_z_peak = N_x * M * 4 + 4096 * M * 8   # X_fp32 + one FP64 tile
        if ooc_z_peak > free_mem * 0.85:
            print(f"[MPDOK] OOC-Z VRAM {ooc_z_peak/1e9:.1f} GB > "
                  f"{free_mem/1e9:.1f} GB free — falling back to OOC tile-stream")
            return gblup_solve_ooc(G, y, lam=lam)
        return gblup_solve_ooc_z(X_raw, p, scale, y, lam)

    if backend == "mpdok":
        # Pre-flight VRAM check: cuSolver LU needs ~3× N² × 8 bytes peak
        # (matrix + LU overwrite + workspace).  Query actual free VRAM so
        # accumulated allocations from prior solves are accounted for.
        peak_needed = 3 * N * N * 8   # bytes (conservative estimate)
        if peak_needed > free_mem * 0.80:
            # Prefer Z-based OOC (HBM-bound) over tile-stream OOC (PCIe-bound)
            if X_raw is not None and p is not None and scale is not None:
                M_x = X_raw.shape[1]
                _tile = 1024          # must match gblup_solve_ooc_z default tile_rows
                _restart = 200        # must match gblup_solve_ooc_z default restart
                ooc_z_peak = (N * M_x * 4            # X_fp32 on GPU
                            + N * (_restart + 1) * 4  # Krylov basis V (FP32)
                            + _tile * M_x * 8          # one FP64 tile in dgemv
                            + N * 8 * 12)              # misc FP64 vectors
                if ooc_z_peak <= free_mem * 0.85:
                    print(f"[MPDOK] N={N}: LU-IR needs {peak_needed/1e9:.1f} GB "
                          f"> {free_mem/1e9:.1f} GB free — routing to OOC-Z")
                    return gblup_solve_ooc_z(X_raw, p, scale, y, lam)
            print(f"[MPDOK] N={N}: peak VRAM needed {peak_needed/1e9:.1f} GB "
                  f"> {free_mem/1e9:.1f} GB free (80% limit) — routing to OOC")
            return gblup_solve_ooc(G, y, lam=lam)

        A_gpu = None
        y_gpu = None
        solver = None
        try:
            from mpdok_ops import MPDOKSolver
            A_gpu = cp.asarray(A, dtype=cp.float64)
            y_gpu = cp.asarray(y, dtype=cp.float64)
            solver = MPDOKSolver()
            alpha_gpu = solver.solve(A_gpu, y_gpu)
            alpha = cp.asnumpy(alpha_gpu)
            del solver, alpha_gpu, A_gpu, y_gpu
        except cp.cuda.memory.OutOfMemoryError:
            # Release all GPU allocations (including partially-constructed solver)
            del solver, A_gpu, y_gpu
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            print(f"[MPDOK] VRAM OOM at N={N} (mid-solve), switching to OOC solver")
            return gblup_solve_ooc(G, y, lam=lam)
        except Exception as e:
            print(f"[MPDOK] GPU solve failed ({e}), falling back to numpy")
            alpha = np.linalg.solve(A, y)

    elif backend == "mpdok_ooc":
        return gblup_solve_ooc(G, y, lam=lam)

    elif backend == "numpy":
        alpha = np.linalg.solve(A, y)

    elif backend == "scipy":
        from scipy.linalg import solve
        alpha = solve(A, y, assume_a="pos")

    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    elapsed = time.perf_counter() - t0

    # Residual ||Aα - y|| / ||y||
    resid = float(np.linalg.norm(A @ alpha - y) / (np.linalg.norm(y) + 1e-30))

    return alpha, {"time_s": elapsed, "residual": resid,
                   "backend": backend, "N": N, "lam": lam}


# ---------------------------------------------------------------------------
# 2. Prediction
# ---------------------------------------------------------------------------

def gblup_predict(G_train: np.ndarray, y_train: np.ndarray,
                  G_cross: np.ndarray, lam: float,
                  backend: str = "mpdok") -> np.ndarray:
    """Fit on training data and predict on new individuals.

    Args:
        G_train:  (N_tr, N_tr) training GRM
        y_train:  (N_tr,) training phenotypes
        G_cross:  (N_te, N_tr) cross-kernel (test-vs-train GRM)
        lam:      regularisation
        backend:  solver backend

    Returns:
        y_hat: (N_te,) predicted phenotypes
    """
    alpha, _ = gblup_solve(G_train, y_train, lam=lam, backend=backend)
    return G_cross @ alpha


# ---------------------------------------------------------------------------
# 3. Backend benchmark
# ---------------------------------------------------------------------------

def benchmark_backends(G: np.ndarray, y: np.ndarray,
                       lam: float = 1e-2) -> dict:
    """Time all three backends on the same system (G + λI)α = y.

    Returns dict keyed by backend name with time_s, residual, alpha.
    """
    results = {}
    for backend in ("mpdok", "mpdok_ooc", "numpy", "scipy"):
        try:
            alpha, stats = gblup_solve(G, y, lam=lam, backend=backend)
            results[backend] = {**stats, "alpha": alpha}
            print(f"  {backend:8s}  {stats['time_s']*1000:8.1f} ms  "
                  f"resid={stats['residual']:.2e}")
        except MemoryError:
            results[backend] = {"error": "OOM", "time_s": float("inf")}
            print(f"  {backend:8s}  OOM")
        except Exception as e:
            results[backend] = {"error": str(e), "time_s": float("inf")}
            print(f"  {backend:8s}  ERROR: {e}")
    return results


# ---------------------------------------------------------------------------
# 4. Lambda (regularisation) sweep via cross-validation
# ---------------------------------------------------------------------------

def cv_lambda_sweep(G: np.ndarray, y: np.ndarray,
                    lambdas: np.ndarray | None = None,
                    k: int = 5, backend: str = "mpdok") -> dict:
    """Grid search over λ values using k-fold CV prediction accuracy.

    Returns dict with lambdas, mean_r, mean_rmse, best_lam.
    """
    from .grm import kfold_indices
    if lambdas is None:
        lambdas = np.logspace(-4, 1, 20)

    splits = kfold_indices(len(y), k=k)
    mean_rs = []

    for lam in lambdas:
        fold_rs = []
        for train, val in splits:
            G_tt = G[np.ix_(train, train)]
            G_vt = G[np.ix_(val, train)]
            y_hat = gblup_predict(G_tt, y[train], G_vt, lam=lam, backend=backend)
            r = np.corrcoef(y[val], y_hat)[0, 1]
            fold_rs.append(r if np.isfinite(r) else 0.0)
        mean_rs.append(np.mean(fold_rs))

    best_idx = int(np.argmax(mean_rs))
    return {"lambdas": lambdas, "mean_r": np.array(mean_rs),
            "best_lam": lambdas[best_idx], "best_r": mean_rs[best_idx]}


# ---------------------------------------------------------------------------
# 5. Heritability estimation (method-of-moments)
# ---------------------------------------------------------------------------

def estimate_h2_mom(G: np.ndarray, y: np.ndarray) -> float:
    """Estimate narrow-sense heritability h² via method-of-moments (Haseman-Elston).

    Regresses off-diagonal phenotypic similarity y_i*y_j against GRM entry G_ij:
        Cov(y_i*y_j, G_ij) / Var(G_ij) ≈ σ²_g

    Then h² = σ²_g / Var(y).

    Fast O(N²) estimator; adequate for demo purposes.
    """
    N = len(y)
    # Upper triangle indices (off-diagonal only)
    ii, jj = np.triu_indices(N, k=1)
    g_off = G[ii, jj]
    y_prod = y[ii] * y[jj]
    # OLS: y_prod = β * g_off + ε
    g_mean = g_off.mean()
    y_mean = y_prod.mean()
    cov = np.mean((g_off - g_mean) * (y_prod - y_mean))
    var_g = np.mean((g_off - g_mean) ** 2)
    sigma2_g = cov / (var_g + 1e-30)
    h2 = sigma2_g / (np.var(y) + 1e-30)
    return float(np.clip(h2, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 6. APY approximation (for accuracy comparison)
# ---------------------------------------------------------------------------

def apy_solve(G: np.ndarray, y: np.ndarray, lam: float,
              n_core: int | None = None,
              seed: int | None = None) -> tuple[np.ndarray, float]:
    """Algorithm for Proven and Young (APY) approximation of G-BLUP.

    Partitions individuals into 'core' (n_core) and 'non-core'. The inverse of
    G is approximated using only the core × core block, avoiding O(N³) full
    factorisation at the cost of prediction accuracy.

    This is the approximation method the industry is FORCED to use at large N
    because exact GBLUP is intractable. MPDOK eliminates the need for it.

    Args:
        G:      (N, N) full GRM
        y:      (N,) phenotype
        lam:    regularisation
        n_core: number of core animals (default: min(N, 2000))
        seed:   if given, randomly sample core animals (reproducible instability
                testing); if None, use first n_core animals (deterministic)

    Returns:
        alpha:  (N,) approximate dual coefficients
        approx_error: ||G_approx - G||_F / ||G||_F (reconstruction quality)
    """
    N = len(y)
    if n_core is None:
        n_core = min(N, 2000)

    if seed is not None:
        rng = np.random.default_rng(seed)
        core = rng.choice(N, size=n_core, replace=False)
        core.sort()
    else:
        core = np.arange(n_core)
    non_core = np.setdiff1d(np.arange(N), core)

    G_cc = G[np.ix_(core, core)]
    G_nc = G[np.ix_(non_core, core)]

    # Core inverse
    A_cc = G_cc.copy()
    np.fill_diagonal(A_cc, np.diag(A_cc) + lam)
    G_cc_inv = np.linalg.inv(A_cc)

    # APY approximation: G_nonccore ≈ G_nc @ G_cc^{-1} @ G_nc^T
    G_nc_approx = G_nc @ G_cc_inv @ G_nc.T

    # Build approximate full matrix
    G_approx = G.copy()
    G_approx[np.ix_(non_core, non_core)] = G_nc_approx

    # Solve with approximate matrix
    A_approx = G_approx.copy()
    np.fill_diagonal(A_approx, np.diag(A_approx) + lam)
    alpha = np.linalg.solve(A_approx, y)

    # Approximation quality
    diff = G_approx - G
    approx_err = np.linalg.norm(diff) / (np.linalg.norm(G) + 1e-30)

    return alpha, float(approx_err)
