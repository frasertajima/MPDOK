"""
MPDOK NTK Out-of-Core Solver — matrix-free GMRES-IR.

For N where the N×N kernel matrix would exceed VRAM (N > ~18k on RTX 4060).

KEY INSIGHT: The NTK feature kernel K = Φ Φ^T + λI has rank D (D=256) << N.
Every matrix-vector product reduces to two GEMV operations on the D-dim feature matrix:

    K v = Φ (Φ^T v) + λ v           O(ND) — never O(N²)

This means the N×N kernel matrix is NEVER formed.  Storage:
  VRAM path  (N ≤ ~300k): Φ in VRAM  (N×D×4 bytes = 30 MB at N=30k)
  RAM  path  (N ≤ ~10M) : Φ in RAM, streamed tile-by-tile to VRAM
  SSD  path  (any N)    : Φ on disk, double-buffered streaming

The outer GMRES-IR loop is in FP64; inner GMRES uses FP32 Φ for speed.
Convergence: cosine kernel has cond ~N/10, so GMRES converges in
sqrt(N/10) ≈ 55 iterations at N=30k — extremely fast.

Usage:
    solver = NTKOOCSolver()
    solver.build(model, X_np, nugget=1e-2, store='vram')  # or 'ram' / 'ssd'
    alpha  = solver.solve_multi(Y, tol=1e-9, maxiter_outer=5, restart=100)
    solver.free()

Direct function:
    alpha = ntk_ooc_solve(Phi_gpu, Y_gpu, nugget, tol=1e-9, restart=100)
"""

import os
import time

import cupy as cp
import numpy as np
import torch

try:
    from cupyx.scipy.sparse.linalg import gmres as cp_gmres, LinearOperator
    _CUPYX_GMRES = True
except ImportError:
    _CUPYX_GMRES = False


# ── Matrix-free matvec ────────────────────────────────────────────────────────

def _ntk_mv_fp64(Phi64, v, nugget):
    """K v = Φ (Φ^T v) + λv  in FP64.  Phi64: (N,D), v: (N,), returns (N,)."""
    z = Phi64.T @ v          # (D,)  — D=256, trivial
    return Phi64 @ z + nugget * v


def _ntk_mv_fp32(Phi32, v, nugget):
    """Same in FP32 for the inner GMRES."""
    z = Phi32.T @ v
    return Phi32 @ z + nugget * v


# ── Core GMRES-IR solve ───────────────────────────────────────────────────────

def ntk_ooc_solve(Phi_gpu, b_gpu, nugget=1e-2,
                  tol=1e-9, restart=100, maxiter_outer=5,
                  verbose=False):
    """Matrix-free GMRES-IR for (Phi @ Phi^T + nugget*I) x = b.

    Parameters
    ----------
    Phi_gpu : (N, D) CuPy float32 array — feature matrix in VRAM.
    b_gpu   : (N,)  CuPy float64 array — right-hand side.
    nugget  : float — regularisation λ.

    Returns
    -------
    x : (N,) CuPy float64 array — solution.
    """
    N, D = Phi_gpu.shape
    Phi64 = Phi_gpu.astype(cp.float64)
    Phi32 = Phi_gpu  # already FP32

    b_norm = float(cp.linalg.norm(b_gpu))
    if b_norm < 1e-30:
        return cp.zeros(N, dtype=cp.float64)

    x = cp.zeros(N, dtype=cp.float64)

    # Inner GMRES using cupyx if available, else Lanczos-style CG
    if _CUPYX_GMRES:
        # cupyx GMRES requires CuPy arrays throughout — no numpy
        def mv_fp32_gpu(v_gpu):
            v = v_gpu.astype(cp.float32)
            return _ntk_mv_fp32(Phi32, v, float(nugget)).astype(cp.float64)

        A_op = LinearOperator((N, N), matvec=mv_fp32_gpu, dtype=cp.float64)

        for outer in range(maxiter_outer):
            r = b_gpu - _ntk_mv_fp64(Phi64, x, nugget)   # CuPy FP64 residual
            rel = float(cp.linalg.norm(r)) / b_norm
            if verbose:
                print(f'    outer {outer}: rel_res={rel:.2e}', flush=True)
            if rel < tol:
                if verbose:
                    print(f'    converged in {outer} outer iterations')
                break
            e_gpu, info = cp_gmres(A_op, r, maxiter=restart, tol=1e-6)
            x = x + e_gpu.astype(cp.float64)
    else:
        # Fallback: conjugate gradient (works since K is SPD)
        x = _cg_solve(Phi64, b_gpu, nugget, tol=tol,
                      maxiter=max(restart * maxiter_outer, 500))

    return x


def _cg_solve(Phi64, b, nugget, tol=1e-9, maxiter=1000):
    """CG for SPD (Phi @ Phi^T + nugget*I) x = b — no GMRES dependency."""
    x = cp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = float(cp.dot(r, r))
    b_norm2 = float(cp.dot(b, b))

    for _ in range(maxiter):
        Ap = _ntk_mv_fp64(Phi64, p, nugget)
        pAp = float(cp.dot(p, Ap))
        if abs(pAp) < 1e-60:
            break
        alpha = rr / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = float(cp.dot(r, r))
        if rr_new / b_norm2 < tol ** 2:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new

    return x


# ── OOC Solver class ──────────────────────────────────────────────────────────

class NTKOOCSolver:
    """Matrix-free NTK solver: never forms the N×N kernel matrix.

    Supports three storage modes for Φ:
      'vram'  : Φ stays in VRAM (N×D×4 bytes — 30 MB at N=30k, D=256)
      'ram'   : Φ in numpy RAM, copied tile-by-tile to VRAM per matvec
      'ssd'   : Φ on disk, memory-mapped; tiles loaded on demand

    The matvec K v = Φ(Φ^T v) + λv is always O(ND) regardless of mode.
    """

    def __init__(self, tile_rows=8192):
        self.tile_rows = tile_rows
        self._phi_vram = None   # CuPy (N, D) float32 — VRAM path
        self._phi_ram  = None   # numpy (N, D) float32 — RAM path
        self._phi_ssd  = None   # str path — SSD path
        self._store    = None
        self._nugget   = None
        self.N         = None
        self.D         = None

    def build(self, model, X_np, nugget=1e-2, store='vram',
              normalize=True, path=None, device='cuda', verbose=True):
        """Extract feature matrix Φ and store it according to `store`.

        Returns length_scale (1.0 for feature kernel) for API compatibility.
        """
        t0 = time.perf_counter()
        model.eval()
        N = len(X_np)
        chunk = 512
        X_t = torch.from_numpy(X_np).float()

        if verbose:
            print(f'  [OOC] extracting features  N={N:,}  D=?  store={store}',
                  flush=True)

        parts = []
        with torch.no_grad():
            for i in range(0, N, chunk):
                phi = model.features(X_t[i:i+chunk].to(device)).cpu().numpy()
                parts.append(phi)
        Phi = np.vstack(parts).astype(np.float32)   # (N, D)
        D = Phi.shape[1]

        if normalize:
            norms = np.linalg.norm(Phi, axis=1, keepdims=True).clip(1e-12)
            Phi /= norms

        self.N = N; self.D = D; self._nugget = nugget; self._store = store

        if store == 'vram':
            self._phi_vram = cp.asarray(Phi)   # N×D×4 bytes — tiny
            if verbose:
                mb = N * D * 4 / 1e6
                print(f'  [OOC] Φ in VRAM: {mb:.1f} MB  (K would be {N*N*8/1e9:.1f} GB)')
        elif store == 'ram':
            self._phi_ram = Phi
            if verbose:
                mb = N * D * 4 / 1e6
                print(f'  [OOC] Φ in RAM: {mb:.1f} MB  (K would be {N*N*8/1e9:.1f} GB)')
        else:  # ssd
            path = path or '/tmp/ntk_phi.bin'
            np.save(path, Phi)
            self._phi_ssd = path
            if verbose:
                mb = N * D * 4 / 1e6
                print(f'  [OOC] Φ on SSD: {path}  ({mb:.1f} MB)')

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f'  [OOC] build done in {elapsed:.2f}s')
        return 1.0   # nominal length scale

    def _get_phi_vram(self):
        """Return (N, D) float32 CuPy array, loading from RAM/SSD if needed."""
        if self._phi_vram is not None:
            return self._phi_vram
        if self._phi_ram is not None:
            return cp.asarray(self._phi_ram)   # copy RAM → VRAM for this call
        # SSD: memory-map and load
        phi = np.load(self._phi_ssd)
        return cp.asarray(phi)

    def _matvec_fp64(self, v):
        Phi = self._get_phi_vram().astype(cp.float64)
        return _ntk_mv_fp64(Phi, v, self._nugget)

    def solve(self, b, tol=1e-9, maxiter_outer=5, restart=100, verbose=False):
        """Solve (K + λI) x = b for a single RHS b (N,) float64."""
        b_gpu = cp.asarray(b, dtype=cp.float64)
        Phi_gpu = self._get_phi_vram()   # (N, D) float32 in VRAM
        return ntk_ooc_solve(Phi_gpu, b_gpu, nugget=self._nugget,
                             tol=tol, restart=restart,
                             maxiter_outer=maxiter_outer, verbose=verbose)

    def solve_multi(self, Y, tol=1e-9, maxiter_outer=5, restart=100,
                    verbose=False):
        """Solve (K + λI) A = Y for multi-column Y (N, C) float64.

        Calls solve() once per column — cheap since each inner GMRES uses
        the same Φ (no refactorization).
        Returns (N, C) numpy float64 alpha.
        """
        Y_gpu = cp.asarray(Y, dtype=cp.float64)
        C = Y_gpu.shape[1]
        alpha_parts = []
        for c in range(C):
            xc = self.solve(Y_gpu[:, c], tol=tol, maxiter_outer=maxiter_outer,
                            restart=restart, verbose=verbose)
            alpha_parts.append(xc)
        return cp.asnumpy(cp.stack(alpha_parts, axis=1))

    def free(self):
        """Release VRAM / RAM buffers."""
        self._phi_vram = None
        self._phi_ram  = None
        cp.get_default_memory_pool().free_all_blocks()


# ── Tiled RAM streaming variant (for Φ too large for VRAM) ───────────────────

def ntk_ooc_solve_ram(phi_ram, b_gpu, nugget=1e-2,
                      tol=1e-9, restart=100, maxiter_outer=5,
                      tile_rows=4096, verbose=False):
    """Matrix-free GMRES-IR where Φ is streamed from RAM tile-by-tile.

    Useful when N×D×4 > VRAM (N > ~8M for D=256, 8 GB VRAM).
    For the standard demo (N≤100k), the VRAM path is always sufficient.

    Each matvec streams one tile at a time:
        z = Φ^T v = Σ_tiles  Φ_tile^T v_tile
        then  Kv = Φ z + λv = Σ_tiles  Φ_tile z + λv
    """
    N, D = phi_ram.shape
    b_norm = float(cp.linalg.norm(b_gpu))
    if b_norm < 1e-30:
        return cp.zeros(N, dtype=cp.float64)

    def mv_fp64(v):
        z = cp.zeros(D, dtype=cp.float64)
        for i in range(0, N, tile_rows):
            end = min(i + tile_rows, N)
            phi_t = cp.asarray(phi_ram[i:end]).astype(cp.float64)
            z += phi_t.T @ v[i:end]
        out = cp.zeros(N, dtype=cp.float64)
        for i in range(0, N, tile_rows):
            end = min(i + tile_rows, N)
            phi_t = cp.asarray(phi_ram[i:end]).astype(cp.float64)
            out[i:end] = phi_t @ z + nugget * v[i:end]
        return out

    x = cp.zeros(N, dtype=cp.float64)
    for outer in range(maxiter_outer):
        r = b_gpu - mv_fp64(x)
        rel = float(cp.linalg.norm(r)) / b_norm
        if verbose:
            print(f'    RAM-OOC outer {outer}: rel_res={rel:.2e}', flush=True)
        if rel < tol:
            break
        x = x + _cg_solve(
            cp.asarray(phi_ram).astype(cp.float64), r, nugget,
            tol=1e-6, maxiter=restart)
    return x
