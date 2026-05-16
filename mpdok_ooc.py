"""
MPDOK — Out-of-Core (OOC) GMRES-IR Solver

For N beyond the VRAM FP32 ceiling (~43K on RTX 4060).

Architecture:
  Build:        GPU computes A_fp32 tiles from coordinates → RAM or SSD binary.
  Outer loop:   FP64 residual r = b − A*x via tiled DGEMV; tiles computed on-the-fly
                from coords — no A_fp64 ever stored.
  Inner GMRES:  FP32 tiled SGEMV; each tile is streamed RAM→VRAM (fast) or
                SSD→RAM→VRAM with a background prefetch thread (double-buffered,
                mirrors the cryo-EM / weatherbench streaming_*_loader.cuf pattern).

Memory at runtime:
  VRAM:  one FP32 tile  (tile_rows × N × 4 bytes)  +  GMRES workspace (N × restart × 4)
  RAM:   full FP32 matrix  (N² × 4 bytes)  — RAM path only
  SSD:   full FP32 matrix  (N² × 4 bytes)  — SSD path only

Usage — RAM path (N ≤ ~108K on 47 GB RAM):
    solver = MPDOKOOCSolver(tile_rows=4096)
    solver.build(coords, gamma=gamma, reg=1e-2, store='ram', verbose=True)
    x = solver.solve(b, tol=1e-5, maxiter_outer=8, restart=50)
    solver.free()

Usage — SSD path (any N, limited by disk and patience):
    solver = MPDOKOOCSolver(tile_rows=4096)
    solver.build(coords, gamma=gamma, reg=1e-2, store='ssd',
                 path='/tmp/A_fp32.bin', verbose=True)
    x = solver.solve(b, tol=1e-5, maxiter_outer=8, restart=20)
    solver.free()
"""

import numpy as np
import cupy as cp


# ── Tile construction ────────────────────────────────────────────────────────

def _build_fp32_tile(coords, sq, i, rows, gamma):
    """Compute A_fp32[i:i+rows, :] on GPU; return as numpy C-order FP32 array."""
    c = coords[i:i + rows]          # (rows, D) FP64 on device
    sq_c = sq[i:i + rows]           # (rows,)   FP64 on device
    D2 = sq_c[:, None] + sq[None, :] - 2.0 * (c @ coords.T)
    cp.maximum(D2, 0.0, out=D2)
    tile = cp.exp(-gamma * D2).astype(cp.float32)
    return cp.asnumpy(tile)         # transfer to RAM, C-order (rows contiguous)


# ── Main solver ──────────────────────────────────────────────────────────────

class MPDOKOOCSolver:
    """
    Out-of-core GMRES-IR for N that exceeds the VRAM FP32 ceiling.

    The FP32 working matrix is cached in RAM or written to an SSD binary file.
    The FP64 outer-residual matrix is never stored — tiles are computed from
    coordinates on demand.

    VRAM footprint during solve: one FP32 tile + GMRES Krylov basis.
    """

    def __init__(self, tile_rows=4096):
        self.tile_rows = tile_rows
        self._store    = None       # 'ram' | 'ssd'
        self._ram_buf  = None       # numpy (N, N) C-order FP32
        self._ssd_path = None       # str: path to SSD binary
        self.N         = None
        self.gamma     = None
        self.reg       = None
        self._coords   = None       # CuPy (N, D) FP64
        self._sq       = None       # CuPy (N,) FP64 squared norms

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, coords, gamma=None, reg=1e-6, store='ram',
              path=None, verbose=True):
        """Compute and cache the FP32 RBF kernel matrix.

        Args:
            coords:  (N, D) FP64 CuPy array.
            gamma:   RBF bandwidth; auto-estimated from coords if None.
            reg:     Diagonal regularisation (default 1e-6).
            store:   'ram' — keep in RAM as numpy array (N² × 4 bytes).
                     'ssd' — stream to a binary file (requires path).
            path:    File path for store='ssd'.
            verbose: Print per-tile progress.

        Returns:
            gamma: bandwidth used.
        """
        import time
        from rbf_kernel import _estimate_gamma

        coords = cp.asarray(coords, dtype=cp.float64)
        N = coords.shape[0]

        if gamma is None:
            gamma = _estimate_gamma(coords)

        self.N       = N
        self.gamma   = gamma
        self.reg     = reg
        self._store  = store
        self._coords = coords
        self._sq     = cp.sum(coords ** 2, axis=1)

        n_tiles  = (N + self.tile_rows - 1) // self.tile_rows
        fp32_gb  = N * N * 4 / 1024**3

        if verbose:
            print(f"  OOC build: N={N:,}  gamma={gamma:.3e}  "
                  f"FP32={fp32_gb:.2f} GB  tiles={n_tiles}  store={store}")

        t0 = time.perf_counter()

        if store == 'ram':
            self._ram_buf = np.empty((N, N), dtype=np.float32)
            for ci, i in enumerate(range(0, N, self.tile_rows)):
                rows = min(self.tile_rows, N - i)
                tile = _build_fp32_tile(coords, self._sq, i, rows, gamma)
                ki = np.arange(rows)
                tile[ki, i + ki] += reg          # diagonal regularisation
                self._ram_buf[i:i + rows, :] = tile
                if verbose:
                    pct = (ci + 1) * 100 // n_tiles
                    print(f"    tile {ci+1}/{n_tiles}  ({pct}%)",
                          end='\r', flush=True)
            self._ssd_path = None

        elif store == 'ssd':
            if path is None:
                raise ValueError("store='ssd' requires a path argument")
            self._ssd_path = path
            self._ram_buf  = None
            tile_cols      = N                   # full row width
            with open(path, 'wb') as fh:
                for ci, i in enumerate(range(0, N, self.tile_rows)):
                    rows = min(self.tile_rows, N - i)
                    tile = _build_fp32_tile(coords, self._sq, i, rows, gamma)
                    ki = np.arange(rows)
                    tile[ki, i + ki] += reg
                    tile.tofile(fh)              # raw FP32, row-major, no header
                    if verbose:
                        pct = (ci + 1) * 100 // n_tiles
                        print(f"    tile {ci+1}/{n_tiles}  ({pct}%)",
                              end='\r', flush=True)
        else:
            raise ValueError(f"store must be 'ram' or 'ssd', got {store!r}")

        t_build = time.perf_counter() - t0
        if verbose:
            print(f"\n  build done in {t_build:.1f} s")

        return gamma

    # ── Tiled SGEMV (inner GMRES, FP32) ─────────────────────────────────────

    def _tiled_sgemv_ram(self, v_fp32):
        """y_fp32 = A_fp32 * v,  streaming tiles from RAM → VRAM."""
        N = self.N
        y = cp.zeros(N, dtype=cp.float32)
        for i in range(0, N, self.tile_rows):
            rows = min(self.tile_rows, N - i)
            tile = cp.asarray(self._ram_buf[i:i + rows, :])  # PCIe transfer
            y[i:i + rows] = tile @ v_fp32
            del tile
        return y

    def _tiled_sgemv_ssd(self, v_fp32):
        """y_fp32 = A_fp32 * v,  streaming from SSD via memory-mapped file.

        np.memmap maps the file into the process virtual address space — the OS
        handles sequential readahead automatically (no explicit prefetch thread
        needed).  For a 9 GB tile read at 2.9 GB/s NVMe the OS readahead pipeline
        keeps the GPU fed continuously.

        For very slow storage (< 1 GB/s) a threading prefetch can be layered on
        top, but memmap + OS readahead is simpler and typically faster because it
        avoids per-tile open/seek/read overhead.
        """
        N = self.N
        y = cp.zeros(N, dtype=cp.float32)

        # Open once as a read-only memory map — no data is copied into RAM
        # upfront; pages are faulted in as each tile slice is accessed.
        A_mmap = np.memmap(self._ssd_path, dtype=np.float32,
                           mode='r', shape=(N, N))
        for i in range(0, N, self.tile_rows):
            rows     = min(self.tile_rows, N - i)
            tile_gpu = cp.asarray(A_mmap[i:i + rows, :])   # page fault + PCIe
            y[i:i + rows] = tile_gpu @ v_fp32
            del tile_gpu

        del A_mmap
        return y

    def _tiled_sgemv(self, v_fp32):
        """Dispatch to RAM or SSD path."""
        if self._store == 'ram':
            return self._tiled_sgemv_ram(v_fp32)
        return self._tiled_sgemv_ssd(v_fp32)

    # ── Tiled DGEMV (outer FP64 residual) — on-the-fly from coords ──────────

    def _tiled_dgemv_fp64(self, v_fp64):
        """y_fp64 = A_fp64 * v,  tiles computed on-the-fly from coordinates.

        A_fp64 is never stored; each tile is computed via cuBLAS DGEMM + exp.
        """
        N      = self.N
        y      = cp.zeros(N, dtype=cp.float64)
        coords = self._coords
        sq     = self._sq

        for i in range(0, N, self.tile_rows):
            rows = min(self.tile_rows, N - i)
            c    = coords[i:i + rows]
            sq_c = sq[i:i + rows]
            D2   = sq_c[:, None] + sq[None, :] - 2.0 * (c @ coords.T)
            cp.maximum(D2, 0.0, out=D2)
            tile = cp.exp(-self.gamma * D2)      # FP64
            ki   = cp.arange(rows)
            tile[ki, i + ki] += self.reg
            y[i:i + rows] = tile @ v_fp64
            del tile, D2

        return y

    # ── Inner GMRES (Python port of inner_gmres from mpdok_solver.cuf) ───────

    def _inner_gmres(self, rhs_fp64, restart):
        """FP32 GMRES: approximately solve A_fp32 * e ≈ rhs_fp64.

        Replicates the Arnoldi / Givens-rotation logic of the Fortran
        inner_gmres subroutine.  The only difference: the SGEMV w = A*v
        uses _tiled_sgemv instead of a single cuBLAS call.

        VRAM at peak: V (N × (restart+1) × 4 bytes) + one tile.
        For N=100K, restart=50: V = 20 MB — negligible.
        """
        N   = self.N
        rhs = rhs_fp64.astype(cp.float32)
        beta = float(cp.linalg.norm(rhs))
        if beta < 1e-30:
            return cp.zeros(N, dtype=cp.float32)

        # Krylov basis — stored on GPU (N × (restart+1), column-major)
        V = cp.empty((N, restart + 1), dtype=cp.float32, order='F')
        V[:, 0] = rhs * (1.0 / beta)

        # Hessenberg and Givens rotation state — tiny, lives on CPU
        H  = np.zeros((restart + 1, restart), dtype=np.float64)
        cs = np.zeros(restart,     dtype=np.float64)
        sn = np.zeros(restart,     dtype=np.float64)
        g  = np.zeros(restart + 1, dtype=np.float64)
        g[0] = beta

        m = restart
        for k in range(restart):
            # w = A_fp32 * V[:,k]   (tiled SGEMV — the key OOC step)
            w = self._tiled_sgemv(V[:, k])

            # Batched Gram-Schmidt pass 1
            h_gpu = V[:, :k + 1].T @ w           # (k+1,) FP32 on device
            h = cp.asnumpy(h_gpu).astype(np.float64)
            H[:k + 1, k] = h
            w -= V[:, :k + 1] @ h_gpu

            # Re-orthogonalisation pass (DGKS): corrects FP32 rounding drift
            # that causes non-monotonic outer convergence for large restart.
            h2_gpu = V[:, :k + 1].T @ w
            H[:k + 1, k] += cp.asnumpy(h2_gpu).astype(np.float64)
            w -= V[:, :k + 1] @ h2_gpu

            nrm = float(cp.linalg.norm(w))
            H[k + 1, k] = nrm

            if nrm > 1e-12:
                V[:, k + 1] = w * (1.0 / nrm)
            else:
                m = k + 1
                break

            # Apply previous Givens rotations to column k of H
            for j in range(k):
                tmp       =  cs[j]*H[j, k] + sn[j]*H[j+1, k]
                H[j+1, k] = -sn[j]*H[j, k] + cs[j]*H[j+1, k]
                H[j,   k] =  tmp

            # New Givens rotation to zero H(k+1, k)
            rho    = np.hypot(H[k, k], H[k+1, k])
            cs[k]  = H[k,   k] / rho
            sn[k]  = H[k+1, k] / rho
            H[k,   k] = rho
            H[k+1, k] = 0.0
            g[k+1] = -sn[k]*g[k]
            g[k]   =  cs[k]*g[k]

            if abs(g[k+1]) / beta < 1e-6:
                m = k + 1
                break

        # Back-substitution: H(1:m,1:m) is upper triangular after Givens
        y = np.zeros(m, dtype=np.float64)
        y[m - 1] = g[m - 1] / H[m - 1, m - 1]
        for j in range(m - 2, -1, -1):
            y[j] = (g[j] - np.dot(H[j, j+1:m], y[j+1:m])) / H[j, j]

        # e = V[:,0:m] * y  (one SGEMV on GPU)
        e = V[:, :m] @ cp.array(y, dtype=cp.float32)
        return e

    # ── Outer GMRES-IR ───────────────────────────────────────────────────────

    def solve(self, b, tol=1e-5, maxiter_outer=20, restart=50, verbose=True):
        """GMRES-IR: outer FP64 residual correction + inner FP32 GMRES.

        Args:
            b:             (N,) right-hand side (CuPy or numpy FP64).
            tol:           Relative residual tolerance.
            maxiter_outer: Max outer refinement iterations.
            restart:       Inner GMRES restart (Krylov dimension).
                           Smaller restart → less VRAM for basis V, more outer iters.
            verbose:       Print outer-iteration residuals and timings.

        Returns:
            x: (N,) FP64 CuPy solution vector.
        """
        import time
        import warnings

        if self.N is None:
            raise RuntimeError("Call build() before solve()")

        b_gpu  = cp.asarray(b, dtype=cp.float64)
        b_norm = float(cp.linalg.norm(b_gpu))

        if b_norm == 0.0:
            return cp.zeros(self.N, dtype=cp.float64)

        x         = cp.zeros(self.N, dtype=cp.float64)
        converged = False

        for outer in range(maxiter_outer):
            t0 = time.perf_counter()
            r   = b_gpu - self._tiled_dgemv_fp64(x)
            rel = float(cp.linalg.norm(r)) / b_norm
            t_dgemv = time.perf_counter() - t0

            if verbose:
                print(f"  outer {outer}: rel_res={rel:.2e}  "
                      f"(FP64 DGEMV {t_dgemv:.1f}s)", flush=True)

            if rel < tol:
                converged = True
                break

            t0 = time.perf_counter()
            e  = self._inner_gmres(r, restart)
            t_gmres = time.perf_counter() - t0

            if verbose:
                print(f"           inner GMRES {t_gmres:.1f}s", flush=True)

            x = x + e.astype(cp.float64)

        if not converged:
            warnings.warn(
                f"MPDOKOOCSolver did not converge within {maxiter_outer} outer "
                f"iterations (final rel_res={rel:.2e})",
                RuntimeWarning, stacklevel=2)

        return x

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def free(self):
        """Release the RAM buffer and GPU coordinate arrays."""
        self._ram_buf  = None
        self._coords   = None
        self._sq       = None
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
