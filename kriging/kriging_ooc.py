"""
Out-of-core Kriging solver — adapts MPDOKOOCSolver for Matérn / GP kernels.

The FP32 covariance matrix is stored in RAM (or SSD) tile by tile.
During the GMRES-IR solve only one tile lives in VRAM at once.

Memory budget:
  N=25k  → FP32 matrix = 2.5 GB RAM,  VRAM = one tile + Krylov basis
  N=50k  → FP32 matrix = 10  GB RAM,  VRAM = one tile + Krylov basis
  N=100k → FP32 matrix = 40  GB RAM,  VRAM = one tile + Krylov basis

store='ram':  all RAM, fastest for matrices < ~40 GB (46 GB system here)
store='ssd':  any size, limited by NVMe throughput (~2.9 GB/s read)
"""

import numpy as np
import cupy as cp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MPDOK.mpdok_ooc import MPDOKOOCSolver


# ── kernel tile builders ──────────────────────────────────────────────────────

_SQRT3 = 1.7320508075688772

def _build_kriging_tile_fp32(coords, sq, i, rows, model, length_scale, sigma2, nugget):
    """Compute one FP32 covariance tile A[i:i+rows, :] on GPU → numpy."""
    c    = coords[i:i + rows]
    sq_c = sq[i:i + rows]
    D2   = sq_c[:, None] + sq[None, :] - 2.0 * (c @ coords.T)
    cp.maximum(D2, 0.0, out=D2)

    if model == 'gaussian':
        K = cp.exp((-1.0 / (2.0 * length_scale ** 2)) * D2)
    elif model == 'exponential':
        K = cp.exp(-cp.sqrt(D2) / length_scale)
    elif model == 'matern32':
        D = cp.sqrt(D2) * (_SQRT3 / length_scale)
        K = (1.0 + D) * cp.exp(-D)
    else:
        raise ValueError(f"Unknown model '{model}'.")

    if sigma2 != 1.0:
        K *= sigma2

    tile = K.astype(cp.float32)

    # Diagonal nugget
    ki = cp.arange(rows)
    tile[ki, i + ki] += nugget

    return cp.asnumpy(tile)


def _build_kriging_tile_fp64(coords, sq, i, rows, model, length_scale, sigma2, nugget):
    """Same as above but stays in FP64 (used for outer-residual DGEMV)."""
    c    = coords[i:i + rows]
    sq_c = sq[i:i + rows]
    D2   = sq_c[:, None] + sq[None, :] - 2.0 * (c @ coords.T)
    cp.maximum(D2, 0.0, out=D2)

    if model == 'gaussian':
        K = cp.exp((-1.0 / (2.0 * length_scale ** 2)) * D2)
    elif model == 'exponential':
        K = cp.exp(-cp.sqrt(D2) / length_scale)
    elif model == 'matern32':
        D = cp.sqrt(D2) * (_SQRT3 / length_scale)
        K = (1.0 + D) * cp.exp(-D)
    else:
        raise ValueError(f"Unknown model '{model}'.")

    if sigma2 != 1.0:
        K *= sigma2

    ki = cp.arange(rows)
    K[ki, i + ki] += nugget
    return K


# ── solver subclass ───────────────────────────────────────────────────────────

class KrigingOOCSolver(MPDOKOOCSolver):
    """Out-of-core kriging covariance solver.

    Overrides the RBF tile builders in MPDOKOOCSolver with Matérn/GP kernels.
    All GMRES-IR logic (tiled SGEMV, Arnoldi, outer residual) is inherited.

    Usage:
        solver = KrigingOOCSolver(tile_rows=4096)
        solver.build_kriging(coords, model='matern32', length_scale=l,
                             store='ram', verbose=True)
        lam = solver.solve(rhs, tol=1e-8, maxiter_outer=8, restart=50)
        solver.free()
    """

    def __init__(self, tile_rows=4096):
        super().__init__(tile_rows=tile_rows)
        self._model        = None
        self._length_scale = None
        self._sigma2       = None
        self._nugget       = None

    def build_kriging(self, coords, model='matern32', length_scale=None,
                      sigma2=1.0, nugget=1e-6, store='ram',
                      path=None, verbose=True):
        """Build and cache the FP32 kriging covariance matrix.

        Args:
            coords:       (N, D) FP64 CuPy array.
            model:        'matern32' | 'gaussian' | 'exponential'.
            length_scale: Auto-estimated if None.
            sigma2:       Variance scale.
            nugget:       Diagonal regularisation.
            store:        'ram' or 'ssd'.
            path:         Required if store='ssd'.
            verbose:      Print build progress.
        """
        import time
        from MPDOK.kriging.kriging_kernel import estimate_length_scale

        coords = cp.asarray(coords, dtype=cp.float64)
        N = coords.shape[0]

        if length_scale is None:
            length_scale = float(estimate_length_scale(coords))

        self._model        = model
        self._length_scale = length_scale
        self._sigma2       = sigma2
        self._nugget       = nugget
        self.N             = N
        self.reg           = nugget
        self._store        = store
        self._coords       = coords
        self._sq           = cp.sum(coords ** 2, axis=1)

        n_tiles = (N + self.tile_rows - 1) // self.tile_rows
        fp32_gb = N * N * 4 / 1e9

        if verbose:
            print(f"  OOC build: N={N:,}  model={model}  l={length_scale:.1f}  "
                  f"FP32={fp32_gb:.2f} GB  tiles={n_tiles}  store={store}")

        t0 = time.perf_counter()

        if store == 'ram':
            self._ram_buf = np.empty((N, N), dtype=np.float32)
            for ci, i in enumerate(range(0, N, self.tile_rows)):
                rows = min(self.tile_rows, N - i)
                tile = _build_kriging_tile_fp32(
                    coords, self._sq, i, rows,
                    model, length_scale, sigma2, nugget)
                self._ram_buf[i:i + rows, :] = tile
                if verbose:
                    pct = (ci + 1) * 100 // n_tiles
                    print(f"    tile {ci+1}/{n_tiles}  ({pct}%)", end='\r', flush=True)
            self._ssd_path = None

        elif store == 'ssd':
            if path is None:
                raise ValueError("store='ssd' requires a path argument.")
            self._ssd_path = path
            self._ram_buf  = None
            with open(path, 'wb') as fh:
                for ci, i in enumerate(range(0, N, self.tile_rows)):
                    rows = min(self.tile_rows, N - i)
                    tile = _build_kriging_tile_fp32(
                        coords, self._sq, i, rows,
                        model, length_scale, sigma2, nugget)
                    tile.tofile(fh)
                    if verbose:
                        pct = (ci + 1) * 100 // n_tiles
                        print(f"    tile {ci+1}/{n_tiles}  ({pct}%)", end='\r', flush=True)
        else:
            raise ValueError(f"store must be 'ram' or 'ssd', got {store!r}.")

        t_build = time.perf_counter() - t0
        if verbose:
            print(f"\n  build done in {t_build:.1f}s")

        return length_scale

    def _tiled_dgemv_fp64(self, v_fp64):
        """Override: outer-residual DGEMV using Matérn FP64 tiles on-the-fly."""
        N = self.N
        y = cp.zeros(N, dtype=cp.float64)
        for i in range(0, N, self.tile_rows):
            rows = min(self.tile_rows, N - i)
            tile = _build_kriging_tile_fp64(
                self._coords, self._sq, i, rows,
                self._model, self._length_scale, self._sigma2, self._nugget)
            y[i:i + rows] = tile @ v_fp64
            del tile
        return y
