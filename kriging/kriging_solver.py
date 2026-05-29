"""
Simple Kriging predictor using MPDOK for the covariance solve.

Simple Kriging (known mean μ):

    C · λ = (z - μ)

where C is the n×n SPD covariance matrix.  Prediction:

    z*(x*) = μ + C*(x*, X) · λ

The critical operation is the SPD solve C·λ = rhs — exactly what MPDOK
accelerates via LU-IR on tensor cores.  The matrix C grows as N² in FP64:
at N=20k it is 3.2 GB; at N=50k it is 20 GB.  SciPy uses dense CPU RAM
and fails around N=15–20k.  MPDOK uses CUDA managed memory and tensor-core
LU factorisation and handles N>100k.

Two backends are exposed:
  - 'mpdok'  : GPU LU-IR via MPDOK; auto-uses cudaMallocManaged for large N
  - 'scipy'  : CPU scipy.linalg.solve (baseline; will OOM/time-out for large N)
  - 'cupy'   : GPU cp.linalg.solve (baseline without IR refinement)
"""

import gc
import time

import cupy as cp
import numpy as np

from MPDOK.kriging.kriging_kernel import (
    build_kriging_cov,
    build_kriging_cov_cpu,
    synthetic_field,
    prediction_grid,
)



def _gpu_memory_reset():
    """Drop live CuPy references then flush the pool — call between backends."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# ── cross-covariance (obs → grid) ─────────────────────────────────────────────

def _kernel_chunk(obs, pred_chunk, sq_obs, sq_pred_chunk, model, length_scale, sigma2):
    """Compute one (N_obs, chunk_size) kernel tile — never stored beyond this call."""
    gram = obs @ pred_chunk.T
    D2   = sq_obs[:, None] + sq_pred_chunk[None, :] - 2.0 * gram
    cp.maximum(D2, 0.0, out=D2)
    if model == 'gaussian':
        K = cp.exp((-1.0 / (2.0 * length_scale ** 2)) * D2)
    elif model == 'exponential':
        K = cp.exp(-cp.sqrt(D2) / length_scale)
    elif model == 'matern32':
        D = cp.sqrt(D2) * (1.7320508075688772 / length_scale)
        K = (1.0 + D) * cp.exp(-D)
    else:
        raise ValueError(f"Unknown model '{model}'.")
    if sigma2 != 1.0:
        K *= sigma2
    return K   # (N_obs, chunk_size)


def build_cross_cov(obs_coords, pred_coords, model='matern32',
                    length_scale=1.0, sigma2=1.0, chunk=2048):
    """Build (N_obs, N_pred) cross-covariance matrix C* in chunks.

    Peak VRAM: N_obs × chunk × 8 bytes (one tile at a time).
    Use cross_cov_predict() instead when N_pred is large — it avoids
    materialising the full matrix entirely.
    """
    obs  = cp.asarray(obs_coords,  dtype=cp.float64)
    pred = cp.asarray(pred_coords, dtype=cp.float64)
    N_obs, N_pred = obs.shape[0], pred.shape[0]
    sq_obs  = cp.sum(obs  ** 2, axis=1)
    sq_pred = cp.sum(pred ** 2, axis=1)

    Cstar = cp.empty((N_obs, N_pred), dtype=cp.float64, order='C')
    for j in range(0, N_pred, chunk):
        end = min(j + chunk, N_pred)
        Cstar[:, j:end] = _kernel_chunk(obs, pred[j:end], sq_obs,
                                        sq_pred[j:end], model, length_scale, sigma2)
    cp.cuda.Stream.null.synchronize()
    return Cstar


def cross_cov_predict(weights, obs_coords, pred_coords, model='matern32',
                      length_scale=1.0, sigma2=1.0, mu=0.0, chunk=2048):
    """Compute mu + weights @ C*(obs, pred) without ever storing full C*.

    Peak VRAM: N_obs × chunk × 8 bytes per pass.
    Safe for any N_pred regardless of VRAM.
    """
    obs  = cp.asarray(obs_coords,  dtype=cp.float64)
    pred = cp.asarray(pred_coords, dtype=cp.float64)
    N_pred  = pred.shape[0]
    sq_obs  = cp.sum(obs  ** 2, axis=1)
    sq_pred = cp.sum(pred ** 2, axis=1)

    z_pred = cp.full(N_pred, mu, dtype=cp.float64)
    for j in range(0, N_pred, chunk):
        end = min(j + chunk, N_pred)
        K_chunk = _kernel_chunk(obs, pred[j:end], sq_obs,
                                sq_pred[j:end], model, length_scale, sigma2)
        z_pred[j:end] += weights @ K_chunk   # (chunk,) — tiny

    cp.cuda.Stream.null.synchronize()
    return z_pred


# ── main predictor ────────────────────────────────────────────────────────────

# VRAM threshold: if matrix + TF32 workspace > 85% of VRAM, use OOC path
# matrix = N²×8, TF32 workspace = N²×4 → combined = N²×12
_VRAM_TOTAL    = cp.cuda.Device(0).mem_info[1]
_OOC_THRESHOLD = _VRAM_TOTAL * 0.85 / 12   # max N² before OOC kicks in


class OrdinaryKriging:
    """Simple Kriging predictor with a choice of backend.

    Solves the SPD system C·λ = (z - μ) where C is the n×n covariance matrix.

    For the 'mpdok' backend, automatically switches to the OOC tiled-GMRES-IR
    path (KrigingOOCSolver) when the matrix + TF32 workspace would exceed 85%
    of VRAM.  The FP32 matrix is stored in RAM (or SSD); only one tile lives
    in VRAM at a time.  No upper bound on N.

    Parameters
    ----------
    model : str
        Covariance model ('matern32', 'gaussian', 'exponential').
    backend : str
        'mpdok', 'cupy', or 'scipy'.
    ooc_store : str
        'ram' or 'ssd' — where to cache the FP32 matrix when OOC is needed.
    ooc_path : str or None
        File path for store='ssd'.
    """

    def __init__(self, model='matern32', backend='mpdok',
                 ooc_store='ram', ooc_path=None):
        self.model     = model
        self.backend   = backend
        self.ooc_store = ooc_store
        self.ooc_path  = ooc_path
        self._coords   = None
        self._weights  = None
        self._mu       = None
        self._l        = None
        self._sigma2   = None
        self.ooc_      = False   # flag: True when last fit used OOC path

    def fit(self, coords, z, length_scale=None, sigma2=1.0, nugget=1e-6):
        """Build C and solve C·λ = (z - μ).

        Parameters
        ----------
        coords : (N, D) array-like
        z      : (N,)   array-like of observations
        """
        coords = cp.asarray(coords, dtype=cp.float64)
        z      = cp.asarray(z,      dtype=cp.float64)
        N = coords.shape[0]

        mu  = float(cp.mean(z))
        rhs = z - mu

        t0 = time.perf_counter()

        if self.backend == 'scipy':
            from scipy.linalg import solve as sp_solve
            coords_np = cp.asnumpy(coords)
            rhs_np    = cp.asnumpy(rhs)
            C_np, l = build_kriging_cov_cpu(coords_np, model=self.model,
                                             length_scale=length_scale,
                                             sigma2=sigma2, nugget=nugget)
            lam = sp_solve(C_np, rhs_np, assume_a='pos')
            self._weights = cp.asarray(lam)
            l = float(l)
            self.ooc_ = False

        elif self.backend == 'cupy':
            C_gpu, l = build_kriging_cov(coords, model=self.model,
                                          length_scale=length_scale,
                                          sigma2=sigma2, nugget=nugget)
            self._weights = cp.linalg.solve(C_gpu, rhs)
            l = float(l)
            self.ooc_ = False

        else:  # mpdok — auto OOC when matrix+workspace exceeds VRAM budget
            use_ooc = N * N > _OOC_THRESHOLD

            if use_ooc:
                from MPDOK.kriging.kriging_ooc import KrigingOOCSolver
                ooc = KrigingOOCSolver(tile_rows=4096)
                # OOC uses FP32 inner GMRES — needs nugget≥1e-2 so cond(C)~1e5
                # and restart=100 so one Krylov sweep is expressive enough.
                ooc_nugget = max(nugget, 1e-2)
                l = ooc.build_kriging(coords, model=self.model,
                                      length_scale=length_scale,
                                      sigma2=sigma2, nugget=ooc_nugget,
                                      store=self.ooc_store,
                                      path=self.ooc_path, verbose=False)
                lam = ooc.solve(rhs, tol=1e-6, maxiter_outer=5, restart=100)
                ooc.free()
                self.ooc_ = True
            else:
                from MPDOK.mpdok_ops import MPDOKSolver
                solver = MPDOKSolver()
                C_gpu, l = build_kriging_cov(coords, model=self.model,
                                              length_scale=length_scale,
                                              sigma2=sigma2, nugget=nugget)
                lam = solver.solve(C_gpu, rhs, maxiter_outer=10)
                self.ooc_ = False

            self._weights = cp.asarray(lam)

        self.fit_time_ = time.perf_counter() - t0
        self._coords   = coords
        self._mu       = mu
        self._l        = float(l)
        self._sigma2   = sigma2
        return self

    def predict(self, pred_coords):
        """Predict at new locations.  Returns (N_pred,) CuPy array.

        Uses cross_cov_predict() internally — peak VRAM is O(N_obs × chunk)
        regardless of N_pred, so any grid size is safe.
        """
        if self._weights is None:
            raise RuntimeError("Call fit() before predict().")

        pred_coords = cp.asarray(pred_coords, dtype=cp.float64)
        return cross_cov_predict(
            self._weights, self._coords, pred_coords,
            model=self.model, length_scale=self._l,
            sigma2=self._sigma2, mu=self._mu,
        )




# ── convenience function for the benchmark ───────────────────────────────────

def run_trial(N, backend='mpdok', model='matern32', seed=42):
    """Run a full kriging trial: generate data, fit, time it."""
    result = {'N': N, 'backend': backend, 'model': model, 'success': False}

    try:
        coords, z = synthetic_field(N, seed=seed)
        ok = OrdinaryKriging(model=model, backend=backend)
        ok.fit(coords, z)
        result['fit_time'] = ok.fit_time_
        result['ooc']      = getattr(ok, 'ooc_', False)
        result['success']  = True

    except MemoryError as e:
        result['error'] = f'MemoryError: {e}'
    except Exception as e:
        result['error'] = str(e)

    _gpu_memory_reset()
    return result
