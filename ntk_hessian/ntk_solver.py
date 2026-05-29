"""
NTK kernel regression solver — MPDOK LU-IR + matrix-free OOC backend.

Kernel ridge regression with the feature / NTK kernel:
    (K + λI) · α = Y
where K = Φ Φ^T + λI, Y is (N, C) one-hot targets, α is (N, C) weights.

Two MPDOK paths:
  In-VRAM (N ≤ ~18k): explicit N×N FP64 kernel in VRAM; solved via TF32 LU-IR.
  OOC     (N > ~18k): matrix-free GMRES-IR using Φ (N×D); K never formed.
                      Φ stored in VRAM / RAM / SSD depending on size.

SciPy baseline: CPU Cholesky on CPU-side kernel matrix; OOMs above N~40k.

Usage:
    solver = NTKSolver(backend='mpdok')
    solver.fit(model, X_train, y_train)
    preds  = solver.predict(X_test)
    acc    = (preds.argmax(1) == y_test).mean()
"""

import gc
import time

import cupy as cp
import numpy as np
import torch

from MPDOK.ntk_hessian.ntk_builder import (build_feature_kernel,
                                            build_feature_kernel_cpu,
                                            make_one_hot, predict_kernel)


def _gpu_memory_reset():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# In-VRAM threshold: K (FP64, N²×8) + LU buf (FP32, N²×4) must fit in VRAM.
# Combined footprint = N²×12; use 85% of available VRAM.
_VRAM_TOTAL      = cp.cuda.Device(0).mem_info[1]
_INVRAM_THRESHOLD = _VRAM_TOTAL * 0.85 / 12   # N² threshold


class NTKSolver:
    """Kernel ridge regression using the learned feature cosine kernel.

    Parameters
    ----------
    backend   : 'mpdok', 'scipy', or 'cupy'.
    nugget    : regularisation λ (default 1e-2 — needed for LU-IR conditioning).
    normalize : L2-normalise feature rows → cosine kernel (default True).
    ooc_store : 'vram', 'ram', or 'ssd' — where OOC Φ lives (default 'vram').
    ooc_path  : file path for ssd store.
    """

    def __init__(self, backend='mpdok', nugget=1e-2, normalize=True,
                 ooc_store='vram', ooc_path=None):
        self.backend   = backend
        self.nugget    = nugget
        self.normalize = normalize
        self.ooc_store = ooc_store
        self.ooc_path  = ooc_path
        self._alpha    = None
        self._X_obs    = None
        self._model    = None
        self._device   = None
        self.ooc_      = False

    def fit(self, model, X_np, y_np, device='cuda', verbose=True):
        """Build kernel and solve (K + λI) α = Y."""
        N  = len(X_np)
        C  = 10
        Y  = make_one_hot(y_np, C)   # (N, C) float64

        self._X_obs  = X_np
        self._model  = model
        self._device = device

        # ── scipy: always CPU kernel + CPU Cholesky ───────────────────────────
        if self.backend == 'scipy':
            K_np, tb = build_feature_kernel_cpu(
                model, X_np, device=device, nugget=self.nugget,
                normalize=self.normalize, verbose=verbose)
            self.build_time_ = tb

            from scipy.linalg import solve as sp_solve
            t0 = time.perf_counter()
            if verbose:
                print(f'  SciPy solve N={N:,} …', flush=True)
            self._alpha = sp_solve(K_np, Y, assume_a='pos')
            self.solve_time_ = time.perf_counter() - t0
            self.ooc_ = False

        # ── mpdok: auto-switch between in-VRAM LU-IR and matrix-free OOC ─────
        elif self.backend == 'mpdok':
            use_ooc = N * N > _INVRAM_THRESHOLD

            if not use_ooc:
                # ── In-VRAM: build N×N kernel, solve with LU-IR ──────────────
                K_gpu, tb = build_feature_kernel(
                    model, X_np, device=device, nugget=self.nugget,
                    normalize=self.normalize, verbose=verbose)
                self.build_time_ = tb

                from MPDOK.mpdok_ops import LUIRSolver
                lu = LUIRSolver()
                t0 = time.perf_counter()
                if verbose:
                    print(f'  MPDOK LU-IR solve N={N:,} …', flush=True)
                lu.factor(K_gpu)
                Y_gpu = cp.asarray(Y)
                alpha_parts = []
                for c in range(C):
                    alpha_parts.append(lu.solve_factored(Y_gpu[:, c],
                                                         maxiter_outer=8))
                lu.free_factored()
                self._alpha = cp.asnumpy(cp.stack(alpha_parts, axis=1))
                self.solve_time_ = time.perf_counter() - t0
                self.ooc_ = False

            else:
                # ── OOC: matrix-free GMRES-IR — K never formed ───────────────
                from MPDOK.ntk_hessian.ntk_ooc import NTKOOCSolver
                ooc = NTKOOCSolver()
                tb = ooc.build(model, X_np, nugget=self.nugget,
                               store=self.ooc_store, normalize=self.normalize,
                               path=self.ooc_path, device=device,
                               verbose=verbose)
                self.build_time_ = tb if isinstance(tb, float) else 0.0

                t0 = time.perf_counter()
                if verbose:
                    print(f'  MPDOK OOC matrix-free solve N={N:,} …', flush=True)
                self._alpha = ooc.solve_multi(Y, tol=1e-9,
                                              maxiter_outer=5, restart=100,
                                              verbose=verbose)
                ooc.free()
                self.solve_time_ = time.perf_counter() - t0
                self.ooc_ = True

        # ── cupy ─────────────────────────────────────────────────────────────
        else:
            K_gpu, tb = build_feature_kernel(
                model, X_np, device=device, nugget=self.nugget,
                normalize=self.normalize, verbose=verbose)
            self.build_time_ = tb
            t0 = time.perf_counter()
            Y_gpu = cp.asarray(Y)
            alpha_gpu = cp.linalg.solve(K_gpu, Y_gpu)
            self._alpha = cp.asnumpy(alpha_gpu)
            self.solve_time_ = time.perf_counter() - t0
            self.ooc_ = False

        self.total_time_ = self.build_time_ + self.solve_time_
        if verbose:
            ooc_tag = ' (OOC/matfree)' if self.ooc_ else ''
            print(f'  Solved in {self.solve_time_:.2f}s  '
                  f'(total: {self.total_time_:.2f}s){ooc_tag}')
        return self

    def predict(self, X_pred_np, chunk=512):
        """Return (N_pred, C) raw scores and (N_pred,) class labels."""
        scores = predict_kernel(
            self._alpha, self._model, self._X_obs, X_pred_np,
            device=self._device, chunk=chunk, normalize=self.normalize)
        return scores, scores.argmax(axis=1)

    def accuracy(self, X_np, y_np, chunk=512):
        _, preds = self.predict(X_np, chunk=chunk)
        return float((preds == y_np).mean())


# ── convenience timing wrapper ─────────────────────────────────────────────────

def time_ntk_solve(model, X_np, y_np, backend, nugget=1e-2,
                   device='cuda', verbose=False):
    """Fit NTKSolver, return (solver, total_time) or (None, None) on OOM."""
    solver = NTKSolver(backend=backend, nugget=nugget)
    try:
        solver.fit(model, X_np, y_np, device=device, verbose=verbose)
        return solver, solver.total_time_
    except (MemoryError, cp.cuda.memory.OutOfMemoryError) as e:
        return None, None
    finally:
        _gpu_memory_reset()
