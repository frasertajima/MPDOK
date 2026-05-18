"""
enkf_solver.py — SPD matrix assembly and assimilation solve for the fleet tracker.

Every assimilation cycle:
  1. Build N×N RBF kernel matrix  A = K(X, X) + λI   (X = vehicle positions)
  2. Solve  A α = y  to get GP regression weights
  3. Reconstruct the denoised traffic field at any grid of query points

Solver backends:
  mpdok  — Fortran GMRES-IR (TF32 tensor cores + FP64 iterative refinement)
  scipy  — scipy.linalg.solve with Cholesky (FP64, CPU, ground truth)
  cupy   — cp.linalg.solve (FP64, GPU direct — only for N where A fits in VRAM)

Memory paths:
  N ≤ ~25 000 on 8 GB VRAM: standard CuPy device allocation.
  N > ~25 000:              build_matrix_managed() uses cudaMallocManaged;
                             A lives in system RAM, CUDA pages it on demand.

Stage 1: N = 6 000  (288 MB matrix, all backends viable)
Stage 2: N = 64 000 (32 GB matrix, managed memory + MPDOK only)
"""

import sys
import time
from pathlib import Path

import numpy as np
import cupy as cp
from scipy.linalg import solve as scipy_solve

_SELF  = Path(__file__).parent
_MPDOK = _SELF.parent
sys.path.insert(0, str(_SELF))   # fleet_sim lives here
sys.path.insert(0, str(_MPDOK))  # mpdok_ops, rbf_kernel live here

from rbf_kernel import build_rbf_kernel
from mpdok_ops import MPDOKSolver

from fleet_sim import Fleet, TrafficField, CITY_SIZE

# Regularisation: governs the noise-vs-smoothness trade-off.
# Higher λ → stronger smoothing, better conditioning.
# λ = 0.1 keeps κ(A) ≈ 30–50 for N = 6 000 vehicles at γ = 0.2 (verified).
_DEFAULT_REG = 0.1

# GP length scale: 1/(2σ²) where σ ≈ 1.6 km.
#
# Choosing the bandwidth:
#   - The RBF auto-estimator uses the mean pairwise distance (~57 km for a
#     100 km city), making the kernel nearly rank-1 and poorly conditioned.
#   - γ = 0.2 → correlation drops to < 1% beyond ~5 km, giving localised
#     structure that matches vehicle density (~1.3 km mean spacing at N=6000).
#   - Verified κ(A) ≈ 37 at N=6000, ensuring TF32 preconditioner quality.
_DEFAULT_GAMMA = 0.2   # km⁻²


# ── Assimilator ───────────────────────────────────────────────────────────────

class FleetAssimilator:
    """Builds and solves the N×N SPD fleet assimilation system.

    The kernel matrix K_ij = exp(-γ||p_i - p_j||²) encodes the spatial
    covariance between vehicle measurements.  Adding λI makes it strictly
    positive definite regardless of how closely vehicles cluster.

    γ is estimated from the vehicle geometry on the first call to
    build_matrix() and then frozen — it is a stable model hyperparameter.
    """

    def __init__(self, reg: float = _DEFAULT_REG,
                 gamma: float = _DEFAULT_GAMMA):
        self.reg     = reg
        self._solver = MPDOKSolver()
        # Pre-seed gamma so build_rbf_kernel never falls back to the
        # mean-pairwise-distance estimate (which spans the full city and
        # produces a near-rank-1, poorly-conditioned matrix).
        self._gamma  = gamma

    # ── Matrix construction ───────────────────────────────────────────────

    def build_matrix(self, positions: np.ndarray):
        """Build N×N SPD kernel matrix from vehicle positions.

        Allocates a standard CuPy device array — use when A fits in VRAM
        (N ≲ 25 000 on 8 GB).

        Args:
            positions: (N, 2) float64 CPU array.

        Returns:
            (A, gamma): Fortran-order FP64 CuPy (N, N) array; bandwidth γ.
        """
        coords = cp.asarray(positions, dtype=cp.float64)
        A, gamma = build_rbf_kernel(coords, gamma=self._gamma, reg=self.reg)
        if self._gamma is None:
            self._gamma = gamma
        return A, gamma

    def build_matrix_managed(self, positions: np.ndarray, verbose: bool = True):
        """Build N×N SPD kernel matrix in CUDA managed memory.

        Suitable for large N where A would exceed VRAM.  The matrix is
        filled from CPU via numpy (no GPU page-fault pressure).

        Args:
            positions: (N, 2) float64 CPU array.
            verbose:   Print build progress.

        Returns:
            (A_managed, gamma): _ManagedArray backed by cudaMallocManaged.
        """
        N = positions.shape[0]
        coords = cp.asarray(positions, dtype=cp.float64)
        A_managed = self._solver.alloc_managed(N)
        _, gamma = build_rbf_kernel(
            coords, gamma=self._gamma, reg=self.reg,
            out=A_managed, verbose=verbose,
        )
        if self._gamma is None:
            self._gamma = gamma
        return A_managed, gamma

    # ── Solvers ───────────────────────────────────────────────────────────

    def solve_mpdok(self, A, y: np.ndarray,
                    maxiter_outer: int = 10, restart: int = 100):
        """Solve A α = y with GMRES-IR (TF32 inner / FP64 outer).

        A may be a CuPy ndarray (standard) or _ManagedArray (large N).
        Returns FP64 CuPy (N,) weight vector.

        Default maxiter_outer=10, restart=100 gives convergence to ~1e-12
        for the vehicle-tracking SPD problem at N=6 000–64 000.
        """
        b = cp.asarray(y, dtype=cp.float64)
        return self._solver.solve(A, b,
                                  maxiter_outer=maxiter_outer,
                                  restart=restart)

    def solve_scipy(self, A, y: np.ndarray) -> np.ndarray:
        """Solve A α = y with scipy.linalg.solve (Cholesky, CPU FP64).

        Ground-truth reference.  Practical only for N ≲ 8 000.
        Returns float64 NumPy (N,) array.
        """
        A_raw = A.array if hasattr(A, 'array') else A
        A_cpu = cp.asnumpy(A_raw) if isinstance(A_raw, cp.ndarray) else A_raw
        return scipy_solve(A_cpu, y, assume_a='pos')

    def solve_cupy(self, A, y: np.ndarray):
        """Solve A α = y with cp.linalg.solve (cuBLAS FP64).

        Works only when A fits entirely in VRAM.
        Returns FP64 CuPy (N,) array.
        """
        A_raw = A.array if hasattr(A, 'array') else A
        b = cp.asarray(y, dtype=cp.float64)
        return cp.linalg.solve(A_raw, b)

    # ── Field reconstruction ──────────────────────────────────────────────

    def reconstruct(self, alpha, train_pos: np.ndarray,
                    query_xy: np.ndarray,
                    chunk: int = 256) -> np.ndarray:
        """Evaluate  f̂(q) = Σ_i α_i · k(q, p_i)  at query points.

        Processes query points in chunks of `chunk` rows so the peak
        VRAM is chunk × N × 8 bytes (≈130 MB at chunk=256, N=64k)
        instead of M × N × 8 bytes all at once.

        Args:
            alpha:     (N,) CuPy or NumPy GP weights from solve_*.
            train_pos: (N, 2) vehicle positions (CPU or GPU).
            query_xy:  (M, 2) query coordinates (CPU or GPU).
            chunk:     query rows processed per GPU pass.

        Returns:
            (M,) float64 NumPy array of reconstructed field values.
        """
        if self._gamma is None:
            raise RuntimeError("Call build_matrix() before reconstruct().")

        tr = cp.asarray(train_pos, dtype=cp.float64)
        al = cp.asarray(alpha,     dtype=cp.float64)
        sq_tr = cp.sum(tr ** 2, axis=1)    # (N,)

        M = query_xy.shape[0]
        out = np.empty(M, dtype=np.float64)

        for start in range(0, M, chunk):
            end = min(start + chunk, M)
            qu_c  = cp.asarray(query_xy[start:end], dtype=cp.float64)
            sq_qu = cp.sum(qu_c ** 2, axis=1)      # (chunk,)
            K = sq_qu[:, None] + sq_tr[None, :] - 2.0 * (qu_c @ tr.T)
            cp.maximum(K, 0.0, out=K)
            cp.exp(-self._gamma * K, out=K)         # (chunk, N)
            out[start:end] = cp.asnumpy(K @ al)
            del qu_c, sq_qu, K

        return out


# ── Correctness verification & benchmark ─────────────────────────────────────

def verify_correctness(N: int = 6000, seed: int = 42,
                       also_cupy: bool = True) -> dict:
    """Solve one assimilation step with MPDOK, SciPy, and (optionally) CuPy.

    Reports relative error of each solver against SciPy and wall-clock timing.
    Passes if MPDOK relative error < 1e-8 (FP64 quality from TF32 preconditioner).
    """
    print(f"\n{'='*64}")
    print(f"  SPD Fleet Tracker — Correctness & Benchmark   N = {N:,}")
    print(f"{'='*64}")

    fleet = Fleet(N=N, seed=seed)
    field = TrafficField()
    y = fleet.measure(field)

    print(f"\n  Vehicles:        {N:,}")
    print(f"  Matrix:          {N}×{N}  →  {N*N*8/1e6:.1f} MB  (FP64)")
    print(f"  Measurements:    min={y.min():.3f}  max={y.max():.3f}"
          f"  σ={y.std():.3f}")

    assimilator = FleetAssimilator()

    # Ensure CUDA context is initialised before any timed calls
    cp.cuda.Device(0).synchronize()

    # ── Build SPD matrix ──────────────────────────────────────────────────
    print(f"\n  Building SPD kernel matrix ...")
    t0 = time.perf_counter()
    A, gamma = assimilator.build_matrix(fleet.positions)
    cp.cuda.Stream.null.synchronize()
    build_ms = (time.perf_counter() - t0) * 1e3
    print(f"  Build:           {build_ms:.0f} ms   (γ = {gamma:.4e})")

    results = {'build_ms': build_ms, 'gamma': gamma}

    # ── SciPy — ground truth ──────────────────────────────────────────────
    print(f"\n  [SciPy]  Cholesky FP64 (CPU) ...")
    t0 = time.perf_counter()
    x_scipy = assimilator.solve_scipy(A, y)
    scipy_ms = (time.perf_counter() - t0) * 1e3
    norm_scipy = np.linalg.norm(x_scipy)
    print(f"  Solve:           {scipy_ms:.0f} ms")
    results['scipy_ms'] = scipy_ms

    # ── MPDOK GMRES-IR ────────────────────────────────────────────────────
    print(f"\n  [MPDOK]  GMRES-IR  TF32 inner / FP64 outer ...")
    # Warm-up: initialise cuBLAS/cuSOLVER handles before timing
    _A_w = cp.eye(128, dtype=cp.float64, order='F')
    _b_w = cp.ones(128, dtype=cp.float64)
    _ = assimilator._solver.solve(_A_w, _b_w)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    x_mpdok = assimilator.solve_mpdok(A, y)
    cp.cuda.Stream.null.synchronize()
    mpdok_ms = (time.perf_counter() - t0) * 1e3

    x_mpdok_np = cp.asnumpy(x_mpdok)
    mpdok_rel  = np.linalg.norm(x_mpdok_np - x_scipy) / norm_scipy
    print(f"  Solve:           {mpdok_ms:.1f} ms")
    print(f"  Rel error vs SciPy:   {mpdok_rel:.2e}")
    results['mpdok_ms']      = mpdok_ms
    results['mpdok_rel_err'] = mpdok_rel

    # ── CuPy direct (optional) ────────────────────────────────────────────
    if also_cupy:
        print(f"\n  [CuPy]   cp.linalg.solve FP64 ...")
        try:
            t0 = time.perf_counter()
            x_cupy = assimilator.solve_cupy(A, y)
            cp.cuda.Stream.null.synchronize()
            cupy_ms = (time.perf_counter() - t0) * 1e3
            x_cupy_np = cp.asnumpy(x_cupy)
            cupy_rel  = np.linalg.norm(x_cupy_np - x_scipy) / norm_scipy
            print(f"  Solve:           {cupy_ms:.1f} ms")
            print(f"  Rel error vs SciPy:   {cupy_rel:.2e}")
            results['cupy_ms']      = cupy_ms
            results['cupy_rel_err'] = cupy_rel
        except Exception as exc:
            print(f"  CuPy: FAILED — {exc}")
            results['cupy_ms'] = None

    # ── Reconstruction spot-check ─────────────────────────────────────────
    print(f"\n  Reconstruction spot-check (5 random probe points) ...")
    rng = np.random.default_rng(999)
    probes = rng.uniform(5.0, CITY_SIZE - 5.0, size=(5, 2))
    true_vals  = field.evaluate(probes, fleet.t)
    recon_vals = assimilator.reconstruct(x_mpdok, fleet.positions, probes)
    for i, (tr, re) in enumerate(zip(true_vals, recon_vals)):
        print(f"    probe {i}: true={tr:.4f}  recon={re:.4f}  "
              f"err={abs(tr-re):.4f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  ── Speedup summary ──────────────────────────────────────")
    print(f"  MPDOK vs SciPy:  {scipy_ms/mpdok_ms:.1f}×")
    if results.get('cupy_ms'):
        print(f"  MPDOK vs CuPy:   {results['cupy_ms']/mpdok_ms:.1f}×")
        print(f"  SciPy vs CuPy:   {scipy_ms/results['cupy_ms']:.2f}×")

    passed = mpdok_rel < 1e-8
    results['passed'] = passed
    print(f"\n  {'✓  PASS' if passed else '✗  FAIL'}  "
          f"(MPDOK rel err {mpdok_rel:.2e}  threshold 1e-8)")
    print(f"{'='*64}\n")
    return results


if __name__ == "__main__":
    verify_correctness(N=6000, also_cupy=True)
