"""
gmres_complex.py — GMRES for complex dense linear systems with GPU matvec.

The standard DenseLinearOperator in dense_krylov.py is real-valued (FP32 on
device).  For the Helmholtz BEM system the matrix is complex N×N.  Rather
than converting to a block-real (2N)×(2N) system (which quadruples memory),
this module operates natively in complex64:

  Memory cost: 8 N² bytes   (complex64: 4 bytes real + 4 bytes imag per element)
  vs block-real in FP32:    16 N² bytes  (2× worse)
  vs block-real in FP64:    32 N² bytes  (4× worse)

On RTX 4060 (8 GB):
  complex64 A fits up to N ≈ 30k   (8 × 30k² ≈ 7.2 GB)
  block-real FP32 fits up to N ≈ 22k

Architecture
------------
ComplexDenseOperator
  - stores A in complex64 on GPU (cuBLAS CGEMM path)
  - residuals computed in complex64 (no persistent fp64 copy; VRAM stays at 8N²)
  - optional diagonal preconditioner via M_inv parameter in gmres_complex()

gmres_complex()
  - restarted GMRES(m) for general complex square systems
  - Arnoldi basis vectors in complex64 on GPU
  - Hessenberg + least-squares on CPU (restart × restart, negligible)
  - batched Gram-Schmidt: h = conj(V[:k+1]) @ w  (one GEMM replaces k+1 dots)
  - compatible with any callable op.matvec(x) returning complex64 CuPy array

Precision floor
  Complex64 GMRES converges to ~1e-7 relative residual (FP32 machine eps ~1e-7).
  For tighter tolerances: apply one step of iterative refinement after GMRES
  (compute exact complex128 residual, solve for correction in complex64).
"""

import numpy as np
import cupy as cp


class ComplexDenseOperator:
    """N×N complex matrix stored in complex64 on GPU for fast Krylov matvec.

    cuBLAS CGEMM (complex float) is used for all matvec calls.  On Ampere/Ada
    GPUs this delivers ~10× the bandwidth vs CPU DGEMM for the same operation.

    No persistent fp64 copy is held — VRAM cost is exactly 8N² bytes.
    Residuals are computed in complex64 (adequate for tol ≥ 1e-7).
    """

    def __init__(self, A):
        """
        Args:
            A: (N, N) complex array — complex64 or complex128 NumPy or CuPy.
               Uploaded and stored as complex64; fp64 input is downcast.
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f'A must be square 2-D, got shape {A.shape}')
        self.N     = A.shape[0]
        self.A_c64 = cp.asarray(A, dtype=cp.complex64)
        cp.cuda.Stream.null.synchronize()
        mb = self.A_c64.nbytes / 1e6
        print(f'ComplexDenseOperator: N={self.N:,}  complex64={mb:.1f} MB on GPU')

    def matvec(self, x):
        """A @ x  in complex64 via cuBLAS CGEMM.  x must be complex64 CuPy (N,).

        Returns complex64 CuPy (N,).
        """
        return cp.matmul(self.A_c64, x)

    def residual_norm(self, x, b):
        """||b − A x||₂ computed in complex64.  Returns Python float.

        No fp64 upcast — keeps VRAM at 8N².  Adequate for tol ≥ 1e-7.
        """
        x_c64 = cp.asarray(x, dtype=cp.complex64)
        b_c64 = cp.asarray(b, dtype=cp.complex64)
        r = b_c64 - cp.matmul(self.A_c64, x_c64)
        return float(cp.linalg.norm(r))

    def diagonal(self):
        """Return diagonal of A as complex64 CuPy (N,) — used for preconditioning."""
        return cp.diag(self.A_c64)

    def free(self):
        """Release VRAM.  Call when the operator is no longer needed."""
        del self.A_c64
        self.A_c64 = None
        cp.get_default_memory_pool().free_all_blocks()


# ── Restarted GMRES (complex64) ───────────────────────────────────────────────

def gmres_complex(op, b, x0=None, tol=1e-6, maxiter=None, restart=50,
                  M_inv=None, verbose=False):
    """Restarted GMRES(m) for complex square systems — GPU complex64 matvec.

    Drop-in analogue of dense_krylov.gmres() but for complex A:
      - Arnoldi basis in complex64 on GPU
      - Hessenberg (restart+1 × restart) in complex128 on CPU
      - Batched Gram-Schmidt via a single GEMM per Arnoldi step
      - Convergence checked on relative ||b − Ax|| / ||b|| in complex64

    Args:
        op:      ComplexDenseOperator (or any object with op.matvec(x) and
                 op.residual_norm(x, b) and attribute op.N).
        b:       RHS — complex64 or complex128 NumPy/CuPy (N,).
        x0:      Initial guess (N,).  Defaults to zero.
        tol:     Convergence on relative ||b−Ax||/||b|| (default 1e-6,
                 near complex64 floor).
        maxiter: Max total matvec calls.  Defaults to 3 × restart.
        restart: Krylov subspace size m (default 50).
        M_inv:   Optional preconditioner callable: M_inv(r) → complex64 CuPy (N,).
                 Applied right-hand side.  Use diagonal_preconditioner() to build.
        verbose: Print residual at each restart.

    Returns:
        x:         Solution as complex64 CuPy (N,).
        history:   List of (total_matvecs, rel_res) at each restart boundary.
        converged: True if tol met; False if maxiter reached or stagnated.
    """
    N = op.N
    if maxiter is None:
        maxiter = 3 * restart

    b_c64   = cp.asarray(b, dtype=cp.complex64)
    b_norm  = float(cp.linalg.norm(b_c64))
    if b_norm == 0.0:
        return cp.zeros(N, dtype=cp.complex64), [(0, 0.0)], True

    if x0 is None:
        x = cp.zeros(N, dtype=cp.complex64)
    else:
        x = cp.asarray(x0, dtype=cp.complex64)

    # Preallocate Arnoldi basis — reused across restarts (no per-restart alloc)
    V_mat = cp.empty((restart + 1, N), dtype=cp.complex64)

    # Hessenberg on CPU — small: (restart+1) × restart complex128
    H_cpu = np.zeros((restart + 1, restart), dtype=np.complex128)

    history    = []
    converged  = False
    total_mv   = 0
    prev_rel   = float('inf')
    stag_tol   = 1e-2      # stop if residual fails to drop by ≥ 1%
    max_rst    = (maxiter + restart - 1) // restart

    for outer in range(max_rst):
        # ── Initial residual for this cycle ────────────────────────────────
        r = b_c64 - op.matvec(x)
        if M_inv is not None:
            r = M_inv(r)

        beta = float(cp.linalg.norm(r))
        if beta == 0.0:
            converged = True
            break

        V_mat[0] = r / beta
        H_cpu[:] = 0.0

        m = 0
        for k in range(restart):
            if total_mv >= maxiter:
                break

            # ── Arnoldi step ───────────────────────────────────────────────
            w = op.matvec(V_mat[k])
            total_mv += 1
            if M_inv is not None:
                w = M_inv(w)

            # Batched Gram-Schmidt (complex):
            #   h[j] = <V[j], w> = conj(V[j]) · w
            #   w -= sum_j h[j] V[j]
            # One CGEMM for the inner products, one CGEMM for the update.
            h_k = cp.matmul(V_mat[:k + 1].conj(), w)          # (k+1,) c64 GPU
            H_cpu[:k + 1, k] = cp.asnumpy(h_k).astype(np.complex128)
            w = w - cp.matmul(V_mat[:k + 1].T, h_k)           # (N,) c64 GPU

            h_next = float(cp.linalg.norm(w))
            H_cpu[k + 1, k] = h_next
            m = k + 1

            if h_next < 1e-14:    # lucky breakdown
                break

            V_mat[k + 1] = w / h_next

        # ── Least-squares solve on CPU: min ||β e₁ − H[:m+1, :m] y|| ─────
        e1 = np.zeros(m + 1, dtype=np.complex128)
        e1[0] = beta
        y, _, _, _ = np.linalg.lstsq(H_cpu[:m + 1, :m], e1, rcond=None)

        # ── Update solution (one batched GEMV on GPU) ──────────────────────
        y_gpu = cp.asarray(y, dtype=cp.complex64)
        x = x + cp.matmul(V_mat[:m].T, y_gpu)

        # ── Convergence check ──────────────────────────────────────────────
        rel = op.residual_norm(x, b_c64) / b_norm
        history.append((total_mv, rel))

        if verbose:
            print(f'  GMRES restart {outer + 1:3d}  '
                  f'mv={total_mv:4d}  rel={rel:.3e}')

        if rel < tol:
            converged = True
            break

        if rel > prev_rel * (1.0 - stag_tol):
            if verbose:
                print(f'  Stagnated: {prev_rel:.3e} → {rel:.3e}')
            break
        prev_rel = rel

    return x, history, converged


# ── Preconditioner helpers ────────────────────────────────────────────────────

def diagonal_preconditioner(op):
    """Build a diagonal (Jacobi) preconditioner from op.diagonal().

    Returns a callable M_inv(r) = diag(A)^{-1} r.
    Works for the Helmholtz BEM where the diagonal dominates at small kh.

    Args:
        op: ComplexDenseOperator with .diagonal() method.

    Returns:
        callable: M_inv(r) — complex64 CuPy (N,) → complex64 CuPy (N,).
    """
    d = op.diagonal()                     # (N,) complex64 GPU
    d_safe = cp.where(cp.abs(d) < 1e-30, cp.ones_like(d), d)
    d_inv  = 1.0 / d_safe                 # (N,) complex64

    def M_inv(r):
        return d_inv * cp.asarray(r, dtype=cp.complex64)

    return M_inv


# ── CPU GMRES baseline (for benchmarking) ────────────────────────────────────

def gmres_scipy_cpu(A_np, b_np, restart=50, tol=1e-6, maxiter=None):
    """scipy.sparse.linalg.gmres with dense FP64 CPU matvec.

    Used as the CPU baseline in Stage 2 benchmarks.  Same algorithm, same
    restart parameter as gmres_complex — isolates the hardware (GPU vs CPU)
    as the independent variable.

    Args:
        A_np: (N, N) complex128 NumPy array.
        b_np: (N,)   complex128 NumPy array.
        restart, tol, maxiter: GMRES parameters.

    Returns:
        x:         (N,) complex128 NumPy solution.
        n_iters:   total iterations taken.
        rel_res:   final ||b−Ax||/||b||.
    """
    from scipy.sparse.linalg import LinearOperator, gmres as sp_gmres

    N  = A_np.shape[0]
    mv_count = [0]

    def matvec_cpu(x):
        mv_count[0] += 1
        return A_np @ x

    linop = LinearOperator((N, N), matvec=matvec_cpu, dtype=np.complex128)
    import scipy
    gmres_kw = {'rtol': tol} if tuple(int(x) for x in scipy.__version__.split('.')[:2]) >= (1, 12) else {'tol': tol}
    x, info = sp_gmres(linop, b_np, restart=restart, maxiter=maxiter,
                        atol=tol * np.linalg.norm(b_np), **gmres_kw)
    rel = np.linalg.norm(b_np - A_np @ x) / np.linalg.norm(b_np)
    return x, mv_count[0], rel
