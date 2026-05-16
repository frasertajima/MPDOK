"""
MPDOK — On-device RBF / GP kernel matrix builder.

Constructs A[i,j] = exp(-gamma * ||c_i - c_j||^2) + reg * delta_ij
entirely on the GPU using the GEMM identity:

    ||a - b||^2  =  ||a||^2  +  ||b||^2  -  2 * a . b

The right-hand term is a single cuBLAS DGEMM; the outer two are
broadcasted row-vectors.  No N×N×D intermediate tensor is ever formed.

For matrices that exceed VRAM (large N), pass out=solver.alloc_managed(N)
so the result is written directly into CUDA managed memory — no host copy,
no intermediate device allocation of the full matrix.

Usage (fits in VRAM):
    import cupy as cp
    from MPDOK.rbf_kernel import synthetic_coords, build_rbf_kernel

    coords = synthetic_coords(N=8192, seed=42)
    A, gamma = build_rbf_kernel(coords)
    # A is (8192, 8192) FP64 CuPy array, Fortran order, ready for MPDOK

Usage (exceeds VRAM — managed memory):
    from MPDOK.mpdok_ops import MPDOKSolver
    from MPDOK.rbf_kernel import synthetic_coords, build_rbf_kernel

    solver = MPDOKSolver()
    coords = synthetic_coords(N=40000, seed=42)
    A = solver.alloc_managed(40000)            # cudaMallocManaged via Fortran
    build_rbf_kernel(coords, out=A)            # fill on-device, chunk by chunk
    x = solver.solve(A, b)
    solver.free_managed()
"""

import ctypes
import cupy as cp
import numpy as np


def synthetic_coords(N, D=2, seed=42, scale=100.0):
    """Generate N synthetic sensor/node coordinates on the GPU.

    Args:
        N:     Number of points.
        D:     Coordinate dimension (default 2).
        seed:  RNG seed for reproducibility.
        scale: Coordinate range [0, scale] in each dimension.

    Returns:
        coords: (N, D) FP64 CuPy array.  Stays on device — no NumPy.
    """
    rng = cp.random.default_rng(seed)
    return rng.uniform(0.0, scale, size=(N, D)).astype(cp.float64)


def _estimate_gamma(coords, max_sample=512):
    """Estimate RBF bandwidth from mean pairwise squared distance.

    Uses a random subsample to keep cost O(max_sample^2) regardless of N.
    """
    n = min(max_sample, coords.shape[0])
    c = coords[:n]
    sq = cp.sum(c ** 2, axis=1)                       # (n,)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (c @ c.T) # (n, n)
    cp.maximum(D2, 0.0, out=D2)
    mean_d2 = float(cp.mean(D2))
    return 1.0 / (2.0 * mean_d2) if mean_d2 > 1e-30 else 1.0


def build_rbf_kernel(coords, gamma=None, reg=1e-6, out=None, chunk=1024):
    """Build the RBF kernel matrix A on the GPU.

    A[i,j] = exp(-gamma * ||c_i - c_j||^2)  +  reg * I

    Uses the GEMM trick for pairwise distances — no N×N×D intermediate.
    Chunked row-wise so peak extra VRAM is O(chunk × N) regardless of N.

    Args:
        coords: (N, D) FP64 CuPy array of coordinates (on device).
        gamma:  RBF bandwidth.  Auto-estimated from coords if None.
        reg:    Diagonal regularization (default 1e-6 for SPD guarantee).
        out:    (N, N) FP64 CuPy array to write into.  Allocates fresh if None.
                Pass solver.alloc_managed(N) here to write into managed memory
                without any intermediate full-matrix allocation.
        chunk:  Number of rows computed per iteration (default 1024).
                Reduce if VRAM is tight; increase for throughput on large VRAM.

    Returns:
        (A, gamma):  A is the kernel matrix (same object as out if provided),
                     gamma is the bandwidth actually used.
    """
    coords = cp.asarray(coords, dtype=cp.float64)
    N = coords.shape[0]

    if gamma is None:
        gamma = _estimate_gamma(coords)

    sq = cp.sum(coords ** 2, axis=1)   # (N,) squared norms, reused every chunk

    if out is None:
        A = cp.empty((N, N), dtype=cp.float64, order='F')
    else:
        # Accept both raw CuPy arrays and _ManagedArray wrappers.
        A = out.array if hasattr(out, 'array') else out

    for i in range(0, N, chunk):
        rows = min(chunk, N - i)
        c_chunk = coords[i:i + rows]           # (rows, D)
        sq_chunk = sq[i:i + rows]              # (rows,)

        # D2[0:rows, :] = sq_chunk[:, None] + sq[None, :] - 2 * c_chunk @ coords.T
        # All three terms are (rows, N) — no full N×N intermediate.
        D2 = sq_chunk[:, None] + sq[None, :] - 2.0 * (c_chunk @ coords.T)
        cp.maximum(D2, 0.0, out=D2)            # clamp floating-point negatives
        A[i:i + rows, :] = cp.exp(-gamma * D2)

    # Diagonal regularization — one pass over diagonal elements.
    idx = cp.arange(N)
    A[idx, idx] += reg

    cp.cuda.Stream.null.synchronize()
    return A, gamma


def _build_rbf_cpu_into(coords_np, gamma, reg, A_np, chunk=512, verbose=False):
    """Compute RBF kernel on CPU and write column-by-column into A_np.

    A_np must be a Fortran-order (column-major) numpy array, typically a
    numpy view of a CUDA managed memory allocation.  Writing from CPU means
    pages stay in system RAM — no GPU-side page-fault thrashing regardless
    of how large the matrix is.

    Uses numpy BLAS (DGEMM) for pairwise distances: O(N²) not O(N²×D).
    """
    N = coords_np.shape[0]
    row_norms = np.sum(coords_np ** 2, axis=1)   # (N,) reused every chunk

    n_chunks = (N + chunk - 1) // chunk
    for ci, j in enumerate(range(0, N, chunk)):
        end = min(j + chunk, N)
        # gram_chunk[:,k] = coords[j+k] · coords[i] for all i → (N, end-j)
        gram_chunk = coords_np @ coords_np[j:end].T
        D2 = row_norms[:, None] + row_norms[None, j:end] - 2.0 * gram_chunk
        np.maximum(D2, 0.0, out=D2)
        col_data = np.exp(-gamma * D2)           # (N, end-j)
        for k in range(j, end):                  # diagonal regularisation
            col_data[k, k - j] += reg
        A_np[:, j:end] = col_data                # contiguous Fortran write
        if verbose and (ci % 10 == 0 or ci == n_chunks - 1):
            print(f"    CPU build: chunk {ci+1}/{n_chunks}  "
                  f"({(ci+1)*100//n_chunks}%)", end='\r', flush=True)

    if verbose:
        print()


def build_rbf_kernel(coords, gamma=None, reg=1e-6, out=None, chunk=1024,
                     verbose=False):
    """Build the RBF kernel matrix A on the GPU (or CPU for large managed arrays).

    A[i,j] = exp(-gamma * ||c_i - c_j||^2)  +  reg * I

    GPU path (default): chunked DGEMM on device; peak extra VRAM is O(chunk×N).
    CPU path (auto):    triggered when out is a managed array and FP64 > 90% VRAM.
                        Writes column-by-column from CPU so pages stay in RAM —
                        no page-fault thrashing for matrices larger than VRAM.

    Args:
        coords:  (N, D) FP64 CuPy array of coordinates (on device).
        gamma:   RBF bandwidth.  Auto-estimated from coords if None.
        reg:     Diagonal regularisation (default 1e-6 for SPD guarantee).
        out:     (N, N) FP64 array to write into.  Allocates fresh if None.
                 Pass solver.alloc_managed(N) to write into managed memory.
        chunk:   GPU-path rows per iteration (default 1024).  Ignored on CPU path.
        verbose: Print progress for the CPU build path.

    Returns:
        (A, gamma): A is the kernel matrix; gamma is the bandwidth used.
    """
    coords = cp.asarray(coords, dtype=cp.float64)
    N = coords.shape[0]

    if gamma is None:
        gamma = _estimate_gamma(coords)

    is_managed = hasattr(out, 'array')
    A_raw = out.array if is_managed else out

    # ── Auto-select build path ─────────────────────────────────────────────
    fp64_bytes = N * N * 8
    vram_total = cp.cuda.Device(0).mem_info[1]
    use_cpu_build = is_managed and fp64_bytes > vram_total * 0.90

    if use_cpu_build:
        # CPU path: numpy BLAS + direct write to managed memory pages in RAM.
        # coords → numpy (cheap: N×D×8 bytes); managed ptr → numpy view.
        coords_np = cp.asnumpy(coords)
        fp64_bytes_actual = N * N * 8
        buf = (ctypes.c_uint8 * fp64_bytes_actual).from_address(
            int(A_raw.data.ptr))
        A_np = np.frombuffer(buf, dtype=np.float64).reshape((N, N), order='F')
        _build_rbf_cpu_into(coords_np, gamma, reg, A_np,
                            chunk=512, verbose=verbose)
        return A_raw, gamma

    # ── GPU path (original) ────────────────────────────────────────────────
    sq = cp.sum(coords ** 2, axis=1)

    if A_raw is None:
        A = cp.empty((N, N), dtype=cp.float64, order='F')
    else:
        A = A_raw

    for i in range(0, N, chunk):
        rows = min(chunk, N - i)
        c_chunk = coords[i:i + rows]
        sq_chunk = sq[i:i + rows]
        D2 = sq_chunk[:, None] + sq[None, :] - 2.0 * (c_chunk @ coords.T)
        cp.maximum(D2, 0.0, out=D2)
        A[i:i + rows, :] = cp.exp(-gamma * D2)

    idx = cp.arange(N)
    A[idx, idx] += reg

    cp.cuda.Stream.null.synchronize()
    return A, gamma


def weather_front(coords, t, noise_std=0.05, seed=None):
    """Simulate a moving 2D thermal front for time-series demo.

    Generates the right-hand side b_t for time step t.  The front is a
    sine wave drifting diagonally — produces a smooth, visually verifiable
    field that stresses the solver's precision without external data.

    Args:
        coords:    (N, 2) FP64 CuPy array of sensor coordinates.
        t:         Integer time step (0-based).
        noise_std: Measurement noise amplitude.
        seed:      RNG seed for the noise (defaults to t for reproducibility).

    Returns:
        b: (N,) FP64 CuPy array — temperature field at time t.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    field = 15.0 + 5.0 * cp.sin((x + y) / 20.0 - 0.5 * t)
    if noise_std > 0:
        rng = cp.random.default_rng(t if seed is None else seed)
        field = field + rng.standard_normal(size=field.shape) * noise_std
    return field.astype(cp.float64)


# -----------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------

def _smoke_test():
    print("\n=== rbf_kernel smoke test ===\n")

    N = 2048
    coords = synthetic_coords(N, seed=0)
    print(f"  coords: {coords.shape}  dtype={coords.dtype}  device={coords.device}")

    A, gamma = build_rbf_kernel(coords)
    print(f"  A:      {A.shape}  dtype={A.dtype}  order={'F' if A.flags['F_CONTIGUOUS'] else 'C'}")
    print(f"  gamma:  {gamma:.6e}")

    # SPD check: diagonal should be 1 + reg, off-diagonal < 1
    diag_val = float(A[0, 0])
    off_val  = float(cp.max(cp.abs(A - cp.diag(cp.diag(A)))))
    print(f"  A[0,0] = {diag_val:.6f}  (expect 1 + 1e-6 = 1.000001)")
    print(f"  max |off-diag| = {off_val:.6f}  (expect < 1)")

    # Symmetry check
    sym_err = float(cp.max(cp.abs(A - A.T)))
    print(f"  symmetry error: {sym_err:.2e}  (expect ~0)")

    # weather_front
    b0 = weather_front(coords, t=0)
    b1 = weather_front(coords, t=1)
    diff = float(cp.mean(cp.abs(b1 - b0)))
    print(f"  weather_front mean |b1 - b0| = {diff:.4f}  (should be nonzero)")

    print("\n=== smoke test passed ===\n")


if __name__ == "__main__":
    _smoke_test()
