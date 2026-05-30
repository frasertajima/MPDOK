"""
Transverse-field Ising Hamiltonians for quantum dynamics demos.

Two regimes:
  integrable  — nearest-neighbour uniform J (ordered quantum magnet)
  chaotic     — all-to-all random J_ij ~ N(0, J²/N)  (SYK-like, quantum chaos)

State space: N qubits → 2^N × 2^N complex128 Hermitian matrix (dense, on GPU).
Practical ceiling on RTX 4060 (8 GB VRAM): N=14 → 16384×16384 ≈ 4 GB.
"""

import numpy as np
import cupy as cp


def _sz_diagonal(n_qubits: int) -> cp.ndarray:
    """Return (n_qubits, 2^n_qubits) array where row i = σᵢᶻ diagonal."""
    N = 1 << n_qubits
    basis = cp.arange(N, dtype=cp.int32)
    diags = cp.empty((n_qubits, N), dtype=cp.float64)
    for i in range(n_qubits):
        bit = (basis >> (n_qubits - 1 - i)) & 1
        diags[i] = 1 - 2 * bit   # +1 for |0⟩, -1 for |1⟩
    return diags


def build_hamiltonian(
    n_qubits: int,
    J:        float | np.ndarray = 1.0,
    h:        float               = 1.0,
    kind:     str                 = 'integrable',
    seed:     int                 = 42,
) -> cp.ndarray:
    """Build the transverse-field Ising Hamiltonian on GPU.

    H = -Σ Jᵢⱼ σᵢᶻσⱼᶻ  -  h Σᵢ σᵢˣ

    kind='integrable' : nearest-neighbour, uniform J (periodic boundary)
    kind='chaotic'    : all-to-all random Jᵢⱼ ~ N(0, J²/N)

    Returns (2^n_qubits, 2^n_qubits) complex128 CuPy array.
    """
    N = 1 << n_qubits
    H = cp.zeros((N, N), dtype=cp.complex128)
    diags = _sz_diagonal(n_qubits)   # (n_qubits, N)

    # ── J matrix ─────────────────────────────────────────────────────────────
    if kind == 'integrable':
        J_mat = np.zeros((n_qubits, n_qubits))
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            a, b = min(i, j), max(i, j)   # ensure upper-triangular storage
            J_mat[a, b] = float(J)
    elif kind == 'chaotic':
        rng   = np.random.default_rng(seed)
        scale = float(J) / np.sqrt(n_qubits)
        J_raw = rng.standard_normal((n_qubits, n_qubits)) * scale
        J_mat = (J_raw + J_raw.T) * 0.5   # symmetrise
    else:
        raise ValueError(f"kind must be 'integrable' or 'chaotic', got {kind!r}")

    # ── σᵢᶻσⱼᶻ diagonal terms ────────────────────────────────────────────────
    idx = cp.arange(N, dtype=cp.int64)
    diag = cp.zeros(N, dtype=cp.float64)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if abs(J_mat[i, j]) < 1e-15:
                continue
            diag -= J_mat[i, j] * diags[i] * diags[j]
    H[idx, idx] = diag.astype(cp.complex128)

    # ── σᵢˣ off-diagonal terms ───────────────────────────────────────────────
    rows = cp.arange(N, dtype=cp.int64)
    for i in range(n_qubits):
        flip = 1 << (n_qubits - 1 - i)
        cols = rows ^ flip
        H[rows, cols] -= h

    return H


def ground_state(H: cp.ndarray) -> tuple[float, cp.ndarray]:
    """Return (ground_energy, ground_state_vector) via full diagonalisation.

    Only feasible for small N (≤ 12).  Use for validation.
    """
    evals, evecs = cp.linalg.eigh(H)
    return float(evals[0].real), evecs[:, 0]


def build_matvec_nn(
    n_qubits: int,
    J:        float = 1.0,
    h:        float = 1.0,
) -> tuple[callable, int]:
    """Matrix-free H×v for nearest-neighbour Ising (no matrix stored).

    H = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ  -  h Σᵢ σᵢˣ   (periodic boundary)

    Memory: O(2^N) for diagonal precomputation + state vectors only.
    No 2^N × 2^N matrix is ever allocated — works for N up to ~22 on 8 GB VRAM.

    Returns (matvec, N) where matvec(psi) → H|ψ⟩ as a CuPy callable.
    """
    N   = 1 << n_qubits
    k   = cp.arange(N, dtype=cp.int64)

    # Precompute diagonal: -J Σᵢ sz_i(k) sz_{i+1}(k)
    diag = cp.zeros(N, dtype=cp.float64)
    for i in range(n_qubits):
        j     = (i + 1) % n_qubits
        bit_i = (k >> (n_qubits - 1 - i)) & 1
        bit_j = (k >> (n_qubits - 1 - j)) & 1
        diag -= J * (1 - 2 * bit_i) * (1 - 2 * bit_j)
    diag_c = diag.astype(cp.complex128)

    # Precompute per-site bit-flip masks
    flips = [cp.int64(1 << (n_qubits - 1 - i)) for i in range(n_qubits)]
    k_fixed = k   # captured in closure

    def matvec(psi: cp.ndarray) -> cp.ndarray:
        w = diag_c * psi
        for flip in flips:
            w = w - h * psi[k_fixed ^ flip]   # σᵢˣ: gather flipped index
        return w

    return matvec, N


def eigenvalue_density(H: cp.ndarray, bins: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of eigenvalues (semicircle law check for chaotic H)."""
    evals = cp.linalg.eigvalsh(H)
    counts, edges = np.histogram(evals.get(), bins=bins, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, counts
