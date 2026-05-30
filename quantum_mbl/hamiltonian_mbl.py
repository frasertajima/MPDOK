"""
MBL (Many-Body Localisation) Hamiltonian — matrix-free H×v and dense variants.

Model: random-field transverse Ising chain
  H = J Σᵢ σᵢᶻσᵢ₊₁ᶻ  +  Σᵢ hᵢ σᵢᶻ  +  Γ Σᵢ σᵢˣ

  J   — nearest-neighbour Ising coupling  (energy scale, typically 1)
  hᵢ  — on-site random disorder field, hᵢ ~ Uniform(−W, +W)
  Γ   — uniform transverse field (makes it quantum, typically 0.5)

Physics:
  W ≪ Wc  →  Thermalising (ETH) phase: entropy grows to Page value
  W ≫ Wc  →  MBL phase: entropy grows logarithmically, imbalance freezes
  Wc ≈ 3.5 for J=1, Γ=0.5

Works with either CuPy (xp=cp) or NumPy (xp=np) arrays — same code path.
"""

import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


# ── initial state ─────────────────────────────────────────────────────────────

def neel_state(n_qubits: int, xp=np, dtype=None):
    """Néel state |↑↓↑↓…⟩ — the standard MBL initial state.

    Imbalance ℐ(0) = 1 by construction. A product state → S(0) = 0.
    """
    dtype = dtype or xp.complex128
    N     = 1 << n_qubits
    psi   = xp.zeros(N, dtype=dtype)
    # Convention: bit=0 → ↑ (sz=+1), bit=1 → ↓ (sz=-1)
    # Néel |↑↓↑↓…⟩: even sites ↑ (bit=0), odd sites ↓ (bit=1)
    # → set bit (n-1-i) = 1 for odd i only
    idx = sum(1 << (n_qubits - 1 - i) for i in range(1, n_qubits, 2))
    psi[idx] = 1.0
    return psi


# ── matrix-free matvec (Krylov path) ─────────────────────────────────────────

def build_matvec_mbl(
    n_qubits: int,
    J:        float = 1.0,
    W:        float = 3.0,
    Gamma:    float = 0.5,
    seed:     int   = 42,
    xp               = np,
):
    """Return a callable matvec(psi) → H|ψ⟩ for the MBL Hamiltonian.

    Works with xp=np (CPU/ThinkPad) or xp=cp (GPU/RTX).
    """
    N = 1 << n_qubits
    k = np.arange(N, dtype=np.int64)   # build diagonal on CPU, upload once

    rng = np.random.default_rng(seed)
    h   = rng.uniform(-W, W, n_qubits)

    # Build diagonal on CPU then upload — avoids keeping k in VRAM
    diag_np = np.zeros(N, dtype=np.float64)
    for i in range(n_qubits):
        j     = (i + 1) % n_qubits
        bit_i = (k >> (n_qubits - 1 - i)) & 1
        bit_j = (k >> (n_qubits - 1 - j)) & 1
        sz_i  = 1 - 2 * bit_i
        sz_j  = 1 - 2 * bit_j
        diag_np += J * sz_i * sz_j
        diag_np += h[i] * sz_i
    diag_c = xp.array(diag_np, dtype=xp.complex128)   # one upload, no k in VRAM

    # Precompute reshape shapes for the bit-flip gather (no index array needed)
    # psi[k ^ flip_i] = psi.reshape(2^i, 2, 2^(n-i-1)) with axis-1 halves swapped
    shapes = [(1 << i, 2, 1 << (n_qubits - 1 - i)) for i in range(n_qubits)]

    def matvec(psi):
        w = diag_c * psi           # 1 alloc: (N,) complex128
        for nblk, _, blk in shapes:
            # Reshape to view — no copy for C-contiguous psi
            pv = psi.reshape(nblk, 2, blk)
            wv = w.reshape(nblk, 2, blk)
            # In-place: avoids full-size temporaries; each slice is N/2 elements
            wv[:, 0, :] -= Gamma * pv[:, 1, :]   # 1 half-size temp: N/2 × 16 B
            wv[:, 1, :] -= Gamma * pv[:, 0, :]   # 1 half-size temp: N/2 × 16 B
        return w

    return matvec, N


# ── diagonal precomputation (Trotter path) ───────────────────────────────────

def build_mbl_diagonal(
    n_qubits: int,
    J:        float = 1.0,
    W:        float = 3.0,
    seed:     int   = 42,
    xp               = np,
    dtype            = None,
):
    """Return (diag, h_fields) for use in Trotter stepping.

    diag: (2^n,) array — diagonal of H_ising = J σᵢᶻσᵢ₊₁ᶻ + hᵢ σᵢᶻ
    h_fields: (n,) disorder realization
    """
    dtype = dtype or xp.float64
    N     = 1 << n_qubits
    k     = xp.arange(N, dtype=xp.int64)

    rng      = np.random.default_rng(seed)
    h_fields = xp.array(rng.uniform(-W, W, n_qubits), dtype=dtype)

    diag = xp.zeros(N, dtype=dtype)
    for i in range(n_qubits):
        j     = (i + 1) % n_qubits
        bit_i = (k >> (n_qubits - 1 - i)) & 1
        bit_j = (k >> (n_qubits - 1 - j)) & 1
        sz_i  = 1 - 2 * bit_i
        sz_j  = 1 - 2 * bit_j
        diag  = diag + J * (sz_i * sz_j).astype(dtype)
        diag  = diag + h_fields[i] * sz_i.astype(dtype)

    return diag, h_fields


# ── level statistics (full diagonalisation, small N only) ────────────────────

def _build_dense_mbl_cpu(n_qubits, J, W, Gamma, seed):
    """Build full dense MBL Hamiltonian in numpy (CPU only, small N ≤ 14)."""
    import scipy.linalg  # noqa: F401 — confirm scipy available
    N   = 1 << n_qubits
    k   = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(seed)
    h   = rng.uniform(-W, W, n_qubits)

    diag = np.zeros(N, dtype=np.float64)
    for i in range(n_qubits):
        j     = (i + 1) % n_qubits
        bit_i = (k >> (n_qubits - 1 - i)) & 1
        bit_j = (k >> (n_qubits - 1 - j)) & 1
        diag += J * (1 - 2 * bit_i) * (1 - 2 * bit_j)
        diag += h[i] * (1 - 2 * bit_i)

    H = np.diag(diag.astype(np.complex128))
    for i in range(n_qubits):
        flip       = 1 << (n_qubits - 1 - i)
        cols       = k ^ flip
        H[k, cols] -= Gamma

    return H


def level_r_ratios(
    n_qubits:       int,
    W:              float,
    n_realizations: int   = 100,
    seed:           int   = 0,
    xp                     = np,
    J:              float = 1.0,
    Gamma:          float = 0.5,
) -> np.ndarray:
    """Compute consecutive level-spacing r-ratios for many disorder samples.

    r_n = min(δ_n, δ_{n+1}) / max(δ_n, δ_{n+1})
    Poisson (MBL):  ⟨r⟩ ≈ 0.386
    GOE   (thermal): ⟨r⟩ ≈ 0.530

    Uses GPU (CuPy) when xp=cp AND N ≤ 12 (H fits in VRAM at 268 MB).
    Falls back to CPU scipy for larger N or when CuPy unavailable.
    N=14 requires 4 GB + Fortran copy = 8 GB — hits VRAM limit, use N ≤ 12.
    """
    N         = 1 << n_qubits
    use_gpu   = (_CUPY_AVAILABLE and xp is not np
                 and N * N * 16 * 2 < 2 * 1024**3)  # fits in 2 GB headroom
    rng       = np.random.default_rng(seed)
    r_vals    = []

    for _ in range(n_realizations):
        rseed = int(rng.integers(1 << 30))
        H_cpu = _build_dense_mbl_cpu(n_qubits, J, W, Gamma, seed=rseed)

        if use_gpu:
            import cupy as cp_local
            H_gpu = cp_local.array(H_cpu)
            evals = cp_local.linalg.eigvalsh(H_gpu).get()
            del H_gpu
            cp_local.get_default_memory_pool().free_all_blocks()
        else:
            import scipy.linalg
            evals = scipy.linalg.eigvalsh(H_cpu)

        evals  = np.sort(evals.real)
        deltas = np.diff(evals)
        r      = (np.minimum(deltas[:-1], deltas[1:])
                  / np.maximum(deltas[:-1], deltas[1:]))
        r_vals.extend(r.tolist())

    return np.array(r_vals)
