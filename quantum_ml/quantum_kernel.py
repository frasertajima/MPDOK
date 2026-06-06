"""
Quantum kernel construction for the MPDOK Quantum ML lab.

Two kernel sources:
  1. PQK (Projected Quantum Kernel) — computed from real IBM hardware Pauli
     projections stored in the CAR T-cell dataset (phi vectors, 180-dim).
     K_ij = exp(-gamma * ||phi_i - phi_j||^2)

  2. IQP simulator — classically simulates the IQP/ZZFeatureMap circuit
     and adds optional shot noise to mimic NISQ device behaviour.
     K_ij = |<psi(x_i)|psi(x_j)>|^2  +  N(0, 1/sqrt(S)) shot noise

  3. GPU-vectorised IQP — computes all N statevectors simultaneously via
     CuPy, then K = |Ψ Ψ†|² in a single ZGEMM.  Scales to N=10,000+.

Both return dense float64 Gram matrices ready for MPDOK.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# 1. PQK kernel from IBM hardware projections
# ---------------------------------------------------------------------------

def pqk_kernel(phi_a: np.ndarray, phi_b: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """RBF kernel in the IBM quantum projection space.

    Args:
        phi_a: (N, D) projection vectors for first set of points
        phi_b: (M, D) projection vectors for second set of points
        gamma: bandwidth; K_ij = exp(-gamma * ||phi_i - phi_j||^2)

    Returns:
        (N, M) float64 kernel matrix
    """
    # Squared Euclidean distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    a2 = np.sum(phi_a ** 2, axis=1, keepdims=True)   # (N, 1)
    b2 = np.sum(phi_b ** 2, axis=1, keepdims=True).T  # (1, M)
    D2 = a2 + b2 - 2.0 * phi_a @ phi_b.T             # (N, M)
    D2 = np.maximum(D2, 0.0)                          # numerical floor
    return np.exp(-gamma * D2)


def pqk_gram(phi: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Symmetric Gram matrix K_ij from a single set of IBM projections."""
    K = pqk_kernel(phi, phi, gamma)
    np.fill_diagonal(K, 1.0)
    return K


def optimal_gamma(phi: np.ndarray) -> float:
    """Median heuristic: gamma = 1 / (2 * median(||phi_i - phi_j||^2))."""
    n = len(phi)
    idx = np.random.choice(n, min(n, 512), replace=False)
    sub = phi[idx]
    a2 = np.sum(sub ** 2, axis=1, keepdims=True)
    D2 = a2 + a2.T - 2.0 * sub @ sub.T
    D2 = np.maximum(D2, 0.0)
    med = np.median(D2[np.triu_indices(len(sub), k=1)])
    return 1.0 / (2.0 * med + 1e-12)


# ---------------------------------------------------------------------------
# 2. IQP / ZZ feature map simulator (classical, exact statevector)
# ---------------------------------------------------------------------------

def _iqp_state(x: np.ndarray, n_layers: int = 2) -> np.ndarray:
    """Exact statevector for an IQP-style circuit on n_qubits = len(x).

    Applies  H^⊗n · exp(i * x_j * Z_j) · exp(i * x_j*x_k * Z_j Z_k) · H^⊗n
    repeated n_layers times.  State lives in C^{2^n}.
    """
    n = len(x)
    dim = 1 << n
    # Hadamard on all qubits: |ψ⟩ = (1/√2^n) * Σ|z⟩
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    for _ in range(n_layers):
        # Single-qubit Z rotations
        phases = np.zeros(dim, dtype=np.float64)
        for q in range(n):
            for z in range(dim):
                if (z >> q) & 1:
                    phases[z] += x[q]
        psi *= np.exp(1j * phases)

        # Two-qubit ZZ interactions (nearest neighbour)
        phases[:] = 0.0
        for q in range(n - 1):
            for z in range(dim):
                b_q  = (z >> q) & 1
                b_q1 = (z >> (q + 1)) & 1
                if b_q and b_q1:
                    phases[z] += x[q] * x[q + 1]
        psi *= np.exp(1j * phases)

        # Hadamard (FFT-based for arbitrary n)
        psi = _hadamard_transform(psi, n)

    return psi


def _hadamard_transform(psi: np.ndarray, n: int) -> np.ndarray:
    """Walsh-Hadamard transform over n qubits, applied to statevector."""
    v = psi.copy()
    for q in range(n):
        step = 1 << q
        for i in range(0, 1 << n, step << 1):
            for j in range(i, i + step):
                a, b = v[j], v[j + step]
                v[j]        = (a + b) / np.sqrt(2)
                v[j + step] = (a - b) / np.sqrt(2)
    return v


def iqp_kernel_matrix(X: np.ndarray, n_layers: int = 2,
                      n_shots: int = 0, rng: np.random.Generator | None = None
                      ) -> np.ndarray:
    """Build IQP kernel Gram matrix for dataset X (N × n_features).

    n_features must be ≤ 20 (statevector grows as 2^n_qubits).
    n_shots > 0 adds shot noise: σ_ij = sqrt(K_ij*(1-K_ij)/n_shots).
    """
    N, nf = X.shape
    assert nf <= 20, f"IQP statevector requires n_qubits ≤ 20, got {nf}"

    states = [_iqp_state(X[i], n_layers) for i in range(N)]
    K = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            fid = abs(np.vdot(states[i], states[j])) ** 2
            K[i, j] = K[j, i] = fid
    np.fill_diagonal(K, 1.0)

    if n_shots > 0:
        if rng is None:
            rng = np.random.default_rng()
        sigma = np.sqrt(np.maximum(K * (1.0 - K), 0.0) / n_shots)
        noise = rng.standard_normal(K.shape)
        noise = (noise + noise.T) / 2.0          # symmetric noise
        K = np.clip(K + sigma * noise, 0.0, 1.0)
        np.fill_diagonal(K, 1.0)

    return K


# ---------------------------------------------------------------------------
# 3. Shot-noise corruption of PQK projections (NISQ device model)
# ---------------------------------------------------------------------------

def add_shot_noise_to_projections(phi: np.ndarray, n_shots: int,
                                  rng: np.random.Generator | None = None
                                  ) -> np.ndarray:
    """Add measurement shot noise to Pauli projection vectors.

    Each expectation value ⟨σ⟩ ∈ [-1,1] has std ≈ 1/√S from S shots.
    Clips to [-1, 1] to preserve valid expectation value range.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal(phi.shape) / np.sqrt(n_shots)
    return np.clip(phi + noise, -1.0, 1.0)


# ---------------------------------------------------------------------------
# 4. Classical RBF baseline kernel
# ---------------------------------------------------------------------------

def rbf_kernel(X_a: np.ndarray, X_b: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Standard RBF kernel K_ij = exp(-gamma * ||x_i - x_j||^2)."""
    a2 = np.sum(X_a ** 2, axis=1, keepdims=True)
    b2 = np.sum(X_b ** 2, axis=1, keepdims=True).T
    D2 = np.maximum(a2 + b2 - 2.0 * X_a @ X_b.T, 0.0)
    return np.exp(-gamma * D2)


def rbf_gram(X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    K = rbf_kernel(X, X, gamma)
    np.fill_diagonal(K, 1.0)
    return K


# ---------------------------------------------------------------------------
# 5. Synthetic benchmark Gram generator (for scaling study)
# ---------------------------------------------------------------------------

def synthetic_quantum_gram(N: int, rank: int = 64,
                           noise: float = 1e-3,
                           seed: int = 42) -> np.ndarray:
    """Generate an N×N SPD quantum-style Gram matrix for scaling benchmarks.

    Constructed as K = V V^T / rank + noise*I  so it mimics a quantum kernel
    (low effective rank, dense, well-conditioned after regularisation).
    """
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((N, rank)) / np.sqrt(rank)
    K = V @ V.T
    K += noise * np.eye(N)
    # Normalise diagonal to 1 (as a proper kernel matrix)
    d = np.sqrt(np.diag(K))
    K = K / np.outer(d, d)
    np.fill_diagonal(K, 1.0)
    return K.astype(np.float64)


# ---------------------------------------------------------------------------
# 6. GPU-vectorised IQP kernel (scales to N=10,000+)
# ---------------------------------------------------------------------------

def _build_basis_phases(n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute ±1 Z-eigenvalue basis table for n_qubits.

    Returns:
        B:    (2^n, n) int array — bit pattern for each basis state
        B_pm: (2^n, n) float64 — B mapped to {+1, -1} (Z eigenvalues)
    """
    dim = 1 << n_qubits
    z_idx = np.arange(dim, dtype=np.int32)
    B = ((z_idx[:, None] >> np.arange(n_qubits, dtype=np.int32)[None, :]) & 1
         ).astype(np.float64)   # (dim, n)
    B_pm = 1.0 - 2.0 * B       # +1 for |0⟩, -1 for |1⟩
    return B, B_pm


def iqp_statevectors_gpu(X_enc: np.ndarray, n_qubits: int) -> 'cp.ndarray':
    """GPU-vectorised IQP statevectors for N samples simultaneously.

    Implements a 1-layer IQP circuit:
      |ψ(x)⟩ = (1/√dim) Σ_z exp(i φ(x,z)) |z⟩
    where φ(x,z) = Σ_j x_j(1-2z_j) + Σ_{j} x_j x_{j+1}(1-2z_j)(1-2z_{j+1})

    Args:
        X_enc:    (N, n_qubits) float64 angles in [0, π]
        n_qubits: number of qubits (dim = 2^n_qubits)

    Returns:
        Psi: (N, 2^n_qubits) CuPy complex128 statevector matrix
    """
    import cupy as cp
    _, B_pm = _build_basis_phases(n_qubits)
    B_pm_gpu = cp.asarray(B_pm)    # (dim, n)
    X_gpu    = cp.asarray(X_enc, dtype=cp.float64)  # (N, n)

    # Single-qubit rotation phases: x_j acts on qubit j
    phases = X_gpu @ B_pm_gpu.T    # (N, n) @ (n, dim) → (N, dim)

    # Nearest-neighbour ZZ interaction phases
    for q in range(n_qubits - 1):
        zz_basis = B_pm_gpu[:, q] * B_pm_gpu[:, q + 1]   # (dim,)
        zz_data  = X_gpu[:, q]    * X_gpu[:, q + 1]       # (N,)
        phases  += cp.outer(zz_data, zz_basis)             # (N, dim)

    dim = 1 << n_qubits
    Psi = cp.exp(1j * phases) / np.sqrt(dim)   # (N, dim) complex128
    return Psi


def iqp_gram_gpu(X_enc: np.ndarray, n_qubits: int,
                 return_psi: bool = False) -> np.ndarray:
    """IQP Gram matrix K_ij = |⟨ψ(x_i)|ψ(x_j)⟩|² via a single CuPy ZGEMM.

    Memory: O(N × 2^n) for Psi + O(N²) for K.  At N=10,000, n=8: ~2.4 GB GPU.

    Args:
        X_enc:      (N, n_qubits) encoded features in [0, π]
        n_qubits:   circuit width
        return_psi: if True, also return the (N, dim) statevector matrix

    Returns:
        K: (N, N) float64 Gram matrix (host numpy array)
    """
    import cupy as cp
    Psi = iqp_statevectors_gpu(X_enc, n_qubits)       # (N, dim) complex
    K_complex = Psi @ Psi.conj().T                     # (N, N) complex  ← ZGEMM
    K = cp.abs(K_complex) ** 2                         # (N, N) real fidelity
    cp.fill_diagonal(K, 1.0)
    K_host = cp.asnumpy(K.astype(cp.float64))
    if return_psi:
        return K_host, cp.asnumpy(Psi)
    return K_host


def iqp_kernel_cross_gpu(X_enc_a: np.ndarray, X_enc_b: np.ndarray,
                         n_qubits: int) -> np.ndarray:
    """Cross-kernel K(X_a, X_b) for prediction on unseen data."""
    import cupy as cp
    Psi_a = iqp_statevectors_gpu(X_enc_a, n_qubits)
    Psi_b = iqp_statevectors_gpu(X_enc_b, n_qubits)
    K = cp.abs(Psi_a @ Psi_b.conj().T) ** 2
    return cp.asnumpy(K.astype(cp.float64))


def encode_for_iqp(X: np.ndarray, n_qubits: int,
                   encoder=None) -> tuple[np.ndarray, object]:
    """PCA + MinMaxScale to [0, π] for IQP angle encoding.

    Args:
        X:        (N, d) raw features
        n_qubits: target dimensionality after PCA
        encoder:  fitted (PCA, MinMaxScaler) tuple; None = fit from X

    Returns:
        X_enc:   (N, n_qubits) float64 angles in [0, π]
        encoder: fitted (PCA, MinMaxScaler) tuple for reuse on test data
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    if encoder is None:
        pca    = PCA(n_components=n_qubits, random_state=0)
        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X_pca  = pca.fit_transform(X)
        X_enc  = scaler.fit_transform(X_pca)
        return X_enc.astype(np.float64), (pca, scaler)
    else:
        pca, scaler = encoder
        X_pca = pca.transform(X)
        X_enc = scaler.transform(X_pca)
        return np.clip(X_enc, 0.0, np.pi).astype(np.float64), encoder
