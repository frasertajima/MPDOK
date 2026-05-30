"""
Physical observables for quantum dynamics.

  magnetization(psi, n)     — per-site ⟨σᵢᶻ⟩, shape (n,)
  total_magnetization(psi)  — Σᵢ ⟨σᵢᶻ⟩ / n  (normalised)
  loschmidt_echo(psi0, psi) — |⟨ψ₀|ψ(t)⟩|²  (return probability)
  entanglement_entropy(psi) — von Neumann entropy S = -Tr(ρ_A log ρ_A)
                               via bipartition SVD
"""

import numpy as np
import cupy as cp


# ── magnetisation ─────────────────────────────────────────────────────────────

def magnetization(psi: cp.ndarray, n_qubits: int) -> np.ndarray:
    """Per-site expectation ⟨σᵢᶻ⟩ for i=0..n-1.  Returns (n,) numpy array."""
    N     = 1 << n_qubits
    basis = cp.arange(N, dtype=cp.int32)
    probs = cp.abs(psi) ** 2          # |⟨k|ψ⟩|²
    mag   = np.empty(n_qubits)
    for i in range(n_qubits):
        bit = (basis >> (n_qubits - 1 - i)) & 1
        sz  = (1 - 2 * bit).astype(cp.float64)
        mag[i] = float(cp.real(cp.dot(probs, sz)))
    return mag


def total_magnetization(psi: cp.ndarray, n_qubits: int) -> float:
    return float(magnetization(psi, n_qubits).mean())


# ── Loschmidt echo ────────────────────────────────────────────────────────────

def loschmidt_echo(psi0: cp.ndarray, psi_t: cp.ndarray) -> float:
    """Return probability |⟨ψ₀|ψ(t)⟩|².  Measures how much the state forgets its origin."""
    return float(cp.abs(cp.dot(psi0.conj(), psi_t)) ** 2)


# ── entanglement entropy ──────────────────────────────────────────────────────

def entanglement_entropy(
    psi:      cp.ndarray,
    n_qubits: int,
    n_A:      int | None = None,
) -> float:
    """Von Neumann entropy S(ρ_A) = -Tr(ρ_A log ρ_A).

    Bipartition: first n_A qubits (subsystem A) vs rest (subsystem B).
    Defaults to n_A = n_qubits // 2 (equal bipartition, maximises possible entropy).

    Method: reshape |ψ⟩ → 2^n_A × 2^n_B matrix M, SVD, S = -Σ σᵢ² log σᵢ².
    """
    n_A   = n_A if n_A is not None else n_qubits // 2
    n_B   = n_qubits - n_A
    M     = psi.reshape(1 << n_A, 1 << n_B)
    s     = cp.linalg.svd(M, compute_uv=False)   # singular values
    s2    = (s ** 2).real
    s2    = s2[s2 > 1e-15]                        # drop numerical zeros
    return float(-cp.sum(s2 * cp.log(s2)))


# ── batch helpers (over trajectory) ──────────────────────────────────────────

def trajectory_observables(
    results,            # list[KrylovResult]
    psi0: cp.ndarray,
    n_qubits: int,
) -> dict:
    """Extract all observables from a trajectory.  Returns dict of numpy arrays."""
    n_t   = len(results)
    mag   = np.zeros((n_t, n_qubits))
    echo  = np.zeros(n_t)
    ee    = np.zeros(n_t)

    for i, r in enumerate(results):
        mag[i]  = magnetization(r.psi_t, n_qubits)
        echo[i] = loschmidt_echo(psi0, r.psi_t)
        ee[i]   = entanglement_entropy(r.psi_t, n_qubits)

    return {
        'magnetization':       mag,         # (n_t, n_qubits)
        'total_magnetization': mag.mean(1), # (n_t,)
        'loschmidt_echo':      echo,        # (n_t,)
        'entanglement_entropy': ee,         # (n_t,)
    }
