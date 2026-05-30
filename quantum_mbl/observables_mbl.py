"""
MBL observables.

imbalance(psi, n)          — ℐ(t) = (1/N) Σᵢ (−1)ⁱ ⟨σᵢᶻ⟩
                              1 for perfect Néel order, 0 for thermal
entanglement_entropy(psi)  — S via SVD (imported from quantum/)
trajectory_mbl_obs(...)    — batch computation over a trajectory
"""

import sys
import numpy as np

sys.path.insert(0, '/var/home/fraser/machine_learning/fortran/examples/'
                   'collected_examples/matrix_dot/tensor13/'
                   'tensor_core_engine_v5')

try:
    import cupy as cp
    _CUPY = True
except ImportError:
    cp    = None
    _CUPY = False


# ── imbalance ─────────────────────────────────────────────────────────────────

def imbalance(psi, n_qubits: int) -> float:
    """ℐ(t) = (1/N) Σᵢ (−1)ⁱ ⟨σᵢᶻ⟩.

    For Néel initial state: ℐ(0) = 1.
    Thermal phase: ℐ(t→∞) → 0.
    MBL phase: ℐ(t→∞) → finite positive value.
    """
    xp    = cp.get_array_module(psi) if _CUPY else np
    N     = 1 << n_qubits
    basis = xp.arange(N, dtype=xp.int32)
    probs = xp.abs(psi) ** 2

    imb = 0.0
    for i in range(n_qubits):
        bit   = (basis >> (n_qubits - 1 - i)) & 1
        sz_i  = 1 - 2 * bit            # +1 for ↑, -1 for ↓
        sign  = (-1) ** i               # +1 for even sites, -1 for odd
        imb  += float(xp.real(xp.dot(probs, (sign * sz_i).astype(xp.float64))))

    return imb / n_qubits


# ── entanglement entropy (SVD on reshaped state) ──────────────────────────────

def entanglement_entropy(psi, n_qubits: int, n_A: int = None) -> float:
    """Von Neumann entropy S(ρ_A) via bipartition SVD.  No density matrix formed."""
    xp  = cp.get_array_module(psi) if _CUPY else np
    n_A = n_A if n_A is not None else n_qubits // 2
    n_B = n_qubits - n_A
    M   = psi.reshape(1 << n_A, 1 << n_B)
    s   = xp.linalg.svd(M, compute_uv=False)
    s2  = (s ** 2).real
    s2  = s2[s2 > 1e-15]
    return float(-xp.sum(s2 * xp.log(s2)))


# ── batch helper ─────────────────────────────────────────────────────────────

def trajectory_mbl_obs(results, n_qubits: int) -> dict:
    """Extract imbalance + entropy from a trajectory (Trotter or Krylov results)."""
    n_t  = len(results)
    imb  = np.zeros(n_t)
    ee   = np.zeros(n_t)
    times = np.zeros(n_t)

    for i, r in enumerate(results):
        psi      = r.psi if hasattr(r, 'psi') else r.psi_t
        imb[i]   = imbalance(psi, n_qubits)
        ee[i]    = entanglement_entropy(psi, n_qubits)
        times[i] = r.t if hasattr(r, 't') else 0.0

    return {'times': times, 'imbalance': imb, 'entropy': ee}
