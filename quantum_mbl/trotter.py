"""
Suzuki-Trotter time evolution for large-N MBL systems.

Second-order (Strang) splitting:
  exp(-i H Δt) ≈ exp(-i H_diag Δt/2) · exp(-i Γ Σσˣ Δt) · exp(-i H_diag Δt/2)

Memory: two vectors of size 2^N (psi + one half-size temporary per σˣ site).
No matrix stored — same principle as the matrix-free Krylov but lower accuracy
(Trotter error O(Δt²) per step vs Krylov error < 10⁻¹³).

Designed for:
  - CPU/numpy path on ThinkPad (80 GB RAM) at N=28, complex64
  - GPU/CuPy path on RTX 4060 at N=24, complex64 or complex128

Trotter error for MBL: qualitative observables (entropy, imbalance) are
coarse enough that float32 / Δt=0.05 is more than sufficient.
"""

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


@dataclass
class TrotterResult:
    psi:        object        # state vector, or None when obs_fns used
    t:          float
    wall_time:  float
    n_steps:    int
    obs:        dict = None   # on-the-fly observables if obs_fns was passed


# ── single Trotter step (in-place, O(N) memory) ───────────────────────────────

def trotter_step(psi, phase_half, Gamma: float, dt: float, n_qubits: int):
    """Apply one second-order Trotter step to psi in-place.

    phase_half : precomputed exp(-0.5i dt diag) — same dtype as psi, no temps created
    Gamma      : transverse field strength
    dt         : time step (only used for σˣ rotation angles)
    """
    # exp(-i H_diag dt/2) — in-place multiply, no temporaries
    psi *= phase_half

    cos_g = float(np.cos(Gamma * dt))
    sin_g = float(np.sin(Gamma * dt))

    for i in range(n_qubits):
        block  = 1 << (n_qubits - i - 1)
        psi_3d = psi.reshape(1 << i, 2, block)
        # Copy BOTH halves to contiguous arrays before operating.
        # psi_3d[:, 1, :] is a strided (non-contiguous) view for middle sites —
        # operating on it directly causes cache thrashing (128 KB stride at site 14
        # for N=28, collapsing effective bandwidth from 50 GB/s to ~1 GB/s).
        tmp0 = psi_3d[:, 0, :].copy()
        tmp1 = psi_3d[:, 1, :].copy()
        psi_3d[:, 0, :] =  cos_g * tmp0 - 1j * sin_g * tmp1
        psi_3d[:, 1, :] = -1j * sin_g * tmp0 + cos_g * tmp1

    # exp(-i H_diag dt/2)
    psi *= phase_half

    return psi


def _make_phase(diag, dt: float, dtype):
    """Precompute exp(-0.5i dt diag) without a temporary astype array."""
    xp = cp.get_array_module(diag) if cp is not None else np
    # Cast diag to complex in-place equivalent: multiply imaginary scalar
    return xp.exp(xp.array(-0.5j * dt, dtype=dtype) * diag.astype(dtype))


# ── full trajectory ───────────────────────────────────────────────────────────

def evolve_trotter(
    psi0,
    diag,
    Gamma:    float,
    times:    np.ndarray,
    dt:       float    = 0.05,
    verbose:  bool     = True,
    obs_fns:  dict     = None,   # {name: fn(psi)->scalar} computed on-the-fly
) -> List[TrotterResult]:
    """Evolve psi0 through all output times using Trotter steps of size dt.

    obs_fns: if provided, observables are computed at each time point and psi
    is NOT stored in the result — prevents N-states × n_times VRAM accumulation.
    E.g. obs_fns={'entropy': lambda p: entanglement_entropy(p, n), 'imb': ...}

    Without obs_fns: stores psi.copy() at each time point (original behaviour,
    but OOMs for large N with many output times).
    """
    xp       = cp.get_array_module(psi0) if cp is not None else np
    times    = np.sort(np.asarray(times, dtype=float))
    psi      = psi0.copy() if xp.iscomplexobj(psi0) else psi0.copy().astype(
                   xp.complex64 if psi0.dtype == xp.float32 else xp.complex128)
    n_qubits = int(np.log2(len(diag)))
    t_now    = 0.0
    results  = []
    t_wall0  = time.perf_counter()

    phase_dt     = _make_phase(diag, dt, psi.dtype)
    _phase_cache = {dt: phase_dt}

    def get_phase(step):
        if step not in _phase_cache:
            _phase_cache[step] = _make_phase(diag, step, psi.dtype)
        return _phase_cache[step]

    for t_target in times:
        t_step_start = time.perf_counter()
        n_steps      = 0

        while t_now < t_target - 1e-12:
            step  = min(dt, t_target - t_now)
            phase = get_phase(round(step, 12))
            psi   = trotter_step(psi, phase, Gamma, step, n_qubits)
            t_now += step
            n_steps += 1

        # Store observables on-the-fly (no psi copy) or full psi (small N only)
        if obs_fns is not None:
            obs_vals = {k: float(fn(psi)) for k, fn in obs_fns.items()}
            stored_psi = None
        else:
            obs_vals   = {}
            stored_psi = psi.copy()

        results.append(TrotterResult(
            psi       = stored_psi,
            t         = t_target,
            wall_time = time.perf_counter() - t_step_start,
            n_steps   = n_steps,
            obs       = obs_vals,
        ))

        if verbose:
            print(f'  t={t_target:.2f}  steps={n_steps}'
                  f'  ({results[-1].wall_time*1000:.0f} ms)', flush=True)

    if verbose:
        print(f'  Trajectory done: {len(times)} points in '
              f'{time.perf_counter()-t_wall0:.1f}s')

    return results
