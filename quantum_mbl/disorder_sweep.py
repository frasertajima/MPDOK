"""
Disorder-averaged MBL sweep engine — Trotter only.

run_sweep_trotter(xp=cp)  — RTX 4060: GPU Trotter, N=24, ~minutes
run_sweep_trotter(xp=np)  — ThinkPad: CPU Trotter, N=28, overnight

Trotter needs only 2 state-vector-sized allocations regardless of N.
Krylov was dropped because the Krylov basis V at N=24 requires >5 GB VRAM
on top of the notebook's existing allocations.

Both paths checkpoint after every (W, realization) — safe to interrupt.
"""

import os
import time

import numpy as np

try:
    import cupy as cp
    _CUPY = True
except ImportError:
    cp    = None
    _CUPY = False

import sys
sys.path.insert(0, '/var/home/fraser/machine_learning/fortran/examples/'
                   'collected_examples/matrix_dot/tensor13/'
                   'tensor_core_engine_v5')

from MPDOK.quantum_mbl.hamiltonian_mbl import build_mbl_diagonal, neel_state
from MPDOK.quantum_mbl.trotter         import evolve_trotter
from MPDOK.quantum_mbl.observables_mbl import entanglement_entropy, imbalance


def run_sweep_trotter(
    n_qubits:       int   = 24,
    W_values:       list  = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0),
    n_realizations: int   = 5,
    t_max:          float = 30.0,
    n_times:        int   = 40,
    dt:             float = 0.05,
    J:              float = 1.0,
    Gamma:          float = 0.5,
    checkpoint_dir: str   = '.',
    dtype                  = None,
    xp                     = None,   # np for CPU (N=28), cp for GPU (N=24)
    verbose:        bool   = True,
) -> dict:
    """Disorder-averaged Trotter sweep.

    xp=cp (default when CuPy available): GPU path, use for N=24 on RTX 4060.
    xp=np: CPU path, use for N=28 overnight on ThinkPad.

    dtype defaults to complex128 for GPU (fast), complex64 for CPU (saves RAM).
    Checkpoints after every (W, realization) — safe to interrupt and resume.
    """
    if xp is None:
        xp = cp if _CUPY else np

    is_gpu = _CUPY and xp is cp
    if dtype is None:
        # complex128 cuSolver SVD workspace OOMs at N≥26 on 8 GB VRAM; complex64
        # halves the state vector and workspace while preserving physics accuracy
        if is_gpu and n_qubits >= 26:
            dtype = xp.complex64
        else:
            dtype = xp.complex128 if is_gpu else np.complex64

    diag_dtype = xp.float64 if is_gpu else np.float32

    times = np.linspace(0, t_max, n_times)
    n_W   = len(W_values)
    avg_entropy   = np.zeros((n_W, n_times))
    avg_imbalance = np.zeros((n_W, n_times))

    N = 1 << n_qubits
    tag = 'GPU' if is_gpu else 'CPU'
    print(f'[{tag}] N={n_qubits}  states={N:,}  '
          f'dtype={dtype}  '
          f'state_vec={N * 16 / 1e9:.2f} GB  '
          f'peak_alloc≈{N * 16 * 2 / 1e9:.2f} GB')

    eta_step = None

    for wi, W in enumerate(W_values):
        ent_sum = np.zeros(n_times)
        imb_sum = np.zeros(n_times)
        t_W     = time.perf_counter()

        for r_idx in range(n_realizations):
            ckpt = os.path.join(checkpoint_dir,
                                f'trotter_N{n_qubits}_W{W:.1f}_r{r_idx}.npz')
            if os.path.exists(ckpt):
                d = np.load(ckpt)
                ent_sum += d['entropy']
                imb_sum += d['imbalance']
                if verbose:
                    print(f'  W={W:.1f} r={r_idx}  [checkpoint]')
                continue

            seed = wi * 1000 + r_idx
            diag, _ = build_mbl_diagonal(n_qubits, J=J, W=W, seed=seed,
                                          xp=xp, dtype=diag_dtype)
            psi0 = neel_state(n_qubits, xp=xp, dtype=dtype)

            if verbose:
                eta_str = (f'  ETA ~{eta_step * n_times / 3600:.1f}h'
                           if eta_step else '')
                print(f'  W={W:.1f}  r={r_idx+1}/{n_realizations}{eta_str}',
                      flush=True)

            # obs_fns: compute observables on-the-fly, never store 40×psi copies
            obs_fns = {
                'entropy':   lambda p: entanglement_entropy(p, n_qubits),
                'imbalance': lambda p: imbalance(p, n_qubits),
            }
            t0      = time.perf_counter()
            traj    = evolve_trotter(psi0, diag, Gamma, times,
                                     dt=dt, verbose=False, obs_fns=obs_fns)
            elapsed = time.perf_counter() - t0
            eta_step = elapsed

            ent_arr = np.array([r.obs['entropy']   for r in traj])
            imb_arr = np.array([r.obs['imbalance'] for r in traj])

            ent_sum += ent_arr
            imb_sum += imb_arr

            np.savez(ckpt, entropy=ent_arr, imbalance=imb_arr,
                     times=times, W=W, r_idx=r_idx)

            # Free GPU pool between realizations
            if is_gpu:
                cp.get_default_memory_pool().free_all_blocks()

            if verbose:
                print(f'    {elapsed:.1f}s  '
                      f'S(t_max)={ent_arr[-1]:.3f}  '
                      f'I(t_max)={imb_arr[-1]:.3f}', flush=True)

        avg_entropy[wi]   = ent_sum / n_realizations
        avg_imbalance[wi] = imb_sum / n_realizations
        if verbose:
            elapsed_W = time.perf_counter() - t_W
            print(f'  W={W:.1f}  avg S={avg_entropy[wi,-1]:.3f}'
                  f'  avg I={avg_imbalance[wi,-1]:.3f}'
                  f'  ({elapsed_W:.1f}s)', flush=True)

    return {
        'W_values':       np.array(W_values),
        'times':          times,
        'avg_entropy':    avg_entropy,
        'avg_imbalance':  avg_imbalance,
        'n_qubits':       n_qubits,
        'n_realizations': n_realizations,
    }
