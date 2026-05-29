"""
Neural Tangent Kernel (NTK) and feature kernel construction.

Two approaches:

1. Feature kernel (fast, any N):
       K(x_i, x_j) = φ(x_i)·φ(x_j) / D
   where φ = model.features() are the penultimate activations.
   This is the 'lazy' or 'NNGP-at-output' kernel.  One forward pass total.
   N=20,000 with D=256 → K is 3.2 GB FP64 — MPDOK handles it, SciPy OOMs.

2. Full empirical NTK (exact, small N ≤ 3000):
       K(x_i, x_j) = ∑_c (∂f_c/∂θ)(x_i) · (∂f_c/∂θ)(x_j)
   Requires N backward passes (or batched via torch.func.vmap + jacrev).
   J matrix is (N, C·P) — only built in chunks, never fully stored.

The MPDOK contribution is in the SOLVE step (ntk_solver.py), not the
build step.  The kernel itself is SciPy-compatible if N is small.

Usage:
    K = build_feature_kernel(model, X_np, nugget=1e-4)  # (N,N) CuPy FP64
    K = build_full_ntk(model, X_np, device='cuda')       # small N only
"""

import gc
import time

import cupy as cp
import numpy as np
import torch
import torch.nn.functional as F


# ── feature kernel ─────────────────────────────────────────────────────────────

def build_feature_kernel(model, X_np, device='cuda', chunk=512,
                          nugget=1e-2, normalize=True, verbose=True):
    """Build N×N feature kernel K = Φ̂ Φ̂^T in CuPy.

    Φ is extracted in CPU chunks (trivial VRAM usage).  By default features
    are L2-normalised (cosine similarity kernel) so that:
      - All diagonal entries are exactly 1 + nugget.
      - Max eigenvalue ≤ N; condition number ≈ N/10 for class-clustered data.
      - LU-IR converges reliably with nugget=1e-2.

    Without normalisation (normalize=False) the kernel is K = Φ Φ^T / D.
    This has max eigenvalue ≈ N·D/10 and condition number ≈ 10^7 for large N,
    which causes LU-IR to diverge — use a much larger nugget (≥1.0) in that case.

    Args:
        model     : MnistMLP (or any model with .features() method).
        X_np      : (N, 784) float32 numpy array.
        device    : torch device for forward pass.
        chunk     : batch size for feature extraction and GEMM.
        nugget    : diagonal regularisation λ (default 1e-2 for LU-IR stability).
        normalize : L2-normalise feature rows before building K (default True).

    Returns:
        K  : (N, N) CuPy float64 array.
        t  : wall-clock build time.
    """
    t0 = time.perf_counter()
    model.eval()
    N = len(X_np)
    X_torch = torch.from_numpy(X_np).float()

    # ── Step 1: extract features → CPU (avoid VRAM pressure) ────────────────
    if verbose:
        print(f'  Extracting features for N={N:,} …', flush=True)
    Phi_parts = []
    with torch.no_grad():
        for i in range(0, N, chunk):
            x_b = X_torch[i:i + chunk].to(device)
            phi = model.features(x_b).cpu().numpy()
            Phi_parts.append(phi)
    Phi_np = np.vstack(Phi_parts).astype(np.float32)   # (N, D)
    D = Phi_np.shape[1]

    if normalize:
        norms = np.linalg.norm(Phi_np, axis=1, keepdims=True).clip(1e-12)
        Phi_np = Phi_np / norms   # unit-norm rows → cosine similarity kernel

    # ── Step 2: K = Phi @ Phi^T via chunked cuBLAS GEMM ─────────────────────
    if verbose:
        label = 'cosine' if normalize else f'Φ Φ^T/D (D={D})'
        print(f'  Building {N:,}×{N:,} kernel ({label}) … '
              f'[{N*N*8/1e9:.2f} GB FP64]', flush=True)

    # Fortran order: no copy needed when LUIRSolver._solve() calls asfortranarray
    K   = cp.empty((N, N), dtype=cp.float64, order='F')
    Phi = cp.asarray(Phi_np)   # (N, D) on GPU — D=256 so very small

    scale = 1.0 if normalize else (1.0 / D)
    for i in range(0, N, chunk):
        end   = min(i + chunk, N)
        Phi_i = Phi[i:end].astype(cp.float64)          # (chunk, D)
        row   = Phi_i @ Phi.T.astype(cp.float64)       # (chunk, N)
        K[i:end, :] = row * scale

    idx = cp.arange(N)
    K[idx, idx] += nugget
    cp.cuda.Stream.null.synchronize()

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f'  Kernel built in {elapsed:.2f}s')
    return K, elapsed


def build_feature_kernel_cpu(model, X_np, device='cuda', chunk=512,
                              nugget=1e-2, normalize=True, verbose=True):
    """Build N×N feature kernel as a NumPy array — no large VRAM allocation.

    Features are extracted on GPU in small chunks (D=256 per sample, trivial VRAM).
    K = Phi @ Phi.T is computed on CPU in NumPy so the full N×N matrix stays in RAM.
    This is the correct path for the SciPy backend — it never fills VRAM with K.
    """
    t0 = time.perf_counter()
    model.eval()
    N = len(X_np)
    X_torch = torch.from_numpy(X_np).float()

    # ── extract features on GPU, collect on CPU ──────────────────────────────
    if verbose:
        print(f'  Extracting features for N={N:,} …', flush=True)
    Phi_parts = []
    with torch.no_grad():
        for i in range(0, N, chunk):
            x_b = X_torch[i:i + chunk].to(device)
            phi = model.features(x_b).cpu().numpy()
            Phi_parts.append(phi)
    Phi = np.vstack(Phi_parts).astype(np.float64)   # (N, D), all on CPU

    if normalize:
        norms = np.linalg.norm(Phi, axis=1, keepdims=True).clip(1e-12)
        Phi   = Phi / norms

    # ── K = Phi @ Phi.T on CPU (no VRAM pressure) ────────────────────────────
    if verbose:
        label = 'cosine' if normalize else f'Φ Φ^T/D'
        print(f'  Building {N:,}×{N:,} kernel ({label}) on CPU … '
              f'[{N*N*8/1e9:.2f} GB]', flush=True)

    scale = 1.0 if normalize else (1.0 / Phi.shape[1])
    K     = (Phi @ Phi.T) * scale
    np.fill_diagonal(K, K.diagonal() + nugget)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f'  Kernel built in {elapsed:.2f}s')
    return K, elapsed


# ── full empirical NTK (small N) ──────────────────────────────────────────────

def build_full_ntk(model, X_np, device='cuda', chunk=64,
                   nugget=1e-6, verbose=True):
    """Full empirical NTK via per-sample Jacobians.  Use only for N ≤ 3000.

    K[i,j] = ∑_c ∇_θ f_c(x_i) · ∇_θ f_c(x_j)

    Jacobians computed via autograd over the model output.
    Returns (N, N) CuPy float64 array.
    """
    from torch.func import vmap, jacrev, functional_call

    t0 = time.perf_counter()
    model.eval()
    N = len(X_np)
    P = sum(p.numel() for p in model.parameters())
    C = 10   # MNIST classes
    if verbose:
        print(f'  Full NTK: N={N:,}  P={P:,}  C={C}', flush=True)

    params  = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def fwd(params, buffers, x):
        return functional_call(model, (params, buffers), (x.unsqueeze(0),)).squeeze(0)

    jac_fn = vmap(jacrev(fwd, argnums=0), in_dims=(None, None, 0))

    X_torch = torch.from_numpy(X_np).float().to(device)
    K_cp = cp.zeros((N, N), dtype=cp.float64)

    # Process in chunks of rows, compute Jacobians, accumulate K block
    J_chunks = []   # list of (chunk, C*P) numpy arrays
    for i in range(0, N, chunk):
        end  = min(i + chunk, N)
        x_b  = X_torch[i:end]
        jac  = jac_fn(params, buffers, x_b)   # dict param_name -> (B, C, *shape)
        # Flatten all parameter Jacobians and classes into (B, C*P)
        jac_flat = torch.cat([v.reshape(end - i, C, -1)
                              for v in jac.values()], dim=2)
        jac_flat = jac_flat.reshape(end - i, C * P).cpu().numpy().astype(np.float32)
        J_chunks.append((i, end, jac_flat))
        if verbose:
            print(f'    Jacobian rows {i:>5}–{end:>5} / {N}', end='\r', flush=True)

    if verbose:
        print()

    # Build K = J @ J^T / C in CuPy
    for (i, ei, Ji) in J_chunks:
        Ji_gpu = cp.asarray(Ji).astype(cp.float64)   # (bi, C*P)
        for (j, ej, Jj) in J_chunks:
            Jj_gpu = cp.asarray(Jj).astype(cp.float64)
            K_cp[i:ei, j:ej] = Ji_gpu @ Jj_gpu.T / C

    idx = cp.arange(N)
    K_cp[idx, idx] += nugget

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f'  Full NTK built in {elapsed:.1f}s')
    return K_cp, elapsed


# ── kernel regression utilities ───────────────────────────────────────────────

def make_one_hot(y_np, n_classes=10):
    """Convert integer labels to (N, C) float64 one-hot array."""
    Y = np.zeros((len(y_np), n_classes), dtype=np.float64)
    Y[np.arange(len(y_np)), y_np] = 1.0
    return Y


def predict_kernel(alpha, model, X_obs_np, X_pred_np,
                   device='cuda', chunk=512, normalize=True, D=None):
    """Kernel regression prediction: ŷ = K*(X_pred, X_obs) @ alpha.

    alpha: (N_obs, C) weights from the kernel solve.
    Returns: (N_pred, C) predictions (raw scores, argmax for class).
    """
    model.eval()
    N_obs  = len(X_obs_np)
    N_pred = len(X_pred_np)
    C = alpha.shape[1] if alpha.ndim == 2 else 1
    alpha_gpu = cp.asarray(alpha)   # (N_obs, C)

    X_obs_torch  = torch.from_numpy(X_obs_np).float()
    X_pred_torch = torch.from_numpy(X_pred_np).float()

    with torch.no_grad():
        Phi_obs_parts = []
        for i in range(0, N_obs, chunk):
            Phi_obs_parts.append(
                model.features(X_obs_torch[i:i+chunk].to(device)).cpu().numpy())
        Phi_obs = np.vstack(Phi_obs_parts).astype(np.float32)  # (N_obs, D)
        if normalize:
            norms = np.linalg.norm(Phi_obs, axis=1, keepdims=True).clip(1e-12)
            Phi_obs = Phi_obs / norms
        D_feat  = Phi_obs.shape[1]
        scale   = 1.0 if normalize else (1.0 / D_feat)

        scores = np.zeros((N_pred, C), dtype=np.float64)
        Phi_obs_gpu = cp.asarray(Phi_obs).astype(cp.float64)

        for i in range(0, N_pred, chunk):
            end      = min(i + chunk, N_pred)
            x_b      = X_pred_torch[i:end].to(device)
            phi_pred = model.features(x_b).cpu().numpy().astype(np.float32)
            if normalize:
                n2 = np.linalg.norm(phi_pred, axis=1, keepdims=True).clip(1e-12)
                phi_pred = phi_pred / n2
            Kstar    = (cp.asarray(phi_pred).astype(cp.float64)
                        @ Phi_obs_gpu.T) * scale  # (chunk, N_obs)
            scores[i:end] = cp.asnumpy(Kstar @ alpha_gpu)

    return scores
