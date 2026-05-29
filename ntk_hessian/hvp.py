"""
Hessian-vector products (HVP) for MNIST MLPs via double backpropagation.

The Hessian H = ∇²_θ L(θ) for an N-parameter model is N×N — completely
uncomputable for any real network.  But the product H·v can be computed
in O(P) time and memory (two backward passes), making it the ideal oracle
for matrix-free eigensolvers (Lanczos, power iteration).

Usage:
    hvp_fn = make_hvp_fn(model, X_batch, y_batch, device)
    Hv = hvp_fn(v)   # v: (P,) torch tensor on device
"""

import torch
import torch.nn.functional as F


# ── parameter utilities ───────────────────────────────────────────────────────

def flatten_params(model):
    """Return all parameters as a single (P,) tensor (view when possible)."""
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def flat_grad(model, loss):
    """Compute ∂loss/∂θ and flatten to (P,) with create_graph=True."""
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True,
                                allow_unused=True)
    parts = []
    for g, p in zip(grads, model.parameters()):
        parts.append(g.reshape(-1) if g is not None
                     else torch.zeros_like(p).reshape(-1))
    return torch.cat(parts)


def flat_grad_no_graph(model, loss):
    """Compute ∂loss/∂θ, flatten to (P,), detach — no graph retained."""
    grads = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
    parts = []
    for g, p in zip(grads, model.parameters()):
        parts.append(g.detach().reshape(-1) if g is not None
                     else torch.zeros_like(p).reshape(-1))
    return torch.cat(parts)


# ── HVP factory ───────────────────────────────────────────────────────────────

def make_hvp_fn(model, X_batch, y_batch, device='cuda'):
    """Return a closure hvp(v) → H·v for the batch Hessian.

    The returned function accepts a (P,) torch float64 tensor on device and
    returns a (P,) float64 tensor.  All intermediate computation stays on GPU.

    The batch Hessian H_batch = ∂²L_batch/∂θ² approximates the full-data
    Hessian; larger batches give better curvature estimates but cost more.

    Implementation: R-operator / double backprop
        g  = ∇L (first backward, create_graph=True)
        gv = g · v (scalar)
        Hv = ∇(gv) (second backward, no graph needed)
    """
    X = X_batch.to(device).float()
    y = y_batch.to(device).long()
    P = sum(p.numel() for p in model.parameters())

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    def hvp(v):
        """v: (P,) torch tensor (float64 or float32) on device."""
        v32 = v.float()

        with torch.enable_grad():
            logits = model(X)
            loss   = F.cross_entropy(logits, y)
            g      = flat_grad(model, loss)            # (P,) with graph
            gv     = (g * v32).sum()                   # scalar
            Hv_list = torch.autograd.grad(gv, model.parameters())
            Hv     = torch.cat([h.reshape(-1) for h in Hv_list]).detach()

        return Hv.double()   # always return FP64 for Lanczos stability

    hvp.P = P
    hvp.device = device
    return hvp


# ── CPU baseline wrapper ──────────────────────────────────────────────────────

def make_hvp_fn_cpu(model_cpu, X_batch, y_batch):
    """Same HVP factory but forces CPU — for SciPy eigsh baseline."""
    return make_hvp_fn(model_cpu, X_batch, y_batch, device='cpu')


# ── gradient norm (sanity check) ──────────────────────────────────────────────

def gradient_norm(model, X_batch, y_batch, device='cuda'):
    """||∇L||₂ over the batch — useful to verify the model is at a minimum."""
    X = X_batch.to(device).float()
    y = y_batch.to(device).long()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    with torch.enable_grad():
        loss = F.cross_entropy(model(X), y)
        g    = flat_grad_no_graph(model, loss)
    return float(g.norm())
