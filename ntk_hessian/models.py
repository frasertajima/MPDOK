"""
MNIST MLP models and data utilities for the NTK/Hessian demo.

Data: /var/home/fraser/machine_learning/data/mnist/mnist.pkl.gz
Format: (train, val, test) each a (X, y) tuple;
        X is (N, 784) float32, y is (N,) int32 labels 0-9.
"""

import gzip
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_PATH = '/var/home/fraser/machine_learning/data/mnist/mnist.pkl.gz'
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')


# ── model ─────────────────────────────────────────────────────────────────────

class MnistMLP(nn.Module):
    """Configurable MLP for MNIST classification.

    Separates backbone (all layers except final) from head (linear classifier)
    so that backbone(x) returns penultimate activations for NTK construction.

    Default hidden=[512, 256] → 535,818 parameters.
    Tiny  hidden=[128,  64] → 109,386 parameters.
    """

    def __init__(self, hidden=(512, 256), activation=nn.ReLU):
        super().__init__()
        dims = [784] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Linear(dims[-1], 10)
        self._D       = dims[-1]   # penultimate dimension

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def features(self, x):
        """Return penultimate activations φ(x) of shape (N, D)."""
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.backbone(x))


# ── data loading ──────────────────────────────────────────────────────────────

def load_mnist(split='train', max_n=None, seed=42,
               path=MNIST_PATH, device='cpu'):
    """Load MNIST from pkl.gz, return (X, y) torch tensors on device.

    split: 'train' (50k), 'val' (10k), or 'test' (10k).
    max_n: if set, subsample deterministically.
    """
    with gzip.open(path, 'rb') as f:
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = pickle.load(f, encoding='latin1')

    splits = {'train': (X_tr, y_tr), 'val': (X_va, y_va), 'test': (X_te, y_te)}
    X_np, y_np = splits[split]

    if max_n and max_n < len(X_np):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_np), max_n, replace=False)
        idx.sort()
        X_np, y_np = X_np[idx], y_np[idx]

    X = torch.from_numpy(X_np).float().to(device)
    y = torch.from_numpy(y_np).long().to(device)
    return X, y


# ── training ──────────────────────────────────────────────────────────────────

def train_mlp(model, device='cuda', epochs=8, lr=1e-3, batch_size=256,
              verbose=True):
    """Train model on MNIST train split.  Returns training history."""
    X_tr, y_tr = load_mnist('train', device=device)
    X_va, y_va = load_mnist('val',   device=device)

    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N     = len(X_tr)
    history = []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_tr[idx])
            loss   = F.cross_entropy(logits, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)

        sched.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_va).argmax(1) == y_va).float().mean().item()

        entry = {'epoch': epoch + 1, 'loss': epoch_loss / N, 'val_acc': val_acc}
        history.append(entry)
        if verbose:
            print(f'  epoch {epoch+1}/{epochs}  loss={entry["loss"]:.4f}  '
                  f'val_acc={val_acc:.4f}  ({time.perf_counter()-t0:.1f}s)')

    return history


def get_model(hidden=(512, 256), device='cuda', retrain=False, verbose=True):
    """Load cached weights or train fresh MnistMLP.

    Weights are cached in ntk_hessian/weights/ keyed by hidden dims.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    key  = '_'.join(map(str, hidden))
    path = os.path.join(WEIGHTS_DIR, f'mnist_mlp_{key}.pt')

    model = MnistMLP(hidden=hidden).to(device)

    if os.path.exists(path) and not retrain:
        model.load_state_dict(torch.load(path, map_location=device))
        if verbose:
            print(f'  Loaded cached weights from {path}')
    else:
        if verbose:
            print(f'  Training MLP {hidden} on MNIST …')
        train_mlp(model, device=device, verbose=verbose)
        torch.save(model.state_dict(), path)
        if verbose:
            print(f'  Saved weights to {path}')

    model.eval()
    val_acc = _eval_accuracy(model, device)
    if verbose:
        print(f'  Val accuracy: {val_acc:.2%}   P = {model.num_params:,} parameters')
    return model


def _eval_accuracy(model, device):
    X_va, y_va = load_mnist('val', device=device)
    with torch.no_grad():
        return (model(X_va).argmax(1) == y_va).float().mean().item()
