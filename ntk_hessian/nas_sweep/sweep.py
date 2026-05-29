"""
NAS sweep engine: train MnistMLP across (depth, width) grid.

All training is done on GPU via PyTorch.  Results are returned as a plain
dataclass so the notebook can do whatever it likes with them.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse data utilities from ntk_hessian sibling package
sys.path.insert(0, '/var/home/fraser/machine_learning/fortran/examples/'
                   'collected_examples/matrix_dot/tensor13/'
                   'tensor_core_engine_v5')

from MPDOK.ntk_hessian.models import MnistMLP, load_mnist


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    depth:    int
    width:    int
    lr:       float
    hidden:   Tuple[int, ...]
    epochs:   int
    val_acc:  float          # best val accuracy across epochs
    final_acc: float         # last-epoch val accuracy
    train_time: float        # wall seconds for all epochs
    history:  List[dict] = field(default_factory=list)
    num_params: int = 0


# ── core training function ────────────────────────────────────────────────────

def _train_one(hidden: tuple, epochs: int, lr: float,
               device: str, X_tr, y_tr, X_va, y_va,
               batch_size: int = 256) -> RunResult:
    model = MnistMLP(hidden=hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N     = len(X_tr)
    history = []
    best_acc = 0.0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        perm       = torch.randperm(N, device=device)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx    = perm[i:i + batch_size]
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
        best_acc = max(best_acc, val_acc)
        history.append({'epoch': epoch + 1,
                        'loss':  epoch_loss / N,
                        'val_acc': val_acc})

    elapsed = time.perf_counter() - t0
    return RunResult(
        depth=len(hidden),
        width=hidden[0],          # first hidden width as representative
        lr=lr,
        hidden=hidden,
        epochs=epochs,
        val_acc=best_acc,
        final_acc=history[-1]['val_acc'],
        train_time=elapsed,
        history=history,
        num_params=model.num_params,
    )


# ── grid sweep ────────────────────────────────────────────────────────────────

def grid_sweep(
    depths: List[int]  = (2, 3, 4, 5),
    widths: List[int]  = (64, 128, 256, 512, 768, 1024),
    epochs: int        = 10,
    lr:     float      = 1e-3,
    device: str        = 'cuda',
    verbose: bool      = True,
) -> List[RunResult]:
    """Train every (depth, width) combination.

    hidden layers are uniform: e.g. depth=3, width=256 → [256, 256, 256].
    Returns list of RunResult in the order trained.
    """
    X_tr, y_tr = load_mnist('train', device=device)
    X_va, y_va = load_mnist('val',   device=device)

    total   = len(depths) * len(widths)
    results = []
    t_sweep = time.perf_counter()

    for i, d in enumerate(depths):
        for j, w in enumerate(widths):
            hidden = tuple([w] * d)
            n      = i * len(widths) + j + 1
            if verbose:
                print(f'  [{n:>3}/{total}]  depth={d}  width={w:>4}  '
                      f'hidden={hidden}', flush=True)
            r = _train_one(hidden, epochs, lr, device,
                           X_tr, y_tr, X_va, y_va)
            results.append(r)
            if verbose:
                print(f'          val_acc={r.val_acc:.4f}  '
                      f'time={r.train_time:.1f}s  '
                      f'params={r.num_params:,}', flush=True)

    if verbose:
        elapsed = time.perf_counter() - t_sweep
        best    = max(results, key=lambda r: r.val_acc)
        print(f'\nSweep done in {elapsed:.1f}s  '
              f'({elapsed/60:.1f} min)')
        print(f'Best: depth={best.depth}  width={best.width}  '
              f'val_acc={best.val_acc:.4f}  hidden={best.hidden}')

    return results


# ── LR × depth × width sweep ─────────────────────────────────────────────────

def lr_sweep(
    depths: List[int]  = (2, 3, 4),
    widths: List[int]  = (128, 256, 512, 1024),
    lrs:    List[float] = (3e-4, 1e-3, 3e-3),
    epochs: int        = 10,
    device: str        = 'cuda',
    verbose: bool      = True,
) -> List[RunResult]:
    """Triple sweep over LR, depth, width."""
    X_tr, y_tr = load_mnist('train', device=device)
    X_va, y_va = load_mnist('val',   device=device)

    configs = [(d, w, lr) for lr in lrs for d in depths for w in widths]
    results = []
    t0      = time.perf_counter()

    for n, (d, w, lr) in enumerate(configs, 1):
        hidden = tuple([w] * d)
        if verbose:
            print(f'  [{n:>3}/{len(configs)}]  depth={d}  width={w}  '
                  f'lr={lr:.0e}', flush=True)
        r = _train_one(hidden, epochs, lr, device, X_tr, y_tr, X_va, y_va)
        results.append(r)
        if verbose:
            print(f'          val_acc={r.val_acc:.4f}  '
                  f'time={r.train_time:.1f}s', flush=True)

    if verbose:
        print(f'\nLR sweep done in {(time.perf_counter()-t0)/60:.1f} min')
    return results


# ── top-N head-to-head ────────────────────────────────────────────────────────

def top_n_extended(
    results: List[RunResult],
    n:       int   = 3,
    epochs:  int   = 30,
    device:  str   = 'cuda',
    verbose: bool  = True,
) -> List[RunResult]:
    """Re-train the top-n architectures for more epochs."""
    ranked = sorted(results, key=lambda r: r.val_acc, reverse=True)[:n]
    X_tr, y_tr = load_mnist('train', device=device)
    X_va, y_va = load_mnist('val',   device=device)
    extended   = []

    for i, r in enumerate(ranked, 1):
        if verbose:
            print(f'  [{i}/{n}]  hidden={r.hidden}  '
                  f'(quick acc={r.val_acc:.4f}) → {epochs} epochs …',
                  flush=True)
        r2 = _train_one(r.hidden, epochs, r.lr, device,
                        X_tr, y_tr, X_va, y_va)
        extended.append(r2)
        if verbose:
            print(f'          val_acc={r2.val_acc:.4f}  '
                  f'time={r2.train_time:.1f}s', flush=True)

    return extended
