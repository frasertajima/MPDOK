"""
acoustic_solver.py — Multi-obstacle BEM Helmholtz solver for the interactive lab.

Multiple scatterers: all panels concatenated into one system; off-diagonal
Green's function blocks give the exact acoustic coupling between obstacles.

Solver paths:
  scipy   — scipy.linalg.solve on complex N×N system
  fortran — LUIRSolver: real (2N)×(2N) block system, TF32 LU + FP64 refinement
"""

import sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve as scipy_solve
from matplotlib.path import Path as MplPath

_ACDIR = Path(__file__).parent.parent / "acoustic_scattering"
_MPDOK = Path(__file__).parent.parent
sys.path.insert(0, str(_ACDIR))
sys.path.insert(0, str(_MPDOK))

from bem_helmholtz import (
    build_bem_matrix_helmholtz,
    make_rhs_helmholtz,
    eval_total_field,
    eval_total_field_gpu,
    to_block_real,
    rhs_to_real,
    sigma_from_real,
)
from geometry import circle_panels, ellipse_panels, joukowski_panels, submarine_panels

# ── GPU ───────────────────────────────────────────────────────────────────────

try:
    import cupy as cp
    HAS_GPU = True
    print("[acoustic_solver] GPU: enabled")
except ImportError:
    HAS_GPU = False
    print("[acoustic_solver] GPU: disabled")

# ── Fortran LU-IR ─────────────────────────────────────────────────────────────

try:
    from mpdok_ops import LUIRSolver
    _luir = LUIRSolver()
    HAS_FORTRAN = True
    print("[acoustic_solver] Fortran LU-IR: enabled")
except Exception as _e:
    HAS_FORTRAN = False
    _luir = None
    print(f"[acoustic_solver] Fortran LU-IR: disabled ({_e})")

# ── Evaluation grid ───────────────────────────────────────────────────────────

NX, NY = 256, 256
DOMAIN = 6.0

_xs = np.linspace(-DOMAIN, DOMAIN, NX)
_ys = np.linspace(-DOMAIN, DOMAIN, NY)
_XX, _YY = np.meshgrid(_xs, _ys)
GRID_PTS = np.stack([_XX.ravel(), _YY.ravel()], axis=1)  # (NX*NY, 2)

# Pre-compute incident-wave phase factor per grid point: x·d for all directions
# (reused every solve without reallocation)
_GRID_X = _XX.ravel().astype(np.float64)  # (NX*NY,)
_GRID_Y = _YY.ravel().astype(np.float64)


# ── Panel generators ──────────────────────────────────────────────────────────

def _make_one(shape_type, n, cx, cy, p):
    if shape_type == "ellipse":
        nodes, normals, lengths = ellipse_panels(n, a=p.get("a", 1.5), b=p.get("b", 0.8))
    elif shape_type == "circle":
        nodes, normals, lengths = circle_panels(n, R=p.get("r", 1.0))
    elif shape_type == "joukowski":
        nodes, normals, lengths = joukowski_panels(n, eps=p.get("eps", 0.12))
    elif shape_type == "submarine":
        nodes, normals, lengths = submarine_panels(n, L=p.get("L", 2.5), R=p.get("R", 0.6))
    elif shape_type == "rect":
        nodes, normals, lengths = _rect_panels(n, w=p.get("w", 2.0), h=p.get("h", 1.2))
    else:
        nodes, normals, lengths = ellipse_panels(n)
    nodes = nodes + np.array([cx, cy])
    return nodes, normals, lengths


def _rect_panels(N, w=2.0, h=1.2):
    perim = 2 * (w + h)
    ds = perim / N
    nodes = np.zeros((N, 2));  normals = np.zeros((N, 2));  lengths = np.full(N, ds)
    for i, s in enumerate((np.arange(N) + 0.5) * ds):
        if s < w:
            nodes[i]=[s-w/2, -h/2];            normals[i]=[0,-1]
        elif s < w+h:
            nodes[i]=[w/2, s-w-h/2];           normals[i]=[1,0]
        elif s < 2*w+h:
            nodes[i]=[w/2-(s-w-h), h/2];       normals[i]=[0,1]
        else:
            nodes[i]=[-w/2, h/2-(s-2*w-h)];    normals[i]=[-1,0]
    return nodes, normals, lengths


def _rect_polygon(cx, cy, w, h):
    """Return the 4 corner vertices of a rectangle (closed polygon for masking)."""
    return np.array([
        [cx - w/2, cy - h/2],
        [cx + w/2, cy - h/2],
        [cx + w/2, cy + h/2],
        [cx - w/2, cy + h/2],
    ])


def _make_all_panels(shapes, n_each):
    all_nodes, all_lengths, per_shape, mask_polys = [], [], [], []
    for sh in shapes:
        nodes, _, lengths = _make_one(sh["type"], n_each, sh["cx"], sh["cy"], sh.get("params", {}))
        all_nodes.append(nodes)
        all_lengths.append(lengths)
        per_shape.append(nodes)
        # Exact polygon for interior masking (rectangles need corners, not midpoints)
        p = sh.get("params", {})
        if sh["type"] == "rect":
            mask_polys.append(_rect_polygon(sh["cx"], sh["cy"], p.get("w", 2.0), p.get("h", 1.2)))
        else:
            mask_polys.append(nodes)
    return np.concatenate(all_nodes), np.concatenate(all_lengths), per_shape, mask_polys


def _mask_multi(grid_pts, mask_polys, p_flat):
    p_out = p_flat.copy()
    for poly in mask_polys:
        p_out[MplPath(poly).contains_points(grid_pts)] = np.nan
    return p_out


# ── Power metrics ─────────────────────────────────────────────────────────────

def _incident_field(k, alpha):
    """Return (p_inc_re, p_inc_im) both (NY,NX) float32."""
    phase = k * (_GRID_X * np.cos(alpha) + _GRID_Y * np.sin(alpha))
    return (np.cos(phase).reshape(NY, NX).astype(np.float32),
            np.sin(phase).reshape(NY, NX).astype(np.float32))


def compute_power_metrics(p_re, p_im, mask, k, alpha):
    """
    Returns (field_power, scattered_power), both normalised by mean |p_inc|².

    field_power:
        ||p_total||² / ||p_inc||²   (exterior mean)
        0 = dead zone / full destructive cancellation  ← what the user finds by dragging
        1 = undisturbed field (obstacle invisible)
       >1 = amplification / resonance

    scattered_power:
        ||p_total − p_inc||² / ||p_inc||²   (exterior mean)
        0 = acoustic cloak (obstacle undetectable)
        1 = typical strong scatterer
       >1 = resonant amplification
    """
    p_inc_re, p_inc_im = _incident_field(k, alpha)
    ext = mask == 0
    n   = ext.sum()
    if n == 0:
        return 1.0, 1.0

    inc  = float(np.mean(p_inc_re[ext]**2 + p_inc_im[ext]**2))
    norm = max(inc, 1e-12)

    field   = float(np.mean(p_re[ext]**2 + p_im[ext]**2)) / norm
    ps_re   = (p_re - p_inc_re)[ext]
    ps_im   = (p_im - p_inc_im)[ext]
    scatter = float(np.mean(ps_re**2 + ps_im**2)) / norm

    # Clamp: values >> 100 indicate an irregular-frequency blow-up, not physics
    field   = min(field,   200.0)
    scatter = min(scatter, 200.0)

    return field, scatter


# Keep old name for any callers
def compute_scattered_power(p_re, p_im, mask, k, alpha):
    _, sp = compute_power_metrics(p_re, p_im, mask, k, alpha)
    return sp


# ── Recommended panel count ───────────────────────────────────────────────────

def recommended_n_panels(k, shapes):
    max_perim = 0.0
    for sh in shapes:
        p = sh.get("params", {})
        t = sh["type"]
        if t == "ellipse":
            a, b = p.get("a",1.5), p.get("b",0.8)
            perim = np.pi*(3*(a+b)-np.sqrt((3*a+b)*(a+3*b)))
        elif t == "circle":   perim = 2*np.pi*p.get("r",1.0)
        elif t == "submarine": perim = 2*p.get("L",2.5)+2*np.pi*p.get("R",0.6)
        elif t == "rect":     perim = 2*(p.get("w",2.0)+p.get("h",1.2))
        else:                 perim = 8.0
        max_perim = max(max_perim, perim)
    return max(60, int(np.ceil(10 * k * max_perim / (2*np.pi))))


# ── Core solve ────────────────────────────────────────────────────────────────

class AcousticSolver:

    def solve(self, shapes, n_panels, k, alpha, solver_type="scipy", use_gpu=True):
        """
        Returns
        -------
        p_re, p_im      : (NY,NX) float32
        mask            : (NY,NX) uint8
        per_shape_nodes : list of (N,2) float32 arrays
        n_rec           : int
        solver_used     : str
        scattered_power : float  (0=cloak, ~1=normal, >1=resonance)
        """
        if not shapes:
            z = np.zeros((NY, NX), dtype=np.float32)
            return z, z, z.astype(np.uint8), [], 100, "scipy", 1.0, 1.0

        nodes, lengths, per_shape, mask_polys = _make_all_panels(shapes, n_panels)

        A = build_bem_matrix_helmholtz(nodes, lengths, k)
        b_complex, _ = make_rhs_helmholtz(nodes, k, alpha)

        # Sanitize: overlapping shapes can produce R=0 off-diagonal → H₀(0)=Inf.
        # Zero those entries (two coincident panels have no net contribution).
        if not np.isfinite(A).all():
            np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # Tikhonov regularization: suppresses blow-up at BEM irregular frequencies
        # (interior Dirichlet eigenvalues where the single-layer matrix is nearly singular).
        # eps ~ 1e-4 * diag_scale leaves normal solutions essentially unchanged.
        diag_scale = float(np.mean(np.abs(np.diag(A))))
        if not np.isfinite(diag_scale) or diag_scale == 0.0:
            diag_scale = 0.1
        A = A + (diag_scale * 1e-4) * np.eye(len(A))

        solver_used = solver_type
        if solver_type == "fortran" and HAS_FORTRAN:
            x_gpu = _luir.solve(to_block_real(A), rhs_to_real(b_complex))
            sigma  = sigma_from_real(cp.asnumpy(x_gpu))
        else:
            if solver_type == "fortran":
                solver_used = "scipy (fallback)"
            sigma = scipy_solve(A, b_complex)

        eval_fn = eval_total_field_gpu if (use_gpu and HAS_GPU) else eval_total_field
        p_flat  = eval_fn(nodes, lengths, sigma, GRID_PTS, k, alpha)

        # Sanitize: blow-up σ from residual irregular-freq issues can propagate to field
        if not np.isfinite(p_flat).all():
            np.nan_to_num(p_flat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        p_masked = _mask_multi(GRID_PTS, mask_polys, p_flat)

        interior = np.isnan(p_masked).reshape(NY, NX).astype(np.uint8)
        p_grid   = p_masked.reshape(NY, NX)
        p_re = np.nan_to_num(p_grid.real, nan=0.0).astype(np.float32)
        p_im = np.nan_to_num(p_grid.imag, nan=0.0).astype(np.float32)

        field_power, scat_power = compute_power_metrics(p_re, p_im, interior, k, alpha)
        bpts  = [n.astype(np.float32) for n in per_shape]
        n_rec = recommended_n_panels(k, shapes)

        return p_re, p_im, interior, bpts, n_rec, solver_used, field_power, scat_power
