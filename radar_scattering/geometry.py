"""
radar_scattering.geometry — 2D panel generators for radar target shapes.

Each generator returns (nodes, normals, lengths):
  nodes:    (N, 2) float64 — panel midpoint coordinates
  normals:  (N, 2) float64 — outward unit normals (CCW convention)
  lengths:  (N,)   float64 — panel arc lengths Δl

Shapes are centred at (or near) the origin.  Panels are ordered CCW so
outward normals point away from the interior.

Re-exports circle_panels from acoustic_scattering for convenience.
"""

import sys, os
import numpy as np

# Allow import from sibling package
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from acoustic_scattering.geometry import (
    circle_panels,
    ellipse_panels,
    joukowski_panels,
    submarine_panels,
    plot_panels,
)

__all__ = [
    'circle_panels', 'ellipse_panels', 'joukowski_panels', 'submarine_panels',
    'square_panels', 'diamond_panels', 'corner_reflector_panels', 'stealth_panels',
    'polygon_panels', 'plot_panels',
]


# ── Generic polygon panel generator ──────────────────────────────────────────

def polygon_panels(N, vertices):
    """N constant panels uniformly distributed around a closed polygon.

    Panels are placed at equal arc-length intervals.  Panel lengths are all
    Δl = perimeter / N (uniform).

    Args:
        N:        Number of panels.
        vertices: (M, 2) array — polygon vertices in CCW order.
                  The last vertex connects back to the first automatically.

    Returns:
        nodes:    (N, 2) panel midpoints.
        normals:  (N, 2) outward unit normals (CCW: rotate tangent 90° CW).
        lengths:  (N,)   panel arc lengths (all equal = perimeter/N).
    """
    verts = np.asarray(vertices, dtype=float)
    M     = len(verts)

    v_next       = np.roll(verts, -1, axis=0)         # (M, 2) next vertex
    edges        = v_next - verts                       # (M, 2) edge vectors
    edge_lengths = np.linalg.norm(edges, axis=1)       # (M,) edge lengths
    perimeter    = edge_lengths.sum()

    ds     = perimeter / N
    s_mids = (np.arange(N) + 0.5) * ds                # panel midpoint arc-positions

    # Cumulative arc lengths at vertex starts (length M+1)
    cum = np.concatenate([[0.0], np.cumsum(edge_lengths)])

    # Which edge does each panel midpoint fall on?
    edge_idx = np.searchsorted(cum[1:], s_mids, side='right')
    edge_idx = np.clip(edge_idx, 0, M - 1)

    # Local fraction along that edge
    s_local = (s_mids - cum[edge_idx]) / edge_lengths[edge_idx]   # (N,) in [0,1]
    s_local = np.clip(s_local, 0.0, 1.0)

    # Panel midpoint positions
    p0    = verts[edge_idx]                             # (N, 2)
    p1    = verts[(edge_idx + 1) % M]                  # (N, 2)
    nodes = p0 + s_local[:, None] * (p1 - p0)          # (N, 2)

    # Outward normals: for CCW polygon, rotate edge tangent 90° clockwise
    # tangent = (tx, ty)  →  outward normal = (ty, −tx)
    tang    = edges[edge_idx]                           # (N, 2) edge directions
    tlen    = edge_lengths[edge_idx]                    # (N,)
    normals = np.stack([tang[:, 1], -tang[:, 0]], axis=1) / tlen[:, None]

    lengths = np.full(N, ds)
    return nodes, normals, lengths


# ── Radar target shapes ───────────────────────────────────────────────────────

def square_panels(N, L=1.0):
    """N panels on a square of side length L (centred at origin).

    Returns (nodes, normals, lengths).

    Radar note: strong backscatter at 0°, 90°, 180°, 270° (flat-plate returns).
    """
    h = L / 2.0
    # CCW: bottom → right → top → left
    verts = np.array([
        [-h, -h],
        [ h, -h],
        [ h,  h],
        [-h,  h],
    ])
    return polygon_panels(N, verts)


def diamond_panels(N, a=1.0, b=None):
    """N panels on a diamond (rhombus) with semi-diagonals a (x) and b (y).

    b defaults to a (square rotated 45°).

    Radar note: strong returns at 45°, 135°, 225°, 315°; minimal frontal RCS
    at 0°/90° — illustrates how faceting shifts the backscatter peaks.
    """
    if b is None:
        b = a
    verts = np.array([
        [ a,  0],   # right
        [ 0,  b],   # top
        [-a,  0],   # left
        [ 0, -b],   # bottom
    ])
    return polygon_panels(N, verts)


def corner_reflector_panels(N, arm_length=2.0, arm_width=0.25):
    """N panels on a thick right-angle dihedral corner reflector.

    Geometry: an L-shape with arms extending along +x and +y, arm thickness
    arm_width.  The concave right-angle corner faces the upper-right (+45°).

    Radar note: strong retroreflection at 45° (upper-right) due to the
    double-bounce mechanism inside the concave corner.
    """
    L = arm_length
    w = arm_width / 2.0
    # CCW: start at inner corner, trace the L-shape
    verts = np.array([
        [-w, -w],   # inner corner
        [ L, -w],   # bottom of horizontal arm
        [ L,  w],   # top of horizontal arm
        [ w,  w],   # junction to vertical arm
        [ w,  L],   # outer top of vertical arm
        [-w,  L],   # inner top of vertical arm
    ])
    return polygon_panels(N, verts)


def stealth_panels(N, length=4.0, half_width=0.4):
    """N panels on a stealth body: elongated diamond (low frontal RCS).

    A high-aspect-ratio diamond with length >> half_width.  The shallow nose
    angle deflects energy off the monostatic (frontal) direction — the
    hallmark of faceted stealth design.

    Args:
        N:          Number of panels.
        length:     Total nose-to-tail length.
        half_width: Maximum half-width (y-extent).

    Radar note: very weak return at 0°/180° (nose/tail-on); peak returns at
    angles where the flat flanks are perpendicular to the radar.
    """
    h = length / 2.0
    verts = np.array([
        [ h,          0],   # nose (right)
        [ 0,  half_width],  # top widest point
        [-h,          0],   # tail (left)
        [ 0, -half_width],  # bottom widest point
    ])
    return polygon_panels(N, verts)


def rounded_stealth_panels(N, length=4.0, half_width=0.4, n_taper=8):
    """N panels on a stealth body with a curved leading edge.

    The sharp diamond nose is replaced with a multi-facet taper —
    approximating the curved leading edge of real stealth airframes.

    Args:
        N:        Number of panels.
        length:   Total length.
        half_width: Maximum half-width.
        n_taper:  Number of facets on each half of the leading edge.

    Returns:
        (nodes, normals, lengths) as usual.
    """
    h = length / 2.0
    # Build upper outline: curved leading edge (n_taper points) + straight aft
    t_lead    = np.linspace(0, np.pi / 2, n_taper + 1)[:-1]  # 0 .. 90°
    x_lead_u  = h - half_width * (1 - np.sin(t_lead)) * (h / half_width)
    y_lead_u  = half_width * np.sin(t_lead)

    # Approximate the leading edge as a set of chord points on an ellipse
    theta_upper = np.linspace(0, np.pi / 2, n_taper + 2)[1:-1]
    x_upper     = h * np.cos(theta_upper)
    y_upper     = half_width * np.sin(theta_upper)

    verts_upper = np.stack([x_upper, y_upper], axis=1)       # leading edge top
    verts_lower = np.stack([x_upper, -y_upper[::-1]], axis=1)  # mirror bottom

    # Full diamond outline: nose → upper → tail → lower
    verts = np.vstack([
        [[h, 0]],
        verts_upper,
        [[0, half_width]],
        [[-h, 0]],
        [[0, -half_width]],
        verts_lower[::-1],
    ])

    return polygon_panels(N, verts)
