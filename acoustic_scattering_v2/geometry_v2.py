"""
geometry_v2.py — 2D panel generators for acoustic BEM, v2.

Extends acoustic_scattering/geometry.py with:
  - perturb_panels()  random normal-direction roughness perturbation
  - char_size table   for scaling roughness consistently across shapes
  - All original shapes re-exported unchanged

All generators return (nodes, normals, lengths):
    nodes:   (N, 2) float64 — panel midpoint coordinates
    normals: (N, 2) float64 — outward unit normals
    lengths: (N,)   float64 — panel arc lengths Δl
"""

import sys
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORIG = os.path.join(_HERE, '..', 'acoustic_scattering')
if _ORIG not in sys.path:
    sys.path.insert(0, _ORIG)

from geometry import (
    circle_panels,
    ellipse_panels,
    joukowski_panels,
    submarine_panels,
    plot_panels,
)

# ── Characteristic sizes for roughness scaling ────────────────────────────────

CHAR_SIZES = {
    'circle':     1.0,    # radius R
    'ellipse':    1.0,    # semi-major axis a
    'joukowski':  1.0,    # chord ~2c
    'submarine':  1.0,    # radius R
}


def perturb_panels(nodes, normals, lengths, eps, char_size, seed=0):
    """Apply random normal-direction roughness to panel centroids.

    Displaces each panel centroid by Δx = N(0, ε·L) in the outward-normal
    direction, where L = char_size.  The perturbation is seeded for
    reproducibility across Monte Carlo studies.

    Arc lengths are unchanged (panel areas are recomputed by the caller
    if needed; for BEM the displacement is assumed small).

    Args:
        nodes:     (N, 2) float64 panel centroids.
        normals:   (N, 2) float64 outward unit normals.
        lengths:   (N,)   float64 arc lengths (returned unchanged).
        eps:       Roughness fraction (displacement std = eps × char_size).
        char_size: Characteristic length for scaling (e.g. radius R).
        seed:      Random seed (int or np.random.Generator).

    Returns:
        nodes_p:  (N, 2) perturbed centroids.
        normals:  (N, 2) unchanged outward normals.
        lengths:  (N,)   unchanged arc lengths.
    """
    rng = np.random.default_rng(seed)
    dx  = rng.normal(0.0, eps * char_size, size=len(nodes))
    return nodes + dx[:, None] * normals, normals, lengths


# ── Convenience: get a geometry by name ──────────────────────────────────────

SHAPES = {
    'circle':    lambda N, k: circle_panels(N),
    'ellipse':   lambda N, k: ellipse_panels(N, a=2.0, b=1.0),
    'joukowski': lambda N, k: joukowski_panels(N, c=1.0, eps=0.1),
    'submarine': lambda N, k: submarine_panels(N, L=4.0, R=1.0),
}

SHAPE_CHAR_SIZES = {
    'circle':    1.0,
    'ellipse':   2.0,   # semi-major axis
    'joukowski': 2.2,   # approximate chord
    'submarine': 1.0,   # radius R
}

# Roughness fractions used in Monte Carlo studies
ROUGHNESS_FRACS = [0.00, 0.02, 0.05, 0.10]
