"""
geometry_3d_targets.py — 5 target meshes + roughness perturbation.

3D analogues of the 5 targets in radar_scattering/geometry.py:

  sphere_mesh          icosphere             (circle analogue)
  cube_mesh            axis-aligned box      (square analogue)
  double_cone_mesh     bipyramid             (diamond analogue)
  dihedral_corner_mesh two orthogonal plates (corner reflector analogue)
  stealth_body_mesh    body of revolution    (stealth body analogue)

Plus:
  perturb_mesh_3d   Gaussian normal-direction roughness perturbation
  TARGETS           list of 5 target dicts with id, name, geom_fn, char_size

All generators return (nodes, normals, areas):
  nodes:   (N, 3) float64 — panel centroids
  normals: (N, 3) float64 — outward unit normals
  areas:   (N,)   float64 — panel areas [m²]
"""

import sys
import os
import numpy as np

_BEM_COBOL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'bem_cobol')
if _BEM_COBOL not in sys.path:
    sys.path.insert(0, _BEM_COBOL)

from geometry_3d import (
    icosphere, _panels_from_triangles, mesh_stats,
)

# Direct re-export
sphere_mesh = icosphere


# ── Cube mesh (corrected panel-count formula) ─────────────────────────────

def cube_mesh(N_target, L=2.0):
    """Axis-aligned cube of side L with ~N_target triangular panels.

    geometry_3d.box_mesh uses ceil(sqrt(N/6)) giving 12n² panels with
    ~2× overshoot.  This version uses round(sqrt(N/12)) so that
    actual count ≈ N_target within ±20%.

    Actual count: 12 * n²  where  n = max(1, round(sqrt(N_target / 12))).
    """
    n    = max(1, int(round(np.sqrt(N_target / 12))))
    base = np.linspace(-L / 2, L / 2, n + 1)

    face_specs = [
        (np.array([0, 0, 1]),  ),   # +z
        (np.array([0, 0, -1]), ),   # -z
        (np.array([1, 0, 0]),  ),   # +x
        (np.array([-1, 0, 0]),),   # -x
        (np.array([0, 1, 0]),  ),   # +y
        (np.array([0, -1, 0]),),   # -y
    ]
    verts, faces = [], []
    for (normal,) in face_specs:
        ax = int(np.argmax(np.abs(normal)))
        t1, t2 = [i for i in range(3) if i != ax]
        for i in range(n):
            for j in range(n):
                u0, u1 = base[i], base[i + 1]
                v0, v1 = base[j], base[j + 1]
                pts = np.zeros((4, 3))
                pts[:, ax] = L / 2 * normal[ax]
                pts[0, t1], pts[0, t2] = u0, v0
                pts[1, t1], pts[1, t2] = u1, v0
                pts[2, t1], pts[2, t2] = u1, v1
                pts[3, t1], pts[3, t2] = u0, v1
                off = len(verts)
                verts += list(pts)
                faces += [[off, off + 1, off + 2],
                          [off, off + 2, off + 3]]

    v = np.array(verts, dtype=np.float64)
    f = np.array(faces, dtype=np.int32)
    return _panels_from_triangles(v, f, outward_from_origin=True)


# ── Stealth body of revolution ────────────────────────────────────────────────

def stealth_body_mesh(N_target, length=4.0, half_width=0.4, n_phi=32):
    """Stealth body of revolution — closed surface with front and rear nose caps.

    The profile traces from front nose (z=+length/2, r≈0) along the belly to the
    rear tail (z=-length/2, r≈0).  Each end is closed with a flat n_phi-gon cap
    connected to a single axial center vertex.

    Previous bug: the face loop used (i+1) % n_z, wrapping the rear-tail ring
    back to the front-nose ring and creating a 3.95m-tall × 0.004m-wide seam
    panel (1007:1 aspect ratio) that doesn't physically exist.

    Fix: open the body loop (no wrap) + explicit front/rear caps.
    r_min raised from 5% to 15% of half_width (reduces cap panel AR from ~13 to ~4).

    Panel count: body = n_body × n_phi × 2, caps = 2 × n_phi
    Total = (n_body + 1) × 2 × n_phi  ≈  N_target.
    """
    # n_body body-strip intervals + 1 makes cap count: total = (n_body+1)*2*n_phi
    n_body = max(4, N_target // (2 * n_phi)) - 1
    n_rings = n_body + 2   # includes both end rings (front nose + rear tail)

    # t from 0 to 2π inclusive — front nose at t=0, rear tail at t=2π
    t = np.linspace(0, 2 * np.pi, n_rings)

    r_profile = half_width * (1 - np.abs(np.cos(t / 2)) ** 3)
    r_profile = np.maximum(r_profile, 0.15 * half_width)  # 15% nose radius
    z_profile = length * (0.5 - t / (2 * np.pi))

    phi_arr = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Ring vertices: shape (n_rings × n_phi)
    verts = []
    for i in range(n_rings):
        for j in range(n_phi):
            verts.append([r_profile[i] * np.cos(phi_arr[j]),
                          r_profile[i] * np.sin(phi_arr[j]),
                          z_profile[i]])

    # Cap center vertices (true axial points at r=0)
    front_cap_idx = len(verts)
    verts.append([0., 0.,  length / 2.0])   # front nose tip (z = +2m)
    rear_cap_idx  = len(verts)
    verts.append([0., 0., -length / 2.0])   # rear tail tip  (z = -2m)

    faces = []

    # Body quads (open loop — no wrap between last and first ring)
    for i in range(n_rings - 1):
        for j in range(n_phi):
            j1 = (j + 1) % n_phi
            a  = i       * n_phi + j
            b  = i       * n_phi + j1
            c  = (i + 1) * n_phi + j1
            d  = (i + 1) * n_phi + j
            faces += [[a, b, c], [a, c, d]]

    # Front cap: ring 0 → axial center (outward normal = +z)
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        faces.append([front_cap_idx, j1, j])

    # Rear cap: last ring → axial center (outward normal = -z)
    base = (n_rings - 1) * n_phi
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        faces.append([rear_cap_idx, base + j, base + j1])

    v = np.array(verts, dtype=np.float64)
    f = np.array(faces, dtype=np.int32)
    return _panels_from_triangles(v, f, outward_from_origin=True)


# ── Double cone (bipyramid) ────────────────────────────────────────────────

def double_cone_mesh(N_target, half_height=1.5, R=1.0):
    """Closed bipyramid with structured quad mesh — greatly reduced apex AR.

    Previous version used n_phi = N_target//2 fan triangles per apex, giving
    apex aspect ratio = slant * n_phi / (2πR) ≈ 734 at N=5120.

    Fix: use a structured n_z × n_phi quad mesh (split into triangles) for the
    cone body, plus one row of short apex fan triangles.  The apex fans span
    only one ring-height (slant/n_z) instead of the full slant, reducing their
    aspect ratio by a factor of n_z.

    n_phi and n_z are chosen jointly so body panels are roughly equilateral:
      panel_width ≈ 2πR/n_phi,  panel_height ≈ slant/n_z
      → n_z / n_phi = slant / (2πR) ≈ 0.287 for the defaults.

    Panel count per cone half:
      body quads:  (n_z - 1) × n_phi × 2  triangles
      apex fans:              n_phi         triangles
      total one half:  n_phi × (2*n_z - 1)
    Grand total (2 halves): 2 × n_phi × (2*n_z - 1)  ≈  N_target.

    Apex fan aspect ratio: ≈ 0.287 × n_phi  (≈ 19 at N=5120 vs 734 before).
    """
    slant = np.sqrt(R ** 2 + half_height ** 2)   # slant height of cone
    ratio = slant / (2.0 * np.pi * R)            # ≈ 0.287 for defaults

    # Solve: 2 * n_phi * (2*n_z - 1) = N_target  with n_z = round(ratio*n_phi)
    # → n_phi² * 2 * (2*ratio - 1/n_phi) ≈ N_target
    # Approximate: n_phi ≈ sqrt(N_target / (4*ratio))
    n_phi = max(8, int(np.sqrt(N_target / (4.0 * ratio))))
    n_z   = max(2, round(ratio * n_phi))

    # Fine-tune n_phi upward until actual count ≥ N_target
    while 2 * n_phi * (2 * n_z - 1) < N_target:
        n_phi += 1
        n_z = max(2, round(ratio * n_phi))

    phi_arr = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Ring z-heights and radii (linear taper from equator to apex)
    # n_z rings: ring 0 = equator, ring n_z-1 = just below apex
    z_rings = np.linspace(0.0, half_height, n_z + 1)[:-1]  # exclude apex
    r_rings = R * (1.0 - z_rings / half_height)

    def _cone_half(z_sign):
        """Generate vertices and faces for one cone half (top or bottom)."""
        verts = []
        for i in range(n_z):
            for j in range(n_phi):
                verts.append([r_rings[i] * np.cos(phi_arr[j]),
                              r_rings[i] * np.sin(phi_arr[j]),
                              z_sign * z_rings[i]])
        apex_idx = len(verts)
        verts.append([0., 0., z_sign * half_height])

        faces = []
        # Body quads: connect ring i to ring i+1 (for i = 0 .. n_z-2)
        for i in range(n_z - 1):
            for j in range(n_phi):
                j1 = (j + 1) % n_phi
                a  = i       * n_phi + j
                b  = i       * n_phi + j1
                c  = (i + 1) * n_phi + j1
                d  = (i + 1) * n_phi + j
                faces += [[a, b, c], [a, c, d]]
        # Apex fans: short triangles from top ring to apex
        for j in range(n_phi):
            j1 = (j + 1) % n_phi
            a  = (n_z - 1) * n_phi + j
            b  = (n_z - 1) * n_phi + j1
            faces.append([apex_idx, b, a])

        return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)

    top_v, top_f = _cone_half(+1.0)
    bot_v, bot_f = _cone_half(-1.0)

    offset = len(top_v)
    v = np.vstack([top_v, bot_v])
    f = np.vstack([top_f, bot_f + offset])
    return _panels_from_triangles(v, f, outward_from_origin=True)


# ── Dihedral corner reflector ──────────────────────────────────────────────

def dihedral_corner_mesh(N_target, arm_length=2.0, double_sided=False):
    """Two orthogonal flat plates meeting at a 90° dihedral angle.

    Plate 1 lies in the xz-plane (y = 0), x ∈ [0, L], z ∈ [-L/2, L/2].
    Plate 2 lies in the yz-plane (x = 0), y ∈ [0, L], z ∈ [-L/2, L/2].
    The dihedral edge runs along the z-axis.

    Normals point into the corner interior:
      Plate 1: +y direction  (faces the yz half-space)
      Plate 2: +x direction  (faces the xz half-space)

    Args:
        N_target:     Target panel count.
        arm_length:   Plate length along each arm [m] (= L).
        double_sided: If True, duplicate each panel with reversed normal,
                      creating a thin closed-body approximation that makes the
                      single-layer BEM well-posed for the exterior Dirichlet
                      problem on open surfaces.

    Actual panel count: 4 * n² (single-sided) or 8 * n² (double-sided),
    where n = max(1, round(sqrt(N_target / 4))).
    """
    L = arm_length
    n = max(1, round(np.sqrt(N_target / 4)))

    def _plate(u_axis, v_axis, offset_axis, u_range, v_range, n_div, normal):
        """Triangulate one rectangular plate into 2*n_div² panels."""
        u = np.linspace(u_range[0], u_range[1], n_div + 1)
        v = np.linspace(v_range[0], v_range[1], n_div + 1)
        nodes_out, norms_out, areas_out = [], [], []
        for i in range(n_div):
            for j in range(n_div):
                # Four corners of quad (i,j)
                def pt(ui, vi):
                    p = np.zeros(3)
                    p[u_axis] = ui
                    p[v_axis] = vi
                    p[offset_axis] = 0.0
                    return p
                p00 = pt(u[i],   v[j])
                p10 = pt(u[i+1], v[j])
                p11 = pt(u[i+1], v[j+1])
                p01 = pt(u[i],   v[j+1])
                for tri in [(p00, p10, p11), (p00, p11, p01)]:
                    a, b, c = tri
                    centroid = (a + b + c) / 3
                    area     = np.linalg.norm(np.cross(b - a, c - a)) / 2
                    n_vec    = np.array(normal, dtype=np.float64)
                    nodes_out.append(centroid)
                    norms_out.append(n_vec)
                    areas_out.append(area)
        return (np.array(nodes_out), np.array(norms_out), np.array(areas_out))

    # Plate 1: xz-plane  (u=x, v=z, offset=y=0), normal = +y = (0,1,0)
    n1, nm1, a1 = _plate(u_axis=0, v_axis=2, offset_axis=1,
                         u_range=(0.0, L), v_range=(-L/2, L/2),
                         n_div=n, normal=[0., 1., 0.])

    # Plate 2: yz-plane  (u=y, v=z, offset=x=0), normal = +x = (1,0,0)
    n2, nm2, a2 = _plate(u_axis=1, v_axis=2, offset_axis=0,
                         u_range=(0.0, L), v_range=(-L/2, L/2),
                         n_div=n, normal=[1., 0., 0.])

    nodes   = np.vstack([n1, n2])
    normals = np.vstack([nm1, nm2])
    areas   = np.concatenate([a1, a2])

    if double_sided:
        nodes   = np.vstack([nodes,   nodes])
        normals = np.vstack([normals, -normals])
        areas   = np.concatenate([areas, areas])

    return nodes.copy(), normals.copy(), areas.copy()


# ── Roughness perturbation ─────────────────────────────────────────────────

def perturb_mesh_3d(nodes, normals, areas, eps, char_size, seed):
    """Apply Gaussian normal-direction roughness to panel centroids.

    Identical formula to the 2D perturb() in generate_stage7_data.py:
      delta ~ N(0, eps * char_size)  per panel
      nodes_p = nodes + delta[:, None] * normals

    Panel areas are unchanged (first-order approximation, exact to O(eps²)).
    Normal vectors are unchanged (perturbation is along the normal direction).

    Args:
        nodes:     (N, 3) float64 — panel centroids.
        normals:   (N, 3) float64 — outward unit normals.
        areas:     (N,)   float64 — panel areas.
        eps:       float  — roughness fraction (0 = smooth).
        char_size: float  — characteristic length scale [m].
        seed:      int    — RNG seed for reproducibility.

    Returns:
        nodes_p, normals, areas  (copies; normals and areas unchanged)
    """
    if eps == 0.0:
        return nodes.copy(), normals.copy(), areas.copy()
    rng    = np.random.default_rng(seed)
    delta  = rng.normal(0.0, eps * char_size, len(nodes))
    return (nodes + delta[:, None] * normals,
            normals.copy(),
            areas.copy())


# ── Target registry ────────────────────────────────────────────────────────

TARGETS = [
    dict(id=0, name='Sphere',
         geom_fn=lambda N, k: sphere_mesh(N, R=1.0),
         char_size=1.0),
    dict(id=1, name='Cube',
         geom_fn=lambda N, k: cube_mesh(N, L=2.0),
         char_size=1.0),
    dict(id=2, name='DblCone',
         geom_fn=lambda N, k: double_cone_mesh(N, half_height=1.5, R=1.0),
         char_size=1.3),
    dict(id=3, name='Dihedral',
         geom_fn=lambda N, k: dihedral_corner_mesh(N, arm_length=2.0),
         char_size=0.4),
    dict(id=4, name='Stealth',
         geom_fn=lambda N, k: stealth_body_mesh(N, length=4.0, half_width=0.4),
         char_size=0.4),
]

ROUGHNESS_FRACS = [0.00, 0.05, 0.10, 0.20]
WAVENUMBERS     = [3.0, 5.0, 8.0, 12.0, 16.0]
N_PANELS        = 5120
N_SEEDS         = 20
