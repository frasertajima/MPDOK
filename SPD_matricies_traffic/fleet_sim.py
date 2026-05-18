"""
fleet_sim.py — City traffic field + N-vehicle random-walk simulator.

Pure NumPy/SciPy: no GPU dependency.  GPU ops happen in enkf_solver.py.

Classes:
    TrafficField  — four drifting Gaussian hotspots at real Vancouver locations
    Fleet         — N vehicles, random-walk attracted to major road network
"""

import numpy as np
from dataclasses import dataclass, field as dc_field
from scipy.spatial import KDTree

CITY_SIZE = 100.0   # km — bounding box [0, CITY_SIZE]²

# Geographic bounds (must match index.html SIM_BOUNDS)
# SW: lat=48.90, lng=-123.50  →  sim (0, 0)
# NE: lat=49.80, lng=-122.10  →  sim (100, 100)
_LNG_W, _LNG_E = -123.50, -122.10
_LAT_S, _LAT_N =   48.90,   49.80


@dataclass
class Incident:
    """A sudden congestion spike placed on the traffic field."""
    x:        float          # sim x (km)
    y:        float          # sim y (km)
    sigma:    float = 5.0    # Gaussian radius (km)
    amp:      float = 0.85   # peak additional congestion
    t_start:  float = 0.0    # sim time when placed
    duration: float = -1.0   # sim steps before expiry  (-1 = permanent)

    def strength(self, t: float) -> float:
        """Amplitude in [0,1] — fades out linearly over the final 20%."""
        if self.duration < 0:
            return 1.0
        frac = (t - self.t_start) / self.duration
        if frac >= 1.0:
            return 0.0
        if frac > 0.8:
            return 1.0 - (frac - 0.8) / 0.2
        return 1.0


def _ll_to_sim(latlng):
    """[(lat,lng), ...] list → (N,2) float64 sim (x,y) array."""
    arr = np.array(latlng, dtype=np.float64)
    x = (arr[:, 1] - _LNG_W) / (_LNG_E - _LNG_W) * CITY_SIZE
    y = (arr[:, 0] - _LAT_S) / (_LAT_N - _LAT_S) * CITY_SIZE
    return np.column_stack([x, y])


def _densify(pts, spacing=0.5):
    """Interpolate extra points along each segment so spacing ≤ `spacing` km."""
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        d = np.linalg.norm(b - a)
        n = max(1, int(np.ceil(d / spacing)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.array(out)


# ---------------------------------------------------------------------------
# Major Metro Vancouver road segments — (lat, lng) waypoints.
# Covers the main grid of arteries within the sim bounding box.
# ---------------------------------------------------------------------------
_ROADS_LL = [
    # ── Trans-Canada / Hwy 1 (E–W backbone, north of Fraser) ──────────────
    [(49.37, -123.27), (49.32, -123.12), (49.29, -123.00),
     (49.25, -122.90), (49.21, -122.82), (49.19, -122.73),
     (49.18, -122.65), (49.14, -122.45)],

    # ── Highway 99 / Oak St Bridge (N–S, west metro) ──────────────────────
    [(48.99, -122.76), (49.07, -122.85), (49.13, -123.02),
     (49.18, -123.09), (49.21, -123.12), (49.25, -123.13)],

    # ── Broadway / W 10th Ave (E–W, central Vancouver) ───────────────────
    [(49.26, -123.25), (49.26, -123.18), (49.26, -123.13),
     (49.26, -123.07), (49.26, -123.00), (49.26, -122.94)],

    # ── Kingsway (diagonal NE through Burnaby) ───────────────────────────
    [(49.22, -123.01), (49.24, -122.95), (49.25, -122.90),
     (49.26, -122.87)],

    # ── Canada Way (E–W, south Burnaby) ──────────────────────────────────
    [(49.23, -123.00), (49.23, -122.93), (49.23, -122.87),
     (49.22, -122.82)],

    # ── Hastings St / Barnet Hwy (E–W, North Burnaby → Port Moody) ───────
    [(49.28, -123.10), (49.29, -123.00), (49.30, -122.93),
     (49.30, -122.88), (49.29, -122.80)],

    # ── Lougheed Hwy (E–W, mid Burnaby → Coquitlam → Pitt Meadows) ───────
    [(49.25, -122.97), (49.26, -122.88), (49.27, -122.80),
     (49.27, -122.72), (49.26, -122.63), (49.24, -122.58)],

    # ── North Road / Clarke Rd (N–S, Burnaby/Coquitlam border) ──────────
    [(49.23, -122.87), (49.27, -122.87), (49.30, -122.87)],

    # ── Granville St (N–S, central Vancouver) ────────────────────────────
    [(49.18, -123.14), (49.23, -123.14), (49.26, -123.14),
     (49.30, -123.14)],

    # ── Cambie St / Cambie Bridge ────────────────────────────────────────
    [(49.20, -123.11), (49.24, -123.11), (49.27, -123.11)],

    # ── Knight St / Clark Dr (N–S, east Vancouver) ───────────────────────
    [(49.20, -123.07), (49.24, -123.07), (49.27, -123.07)],

    # ── Boundary Rd (N–S, Van/Burnaby boundary) ──────────────────────────
    [(49.20, -123.00), (49.25, -123.00), (49.28, -123.00)],

    # ── Scott Rd / 120 St (N–S, Surrey) ──────────────────────────────────
    [(49.01, -122.80), (49.07, -122.80), (49.13, -122.80),
     (49.20, -122.80)],

    # ── King George Blvd (N–S diagonal, Surrey) ──────────────────────────
    [(49.01, -122.84), (49.07, -122.84), (49.13, -122.84),
     (49.19, -122.83)],

    # ── Fraser Hwy (E–W, mid-Surrey to Langley) ──────────────────────────
    [(49.18, -122.97), (49.17, -122.87), (49.15, -122.80),
     (49.12, -122.73), (49.10, -122.65), (49.08, -122.55)],

    # ── 64 Ave (E–W, south Surrey / Cloverdale) ──────────────────────────
    [(49.10, -122.87), (49.10, -122.80), (49.10, -122.72),
     (49.10, -122.65)],

    # ── 200 St (N–S, Langley) ─────────────────────────────────────────────
    [(49.01, -122.67), (49.07, -122.67), (49.13, -122.67),
     (49.18, -122.67)],

    # ── Dewdney Trunk Rd (E–W, Mission/Maple Ridge) ──────────────────────
    [(49.22, -122.70), (49.22, -122.60), (49.22, -122.50),
     (49.22, -122.40), (49.23, -122.30)],

    # ── Lougheed Hwy east extension (Maple Ridge) ────────────────────────
    [(49.24, -122.58), (49.23, -122.47), (49.22, -122.33)],

    # ── North Vancouver: Main St / Lonsdale ──────────────────────────────
    [(49.31, -123.07), (49.34, -123.07), (49.37, -123.07),
     (49.40, -123.07)],

    # ── Marine Dr (North Van E–W) ─────────────────────────────────────────
    [(49.31, -123.14), (49.31, -123.08), (49.32, -123.00),
     (49.33, -122.92), (49.34, -122.87)],

    # ── Westwood Plateau / Mariner Way (Coquitlam N–S) ───────────────────
    [(49.28, -122.77), (49.31, -122.77), (49.34, -122.77)],

    # ── Pitt Meadows / Maple Ridge ridge road ────────────────────────────
    [(49.23, -122.63), (49.24, -122.55), (49.25, -122.47),
     (49.27, -122.40)],

    # ── Delta / Ladner main roads (south metro) ───────────────────────────
    [(49.07, -122.96), (49.08, -123.03), (49.09, -123.08),
     (49.10, -123.15)],
    [(49.06, -123.06), (49.06, -122.98), (49.06, -122.90)],
]


def _build_road_tree():
    """Build KDTree of densely-sampled road network points."""
    all_pts = []
    for seg_ll in _ROADS_LL:
        sim_pts = _ll_to_sim(seg_ll)
        all_pts.append(_densify(sim_pts, spacing=0.4))
    pts = np.vstack(all_pts)
    return KDTree(pts), pts


_ROAD_TREE = None
_ROAD_PTS  = None


def _get_road_tree():
    global _ROAD_TREE, _ROAD_PTS
    if _ROAD_TREE is None:
        _ROAD_TREE, _ROAD_PTS = _build_road_tree()
    return _ROAD_TREE, _ROAD_PTS


class TrafficField:
    """Time-varying 2D traffic congestion field.

    Four Gaussian hotspots at real Metro Vancouver traffic corridors,
    plus any active Incident objects added at runtime.
    Field values are in [0, 1]: 0 = open road, 1 = standstill.
    """

    def __init__(self):
        self.incidents: list[Incident] = []

    def add_incident(self, inc: Incident):
        self.incidents.append(inc)

    def clear_incidents(self):
        self.incidents = []

    def _prune_incidents(self, t: float):
        """Remove expired incidents in-place."""
        self.incidents = [i for i in self.incidents
                          if i.duration < 0 or (t - i.t_start) < i.duration]

    # Columns: cx0, cy0, σ (km), amplitude, vx, vy, pulse_ω
    # Positions in sim (x,y) at real Metro Van locations:
    #   Downtown Vancouver:    lat=49.28, lng=-123.12  → x≈27, y≈42
    #   Metrotown/Burnaby:     lat=49.23, lng=-122.99  → x≈36, y≈37
    #   Surrey Central:        lat=49.19, lng=-122.85  → x≈46, y≈32
    #   Coquitlam/Port Coquit: lat=49.27, lng=-122.82  → x≈49, y≈41
    _HOTSPOTS = np.array([
        [27.0, 42.0,  9.0, 0.90,  0.06,  0.04, 0.15],   # Downtown Vancouver
        [36.0, 37.0,  8.0, 0.80, -0.05,  0.07, 0.20],   # Metrotown/Burnaby
        [46.0, 32.0, 11.0, 0.70,  0.04, -0.06, 0.12],   # Surrey Central
        [49.0, 41.0,  8.0, 0.65, -0.04, -0.03, 0.25],   # Coquitlam corridor
    ], dtype=np.float64)

    def evaluate(self, xy: np.ndarray, t: float) -> np.ndarray:
        """Field values at (M, 2) positions xy at time t.

        Returns (M,) float64, clipped to [0, 1].
        """
        out = np.zeros(xy.shape[0], dtype=np.float64)
        for cx0, cy0, sigma, amp, phase_x, phase_y, omega in self._HOTSPOTS:
            cx = cx0 + 5.0 * np.sin(omega * t + phase_x)
            cy = cy0 + 5.0 * np.cos(omega * t * 0.7 + phase_y)
            a  = amp * (0.85 + 0.15 * np.sin(2.0 * omega * t))
            dx = xy[:, 0] - cx
            dy = xy[:, 1] - cy
            out += a * np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
        # Active incidents
        for inc in self.incidents:
            s = inc.strength(t)
            if s <= 0:
                continue
            dx = xy[:, 0] - inc.x
            dy = xy[:, 1] - inc.y
            out += s * inc.amp * np.exp(-(dx ** 2 + dy ** 2) / (2.0 * inc.sigma ** 2))
        return np.clip(out, 0.0, 1.0)

    def evaluate_grid(self, nx: int, ny: int, t: float) -> np.ndarray:
        """Evaluate on an (nx × ny) pixel grid.  Returns (ny, nx) float64."""
        xs = np.linspace(0.0, CITY_SIZE, nx)
        ys = np.linspace(0.0, CITY_SIZE, ny)
        gx, gy = np.meshgrid(xs, ys)
        xy = np.column_stack([gx.ravel(), gy.ravel()])
        return self.evaluate(xy, t).reshape(ny, nx)


class Fleet:
    """N vehicles performing random-walk attracted to major road network.

    Each step: random walk noise + soft pull toward nearest road point.
    Vehicles naturally cluster along streets without hard constraints.

    Attributes:
        positions: (N, 2) float64 — current vehicle coordinates.
        t:         float — elapsed simulation time.
    """

    ROAD_PULL = 0.75   # fraction of distance toward nearest road per step

    def __init__(self, N: int = 6000, seed: int = 0):
        self.N    = N
        self._rng = np.random.default_rng(seed)
        self.t    = 0.0
        self.dt   = 1.0
        self.step_std    = 0.35    # km random displacement (σ) per step
        self.centre_pull = 0.0003  # gentle drift toward city centre

        # Initialise positions scattered around road network
        tree, pts = _get_road_tree()
        chosen = self._rng.integers(0, len(pts), size=N)
        jitter = 1.5   # km spread around each road point
        self.positions = np.clip(
            pts[chosen] + self._rng.standard_normal((N, 2)) * jitter,
            0.0, CITY_SIZE,
        ).astype(np.float64)

    def step(self, field: "TrafficField | None" = None):
        """Advance all N vehicles one time step.

        If field is supplied, vehicles slow down proportionally to local
        congestion: gridlock (1.0) → ~15 % of free-flow speed;
        clear road (0.0) → full step_std.  This makes jams visible as
        dense, barely-moving clusters.
        """
        centre = np.full(2, CITY_SIZE * 0.5)

        if field is not None:
            congestion = field.evaluate(self.positions, self.t)   # (N,) in [0,1]
            speed_scale = 1.0 - 0.85 * congestion                 # 1.0 → 0.15
        else:
            speed_scale = 1.0

        noise = (self._rng.standard_normal((self.N, 2))
                 * self.step_std
                 * speed_scale[:, None])
        drift  = (centre - self.positions) * self.centre_pull

        proposed = np.clip(self.positions + noise + drift, 0.0, CITY_SIZE)

        # Pull toward nearest road node
        tree, tree_pts = _get_road_tree()
        _, idx = tree.query(proposed)
        nearest = tree_pts[idx]
        proposed += (nearest - proposed) * self.ROAD_PULL

        self.positions = proposed
        self.t += self.dt

    def measure(self, field: TrafficField,
                noise_std: float = 0.08) -> np.ndarray:
        """Noisy congestion reading at each vehicle's position.

        Returns (N,) float64 — true field + N(0, noise_std²) noise.
        """
        truth = field.evaluate(self.positions, self.t)
        noise = self._rng.standard_normal(self.N) * noise_std
        return truth + noise
