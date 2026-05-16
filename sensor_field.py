"""
rbf_spatial — Synthetic multi-field sensor simulation.

Generates realistic N-sensor networks with multiple independent physical
fields (temperature, pressure, humidity) that share the same spatial
geometry but have distinct time-evolving signals.  No external data files.

Usage:
    from sensor_field import SensorNetwork
    net = SensorNetwork(N=8192, seed=42)
    A, gamma = net.build_kernel()          # shared RBF matrix
    b_temp = net.field('temperature', t=0)
    b_pres = net.field('pressure',    t=0)
    b_humi = net.field('humidity',    t=0)
"""

import cupy as cp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MPDOK'))
from rbf_kernel import synthetic_coords, build_rbf_kernel, weather_front


# ── Field definitions ─────────────────────────────────────────────────────

def _temperature(coords, t):
    """Moving warm front (°C).  Range ~10–20°C."""
    x, y = coords[:, 0], coords[:, 1]
    base  = 15.0 + 5.0 * cp.sin((x + y) / 20.0 - 0.5 * t)
    rng   = cp.random.default_rng(t)
    return base + rng.standard_normal(size=base.shape) * 0.05

def _pressure(coords, t):
    """Rotating low-pressure system (hPa offset from 1013).  Range ±15 hPa."""
    x, y  = coords[:, 0], coords[:, 1]
    cx    = 50.0 + 10.0 * cp.cos(t * 0.3)   # centre drifts
    cy    = 50.0 + 10.0 * cp.sin(t * 0.3)
    r2    = (x - cx)**2 + (y - cy)**2
    field = -12.0 * cp.exp(-r2 / 800.0) + 3.0 * cp.sin(x / 15.0)
    rng   = cp.random.default_rng(t + 1000)
    return field + rng.standard_normal(size=field.shape) * 0.1

def _humidity(coords, t):
    """Humidity anomaly (%) — coastal gradient + diurnal cycle."""
    x, y  = coords[:, 0], coords[:, 1]
    coast = cp.exp(-x / 30.0) * 40.0          # higher near x=0
    diurn = 10.0 * cp.sin(2.0 * cp.pi * t / 24.0 - y / 25.0)
    rng   = cp.random.default_rng(t + 2000)
    return coast + diurn + rng.standard_normal(size=x.shape) * 0.5


FIELDS = {
    'temperature': (_temperature, '°C',  (10, 20)),
    'pressure':    (_pressure,    'hPa', (-15, 15)),
    'humidity':    (_humidity,    '%',   (0, 80)),
}


class SensorNetwork:
    """N irregularly-spaced sensors sharing one RBF kernel matrix.

    The matrix A depends only on sensor geometry, not on the field values.
    So it is built once and reused across all fields and all time steps —
    the core motivation for the LU-IR pre-factored API.
    """

    def __init__(self, N=8192, D=2, seed=42, scale=100.0, reg=1e-2):
        self.N      = N
        self.reg    = reg
        self.coords = synthetic_coords(N, D=D, seed=seed, scale=scale)
        self._A     = None
        self._gamma = None

    def build_kernel(self, chunk=1024):
        """Build the shared N×N RBF kernel matrix.  Returns (A, gamma).

        A is a (N, N) FP64 CuPy array in Fortran order, ready for MPDOK.
        """
        A, gamma = build_rbf_kernel(self.coords, reg=self.reg, chunk=chunk)
        self._A     = A
        self._gamma = gamma
        return A, gamma

    @property
    def A(self):
        if self._A is None:
            raise RuntimeError("Call build_kernel() first.")
        return self._A

    def field(self, name, t):
        """Return the RHS vector b for field `name` at time step t.

        Args:
            name: 'temperature', 'pressure', or 'humidity'
            t:    Integer time step (0-based, e.g. hours)

        Returns:
            b: (N,) FP64 CuPy array — sensor readings at time t
        """
        if name not in FIELDS:
            raise ValueError(f"Unknown field '{name}'. Choose from: {list(FIELDS)}")
        fn, _, _ = FIELDS[name]
        return fn(self.coords, t).astype(cp.float64)

    def all_fields(self, t):
        """Return dict of all three field vectors at time t."""
        return {name: self.field(name, t) for name in FIELDS}

    def field_range(self, name):
        """Approximate physical range tuple (lo, hi) for display scaling."""
        return FIELDS[name][2]

    def field_unit(self, name):
        return FIELDS[name][1]
