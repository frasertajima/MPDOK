#!/usr/bin/env python3
"""
Fleet Tracker — MPDOK GMRES-IR, auto-scaling from N=6k to N=64k+.

Two independent loops share state through a thread lock:

  vehicle_loop  — advances fleet at ANIM_HZ regardless of solve speed.
                  This keeps the dot animation fluid even during long OOC cycles.

  solve_loop    — builds the N×N SPD matrix and solves with the best available
                  backend, then updates the reconstructed field.

Solver auto-selection:
  FP32 matrix ≤ 60 % VRAM  → standard MPDOKSolver (Fortran GMRES-IR, on-device)
  FP32 matrix  > 60 % VRAM  → MPDOKOOCSolver      (Python GMRES-IR, RAM tiles)

Usage:
    cd MPDOK/SPD_matrices_vehicle_tracking && python server.py
    Open http://localhost:8787/

Change N_VEHICLES at the top to scale up or down.
"""

import asyncio
import json
import struct
import sys
import threading
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np

_SELF = Path(__file__).parent
_MPDOK = _SELF.parent
sys.path.insert(0, str(_SELF))
sys.path.insert(0, str(_MPDOK))

import cupy as cp
from mpdok_ooc import MPDOKOOCSolver

from enkf_solver import _DEFAULT_GAMMA, _DEFAULT_REG, FleetAssimilator
from fleet_sim import CITY_SIZE, Fleet, Incident, TrafficField

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
except ImportError:
    raise SystemExit("pip install fastapi uvicorn")

# ── Configuration ─────────────────────────────────────────────────────────────
PORT = 8787
N_VEHICLES = 256_000  # ← change me
NX, NY = 64, 64  # reconstruction grid
ANIM_HZ = 5  # vehicle animation rate (Hz) — independent of solve speed
OOC_TILE = 512  # rows per OOC tile (safe for 4 GB VRAM)
OOC_RESTART = 20  # inner GMRES restart (less VRAM for Krylov basis)
OOC_OUTER = 5  # outer iterations
OOC_TOL = 5e-3  # tighter than observation noise → good enough visually
OOC_SSD_PATH = "/tmp/mpdok_kernel.bin"  # ← SSD streaming scratch file

N_TRAIL = 5  # number of tracked (highlighted) vehicles
TRAIL_LEN = 80  # positions kept per vehicle trail

WEB_DIR = _SELF
app = FastAPI(title="Fleet Tracker")

# ── Solver auto-selection ─────────────────────────────────────────────────────
_vram_total = cp.cuda.Device(0).mem_info[1]
_fp32_bytes = N_VEHICLES**2 * 4
_ram_bytes = _fp32_bytes  # same size needed for store='ram'
USE_OOC = _fp32_bytes > _vram_total * 0.60
# Use SSD streaming when matrix also exceeds available system RAM
import psutil as _psutil
USE_SSD = USE_OOC and _ram_bytes > _psutil.virtual_memory().available * 0.80
OOC_STORE = "ssd" if USE_SSD else "ram"

SOLVER_MODE = 2 if USE_SSD else (1 if USE_OOC else 0)  # 0=standard, 1=ooc-ram, 2=ooc-ssd


# ── Shared state ──────────────────────────────────────────────────────────────


class SimState:
    def __init__(self):
        self.lock = threading.Lock()

        self.fleet = Fleet(N=N_VEHICLES, seed=0)
        self.field = TrafficField()
        self.assimilator = FleetAssimilator()  # owns MPDOKSolver handle
        self.ooc_solver = MPDOKOOCSolver(tile_rows=OOC_TILE) if USE_OOC else None

        xs = np.linspace(0.0, CITY_SIZE, NX)
        ys = np.linspace(0.0, CITY_SIZE, NY)
        gx, gy = np.meshgrid(xs, ys)
        self.query_grid = np.column_stack([gx.ravel(), gy.ravel()])

        # -- Vehicle state (written by vehicle_loop) --
        self.positions = self.fleet.positions.astype(np.float32)
        self.y_vals = np.zeros(N_VEHICLES, dtype=np.float32)  # per-vehicle congestion
        self.pending_y = None  # latest measurements, ready for solver
        self.pending_pos = None  # positions matching pending_y

        # -- Field state (written by solve_loop) --
        self.true_field = np.zeros(NX * NY, dtype=np.float32)
        self.recon_field = np.zeros(NX * NY, dtype=np.float32)

        # -- Metrics --
        self.frame_number = 0
        self.step_count = 0  # vehicle steps taken
        self.assim_step = 0  # completed solve cycles
        self.build_s = 0.0
        self.solve_s = 0.0
        self.cycle_s = 0.0  # build + solve combined
        self.outer_iter = 0  # OOC outer iteration in progress
        self.scipy_ms = 0.0
        self.rel_error = 0.0
        self.baseline_speedup = 1.0
        self.elapsed_assim_s = 0.0  # seconds since last field update
        self._assim_ts = time.perf_counter()

        # -- Capability flags (set at startup) --
        self.cupy_oom = False
        self.scipy_est_s = 0.0  # estimated scipy time in seconds

        # -- Trails (written by vehicle_loop, read by pack_frame) --
        _rng = np.random.default_rng(7)
        self.trail_indices = _rng.choice(
            N_VEHICLES, size=N_TRAIL, replace=False
        ).tolist()
        self.trail_replace_idx = 0
        # Each deque holds (x, y) float32 tuples; filled with start pos initially
        self.trail_history = []
        for vi in self.trail_indices:
            p = self.fleet.positions[vi].astype(np.float32)
            self.trail_history.append(
                deque([(p[0], p[1])] * TRAIL_LEN, maxlen=TRAIL_LEN)
            )

        # -- Control --
        self.running = True
        self.paused = False
        self.pending: dict = {}


state = SimState()


# ── Startup benchmark ─────────────────────────────────────────────────────────


def _run_startup_benchmark():
    """
    1. Probe CuPy (expect OOM for large N).
    2. Estimate SciPy time by scaling from N=6k result.
    3. Time one MPDOK solve (standard or OOC) to set baseline_speedup.
    """
    fp64_gb = N_VEHICLES**2 * 8 / 1e9
    fp32_gb = N_VEHICLES**2 * 4 / 1e9
    vram_gb = _vram_total / 1e9
    print(
        f"  Matrix: {N_VEHICLES:,}×{N_VEHICLES:,}  "
        f"FP64={fp64_gb:.1f}GB  FP32={fp32_gb:.1f}GB  VRAM={vram_gb:.1f}GB"
    )
    _smode = "OOC-SSD" if USE_SSD else ("OOC-RAM" if USE_OOC else "standard MPDOK")
    print(f"  Solver: {_smode}")

    # ── CuPy probe ────────────────────────────────────────────────────────────
    print("\n  [CuPy probe] attempting cp.linalg.solve ...")
    try:
        A_test = cp.eye(min(N_VEHICLES, 100), dtype=cp.float64)
        _A_big = cp.zeros((N_VEHICLES, N_VEHICLES), dtype=cp.float32)
        del _A_big
        state.cupy_oom = False
        print("  [CuPy probe] allocation succeeded")
    except cp.cuda.memory.OutOfMemoryError:
        state.cupy_oom = True
        print(
            f"  [CuPy probe] ✗ OutOfMemoryError — "
            f"{fp32_gb:.1f} GB FP32 > {vram_gb:.1f} GB VRAM"
        )
    cp.get_default_memory_pool().free_all_blocks()

    # ── SciPy estimate (scale from N=6k: 1.6s → O(N³)) ──────────────────────
    # N=6000 SciPy: ~1.6s → scale: (N/6000)^3 × 1.6
    scipy_ref_s, N_ref = 1.6, 6_000
    est = scipy_ref_s * (N_VEHICLES / N_ref) ** 3
    state.scipy_est_s = est
    print(f"  [SciPy estimate] ~{_fmt_time(est)} (O(N³) from N={N_ref:,} baseline)")

    # ── MPDOK baseline ────────────────────────────────────────────────────────
    print(f"\n  [MPDOK] warming up ...")
    fleet_tmp = Fleet(N=N_VEHICLES, seed=0)
    field_tmp = TrafficField()
    y = fleet_tmp.measure(field_tmp)
    coords = cp.asarray(fleet_tmp.positions, dtype=cp.float64)

    if USE_OOC:
        ooc = MPDOKOOCSolver(tile_rows=OOC_TILE)
        t0 = time.perf_counter()
        ooc.build(
            coords, gamma=_DEFAULT_GAMMA, reg=_DEFAULT_REG,
            store=OOC_STORE, path=OOC_SSD_PATH if USE_SSD else None, verbose=False
        )
        build_s = time.perf_counter() - t0
        print(f"  [MPDOK OOC] build: {build_s:.1f}s")

        t0 = time.perf_counter()
        b = cp.asarray(y, dtype=cp.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x = ooc.solve(
                b,
                tol=OOC_TOL,
                maxiter_outer=OOC_OUTER,
                restart=OOC_RESTART,
                verbose=True,
            )
        solve_s = time.perf_counter() - t0
        cycle_s = build_s + solve_s
        ooc.free()
    else:
        a = state.assimilator
        # warm CUDA handles
        _aw = cp.eye(128, dtype=cp.float64, order="F")
        _ = a._solver.solve(_aw, cp.ones(128, dtype=cp.float64))
        cp.cuda.Stream.null.synchronize()
        A, _ = a.build_matrix(fleet_tmp.positions)
        t0 = time.perf_counter()
        xm = a.solve_mpdok(A, y)
        cp.cuda.Stream.null.synchronize()
        mpdok_ms = (time.perf_counter() - t0) * 1e3
        build_s = 0.039  # approx N=6k build in seconds
        solve_s = mpdok_ms / 1e3
        cycle_s = solve_s

        # compare against scipy for the speedup badge
        t0 = time.perf_counter()
        xs = a.solve_scipy(A, y)
        scipy_ms = (time.perf_counter() - t0) * 1e3
        state.scipy_ms = scipy_ms
        rel = float(np.linalg.norm(cp.asnumpy(xm) - xs) / np.linalg.norm(xs))
        state.rel_error = rel
        spd = scipy_ms / mpdok_ms
        state.baseline_speedup = spd
        print(
            f"  [MPDOK std] {mpdok_ms:.0f}ms  SciPy {scipy_ms:.0f}ms  "
            f"speedup {spd:.1f}×  rel_err {rel:.2e}"
        )

    if USE_OOC:
        # Speedup vs SciPy estimated
        spd = state.scipy_est_s / cycle_s if cycle_s > 0 else 1.0
        state.baseline_speedup = spd
        state.cycle_s = cycle_s
        print(
            f"  [MPDOK OOC] cycle {_fmt_time(cycle_s)}  "
            f"vs SciPy est {_fmt_time(state.scipy_est_s)}  "
            f"speedup {spd:.1f}×"
        )

    del fleet_tmp


def _fmt_time(s):
    if s < 2:
        return f"{s:.2f}s"
    if s < 120:
        return f"{s:.0f}s"
    return f"{s / 60:.1f}min"


# ── Vehicle animation loop ────────────────────────────────────────────────────


def vehicle_loop():
    """Advances fleet at ANIM_HZ.  Queues latest (positions, y) for solver."""
    fleet = state.fleet
    field = state.field
    dt = 1.0 / ANIM_HZ

    while state.running:
        t0 = time.perf_counter()
        with state.lock:
            paused = state.paused

        if paused:
            time.sleep(dt)
            continue

        field._prune_incidents(fleet.t)
        fleet.step(field)
        y = fleet.measure(field)

        # True field on grid (lightweight; never involves the SPD solve)
        true_flat = field.evaluate(state.query_grid, fleet.t)
        true_grid = true_flat.reshape(NY, NX).astype(np.float32)

        pos32 = fleet.positions.astype(np.float32)
        y32 = np.clip(y, 0.0, 1.0).astype(np.float32)
        with state.lock:
            state.positions = pos32
            state.y_vals = y32
            state.pending_y = y
            state.pending_pos = fleet.positions.copy()
            state.true_field = true_grid.ravel()
            state.step_count += 1
            state.elapsed_assim_s = time.perf_counter() - state._assim_ts
            state.frame_number += 1
            for k, vi in enumerate(state.trail_indices):
                state.trail_history[k].append((pos32[vi, 0], pos32[vi, 1]))

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


# ── Solve loop ────────────────────────────────────────────────────────────────


class _OOCProgress:
    """Monkey-patch hook: updates state.outer_iter on each outer residual."""

    def __init__(self):
        self.outer = 0

    def outer_start(self, i):
        self.outer = i
        with state.lock:
            state.outer_iter = i


def solve_loop():
    """Runs one build+solve cycle whenever new measurements are pending."""
    a = state.assimilator

    while state.running:
        # Wait for new measurements from the vehicle loop
        with state.lock:
            y = state.pending_y
            pos = state.pending_pos

        if y is None:
            time.sleep(0.05)
            continue

        with state.lock:
            paused = state.paused
        if paused:
            time.sleep(0.1)
            continue

        t_cycle = time.perf_counter()

        # Consume the pending batch (so next cycle gets fresh data)
        with state.lock:
            state.pending_y = None
            state.pending_pos = None

        coords = cp.asarray(pos, dtype=cp.float64)

        if USE_OOC:
            # ── OOC path ──────────────────────────────────────────────────
            ooc = state.ooc_solver

            t0 = time.perf_counter()
            ooc.build(
                coords,
                gamma=_DEFAULT_GAMMA,
                reg=_DEFAULT_REG,
                store=OOC_STORE,
                path=OOC_SSD_PATH if USE_SSD else None,
                verbose=False,
            )
            build_s = time.perf_counter() - t0

            b = cp.asarray(y, dtype=cp.float64)
            with state.lock:
                state.build_s = build_s
                state.outer_iter = 0

            # Patch solve to report outer iterations
            t0 = time.perf_counter()
            orig_dgemv = ooc._tiled_dgemv_fp64
            _outer_count = [0]

            def _patched_dgemv(v):
                with state.lock:
                    state.outer_iter = _outer_count[0]
                _outer_count[0] += 1
                return orig_dgemv(v)

            ooc._tiled_dgemv_fp64 = _patched_dgemv
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                alpha = ooc.solve(
                    b,
                    tol=OOC_TOL,
                    maxiter_outer=OOC_OUTER,
                    restart=OOC_RESTART,
                    verbose=False,
                )
            ooc._tiled_dgemv_fp64 = orig_dgemv
            solve_s = time.perf_counter() - t0

        else:
            # ── Standard MPDOK path ────────────────────────────────────────
            t0 = time.perf_counter()
            A, _ = a.build_matrix(pos)
            cp.cuda.Stream.null.synchronize()
            build_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            alpha = a.solve_mpdok(A, y)
            cp.cuda.Stream.null.synchronize()
            solve_s = time.perf_counter() - t0

        # ── Reconstruct field ──────────────────────────────────────────────
        # Release any cached VRAM before the chunked cross-kernel pass.
        cp.get_default_memory_pool().free_all_blocks()
        recon_flat = a.reconstruct(alpha, pos, state.query_grid)
        recon_grid = recon_flat.reshape(NY, NX).astype(np.float32)

        cycle_s = time.perf_counter() - t_cycle

        with state.lock:
            state.recon_field = recon_grid.ravel()
            state.build_s = build_s
            state.solve_s = solve_s
            state.cycle_s = cycle_s
            state.assim_step += 1
            state._assim_ts = time.perf_counter()
            state.elapsed_assim_s = 0.0

        if USE_OOC:
            print(
                f"  [assim {state.assim_step}] "
                f"build={build_s:.1f}s  solve={solve_s:.1f}s  "
                f"cycle={cycle_s:.1f}s",
                flush=True,
            )


# ── Binary frame packer ───────────────────────────────────────────────────────
#
# Header (20 × 4 = 80 bytes), little-endian:
#   ii     : frame(i32)  step(i32)
#   fffff  : build_s(f32)  solve_s(f32)  cycle_s(f32)  rel_err(f32)  speedup(f32)
#   iiii   : N(i32)  NX(i32)  NY(i32)  paused(i32)
#   ffff   : city(f32)  elapsed_assim(f32)  scipy_est_s(f32)  scipy_ms(f32)
#   iiii   : solver_mode(i32)  cupy_oom(i32)  outer_iter(i32)  assim_step(i32)
#
_HDR_FMT = "<" + "ii" + "fffff" + "iiii" + "ffff" + "iiii"  # 20 values × 4 B = 80 B


def pack_frame() -> bytes:
    with state.lock:
        fn = state.frame_number
        step = state.step_count
        bs = float(state.build_s)
        ss = float(state.solve_s)
        cs = float(state.cycle_s)
        rerr = float(state.rel_error)
        spdup = float(state.baseline_speedup)
        N = int(state.fleet.N)
        paused = int(state.paused)
        city = float(CITY_SIZE)
        ela = float(state.elapsed_assim_s)
        spest = float(state.scipy_est_s)
        spms = float(state.scipy_ms)
        smode = int(SOLVER_MODE)
        coom = int(state.cupy_oom)
        oiter = int(state.outer_iter)
        astep = int(state.assim_step)
        tf = state.true_field.copy()
        rf = state.recon_field.copy()
        pos = state.positions.copy()
        yv = state.y_vals.copy()
        t_hist = [list(state.trail_history[k]) for k in range(N_TRAIL)]
        incidents = [
            (i.x, i.y, i.strength(state.fleet.t)) for i in state.field.incidents
        ]

    header = struct.pack(
        _HDR_FMT,
        fn,
        step,
        bs,
        ss,
        cs,
        rerr,
        spdup,
        N,
        NX,
        NY,
        paused,
        city,
        ela,
        spest,
        spms,
        smode,
        coom,
        oiter,
        astep,
    )

    # Trail data: int32 n_trail, int32 trail_len, then N_TRAIL×TRAIL_LEN×2 float32
    trail_arr = np.zeros((N_TRAIL, TRAIL_LEN, 2), dtype=np.float32)
    for k, hist in enumerate(t_hist):
        for j, (x, y) in enumerate(hist):
            trail_arr[k, j, 0] = x
            trail_arr[k, j, 1] = y
    trail_header = struct.pack("<ii", N_TRAIL, TRAIL_LEN)

    # Incident data: int32 count, then float32 (x, y, strength) per incident
    inc_arr = (
        np.array(incidents, dtype=np.float32).reshape(-1, 3)
        if incidents
        else np.zeros((0, 3), dtype=np.float32)
    )
    inc_header = struct.pack("<i", len(incidents))

    return (
        header
        + tf.tobytes()
        + rf.tobytes()
        + pos.ravel().tobytes()
        + yv.tobytes()
        + trail_header
        + trail_arr.tobytes()
        + inc_header
        + inc_arr.tobytes()
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    last_frame = -1
    caps = {
        "N": N_VEHICLES,
        "city_size": CITY_SIZE,
        "nx": NX,
        "ny": NY,
        "solver_mode": SOLVER_MODE,
        "ooc_outer": OOC_OUTER,
        "cupy_oom": state.cupy_oom,
    }
    await ws.send_text(json.dumps(caps))

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.005)
                cmd = json.loads(msg)
                with state.lock:
                    state.pending.update(cmd)
                    if "paused" in cmd:
                        state.paused = bool(cmd["paused"])
                    if "incident" in cmd:
                        d = cmd["incident"]
                        inc = Incident(
                            x=float(d["sx"]), y=float(d["sy"]), t_start=state.fleet.t
                        )
                        state.field.add_incident(inc)
                    if "clear_incidents" in cmd:
                        state.field.clear_incidents()
                    if "track" in cmd:
                        vi = int(cmd["track"])
                        if 0 <= vi < N_VEHICLES:
                            k = state.trail_replace_idx % N_TRAIL
                            state.trail_replace_idx += 1
                            state.trail_indices[k] = vi
                            p = state.positions[vi]
                            state.trail_history[k] = deque(
                                [(p[0], p[1])] * TRAIL_LEN, maxlen=TRAIL_LEN
                            )
            except asyncio.TimeoutError:
                pass

            with state.lock:
                cur = state.frame_number
            if cur != last_frame:
                await ws.send_bytes(pack_frame())
                last_frame = cur
            else:
                await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        pass


@app.get("/")
async def root():
    return FileResponse(WEB_DIR / "index.html")


@app.on_event("startup")
async def startup():
    fp32_gb = N_VEHICLES**2 * 4 / 1e9
    print("=" * 62)
    _smode2 = "OOC-SSD" if USE_SSD else ("OOC-RAM" if USE_OOC else "Standard")
    print(f"  SPD Fleet Tracker — MPDOK {_smode2}")
    print(f"  N = {N_VEHICLES:,}  matrix = {fp32_gb:.1f} GB FP32")
    print("=" * 62)
    print(f"\033[92m  Open: http://localhost:{PORT}/\033[0m\n")

    # Startup benchmark runs in a background thread so uvicorn binds the port
    # immediately and the browser can connect straight away.  The vehicle and
    # solve loops start as soon as the benchmark finishes.
    def _bg():
        cp.cuda.Device(0).synchronize()
        _run_startup_benchmark()
        threading.Thread(target=vehicle_loop, daemon=True).start()
        threading.Thread(target=solve_loop, daemon=True).start()

    threading.Thread(target=_bg, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
