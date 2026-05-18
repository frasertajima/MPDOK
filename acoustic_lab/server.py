#!/usr/bin/env python3
"""
Acoustic Scattering Lab — interactive 2D Helmholtz BEM server.

Features: multiple draggable obstacles, Fortran LU-IR solver, k/α sweep,
scattered-power metric, Nelder-Mead position optimiser.

Usage:
    cd MPDOK/acoustic_lab && python server.py
    Open http://localhost:8766/
"""

import asyncio
import json
import struct
import threading
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from acoustic_solver import (
    AcousticSolver, compute_scattered_power,
    NX, NY, DOMAIN, HAS_FORTRAN,
)

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
except ImportError:
    raise SystemExit("pip install fastapi uvicorn")

PORT    = 8766
WEB_DIR = Path(__file__).parent
app     = FastAPI(title="Acoustic Scattering Lab")

FIELD_BYTES = NX * NY * 4
MASK_BYTES  = NX * NY


# ── Shared state ──────────────────────────────────────────────────────────────

class LabState:
    def __init__(self):
        self.lock = threading.Lock()
        self._next_id = 1

        self.shapes = [
            {"id": 0, "type": "ellipse", "cx": 0.0, "cy": 0.0,
             "params": {"a": 1.5, "b": 0.8}}
        ]

        self.k           = 8.0
        self.alpha       = 0.0
        self.n_panels    = 200
        self.solver_type = "scipy"
        self.use_gpu     = True

        self.sweep_k     = False
        self.sweep_alpha = False
        self.dk          = 0.08
        self.dalpha      = 0.03
        self.k_min, self.k_max = 1.0, 30.0

        # Output fields
        self.p_re        = np.zeros((NY, NX), dtype=np.float32)
        self.p_im        = np.zeros((NY, NX), dtype=np.float32)
        self.mask        = np.zeros((NY, NX), dtype=np.uint8)
        self.boundaries  = []
        self.shape_ids   = []
        self.n_rec       = 200
        self.solver_used = "scipy"
        self.field_power = 1.0    # ||p_total||²   / ||p_inc||²  (0=dead zone)
        self.scat_power  = 1.0    # ||p_scattered||² / ||p_inc||²  (0=cloak)
        self.opt_mode    = "field"  # "field" = minimise |p_total|², "scatter" = cloak

        self.frame_number = 0
        self.solve_ms     = 0.0
        self.dirty        = True
        self.running      = True

        # Optimiser
        self.optimising   = False
        self.opt_iter     = 0
        self.opt_best     = 1.0
        self.opt_max_iter = 300

        self.pending: dict = {}

    def next_id(self):
        i = self._next_id;  self._next_id += 1;  return i


state = LabState()


# ── Simulation thread ─────────────────────────────────────────────────────────

def simulation_loop():
    solver = AcousticSolver()

    while state.running:
        t0 = time.perf_counter()

        with state.lock:
            p = state.pending

            for key, cast in [("k", float), ("alpha", float),
                               ("n_panels", int), ("use_gpu", bool),
                               ("solver_type", str), ("dk", float), ("dalpha", float)]:
                if key in p:
                    val = cast(p.pop(key))
                    if key == "k":        val = max(state.k_min, min(state.k_max, val))
                    if key == "n_panels": val = max(60, min(500, val))
                    setattr(state, key, val)
                    if key not in ("dk", "dalpha", "use_gpu", "solver_type"):
                        state.dirty = True

            if "solver_type" in p or "use_gpu" in p:
                state.dirty = True

            for sw in ("sweep_k", "sweep_alpha"):
                if sw in p: setattr(state, sw, bool(p.pop(sw)))

            if "add_shape" in p:
                sh = p.pop("add_shape");  sh["id"] = state.next_id()
                state.shapes.append(sh);  state.dirty = True

            if "remove_shape" in p:
                rid = int(p.pop("remove_shape"))
                state.shapes = [s for s in state.shapes if s["id"] != rid]
                state.dirty = True

            if "move_shape" in p:
                mv = p.pop("move_shape"); sid = int(mv["id"])
                for s in state.shapes:
                    if s["id"] == sid:
                        s["cx"] = float(mv["cx"]); s["cy"] = float(mv["cy"]); break
                state.dirty = True

            if "update_shape" in p:
                upd = p.pop("update_shape"); sid = int(upd["id"])
                for s in state.shapes:
                    if s["id"] == sid:
                        if "type"   in upd: s["type"] = upd["type"]
                        if "params" in upd: s["params"].update(upd["params"])
                        break
                state.dirty = True

            if "opt_mode" in p:
                state.opt_mode = p.pop("opt_mode")

            if "start_optimize" in p:
                p.pop("start_optimize")
                if not state.optimising:
                    state.optimising = True
                    state.opt_iter   = 0
                    state.opt_best   = state.field_power if state.opt_mode == "field" else state.scat_power
                    shapes_snap = [dict(s, params=dict(s["params"])) for s in state.shapes]
                    opt_thread = threading.Thread(
                        target=_optimise,
                        args=(solver, shapes_snap, state.opt_mode),
                        daemon=True
                    )
                    opt_thread.start()

            if "stop_optimize" in p:
                p.pop("stop_optimize");  state.optimising = False

            if state.sweep_k:
                state.k = state.k_min if state.k + state.dk > state.k_max else state.k + state.dk
                state.dirty = True
            if state.sweep_alpha:
                state.alpha = (state.alpha + state.dalpha) % (2 * np.pi)
                state.dirty = True

            dirty       = state.dirty and not state.optimising
            k, alpha    = state.k, state.alpha
            n_panels    = state.n_panels
            solver_type = state.solver_type
            use_gpu     = state.use_gpu
            shapes      = [dict(s, params=dict(s["params"])) for s in state.shapes]

        if dirty:
            ts = time.perf_counter()
            try:
                p_re, p_im, mask, bpts, n_rec, su, fp, sp = solver.solve(
                    shapes, n_panels, k, alpha, solver_type, use_gpu
                )
                with state.lock:
                    state.p_re = p_re;  state.p_im = p_im;  state.mask = mask
                    state.boundaries  = bpts
                    state.shape_ids   = [s["id"] for s in shapes]
                    state.n_rec       = n_rec
                    state.solver_used = su
                    state.field_power = fp
                    state.scat_power  = sp
                    state.frame_number += 1
                    state.solve_ms    = (time.perf_counter() - ts) * 1000
                    state.dirty       = False
            except Exception as exc:
                print(f"[solver] {exc}")
                with state.lock:
                    state.dirty = False

        elapsed = time.perf_counter() - t0
        if elapsed < 1/30:
            time.sleep(1/30 - elapsed)


# ── Nelder-Mead optimiser (runs in its own thread) ────────────────────────────

def _optimise(solver, shapes_snap, opt_mode="field"):
    """Minimise field_power (dead zone) or scattered_power (cloak) over positions."""
    shape_ids = [s["id"] for s in shapes_snap]
    x0 = np.array([[s["cx"], s["cy"]] for s in shapes_snap], dtype=float).ravel()

    # bounds: keep shapes within 80% of domain
    lim = DOMAIN * 0.80

    def objective(x):
        if not state.optimising:
            raise StopIteration

        # Enforce bounds with heavy penalty (Nelder-Mead ignores bounds natively)
        shapes = [dict(s, params=dict(s["params"])) for s in shapes_snap]
        penalty = 0.0
        for i, s in enumerate(shapes):
            cx, cy = float(x[2*i]), float(x[2*i+1])
            # soft wall
            for v, lim_ in [(cx, lim), (cy, lim)]:
                if abs(v) > lim_:
                    penalty += (abs(v) - lim_) ** 2 * 10
            s["cx"] = cx;  s["cy"] = cy

        with state.lock:
            k, alpha    = state.k, state.alpha
            n_panels    = state.n_panels
            solver_type = state.solver_type
            use_gpu     = state.use_gpu

        try:
            p_re, p_im, mask, bpts, n_rec, su, fp, sp = solver.solve(
                shapes, n_panels, k, alpha, solver_type, use_gpu
            )
        except Exception:
            return 1e6

        metric = fp if opt_mode == "field" else sp
        total  = metric + penalty

        with state.lock:
            state.opt_iter += 1
            if total < state.opt_best:
                state.opt_best   = total
                state.p_re       = p_re;  state.p_im = p_im;  state.mask = mask
                state.boundaries = bpts
                state.shape_ids  = [s["id"] for s in shapes]
                state.field_power = fp
                state.scat_power  = sp
                state.frame_number += 1
                state.solve_ms = 0.0
                for i, sid in enumerate(shape_ids):
                    for ms in state.shapes:
                        if ms["id"] == sid:
                            ms["cx"] = float(x[2*i]);  ms["cy"] = float(x[2*i+1]);  break

        return total

    try:
        minimize(
            objective, x0, method="Nelder-Mead",
            options={"maxiter": state.opt_max_iter, "xatol": 0.02, "fatol": 5e-4,
                     "adaptive": True},
        )
    except StopIteration:
        pass
    except Exception as exc:
        print(f"[optimiser] {exc}")
    finally:
        with state.lock:
            state.optimising = False


# ── Binary frame packer ───────────────────────────────────────────────────────
#
# Header (36 bytes):
#   frame(i32) solve_ms(f32) k(f32) alpha(f32) n_shapes(i32)
#   n_rec(i32) flags(i32) scat_power(f32) opt_iter(i32)
#
# flags: bit0=sweep_k  bit1=sweep_alpha  bit2=has_fortran
#        bit3=fortran_active  bit4=optimising
#
def pack_frame() -> bytes:
    with state.lock:
        fn   = state.frame_number;   ms    = state.solve_ms
        k    = state.k;              alpha  = state.alpha
        nsh  = len(state.boundaries); nrec  = state.n_rec
        fp   = state.field_power;    sp     = state.scat_power
        oi   = state.opt_iter;       ob     = state.opt_best
        om   = 1 if state.opt_mode == "field" else 0   # 1=field, 0=scatter
        p_re = state.p_re;           p_im  = state.p_im
        mask = state.mask
        bds  = list(state.boundaries)
        sids = list(state.shape_ids)
        flags = (
            (1 if state.sweep_k     else 0) |
            (2 if state.sweep_alpha else 0) |
            (4 if HAS_FORTRAN       else 0) |
            (8 if (state.solver_type == "fortran" and HAS_FORTRAN) else 0) |
            (16 if state.optimising else 0)
        )

    # Header: 11 × 4 = 44 bytes
    # frame solve_ms k alpha n_shapes n_rec flags field_power scat_power opt_iter opt_best opt_mode(i32)
    header = struct.pack("<i f f f i i i f f i f i",
                         fn, ms, k, alpha, nsh, nrec, flags, fp, sp, oi, ob, om)

    parts = [header, p_re.tobytes(), p_im.tobytes(), mask.tobytes()]
    for sid, bd in zip(sids, bds):
        parts.append(struct.pack("<ii", sid, len(bd)))
    for bd in bds:
        parts.append(bd.tobytes())
    return b"".join(parts)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    last_frame = -1

    with state.lock:
        caps = {"has_fortran": HAS_FORTRAN,
                "shapes": list(state.shapes)}
    await ws.send_text(json.dumps(caps))

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.005)
                cmd = json.loads(msg)
                with state.lock:
                    state.pending.update(cmd)
            except asyncio.TimeoutError:
                pass

            with state.lock:
                cur = state.frame_number

            if cur != last_frame:
                await ws.send_bytes(pack_frame())
                last_frame = cur
            else:
                await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        pass


@app.get("/")
async def root():
    return FileResponse(WEB_DIR / "index.html")


@app.on_event("startup")
async def startup():
    print("=" * 60)
    print(f"  Acoustic Scattering Lab")
    print(f"  Fortran LU-IR: {'enabled' if HAS_FORTRAN else 'disabled'}")
    print("=" * 60)
    threading.Thread(target=simulation_loop, daemon=True).start()
    print(f"\033[92m  Open: http://localhost:{PORT}/\033[0m")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
