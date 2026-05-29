"""
Real-world kriging demo: NOAA GHCN-Daily temperature observations, CONUS.

Compares SciPy vs MPDOK on actual irregular station networks.
Generates a publication-quality interpolated TMAX map of the contiguous US.

Usage:
    conda run -n py314 python realworld_demo.py
    conda run -n py314 python realworld_demo.py --date 2024-07-04 --max-n 12000
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import cupy as cp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from MPDOK.kriging.fetch_noaa import fetch_observations
from MPDOK.kriging.kriging_kernel import build_kriging_cov, build_kriging_cov_cpu, estimate_length_scale
from MPDOK.kriging.kriging_solver import OrdinaryKriging, _gpu_memory_reset


# ── data loading ─────────────────────────────────────────────────────────────

def load_stations(date='2024-07-04', element='TMAX', region='conus',
                  max_n=None, seed=42):
    """Load real station data, optionally subsample to max_n."""
    obs = fetch_observations(date, element=element, region=region)

    # Remove outliers (sensor faults, bad QC)
    lo, hi = obs['value'].quantile(0.005), obs['value'].quantile(0.995)
    obs = obs[(obs['value'] >= lo) & (obs['value'] <= hi)].copy()
    obs = obs.reset_index(drop=True)

    if max_n and len(obs) > max_n:
        obs = obs.sample(n=max_n, random_state=seed).reset_index(drop=True)

    print(f'  Stations: {len(obs):,}   '
          f'{element}: {obs["value"].min():.1f}–{obs["value"].max():.1f} °C')
    return obs


# ── timing comparison ─────────────────────────────────────────────────────────

def time_backend(coords_gpu, z_gpu, backend, nugget=1e-6):
    """Run kriging fit on real data and return timing + solver."""
    t0 = time.perf_counter()
    ok = OrdinaryKriging(model='matern32', backend=backend)
    ok.fit(coords_gpu, z_gpu, nugget=nugget)
    elapsed = time.perf_counter() - t0
    ooc = getattr(ok, 'ooc_', False)
    return ok, elapsed, ooc


# ── prediction grid (CONUS bounding box) ─────────────────────────────────────

def make_grid(obs, n_lat=150, n_lon=350):
    """Return a (M, 2) lon/lat grid spanning the observation bounding box."""
    lat_min = float(obs['lat'].min()); lat_max = float(obs['lat'].max())
    lon_min = float(obs['lon'].min()); lon_max = float(obs['lon'].max())
    # small padding
    pad_lat = (lat_max - lat_min) * 0.03
    pad_lon = (lon_max - lon_min) * 0.03
    lats = cp.linspace(lat_min - pad_lat, lat_max + pad_lat, n_lat)
    lons = cp.linspace(lon_min - pad_lon, lon_max + pad_lon, n_lon)
    ll, lo = cp.meshgrid(lats, lons, indexing='ij')
    grid = cp.stack([lo.ravel(), ll.ravel()], axis=1).astype(cp.float64)
    extent = [float(lons[0]), float(lons[-1]), float(lats[0]), float(lats[-1])]
    return grid, (n_lat, n_lon), extent


# ── main ──────────────────────────────────────────────────────────────────────

def run(date='2024-07-04', element='TMAX', region='conus',
        max_n=None, out_dir=None):
    out_dir = out_dir or HERE
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Real-World Kriging: NOAA GHCN-Daily {element}  {date}')
    print(f'{"="*60}')

    # ── load data ─────────────────────────────────────────────────────────────
    obs = load_stations(date, element=element, region=region, max_n=max_n)
    N   = len(obs)

    # coords: (N, 2) array of [lon, lat] in decimal degrees
    coords_np = obs[['lon', 'lat']].values.astype(np.float64)
    z_np      = obs['value'].values.astype(np.float64)
    coords_gpu = cp.asarray(coords_np)
    z_gpu      = cp.asarray(z_np)

    print(f'\n  N = {N:,}   matrix size = {N**2*8/1e9:.2f} GB FP64')

    # ── scipy timing (build + solve) ──────────────────────────────────────────
    print(f'\n  SciPy  ... ', end='', flush=True)
    if N <= 20_000:
        _, t_scipy, _ = time_backend(coords_gpu, z_gpu, 'scipy')
        print(f'{t_scipy:.2f}s  (matrix build + solve)')
        scipy_ok = True
    else:
        print('skipped (N > 20k, would take many minutes)')
        t_scipy = None; scipy_ok = False

    # ── MPDOK timing ──────────────────────────────────────────────────────────
    _gpu_memory_reset()
    print(f'  MPDOK  ... ', end='', flush=True)
    ok_mpdok, t_mpdok, ooc = time_backend(coords_gpu, z_gpu, 'mpdok')
    ooc_tag = ' (OOC/RAM)' if ooc else ''
    print(f'{t_mpdok:.2f}s{ooc_tag}')

    if t_scipy:
        print(f'\n  Speedup: {t_scipy/t_mpdok:.1f}× faster than SciPy')

    # ── interpolate onto prediction grid ─────────────────────────────────────
    print(f'\n  Predicting on grid …', flush=True)
    grid, (n_lat, n_lon), extent = make_grid(obs)
    t0 = time.perf_counter()
    z_grid = ok_mpdok.predict(grid)
    t_pred = time.perf_counter() - t0
    z_grid_np = cp.asnumpy(z_grid).reshape(n_lat, n_lon)
    print(f'  Prediction: {n_lat}×{n_lon} grid in {t_pred:.2f}s')

    # ── plot ──────────────────────────────────────────────────────────────────
    _make_map(obs, z_grid_np, n_lat, n_lon, extent, date, element, N,
              t_mpdok, t_scipy, ooc_tag, out_dir)
    _make_scatter(obs, coords_np, extent, out_dir)

    return {
        'N': N, 'date': date, 'element': element,
        't_scipy': t_scipy, 't_mpdok': t_mpdok, 'ooc': ooc,
        'speedup': t_scipy / t_mpdok if t_scipy else None,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def _make_map(obs, z_grid, n_lat, n_lon, extent, date, element, N,
              t_mpdok, t_scipy, ooc_tag, out_dir):
    """Two-panel figure with cartopy projections and coastlines."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    z_np = obs['value'].values
    vmin, vmax = float(np.percentile(z_np, 2)), float(np.percentile(z_np, 98))
    cmap = 'RdYlBu_r'
    lon0, lon1, lat0, lat1 = extent

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18, 7), facecolor='#0d1117')

    # ── left: observations ───────────────────────────────────────────────────
    ax0 = fig.add_subplot(1, 2, 1, projection=proj, facecolor='#0d1117')
    ax0.set_extent([lon0, lon1, lat0, lat1], crs=proj)
    ax0.add_feature(cfeature.OCEAN, facecolor='#111827', zorder=0)
    ax0.add_feature(cfeature.LAND,  facecolor='#1f2937', zorder=0)
    ax0.add_feature(cfeature.COASTLINE, edgecolor='#6b7280', linewidth=0.5, zorder=2)
    ax0.add_feature(cfeature.BORDERS,   edgecolor='#4b5563', linewidth=0.3, zorder=2)

    sc = ax0.scatter(obs['lon'], obs['lat'],
                     c=z_np, cmap=cmap, s=5, vmin=vmin, vmax=vmax,
                     linewidths=0, rasterized=True,
                     transform=proj, zorder=3)
    ax0.set_title(f'GHCN-Daily Observations  (N = {N:,})\n{element}  ·  {date}',
                  fontsize=11, color='white', pad=8)
    ax0.gridlines(color='#374151', linewidth=0.3, alpha=0.6)
    cb0 = plt.colorbar(sc, ax=ax0, orientation='vertical',
                       fraction=0.03, pad=0.04, shrink=0.85)
    cb0.set_label('Max Temp  (°C)', color='white', fontsize=9)
    cb0.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb0.ax.yaxis.get_ticklabels(), color='white')

    # ── right: kriged surface ────────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 2, projection=proj, facecolor='#0d1117')
    ax1.set_extent([lon0, lon1, lat0, lat1], crs=proj)
    ax1.add_feature(cfeature.OCEAN, facecolor='#111827', zorder=0)

    im = ax1.imshow(z_grid, origin='lower',
                    extent=[lon0, lon1, lat0, lat1],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    transform=proj, aspect='auto',
                    interpolation='bilinear', zorder=1)
    ax1.add_feature(cfeature.COASTLINE, edgecolor='#e5e7eb', linewidth=0.7, zorder=2)
    ax1.add_feature(cfeature.BORDERS,   edgecolor='#9ca3af', linewidth=0.3, zorder=2)

    speedup_str = (f'{t_scipy/t_mpdok:.1f}× faster than SciPy'
                   if t_scipy else 'SciPy skipped (N too large)')
    ax1.set_title(f'Kriged Surface  ·  MPDOK LU-IR{ooc_tag}\n'
                  f'Fit: {t_mpdok:.1f}s  ·  {speedup_str}',
                  fontsize=11, color='white', pad=8)
    ax1.gridlines(color='#374151', linewidth=0.3, alpha=0.6)
    cb1 = plt.colorbar(im, ax=ax1, orientation='vertical',
                       fraction=0.03, pad=0.04, shrink=0.85)
    cb1.set_label('Max Temp  (°C)', color='white', fontsize=9)
    cb1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color='white')

    fig.suptitle(
        'MPDOK Tensor-Core Kriging  ·  10,795 Real NOAA GHCN-Daily Stations  ·  July 4 2024',
        fontsize=13, color='white', y=1.00
    )
    plt.tight_layout(pad=1.5)
    out_path = os.path.join(out_dir, f'realworld_map_{date}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f'  Map saved to {out_path}')
    plt.close(fig)


def _make_scatter(obs, coords_np, extent, out_dir):
    """Station density map."""
    lon0, lon1, lat0, lat1 = extent
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.scatter(coords_np[:, 0], coords_np[:, 1],
               c=coords_np[:, 1], cmap='plasma',
               s=2, alpha=0.6, linewidths=0, rasterized=True)
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_title('GHCN Station Distribution — Irregular, Non-Grid',
                 color='white', fontsize=12)
    ax.set_xlabel('Longitude', color='white')
    ax.set_ylabel('Latitude',  color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'realworld_stations.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0d1117')
    print(f'  Station map saved to {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date',    default='2024-07-04')
    parser.add_argument('--element', default='TMAX')
    parser.add_argument('--region',  default='conus')
    parser.add_argument('--max-n',   type=int, default=None)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    result = run(args.date, args.element, args.region, args.max_n)
    print(f'\nResult: {result}')
