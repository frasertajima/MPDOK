"""
Download and preprocess NOAA GHCN-Daily data for the kriging real-world demo.

Data sources (no API key required):
  - Station inventory: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
  - Annual observations: https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{YEAR}.csv.gz

Output: data/noaa_tmax_{DATE}.parquet
  columns: lat, lon, tmax_C   (station coordinates + temperature)

Usage:
    conda run -n py314 python fetch_noaa.py --date 2024-07-04
    conda run -n py314 python fetch_noaa.py --date 2024-01-15 --region us
"""

import argparse
import gzip
import io
import os
import time

import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

STATIONS_URL = 'https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt'
ANNUAL_URL   = 'https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{year}.csv.gz'

# Fixed-width column specs for ghcnd-stations.txt
STATION_COLSPECS = [(0,11), (12,20), (21,30), (31,37), (38,40), (41,71)]
STATION_NAMES    = ['id', 'lat', 'lon', 'elev', 'state', 'name']


# ── station inventory ─────────────────────────────────────────────────────────

def fetch_station_inventory(cache_path=None, verbose=True):
    """Download and parse the GHCN-Daily station inventory.

    Returns a DataFrame with columns: id, lat, lon, elev, state, name.
    """
    cache_path = cache_path or os.path.join(DATA_DIR, 'ghcnd_stations.parquet')

    if os.path.exists(cache_path):
        if verbose:
            print(f'  [cache] loading station inventory from {cache_path}')
        return pd.read_parquet(cache_path)

    if verbose:
        print('  Downloading GHCN-Daily station inventory …', flush=True)

    t0 = time.perf_counter()
    r = requests.get(STATIONS_URL, timeout=60)
    r.raise_for_status()

    df = pd.read_fwf(
        io.StringIO(r.text),
        colspecs=STATION_COLSPECS,
        names=STATION_NAMES,
        dtype={'id': str, 'state': str, 'name': str},
    )
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    df = df[(df['lat'] != 0.0) | (df['lon'] != 0.0)]   # remove (0,0) placeholders

    df.to_parquet(cache_path, index=False)
    if verbose:
        print(f'  {len(df):,} stations  ({time.perf_counter()-t0:.1f}s)')
    return df


def filter_region(df, region='conus'):
    """Subset to a geographic region.

    Regions:
        'conus'     : contiguous US (lat 24-50, lon -125 to -65)
        'us'        : all US territories (state code not empty)
        'northam'   : US + Canada + Mexico
        'global'    : no filter
    """
    if region == 'conus':
        mask = (
            (df['lat'] >= 24.0) & (df['lat'] <= 50.0) &
            (df['lon'] >= -125.0) & (df['lon'] <= -65.0)
        )
    elif region == 'us':
        mask = df['state'].str.strip().str.len() == 2
    elif region == 'northam':
        mask = (
            (df['lat'] >= 14.0) & (df['lat'] <= 72.0) &
            (df['lon'] >= -170.0) & (df['lon'] <= -50.0)
        )
    else:
        mask = pd.Series(True, index=df.index)

    return df[mask].copy()


# ── annual observations ───────────────────────────────────────────────────────

def fetch_observations(date, element='TMAX', region='conus',
                       annual_cache=None, verbose=True):
    """Stream the GHCN-Daily by_year file and extract one date's observations.

    The file is sorted by station ID then date within each station — we must
    scan the whole file to collect all records for the target date.  The
    compressed file is typically 50-200 MB; streaming avoids full decompression.

    Args:
        date:    'YYYY-MM-DD' string.
        element: 'TMAX', 'TMIN', 'TAVG', 'PRCP', etc.
        region:  station filter passed to filter_region().

    Returns:
        DataFrame with columns: id, lat, lon, value
    """
    year = date[:4]
    target = date.replace('-', '')   # YYYYMMDD

    out_path = os.path.join(DATA_DIR, f'noaa_{element.lower()}_{date}_{region}.parquet')
    if os.path.exists(out_path):
        if verbose:
            print(f'  [cache] loading observations from {out_path}')
        return pd.read_parquet(out_path)

    # ── station coordinates ──────────────────────────────────────────────────
    stations = fetch_station_inventory(verbose=verbose)
    stations = filter_region(stations, region)
    sta_idx  = stations.set_index('id')
    if verbose:
        print(f'  Region stations: {len(sta_idx):,}')

    # ── stream annual observations file ──────────────────────────────────────
    url = ANNUAL_URL.format(year=year)
    annual_cache = annual_cache or os.path.join(DATA_DIR, f'ghcnd_{year}.csv.gz')

    if not os.path.exists(annual_cache):
        if verbose:
            print(f'  Downloading {url} …', flush=True)
        t0 = time.perf_counter()
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(annual_cache, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if verbose and total:
                        pct = downloaded * 100 // total
                        mb  = downloaded // 1024 // 1024
                        print(f'    {mb} MB / {total//1024//1024} MB  ({pct}%)',
                              end='\r', flush=True)
        if verbose:
            print(f'\n  Download complete in {time.perf_counter()-t0:.1f}s')
    else:
        if verbose:
            print(f'  [cache] using {annual_cache}')

    # ── parse: filter to target date + element ───────────────────────────────
    if verbose:
        print(f'  Scanning {year}.csv.gz for {element} on {date} …', flush=True)

    t0 = time.perf_counter()
    records = []
    with gzip.open(annual_cache, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split(',')
            if len(parts) < 4:
                continue
            sid, obs_date, elem = parts[0], parts[1], parts[2]
            if obs_date != target or elem != element:
                continue
            if sid not in sta_idx.index:
                continue
            try:
                val = float(parts[3])
            except ValueError:
                continue
            if val in (-9999, -9999.0):   # GHCN missing value flag
                continue
            records.append({'id': sid, 'raw_value': val})

    if verbose:
        print(f'  Parsed {len(records):,} records in {time.perf_counter()-t0:.1f}s')

    if not records:
        raise ValueError(f'No {element} records found for {date} in {region} region.')

    obs = pd.DataFrame(records)
    obs = obs.merge(sta_idx[['lat', 'lon']], left_on='id', right_index=True)

    # GHCN-Daily stores TMAX/TMIN in tenths of a degree C
    if element in ('TMAX', 'TMIN', 'TAVG'):
        obs['value'] = obs['raw_value'] / 10.0   # → degrees Celsius
    else:
        obs['value'] = obs['raw_value']

    obs = obs[['id', 'lat', 'lon', 'value']].drop_duplicates('id')
    obs.to_parquet(out_path, index=False)

    if verbose:
        print(f'  Saved {len(obs):,} station records to {out_path}')
        print(f'  Value range: {obs["value"].min():.1f} – {obs["value"].max():.1f}')

    return obs


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date',    default='2024-07-04',
                        help='Observation date YYYY-MM-DD')
    parser.add_argument('--element', default='TMAX',
                        help='GHCN element: TMAX, TMIN, PRCP, SNWD, …')
    parser.add_argument('--region',  default='conus',
                        choices=['conus', 'us', 'northam', 'global'])
    args = parser.parse_args()

    print(f'\nFetching {args.element} for {args.date}  region={args.region}\n')
    obs = fetch_observations(args.date, element=args.element, region=args.region)
    print(f'\nReady: {len(obs):,} stations')
    print(obs[['lat', 'lon', 'value']].describe())
