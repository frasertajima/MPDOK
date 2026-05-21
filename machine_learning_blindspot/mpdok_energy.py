#!/usr/bin/env python3
"""
mpdok_energy.py — MPDOK vs static ML on energy price time series

The central question: does MPDOK's dynamic network detection outperform
"standard ML" when the underlying correlation structure changes over time?

Three models:
  Static Ridge  — trained once on 2000-2013, coefficients frozen forever
                  (the "just replicate static analysis at each step" baseline)
  Rolling Ridge — re-fits every month on last LOOKBACK months (Stage 1)
  MPDOK Rolling — rolling + per-feature penalties from resolvent (Stage 2)

Target: WTI crude oil implied log-price, one month at a time
Network: nat gas, gasoline, heating oil, Brent crude, dollar, industrial output

Regime changes expected to stress static model:
  2014-2016  US shale price war — WTI halved in 6 months
  2020       COVID demand collapse
  2022       Ukraine war — nat gas → heating oil cascade dominates
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

_HERE     = os.path.dirname(os.path.abspath(__file__))
_KEY_FILE = os.path.join(_HERE, 'fred_rate_predictor', 'FRED_API_KEY.txt')

ALPHA    = 0.85   # resolvent damping
LOOKBACK = 36     # rolling window months
PENALTY_CAP = 200.0

FETCH_START = '2000-01-01'
TRAIN_END   = '2013-12-31'   # static model trained up to here

# ── Energy series universe ────────────────────────────────────────────────────

UNIVERSE = {
    'NATGAS':   ('MHHNGSP',    'log',  'Henry Hub nat gas ($/MMBtu)'),
    'GASOLINE': ('GASREGCOVW', 'log',  'US regular gasoline ($/gal)'),
    'HEATING':  ('DHOILNYH',   'log',  'No.2 heating oil NY Harbor ($/gal)'),
    'DOLLAR':   ('DTWEXBGS',   'log',  'Broad trade-weighted dollar'),
    'INDPROD':  ('INDPRO',     'log',  'Industrial production'),
    'CFNAI':    ('CFNAI',      'diff', 'Chicago Fed Activity Index'),
}
# Brent excluded: it's essentially WTI re-labelled — including it would give
# the static model a trivial shortcut and mask the real cascade dynamics.
TARGET_CODE = 'DCOILWTICO'   # WTI crude — predict this


# ── FRED fetch ────────────────────────────────────────────────────────────────

def _get_fred():
    from fredapi import Fred
    if not os.path.exists(_KEY_FILE):
        raise FileNotFoundError(f'FRED API key not found: {_KEY_FILE}')
    return Fred(api_key=open(_KEY_FILE).read().strip())


def _fetch_one(fred, code):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = fred.get_series(code, observation_start=FETCH_START)
    if s is None or len(s) == 0:
        return pd.Series(dtype=float, name=code)
    s.index = pd.to_datetime(s.index)
    return s.resample('ME').mean().ffill(limit=3).rename(code)


def fetch_energy():
    print(f'  Fetching {len(UNIVERSE)} energy series + WTI target…')
    fred = _get_fred()

    levels = {}
    for node_id, (code, transform, desc) in UNIVERSE.items():
        try:
            s = _fetch_one(fred, code)
            if len(s) >= 12:
                levels[node_id] = s
                print(f'    {node_id:<10} {code:<16} {len(s)} months  ✓  {desc}')
            else:
                print(f'    {node_id:<10} {code:<16} too short — skipped')
        except Exception as e:
            print(f'    {node_id:<10} FAILED: {e}')

    wti_raw = _fetch_one(fred, TARGET_CODE)
    print(f'    {"WTI":<10} {TARGET_CODE:<16} {len(wti_raw)} months  ✓  (target)')

    df = pd.DataFrame(levels)
    common = df.index.intersection(wti_raw.index)
    df    = df.loc[common].ffill(limit=2).dropna()
    wti   = wti_raw.loc[df.index]

    # Log-levels for all series
    log_df  = np.log(df.clip(lower=1e-6))
    # CFNAI stays as-is (signed index, cannot log)
    if 'CFNAI' in df.columns:
        log_df['CFNAI'] = df['CFNAI']
    log_wti = np.log(wti.clip(lower=1e-6))

    print(f'\n  Ready: {len(log_df.columns)} predictors × '
          f'{len(log_df)} months  '
          f'({log_df.index[0].strftime("%Y-%m")} → '
          f'{log_df.index[-1].strftime("%Y-%m")})')
    return log_df, log_wti


# ── MPDOK resolvent ────────────────────────────────────────────────────────────

def mpdok_influences(X_win, y_win, feature_names):
    """
    Build [features | WTI] correlation matrix on window,
    return per-feature resolved influence on WTI.
    """
    data = np.column_stack([X_win, y_win])
    std  = data.std(axis=0, ddof=1)
    std[std < 1e-8] = 1e-8
    data_z = (data - data.mean(axis=0)) / std
    A = (data_z.T @ data_z) / max(len(data_z) - 1, 1)
    n = A.shape[0]

    row_max = np.abs(A).sum(axis=1).max()
    if row_max < 1e-8:
        return {f: 1.0 for f in feature_names}
    A_hat = A / row_max
    R = np.linalg.solve(np.eye(n) - ALPHA * A_hat, np.eye(n))

    target_idx = n - 1
    return {name: abs(R[i, target_idx]) for i, name in enumerate(feature_names)}


# ── Ridge solvers ──────────────────────────────────────────────────────────────

def _fit_ridge(X, y, penalties):
    """Solve ridge with per-feature penalties; intercept unpunished."""
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    pen_full = np.concatenate([[0.0], penalties])
    A = Xb.T @ Xb + np.diag(pen_full)
    return np.linalg.solve(A, Xb.T @ y)


def _predict(X, coefs):
    Xb = np.column_stack([np.ones(len(X)), X])
    return Xb @ coefs


# ── Rolling prediction loop ───────────────────────────────────────────────────

def run_models(log_df, log_wti):
    features = list(log_df.columns)
    n_feat   = len(features)
    dates    = log_df.index
    n        = len(dates)

    # Storage
    pred_static  = pd.Series(np.nan, index=dates)
    pred_rolling = pd.Series(np.nan, index=dates)
    pred_mpdok   = pd.Series(np.nan, index=dates)
    influence_log = {}   # date → {feature: score}

    # ── Static model: fit once on pre-2014 data ───────────────────────────────
    train_mask = dates <= pd.Timestamp(TRAIN_END)
    X_static = log_df.loc[train_mask].values
    y_static = log_wti.loc[train_mask].values
    uniform_pen = np.ones(n_feat)
    static_coefs = _fit_ridge(X_static, y_static, uniform_pen)

    # Apply static coefficients to all months
    pred_static[:] = _predict(log_df.values, static_coefs)

    # ── Rolling models ────────────────────────────────────────────────────────
    for t in range(LOOKBACK, n):
        win = slice(t - LOOKBACK, t)
        X_win = log_df.values[win]
        y_win = log_wti.values[win]

        # Rolling Ridge (uniform penalty)
        r_coefs = _fit_ridge(X_win, y_win, uniform_pen)
        pred_rolling.iloc[t] = _predict(log_df.values[[t]], r_coefs)[0]

        # MPDOK rolling — penalties centred at the Stage 1 level so the
        # comparison is fair: high-influence features get less shrinkage,
        # low-influence features get more, mean penalty == uniform penalty.
        inf = mpdok_influences(X_win, y_win, features)
        inf_vec = np.array([inf[f] for f in features])
        inf_vec = np.clip(inf_vec, 1e-6, None)
        mean_inf = inf_vec.mean()
        mpdok_pen = mean_inf / inf_vec   # high influence → penalty < mean_inf
        m_coefs = _fit_ridge(X_win, y_win, mpdok_pen)
        pred_mpdok.iloc[t] = _predict(log_df.values[[t]], m_coefs)[0]

        influence_log[dates[t]] = inf

    return pred_static, pred_rolling, pred_mpdok, influence_log


# ── Metrics ───────────────────────────────────────────────────────────────────

def rmse(actual, predicted, mask=None):
    if mask is not None:
        actual, predicted = actual[mask], predicted[mask]
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    if valid.sum() == 0:
        return np.nan
    diff = actual[valid] - predicted[valid]
    return float(np.sqrt((diff ** 2).mean()))


def r2(actual, predicted, mask=None):
    if mask is not None:
        actual, predicted = actual[mask], predicted[mask]
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    if valid.sum() < 2:
        return np.nan
    a, p = actual[valid], predicted[valid]
    ss_res = ((a - p) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def dir_acc(actual, predicted, mask=None):
    """Directional accuracy: fraction where sign(actual chg) == sign(pred chg)."""
    if mask is not None:
        actual, predicted = actual[mask], predicted[mask]
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    if valid.sum() < 2:
        return np.nan
    a, p = actual[valid], predicted[valid]
    da = np.diff(a)
    dp = np.diff(p)
    return float((np.sign(da) == np.sign(dp)).mean())


# ── Regime periods ────────────────────────────────────────────────────────────

REGIMES = [
    ('2003-2007', '2003-01', '2007-12', 'Pre-GFC stable'),
    ('2008-2009', '2008-01', '2009-12', 'GFC crash & recovery'),
    ('2010-2013', '2010-01', '2013-12', 'Recovery (static train end)'),
    ('2014-2016', '2014-01', '2016-12', 'Shale price war ← key test'),
    ('2017-2019', '2017-01', '2019-12', 'Recovery'),
    ('2020',      '2020-01', '2020-12', 'COVID collapse'),
    ('2021-2022', '2021-01', '2022-12', 'Surge + Ukraine cascade'),
    ('2023-2024', '2023-01', '2024-12', 'Normalisation'),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print('── MPDOK Energy Price Time Series Experiment ────────────────────────')
    print(f'   Target: WTI crude  |  Predictors: nat gas, gasoline, heating,')
    print(f'   Brent, dollar, industrial output, CFNAI')
    print(f'   Static model trained: {FETCH_START} → {TRAIN_END}')
    print(f'   Rolling window: {LOOKBACK} months  |  Resolvent α={ALPHA}')
    print()

    log_df, log_wti = fetch_energy()
    print()

    print('  Running rolling predictions…')
    pred_static, pred_rolling, pred_mpdok, influence_log = run_models(log_df, log_wti)
    print('  Done.\n')

    actual  = log_wti.values
    dates   = log_wti.index

    # ── Overall results (rolling period only — after warm-up) ─────────────────
    roll_mask = ~np.isnan(pred_rolling.values)

    print('Overall results (rolling period):')
    print(f'  {"Model":<28}  {"RMSE (log)":>10}  {"R²":>8}  {"Dir Acc":>8}')
    print(f'  {"─"*28}  {"─"*10}  {"─"*8}  {"─"*8}')
    for label, pred in [('Static Ridge (frozen 2000-13)', pred_static.values),
                         ('Rolling Ridge (Stage 1)',       pred_rolling.values),
                         ('MPDOK Rolling (Stage 2)',       pred_mpdok.values)]:
        print(f'  {label:<28}  {rmse(actual, pred, roll_mask):10.4f}'
              f'  {r2(actual, pred, roll_mask):8.4f}'
              f'  {dir_acc(actual, pred, roll_mask):8.3f}')
    print()

    # ── Regime breakdown ──────────────────────────────────────────────────────
    print('RMSE by regime:')
    print(f'  {"Period":<12}  {"Static":>8}  {"Rolling":>8}  {"MPDOK":>8}'
          f'  {"MPDOK vs Roll":>13}  {"Notes"}')
    print(f'  {"─"*12}  {"─"*8}  {"─"*8}  {"─"*8}  {"─"*13}  {"─"*26}')
    for period, start, end, note in REGIMES:
        mask = ((dates >= pd.Timestamp(start)) &
                (dates <= pd.Timestamp(end)))
        mask_arr = mask & roll_mask
        if mask_arr.sum() < 3:
            continue
        rs  = rmse(actual, pred_static.values,  mask_arr)
        rr  = rmse(actual, pred_rolling.values, mask_arr)
        rm  = rmse(actual, pred_mpdok.values,   mask_arr)
        delta = rm - rr
        arrow = '▲ worse' if delta > 0.002 else ('▼ better' if delta < -0.002 else '≈ same')
        print(f'  {period:<12}  {rs:8.4f}  {rr:8.4f}  {rm:8.4f}'
              f'  {delta:>+8.4f} {arrow:<6}  {note}')
    print()

    # ── MPDOK influence evolution — key years ─────────────────────────────────
    features = list(log_df.columns)
    snap_years = [2006, 2009, 2015, 2020, 2022, 2024]
    print('MPDOK resolved influence on WTI by year (top 3):')
    print(f'  {"Year":<6}  {"#1 influence":>16}  {"#2 influence":>16}  {"#3 influence":>16}')
    print(f'  {"─"*6}  {"─"*16}  {"─"*16}  {"─"*16}')
    for yr in snap_years:
        yr_dates = [d for d in influence_log if d.year == yr]
        if not yr_dates:
            continue
        # Average influence across all months in that year
        avg = {f: np.mean([influence_log[d][f] for d in yr_dates]) for f in features}
        ranked = sorted(avg.items(), key=lambda x: -x[1])[:3]
        cols = [f'{n} ({s:.3f})' for n, s in ranked]
        print(f'  {yr:<6}  {cols[0]:>16}  {cols[1]:>16}  {cols[2]:>16}')
    print()

    # ── Worst static periods (where freezing hurts most) ─────────────────────
    print('Static model breakdown — months where |static error| >> |rolling error|:')
    s_err  = np.abs(actual - pred_static.values)
    r_err  = np.abs(actual - pred_rolling.values)
    excess = s_err - r_err
    worst_idx = np.where(roll_mask)[0]
    worst_idx = sorted(worst_idx, key=lambda i: -excess[i])[:10]
    print(f'  {"Date":<10}  {"Actual WTI":>10}  {"Static err":>10}  '
          f'{"Rolling err":>11}  {"Excess err":>10}')
    print(f'  {"─"*10}  {"─"*10}  {"─"*10}  {"─"*11}  {"─"*10}')
    for i in worst_idx:
        d   = dates[i]
        act = np.exp(actual[i])
        se  = np.abs(actual[i] - pred_static.values[i])
        re  = np.abs(actual[i] - pred_rolling.values[i])
        print(f'  {d.strftime("%Y-%m"):<10}  ${act:>9.2f}  {se:>10.4f}  '
              f'{re:>11.4f}  {excess[i]:>+10.4f}')
    print()

    print('Key insight:')
    overall_static  = rmse(actual, pred_static.values,  roll_mask)
    overall_rolling = rmse(actual, pred_rolling.values, roll_mask)
    overall_mpdok   = rmse(actual, pred_mpdok.values,   roll_mask)
    print(f'  Static vs Rolling: {(overall_static - overall_rolling):+.4f} log-RMSE'
          f' (static is {"worse" if overall_static > overall_rolling else "better"})')
    print(f'  MPDOK vs Rolling:  {(overall_mpdok - overall_rolling):+.4f} log-RMSE'
          f' (MPDOK is {"better" if overall_mpdok < overall_rolling else "worse"})')
    print(f'  Network restructuring visible in influence table above.')
    print()


if __name__ == '__main__':
    main()
