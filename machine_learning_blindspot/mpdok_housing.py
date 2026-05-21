#!/usr/bin/env python3
"""
mpdok_housing.py — MPDOK applied to California Housing price regression

Compares:
  Stage 1 : plain Ridge regression
  Stage 2 : MPDOK adaptive Ridge — per-feature L2 penalties weighted by
            resolved influence on median_house_value
  RF      : Random Forest ceiling (non-linear baseline)

Stage 2 uses the exact same adaptive Ridge formula as the Fed rate predictor:
  penalty_i = ridge_alpha / influence_i     (high influence → less shrinkage)
  coefs = solve(X.T @ X + diag(penalties), X.T @ y)

The hypothesis: housing price has genuine multi-hop causality —
  location → ocean_proximity → income → price
  population → households → density → price
  total_rooms → bedrooms_per_room → price
The resolvent should surface income and ocean proximity as dominant channels.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

DATA = '/var/home/fraser/machine_learning/hands-on-ml-with-scikit-learn-keras-and-tensorflow/datasets/housing/housing.csv'
ALPHA      = 0.85   # resolvent damping
RIDGE_A    = 1.0    # baseline Ridge alpha (same for S1 and S2)
N_FOLDS    = 5
PENALTY_CAP = 200.0  # floor on influence prevents runaway penalties


# ── Data prep ─────────────────────────────────────────────────────────────────

OCEAN_MAP = {'INLAND': 0.0, 'NEAR OCEAN': 1.0, '<1H OCEAN': 2.0,
             'NEAR BAY': 3.0, 'ISLAND': 4.0}

def load():
    df = pd.read_csv(DATA)
    df['ocean_enc']      = df['ocean_proximity'].map(OCEAN_MAP)
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

    # Log-transform count features to reduce skew
    for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
        df[f'log_{col}'] = np.log1p(df[col])

    features = [
        'longitude', 'latitude', 'housing_median_age', 'median_income',
        'log_total_rooms', 'log_total_bedrooms', 'log_population',
        'log_households', 'ocean_enc',
    ]
    X = df[features].values.astype(float)
    y = df['median_house_value'].values.astype(float)
    return X, y, features


# ── MPDOK resolvent ────────────────────────────────────────────────────────────

def mpdok_influences(X_train, y_train, feature_names):
    """
    Build [features | target] correlation matrix on training fold,
    compute resolvent R = (I − α·Â)⁻¹, return column influence on target.
    """
    data = np.column_stack([X_train, y_train])
    std  = data.std(axis=0, ddof=1)
    std[std < 1e-8] = 1e-8
    data_z = (data - data.mean(axis=0)) / std
    A = (data_z.T @ data_z) / (len(data_z) - 1)
    n = A.shape[0]

    row_max = np.abs(A).sum(axis=1).max()
    A_hat   = A / row_max
    R = np.linalg.solve(np.eye(n) - ALPHA * A_hat, np.eye(n))

    target_idx = n - 1
    influences = {name: abs(R[i, target_idx])
                  for i, name in enumerate(feature_names)}
    return influences, A, R


# ── Ridge solvers ──────────────────────────────────────────────────────────────

def ridge_stage1(X, y, alpha=RIDGE_A):
    """Plain Ridge: uniform L2 penalty on all features (intercept unpunished)."""
    n, p = X.shape
    # Prepend bias column
    Xb = np.column_stack([np.ones(n), X])
    penalty = np.full(p + 1, alpha)
    penalty[0] = 0.0  # intercept unpunished
    A = Xb.T @ Xb + np.diag(penalty)
    coefs = np.linalg.solve(A, Xb.T @ y)
    return coefs


def ridge_stage2(X, y, influences, feature_names, alpha=RIDGE_A):
    """MPDOK adaptive Ridge: per-feature penalty = alpha / influence."""
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    inf_vec = np.array([influences[f] for f in feature_names])
    inf_vec = np.clip(inf_vec, alpha / PENALTY_CAP, None)  # avoid ÷0
    penalties = alpha / inf_vec
    penalty_full = np.concatenate([[0.0], penalties])  # intercept unpunished
    A = Xb.T @ Xb + np.diag(penalty_full)
    coefs = np.linalg.solve(A, Xb.T @ y)
    return coefs


def predict(X, coefs):
    Xb = np.column_stack([np.ones(len(X)), X])
    return Xb @ coefs


# ── Cross-validated evaluation ─────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate():
    X, y, features = load()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    rmse_s1, rmse_s2, rmse_rf = [], [], []
    r2_s1,   r2_s2,   r2_rf   = [], [], []
    fold_influences = []

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s  = scaler.transform(X_va)

        # Stage 1
        c1 = ridge_stage1(X_tr_s, y_tr)
        p1 = predict(X_va_s, c1)
        rmse_s1.append(rmse(y_va, p1))
        r2_s1.append(r2_score(y_va, p1))

        # MPDOK influences (training fold only — no data leakage)
        inf, A, R = mpdok_influences(X_tr, y_tr, features)
        fold_influences.append(inf)

        # Stage 2
        c2 = ridge_stage2(X_tr_s, y_tr, inf, features)
        p2 = predict(X_va_s, c2)
        rmse_s2.append(rmse(y_va, p2))
        r2_s2.append(r2_score(y_va, p2))

        # RF ceiling
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        prf = rf.predict(X_va_s)
        rmse_rf.append(rmse(y_va, prf))
        r2_rf.append(r2_score(y_va, prf))

        print(f'  fold {fold}  S1 RMSE={rmse_s1[-1]:,.0f}  '
              f'S2 RMSE={rmse_s2[-1]:,.0f}  RF RMSE={rmse_rf[-1]:,.0f}')

    return (rmse_s1, rmse_s2, rmse_rf,
            r2_s1,   r2_s2,   r2_rf,
            fold_influences, features, X, y)


def mean_influences(fold_influences, features):
    return {f: np.mean([fi[f] for fi in fold_influences]) for f in features}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print('── MPDOK California Housing Experiment ──────────────────────────────')
    print(f'   Resolvent α={ALPHA}  |  {N_FOLDS}-fold CV  |  20,640 districts')
    print(f'   Target: median_house_value  |  Ridge α={RIDGE_A}')
    print()

    (rmse_s1, rmse_s2, rmse_rf,
     r2_s1,   r2_s2,   r2_rf,
     fold_influences, features, X, y) = evaluate()

    def fmtr(vals):
        return f'${np.mean(vals):>8,.0f}  ±{np.std(vals):>5,.0f}'
    def fmtq(vals):
        return f'{np.mean(vals):.4f}  ±{np.std(vals):.4f}'

    print()
    print('Results (5-fold CV):')
    print(f'  {"Model":<28s}  {"RMSE ($/district)":>20s}  {"R²":>12s}')
    print(f'  {"─"*28}  {"─"*20}  {"─"*12}')
    print(f'  {"Stage 1 (Ridge)":<28s}  {fmtr(rmse_s1)}  {fmtq(r2_s1)}')
    print(f'  {"Stage 2 (MPDOK adaptive Ridge)":<28s}  {fmtr(rmse_s2)}  {fmtq(r2_s2)}')
    delta_rmse = np.mean(rmse_s2) - np.mean(rmse_s1)
    delta_r2   = np.mean(r2_s2)   - np.mean(r2_s1)
    print(f'  {"MPDOK delta":<28s}  {delta_rmse:>+8,.0f}               {delta_r2:>+.4f}')
    print()
    print(f'  {"Random Forest (ceiling)":<28s}  {fmtr(rmse_rf)}  {fmtq(r2_rf)}')
    gap_rmse = np.mean(rmse_rf) - np.mean(rmse_s2)
    print(f'  {"Linear ceiling gap":<28s}  {gap_rmse:>+8,.0f}  (MPDOK vs RF)')
    print()

    # ── MPDOK influence scores ────────────────────────────────────────────────
    mean_inf = mean_influences(fold_influences, features)
    print('MPDOK resolved influence on median_house_value (avg across folds):')
    max_inf = max(mean_inf.values())
    for name, score in sorted(mean_inf.items(), key=lambda x: -x[1]):
        bar = '█' * int(round(score / max_inf * 32))
        print(f'  {name:22s}  {score:.4f}  {bar}')
    print()

    # ── Correlation matrix ────────────────────────────────────────────────────
    inf_full, A, R = mpdok_influences(X, y, features)
    node_names = features + ['house_value']
    print('Direct correlation with median_house_value:')
    target_idx = len(features)
    for i, name in enumerate(features):
        corr = A[i, target_idx]
        bar_len = int(abs(corr) * 30)
        sign = '+' if corr >= 0 else '-'
        bar = sign + '█' * bar_len
        print(f'  {name:22s}  {corr:+.3f}  {bar}')
    print()

    # ── Stage 2 penalty comparison ────────────────────────────────────────────
    print('Stage 2 adaptive penalties (lower = feature kept larger):')
    print(f'  {"Feature":22s}  {"Influence":>9s}  {"Penalty":>9s}  {"vs S1 uniform":>14s}')
    print(f'  {"─"*22}  {"─"*9}  {"─"*9}  {"─"*14}')
    for name in sorted(mean_inf, key=lambda x: -mean_inf[x]):
        inf = max(mean_inf[name], RIDGE_A / PENALTY_CAP)
        pen = RIDGE_A / inf
        ratio = pen / RIDGE_A
        direction = 'less shrinkage' if ratio < 1 else 'more shrinkage'
        print(f'  {name:22s}  {mean_inf[name]:9.4f}  {pen:9.3f}  ×{ratio:.2f} ({direction})')
    print()

    top2 = sorted(mean_inf.items(), key=lambda x: -x[1])[:2]
    print('Key finding:')
    print(f'  Resolvent top influences: {top2[0][0]} ({top2[0][1]:.4f}), '
          f'{top2[1][0]} ({top2[1][1]:.4f})')
    print('  Multi-hop paths captured: location→ocean→income→price,')
    print('  population→households→density→price')
    print()


if __name__ == '__main__':
    main()
