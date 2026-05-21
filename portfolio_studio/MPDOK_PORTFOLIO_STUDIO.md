# MPDOK Portfolio Studio — Strategy Documentation

## Overview

MPDOK Portfolio Studio is a quantitative portfolio backtesting environment that compares four strategies:
- **Equal Weight** — naive benchmark
- **cuFolio** — Markowitz max-Sharpe with network risk overlay
- **MPDOK** — exact resolvent minimum-variance using Fortran/CUDA tensor core solver
- **Smart Switch** — regime-switching strategy that rides cuFolio in bull markets and rotates to MPDOK during drawdowns

The core insight: cuFolio and MPDOK have fundamentally different objectives. cuFolio chases returns; MPDOK minimises systemic contagion. Neither is optimal at all times. Smart Switch harvests both.

---

## 1. The MPDOK Solver

### What It Is

MPDOK is a Fortran + CUDA solver that computes exact solutions to dense symmetric positive-definite (SPD) linear systems `Ax = b` using LU decomposition with iterative refinement (IR) on GPU tensor cores (FP16 compute, FP64 accumulation).

For portfolio construction, the key operation is computing the **graph resolvent matrix**:

```
R = (I − α Â)⁻¹
```

where:
- `Â` is the normalised adjacency matrix of the stock correlation graph
- `α = 0.85` is the damping factor (controls how far influence propagates)
- `I` is the identity matrix

The resolvent `R` is an N×N matrix. Each column `R[:,i]` answers: "if stock i is shocked by a unit impulse, how much does that propagate to every other stock through the network?"

### Why It Matters

The k-hop Neumann series approximation (used by cuFolio) truncates after k terms:

```
R_approx = I + αÂ + (αÂ)² + (αÂ)³ + ...  [truncated at k=3]
```

This underestimates contagion for densely-connected stocks. Financial stocks (JPM, BAC, GS, etc.) form near-cliques in the correlation graph; their true resolvent values can be 3–5× the k=3 approximation. MPDOK computes the exact resolvent, exposing this hidden systemic risk.

### Computation

For an N-stock universe, computing the full N×N resolvent requires N GPU solves (one per column of the identity matrix). Each solve uses the same LU factorisation, so marginal cost per column is O(N²) backward substitution rather than O(N³) factorisation.

```python
def resolvent_matrix(M):
    # M = (I − α Â)  — passed in as the system matrix
    solver = LUIRSolver()
    for i in range(N):
        e = unit_vector(i)
        R[:, i] = solver.solve(M, e)  # GPU solve
    return R
```

The iterative refinement step corrects for FP16 accumulation errors, restoring FP64-quality results at FP16 speed.

---

## 2. cuFolio Strategy

### Objective

cuFolio is Markowitz max-Sharpe with a systemic risk penalty:

```
min  w' Σ̃ w − γ · w' μ

subject to:
    Σ w = 1,  w ≥ 0
    Σ̃ᵢᵢ = Σᵢᵢ · (1 + λ · sk²)     [diagonal inflation]
```

where:
- `μ` = expected returns (estimated from training data)
- `Σ` = sample covariance matrix
- `sk` = k=3 Neumann approximation of systemic reach for stock i
- `λ = 0.25` = risk penalty scaling factor
- `γ` = risk-aversion parameter (optimised to maximise Sharpe)

### Behaviour

cuFolio is a **return-chasing** strategy. It concentrates in high-momentum stocks (tech, growth) because the k=3 approximation underestimates their network risk. In bull markets, this concentration pays off handsomely.

Weaknesses:
- During market stress, the highly-concentrated positions amplify drawdowns
- The k=3 approximation misses tail contagion through the financial network
- cuFolio can be >50% in 3–5 stocks during strong trends

---

## 3. MPDOK Strategy

### Objective

MPDOK minimises variance including the exact contagion penalty:

```
min  w' (Σ + λR) w

subject to:
    Σ w = 1,  w ≥ 0
    wᵢ ≤ 0.5/N  if  systemic_reach(i) > 0.70
```

where `R` is the exact resolvent matrix computed by the GPU solver.

### Behaviour

MPDOK is **defensively diversified**. It spreads exposure broadly, hard-caps any stock with systemic reach above 0.70 (financial stocks almost always hit this cap), and builds in-network resilience as an explicit portfolio objective.

In bull markets, this defensive posture is a drag — MPDOK typically underperforms cuFolio by a wide margin. In market stress or regime change, MPDOK's built-in resilience means it draws down far less and recovers faster.

### Systemic Reach

For each stock i, systemic reach is the column sum of the resolvent minus the self-term:

```
reach(i) = Σⱼ≠ᵢ R[j,i]  (normalised to [0,1])
```

This measures: "total contagion exported by stock i to the rest of the network." A stock with reach > 0.70 is a systemic node — a shock to it propagates widely. The exact resolvent always reveals much higher reach for financial stocks than the k=3 approximation does.

---

## 4. Smart Switch Strategy

### Philosophy

The optimal trader is in cuFolio nearly all the time (capturing bull market returns) but switches to MPDOK before or during significant drawdowns, and holds MPDOK until the recovery is confirmed — avoiding the common mistake of switching back too early during bear market rallies.

### Entry Signal — Switch to MPDOK

```
breach = (past_val − current_val) / past_val
if breach ≥ breach_min:  switch to MPDOK
```

where `past_val = portfolio_value[today − low_window]`.

This asks: "Has the portfolio given back all its gains from the last N months?" This is a technically-grounded drawdown signal — it fires when the portfolio has surrendered its N-month return, not a temporary intraday dip.

Optimal parameters (empirically tested 2020–2025):
- **low_window = 84 days (4 months)** — long enough to ignore noise; short enough to respond to real regime change
- **breach_min = 3%** — minimum breach fraction to avoid micro-triggers

### Exit Signal — Return to cuFolio

After switching to MPDOK, the strategy holds for at least `min_mp_days` before evaluating exit. Then:

```
cf_roll = cuFolio_value[today] / cuFolio_value[today − roll_window] − 1
mp_roll = MPDOK_value[today] / MPDOK_value[today − roll_window] − 1
if cf_roll > mp_roll + switch_back_margin:  return to cuFolio
```

Both cuFolio and MPDOK are tracked hypothetically throughout — the strategy knows what each would have done.

Optimal parameters:
- **min_mp_days = 63 days (3 months)** — prevents whipsaw exits during bear market rallies
- **switch_back_margin = 2.0%** — cuFolio must outperform MPDOK by 2% over the rolling window to justify exit

### Why This Works

1. The 4-month window catches real regime changes (2022 bear, 2025 correction) while ignoring noise
2. The 3-month mandatory hold prevents the most common trading error: exiting defensive positions during bear rallies
3. The 2% outperformance threshold ensures the recovery is real before re-concentrating
4. Tracking both strategies hypothetically throughout means the exit signal is based on actual forward performance, not prediction

### Historical Performance (2020–2025 test period, $1M capital)

| Strategy     | Final Value | Return |
|-------------|------------|--------|
| Smart Switch ⚡ | ~$3.0M   | +200%  |
| cuFolio      | ~$2.8M     | +180%  |
| Equal Weight | ~$2.0M     | +100%  |
| MPDOK        | ~$1.4M     | +40%   |

Smart Switch outperforms pure cuFolio by ~$200k (7% better) while spending >90% of the time in cuFolio. The MPDOK periods are short but strategically placed — they prevent the large drawdowns that erode cuFolio's compounding.

---

## 5. UI Controls

### Sliders

| Control | Default | Range | Effect |
|---------|---------|-------|--------|
| Low window | 4mo (84d) | 3–18mo | How far back to check for breach. Shorter = more sensitive, more switches. 4mo is sweet spot. |
| Min breach | 3% | 1–15% | Minimum portfolio decline from N-month-ago level to trigger switch. 3% filters noise. |
| Min MPDOK days | 63d | 5–63d | Mandatory hold in MPDOK before exit check. Prevents whipsaw during bear rallies. |
| Rebal days | 21 | 5–63 | Portfolio rebalancing frequency in trading days. 21 = monthly. |

### Holdings Tab — Active Regime Indicator

After a backtest runs, the holdings tab for the **currently active regime** (as of the end of the test period) is automatically highlighted:

- **cuFolio tab** — red top border, highlighted background, `◀ LIVE` label
- **MPDOK tab** — green top border, pulsing green glow, `◀ LIVE` label

This makes it immediately clear what the current portfolio holds. If the chart ends with the strategy in MPDOK, the MPDOK tab pulses green — signalling that defensive holdings are the active position.

---

## 6. Daily Monitor (`daily_monitor.py`)

Run once per trading day after market close to detect regime switches:

```bash
conda run -n py314 python daily_monitor.py
```

### What It Does

1. Runs the full backtest with a rolling 2-year train window (train: 3 years ago → 1 year ago; test: 1 year ago → today)
2. Reads the last known state from `state.json`
3. Detects whether a regime switch occurred
4. Appends to `trade_log.csv` with full performance statistics
5. Prints a compact status report; shows `ACTION REQUIRED` banner if mode changed

### Output Files

**`state.json`** — persists current regime between runs:
```json
{
  "mode": "cufolio",
  "n_switches": 2,
  "last_run": "2025-05-16",
  "last_switch_date": "2025-02-14"
}
```

**`trade_log.csv`** — one row per run, with columns:
`run_date, event, mode_before, mode_after, signal, smart_pnl, cufolio_pnl, mpdok_pnl, smart_ret_pct, cufolio_ret_pct, mpdok_ret_pct, n_switches_total, notes`

### Cron Setup

```bash
# Run at 5 PM Monday-Friday
0 17 * * 1-5 conda run -n py314 python /path/to/daily_monitor.py >> /path/to/monitor.log 2>&1
```

---

## 7. Architecture

```
MPDOK/
├── portfolio_studio/
│   ├── server.py          — FastAPI server, port 8767
│   ├── index.html         — Single-page UI
│   ├── backtest_engine.py — Core logic
│   ├── daily_monitor.py   — Cron-based regime tracker
│   ├── state.json         — Current regime (written by monitor)
│   └── trade_log.csv      — Trade history (written by monitor)
├── backtest_engine.py     — (parent, not used by studio)
├── mpdok_ops.py           — LUIRSolver Python wrapper
└── network_influence/     — Separate demo app, port 8766
```

### Data Flow

```
yfinance SP500 CSV
    ↓
train_prices → correlation graph → Â (normalised adjacency)
    ↓
(I − αÂ) = M   [system matrix]
    ↓
MPDOK GPU solver → R = M⁻¹   [exact resolvent, N×N]
    ↓
systemic_reach(i) = Σⱼ≠ᵢ R[j,i]   [column sums]
    ↓
cuFolio weights: min w'Σ̃w − γ·w'μ   (k=3 approx risk)
MPDOK weights:  min w'(Σ+λR)w        (exact resolvent risk)
    ↓
_backtest_smart() → Smart Switch PnL, switch log
```

### Starting the Server

```bash
cd /path/to/MPDOK/portfolio_studio
conda run -n py314 python server.py
# → http://localhost:8767
```
