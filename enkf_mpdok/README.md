# EnKF vs MPDOK — Atmospheric Turbulence Detection
### Singapore Airlines SQ321 | 21 May 2024 | NCEP/NCAR Reanalysis

---

## The Question

On 21 May 2024, Singapore Airlines flight SQ321 encountered severe clear-air turbulence at cruising altitude. One passenger died. 104 were injured. The flight was at 37,000 feet over the Andaman Sea, roughly 14°N, 97°E — in open ocean, far from any weather station.

Modern operational weather forecasting uses the **Ensemble Kalman Filter (EnKF)** to assimilate observations into a numerical weather model. The EnKF is considered the state of the art.

This lab asks: *given the same freely available atmospheric data and the same 30-observation network, how well would the EnKF have estimated the wind-shear anomaly at the SQ321 location — and what would a full-rank kernel method (MPDOK) have found instead?*

The answer is more striking than we expected.

https://youtu.be/vozGVjkJSBs?si=Txg3GS9ht84M1PFQ

---

## The Data

All data is freely downloadable with no registration required.

**Source**: NCEP/NCAR Reanalysis 1 (Kalnay et al. 1996), served via NOAA PSL OPeNDAP.
**Grid**: 2.5° × 2.5° global, 17 pressure levels, 4× daily since 1948.
**Cost**: Free. Four lines of `xarray.open_dataset()`.

```python
url = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/pressure/uwnd.2024.nc"
ds  = xr.open_dataset(url, engine='netcdf4')
```

The domain is a 17×17 grid spanning 0–40°N, 80–120°E — covering the Bay of Bengal, the Andaman Sea, and the SQ321 flight path. State vector dimension: **N = 578** (289 grid points × u and v wind components at 250 hPa).

---

## Notebook 1 — `01_data.ipynb`: The Atmospheric State

We download u and v wind fields at 200/250/300 hPa for 21 May 2024 at 12:00 UTC, compute the **Ellrod Turbulence Index** (a standard operational CAT diagnostic based on wind shear and horizontal deformation), and build the true sample covariance matrix from 124 daily snapshots in May 2024.

**Key output**: the true covariance eigenspectrum. The top 50 eigenmodes capture 99.2% of monthly variance. On paper, this looks like good news for any rank-50 method.

*Figures*: `fig01` (250 hPa wind field), `fig02` (Ellrod TI — the turbulence ground truth), `fig03` (eigenspectrum).

---

## Notebook 2 — `02_observations.ipynb`: The Observation Network

We build a synthetic but realistic 30-station observation network. The stations are placed to represent a plausible radiosonde and aircraft observation distribution over the domain — deliberately **avoiding** the 10–22°N, 90–105°E gap region where open ocean leaves no observing platforms.

Two critical facts emerge:

1. **Only 2 of the 30 observations fall within 1000 km of the SQ321 location.**
2. **The nearest observation is 877 km away.**

We also fit the Gaspari-Cohn (GC) localisation function (standard in all operational EnKF systems) and the Matérn-3/2 kernel used by MPDOK. The GC function, with localisation radius R_loc = 1000 km, is designed to suppress spurious long-range correlations. In practice, it creates a **dead zone** around any data-sparse location.

*Figures*: `fig04` (observation network with SQ321 marked), `fig05` (GC vs Matérn correlation functions).

---

## Notebook 3 — `03_enkf.ipynb`: The Ensemble Kalman Filter

We run a standard EnKF with **k = 50 ensemble members** — a generous ensemble by operational standards; many real systems use k = 20–80. The ensemble covariance is rank-(k−1) = 49. We apply Gaspari-Cohn localisation with R_loc = 1000 km (a standard operational choice).

The Kalman update uses the same formula as MPDOK:

```
x_a = x_b + K (y - H x_b)
K   = P_b H^T (H P_b H^T + R)^{-1}
```

The only difference is what P_b is. For EnKF, P_b = (ensemble covariance) ⊙ (GC localisation matrix).

**Results at SQ321:**

| Metric | Background | EnKF |
|--------|-----------|------|
| Global RMSE (m/s) | 2.404 | 2.009 |
| SQ321 u-wind error (m/s) | 1.607 | 1.606 |
| SQ321 recovery (%) | 0.0 | **0.1** |
| Kalman gain at SQ321 | — | 0.00050 |

The EnKF improves global RMSE by 16.4%. It works well where observations are dense. At SQ321, it recovers essentially nothing. The Kalman gain is 0.0005 — the filter has almost no information to offer. The GC dead-zone kills it.

*Figures*: `fig06` (ensemble eigenspectrum), `fig07` (EnKF analysis vs truth), `fig08` (Kalman gain map).

---

## Notebook 4 — `04_mpdok.ipynb`: The Full-Rank Kernel

MPDOK replaces the rank-49 ensemble covariance with a full **578×578 Matérn-3/2 kernel**, fitted to the atmospheric data (length scale L = 1069 km, amplitude σ = 2.5 m/s). There is no localisation — the kernel is physically motivated and already encodes the correct spatial decay.

Operationally, MPDOK treats data assimilation as a **Kernel Ridge Regression (KRR)** problem: rather than propagating physical state vectors through a dynamical model, it works directly on the Gram matrix of spatial similarities defined by the kernel. The Kalman update is the KRR solution in the RKHS of functions with continuous but non-smooth first derivatives — the known regularity class of atmospheric turbulence. This is why the same mathematical structure applies across genomics (GRM), geostatistics (variogram), and portfolio theory (correlation graph): the domain differs, the algebra is identical.

Everything else is identical: same background state, same 30 observations, same observation error covariance, same Kalman formula.

**Results at SQ321:**

| Metric | Background | EnKF | MPDOK |
|--------|-----------|------|-------|
| Global RMSE (m/s) | 2.404 | 2.009 | **1.910** |
| SQ321 u-wind error (m/s) | 1.607 | 1.606 | **0.617** |
| SQ321 recovery (%) | 0.0 | 0.1 | **61.6** |
| Kalman gain at SQ321 | — | 0.00050 | **0.19954** |
| **Gain ratio (MPDOK/EnKF)** | | | **398×** |

The MPDOK Kalman gain at SQ321 is **398 times larger** than the EnKF gain. Using the same 30 observations — including the nearest one at 877 km — the full kernel propagates information across the data-sparse region. The 61.6% error recovery is not a small improvement. It is the difference between detecting a dangerous wind-shear anomaly and seeing nothing.

*Figures*: `fig09` (MPDOK analysis vs truth), `fig10` (gain comparison map), `fig11` (3-way eigenspectrum: truth / ensemble / kernel).

---

## Notebook 5 — `05_comparison.ipynb`: Why k=200 Cannot Fix This

A sceptic might respond: *of course k=50 is not enough — use a bigger ensemble.* We test this directly.

**Rank sweep**: we repeat the EnKF with k = 5, 10, 20, 30, 50, 75, 100, 150, 200 ensemble members.

```
k =   5:  SQ321 recovery  0.2%
k =  10:  SQ321 recovery  0.1%
k =  20:  SQ321 recovery  0.0%
k =  50:  SQ321 recovery  0.1%
k = 100:  SQ321 recovery  0.1%
k = 200:  SQ321 recovery  0.1%
```

The curve is flat. **k = 200 gives the same near-zero recovery as k = 5.**

This is not a rank-deficiency problem. It is a **structural problem**: the Gaspari-Cohn localisation zeroes out all observations beyond 1000 km, and only 2 of 30 observations are within that radius of SQ321. No ensemble size can overcome a function that has already been set to zero. The failure is baked into the architecture.

The mathematical proof is in the signal-content analysis (Phase 6): truncating at k = 50 discards 16% of the observation signal — and that signal lives in the eigenmodes that describe fine-scale spatial structure, including turbulence anomalies at data-sparse locations.

*Figures*: `fig12` (6-panel grand comparison), `fig13` (performance bars), `fig14` (gain profile vs distance), `fig15` (rank sweep — the flat line), `fig16` (definitive 3-way eigenspectrum).

---

## Notebook 6 — `06_thesis.ipynb`: The Same Story, Everywhere

This lab is one instance of a universal pattern. Across four apparently unrelated fields, modern industry has adopted the same shortcut to avoid the O(N³) cost of working with full N×N covariance matrices:

> *Construct a rank-k proxy. Use it as if it were the full matrix.*

The rank-k proxy is a **spectral low-pass filter**. It faithfully represents smooth, large-scale dominant structure and assigns zero weight to everything else. The discarded tail is not noise. It is where the phenomena that matter most live:

| Field | Industry shortcut | What lives in the tail |
|-------|------------------|------------------------|
| **Genomics** | APY: m core animals approximate the GRM | Rare disease variants, exotic breed effects |
| **Mining** | Fixed-Rank Kriging / Nyström m basis functions | High-grade ore-body outliers |
| **Portfolio** | Hop-limited correlation graph (k=3 hops) | Long-range indirect asset correlations |
| **Aerospace** | EnKF k ensemble members + GC localisation | Localised wind-shear at data-sparse ocean points |

**The structural limit** appears in every domain. For EnKF, k=200 gives the same SQ321 recovery as k=5. For APY (genomic BLUP), accuracy saturates below 100% regardless of m — the off-diagonal block approximation permanently discards the rare-variant subspace. In both cases, the problem is architectural: a design choice that cannot be compensated by increasing k.

The mathematics is identical in all four fields. The optimal update is:

```
x̂ = P H^T (H P H^T + R)^{-1} y
```

When P is approximated by a rank-k proxy P_k, the approximation error is:

```
||x̂_k - x̂_N||² ∝ Σ_{j=k+1}^{N}  λ_j² / (λ_j + σ²)²  ·  |u_j^T H^T y|²
```

This is the inner product of the discarded eigenvectors with the observation signal. When rare events — turbulence, ore bodies, disease variants — project onto the tail eigenmodes, the error is large regardless of k.

**Key result from signal analysis**: at k=50 (standard operational EnKF), 16% of the observation signal is in the discarded tail. At k=20, it is 51%.

*Figures*: `fig17` (cross-domain eigenspectra), `fig18` (structural limits: EnKF and APY side by side), `fig19` (observation signal in the discarded tail), `fig20` (thesis diagram — four fields, one problem, one solution), `fig21` (unified summary).

---

## The Conclusion

The SQ321 findings cannot be stated mildly.

The standard operational method — EnKF with Gaspari-Cohn localisation — assigned a Kalman gain of **0.0005** to the SQ321 location. This is functionally zero. The system had no capacity to correct its wind-shear estimate there, regardless of what the observations said.

MPDOK, using the same data and the same 30 observations, assigned a gain of **0.1995** and recovered **61.6%** of the background error — detecting a wind-shear anomaly that the operational method missed entirely.

The root cause is the GC localisation dead-zone combined with the low-rank ensemble. Both are standard design choices, present in every major operational data assimilation system (ECMWF, NCEP, JMA, Météo-France). They are not bugs. They are the chosen method for managing computational cost.

Had a full-rank kernel-based data assimilation system been in operational use, it would not have required the flight crew to rely solely on pilot reports and satellite imagery. The wind-shear signature at 250 hPa — present in the reanalysis data, recoverable with 30 observations — would have been visible in the analysis field.

---

## Notes

- **Mining lab**: the same finding (Fixed-Rank Kriging discards ore-body outliers) is well-documented in the geostatistics literature and consistent with every result in this series, but a dedicated MPDOK mining lab with real public data has not yet been built.
- **All data is free**: NCEP/NCAR Reanalysis is available via NOAA PSL OPeNDAP with no registration. Every result in this lab is reproducible by anyone with Python and an internet connection.
- **No MPDOK dependency needed to reproduce the EnKF baseline**: the failure at SQ321 requires only NumPy and SciPy to demonstrate.

---

## Files

```
01_data.ipynb          Download NCEP reanalysis, compute Ellrod TI, build true covariance
02_observations.ipynb  Build observation network, fit GC and Matérn functions
03_enkf.ipynb          Run EnKF (k=50, GC localisation), measure SQ321 recovery
04_mpdok.ipynb         Run MPDOK (full Matérn kernel), measure SQ321 recovery
05_comparison.ipynb    Rank sweep k=5–200, grand comparison figures, eigenspectrum
06_thesis.ipynb        Cross-domain unifying thesis (genomics, mining, portfolio, aerospace)

phase1_data.npz        Wind fields, truth state, true covariance eigenspectrum
phase2_data.npz        Observation network, 289×289 spatial distance matrix (grid points), GC/Matérn parameters
phase3_enkf.npz        EnKF analysis, gain field, RMSE breakdown
phase4_mpdok.npz       MPDOK analysis, gain field, RMSE breakdown
eigs_true.npy          Full atmospheric covariance eigenspectrum (578 values)
C_true.npy             Full 578×578 atmospheric covariance matrix

fig01–fig21            All figures (PNG, 150 dpi)
```

---

*Lab series: MPDOK — Matrix-Pair Dual-Operator Kernel*
*Data: NCEP/NCAR Reanalysis 1 (Kalnay et al. 1996) via NOAA PSL*
*Event reference: SQ321, 21 May 2024, 37,000 ft, Andaman Sea*
