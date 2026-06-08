# MPDOK Mining Lab — Fixed-Rank Kriging and the Masking Effect
### Carlin Trend, Nevada | USGS NURE-HSSR Stream-Sediment Geochemistry

---

## The Question

The Carlin Trend in north-central Nevada is the second-largest gold province on Earth. Its gold is "invisible" — locked in arsenian pyrite at concentrations too fine for conventional optical methods. Every major mining decision in this region is made on geochemical data: stream sediment, soil, and drillhole assay results, processed through spatial interpolation models that predict grade between sample points.

The industry-standard spatial approximation for large datasets is **Fixed-Rank Kriging (FRK)** — the geostatistics implementation of the Nyström approximation, which represents the covariance structure using $m$ inducing-point basis functions rather than the full $N \times N$ Gram matrix.

This lab asks: *given real USGS stream-sediment gold data from the Carlin Trend region and a standard kriging setup, how well does FRK at operational ranks (m = 5–200) recover high-grade gold anomalies — and what does MPDOK find instead?*

The answer reveals a masking effect that has deep economic consequences.

---

## The Masking Effect

Raw stream-sediment gold data carries a **55.7% nugget effect** — more than half of all Au variation is purely localized, uncorrelated random noise arising from sampling errors, assay variability, and micro-scale mineralization changes occurring over distances shorter than the sample spacing. Because this variance is genuinely random, no spatial interpolation method can recover it. Both FRK and MPDOK operate against the same nugget ceiling.

The consequence is an empirical masking effect: with 55.7% of variance being irrecoverable white noise, the performance gap between FRK (a structurally limited low-rank method) and MPDOK (a full-rank method) appears modest — 13 percentage points at m=20 versus the MPDOK ceiling.

**But the structural gap is always there.** The nugget does not fix FRK's architectural problem. It hides it.

The nugget adds $\sigma^2_{\text{nugget}} \mathbf{I}$ to the kernel, which shifts every eigenvalue upward by the same scalar amount. This lifts the entire eigenspectrum uniformly — it compresses relative differences between methods without changing the kernel geometry that makes FRK blind to short-range structure. When the nugget drops to 6% (as in the controlled synthetic experiment), the same structural gap widens from 13pp to 42pp.

> *The nugget was masking the failure. The failure was always there.*

This mirrors the EnKF failure at SQ321 exactly. In the aerospace case, the Gaspari-Cohn dead zone was the masking mechanism — it zeroed out all observation information beyond 1000km, making the localized turbulence signal irrecoverable regardless of ensemble size. In the mining case, the nugget is the masking mechanism — it compresses the irrecoverable-noise fraction until both methods look equally limited. In both cases, the structural architectural flaw survives intact under the mask.

---

## The Data

**Source**: USGS National Uranium Resource Evaluation — Heavy-Mineral Sand Sample Reanalysis (NURE-HSSR), downloaded from ScienceBase (freely available, no registration).

**Region**: Nevada, filtered to the Carlin Trend geochemical province (39.5–42.0°N, 117.5–114.5°W).

**Records**: 13,828 Nevada samples; 4,106 within the Carlin Trend bounding box; 800-sample spatial subsample used for eigenspectrum analysis.

**Elements**: Au (ppm), As (ppm), Sb (ppm), Ag (ppm), Cu (ppm), Zn (ppm), Tl (ppm).

**Key statistics**:
- Au P95: 0.011 ppm (high-grade threshold)
- Fitted Matérn-3/2 variogram: nugget = 0.4536, C0 = 0.3604, ℓ = 34.4 km
- Nugget fraction: 55.7% of total sill

All coordinates in NAD27 (negligible ~100m shift to WGS84 at this scale).

---

## The Physical Limit

For a domain of area $A$ km², $m$ inducing points placed on an **idealised regular grid** are spaced $\sqrt{A/m}$ km apart. For a completely random placement, the expected nearest-neighbour distance is $\frac{1}{2}\sqrt{A/m}$.

For the 69,000 km² Carlin Trend study domain and m=20:

| Placement | Spacing at m=20 | vs ℓ_short (10 km) |
|-----------|:---------------:|:-------------------:|
| Idealised regular grid | **59 km** | 5.9× |
| Random | **~30 km** | ~3× |

The Matérn-3/2 correlation at 59 km for ℓ=10km is **0.007** — essentially zero. Achieving correlation = 0.5 (the resolution threshold) for ℓ=10km requires m > 400 in this domain. For the fitted Au variogram (ℓ=34km), the threshold is m* ≈ 63.

**FRK at practical ranks (m=20–100) cannot represent short-range mineralisation structure. This is geometry, not statistics.**

---

## Notebook 1 — `01_data.ipynb`: The Carlin Trend Dataset

We load and clean the NURE-HSSR Nevada data, map the Au and As distributions across the Carlin Trend, and compute the eigenspectrum of the fitted Matérn-3/2 kernel at 800 subsampled locations.

The eigenspectrum reveals the structural gap: the top-50 eigenmodes capture 70.7% of kernel variance but only 22.9% of high-grade Au signal (>p95). The HG signal disproportionately occupies the spectral tail that FRK discards. This is not a numerical artifact — it reflects the physical fact that high-grade mineralization is a localized, short-range phenomenon embedded in a much smoother geological background.

*Figures*: `fig01` (Nevada Au map), `fig02` (Carlin Trend Au + As maps), `fig03` (Au distribution + exceedance), `fig04` (3-panel eigenspectrum: decay / cumulative signals / structural gap).

---

## Notebook 2 — `02_variogram.ipynb`: Variogram Fitting and Cross-Validation

We fit the Matérn-3/2 variogram by weighted least squares (Cressie weights: $n_{\text{pairs}} / \hat{\gamma}^2$), establishing the nugget (55.7%), range (34.4 km), and partial sill (0.3604). These parameters drive all subsequent kriging predictions.

A spatial holdout (Carlin Trend core withheld as test zone) shows that even MPDOK achieves only 6.5% improvement at HG sites — because every training sample within 26–55km has background-level Au (~0.003–0.007 ppm) regardless of model. This is the nugget ceiling in action.

The random 80/20 split cross-validation gives cleaner numbers:

| Method | HG improvement over null |
|--------|:------------------------:|
| FRK m=5 | 6.4% |
| FRK m=20 | 10.2% |
| FRK m=50 | 17.5% |
| FRK m=200 | 21.7% |
| **MPDOK** | **23.0%** |

The gap is real but modest — 13pp at m=20. The nugget is doing its masking work.

*Figures*: `fig05` (variogram fit), `fig06` (FRK rank sweep with null baseline), `fig07` (predicted vs true scatter), `fig08` (spatial maps FRK m=50 vs MPDOK).

---

## Notebook 3 — `03_mpdok.ipynb`: The Synthetic Control Experiment

To expose the structural gap that the nugget is hiding, we simulate a **nested Matérn-3/2 field** at the actual NURE-HSSR sample locations:

$$y(s) = y_{\text{long}}(s) + y_{\text{short}}(s) + \varepsilon(s)$$
$$y_{\text{long}} \sim \mathcal{GP}(0,\, 0.30 \cdot M_{3/2}(\ell = 80\,\text{km}))$$
$$y_{\text{short}} \sim \mathcal{GP}(0,\, 0.50 \cdot M_{3/2}(\ell = 10\,\text{km}))$$
$$\varepsilon \sim \mathcal{N}(0,\, 0.05 \cdot I)$$

This represents what block-averaged drillhole grades, multi-element pathfinder scores (As + Sb + Tl combined), or airborne geophysics would look like — spatially structured data without the raw Au sample nugget masking the structure.

| Method | Short-range peak improvement | Gap vs MPDOK |
|--------|:----------------------------:|:------------:|
| Null predictor | 0% | 45 pp |
| FRK m=20 | **3.2%** | **42 pp** |
| FRK m=50 | 27.3% | 18 pp |
| FRK m=200 | 42.7% | 2 pp |
| **MPDOK** | **45.2%** | — |

MPDOK correlation with truth at short-range peaks: **r = 0.92**. FRK m=20: **r = −0.05** (random, equivalent to null). The masking is gone. The structural gap is 42 percentage points.

*Figures*: `fig09` (synthetic field — long + short + observed), `fig10` (rank sweep), `fig11` (predicted vs true scatter), `fig12` (4-panel spatial maps).

---

## Notebook 4 — `04_comparison.ipynb`: Proof, Mechanism, Grand Summary

The assembly notebook. Four panels of evidence:

1. **Spacing analysis** (fig13): For each m, the inducing-point spacing is computed for the actual domain. The Matérn-3/2 correlation of the short-range (ℓ=10km), fitted (ℓ=34km), and long-range (ℓ=80km) components at that spacing is plotted. Annotated: m* for ℓ=34km is ~63; m* for ℓ=10km is >400.

2. **Spectral proof** (fig14): Cumulative kernel variance vs cumulative anomalous signal content for both kernels. The shaded gap is what FRK discards. At m=20: the structural gap in the nested kernel is ~40pp.

3. **Cross-dataset rank sweep** (fig15): Both datasets on the same improvement-percentage axis. The nugget compression on the left (real Au) vs the exposed gap on the right (synthetic).

4. **Grand 6-panel summary** (fig16): Spacing + two eigenspectra + two rank sweeps + summary bar chart. The bar chart shows FRK m=5 through m=200 and MPDOK for both datasets side by side.

---

## The Economic Trap

A mining company relying on FRK at operational ranks faces two symmetric cost categories that together constitute a capital allocation trap:

### A — Unnecessary Drilling Cost

To resolve a 10km mineralisation target with FRK, a company must either expand m beyond 400 (requiring massive infill drilling campaigns to prevent matrix ill-conditioning) or accept the spatial blur and miss the target entirely. Both outcomes destroy capital: the first through unnecessary drill footage at $\$50–\$300$ per metre, the second through missed ore.

### B — Ore vs. Waste Misclassification

FRK's spatial low-pass filter does two things simultaneously:

- **Revenue loss**: a localised high-grade zone that is economically viable to extract is diluted by the smoothing into surrounding low-grade rock. The model reports it as sub-economic waste. The company leaves gold in the ground.

- **Milling waste**: the smooth predicted-grade surface extends moderate values into adjacent areas of actually barren rock. The mining team blasts, transports, and processes worthless material through the milling circuit at full operational cost — typically $\$10–\$30$ per tonne of ore milled — for zero yield.

MPDOK eliminates both failure modes simultaneously. By operating in the full RKHS without rank truncation, it recovers both the smooth regional geological background (equally well as FRK) and the localised high-grade mineralisation zones (which FRK misses). No rank parameter. No capital-efficiency trade-off.

---

## Cross-Domain Pattern

This lab is one instance of a universal pattern. Across four apparently unrelated industries, modern practice has adopted the same shortcut to avoid the O(N³) cost of full covariance inference:

> *Construct a rank-k proxy. Use it as if it were the full matrix.*

The rank-k proxy is a spatial low-pass filter. It faithfully represents smooth, large-scale structure and assigns zero weight to everything in the spectral tail. The discarded tail is not noise. It is where the phenomena that determine outcomes live:

| Industry lab | Low-rank proxy | High-frequency failure mode |
|---|---|---|
| **Aerospace (SQ321)** | EnKF k=50 ensemble + Gaspari-Cohn localisation | Localised wind-shear at data-sparse ocean locations → **turbulence, fatality** |
| **Mining (Carlin Trend)** | FRK m=20 inducing points | Short-range high-grade mineralisation → **ore/waste misclassification, capital loss** |
| **Genomics (APY)** | m core animals approximating the GRM | Rare disease variants, exotic breed effects → **BLUP accuracy ceiling** |
| **Portfolio theory** | Hop-limited correlation graph (k=3 hops) | Long-range indirect asset correlations → **unhedged tail exposure** |

In every case, MPDOK — by retaining the full Gram matrix and solving via Cholesky or iterative refinement — recovers the high-frequency signal without a rank parameter to tune and without the associated masking of architectural failure.

---

## The Conclusion

The findings of this lab can be stated without qualification.

FRK at m=20 across the 69,000 km² Carlin Trend domain cannot resolve spatial structure at the 10km mineralisation scale. This is not a statistical limitation — it is a geometric one, enforced by the 59km inducing-point spacing that is 5.9× larger than the correlation length of economic interest. No amount of additional training data changes this, because the constraint is in the basis function geometry, not the sample size.

In the real NURE-HSSR Au data, the 55.7% nugget effect masks this failure, compressing the MPDOK advantage to 13 percentage points at HG locations. In the synthetic nested experiment — which represents what block-averaged drillhole or pathfinder geochemistry data would look like — the same structural limit produces a 42 percentage point gap, with MPDOK achieving r=0.92 correlation at short-range peaks and FRK m=20 achieving r=−0.05 (random).

The nugget was masking the failure. The failure was always there. MPDOK exposes it.

---

## Notes

- **All data is free**: the NURE-HSSR dataset is available from USGS ScienceBase at no cost and with no registration. Every result in this lab is fully reproducible.
- **NAD27 coordinates**: NURE data uses NAD27. The WGS84 shift (~100m) is negligible at these scales.
- **The spatial holdout experiment** (Carlin Trend core as test zone, Notebook 2) confirms that even MPDOK cannot extrapolate high-grade Au across 26–55km gaps in a nugget-dominated dataset. This is not a failure — it is honest. The nugget represents genuine irreducible uncertainty.

---

## Files

```
01_data.ipynb          Load NURE-HSSR data, map Au/As, compute Matérn eigenspectrum
02_variogram.ipynb     Fit variogram, run FRK + MPDOK cross-validation on real Au
03_mpdok.ipynb         Synthetic nested-Matérn control experiment at NURE locations
04_comparison.ipynb    Spacing analysis, spectral proof, grand summary figures

mining_phase1.npz      Eigenspectrum, 800×800 distance matrix, variogram statistics
mining_phase2.npz      FRK/MPDOK cross-validation results on real Au
mining_phase3.npz      FRK/MPDOK results on synthetic nested field
mining_phase4.npz      Spacing analysis, nested eigenspectrum, improvement arrays

nevada_nure_raw.csv    Raw NURE-HSSR Nevada data (13,828 records, 6.5 MB)

fig01–fig16            All figures (PNG, 150 dpi)
```

---

*Lab series: MPDOK — Matrix-Pair Dual-Operator Kernel*
*Data: USGS NURE-HSSR via ScienceBase (public domain)*
*Region: Carlin Trend, north-central Nevada (39.5–42°N, 117.5–114.5°W)*
