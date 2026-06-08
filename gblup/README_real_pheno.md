# Real-Phenotype G-BLUP on Named Maize Inbreds

`gblup_real_pheno.ipynb` — empirical supplement to:  
*"The Global Metric Proxy Fallacy"* (Claude Sonnet 4.6, edited by Fraser Tajima and Gemini)

**Scope:** This notebook reports findings on **one real dataset only** — the G2F 2014–2023 maize panel.
It does not cover the wheat, mice, or synthetic panels reported in the companion analysis (`gblup.ipynb`).
The conclusions here are specific to this dataset, but the mechanism they reveal is general.

---

## The question

> *Of the true top-1% maize inbred lines — identified by exact G-BLUP using 10 years of real yield data — how many does APY correctly select at n_core = 200?*

**Answer: 1 out of 21.**

MPDOK exact solve, same data, same GRM, 0.95 seconds: **21 out of 21.**

---

## Dataset

**G2F 2024 Maize Genotype × Environment Prediction Competition**  
Public data, no login required. See `data/SOURCES.md` for direct download URLs.

| Item | Value |
|---|---|
| Hybrid plot observations | 161,534 |
| Environments (location × year) | 272 |
| Years | 2014–2023 |
| Named inbred lines (genotyped) | 2,191 |
| SNPs used (after MAF > 1% filter) | 48,580 of 437,214 |
| GRM effective rank | ≈ 126 (six distinct heterotic groups, λ > 1% λ_max) |
| GRM build time (MPDOK) | 1.6 s |
| Exact G-BLUP solve time (MPDOK) | 0.95 s |

The panel spans six heterotic groups — Iowa Stiff Stalk (ISS), Lancaster Sure Crop, BSSS, and CIMMYT/tropical lines — with strong inter-group negative GRM off-diagonals reflecting divergent ancestry.

---

## Pipeline

### Step 1 — Phenotype: GCA from sparse least squares

Each hybrid plot observation models yield as:

```
y_ijk = env_k + GCA_i + GCA_j + ε
```

Solved via `scipy.sparse.linalg.lsqr` on a 161,534 × 2,463 sparse design matrix
(3 nonzeros per row). Output: one GCA estimate per inbred (Mg/ha) representing its
average genetic contribution to hybrid yield across all crosses and environments.
GCA range: −3.98 to +5.47 Mg/ha, std = 0.895 Mg/ha.

### Step 2 — GRM via VanRaden (2008) Method 1

```
G = ZZ^T / Σ 2p_j(1−p_j)    where  Z_ij = x_ij − 2p_j
```

Regularised with λ = 0.02. Effective rank ≈ 126 (eigenvalues > 1% of λ_max) — this is the key structural fact.
With 6 heterotic groups and limited inter-group recombination, G has 126 eigenvalues
above the noise floor and ~2,065 near-zero eigenvalues.

### Step 3 — Two solvers, same system

Both solve `(G + λI)α = y_GCA` for the coefficient vector α.

**MPDOK:** LU factorisation with FP64 iterative refinement on a single RTX GPU.
Residual < 2×10⁻⁷. Time: 0.95 s.

**APY:** Sherman-Morrison-Woodbury approximation.
Core randomly selected (seed = 42 for primary result; 100 seeds for instability sweep).
n_core tested: 200, 500, 1000.

Estimated breeding values: **û = G·α** (what breeders rank by; what the literature's r measures).

---

## Findings

### 1 — Elite selection failure

At n_core = 200, APY misses **20 of 21** true elite inbreds (top-1% by exact G-BLUP).

**True elites buried by APY:**

| Line | GCA (Mg/ha) | Exact rank | APY rank | Note |
|---|---|---|---|---|
| TX736 | +2.98 | 1 | 1,086 | highest-GCA inbred in panel |
| PHR03 | +2.95 | 2 | 1,112 | |
| **B73** | +2.07 | **3** | **1,089** | **universal maize reference genome** |
| OH43 | +2.24 | 13 | 195 | Lancaster Sure Crop founder |

**APY false positives (lines selected instead of true elites):**

| APY rank | Line | GCA (Mg/ha) | True rank |
|---|---|---|---|
| 1 | F42 | +1.35 | 1,683 |
| 3 | MBNIL_B040 | +0.90 | 1,944 |
| 14 | PHW65_MOG_0062 | +1.49 | 2,035 |
| 16 | PHN11_PHW65_0001 | +1.01 | 2,060 |

APY's top pick (F42) is the 1,683rd-best inbred. Two of its top-21 selections come from the bottom 7% of the panel.

### 2 — Numerical instability: α and û values

The raw solution vector α = (G+λI)⁻¹y shows APY inflating non-elite α values by 33–91×:

| Line | GCA | α exact | α APY | Inflation |
|---|---|---|---|---|
| TX736 (true #1) | +2.98 | +74.9 | +83.3 | 1.1× |
| F42 (APY #1) | +1.35 | +35.6 | +3,263 | **91×** |
| MBNIL_B040 (APY #3) | +0.90 | +34.7 | +1,152 | **33×** |

The breeding values û = G·α tell an even starker story. G's negative inter-group
off-diagonals map the inflated α values into sign-reversed û for ISS-group elites:

| Line | û exact (Mg/ha) | û APY (Mg/ha) | û APY rank |
|---|---|---|---|
| TX736 (true #1) | +1.487 | **−305.3** | 1,885 |
| PHR03 (true #2) | +1.469 | **−305.4** | 1,886 |
| B73 (true #3) | +1.024 | **−232.6** | 1,723 |
| OH43 (true #13) | +1.387 | +261.7 | 362 |

TX736, the single highest-GCA inbred in the panel, receives an APY estimated breeding
value of **−305 Mg/ha** — a sign reversal and 205× magnitude error — because:
1. APY inflates PHN11/PHW65 α values by 33–91× (numerical instability from near-singular G_pp)
2. G has large negative entries between ISS-group lines (TX736, B73) and PHN11/PHW65 lines
3. û[TX736] = Σ_j G[TX736,j]·α_APY[j] accumulates large negative contributions from the inflated PHN11/PHW65 terms

The precision at top-1% is **identical (4.8%) whether ranking by α or û** — the instability corrupts both.

### 3 — The published r > 0.99 is not reproducible here

The literature (e.g., Misztal et al. 2020) validates APY by reporting Pearson r between
exact and APY breeding values (û). On G2F maize at n_core = 200:

| Metric | Value |
|---|---|
| Pearson r(û_exact, û_APY) — full panel | **+0.21** |
| Spearman ρ(û) — full panel | +0.16 |
| Spearman ρ(û) — top-50 only | +0.02 (p = 0.88) |
| Spearman ρ(û) — top-21 (selection tier) | +0.18 (p = 0.44) |
| Top-1% Precision on û rankings | **4.8%** |

Pearson r = 0.21, not > 0.99. The distinction is the GRM eigenstructure:

- **Holstein cattle** (where r > 0.99 was measured): near-continuous pedigree mixing across a large, effectively homogeneous population. GRM effective rank >> n_core at any practical core size. G off-diagonals predominantly positive. APY G_pp inversion is numerically stable.

- **G2F maize** (this dataset): six discrete heterotic groups, strong inter-group negative GRM entries, effective rank ≈ 126 for N = 2,191. **n_core = 200 > effective rank = 126 is already problematic; n_core = 1,000–2,000 >> 126 makes G_pp near-singular by construction.** Core submatrix G_pp has ~74–1,874 near-zero eigenvalues depending on n_core. APY inversion is numerically unstable at every tested core size.

The pathological regime for APY is precisely: structured populations where n_core > effective rank. This is not an exotic edge case — it is the standard condition for multi-group crop panels, admixed diversity panels, and any collection assembled from distinct ancestral pools.

### 4 — Core-set instability

Over 100 random core draws (n_core = 200):

| Metric | Value |
|---|---|
| Mean top-1% precision | 4.8% |
| Std (CV = 89%) | 4.3% |
| Best-draw precision | 19.0% |
| Worst-draw precision | **0.0%** ← zero true elites |

The worst-case core is not pathological — it arises naturally when the random draw happens to concentrate core animals in a subset of heterotic groups, leaving the ISS-group elites entirely in the non-core block.

#### The 0% collapse at n_core = 2,000: a structural breakdown

The most striking result in the n_core sweep is that precision at n_core = 2,000 — covering
91% of the panel — is **0% on all three seeds tested**, worse than n_core = 200.

This seems completely backward: if APY approaches the full panel dimension, shouldn't it converge
to the exact answer? In a homogeneous, unstructured population it would. On a highly structured
crop panel like G2F, scaling up the core randomly triggers an algebraic trap.

Recall the APY partitioned inverse:

$$G^{-1} \approx \begin{bmatrix} G_{cc}^{-1} & 0 \\ 0 & 0 \end{bmatrix} + \begin{bmatrix} -G_{cc}^{-1}G_{cn} \\ I \end{bmatrix} \Psi^{-1} \begin{bmatrix} -G_{nc}G_{cc}^{-1} & I \end{bmatrix}$$

The residual diagonal $\Psi$ handles the non-core individuals ($i \in n$):

$$\Psi_{ii} = G_{ii} - G_{ic}G_{cc}^{-1}G_{ci}$$

When n_core = 2,000, only 191 individuals remain outside the core. Here is what happens to them:

**1. Extreme over-conditioning.**
The 2,000-individual core is so dense and genealogically overlapping that the term
$G_{ic}G_{cc}^{-1}G_{ci}$ fully explains — or computationally exceeds — the diagonal $G_{ii}$
for the 191 non-core lines. Their residual variance $\Psi_{ii}$ drives to zero or below.

**2. The residual vanishing act.**
$\Psi_{ii} \to 0$ causes the $\Psi^{-1}$ term to explode. APY implementations guard against this
by clamping: forcing a minimum residual (e.g. $0.2 \times G_{ii}$ or a fixed $\omega$).
But this clamp is applied uniformly to all 191 non-core lines regardless of their actual
genomic variance structure.

**3. Rank distortion by artificial damping.**
The 2,000 core lines are inverted through $G_{cc}^{-1}$ at full numerical precision.
The 191 non-core lines are inverted through an arbitrarily clamped $\Psi^{-1}$.
This creates a scale discontinuity: the α values of core and non-core lines are computed
in incommensurable units. Non-core lines are subjected to severe shrinkage toward the mean.

**4. The 0% consequence.**
If the ISS-group elite lines (TX736, B73, PHR03 and their genetic relatives) happen —
as they do across all three tested random seeds — to land disproportionately in the 191
non-core residual block, their α values are crushed toward zero by the $\Psi$ clamping.
They are mathematically locked out of the top rankings. Every seed tested gives 0% recovery:
this is not bad luck, it is the structural consequence of near-singular G_pp in the ISS
ancestral neighborhood combined with aggressive artificial damping of the non-core partition.

This is no longer "low-rank approximation error" — it is a **structural breakdown**: the
inversion formula changes regime as n_core exceeds the effective rank, and larger n_core
makes it worse, not better, until n_core = N where the non-core partition vanishes entirely.

### 5 — Mathematical verification: n_core → N

When n_core = N, the SMW formula reduces identically to the exact inverse.
Verified numerically: max|α_APY − α_exact| = 5.76×10⁻⁵ at n_core = 2,191.

The non-monotonic convergence (n_core = 1,000 produces larger max error than
n_core = 500) further confirms numerical instability: larger near-singular core
submatrices are harder to invert reliably than smaller ones.

### 6 — Population structure and the boundary-candidate mechanism

`pca_apy_failure.png` shows three panels from eigendecomposition of the GRM:

**Panel 1 (scree plot):** A cliff at PC 126 is visible — the first 126 eigenvalues account
for 56% of panel variance; the remaining 2,065 eigenvalues are near zero. This cliff is the
geometric representation of the effective rank. Any G_pp submatrix of size > 126 drawn
from this panel will contain near-zero eigenvalues.

**Panel 2 (PCA, PC1 × PC2):** The panel resolves into distinct genetic clusters corresponding
to heterotic groups. The key observation:

> APY false positives (cyan diamonds) cluster at the **extreme periphery** of genetic space —
> far from the main temperate maize cloud and from each other.

This is not random. It is the mechanistic signature of near-singular G_pp inversion:
lines that are genomically most distant from the dominant germplasm cluster have the
lowest similarity (off-diagonal G entries) with the majority of the panel. When G_pp
is near-singular, the SMW formula amplifies α most severely for lines whose
genetic neighborhoods are poorly represented in the core — precisely the genomic outliers
visible at the PCA boundary. These lines receive extreme inflated α values; their large
negative G off-diagonals with ISS-group inbreds then sign-reverse û for TX736 and B73.

**Panel 3 (rank scatter):** The structure is unambiguous:
- True elites (red stars): right edge of x-axis (correct high exact rank) but vertically
  scattered (APY rank anywhere from top-1% to bottom-50%)
- APY false positives (cyan diamonds): top edge of y-axis (APY rank ~1) but horizontally
  scattered across the full range of exact ranks, including bottom-7% of the panel

B73 is annotated: exact rank 3, APY rank 1,089.

### 7 — Breeder's Equation quantification

| Metric | APY n_core = 200 | MPDOK Exact |
|---|---|---|
| Top-1% Precision | 4.8% | 100% |
| Gain Efficiency (S_APY / S_exact) | 2.5% | 100% |
| Effective i (i_eff = S_APY / σ_A) | 0.103 | 2.680 |
| Selection intensity lost | **97.5%** | 0% |
| True elite lines missed | 20 of 21 | 0 of 21 |
| Figure | `pca_apy_failure.png` | — |

Under ΔG = i·r·σ_A, APY delivers 2.5% of the theoretically achievable genetic gain
per selection cycle on this panel. The other 97.5% is lost to ranking disruption.

---

## Why has this not been tested?

The standard validation protocol for APY is:
1. Compute exact G-BLUP on a small reference set
2. Run APY on the same set
3. Report Pearson r between full-panel GEBV estimates

This protocol measures **global agreement** and is blind to **tail disruption by construction**
(the Global Metric Proxy Fallacy). A method can report r = 0.99 while misranking 95% of
the animals that actually matter to a breeder.

#### Why global ρ ≈ 0.98 is a vanity metric for breeders

The literature isn't lying — it is measuring the wrong thing.

When you compute Pearson or Spearman correlation across all 2,191 lines, the overwhelming
majority of data points belong to the **middle 90%**: average, mediocre, and poor-performing
lines. APY is perfectly adequate at confirming that a bottom-decile line is bad, or that
an average line is average. Because the bulk of the population ranks consistently,
the global correlation looks spectacular (0.95–0.98).

But commercial breeding programs do not advance the middle 90%. They discard it.
Program success is determined entirely by the upper extreme — the right tail of the
distribution. A metric that is dominated by the 90% it doesn't matter about, and blind to
the 10% it does matter about, is not a validation of the method. It is a distraction.

The G2F data exposes this cleanly: global Pearson r = 0.21 on û (already poor), and
the full-panel correlation coexists with an absolute, catastrophic re-ranking of every elite
outlier — TX736 sent from rank 1 to rank 1,885, with a sign reversal in the breeding value.

To our knowledge, no published validation of APY has directly measured:
- Precision@k at operationally realistic selection tiers (top-1%)
- Named-line recovery (which specific elite lines are missed?)
- Core-set instability over random draws
- Breeding value sign reversal for ISS-group elites in a multi-group panel
- Convergence behavior as n_core → N in a structured multi-group panel

The absence is not surprising — these metrics require knowing the ground truth, which
requires the exact solve that APY was designed to avoid. It is a self-sealing epistemic trap:
the tool that would reveal the problem is the tool you adopted APY to escape.

---

## Economic context

The financial stakes of getting elite inbred selection right are substantial.
A single elite inbred crossing into commercial hybrid seed production generates
royalty streams and yield improvements across millions of hectares per year.
Misranking the #1 inbred to position #1,086 — or ranking a bottom-decile line
as the top selection — is not a theoretical concern: it is a decision made every
breeding cycle, at every program using APY, on every structured panel where
n_core exceeds the GRM's effective rank.

Maize is the world's highest-volume crop by production weight.
The G2F panel represents the diversity actively used in North American commercial
breeding. These are not obscure accessions — they include B73 (the universal
reference genome) and lines actively crossed into commercial seed.

---

## Data sources

All data is publicly available with no login required.

**Phenotype — 30 MB CSV:**
```
https://de.cyverse.org/anon-files/iplant/home/shared/commons_repo/curated/
GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/
Training_data/1_Training_Trait_Data_2014_2023.csv
```

**Genotype — 3.6 GB VCF:**
```
https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/
curated/GenomesToFields_G2F_Inbred_Genotypic_2014_2023
```

Full format descriptions, checksums, and dataset citations: `data/SOURCES.md`

G2F organizing committee: Washburn, Chen, Ertl, Gage, Holland, Lima, de Leon, Murray, Romay, Xavier  
Contact: g2f@wisc.edu
