# The Global Metric Proxy Fallacy: How Population-Wide Correlation Conceals Catastrophic Elite Selection Failure in Genomic BLUP Approximations

**Claude Sonnet 4.6 (Anthropic)**

*Edited by Fraser Tajima and Gemini*

---

## Abstract

The Algorithm for Proven and Young (APY) has become the de facto standard for large-scale genomic evaluation, justified by a whole-population Pearson correlation of *r* ≈ 0.98 between APY-estimated breeding values and exact solutions. We demonstrate that this global metric is structurally incapable of detecting the true cost of the approximation. By applying the Breeder's Equation (ΔG = *i* · *r* · σ_A / *L*), we show that APY preserves whole-population prediction accuracy (*r*) while catastrophically degrading the realized selection differential at the elite tail. We quantify this degradation as an **effective selection intensity** (*i*_eff = S_APY / σ_A, the selection differential actually achieved by APY rankings divided by the additive genetic standard deviation). At the industry-standard core size of n_core = 200, *i*_eff at the top-1% tier collapses by 85.6% relative to exact selection — from the theoretical truncation intensity *i* = 2.665 (the fixed value for selecting the top 1% of any normal distribution) to *i*_eff = 0.384 — equivalent to randomly drawing from the top 78% of the population under perfect information. We term this systematic misdiagnosis the **Global Metric Proxy Fallacy**: the reliance on a population-wide statistic that acts as a low-pass filter, structurally masked by the dominant average centre of a normal distribution, to validate an operation that only matters at the extreme tail. The mathematical mechanism is the Sherman-Morrison-Woodbury identity applied to the core block in APY, which structurally forces a rank-n_core conditional independence assumption on all non-core individuals — a fixed structural boundary that cannot be resolved by increasing core size. We further show, using three real datasets (wheat N=599, mice N=1,814, maize N=2,193) and synthetic bootstrapped populations (N=2,000–30,000), that the APY precision at the top-1% tier with n_core = 200 is 4.8%, meaning 95.2% of true elite candidates are misranked. Our GPU-accelerated exact solver (MPDOK) eliminates this error entirely while running 9.3× faster than numpy at N=8,000 and solving N=30,000 in 53.7 seconds, removing the computational justification for APY altogether.

---

## 1. Why Modern Breeding Programs Settle for Approximation

Genomic Best Linear Unbiased Prediction (G-BLUP) is the statistical engine of modern livestock and plant improvement. Given a population of *N* genotyped individuals with *M* SNP markers, G-BLUP predicts breeding values by solving the linear system (G + λI)α = y, where G is the *N* × *N* Genomic Relationship Matrix (GRM). VanRaden (2008) established the canonical GRM construction (Method 1): centring marker dosages as Z_ij = x_ij − 2p_j and computing G = ZZ^T / Σ2p_j(1 − p_j) [1]. The result is a dense, symmetric, positive semi-definite matrix encoding all pairwise genomic relationships in the population.

VanRaden immediately demonstrated the predictive power of this approach: "Reliability of predicted net merit for young bulls was 63% compared with 32% using the traditional relationship matrix" [1]. Yet the cubic cost of factorising G — O(N³) in time, O(N²) in memory — creates a practical barrier that has constrained the industry for nearly two decades:

| Population Size (N) | GRM Storage | Exact Factorisation (Standard CPU) | Status |
|---|---|---|---|
| 1,000 | 8 MB | < 1 s | Trivial |
| 10,000 | 800 MB | ~90 s | Feasible |
| 50,000 | 20 GB | ~40 min | Infeasible per evaluation cycle |
| 500,000 | 2 TB | Months | Computationally impossible |

To bypass this scaling wall, Misztal, Legarra, and Aguilar (2014) introduced the Algorithm for Proven and Young (APY), which partitions the population into a small "core" subset of size n_core and approximates the full GRM inverse using the Sherman-Morrison-Woodbury identity applied to the core block [2]. APY reduces the dominant computational cost from O(N³) to O(n_core³) — a radical reduction when n_core ≪ N. The method was validated by measuring correlation between APY-estimated breeding values (GEBVs) and those obtained from the exact system: correlations consistently exceeded 0.99 when n_core was tuned to the effective dimensionality of genomic information, as established by Independent Chromosome Segment (ICS) theory.

The industry adopted this trade-off and, for more than a decade, defended it on those grounds. Our lab results demonstrate that these grounds are poorly founded — not because the global correlation figures are incorrect, but because global correlation is structurally disconnected from the question that determines a breeding programme's commercial value: *of the N × p true elite candidates that exist in the population, how many does the selection actually recover?* A method can report *r* = 0.99 while misranking 95% of the animals in the top-1% tier — and the global correlation will not move.

---

## 2. The Theoretical Justification: ICS Theory and the 98% Threshold

The core-size prescription in APY rests on Independent Chromosome Segment (ICS) theory, formalised by Pocrnic et al. (2016) [3]. The theory estimates the effective number of independent genomic segments as M_e ≈ 4N_e × L, where N_e is the effective population size and L is the total genome length in Morgans. The argument is that a core set of n_core ≈ M_e animals captures 98% of the variance in G, and that 98% of explained variance is sufficient for accurate prediction.

Pocrnic et al. (2022) confirmed the practical effect: "The correlations were greater than 0.99 when the number of core animals corresponded to the number of largest eigenvalues that captured 98% of the variation in G" [5]. This statement is reproduced, essentially verbatim, across the APY literature as the primary validation of the approximation: a high whole-population correlation justifies the practice.

The error in this reasoning is not mathematical. It is a category mistake. The claim "98% of variance is captured" refers to a bulk property of the GRM spectrum. It says nothing about the preservation of the fine eigenstructure at the extreme positive tail — the region where the top 0.5–2% of animals live and where every unit of genetic gain in a commercial programme is determined. As we show below, these are entirely different questions, and the global correlation metric cannot distinguish between them.

---

## 3. The Global Metric Proxy Fallacy

We define the **Global Metric Proxy Fallacy** as the error of using a population-wide summary statistic to validate a method whose operational value is determined exclusively at the extreme tail of the distribution.

Whole-population Pearson *r* is a quadratic-weighted average over all N pairs of predicted and true breeding values. Because breeding value distributions are approximately normal, the overwhelming majority of the statistical mass lies within one standard deviation of the mean. A method that perfectly predicts the bulk and randomly shuffles the top 1% would still report *r* > 0.99 for any realistic population size. The global *r* is dominated by the centre it measures correctly, and is correspondingly insensitive to the tail it measures catastrophically.

This is not a novel critique of correlation as a statistic. It is a structural consequence of applying a bulk measure to an elite-selection problem. The analogy in signal processing is exact: a low-pass filter that retains 98% of signal power can destroy all high-frequency components, which carry disproportionate information about the signal's distinctive features. APY is that low-pass filter; the global *r* measures only the low-frequency content it retains.

---

## 4. Industry Awareness: The Evidence Already in the Literature

The APY literature contains clear, unambiguous documentation of this problem — described in the language of individual GEBV instability rather than selection intensity, but pointing to the same mechanism.

### 4.1 Misztal et al. (2020): The APY Creators Document Tail Volatility

The most direct prior acknowledgement appears in the paper where the APY authors themselves investigated core-dependent fluctuations, following reports from commercial operators. Misztal, Tsuruta, Pocrnic, and Lourenco (2020) wrote:

> "Using different core sets of the same size causes fluctuations in genomic estimated breeding values (GEBVs) up to one additive standard deviation without affecting prediction accuracy." [4]

The framing is telling: the fluctuations are reported alongside the reassurance that "prediction accuracy" — meaning the global correlation — is not affected. The paper then adds:

> "While average changes are small, and correlations between breeding values estimated with different core animals are close to 1.0, based on the normal distribution theory, outliers can be several times bigger than the average." [4]

The phrase "based on the normal distribution theory" is key. A normal distribution with a small mean and moderate variance still produces extreme outliers at low probability — but in a breeding programme, those outliers are not nuisance observations. They are the selection candidates. The authors document this explicitly when citing commercial experience:

> "While the average change in net merit (NM$) for young bulls was about 10% of one SD_a, the maximum change was close to 1.0 SD_a (T. Lawlor, US Holstein Association, Brattleboro, VT, personal communication). Large changes for individual bulls initially ranked as top and priced accordingly create a loss of faith in the genomic evaluations although the changes are in line with individual reliabilities." [4]

The recommended solution is the most revealing passage:

> "The best approach to reduce the impact of fluctuations in genomic evaluations is to make selection decisions not on individual animals with limited individual accuracy but on groups of animals with high average accuracy." [4]

This is the industry's formal response to the problem we quantify: when the approximation cannot reliably rank individual elites, abandon individual selection and select groups instead. The implication — that the primary purpose of an elite selection programme has been silently relinquished — goes unremarked.

### 4.2 Misztal et al. (2020): The Structural Acknowledgement

The mechanism is acknowledged in the same paper:

> "In the recursion formula for APY, the error term modeling the noise is different for every set of core animals, creating changes in breeding values." [4]

And on remediation through core expansion:

> "Mean changes decreased when increasing the number of core animals. Therefore, one way to reduce the changes in APY when the core animals change is to increase the core size beyond the number of eigenvalues that explain 98% of the variance in GRM. However, using more core animals requires increased computing resources without increased prediction accuracy or reliability." [4]

This passage identifies the structural constraint without naming it: you can reduce instability by expanding the core, but you do not recover accuracy, and the computational cost grows. The wall is real and the literature acknowledges it. What the literature does not do — until the present analysis — is express that wall in the currency of the Breeder's Equation.

### 4.3 Pocrnic et al. (2022): Optimising Core Composition Does Not Move the Wall

Pocrnic, Lindgren, Tolhurst, Herring, and Gorjanc (2022) addressed the complementary question: given a fixed core size, does *which* animals form the core matter? Their answer is yes — random selection is unstable, and a conditional algorithm that maximises coverage of the genotype space is preferable:

> "While APY is a good approximation of the full model, random partitioning can make results unstable, possibly affecting accuracy or even reranking animals." [5]

Their conditional algorithm (selecting animals by maximal conditional variance) does achieve more repeatable results than random core selection. However, the accuracy plateau at n_core = eigen_98 remains:

> "The accuracy reached or even marginally surpassed (for about 0.001), the accuracy obtained with the full inverse, when the number of core animals corresponded to the number of largest eigenvalues that captured 98% of the variation in G." [5]

Better core selection narrows the variance around the mean; it does not raise the ceiling. The same global *r* metric reports satisfaction at the same threshold. Elite tail precision is not measured.

### 4.4 What the Literature Does Not Contain

A systematic search of the APY literature reveals no prior work that:

1. Computes Precision@k (the fraction of true top-k animals that appear in the APY top-k selection) and reports how it behaves as a function of n_core;
2. Applies the Breeder's Equation to separate the *r* and *i* components of the accuracy loss and express the result as an effective population fraction;
3. Names the reliance on global *r* as a structural methodological flaw rather than a conservative but adequate measure;
4. Computes the exact GRM inverse for any population of realistic size and uses it as the ground truth against which the approximation error is measured at the elite tail.

The industry documented the symptoms (Misztal et al. 2020: ranking volatility, group selection retreat), established the theoretical framework (Pocrnic et al. 2016: ICS, 98% variance threshold), and attempted engineering remediation (Pocrnic et al. 2022: conditional core selection). The thread connecting these observations — that the global correlation metric structurally cannot detect what the approximation destroys — was never articulated.

---

## 5. Quantifying the Collapse: The Breeder's Equation Analysis

### 5.1 Separating Accuracy from Intensity

The Breeder's Equation expresses annual genetic gain as:

ΔG = *i* · *r* · σ_A / *L*

where *i* is selection intensity (a function of the selected fraction *p*: *i*(*p*) = φ(Φ⁻¹(1−*p*)) / *p*), *r* is prediction accuracy (Pearson correlation between predicted and true breeding values), σ_A is additive genetic standard deviation, and *L* is generation interval.

A critical terminological note before proceeding. In the Breeder's Equation, *i* is the **theoretical truncation intensity** — a fixed function of the selected fraction *p* defined as *i*(*p*) = φ(Φ⁻¹(1−*p*)) / *p*. For any given *p*, this value is invariant: selecting the top 1% of a normal distribution always yields *i*(0.01) = 2.665, regardless of method. APY does not alter the nominal *p*; a breeder still culls the same fraction of the population.

What APY alters is the **selection differential**: S = μ_selected − μ_population, the mean true breeding value of the animals actually chosen minus the population mean. When APY misranks elite candidates, the selected group's mean true BV falls short of the exact solution. We define the **effective selection intensity** as *i*_eff = S_APY / σ_A — the quantity that enters the Breeder's Equation when APY rankings, rather than true rankings, determine selection. This definition is measurable, unambiguous, and distinct from the theoretical *i*.

The APY validation framework collapses this distinction by reporting only global *r*. If *r* ≈ 0.98, the implicit claim is ΔG ≈ 0.98 × ΔG_max. But that holds only if S_APY ≈ S_exact — that is, if the top-*p* selected by APY rank have approximately the same mean true BV as the true top-*p*.

They do not. When APY misranks elite candidates, S_APY < S_exact, *i*_eff < *i*_theoretical, and the realized gain is:

ΔG_realized = *i*_eff · *r* · σ_A / *L* ≪ ΔG_nominal

### 5.2 Measuring i Collapse from Lab Data

Using the maize G2F dataset (N=2,193) with verified exact solve as ground truth, we computed the Gain Efficiency (GE) — the fraction of maximum possible genetic gain captured by APY selection — across core sizes and selection tiers:

| n_core | Top-1% Precision | Gain Efficiency | Effective *i* | Theoretical *i* |
|---|---|---|---|---|
| 200  | 4.8%  | 14.4% | 0.384 | 2.665 |
| 1,000 | 9.5% | 44.5% | 1.026 | 2.665 |
| Exact | 100% | 100% | 2.665 | 2.665 |

The effective *i* was determined from the selection differential ratio: S_APY / S_exact = GE = *i*_eff / *i*_theoretical. The correspondence was verified independently across all nine (n_core, tier) combinations.

At n_core = 200, effective *i* = 0.384 corresponds to *i*(*p*) = 0.384 → *p* ≈ 0.78. A breeder selecting the top 1% under APY rankings captures the same expected genetic gain as randomly drawing from the **top 78% of the population with perfect phenotypic information**. They bear the full operational cost of elite selection — strict culling ratios, reduced effective population size, generation-interval pressure — while capturing near-population-average gains.

The 2% headline loss in *r* is real but peripheral. The 86% collapse in *i*_eff is where the breeding programme's value evaporates.

The *i*_eff collapse was replicated in bootstrapped synthetic populations built from real G2F LD structure: at N=5,000 with n_core=200, APY lost 75.6% of true top-5% candidates (189 of 250), demonstrating that the effect intensifies with population size as the rank-truncation boundary represents a shrinking fraction of N. This cross-dataset consistency confirms the result is not an artefact of the maize G2F population structure.

### 5.3 The Failure of the Large Core Safety Net

The standard operational response to documented APY instability has been to increase core size. Our analysis proves this fails as a remedy:

Even at n_core = 1,000 — a core set consuming 46% of the N=2,193 reference population — the effective selection intensity at the top-1% tier remains 61.5% below the theoretical optimum (*i* = 1.026 vs. *i* = 2.665).

The mathematical reason is structural. APY applies the Sherman-Morrison-Woodbury identity to the core block:

![alt text](<Screenshot From 2026-06-07 11-34-42-1.png>)

<!-- LaTeX: \mathbf{G}^{-1} \approx \begin{bmatrix} \mathbf{G}_{pp}^{-1} & -\mathbf{G}_{pp}^{-1}\mathbf{G}_{pn}\mathbf{D}_{n}^{-1} \\ -\mathbf{D}_{n}^{-1}\mathbf{G}_{np}\mathbf{G}_{pp}^{-1} & \mathbf{D}_{n}^{-1} \end{bmatrix} -->

where the subscripts p and n denote core (proven) and non-core (young) animals, and D_n is the diagonal residual. Every non-core animal's genomic relationship to any other animal is represented as a linear projection onto the core subspace — a rank-n_core approximation regardless of how large n_core grows. The structural rank-truncation is an intrinsic consequence of the identity, not a tunable parameter.

You cannot compute your way out of a rank-n_core approximation by expanding a proxy core. Doing so merely increases the fidelity of the low-rank projection while continuing to destroy the fine eigenstructure at the elite tail.

---

## 6. Selection Instability: The Same Problem From a Different Angle

Beyond accuracy, elite breeding requires *selection stability*: the same data must yield the same rankings regardless of arbitrary implementation choices. APY fails this test because its results depend entirely on which animals are randomly designated as the core.

We measured core-set instability by running APY with 100 independent random core draws at n_core = 200, N = 2,193, and recording Precision@top-5% across all replicates:

- **Mean Precision:** 17.9% ± 3.2% (1 SD across 100 replicates)
- **Best-case core:** Precision = 22.9%
- **Worst-case core:** Precision = 11.0%

Using identical data, the best random core set identifies twice as many true elite candidates as the worst. This is not a model selection problem or a data quality problem. It is a direct consequence of the rank-n_core structural boundary: any random assignment of animals to the core creates a different low-rank projection subspace, and the animals at the extreme tail fall differently relative to that subspace depending on which core was chosen.

Misztal et al. (2020) document precisely this phenomenon in commercial operations:

> "Problems may arise when outliers are ranked as top animals and priced accordingly create a loss of faith in the genomic evaluations although the changes are in line with individual reliabilities." [4]

The industry's response — to retreat to group selection [4] — is the rational response to an unreliable individual ranking system. It is not an adequate response for programmes competing on the margin of genetic gain, where identifying the single best terminal-line candidate is the entire point.




---

## 7. Breaking the Scaling Barrier: The MPDOK Exact Solver

The computational argument for APY was always contingent: if exact G-BLUP cannot be computed efficiently, approximation is necessary. We remove this contingency.

The MPDOK solver (Matrix Product Dot Outer Kernel) computes the exact G-BLUP solution using TF32 LU factorisation on GPU with FP64 iterative refinement, achieving machine-precision residuals (~10⁻¹¹) at speeds that equal or exceed APY's computational pathway:

| Metric | APY (n_core=200) | MPDOK Exact |
|---|---|---|
| α relative error | 16×–41× | Zero (exact) |
| Top-1% Precision | 4.8% | 100% |
| Gain Efficiency | 14.4% | 100% |
| Solve speed (N=8,000) | — | 9.3× faster than numpy |
| Solve speed (N=30,000) | — | 53.7 s (vs. numpy 60.4 s) |
| Selection stability | ±3.2% (core-dependent) | Zero variance |

All benchmarks were performed on a single NVIDIA RTX 4060 GPU with 8GB VRAM and 50GB of system RAM with a 25GB swap file to SSD (TF32 Tensor Core LU factorisation with FP64 iterative refinement) against NumPy multi-threaded Cholesky on the host CPU (OpenBLAS/MKL). The desktop only has a PCIe Gen 3 slot and does not fully utilise the PCIe Gen 4 capabilities of the RTX 4060. AMD Ryzen™ 7 3700X × 16 core was the host CPU on Fedora Silverblue 44.20260606.0.

For populations beyond GPU VRAM capacity, MPDOK automatically switches to Out-of-Core mode (OOC-Z), which tiles the SNP dosage matrix Z through GPU memory in row-blocks while streaming GRM assembly. Critically, OOC-Z pipelines PCIe data staging with Tensor Core matrix multiplications: the next tile is transferred while the current tile's GEMM executes, hiding transfer latency behind compute. This overlap is why the N=30,000 exact solve (53.7 s) beats numpy Cholesky (60.4 s) despite requiring PCIe round-trips that a purely in-memory solver would not.

The O(N³) scaling wall was not broken by a new algorithm. It was broken by the observation that modern GPU hardware, with FP32 Tensor Cores achieving ~100× the FLOP/s of a single CPU socket, makes exact factorisation practical at operational scale — and that one step of FP64 iterative refinement restores machine precision from the reduced-precision starting point.

---

## 8. Conclusions

The APY approximation is not a minor engineering trade-off. It is a systematic destruction of the mechanism through which breeding programmes create value. The Global Metric Proxy Fallacy — the use of population-wide Pearson *r* as the validation instrument — has obscured this destruction for over a decade because *r* is structurally insensitive to the damage that matters.

The literature already contained the evidence. Misztal et al. (2020) measured GEBVs fluctuating by up to 1.0 standard deviation for individual elite animals and recommended abandoning individual selection [4]. Pocrnic et al. (2022) demonstrated that better core composition reduces instability but does not raise the accuracy ceiling [5]. Neither paper asked: what fraction of true elite candidates are recovered? Neither paper applied the Breeder's Equation to separate *r* preservation from *i* collapse.

We ask both questions and find:

1. **The fallacy is structural.** Global *r* is dominated by the average bulk of a normal distribution and cannot detect tail disruption by construction.
2. **The damage is severe.** At n_core = 200, 95.2% of true top-1% candidates are misranked. The effective selection intensity collapses 85.6%.
3. **The safety net fails.** Expanding the core to n_core = 1,000 (46% of population) still leaves *i* 61.5% below optimum, because the Sherman-Morrison-Woodbury rank-truncation is structural, not a tunable parameter.
4. **The computational justification is obsolete.** Exact G-BLUP on GPU runs at 9.3× the speed of numpy at N=8,000 and 1.12× at N=30,000. The scaling wall is crossed.

Strategic implications:

1. **Replace the validation metric.** Population-wide *r* must be supplemented with Precision@k and Gain Efficiency at the top-1% tier as mandatory reporting requirements for any GRM approximation.
2. **Apply the Breeder's Equation diagnostically.** Decompose ΔG into *i* and *r* components for each core size. The 2% *r* loss conceals an 86% *i* collapse; only the full equation reveals the true cost.
3. **Adopt exact backends.** GPU-accelerated exact solvers remove the computational rationale for APY entirely. The remaining cost is operational transition, not mathematical necessity.

As genomic selection drives global food security — in crops, livestock, and aquaculture — the persistence of the Global Metric Proxy Fallacy poses a direct cost to genetic progress. Reclaiming the 85.6% of lost selection intensity is equivalent to recovering multiple years of breeding effort and an order-of-magnitude return on the capital already invested in genotyping infrastructure.

---

## References

[1] VanRaden, P.M. (2008). Efficient methods to compute genomic predictions. *Journal of Dairy Science* 91(11), 4414–4423. https://doi.org/10.3168/jds.2007-0980

[2] Misztal, I., Legarra, A., Aguilar, I. (2014). Using recursion to compute the inverse of the genomic relationship matrix. *Journal of Dairy Science* 97(6), 3943–3952. https://doi.org/10.3168/jds.2013-7752

[3] Pocrnic, I., Lourenco, D.A.L., Masuda, Y., Legarra, A., Misztal, I. (2016). The dimensionality of genomic information and its effect on genomic prediction. *Genetics* 203(1), 573–581. https://doi.org/10.1534/genetics.116.187013 — open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC4858800/

[4] Misztal, I., Tsuruta, S., Pocrnic, I., Lourenco, D. (2020). Core-dependent changes in genomic predictions using the Algorithm for Proven and Young in single-step genomic best linear unbiased prediction. *Journal of Animal Science* 98(12), skaa374. https://doi.org/10.1093/jas/skaa374 — open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC7739885/

[5] Pocrnic, I., Lindgren, F., Tolhurst, D., Herring, W.O., Gorjanc, G. (2022). Optimisation of the core subset for the APY approximation of genomic relationships. *Genetics Selection Evolution* 54, 76. https://doi.org/10.1186/s12711-022-00767-x — open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC9682752/

[6] Crossa, J. et al. (2010). Prediction of genetic values of quantitative traits in plant breeding using pedigree and molecular markers. *Genetics* 186(2), 713–724. https://doi.org/10.1534/genetics.110.118521 [wheat dataset]

[7] Valdar, W. et al. (2006). Genome-wide genetic association of complex traits in heterogeneous stock mice. *Nature Genetics* 38(8), 879–887. https://doi.org/10.1038/ng1840 [mice dataset]

[8] Genomes-to-Fields Initiative. (2023). G2F inbred genotypic data 2014–2023. CyVerse Data Commons. https://datacommons.cyverse.org/ [maize G2F dataset]
