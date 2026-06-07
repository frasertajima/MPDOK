"""
Genomic Relationship Matrix (GRM) construction.

VanRaden (2008) Method 1:
    Z_ij = (x_ij - 2p_j) / sqrt(2 * Σ p_j(1-p_j))
    G = Z Z^T

where x_ij ∈ {0,1,2} is the SNP dosage (copies of minor allele) for individual
i at locus j, and p_j is the minor allele frequency at locus j.

The resulting G is an N×N symmetric positive semi-definite matrix with diagonal
entries ≈1 (self-relationship) and off-diagonal entries ≈ 2*IBD probability.
"""

import numpy as np
import time


# ---------------------------------------------------------------------------
# 1. Core GRM construction
# ---------------------------------------------------------------------------

def compute_grm(X: np.ndarray, method: str = "vanraden1",
                min_maf: float = 0.01) -> tuple[np.ndarray, dict]:
    """Build Genomic Relationship Matrix from SNP dosage matrix.

    Args:
        X:       (N, M) int8/float array, entries ∈ {0, 1, 2} (minor allele dosage)
        method:  "vanraden1" (default) or "simple" (X @ X.T / M)
        min_maf: minor allele frequency filter (remove monomorphic SNPs)

    Returns:
        G:    (N, N) float64 GRM, symmetric SPD after regularisation
        info: dict with allele frequencies, scale factor, M_used
    """
    X = np.asarray(X, dtype=np.float64)
    N, M = X.shape

    # Allele frequencies (p_j = mean dosage / 2)
    p = X.mean(axis=0) / 2.0          # (M,)

    # MAF filter
    maf = np.minimum(p, 1.0 - p)
    keep = maf >= min_maf
    X = X[:, keep]
    p = p[keep]
    M_used = keep.sum()

    if method == "vanraden1":
        # Centre by 2p
        Z = X - 2.0 * p[np.newaxis, :]               # (N, M)
        # Scale factor = 2 Σ p_j(1-p_j)
        scale = 2.0 * np.sum(p * (1.0 - p))
        G = (Z @ Z.T) / scale
    else:
        # Simple: X X^T / M
        G = (X @ X.T) / M_used
        scale = M_used

    info = {"p": p, "scale": scale, "M_used": int(M_used), "M_total": M,
            "N": N, "method": method,
            "X_filtered": X.astype(np.float32)}   # kept for Z-based OOC solver
    return G.astype(np.float64), info


def regularise_grm(G: np.ndarray, lam: float) -> np.ndarray:
    """Return G + lam * I (in-place addition on copy)."""
    A = G.copy()
    np.fill_diagonal(A, np.diag(A) + lam)
    return A


# ---------------------------------------------------------------------------
# 2. VCF parser (streaming, no external dependencies)
# ---------------------------------------------------------------------------

def parse_vcf_genotypes(vcf_path: str, subsample_step: int = 1,
                        max_snps: int | None = None,
                        verbose: bool = True) -> tuple[np.ndarray, list, list]:
    """Stream-parse a VCF file into a SNP dosage matrix.

    Handles biallelic SNPs only. Multiallelic and non-PASS variants are
    kept (the G2F dataset is minimally filtered, as noted in its readme).
    Missing genotypes (./.) are imputed to the column mean.

    Args:
        vcf_path:        path to .vcf file (uncompressed)
        subsample_step:  keep every k-th SNP (1 = all, 9 = ~1/9 of SNPs)
        max_snps:        stop after this many SNPs (None = all)
        verbose:         print progress every 50k SNPs

    Returns:
        X:        (N, M) int8 dosage matrix
        samples:  list of N sample IDs
        snp_ids:  list of M SNP IDs (CHROM:POS)
    """
    samples = []
    rows = []
    snp_ids = []
    n_seen = 0
    n_kept = 0
    t0 = time.time()

    with open(vcf_path, "r") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                samples = cols[9:]               # sample IDs start at col 9
                continue

            # Data line
            n_seen += 1
            if (n_seen - 1) % subsample_step != 0:
                continue

            parts = line.rstrip("\n").split("\t")
            chrom, pos, snp_id = parts[0], parts[1], parts[2]
            if snp_id == ".":
                snp_id = f"{chrom}:{pos}"

            gt_fields = parts[9:]
            dosage = np.empty(len(gt_fields), dtype=np.int8)
            for i, gt in enumerate(gt_fields):
                alleles = gt[:3] if len(gt) >= 3 else gt  # take GT before any ':'
                if "." in alleles:
                    dosage[i] = -1   # missing, impute later
                else:
                    a0 = int(alleles[0])
                    a1 = int(alleles[2]) if len(alleles) > 2 else int(alleles[-1])
                    dosage[i] = a0 + a1

            # Impute missing to rounded column mean
            missing = dosage < 0
            if missing.any():
                mean_dos = dosage[~missing].mean() if (~missing).sum() > 0 else 1
                dosage[missing] = int(round(mean_dos))

            rows.append(dosage)
            snp_ids.append(snp_id)
            n_kept += 1

            if verbose and n_kept % 5000 == 0:
                elapsed = time.time() - t0
                rate = n_seen / elapsed
                print(f"  SNPs read: {n_seen:,}  kept: {n_kept:,}  "
                      f"({elapsed:.0f}s, {rate:.0f} lines/s)")

            if max_snps and n_kept >= max_snps:
                break

    X = np.array(rows, dtype=np.int8).T   # transpose: (N, M)
    if verbose:
        print(f"Parsed {n_seen:,} SNPs total, kept {n_kept:,}. "
              f"Matrix: {X.shape}  ({X.nbytes/1e9:.2f} GB)")
    return X, samples, snp_ids


# ---------------------------------------------------------------------------
# 3. Synthetic GRM with realistic genomic eigenspectrum (for scaling study)
# ---------------------------------------------------------------------------

def bootstrap_grm(X_real: np.ndarray, N_target: int,
                  seed: int = 42,
                  min_maf: float = 0.01) -> tuple[np.ndarray, dict]:
    """Build a large GRM by bootstrapping individuals from a real SNP matrix.

    Sampling with replacement from real haplotypes preserves LD structure and
    allele frequency spectrum — the resulting GRM has a realistic eigenspectrum
    unlike a random SPD matrix.

    Args:
        X_real:   (N_real, M) real SNP dosage matrix (int8 dosages 0/1/2)
        N_target: target population size (may be >> N_real)
        seed:     RNG seed
        min_maf:  MAF filter passed to compute_grm (default 0.01)

    Returns:
        G:    (N_target, N_target) GRM
        info: dict with metadata including 'X_filtered' (float32, MAF-filtered
              dosage matrix) and 'p', 'scale' for use with gblup_solve_ooc_z
    """
    rng = np.random.default_rng(seed)
    N_real, M = X_real.shape
    idx = rng.integers(0, N_real, size=N_target)
    X_boot = X_real[idx].astype(np.float64)
    G, info = compute_grm(X_boot, method="vanraden1", min_maf=min_maf)
    info["bootstrap_N_real"] = N_real
    info["bootstrap_N_target"] = N_target
    return G, info


def simulate_phenotype(G: np.ndarray, h2: float = 0.4,
                       seed: int = 0) -> np.ndarray:
    """Simulate a quantitative phenotype under an additive genetic model.

    Model:  y = g + e
      g ~ MVN(0, h² * G)         genomic breeding values
      e ~ MVN(0, (1-h²) * I)     environmental noise

    Args:
        G:    (N, N) GRM
        h2:   narrow-sense heritability
        seed: RNG seed

    Returns:
        y: (N,) simulated phenotype, mean-centred and unit-variance
    """
    rng = np.random.default_rng(seed)
    N = G.shape[0]

    # Genomic values via Cholesky of h² * G (add small nugget for stability)
    G_stable = h2 * G + 1e-6 * np.eye(N)
    try:
        L = np.linalg.cholesky(G_stable)
        g = L @ rng.standard_normal(N)
    except np.linalg.LinAlgError:
        # Fall back to eigendecomposition
        w, v = np.linalg.eigh(G_stable)
        w = np.maximum(w, 0.0)
        g = v @ (np.sqrt(w) * rng.standard_normal(N))

    e = np.sqrt(1.0 - h2) * rng.standard_normal(N)
    y = g + e
    y = (y - y.mean()) / y.std()
    return y.astype(np.float64)


# ---------------------------------------------------------------------------
# 4. Cross-validation utilities
# ---------------------------------------------------------------------------

def kfold_indices(N: int, k: int = 5, seed: int = 42) -> list[tuple]:
    """Return list of (train_idx, val_idx) tuples for k-fold CV."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train, val))
    return splits


def cv_accuracy(G: np.ndarray, y: np.ndarray, lam: float,
                k: int = 5, backend: str = "mpdok") -> dict:
    """k-fold cross-validated prediction accuracy for GBLUP.

    Returns dict with:
        r2:       mean Pearson r² across folds
        rmse:     root mean squared error
        per_fold: list of per-fold (r, rmse)
    """
    from .gblup import gblup_predict
    splits = kfold_indices(len(y), k=k)
    rs, rmses = [], []
    for train, val in splits:
        G_tt = G[np.ix_(train, train)]
        G_vt = G[np.ix_(val, train)]
        y_t  = y[train]
        y_v  = y[val]
        y_hat = gblup_predict(G_tt, y_t, G_vt, lam=lam, backend=backend)
        corr = np.corrcoef(y_v, y_hat)[0, 1]
        rs.append(corr)
        rmses.append(np.sqrt(np.mean((y_v - y_hat) ** 2)))
    return {"r2": np.mean(rs) ** 2, "r": np.mean(rs),
            "rmse": np.mean(rmses), "per_fold": list(zip(rs, rmses))}
