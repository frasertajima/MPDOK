"""
Download and prepare all datasets for the GBLUP lab.

Run once:  conda run -n py314 python prepare_data.py

Datasets:
  1. BGLR wheat  — N=599,  M=1,279 SNPs, 4 grain-yield traits
  2. BGLR mice   — N=1814, M=10,346 SNPs, obesity BMI trait
  3. G2F maize   — N=2193, M=437,214 SNPs (subsampled), sequenced inbred lines
                   VCF must be in data/ before running (large file, ~3.6 GB)

Outputs (all in data/):
  wheat.npz   — X, Y, A (pre-built GRM), sets, trait_names
  mice.npz    — X, y_bmi, A (pre-built GRM), chr, mbp
  g2f.npz     — X (subsampled SNPs), samples, snp_ids
"""

import os, sys, urllib.request, io, time
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

MPDOK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MPDOK_ROOT not in sys.path:
    sys.path.insert(0, MPDOK_ROOT)

from gblup.grm import parse_vcf_genotypes, compute_grm


def _fetch_rdata(url: str) -> dict:
    import rdata
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    parsed = rdata.parser.parse_data(raw)
    return rdata.conversion.convert(parsed)


# ---------------------------------------------------------------------------
# 1. Wheat (BGLR)
# ---------------------------------------------------------------------------
def prepare_wheat(force: bool = False):
    out = os.path.join(DATA_DIR, "wheat.npz")
    if os.path.exists(out) and not force:
        print(f"wheat.npz exists, skipping.")
        return

    print("Downloading BGLR wheat dataset...")
    url = "https://raw.githubusercontent.com/gdlc/BGLR-R/master/data/wheat.RData"
    data = _fetch_rdata(url)

    X  = np.array(data["wheat.X"],    dtype=np.float32)   # (599, 1279)
    Y  = np.array(data["wheat.Y"],    dtype=np.float32)   # (599, 4)
    A  = np.array(data["wheat.A"],    dtype=np.float64)   # (599, 599) GRM
    sets = np.array(data["wheat.sets"], dtype=np.int8)    # (599,) CV folds

    np.savez_compressed(out, X=X, Y=Y, A=A, sets=sets,
                        trait_names=np.array(["E1", "E2", "E3", "E4"], dtype="U4"))
    print(f"Saved {out}  X={X.shape} Y={Y.shape} A={A.shape}")


# ---------------------------------------------------------------------------
# 2. Mice (BGLR)
# ---------------------------------------------------------------------------
def prepare_mice(force: bool = False):
    out = os.path.join(DATA_DIR, "mice.npz")
    if os.path.exists(out) and not force:
        print(f"mice.npz exists, skipping.")
        return

    print("Downloading BGLR mice dataset...")
    url = "https://raw.githubusercontent.com/gdlc/BGLR-R/master/data/mice.RData"
    data = _fetch_rdata(url)

    X   = np.array(data["mice.X"],          dtype=np.float32)  # (1814, 10346)
    A   = np.array(data["mice.A"],          dtype=np.float64)  # (1814, 1814) GRM
    df  = data["mice.pheno"]
    bmi = np.array(df["Obesity.BMI"],       dtype=np.float64)  # (1814,)
    blen= np.array(df["Obesity.BodyLength"],dtype=np.float64)  # (1814,)
    mp  = data["mice.map"]
    # Force unicode dtype so npz can be loaded without allow_pickle
    chr_raw = mp["chr"].values if hasattr(mp["chr"], "values") else list(mp["chr"])
    chrom = np.array([str(c) for c in chr_raw], dtype="U8")
    mbp   = np.array(mp["mbp"], dtype=np.float32)

    np.savez_compressed(out, X=X, A=A, y_bmi=bmi, y_blen=blen,
                        chrom=chrom, mbp=mbp)
    print(f"Saved {out}  X={X.shape} A={A.shape} BMI={bmi.shape}")


# ---------------------------------------------------------------------------
# 3. G2F maize VCF
# ---------------------------------------------------------------------------
def prepare_g2f(subsample_step: int = 9, max_snps: int | None = None,
                force: bool = False):
    out = os.path.join(DATA_DIR, "g2f.npz")
    vcf = os.path.join(DATA_DIR, "inbreds_G2F_2014-2023_437k.vcf")

    if os.path.exists(out) and not force:
        print(f"g2f.npz exists, skipping.")
        return
    if not os.path.exists(vcf):
        raise FileNotFoundError(
            f"G2F VCF not found at {vcf}\n"
            "Download from: https://datacommons.cyverse.org/browse/iplant/home/"
            "shared/commons_repo/curated/GenomesToFields_G2F_genotypic_data_2014_to_2023"
        )

    vcf_size = os.path.getsize(vcf) / 1e9
    print(f"Parsing G2F VCF ({vcf_size:.2f} GB), subsample_step={subsample_step} ...")
    t0 = time.time()
    X, samples, snp_ids = parse_vcf_genotypes(vcf, subsample_step=subsample_step,
                                               max_snps=max_snps, verbose=True)
    print(f"Parsed in {time.time()-t0:.0f}s.  Matrix: {X.shape}")

    # Convert string lists to byte arrays for npz storage
    samples_arr = np.array(samples, dtype="U64")
    snps_arr    = np.array(snp_ids,  dtype="U32")

    np.savez_compressed(out, X=X, samples=samples_arr, snp_ids=snps_arr)
    print(f"Saved {out}  X={X.shape}  N={len(samples)}  M={len(snp_ids)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--step",     type=int, default=9,
                   help="VCF subsample step (default 9 → ~50k SNPs)")
    p.add_argument("--max-snps", type=int, default=None)
    p.add_argument("--force",    action="store_true")
    args = p.parse_args()

    prepare_wheat(force=args.force)
    prepare_mice(force=args.force)
    prepare_g2f(subsample_step=args.step, max_snps=args.max_snps, force=args.force)
    print("\nAll datasets ready.")
