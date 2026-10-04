"""Build the dog OMIA positives panel on ROS_Cfam_1.0.
Dog OMIA is build-heterogeneous: 227/282 on CanFam3.1, 3 native ROS_Cfam_1.0 (+55 on other
builds, skipped for now). We liftOver the CanFam3.1 majority CanFam3.1(canFam3) -> ROS_Cfam(canFam6)
with pyliftover, reverse-complementing ref/alt on strand flips, and add the ROS_Cfam natives.
ref-allele verification against the genome happens downstream (extract_windows_local).

  python dog_liftover_panel.py --chain data/raw/liftover/canFam3ToCanFam6.over.chain \
      --out data/interim/dog_omia_pos.parquet
"""
import argparse, sys
import polars as pl
from pyliftover import LiftOver

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC = "data/interim/omia_multispecies_positives.parquet"
COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = pl.read_parquet(SRC).filter(pl.col("species") == "dog")
    lo = LiftOver(a.chain)
    rows, n_lifted, n_fail = [], 0, 0

    # 1) CanFam3.1 -> ROS_Cfam_1.0 via liftOver
    for r in src.filter(pl.col("native_assembly") == "CanFam3.1").iter_rows(named=True):
        c = "chr" + str(r["chrom_label"])
        res = lo.convert_coordinate(c, int(r["pos"]) - 1)   # pyliftover is 0-based
        if not res:
            n_fail += 1
            continue
        nc, npos, strand, _ = res[0]
        chrom = nc.replace("chr", "")
        ref, alt = r["ref"].upper(), r["alt"].upper()
        if strand == "-":
            ref, alt = COMP.get(ref, ref), COMP.get(alt, alt)
        rows.append({"chrom": chrom, "pos": npos + 1, "ref": ref, "alt": alt,
                     "gene": r["gene"], "phenotype": r["phenotype"],
                     "omia_variant_id": r["omia_variant_id"], "src_build": "CanFam3.1->lift"})
        n_lifted += 1

    # 2) ROS_Cfam_1.0 natives (direct)
    for r in src.filter(pl.col("native_assembly") == "ROS_Cfam_1.0").iter_rows(named=True):
        rows.append({"chrom": str(r["chrom_label"]), "pos": int(r["pos"]),
                     "ref": r["ref"].upper(), "alt": r["alt"].upper(),
                     "gene": r["gene"], "phenotype": r["phenotype"],
                     "omia_variant_id": r["omia_variant_id"], "src_build": "ROS_Cfam_native"})

    out = pl.DataFrame(rows).with_columns([
        (pl.col("chrom") + "_" + pl.col("pos").cast(pl.Utf8) + "_" + pl.col("ref") + "_" + pl.col("alt")).alias("variant_id"),
        pl.lit(1).alias("label"),
    ]).unique(subset=["variant_id"])
    out.write_parquet(a.out)
    print(f"dog positives: lifted {n_lifted} (failed {n_fail}) + natives -> {out.height} unique on ROS_Cfam_1.0")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
