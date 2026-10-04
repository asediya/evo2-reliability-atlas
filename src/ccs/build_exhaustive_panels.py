"""Build the EXHAUSTIVE cloud panels beyond the 8192 core:
  - atlas window-sweep at 2048 and 4096 bp (9 species) -> readout/context robustness sweep
  - the FULL 20k eQTL panel at 8192 bp -> the definitive regulatory number at scale
Reuses the VALIDATED extraction from build_atlas8192 (correct per-species genome builds, 100% ref-match)
and build_ablation_eqtl (pig). CPU-only; run before renting so the GPU does pure scoring.
  python src/ccs/build_exhaustive_panels.py
"""
import sys
import polars as pl
from pathlib import Path
sys.path.insert(0, "src")
from ccs.build_atlas8192 import SPECIES, load_fai, fetch, parse_vid
from ccs.build_ablation_eqtl import extract as eqtl_extract   # extract(df[chrom,pos,ref,alt], W) on pig genome

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_atlas(W):
    HALF = W // 2
    out = Path(f"data/interim/atlas{W}"); out.mkdir(parents=True, exist_ok=True)
    gn = gok = 0
    # SPECIES values are 3-tuples (stem, genome, posfile) -- ccs.build_atlas8192, which defines it,
    # unpacks three. Unpacking two raises "ValueError: too many values to unpack (expected 2)" on the
    # first species.
    for sp, (stem, genome, _posfile) in SPECIES.items():
        d = pl.read_parquet(f"data/interim/{stem}.parquet").select(["variant_id", "label"])
        idx = load_fai(genome + ".fai")
        vid, rs, alts, vo, lab, ok = [], [], [], [], [], []
        with open(genome, "rb") as fh:
            for v, l in d.iter_rows():
                try:
                    c, p, rf, al = parse_vid(v)
                except Exception:
                    continue
                s = fetch(fh, idx, str(c), p - HALF, p + (W - HALF - 1))
                if s is None or len(s) != W:
                    continue
                vid.append(v); rs.append(s); alts.append(s[:HALF] + al.upper() + s[HALF + 1:])
                vo.append(HALF); lab.append(l); ok.append(s[HALF] == rf.upper())
        df = pl.DataFrame({"variant_id": vid, "ref_seq": rs, "alt_seq": alts,
                           "var_off": vo, "label": lab, "ref_ok": ok})
        df.write_parquet(out / f"{sp}_windows_{W}.parquet")
        nok = int(sum(ok)); gn += df.height; gok += nok
        print(f"  atlas{W} {sp:8} {df.height:5} | ref-match {100*nok/max(1,df.height):.1f}%", flush=True)
    print(f"  atlas{W} TOTAL {gok}/{gn} ({100*gok/max(1,gn):.1f}%)", flush=True)


def build_full_eqtl():
    cand = pl.read_parquet("data/interim/eqtl_candidates.parquet")
    df = eqtl_extract(cand, 8192)   # pig genome; adds ref_ok
    nok = int(df["ref_ok"].sum())
    Path("data/interim/ablation").mkdir(parents=True, exist_ok=True)
    df.write_parquet("data/interim/ablation/eqtl_full_8192.parquet")
    print(f"  eqtl_full_8192 {df.height}/{cand.height} | ref-match {100*nok/max(1,df.height):.1f}%", flush=True)


if __name__ == "__main__":
    # The two builds are independent and each is guarded on its own, so an atlas-sweep
    # failure does not take the eQTL build with it.
    rc = 0
    for label, fn in (("atlas window-sweep (2048, 4096)", lambda: (build_atlas(2048), build_atlas(4096))),
                      ("full 20k eQTL @ 8192", build_full_eqtl)):
        print("== %s ==" % label, flush=True)
        try:
            fn()
        except Exception as e:
            print("  FAILED: %s: %s" % (type(e).__name__, e), flush=True)
            rc = 1
    sys.exit(rc)
