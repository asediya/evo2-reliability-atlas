"""MASTER reliability-atlas table (development log, superseded): one row per species tying together every arm —
discrimination (Evo2-40B vs conservation vs 1B), fidelity (block-streaming vs hosted r), calibration
(ECE none -> transferred), what the FM adds over conservation (ΔFM, low-conservation AUROC), and the
deployable trust budget (abstention error 100% -> min). This IS the "calibrated cross-species reliability
atlas" the title promises. Merges the per-arm parquets; re-run after the atlas/analyses refresh.
  python src/ccs/build_reliability_atlas.py
"""
import os, sys, time
import polars as pl
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROC = "data/processed"
EV = "logs/status/events.log"; ST = "logs/status/reliability.status"; MD = "logs/reliability_atlas.md"
ORDER = ["human", "cattle", "dog", "cat", "horse", "sheep", "pig", "goat", "chicken"]


def ev(m):
    with open(EV, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] atlas: {m}\n")
def st(m):
    with open(ST, "w", encoding="utf-8") as f: f.write(m + "\n")


def g(row, k):
    v = row.get(k)
    return v if v is not None else None


def fmt(v, nd=3, pct=False, plus=False):
    if v is None: return "—"
    if pct: return f"{v:.0%}"
    return f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"


def main():
    st("RUNNING | assembling master reliability atlas")
    atlas = pl.read_parquet(f"{PROC}/atlas_40b.parquet")
    cal = pl.read_parquet(f"{PROC}/calibration_transfer.parquet") if os.path.exists(f"{PROC}/calibration_transfer.parquet") else None
    dec = pl.read_parquet(f"{PROC}/decomposition.parquet") if os.path.exists(f"{PROC}/decomposition.parquet") else None
    ab = pl.read_parquet(f"{PROC}/abstention_trust_budget.parquet") if os.path.exists(f"{PROC}/abstention_trust_budget.parquet") else None

    A = {r["species"]: r for r in atlas.to_dicts()}
    C = {r["species"]: r for r in cal.to_dicts()} if cal is not None else {}
    D = {r["sp"]: r for r in dec.to_dicts()} if dec is not None else {}
    B = {r["sp"]: r for r in ab.to_dicts()} if ab is not None else {}

    order = [s for s in ORDER if s in A] + [s for s in A if s not in ORDER]
    rows = []
    for sp in order:
        a = A[sp]; c = C.get(sp, {}); d = D.get(sp, {}); b = B.get(sp, {})
        rows.append(dict(
            species=sp, clade=c.get("clade", "—"), n=a["n"], pos=a["n_pos"],
            auroc_40b=g(a, "auroc_evo2_40b"), d_cons=g(a, "delta_40b_vs_cons"),
            d_1b=g(a, "delta_40b_vs_1b"), xcheck_r=g(a, "xcheck_local_vs_api_r"),
            ece_none=g(c, "ece_none"), ece_transfer=g(c, "ece_transfer"),
            d_fm=g(d, "d_fm"), lowcons=g(d, "a_f_lo"),
            trust_e100=g(b, "e100"), trust_emin=g(b, "emin"), trust_cov=g(b, "best_cov")))

    hdr = ["species", "clade", "N", "pos", "AUROC 40B", "Δvs cons", "Δvs 1B", "streaming≈hosted r",
           "ECE none→transfer", "ΔFM", "Evo2 @low-cons", "trust err 100%→min@cov"]
    lines = ["# Calibrated cross-species reliability atlas — master table "
             "(development log; superseded by the submitted Table 2)", "",
             "One row per species. **AUROC 40B** = Evo2-40B zero-shot; **Δvs cons/1B** = paired gain over best "
             "conservation / Evo2-1B; **streaming≈hosted r** = block-streaming vs hosted 40B; **ECE** = calibration "
             "(no-cal → transferred); **ΔFM** = signal the FM adds beyond conservation; **@low-cons** = Evo2 AUROC "
             "at below-median-conservation sites; **trust** = abstention selective error (100% coverage → min).", "",
             "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["species"], r["clade"], str(r["n"]), str(r["pos"]),
            fmt(r["auroc_40b"]), fmt(r["d_cons"], plus=True), fmt(r["d_1b"], plus=True),
            fmt(r["xcheck_r"], nd=4),
            f'{fmt(r["ece_none"])}→{fmt(r["ece_transfer"])}',
            fmt(r["d_fm"], plus=True), fmt(r["lowcons"]),
            f'{fmt(r["trust_e100"])}→{fmt(r["trust_emin"])}@{fmt(r["trust_cov"], pct=True)}',
        ]) + " |")

    done40 = [r for r in rows if r["auroc_40b"] is not None]
    lines += ["",
              f"Species with 40B scored: {len(done40)}/{len(rows)}. "
              f"Mean AUROC-40B = {sum(r['auroc_40b'] for r in done40)/len(done40):.3f}." if done40 else ""]
    open(MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    pl.DataFrame(rows).write_parquet(f"{PROC}/reliability_atlas.parquet")
    print("\n".join(lines))
    st(f"DONE | master reliability atlas: {len(done40)}/{len(rows)} species with 40B; {MD}")
    ev(f"DONE: master reliability-atlas table assembled ({len(done40)}/{len(rows)} species with 40B) -> {MD}")


if __name__ == "__main__":
    main()
