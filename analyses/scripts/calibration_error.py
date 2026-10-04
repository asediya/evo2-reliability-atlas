# -*- coding: utf-8 -*-
"""C4 — the ECE binning flip is a documented property, not an embarrassment. Quantify it.

The supplement reports that conclusions move between equal-width and equal-mass binning and
between 10 and 15 bins, and treats that as an awkwardness to disclose. It is published behaviour:

  Roelofs et al. 2022 (arXiv:2012.08668)   bias as a function of bin count and scheme; equal-mass
                                           is less biased than equal-width
  Kumar, Liang & Ma 2019 (arXiv:1909.10155) the plug-in binned estimator is biased DOWNWARD, with
                                           a debiased estimator
  Vaicenavicius et al. 2019 (arXiv:1902.06977) different binnings estimate genuinely DIFFERENT
                                           estimands, so a flip is an identification issue rather
                                           than imprecision

This computes, for every calibrator arm: L1 ECE under six binning schemes with bootstrap
intervals, and the debiased L2 calibration error with its interval. The debiased plug-in for the
squared error subtracts the per-bin sampling variance, which is what the downward bias is:

    CE2_debiased = sum_b (n_b/n) [ (mean_p_b - mean_y_b)^2 - mean_y_b(1 - mean_y_b)/(n_b - 1) ]

    python analyses/scripts/calibration_error.py
"""
import io
import json
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "analyses/results"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260804)

ARMS = [("isotonic_LOSO", "isotonic transfer, leave-one-species-out"),
        ("platt_LOSO", "Platt transfer, leave-one-species-out"),
        # The label states the part that matters: this sigmoid is fitted on ALL
        # SPECIES POOLED, so it is scored in-sample and is NOT comparable with the LOSO arms
        # above. The TRAP note further down this file explains the hazard.
        ("global_sigmoid", "a two-parameter sigmoid fitted once on ALL species pooled -- "
                           "in-sample, not comparable with the LOSO arms"),
        ("oracle_isotonic", "in-species oracle (unattainable bound)")]


def edges(p, nbins, scheme):
    if scheme == "width":
        return np.linspace(0.0, 1.0, nbins + 1)
    q = np.quantile(p, np.linspace(0, 1, nbins + 1))
    q[0], q[-1] = 0.0, 1.0
    return np.unique(q)


def ece_l1(p, y, nbins, scheme):
    e = edges(p, nbins, scheme)
    idx = np.clip(np.digitize(p, e[1:-1], right=True), 0, len(e) - 2)
    tot = 0.0
    for b in range(len(e) - 1):
        m = idx == b
        if not m.any():
            continue
        tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def ce2_debiased(p, y, nbins, scheme):
    """Debiased squared calibration error. Bins with a single member contribute no correction and
    are dropped from it, which is stated rather than silently handled."""
    e = edges(p, nbins, scheme)
    idx = np.clip(np.digitize(p, e[1:-1], right=True), 0, len(e) - 2)
    tot, dropped = 0.0, 0
    n = len(p)
    for b in range(len(e) - 1):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        yb, pb = y[m].mean(), p[m].mean()
        if nb < 2:
            dropped += 1
            tot += (nb / n) * (pb - yb) ** 2
            continue
        tot += (nb / n) * ((pb - yb) ** 2 - yb * (1 - yb) / (nb - 1))
    return float(tot), dropped


def boot(fn, p, y, B=1000):
    n = len(p)
    v = []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        try:
            v.append(fn(p[i], y[i]))
        except Exception:
            pass
    v.sort()
    return [float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v))])] if v else [None, None]


def main():
    d = pl.read_parquet("reports/_recon_pervariant_trust.parquet")
    species = d["species"].unique().to_list()
    out = {"_generated_by": "analyses/scripts/calibration_error.py",
           "_citations": ["arXiv:2012.08668", "arXiv:1909.10155", "arXiv:1902.06977",
                          "arXiv:1904.01685"],
           "_why": "The supplement discloses that the ECE conclusion moves with the binning rule. "
                   "That is published behaviour with a published fix. Reporting the spread as an "
                   "identification issue, plus a debiased estimator with an interval, turns the "
                   "most awkward note in the supplement into a contribution.",
           "_estimands": "L1 ECE under six binning schemes, macro over species. Debiased L2 "
                         "calibration error (Kumar 2019), macro over species. The two are "
                         "different quantities and are not compared to each other.",
           "arms": {}}

    for col, why in ARMS:
        if col not in d.columns:
            continue
        row = {"_what": why, "l1_ece": {}, "debiased_l2": {}}
        for scheme in ("width", "mass"):
            for nb in (10, 15, 20):
                vals, dbg = [], []
                for sp in species:
                    s = d.filter(pl.col("species") == sp)
                    p = s[col].to_numpy().astype(float)
                    y = s["label"].to_numpy().astype(float)
                    if len(p) < 40:
                        continue
                    vals.append(ece_l1(p, y, nb, scheme))
                    dbg.append(ce2_debiased(p, y, nb, scheme)[0])
                key = "%s_%d" % (scheme, nb)
                row["l1_ece"][key] = float(np.mean(vals))
                row["debiased_l2"][key] = float(np.mean(dbg))
        v = list(row["l1_ece"].values())
        row["l1_spread_across_schemes"] = float(max(v) - min(v))
        row["l1_min"], row["l1_max"] = float(min(v)), float(max(v))
        # one pooled interval, on the scheme Roelofs recommends
        p = d[col].to_numpy().astype(float)
        y = d["label"].to_numpy().astype(float)
        row["pooled_l1_ece_mass15"] = ece_l1(p, y, 15, "mass")
        row["pooled_l1_ece_mass15_ci95"] = boot(lambda a, b: ece_l1(a, b, 15, "mass"), p, y)
        c2, dropped = ce2_debiased(p, y, 15, "mass")
        row["pooled_debiased_l2"] = c2
        row["pooled_debiased_l2_ci95"] = boot(lambda a, b: ce2_debiased(a, b, 15, "mass")[0], p, y)
        row["pooled_debiased_l2_negative"] = bool(c2 < 0)
        row["singleton_bins_dropped_from_correction"] = dropped
        out["arms"][col] = row

    # ---- the claim the manuscript actually makes, tested against the right object --------------
    # TRAP: the per-variant table's
    # `global_sigmoid` column is a sigmoid fitted ONCE ON ALL SPECIES POOLED. The manuscript's
    # "Platt transfer IS a trivial sigmoid" claim is about `trivial_sigmoid_LOSO`, a sigmoid
    # fitted under the SAME leave-one-species-out protocol as the Platt transfer. Those two are
    # identical by construction; the pooled-fit column is a different estimator and differs from
    # the Platt arm on every one of the 11,130 variants. Comparing the wrong pair produces an
    # apparent contradiction of a load-bearing claim.
    grid = json.load(io.open("reports/fig4_reconciliation.json", encoding="utf-8"))["ece_grid"]
    same, checked = True, []
    for scheme, cell in grid.items():
        if "platt_LOSO" in cell and "trivial_sigmoid_LOSO" in cell:
            a_, b_ = cell["platt_LOSO"], cell["trivial_sigmoid_LOSO"]
            eq = (a_["macro"] == b_["macro"] and a_["pooled"] == b_["pooled"]
                  and all(a_["per_species"][s] == b_["per_species"][s] for s in a_["per_species"]))
            checked.append(scheme)
            same = same and eq
    out["platt_transfer_is_the_trivial_sigmoid"] = {
        "identical_in_every_deposited_binning_scheme": bool(same),
        "schemes_checked": checked,
        "_what_was_compared": "platt_LOSO against trivial_sigmoid_LOSO in reports/"
                              "fig4_reconciliation.json ece_grid, macro, pooled and per species",
        "_naming_hazard": "The per-variant table's `global_sigmoid` column is a POOLED fit, not "
                          "the LOSO trivial sigmoid, and differs from the Platt arm on all "
                          "11,130 variants (max |diff| 0.099). The two must not be conflated.",
    }
    pv_a = d["platt_LOSO"].to_numpy().astype(float)
    pv_b = d["global_sigmoid"].to_numpy().astype(float)
    out["pooled_fit_sigmoid_is_a_different_estimator"] = {
        "max_abs_difference_from_platt_LOSO": float(np.max(np.abs(pv_a - pv_b))),
        "n_variants_differing": int((pv_a != pv_b).sum()),
        "n_variants": int(len(pv_a)),
        "_note": "Reported so nobody repeats the mistake of reading this column as the trivial "
                 "sigmoid the manuscript's null refers to."}

    p = os.path.join(OUT, "calibration_error.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    print("  macro L1 ECE by binning scheme (lower is better)")
    print("  %-22s %8s %8s %8s %8s %8s %8s | spread" %
          ("arm", "w10", "w15", "w20", "m10", "m15", "m20"))
    for col, _ in ARMS:
        if col not in out["arms"]:
            continue
        r = out["arms"][col]["l1_ece"]
        print("  %-22s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f | %.4f"
              % (col, r["width_10"], r["width_15"], r["width_20"],
                 r["mass_10"], r["mass_15"], r["mass_20"],
                 out["arms"][col]["l1_spread_across_schemes"]))
    print()
    for col, _ in ARMS:
        if col not in out["arms"]:
            continue
        r = out["arms"][col]
        print("  %-22s pooled L1 %.4f %s   debiased L2 %.5f %s"
              % (col, r["pooled_l1_ece_mass15"],
                 "[%.4f, %.4f]" % tuple(r["pooled_l1_ece_mass15_ci95"]),
                 r["pooled_debiased_l2"],
                 "[%.5f, %.5f]" % tuple(r["pooled_debiased_l2_ci95"])))
    t = out["platt_transfer_is_the_trivial_sigmoid"]
    print()
    print("  Platt transfer IS the LOSO trivial sigmoid, in all %d deposited binning schemes: %s"
          % (len(t["schemes_checked"]), t["identical_in_every_deposited_binning_scheme"]))
    u = out["pooled_fit_sigmoid_is_a_different_estimator"]
    print("  (the POOLED-fit sigmoid is a different estimator: differs on %s of %s variants, "
          "max |diff| %.3f)" % ("{:,}".format(u["n_variants_differing"]),
                                "{:,}".format(u["n_variants"]),
                                u["max_abs_difference_from_platt_LOSO"]))
    print()
    print("  wrote %s" % p)


if __name__ == "__main__":
    main()
