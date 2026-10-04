# -*- coding: utf-8 -*-
"""Run the five adversarial panel-level controls and deposit their output.

It was noted that the five `check_*.py` scripts in this directory had no deposited
output and were never reported; E80/E81 asked specifically for the composition and spatial controls
to be run and reported. This runner executes all five against the current `verify_table.parquet`,
writes the raw stdout to `reports/verify_controls.txt`, and parses the headline numbers into
`reports/verify_controls.json` so the manuscript can cite them.

Note on scope: `check_leakage/gain/permutation` were written for an exploratory
evo2+phylop *ensemble* (out-of-fold AUROC ~0.969) that is NOT a result in the manuscript — the paper
reports Evo 2-40B alone. Their value here is as PANEL-level controls: the composition-matching and
spatial-proximity checks (E80/E81) bear directly on the atlas panel the paper does use, since the
panel is shared. The ensemble numbers are reported as provenance, not as a manuscript claim.

    python src/ccs/verify/run_controls.py
"""
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
PY = sys.executable
CHECKS = ["check_composition", "check_leakage", "check_gain", "check_spatial", "check_permutation"]


def run(name):
    p = subprocess.run([PY, os.path.join(HERE, name + ".py")],
                       capture_output=True, text=True, cwd=ROOT)
    return (p.stdout or "") + (p.stderr or "")


def num(pattern, text, default=None, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def main():
    outputs = {name: run(name) for name in CHECKS}
    raw = "\n".join(f"===== {n} =====\n{outputs[n]}" for n in CHECKS)
    io.open(os.path.join(ROOT, "reports/verify_controls.txt"), "w", encoding="utf-8",
            newline="\n").write(raw)

    comp, spat, perm = outputs["check_composition"], outputs["check_spatial"], outputs["check_permutation"]
    gain, leak = outputs["check_gain"], outputs["check_leakage"]
    j = {
        "_scope": "panel-level adversarial controls on data/interim/verify_table.parquet; "
                  "leakage/gain/permutation concern an exploratory evo2+phylop ensemble not reported "
                  "in the manuscript (see docstring); composition + spatial bear on the shared panel.",
        "composition": {
            "composition_only_auroc": num(r"composition-ONLY.*?AUROC:\s*([\d.]+)", comp),
            "trinuc_chi2_p": num(r"chi2 p=([\d.]+)", comp),
            "gc_mwu_p": num(r"GC MWU p=([\d.]+)", comp),
            "gc_mean_pos": num(r"GC mean pos=([\d.]+)", comp),
            "gc_mean_neg": num(r"neg=([\d.]+)", comp),
            "criterion": "composition-only (trinuc-context + local GC) AUROC should be ~0.5 if matched",
        },
        "spatial": {
            "neg_within_1kb": num(r"within 1kb.*?:\s*(\d+)", spat, cast=int),
            "neg_within_10kb": num(r"within 10kb:\s*(\d+)", spat, cast=int),
            "neg_within_50kb": num(r"within 50kb:\s*(\d+)", spat, cast=int),
            "auroc_full": num(r"AUROC full ensemble:\s*([\d.]+)", spat),
            "auroc_drop_50kb": num(r"dropping <50kb.*?:\s*([\d.]+)", spat),
        },
        "permutation": {
            "real_auroc": num(r"real ensemble out-of-fold AUROC:\s*([\d.]+)", perm),
            "null_mean": num(r"null: mean=([\d.]+)", perm),
            "p_value": num(r"permutation p-value:\s*([\d.]+)", perm),
        },
        "ensemble_provenance": {
            "evo2_auroc": num(r"evo2=([\d.]+)", gain),
            "phylop_auroc": num(r"phylop=([\d.]+)", gain),
            "ensemble_auroc": num(r"ensemble=([\d.]+)", gain),
            "leakage_gap_insample_minus_oof": num(r"gap \(in-sample - oof\):\s*([\d.]+)", leak),
            "note": "ensemble is exploratory and NOT a manuscript result",
        },
    }
    io.open(os.path.join(ROOT, "reports/verify_controls.json"), "w", encoding="utf-8",
            newline="\n").write(json.dumps(j, indent=2))
    print("wrote reports/verify_controls.txt and reports/verify_controls.json")
    print(json.dumps(j, indent=1))


if __name__ == "__main__":
    main()
