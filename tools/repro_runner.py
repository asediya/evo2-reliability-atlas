"""From-scratch reproduction runner.

Re-runs every analysis script that produces a number in the paper, fresh, and
records exit status, duration and output. Sequential by design: these scripts
share parquet outputs under data/processed, so parallel execution would race.

The point is NOT to confirm what we remember. It is to regenerate every number
from the cached Evo2/NT scores + raw panels and then diff the regenerated
artifacts against the sealed baseline in reports/_repro_baseline/.

Usage:
    python tools/repro_runner.py            # run all pending
    python tools/repro_runner.py --only build_atlas
    python tools/repro_runner.py --reset    # forget prior state

State lives in reports/repro_state.json so the run is resumable across restarts.
Per-script output goes to reports/repro_logs/<script>.log.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _interpreter():
    """The interpreter to run each analysis script with.

    ROOT/.venv is the normal case, but it does not exist inside a git worktree (the venv lives
    in the main checkout), and it will not exist for a referee who installed the deposited code
    into their own environment. In both cases the harness died at startup with "interpreter not
    found" rather than running anything. Fall back to whatever is running this file, and let
    CCS_PYTHON override for a deliberately different environment.
    """
    env = os.environ.get("CCS_PYTHON")
    if env and os.path.exists(env):
        return env
    venv = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv):
        return venv
    venv_posix = os.path.join(ROOT, ".venv", "bin", "python")
    if os.path.exists(venv_posix):
        return venv_posix
    return sys.executable


PY = _interpreter()
LOGS = os.path.join(ROOT, "reports", "repro_logs")
STATE = os.path.join(ROOT, "reports", "repro_state.json")

# Dependency-ordered. Annotations and panels first, then the atlas they feed,
# then the trust layer, then the independent arms, then reconciliation/tables.
ORDER = [
    "annotate_consequence",
    "build_atlas",
    "build_atlas8192",
    "build_conservation_matched",
    "build_reliability_atlas",
    "calibration_transfer",
    "build_calibration_robustness",
    "build_calibration_rigor",
    "build_calibration_methods",
    "build_prevalence_correction",
    "build_conformal",
    "build_conformal_robustness",
    "build_abstention",
    "build_decision_panel",
    "build_beats_conservation_ci",
    "build_decomposition",
    "build_selection_analysis",
    "build_bat_analysis",
    "build_eqtl_analysis",
    "build_eqtl_calibration",
    "build_eqtl_abstention",
    "build_clinvar_calibration",
    "build_clinvar_ci",
    "build_nt_backbone",
    "build_nt_conformal",
    "build_nt_prevalence_fix",
    "build_hardening_cis",
    "build_downsampling_defense",
    "build_crc_clinvar",
    "build_crc_certificate",
    "eval_brca1_delta",
    "fig6_stats",
    "fig4_reconcile",
    "build_tost_equivalence",
    "build_readout_headtohead",
    "build_phylop_crosscheck",
    # The figure-data / statistics layer. fig2_data, fig3_stats, fig4_*_stats and fig5_stats all
    # produce numbers that are printed on a canvas, so they are re-run here too.
    # An audit that skips the layer the reader actually looks at is not an audit.
    "compile_results",
    "type_matched_atlas",
    "score_gerp_baseline",
    "rebuild_trust_layer_8192",
    "analyze_trust_layer",
    "annotate_brca1_residues",
    "compute_auprc",
    "build_emin",
    "build_fig3_consequence",
    "build_locus_clustered_ci",
    "build_readout_effect_fullpanel",
    "build_clinvar_reviewstatus",
    "playbook_checks",
    "fig2b_data",
    "fig2_data",
    "fig3_data",
    "fig3_stats",
    "fig4_asym_stats",
    "fig4_field_stats",
    "fig4_matrix_stats",
    "fig5_stats",
    # tables and the supplement consume everything above, so they stay last.
    "build_tables",
    "build_supplementary",
]

TIMEOUT = 2400  # 40 min per script; nothing here should approach it


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(io.open(STATE, encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(st):
    io.open(STATE, "w", encoding="utf-8", newline="\n").write(
        json.dumps(st, indent=2, sort_keys=True) + "\n"
    )


def run_one(name, state):
    script = os.path.join(ROOT, "src", "ccs", name + ".py")
    if not os.path.exists(script):
        state[name] = {"status": "MISSING", "detail": script}
        return state[name]

    log_path = os.path.join(LOGS, name + ".log")
    t0 = time.time()
    try:
        p = subprocess.run(
            [PY, script],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT,
        )
        out = p.stdout.decode("utf-8", errors="replace")
        rc = p.returncode
        status = "OK" if rc == 0 else "FAIL"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", errors="replace")
        out += "\n\n*** TIMEOUT after %ds ***\n" % TIMEOUT
        rc, status = -9, "TIMEOUT"

    dt = time.time() - t0
    io.open(log_path, "w", encoding="utf-8", newline="\n").write(out)

    # keep the tail in state so a summary does not require opening every log
    tail = "\n".join(out.strip().split("\n")[-25:])
    state[name] = {
        "status": status,
        "returncode": rc,
        "seconds": round(dt, 1),
        "log": os.path.relpath(log_path, ROOT).replace("\\", "/"),
        "tail": tail,
    }
    return state[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run just these scripts")
    ap.add_argument("--reset", action="store_true", help="discard prior state")
    ap.add_argument("--rerun-failed", action="store_true", help="retry non-OK entries")
    args = ap.parse_args()

    if not os.path.exists(PY):
        print("FATAL: interpreter not found at %s" % PY)
        return 2
    os.makedirs(LOGS, exist_ok=True)

    state = {} if args.reset else load_state()
    targets = args.only if args.only else ORDER

    for i, name in enumerate(targets, 1):
        prev = state.get(name, {}).get("status")
        if prev == "OK" and not args.only:
            print("[%2d/%d] SKIP %s (already OK)" % (i, len(targets), name), flush=True)
            continue
        if prev and prev != "OK" and not (args.rerun_failed or args.only):
            print("[%2d/%d] SKIP %s (prior %s; use --rerun-failed)"
                  % (i, len(targets), name, prev), flush=True)
            continue

        print("[%2d/%d] RUN  %s ..." % (i, len(targets), name), end=" ", flush=True)
        res = run_one(name, state)
        save_state(state)
        print("%s (%.1fs)" % (res["status"], res.get("seconds", 0)), flush=True)

    ok = sum(1 for v in state.values() if v.get("status") == "OK")
    bad = [k for k, v in state.items() if v.get("status") != "OK"]
    print("\n=== reproduction summary ===")
    print("OK: %d / %d" % (ok, len(state)))
    if bad:
        print("NOT OK: %s" % ", ".join(sorted(bad)))
        # Non-zero, so a harness whose whole purpose is to detect a script that
        # did not reproduce never reports success while naming the scripts that failed.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
