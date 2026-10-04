#!/usr/bin/env python3
"""Capture the GPU scoring environment from a live scoring host, as one JSON record.

    python tools/capture_scoring_environment.py --out reports/scoring_environment.json \
        --checkpoint-dir /path/to/evo2/checkpoints

Why this exists
---------------
Every published Evo 2 score in this paper was produced on a cloud GPU host, not on the machine the
analysis layer runs on, inside the container whose recipe is deposited under `container/`; its
README gives the base-image digest and the versions of evo2, vortex, PyTorch, TransformerEngine,
CUDA and cuDNN. `requirements.txt` pins the ANALYSIS layer only and says so; `evo2`, `vortex` and
TransformerEngine live solely on the scoring host.

Run this script on a scoring host, with the same interpreter and environment that performs
scoring, and it records — from the live environment, not from anyone's memory:

  * interpreter, OS, CPU and hostname
  * `pip freeze` in full, plus an explicit version probe for torch, transformer_engine, evo2,
    vortex, flash_attn, transformers, einops and huggingface_hub
  * CUDA runtime and driver versions, NVIDIA driver, and every visible GPU with its memory
  * torch's compiled CUDA/cuDNN versions and the matmul/TF32 precision flags actually in force
  * SHA-256, size and mtime of every checkpoint file under --checkpoint-dir (or the HF cache),
    so a checkpoint can be identified by content rather than by name
  * the deterministic scoring policy as declared on the command line (--batch-size, --precision,
    --window), recorded alongside everything else so the policy and the environment cannot drift

It writes one JSON file, which a reader who rebuilds the container can compare with
`container/README.md` to confirm that the rebuilt environment matches the one that scored.

What it does NOT do
-------------------
It does not rescore anything and it does not invent a value. Any field it could not determine is
written as null with a `_unavailable` note saying why, so a reader can tell a missing probe from a
missing package. Fields you must still supply by hand are listed under `manual_todo` in the output.

Fixture mode
------------
    python tools/capture_scoring_environment.py --emit-fixture-template reports/scoring_fixture.json

writes a template listing a small set of variants whose scores a referee could recompute
end-to-end. Fill in the expected scores on the scoring host; `--check-fixture` then compares a
fresh scoring run against it. The template is deliberately empty of numbers: this script never
fabricates an expected value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

PROBE = ["torch", "transformer_engine", "evo2", "vortex", "flash_attn", "transformers",
         "einops", "huggingface_hub", "numpy", "safetensors"]


def _run(cmd):
    """Run a command, returning stripped stdout or None. Never raises."""
    try:
        exe = shutil.which(cmd[0])
        if exe is None:
            return None
        out = subprocess.run([exe] + cmd[1:], capture_output=True, text=True, timeout=120)
        return out.stdout.strip() or None
    except Exception:
        return None


def _probe(cmd, why):
    """Run an external probe for the record: its output, or null WITH a note saying why.

    A bare null cannot distinguish "the tool is not on PATH" from "the tool ran and said
    nothing", and this script's contract is that every undetermined field carries a
    `_unavailable` note. nvcc in particular is absent on hosts that install only the CUDA
    runtime, which is a fact about the host worth recording rather than losing.
    """
    if shutil.which(cmd[0]) is None:
        return {"output": None, "_unavailable": f"{cmd[0]} not found on PATH — {why}"}
    out = _run(cmd)
    if out is None:
        return {"output": None,
                "_unavailable": f"{cmd[0]} is on PATH but produced no output — {why}"}
    return {"output": out}


def _version(mod):
    try:
        m = __import__(mod)
    except Exception as e:
        return {"installed": False, "version": None, "_unavailable": f"{type(e).__name__}: {e}"}
    v = getattr(m, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version as _v
            v = _v(mod)
        except Exception:
            v = None
    return {"installed": True, "version": v, "file": getattr(m, "__file__", None),
            **({"_unavailable": "module has no __version__ and no distribution metadata"}
               if v is None else {})}


def _torch_block():
    try:
        import torch
    except Exception as e:
        return {"_unavailable": f"torch not importable: {type(e).__name__}: {e}"}
    b = {"version": torch.__version__,
         "compiled_cuda": getattr(torch.version, "cuda", None),
         "compiled_cudnn": (torch.backends.cudnn.version()
                            if torch.backends.cudnn.is_available() else None),
         "cuda_available": bool(torch.cuda.is_available()),
         "float32_matmul_precision": None,
         "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
         "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
         "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
         "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
         "devices": []}
    try:
        b["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:
        pass
    if b["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            b["devices"].append({"index": i, "name": p.name,
                                 "capability": f"{p.major}.{p.minor}",
                                 "total_memory_bytes": int(p.total_memory),
                                 "multi_processor_count": int(p.multi_processor_count)})
    else:
        b["_unavailable"] = ("torch.cuda.is_available() is False — run this on the scoring host, "
                             "with the GPU visible, or the record will not describe the machine "
                             "that produced the scores")
    return b


def _hash_checkpoints(root, limit_bytes):
    """SHA-256 every plausible checkpoint file under root."""
    if not root:
        return {"_unavailable": "no --checkpoint-dir given and no HF cache found; "
                               "pass --checkpoint-dir to identify checkpoints by content"}
    if not os.path.isdir(root):
        return {"_unavailable": f"--checkpoint-dir does not exist: {root}"}
    exts = (".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".json", ".yml", ".yaml")
    out = {"root": os.path.abspath(root), "files": [], "skipped_larger_than_bytes": limit_bytes}
    for dirpath, _dirs, names in os.walk(root):
        for n in sorted(names):
            if not n.endswith(exts):
                continue
            p = os.path.join(dirpath, n)
            try:
                st = os.stat(p)
            except OSError:
                continue
            rec = {"path": os.path.relpath(p, root), "size_bytes": st.st_size,
                   "mtime_epoch": int(st.st_mtime), "sha256": None}
            if limit_bytes and st.st_size > limit_bytes:
                rec["_unavailable"] = (f"larger than --max-hash-bytes ({limit_bytes}); "
                                       "re-run with a higher limit to hash it")
            else:
                h = hashlib.sha256()
                try:
                    with open(p, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 22), b""):
                            h.update(chunk)
                    rec["sha256"] = h.hexdigest()
                except OSError as e:
                    rec["_unavailable"] = f"unreadable: {e}"
            out["files"].append(rec)
    if not out["files"]:
        out["_unavailable"] = f"no checkpoint-shaped files found under {root}"
    return out


def _default_ckpt_dir():
    for env in ("EVO2_CHECKPOINT_DIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        v = os.environ.get(env)
        if v and os.path.isdir(v):
            return v
    d = os.path.expanduser("~/.cache/huggingface")
    return d if os.path.isdir(d) else None


FIXTURE_TEMPLATE = {
    "_what": "End-to-end scoring fixture. A referee runs the scorer on these variants and compares "
             "against expected_score. Fill expected_score on the scoring host; leave nothing blank.",
    "_how": "python tools/capture_scoring_environment.py --check-fixture reports/scoring_fixture.json "
            "--scored <parquet written by a fresh scoring run>",
    "_policy": {"checkpoint": None, "checkpoint_sha256": None, "window_bp": None,
                "readout": None, "precision": None, "batch_size": None,
                "scorer_command": None,
                "_note": "every field is REQUIRED for --check-fixture; a null anywhere fails"},
    "_tolerance": {"abs": 1e-6, "_note": "FP8 scaling is batch-dependent; if a fresh run does not "
                                         "match bit-for-bit, record the tolerance that does and say "
                                         "so here rather than loosening it silently."},
    "variants": [
        {"variant_id": None, "species": None, "chrom": None, "pos": None,
         "ref": None, "alt": None, "expected_score": None}
    ],
}


def _load_scored(path):
    """Return {variant_id: score} from a parquet or JSON produced by a scoring run."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict):
            return {str(k): float(v) for k, v in d.items()}
        return {str(r["variant_id"]): float(r["score"]) for r in d}
    import polars as pl
    df = pl.read_parquet(path)
    vid = next((c for c in df.columns if c.lower() in ("variant_id", "id")), None)
    sc = next((c for c in df.columns
               if c.lower() in ("score", "expected_score") or "evo2" in c.lower()), None)
    if vid is None or sc is None:
        raise SystemExit(f"{path}: need a variant-id column and a score column; "
                         f"found {df.columns}")
    return {str(a): float(b) for a, b in zip(df[vid].to_list(), df[sc].to_list())}


def check_fixture(fixture_path, scored_path):
    """Compare a fresh scoring run against the filled-in fixture. Exit non-zero on any mismatch."""
    with open(fixture_path, encoding="utf-8") as fh:
        fx = json.load(fh)
    rows = fx.get("variants") or []
    # A fixture is provenance, not a number pair. Every row must carry the coordinates and
    # alleles that define the variant, and the fixture must name the checkpoint, readout,
    # precision, batch policy and window it was scored under -- otherwise a "PASS" only says
    # that two floats matched, which is exactly the false assurance a referee flagged.
    ROW_REQ = ("variant_id", "species", "chrom", "pos", "ref", "alt", "expected_score")
    POL_REQ = ("checkpoint", "checkpoint_sha256", "window_bp", "readout", "precision",
               "batch_size", "scorer_command")
    pol = fx.get("_policy") or {}
    missing_pol = [k for k in POL_REQ if pol.get(k) in (None, "")]
    bad_rows = [i for i, r in enumerate(rows) if any(r.get(k) in (None, "") for k in ROW_REQ)]
    if not rows or bad_rows or missing_pol:
        print(f"FIXTURE NOT READY: {len(bad_rows)} of {len(rows)} rows have a null field among "
              f"{ROW_REQ}; policy fields missing: {missing_pol or 'none'}.")
        print("Fill them in on the scoring host. This script will not invent an expected score "
              "or a provenance field.")
        return 2
    for r in rows:
        # `x in "ACGT"` is a SUBSTRING test: it accepts "AC", "CG", "ACGT". Both readouts in
        # this paper are defined only for a single-base substitution, so test the length too.
        ref, alt = str(r["ref"]).upper(), str(r["alt"]).upper()
        if len(ref) != 1 or ref not in "ACGT" or len(alt) != 1 or alt not in "ACGT" \
                or ref == alt:
            print(f"FIXTURE INVALID: {r['variant_id']}: ref/alt must be distinct single bases.")
            return 2
        try:
            int(r["pos"])
        except (TypeError, ValueError):
            print(f"FIXTURE INVALID: {r['variant_id']}: pos is not an integer.")
            return 2
    if not scored_path:
        print("--check-fixture needs --scored <parquet or json from a fresh scoring run>")
        return 2
    got = _load_scored(scored_path)
    tol = float((fx.get("_tolerance") or {}).get("abs", 1e-6))
    missing, bad = [], []
    for r in rows:
        vid = str(r["variant_id"])
        if vid not in got:
            missing.append(vid); continue
        d = abs(got[vid] - float(r["expected_score"]))
        if d > tol:
            bad.append((vid, r["expected_score"], got[vid], d))
    print(f"fixture {fixture_path}: {len(rows)} variant(s), tolerance {tol:g}")
    for vid in missing:
        print(f"  MISSING from the scoring run: {vid}")
    for vid, exp, act, d in bad:
        print(f"  MISMATCH {vid}: expected {exp!r}, got {act!r}, |diff| {d:g}")
    if missing or bad:
        print(f"FAIL — {len(missing)} missing, {len(bad)} mismatched")
        return 1
    print("PASS — every fixture variant reproduced within tolerance")
    return 0


def self_test():
    """Negative and positive controls for --check-fixture. Exit 0 only if every case behaves."""
    import json, tempfile, os
    base = json.loads(json.dumps(FIXTURE_TEMPLATE))
    good_pol = {"checkpoint": "evo2_40b", "checkpoint_sha256": "0" * 64, "window_bp": 8192,
                "readout": "mean_log_likelihood", "precision": "fp8", "batch_size": 4,
                "scorer_command": "python score.py --window 8192"}
    good_row = {"variant_id": "v1", "species": "dog", "chrom": "1", "pos": 100,
                "ref": "A", "alt": "G", "expected_score": -1.234567}
    cases = [  # (label, policy, row, scored, expected exit)
        ("null metadata, matching score", {}, {"variant_id": "v1", "expected_score": -1.234567}, {"v1": -1.234567}, 2),
        ("null policy only", {}, good_row, {"v1": -1.234567}, 2),
        ("null coordinates only", good_pol, {**good_row, "pos": None}, {"v1": -1.234567}, 2),
        ("same ref and alt", good_pol, {**good_row, "alt": "A"}, {"v1": -1.234567}, 2),
        # Multi-base alleles. The original test was `x in "ACGT"`, a substring test that
        # accepted every one of these; the suite covered only ref == alt, so it passed.
        ("multi-base ref AC", good_pol, {**good_row, "ref": "AC"}, {"v1": -1.234567}, 2),
        ("multi-base alt CG", good_pol, {**good_row, "alt": "CG"}, {"v1": -1.234567}, 2),
        ("multi-base ref ACGT", good_pol, {**good_row, "ref": "ACGT"}, {"v1": -1.234567}, 2),
        ("multi-base ref ACG", good_pol, {**good_row, "ref": "ACG", "alt": "T"}, {"v1": -1.234567}, 2),
        ("multi-base both CG/GT", good_pol, {**good_row, "ref": "CG", "alt": "GT"}, {"v1": -1.234567}, 2),
        ("non-ACGT base N", good_pol, {**good_row, "alt": "N"}, {"v1": -1.234567}, 2),
        ("lowercase a/g accepted", good_pol, {**good_row, "ref": "a", "alt": "g"}, {"v1": -1.234567}, 0),
        ("complete, matching", good_pol, good_row, {"v1": -1.234567}, 0),
        ("complete, mismatched", good_pol, good_row, {"v1": -1.2}, 1),
        ("complete, variant absent", good_pol, good_row, {"v9": -1.234567}, 1),
    ]
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for lab, pol, row, scored, want in cases:
            fx = dict(base); fx["_policy"] = {**base["_policy"], **pol}; fx["variants"] = [row]
            fp, sp = os.path.join(td, "fx.json"), os.path.join(td, "sc.json")
            json.dump(fx, open(fp, "w")); json.dump(scored, open(sp, "w"))
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                got = check_fixture(fp, sp)
            flag = "ok " if got == want else "BAD"
            ok &= got == want
            print(f"  {flag}  {lab:<30} exit {got} (expected {want})")
    print("SELF-TEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/scoring_environment.json")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="directory holding the Evo 2 checkpoints (default: HF cache if present)")
    ap.add_argument("--max-hash-bytes", type=int, default=0,
                    help="skip hashing files larger than this many bytes (0 = hash everything)")
    ap.add_argument("--batch-size", default=None, help="scoring batch size actually used")
    ap.add_argument("--precision", default=None, help="e.g. fp8, bf16, fp16, fp32")
    ap.add_argument("--window", default=None, help="e.g. 8192 or 1001")
    ap.add_argument("--command", default=None, help="the exact scoring command, quoted")
    ap.add_argument("--emit-fixture-template", metavar="PATH", default=None)
    ap.add_argument("--check-fixture", metavar="PATH", default=None,
                    help="compare a fresh scoring run against a filled-in fixture")
    ap.add_argument("--scored", metavar="PATH", default=None,
                    help="parquet/JSON from a fresh scoring run, for --check-fixture")
    ap.add_argument("--self-test", action="store_true",
                    help="run the negative/positive controls for --check-fixture and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.check_fixture:
        return check_fixture(args.check_fixture, args.scored)

    if args.emit_fixture_template:
        os.makedirs(os.path.dirname(args.emit_fixture_template) or ".", exist_ok=True)
        with open(args.emit_fixture_template, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(FIXTURE_TEMPLATE, fh, indent=2)
            fh.write("\n")
        print(f"wrote fixture template {args.emit_fixture_template} — fill in every null on the "
              f"scoring host; this script will not invent an expected score")
        return 0

    ckpt_dir = args.checkpoint_dir or _default_ckpt_dir()
    rec = {
        "_schema": "ccs/scoring-environment/1",
        "_what": "The GPU scoring environment, captured from the live host by "
                 "tools/capture_scoring_environment.py. Every field is read from the machine; "
                 "fields that could not be determined are null and carry a _unavailable note.",
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python_version": sys.version,
            "python_executable": sys.executable,
        },
        "packages_probed": {m: _version(m) for m in PROBE},
        "pip_freeze": (_run([sys.executable, "-m", "pip", "freeze"]) or "").splitlines() or None,
        "torch": _torch_block(),
        "nvidia_smi": _probe(["nvidia-smi",
                              "--query-gpu=name,driver_version,memory.total,compute_cap",
                              "--format=csv,noheader"],
                             "run this on the scoring host, with the GPU visible"),
        "nvcc": _probe(["nvcc", "--version"],
                       "the CUDA runtime alone does not install nvcc; record the CUDA toolkit "
                       "version by hand if the scoring host has no nvcc"),
        "checkpoints": _hash_checkpoints(ckpt_dir, args.max_hash_bytes),
        "scoring_policy": {
            "batch_size": args.batch_size,
            "precision": args.precision,
            "window_bp": args.window,
            "command": args.command,
            "_note": "These four are declared on the command line, not detected. If any is null, "
                     "re-run with the corresponding flag — a null here is the same gap the "
                     "referees flagged.",
        },
    }
    todo = []
    if not args.batch_size or not args.precision or not args.window or not args.command:
        todo.append("scoring_policy: pass --batch-size, --precision, --window and --command")
    if rec["torch"].get("_unavailable"):
        todo.append("torch/CUDA: run this on the scoring host with the GPU visible")
    if rec["checkpoints"].get("_unavailable"):
        todo.append("checkpoints: pass --checkpoint-dir pointing at the Evo 2 weights")
    ckpt_files = rec["checkpoints"].get("files") or []
    unhashed = [f for f in ckpt_files if not f.get("sha256")]
    if unhashed:
        todo.append(f"checkpoints: {len(unhashed)} of {len(ckpt_files)} file(s) carry no sha256 "
                    "— re-run without --max-hash-bytes (0 = hash everything); a checkpoint "
                    "identified by name and size only is not identified by content")
    if not rec["pip_freeze"]:
        todo.append("pip_freeze: empty — `python -m pip freeze` produced nothing in the scoring "
                    "interpreter (no pip module, or pip failed). Record the environment another "
                    "way (conda list / uv pip freeze) and paste it in by hand")
    for m in ("evo2", "vortex", "transformer_engine"):
        if not rec["packages_probed"][m]["installed"]:
            todo.append(f"{m}: not importable here — run on the scoring host")
    rec["manual_todo"] = todo or ["none — this record is complete"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"wrote {args.out}")
    for t in rec["manual_todo"]:
        print(f"  TODO: {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
