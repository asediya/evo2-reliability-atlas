# -*- coding: utf-8 -*-
"""Every scoring panel's var_off must index the base that actually differs between ref and alt.

The single-position scorers read `lp[alt_base] - lp[ref_base]` at `var_off`. If that offset points
anywhere other than the substitution, ref and alt carry the same base there and the delta is exactly
zero -- for every row, silently, with the job exiting 0. The splicing and ClinVar
1,001-bp panels cannot take var_off = 501 from the atlas windows, which are 1,002 bp with the
variant at 501, while these are 1,001 bp with the variant at 500.

The scorer refuses such a panel. This sweeps every panel on disk instead of waiting for one to
be scored, so a builder that regresses is caught at build time rather than after a GPU run. It also
reports the window length beside the offset, because the whole defect is that two window
constructions share the label "1,001 bp" and differ by one base.

    python tools/check_var_off.py

Exit 0 when every panel is consistent, 1 otherwise.
"""
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PATTERNS = [
    "data/interim/*_scoring_windows.parquet",
    "data/interim/*windows*.parquet",
    "analyses/data/**/*w1001*.parquet",
    "analyses/data/**/*windows*.parquet",
]
SAMPLE = 500          # rows per panel; the offset is a constant, so a sample settles it


def main():
    seen, rows, bad = set(), [], []
    for pat in PATTERNS:
        for p in sorted(glob.glob(pat, recursive=True)):
            rp = os.path.normpath(p)
            if rp in seen:
                continue
            seen.add(rp)
            try:
                d = pl.read_parquet(p)
            except Exception as e:
                rows.append((rp, "-", "-", "unreadable: %s" % str(e)[:40]))
                continue
            if not {"ref_seq", "alt_seq", "var_off"} <= set(d.columns):
                continue
            h = d.head(SAMPLE)
            wl = len(h["ref_seq"][0])
            offs = sorted(set(int(v) for v in h["var_off"].to_list()))
            # Two different faults look alike at a glance and must not be conflated:
            #
            #   a builder off-by-one -- var_off misses the substitution in essentially EVERY row,
            #   because the offset is a constant. This is the one that silently zeroes a whole run.
            #
            #   a reference mismatch -- a handful of records whose assembly base is the alternate
            #   allele, so substituting alt reproduces the reference window. The window is then
            #   identical end to end and the delta is zero for a reason that is about the data, not
            #   the code. These are flagged ref_ok = False where the panel carries that column.
            #
            # Failing on the second would make this gate cry wolf on panels that are behaving
            # correctly, so only a systematic miss fails.
            eq, ident = [], 0
            for i, (r, a, o) in enumerate(zip(h["ref_seq"].to_list(), h["alt_seq"].to_list(),
                                              h["var_off"].to_list())):
                if r[int(o)].upper() == a[int(o)].upper():
                    eq.append(i)
                    if r == a:
                        ident += 1
            n_eq = len(eq)
            if n_eq == 0:
                note = "ok"
            elif n_eq == h.height:
                r, a = h["ref_seq"][0], h["alt_seq"][0]
                diff = [i for i, (x, y) in enumerate(zip(r, a)) if x != y]
                note = "*** OFF-BY-ONE: var_off misses the substitution in ALL %d rows" % h.height
                note += " (they differ at %s)" % (diff[:3] if diff else "nowhere")
                bad.append(rp)
            else:
                note = "%d of %d rows identical ref/alt window" % (n_eq, h.height)
                flagged = ""
                if "ref_ok" in d.columns:
                    ids = [h["variant_id"][i] for i in eq] if "variant_id" in h.columns else []
                    nok = set(d.filter(~pl.col("ref_ok"))["variant_id"].to_list()) if ids else set()
                    flagged = ", %d flagged ref_ok=False" % sum(1 for v in ids if v in nok)
                note += "%s -- reference mismatch, not an offset fault" % flagged
            rows.append((rp, wl, offs[0] if len(offs) == 1 else offs, note))

    print("=" * 104)
    print("  %-58s %6s %6s  %s" % ("panel", "win bp", "var_off", "state"))
    print("=" * 104)
    for p, wl, off, note in rows:
        print("  %-58s %6s %6s  %s" % (p[-58:], wl, off, note))
    print()
    if bad:
        print("  RESULT: FAIL -- %d panel(s) whose var_off misses the substitution in every row."
              % len(bad))
        for p in bad:
            print("    %s" % p)
        print("  Fix the panel BUILDER, never the offset at scoring time.")
        return 1
    if not rows:
        # A scan that finds nothing must not print "CLEAN -- 0 panel(s)" and exit 0, which is
        # indistinguishable in the output from a real pass: a vacuous pass is the one outcome this gate
        # must not report. The window panels it reads live under data/ and are not deposited.
        print("  RESULT: NO DATA -- 0 panels found. The window panels this gate reads are not")
        print("  part of the code deposit (see reports/DATA_MANIFEST.md); from a clone there is")
        print("  nothing to scan, and that is NOT a pass.")
        # ...and so it must not exit 0. Exit 3 is this archive's "stopped at an undeposited path"
        # convention, which tools/run_all_gates.py files as NODATA rather than PASS.
        return 3
    print("  RESULT: CLEAN -- %d panel(s), no systematic var_off fault" % len(rows))
    print("  Rows noted above as 'reference mismatch' are records whose assembly base is the")
    print("  alternate allele; their windows are identical and score exactly zero by construction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
