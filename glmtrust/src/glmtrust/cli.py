"""Command-line interface: ``glmtrust <command> INPUT [options]``.

INPUT is one table (``.parquet``, ``.csv`` or ``.tsv``) with a score column, a label column and,
optionally, a group column. Parquet needs the optional ``polars`` dependency; CSV/TSV use only the
standard library.

    glmtrust evaluate  PANEL.parquet --score-col SCORE --label-col LABEL --group-col GROUP
    glmtrust calibrate PANEL.csv     --score-col SCORE --label-col LABEL --out probs.csv
    glmtrust transfer  PANEL.parquet --score-col SCORE --label-col LABEL --group-col GROUP --out probs.csv
    glmtrust audit     PANEL.parquet --score-col A --readout R --score-col B --readout R --label-col LABEL
    glmtrust reach     REACH.parquet --label-col LABEL --reach-prefix reach__ --covered COVERED.csv
    glmtrust baseline  PANEL.parquet --label-col LABEL --group-col GENE --class-col CONSEQUENCE

No sample panel ships with the package. To see the tool run on real data, use
benchmarks/reproduce_paper_trust_layer.py, which carries its own fixture.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

from . import __version__
from .audit import DEFAULT_LAMBDAS
from .calibration import cross_conformal_calibrate
from .pipeline import TrustLayer
from .transfer import leave_one_group_out


def _load(path, columns):
    """Return {column: object-array} from a parquet/csv/tsv file.

    `columns=None` reads every column, which is what --auto needs in order to see the candidates
    it is choosing between.
    """
    wanted = None if columns is None else [c for c in columns if c]
    # A missing input file otherwise surfaced as a polars traceback from three frames down, which is
    # what a reader copying a command out of the README saw first. The tool already exits cleanly on
    # a missing column; it should do the same on a missing file.
    if not os.path.exists(path):
        sys.exit("no such file: %s" % path)
    if path.endswith(".parquet"):
        try:
            import polars as pl
        except ImportError:
            sys.exit("reading .parquet needs the optional dependency: pip install 'glmtrust[io]'")
        df = pl.read_parquet(path)
        if wanted is None:
            wanted = list(df.columns)
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            sys.exit("column(s) not found in %s: %s (have: %s)" % (path, missing, df.columns))
        return {c: df[c].to_numpy() for c in wanted}
    delim = "\t" if path.endswith(".tsv") else ","
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=delim))
    if not rows:
        sys.exit("empty input file: %s" % path)
    if wanted is None:
        wanted = list(rows[0])
    missing = [c for c in wanted if c not in rows[0]]
    if missing:
        sys.exit("column(s) not found in %s: %s (have: %s)" % (path, missing, list(rows[0])))
    return {c: np.array([r[c] for r in rows], dtype=object) for c in wanted}


def _arrays(args):
    """Score, label and group arrays, restricted to the variants the scorer actually scored.

    A real panel carries no-calls: on dbNSFP's 328,328 variants at one review star or better,
    REVEL returns no value for 40% of them -- 65% of the positives and 15% of the negatives -- and a
    scorer that declines is the case this package exists for. The calibration and conformal layers
    fit a model, so they cannot take a NaN, and passing one through produced a raw scikit-learn
    traceback on the most ordinary input there is.

    Dropping them here is the right behaviour for `calibrate`, `evaluate` and `transfer`, which are
    about the scores that exist. It is the wrong behaviour for the question "what do the missing ones
    cost", so the message points at `glmtrust audit`, which is the command that answers it.
    """
    data = _load(args.input, [args.score_col, args.label_col, getattr(args, "group_col", None)])
    scores = _col(data, args.score_col)
    labels = _col(data, args.label_col).astype(int)
    groups = data[args.group_col] if getattr(args, "group_col", None) else None

    keep = np.isfinite(scores)
    n_drop = int((~keep).sum())
    if n_drop:
        if keep.sum() == 0:
            sys.exit("every value in --score-col %s is missing; nothing to fit" % args.score_col)
        sys.stderr.write(
            "note: %d of %d variants have no %s score and are excluded from the fit.\n"
            "      What that missingness costs is what 'glmtrust audit' measures.\n"
            % (n_drop, len(scores), args.score_col))
        scores, labels = scores[keep], labels[keep]
        if groups is not None:
            groups = groups[keep]
    return scores, labels, groups


def _write_rows(path, header, rows):
    """Write rows to path, honouring the extension.

    This always wrote CSV whatever the filename said, so `--out foo.parquet` produced a CSV named
    .parquet that every parquet reader then refused to open. Silently writing a different format
    from the one requested is worse than refusing: the file exists, looks right in a listing, and
    fails only when someone tries to read it.
    """
    if str(path).lower().endswith(".parquet"):
        try:
            import polars as pl
        except ImportError:
            raise SystemExit("writing %s needs polars for parquet output; install polars, or "
                             "choose a .csv path" % path)
        pl.DataFrame({h: [r[i] for r in rows] for i, h in enumerate(header)}).write_parquet(path)
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


#: What a delimited file writes where a scorer returned nothing. A parquet column carries a real
#: null and arrives as NaN; a CSV or TSV carries one of these strings, and every one of them used to
#: take down the whole column with "could not convert string to float: ''". That is the one input
#: this package exists to measure, so it has to survive the front door.
_NA = {"", "na", "n/a", "nan", "null", "none", ".", "-", "?"}


def _to_float(v):
    """One cell as a float, with the usual no-call sentinels mapped to NaN.

    A value that is neither a number nor a recognised sentinel still raises, by name and with the
    offending string, rather than being swallowed as missing: silently reading a typo as a no-call
    would change the reach this package reports.
    """
    import numpy as _np
    if v is None:
        return _np.nan
    if isinstance(v, (int, float, _np.integer, _np.floating)):
        return float(v)
    t = str(v).strip()
    if t.lower() in _NA:
        return _np.nan
    try:
        return float(t)
    except ValueError:
        raise ValueError(
            "cannot read %r as a number. If it means 'not scored', write it as an empty field or "
            "one of %s; glmtrust will not guess, because reading a typo as a no-call would change "
            "the reach it reports." % (v, sorted(x for x in _NA if x))) from None


def _col(tab, name):
    """One column as a float array, whatever the table backend."""
    import numpy as _np
    try:
        return _np.asarray(tab[name], dtype=float)
    except (TypeError, ValueError):
        return _np.array([_to_float(v) for v in list(tab[name])], dtype=float)


def cmd_evaluate(args):
    scores, labels, groups = _arrays(args)
    layer = TrustLayer(calibration=args.calibration, conformal=args.conformal,
                       alpha=args.alpha, coverage=args.coverage, seed=args.seed)
    report = layer.evaluate(scores, labels, groups)
    print(layer.summary(scores, labels, groups))
    if args.out:
        json.dump(report, open(args.out, "w", encoding="utf-8"), indent=2)
        print("\nwrote %s" % args.out)
    return 0


def cmd_calibrate(args):
    scores, labels, _ = _arrays(args)
    probs = cross_conformal_calibrate(scores, labels, method=args.calibration, seed=args.seed)
    if args.out:
        _write_rows(args.out, ["index", "score", "label", "probability"],
                    [[i, scores[i], labels[i], probs[i]] for i in range(len(probs))])
        print("wrote %s (%d calibrated probabilities)" % (args.out, len(probs)))
    else:
        for p in probs:
            print("%.6f" % p)
    return 0


def cmd_transfer(args):
    scores, labels, groups = _arrays(args)
    if groups is None:
        sys.exit("transfer needs --group-col")
    probs, report = leave_one_group_out(scores, labels, groups, method=args.calibration)
    for g, r in report.items():
        if g == "__macro__":
            print("MACRO  AUROC %.3f  ECE %.4f  over %d groups"
                  % (r["auroc"], r["ece"], r["n_groups"]))
        elif "auroc" in r:
            print("%-16s n=%-5d AUROC %.3f  ECE %.4f%s"
                  % (str(g), r["n"], r["auroc"], r["ece"],
                     "   below chance: check the score's sign on this group"
                     if r.get("below_chance") else ""))
        else:
            print("%-16s n=%-5d %s" % (str(g), r["n"], r.get("note", "")))
    if args.out:
        _write_rows(args.out, ["index", "group", "score", "label", "transferred_probability"],
                    [[i, groups[i], scores[i], labels[i], probs[i]] for i in range(len(probs))])
        print("wrote %s" % args.out)
    return 0


# Columns that are numeric but are not scores. Kept as substrings because annotation tables prefix
# and suffix freely (hg38_pos, pos_1based, #chr). Matched case-insensitively against the whole name.
_NOT_A_SCORE = ("pos", "position", "start", "end", "chrom", "chr", "label", "class", "stars",
                "rank", "id", "index", "count", "n_", "_n", "length", "len", "af", "freq",
                "allele", "gene", "strand", "build", "assembly")


def _detect_score_columns(tab, label_col, strata_col=None):
    """Guess which columns are scores, and be loud about the guess.

    Annotation tables carry dozens of numeric columns that are coordinates, counts, ranks and
    frequencies. Silently auditing those would produce a page of confident nonsense, so the rule is
    deliberately conservative and the caller prints what it selected. A column is a candidate when
    it parses as float for most rows and its name does not look like bookkeeping.
    """
    import numpy as _np

    out = []
    for name, col in tab.items():
        if name in (label_col, strata_col):
            continue
        low = str(name).lower()
        if any(tok == low or tok in low.split("_") or low.startswith(tok + "_")
               or low.endswith("_" + tok) for tok in _NOT_A_SCORE):
            continue
        try:
            v = _np.asarray(col, dtype=float)
        except (TypeError, ValueError):
            # A delimited file writes no-calls as strings, so a genuine score column with
            # missingness fails the bulk cast. Retry per element before discarding it: silently
            # skipping such a column made --auto pass over exactly the scorers whose reach this
            # package is meant to report.
            try:
                v = _np.array([_to_float(x) for x in col], dtype=float)
            except ValueError:
                continue
        finite = _np.isfinite(v)
        if finite.sum() < 2 or _np.unique(v[finite]).size < 3:
            continue          # constant or near-constant: not a score
        out.append(name)
    return out


def cmd_audit(args):
    """Audit a comparison between two or more scorers before believing its headline."""
    import json as _json

    from .audit import Scorer, audit as _audit

    if args.auto:
        if args.score_col:
            sys.exit("--auto selects the score columns itself; do not also pass --score-col")
        if len(args.readout) != 1:
            sys.exit("--auto needs exactly one --readout. Giving one readout for every detected "
                     "column asserts they are comparable; if they are not, name them individually "
                     "with --score-col/--readout instead.")
        tab = _load(args.input, None)
        if args.label_col not in tab:
            sys.exit("label column %r not found; have: %s" % (args.label_col, sorted(tab)))
        cols = [c for c in _detect_score_columns(tab, args.label_col, args.strata_col)
                if c != args.cluster_col]
        if len(cols) < 1:
            sys.exit("--auto found no score-like column in %s. Name them with --score-col."
                     % args.input)
        print("--auto selected %d score column(s): %s" % (len(cols), ", ".join(cols)))
        print("  skipped as bookkeeping: %s"
              % ", ".join(c for c in tab if c not in cols
                          and c not in (args.label_col, args.strata_col, args.cluster_col))
              or "(none)")
        print()
        args.score_col = cols
        args.readout = list(args.readout) * len(cols)
    else:
        if not args.score_col:
            sys.exit("give --score-col (repeatable), or --auto to detect them")
        want = list(args.score_col) + [args.label_col]
        if args.strata_col:
            want.append(args.strata_col)
        if args.cluster_col:
            want.append(args.cluster_col)
        tab = _load(args.input, want)

    labels = _col(tab, args.label_col)
    if len(args.score_col) != len(args.readout):
        sys.exit("give one --readout per --score-col (%d score columns, %d readouts)"
                 % (len(args.score_col), len(args.readout)))
    # strata are categorical, so they bypass _col's float conversion
    strata = np.asarray(tab[args.strata_col], dtype=object) if args.strata_col else None
    if args.cluster_col and args.cluster_col not in tab:
        sys.exit("cluster column %r not found; have: %s" % (args.cluster_col, sorted(tab)))
    cluster = np.asarray(tab[args.cluster_col], dtype=object) if args.cluster_col else None
    flip = set(args.lower_is_worse)
    unknown = sorted(flip - set(args.score_col))
    if unknown:
        sys.exit("--lower-is-worse names %s, which %s not among the scored columns (%s)"
                 % (", ".join(unknown), "is" if len(unknown) == 1 else "are",
                    ", ".join(args.score_col)))
    scorers = [Scorer(c, _col(tab, c), readout=r, higher_is_worse=c not in flip)
               for c, r in zip(args.score_col, args.readout)]
    rep = _audit(labels, scorers, strata=strata, n_boot=args.n_boot, seed=args.seed,
                 min_stratum=args.min_stratum, min_class=args.min_class, cluster=cluster)
    print(rep)
    if args.out:
        def _plain(obj):
            d = dict(obj.__dict__)
            if "strata" in d:
                d["strata"] = [dict(t.__dict__) for t in d["strata"]]
            return d
        payload = {"scorers": [_plain(s) for s in rep.scorers],
                   "pairs": [dict(p.__dict__) for p in rep.pairs],
                   "warnings": rep.warnings}
        _json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=2, default=float)
        print("\nwrote %s" % args.out)
    if args.card:
        from .card import write_card
        write_card(rep, args.card, title=args.card_title or "glmtrust audit",
                   subtitle="%s   n=%s" % (args.input, format(len(labels), ",")))
        print("wrote %s" % args.card)
    return 0


def _lambdas(text):
    """--lambdas as a list of floats, each a contamination share in [0, 1]."""
    try:
        vals = [float(t) for t in str(text).split(",") if t.strip()]
    except ValueError:
        sys.exit("--lambdas takes comma-separated numbers in [0, 1], e.g. 0.05,0.1,1; got %r" % text)
    bad = [v for v in vals if not 0.0 <= v <= 1.0]
    if bad:
        sys.exit("--lambdas: each value is a share of the unscored pairs and must lie in [0, 1]; "
                 "got %s" % bad)
    return vals


def cmd_reach(args):
    """Reach accounting from reach indicators, with covered AUROCs optional."""
    import json as _json

    from .audit import reach_audit

    lambdas = _lambdas(args.lambdas)
    tab = _load(args.input, None)
    if args.label_col not in tab:
        sys.exit("label column %r not found; have: %s" % (args.label_col, sorted(tab)))
    cols = list(args.reach_col)
    if args.reach_prefix:
        cols += [c for c in tab if c.startswith(args.reach_prefix) and c not in cols]
    if not cols:
        sys.exit("name the reach columns with --reach-col (repeatable) or --reach-prefix")
    missing = [c for c in cols if c not in tab]
    if missing:
        sys.exit("reach column(s) not found: %s" % ", ".join(missing))
    strip = args.reach_prefix or ""
    name = {c: (c[len(strip):] if strip and c.startswith(strip) else c) for c in cols}
    keep = np.ones(len(tab[args.label_col]), dtype=bool)
    if args.subset_col:
        if args.subset_col not in tab:
            sys.exit("subset column %r not found" % args.subset_col)
        keep = np.asarray([str(v) == args.subset_value for v in tab[args.subset_col]])
        if not keep.any():
            sys.exit("no row has %s == %r" % (args.subset_col, args.subset_value))
    labels = _col(tab, args.label_col)[keep]
    reach = {name[c]: np.asarray(tab[c])[keep] for c in cols}
    covered = None
    if args.covered:
        ctab = _load(args.covered, [args.covered_name_col, args.covered_auroc_col])
        covered = {str(n): float(v) for n, v in
                   zip(ctab[args.covered_name_col], ctab[args.covered_auroc_col])}
    cluster = None
    if args.cluster_col:
        if args.cluster_col not in tab:
            sys.exit("cluster column %r not found" % args.cluster_col)
        cluster = np.asarray(tab[args.cluster_col], dtype=object)[keep]
    rep = reach_audit(labels, reach, covered_auroc=covered, cluster=cluster,
                      n_boot=args.n_boot, seed=args.seed, lambdas=lambdas)
    print(rep)
    if args.out:
        payload = {"scorers": [dict(s.__dict__) for s in rep.scorers],
                   "n_pairs": rep.n_pairs, "feasible": rep.feasible,
                   "feasible_sharp": rep.feasible_sharp, "identified": rep.identified,
                   "identified_monotone": rep.identified_monotone,
                   "identified_pairs": rep.identified_pairs, "interval_level": rep.interval_level,
                   "warnings": rep.warnings, "breakdown": rep.breakdown,
                   "frontier": rep.frontier,
                   "median_breakdown_undecided": rep.median_breakdown_undecided}
        _json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=2, default=float)
        print("\nwrote %s" % args.out)
    return 0


def cmd_baseline(args):
    """Sequence-blind baselines: out-of-fold group and class positive rates on the table's labels."""
    import json as _json
    from dataclasses import asdict

    from .baseline import sequence_blind

    if not (args.group_col or args.class_col):
        sys.exit("baseline needs --group-col (usually gene), --class-col (usually consequence) "
                 "or both")
    if args.seeds < 1:
        sys.exit("--seeds is the number of fold seeds, 0 to SEEDS - 1; give at least 1")
    tab = _load(args.input, [args.label_col, args.group_col, args.class_col])
    # groups and classes are categorical, so they bypass _col's float conversion
    groups = np.asarray(tab[args.group_col], dtype=object) if args.group_col else None
    classes = np.asarray(tab[args.class_col], dtype=object) if args.class_col else None
    try:
        rep = sequence_blind(_col(tab, args.label_col), groups=groups, classes=classes,
                             folds=args.folds, alpha=args.alpha, seeds=range(args.seeds))
    except ValueError as e:
        sys.exit("baseline: %s" % e)
    print(rep)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            _json.dump(asdict(rep), fh, indent=2)
        print("\nwrote %s" % args.out)
    return 0


def _common(sub, groups=True):
    sub.add_argument("input", help="parquet/csv/tsv table")
    sub.add_argument("--score-col", required=True)
    sub.add_argument("--label-col", required=True)
    if groups:
        sub.add_argument("--group-col", default=None)
    # Runnability report T5: this defaulted to isotonic while the Methods state the per-variant
    # deliverable is the two-parameter Platt posterior, so the documented command handed a reader
    # a different arm from the paper. The two differ materially on the paper's own panel (AUROC
    # 0.820 against 0.847, lift 2.37 against 2.56). The tool now ships what the paper reports.
    sub.add_argument("--calibration", default="platt", choices=["isotonic", "platt"])
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--out", default=None)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="glmtrust", description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version="glmtrust %s" % __version__)
    sub = ap.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="cross-validated trust-layer report")
    _common(ev)
    ev.add_argument("--conformal", default="mondrian", choices=["mondrian", "split"])
    ev.add_argument("--alpha", type=float, default=0.1)
    ev.add_argument("--coverage", type=float, default=0.85)
    ev.set_defaults(func=cmd_evaluate)

    ca = sub.add_parser("calibrate", help="write out-of-fold calibrated probabilities")
    _common(ca, groups=False)
    ca.set_defaults(func=cmd_calibrate)

    au = sub.add_parser("audit",
                        help="reach/readout audit of a comparison between scorers")
    au.add_argument("input", help="parquet/csv/tsv table")
    au.add_argument("--score-col", action="append", default=[],
                    help="a score column; repeat for each scorer")
    au.add_argument("--auto", action="store_true",
                    help="detect the score columns instead of naming them, and print the choice; "
                         "needs exactly one --readout, which asserts they are comparable")
    au.add_argument("--readout", required=True, action="append",
                    help="the readout that produced the matching --score-col; repeat")
    au.add_argument("--label-col", required=True)
    au.add_argument("--strata-col", default=None,
                    help="categorical column (usually variant consequence) to hold fixed, so that "
                         "class-dependent reach can be separated from composition")
    au.add_argument("--min-stratum", type=int, default=40,
                    help="smallest stratum to report (default 40)")
    au.add_argument("--min-class", type=int, default=10,
                    help="variants of EACH label required per stratum (default 10)")
    au.add_argument("--lower-is-worse", action="append", default=[], metavar="SCORE_COL",
                    help="a score column where LOWER values mean more damaging; repeat. Without "
                         "it every score is read as higher-is-worse")
    au.add_argument("--cluster-col", default=None,
                    help="group column (gene, locus): every interval becomes a bootstrap over "
                         "whole groups instead of a variant-level one")
    au.add_argument("--card", default=None,
                    help="also write a self-contained HTML audit card to this path")
    au.add_argument("--card-title", default=None)
    au.add_argument("--n-boot", type=int, default=2000)
    au.add_argument("--seed", type=int, default=0)
    au.add_argument("--out", default=None)
    au.set_defaults(func=cmd_audit)

    re_ = sub.add_parser("reach",
                         help="reach accounting from reach indicators; covered AUROCs optional")
    re_.add_argument("input", help="parquet/csv/tsv table with a label and reach columns")
    re_.add_argument("--label-col", required=True)
    re_.add_argument("--reach-col", action="append", default=[],
                     help="a boolean or 0/1 reach column, or a raw score with NaN where "
                          "declined; repeat")
    re_.add_argument("--reach-prefix", default=None,
                     help="take every column with this prefix, and strip it to name the scorer")
    re_.add_argument("--covered", default=None,
                     help="table of covered AUROCs, one row per scorer, to count identified pairs")
    re_.add_argument("--covered-name-col", default="pred")
    re_.add_argument("--covered-auroc-col", default="a_cov")
    re_.add_argument("--subset-col", default=None,
                     help="restrict to rows where this column equals --subset-value")
    re_.add_argument("--subset-value", default=None)
    re_.add_argument("--cluster-col", default=None,
                     help="group column: class-gap intervals become a bootstrap over whole groups")
    re_.add_argument("--lambdas", default=",".join("%g" % v for v in DEFAULT_LAMBDAS),
                     help="comma-separated contamination shares in [0, 1] at which the frontier "
                          "counts ordered pairs, given covered AUROCs (default %(default)s)")
    re_.add_argument("--n-boot", type=int, default=2000)
    re_.add_argument("--seed", type=int, default=0)
    re_.add_argument("--out", default=None)
    re_.set_defaults(func=cmd_reach)

    bl = sub.add_parser("baseline",
                        help="sequence-blind baselines: out-of-fold group and class positive rates")
    bl.add_argument("input", help="parquet/csv/tsv table")
    bl.add_argument("--label-col", required=True)
    bl.add_argument("--group-col", default=None,
                    help="categorical column scored by its own out-of-fold positive rate (gene)")
    bl.add_argument("--class-col", default=None,
                    help="categorical column (consequence); with --group-col it also gives the "
                         "group-within-class baseline")
    bl.add_argument("--folds", type=int, default=5, help="label-stratified folds (default 5)")
    bl.add_argument("--alpha", type=float, default=1.0,
                    help="pseudo-count by which each rate is shrunk (default 1.0)")
    bl.add_argument("--seeds", type=int, default=8,
                    help="average over fold seeds 0 to SEEDS - 1 (default 8)")
    bl.add_argument("--out", default=None, help="also write the report as JSON")
    bl.set_defaults(func=cmd_baseline)

    tr = sub.add_parser("transfer", help="leave-one-group-out calibration transfer")
    _common(tr)
    tr.set_defaults(func=cmd_transfer)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
