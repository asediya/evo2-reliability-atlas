# -*- coding: utf-8 -*-
"""Emit the cattle sample table the Methods promise is deposited.

The Methods state that the 7,394 individuals behind the cattle allele-frequency panel are described
by "the deposited sample table", and that "the individual accessions are in the deposited sample
table for anyone who wishes to reconstruct it". No such table was in the deposit, so that sentence
was false about its own archive. This writes it from the PLINK2 .psam files the panel was actually
built from, so the claim is backed by the artefact rather than by a promise.

    python src/ccs/build_cattle_sample_table.py
    -> reports/cattle_sample_table.tsv
"""
import glob
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PSAM = sorted(glob.glob("data/interim/geno_pgen/*.psam"))
OUT = "reports/cattle_sample_table.tsv"

if not PSAM:
    raise SystemExit("no .psam files under data/interim/geno_pgen/ — cannot build the sample table")

# Every per-chromosome .psam carries the same cohort; read them all and confirm that, rather than
# trusting the first, because a mismatch would mean the panel is not one cohort.
sets = {}
for p in PSAM:
    ids = []
    col = 0                                   # .psam is "#IID SEX"; take the column the header names
    with io.open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.lstrip("#").split()
                col = header.index("IID") if "IID" in header else 0
                continue
            parts = line.split()
            if parts and col < len(parts):
                ids.append(parts[col])
    sets[os.path.basename(p)] = ids

first = list(sets.values())[0]
same = all(v == first for v in sets.values())
print("%d .psam file(s); identical cohort across all: %s" % (len(sets), same))
if not same:
    sizes = {k: len(v) for k, v in sets.items()}
    print("  cohort sizes differ: %s" % sizes)

ids = first
with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("# Cattle allele-frequency panel: individual identifiers as carried in the PLINK2\n")
    fh.write("# .psam files the panel was built from (data/interim/geno_pgen/*.psam).\n")
    fh.write("# These are the sample identifiers of the aggregated public resequencing cohort; the\n")
    fh.write("# panel is an aggregation rather than a single published study, so they are given as\n")
    fh.write("# the record of what was used, not as a claim about any one cohort's provenance.\n")
    fh.write("sample_id\n")
    for s in ids:
        fh.write("%s\n" % s)

print("wrote %s: %d individuals" % (OUT, len(ids)))
if len(ids) != 7394:
    print("NOTE: the Methods state 7,394 individuals; this table has %d. "
          "Reconcile before submission." % len(ids))
