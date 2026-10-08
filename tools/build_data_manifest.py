# -*- coding: utf-8 -*-
"""Generate reports/DATA_MANIFEST.md — the inventory of raw/intermediate inputs the analysis reads.

Why this file exists. Scripts under src/ and tools/ read ~120
distinct paths under data/. That tree is large (genome FASTAs, per-species window files, and
the 8,192-bp score parquets), so it cannot travel inside Additional file 2. It was flagged that the deposit therefore contained scripts that abort on missing inputs with no
statement of what the inputs are or where they come from. This manifest is that statement: it lists
every data/ path the deposited code references, records the total size, and points to the public
sources and to the recompute layers (JSON + small parquets, both deposited) from which every
PUBLISHED number is re-derivable WITHOUT the raw data.

The path list is grepped from the code so it cannot drift from what the scripts actually open.

    python tools/build_data_manifest.py   -> reports/DATA_MANIFEST.md
"""
import glob
import io
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = "reports/DATA_MANIFEST.md"
# the character class included '.', so a path already ending in an extension kept
# matching through the next one and the pig genome FASTA index was listed with its extension
# doubled -- a path no script opens. The manifest's own header promises the list cannot drift from
# what the code opens, so an over-match falsifies that promise. Directory segments may not contain
# The alternation is also the manifest's blind spot: an input whose extension is missing from it
# is invisible to a file whose header claims the path list cannot drift from what the scripts
# open. Five were, and each is a real input: the splicing panel's source table (.xlsx), the
# BRCA1 structure (.pdb), the eQTL embeddings (.npz), the liftover chain (.chain) and the
# snpEff build config (.config).
# a dot; only the final extension does, and the alternation is longest-first so the compressed VCF
# suffix wins over the bare one.
#
# Note for anyone editing the comments in this file: the inventory is grepped from tools/**/*.py,
# so a literal path written in a comment here is scanned as if a script opened it. Describe paths
# in words rather than writing them out.
# The optional leading directory is load-bearing. The splicing and ClinVar scorers open their
# inputs from a path that carries one more component than the raw tree does, and a pattern anchored
# on the bare inventory directory recorded them one level too shallow -- non-existent paths, in the
# one document whose job is to tell a reader where an input lives.
PAT = re.compile(r"(?:analyses/)?data/(?:[A-Za-z0-9_%{}-]+/)*[A-Za-z0-9_%{}.-]+?"
                 r"\.(?:vcf\.gz|parquet|csv|vcf|fai|fa|bw|bed|txt|tsv|json|npy|xlsx|npz|pdb|chain|config)(?![A-Za-z0-9.])")


def collect_paths():
    files = []
    # analyses/scripts/ is deposited and reads its own inputs -- the splicing panel, the
    # ClinVar windows, the strand ladder. Leaving it out made the manifest silently incomplete
    # for three of the paper's results.
    for pat in ("src/**/*.py", "tools/**/*.py", "glmtrust/**/*.py",
                "analyses/scripts/**/*.py"):
        files.extend(glob.glob(pat, recursive=True))
    found = set()
    for fp in files:
        txt = io.open(fp, encoding="utf-8", errors="ignore").read()
        for m in PAT.finditer(txt):
            found.add(m.group(0))
    # normalise format placeholders (%s, {sp}) to <species> so the list is a template inventory
    norm = set()
    for p in found:
        norm.add(re.sub(r"%s|%d|\{[a-z_]+\}", "<var>", p))
    return sorted(norm)


def _data_dependent_counts():
    """(scripts under src/ that reference a data/ path, total scripts under src/).

    The manifest named three scripts as the exception to "reproduces without
    touching data/". The exception is the majority, so the count is measured rather than asserted.

    this counted every file in the WORKING tree, and the code deposit excludes
    the superseded figure modules, so the shipped manifest quoted a denominator larger than the
    tree a reader holds -- in the one file whose header says it cannot drift. Count what ships,
    using the deposit's own exclusion rule.
    """
    srcs = [f for f in glob.glob("src/**/*.py", recursive=True)
            if "_superseded_" not in f.replace("\\", "/")]
    dep = sum(1 for f in srcs
              if re.search(r"[\"']data/", io.open(f, encoding="utf-8", errors="ignore").read()))
    return dep, len(srcs)


def group(paths):
    buckets = {}
    for p in paths:
        parts = p.split("/")
        top = "/".join(parts[:2]) + "/"        # e.g. data/processed/
        buckets.setdefault(top, []).append(p)
    return buckets


def main():
    paths = collect_paths()
    buckets = group(paths)
    W = [].append
    out = []
    out.append("# Data manifest — inputs required to regenerate the analysis from scratch\n")
    out.append(
        "*This file, its path inventory and the prose around it, is written by "
        "`tools/build_data_manifest.py`, which greps the deposited code, so the list cannot drift "
        "from what the scripts actually open. It inventories what the code references; not every "
        "listed path is produced by the published analysis.*\n")
    out.append(
        "\n## Why the raw data is not in the deposit\n\n"
        "The `data/` tree for this study is not deposited — nine genome FASTAs, per-species "
        "variant-window files, and the 8,192-bp Evo 2 score parquets — which exceeds any reasonable "
        "supplementary-file limit. It is therefore **not** included in Additional file 2.\n\n"
        "**This does not block reproduction of the published numbers.** Every value in the "
        "manuscript, tables and figures is held in the recompute layer — the `reports/*.json` "
        "files and the small `reports/*.parquet` files, both shipped in Additional file 2 — which the "
        "table and figure builders read directly. Running `src/ccs/build_tables.py` against those "
        "deposited artifacts regenerates the deposit's own tables report, `reports/tables.md`, without "
        "touching `data/`, while `src/ccs/build_supplementary.py` does not produce the submitted "
        "supplementary table set, which ships as the export `reports/supplementary_tables.md`; `tools/verify_from_data.py`, by contrast, re-derives published values "
        "from the raw score and label files under `data/`. The `fig*_stats.py` scripts are **not** in that class: "
        "they read "
        "per-variant panels under `data/` and exit non-zero without it. So do most deposited "
        "analysis scripts — %d of the %d under `src/` reference a `data/` path, including "
        "`build_tost_equivalence.py`, `build_phylop_crosscheck.py` and the `fig*_stats.py` "
        "family. What they emit — the recompute JSONs and small parquets — IS deposited, so "
        "every published value stays checkable where the script that produced it cannot be re-run "
        "here. The raw tree is needed to re-run Evo 2 scoring end-to-end (a GPU workload) and to "
        "rebuild Additional file 1's Figures S3, S5 and S6 (see FIGURES.md). Figures 9 and 10 "
        "and Additional file 1's Figures S1, S2, S4 and S7 rebuild from deposited artefacts, "
        "and Figures 1 to 8 and Additional file 1's Figures S8 to S12 are built in Additional file 3 and "
        "need nothing from `data/`.\n"
        % _data_dependent_counts())
    # Three deposited artefacts are read by deposited scripts but written by
    # none, so a reproducibility referee finds a hole in the provenance chain and cannot tell a
    # deliberate deposited input from a leftover. The numbers themselves all reproduce; only the
    # origin was undocumented. Stated here because this file is where a reader looks for provenance.
    out.append(
        "\n## Deposited `reports/` artefacts with no in-deposit writer\n\n"
        "Three artefacts in the deposit are inputs rather than outputs: deposited scripts read "
        "them, but no deposited script writes them. They are listed here so the provenance chain "
        "has no unexplained node.\n\n"
        "- `reports/_recon_pervariant_trust.parquet` — per-variant trust-layer reconstruction "
        "(species, variant_id, label, score and the four calibration posteriors) for the 8,192-bp "
        "panel. It is a deposited **input**, consumed by `build_emin.py`, "
        "`build_fig3_consequence.py`, `build_locus_clustered_ci.py`, `build_ablation_fp8.py`, "
        "`build_raw_ece.py`, `fig4_trust.py` and `analyses/scripts/calibration_error.py`. "
        "Produced by the trust-layer build from `data/processed/scores_cloud/`, which is not "
        "deposited; the file is shipped so those consumers run without it.\n"
        "- `reports/ablation_fp8.parquet` — the FP8 batch-size control's per-variant scores at both "
        "batch sizes (variant_id, delta_b4, delta_b32; 600 rows). `build_ablation_fp8.py` reads it "
        "and regenerates `reports/ablation_fp8.json` from it, joining labels from "
        "`_recon_pervariant_trust.parquet`.\n"
        "- `reports/fig4_leak.json` — the selective-layer confusion counts (missed positives, "
        "false alarms, caught, reaching the clinician); Additional file 1 prints the first three "
        "in Table S28, the refusal total in Table S27's notes and the share of errors refused "
        "(35.4%) in Note S52. Every stored count "
        "recomputes from the deposited `reports/fig4_pervariant.parquet` at the 15% cut.\n")
    out.append(
        "\n## Public sources of the raw inputs\n\n"
        "Exact releases and assembly accessions are in the manuscript Methods and Availability of "
        "data and materials; the primary sources are:\n\n"
        "- **Disease variants (positives):** OMIA (Online Mendelian Inheritance in Animals, "
        "<https://omia.org>) for the eight non-human species; NCBI ClinVar "
        "(<https://www.ncbi.nlm.nih.gov/clinvar/>, release dated in Methods) for the human panel.\n"
        "- **Negatives and genome assemblies:** population variants from Ensembl Variation for seven non-human species, "
        "the PigGTEx v0 genotype panel for pig, and ClinVar Benign/Likely_benign records for human; the Ensembl per-species releases and GenBank assembly accessions are listed in Methods.\n"
        "- **Regulatory (eQTL) panel:** PigGTEx fine-mapped causal *cis*-eQTLs (SuSiE-inf); their allele "
        "frequencies from PigGTEx's molQTL genotype release (Science Data Bank, doi:10.57760/sciencedb.08089), "
        "which `analyses/scripts/build_piggtex_maf.py` identifies by URL and MD5.\n"
        "- **Conservation tracks:** GERP from the per-species Ensembl bigWig tracks (see `src/ccs/query_gerp.py`); phyloP from UCSC where a genuine track exists (human hg38 phyloP100way, chicken galGal6 phyloP77way) and, for cattle (bosTau9), from Zenodo record 13332541 (Roslin) over a Cactus 241-way alignment — a different alignment and species set, described in Additional file 1: Table S8.\n"
        "- **Label-free panels:** the cattle population allele-frequency panel and the pilot bat "
        "cohort VCF described in Methods.\n")
    out.append("\n## Path inventory (%d distinct templates; `<var>` = species or sample)\n"
                % len(paths))
    labels = {
        "data/raw/": "Raw downloads — genome FASTAs, OMIA/ClinVar truth sets, source VCFs.",
        "data/external/": "Third-party tracks and panels obtained externally.",
        "data/interim/": "Intermediate build products — variant windows, candidate tables, pilots.",
        "data/processed/": "Scored outputs — Evo 2 / NT / conservation scores per panel.",
    }
    for top in sorted(buckets):
        out.append("\n### `%s` — %s\n" % (top, labels.get(top, "")))
        for p in buckets[top]:
            out.append("- `%s`" % p)
    out.append("\n")
    io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print("wrote %s (%d path templates across %d groups)" % (OUT, len(paths), len(buckets)))


if __name__ == "__main__":
    main()
