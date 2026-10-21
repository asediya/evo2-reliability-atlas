# Reproducing

What regenerates from this repository alone, what needs the raw data tree, and where the guarantees
stop. The short version is in the [README](../README.md); this is the contract.

## Environment

Python 3.12 for the analysis layer. `glmtrust` declares Python 3.9 or later and was tested under 3.12.10.

```bash
pip install numpy polars pandas scipy scikit-learn pyarrow matplotlib pymupdf
pip install './glmtrust[io]'   # the package itself; the io extra pulls polars, which the benchmark needs
```

That is everything the recompute layer and the clone-runnable figure builders need. `pymupdf`
is used by `fig2_split`, which draws Additional file 1's Figure S7, and by
`style_main`, the style
module of the main figures, which reads each PDF it writes back to check the page. The
submitted PDFs were written under matplotlib 3.11.2; a rebuild under another matplotlib matches
them in content and geometry but not pixel for pixel.

Scripts that read the undeposited `data/` tree need more, and none of it is required for anything
above: `duckdb` and `pybigtools` for the panel and phyloP builders, `pyBigWig` for the older GERP
and phyloP queries, `pyfaidx` for window extraction,
`biopython` and `freesasa` for the BRCA1 structure figure, `joblib` where a script parallelises, `requests` for the BRCA1 panel download,
and `torch`, `transformers`, `fair-esm`, `evo2` and `vortex` on a GPU scoring host.

`requirements.txt` records the environment that produced every published number. Install from the
commands above rather than from its pins: `numpy==2.5.1` needs Python 3.12, and `torch`'s `+cu124`
build is served only by download.pytorch.org. `torch` is needed only on a GPU scoring host.

Run scripts from the archive root. Some locate it from their own path and honour `CCS_ROOT`; many
open `reports/` relative to the working directory.

## Three tiers of reproducibility

| tier | what it needs | what it gives |
|---|---|---|
| **Package benchmark** | `pip install './glmtrust[io]'`, nothing else | the 8,192-bp trust-layer numbers, from a fixture that ships with the package |
| **Recompute layer** | this repository | the deposit's own tables D1 to D4 (Additional file 3 recomputes the published tables) and the figures listed below |
| **From raw data** | the raw `data/` tree, and a GPU for scoring | everything, including the Evo 2 scores themselves |

Most readers want the first two. Neither needs the raw tree.

### Tier 1: the package benchmark

```bash
pip install './glmtrust[io]'
python glmtrust/benchmarks/reproduce_paper_trust_layer.py
```

The `[io]` extra is not optional for this command. polars sits in that extra rather than in the
core dependencies, and the benchmark reads the study parquets through it, so `pip install
./glmtrust` without the extra exits 1 with "this benchmark needs polars to read the study
parquets". The quotes matter in zsh, which treats the brackets as a glob.

Exits zero when the order-invariant figures match within tolerance (415 errors, macro ECE 0.0209) and the published pooled capture 0.6096 and pooled lift 4.0676 lie inside their row-order ranges; the macro lift, 3.9343, is printed beside them. The 11,109 variant-level
scores it needs travel with the package as a 119 KiB fixture.

### Tier 2: the recompute layer

Every figure has a recompute layer. Most figure modules draw only from the JSON it emits; `figS1_missing_panels`
reads `reports/fig4_pervariant.parquet`, `fig7_atlas` and `fig8_readout` compute their AUROCs and locus bootstraps at
build time from Additional file 3's per-variant atlas, and `fig_brca1_composite` computes its AUROC from `data/`.
No module types a float by hand. The tables read the same JSONs and assert their headline values, so
drift fails the build instead of reaching the paper.

```bash
python -m src.ccs.build_tables      # -> reports/tables.{json,md}
python tools/verify_from_data.py    # the count is printed by the run itself; from a clone,
                                    # 23 pass, none of them from raw data, and the summary
                                    # prints the evidence class of every check
```

`build_tables.py` runs from a clone and prints `8 canonical values asserted, all match`.
`verify_from_data.py` reads the raw score and label files, so from a clone it re-derives the 23
checks it can, lists each of the other sections with the input it could not reach, and exits **3** — this archive's
"stopped at an undeposited path" code. That is the intended behaviour: it is the one check that recomputes from raw data instead of
from the deposited layer, so it cannot be made to pass without that data. It is not a broken
script. All ten of its
sections appear in the summary table whether or not they could run, so a section that was skipped
cannot be mistaken for one that passed.

In the numbering of this submission: Figures 9 and 10 and Additional file 1's Figures S1, S2, S4 and S7
rebuild from deposited artefacts. Additional file 1's Figures S3, S5 and S6 additionally
read `data/` and exit non-zero without it; Figures 1 to 8 and Additional file 1's Figures S8 to S12 are
built in Additional file 3 and need nothing from `data/`. `FIGURES.md` records which builder is in which
tier and maps each builder-output filename to its figure number; its last column is this submission's
numbering.

Three of these six run from a clone alone; `fig7_atlas`, `fig8_readout` and `fig2_split`
additionally need Additional file 3's tables: `tables/fig1_atlas_pervariant.parquet`, `table2_cluster_aware_ci.tsv`
and `refseq_chromosome_map.csv`, and `fig8_readout` also `atlas_gerp_windows.parquet`, none of which ships
in Additional file 2. Point `CCS_TABLES` at that folder, as the command below shows. Without it those three
exit 1, naming the file they want; the other three exit 0.

```bash
export CCS_TABLES=<Additional file 3>/tables
python -m src.ccs.fig7_atlas              # Figure 9
python -m src.ccs.fig8_readout            # Figure 10
python -m src.ccs.fig2_split              # Additional file 1: Figure S7 (needs pymupdf)
python -m src.ccs.fig3_rebuild            # Additional file 1: Figure S2
python -m src.ccs.fig4_trust              # Additional file 1: Figure S4
python -m src.ccs.figS1_missing_panels    # Additional file 1: Figure S1
```

Building the figures writes `reports/figures/*.pdf` and `*.png`. Those are build outputs, not
shipped files, so they are in no skipped directory and carry no skipped suffix, and running
`python3 check_manifest.py` afterwards reports them as **undeclared** and exits 1. That is the
integrity check working correctly on files the archive never shipped, not a packaging fault:
delete them, or run the check on a fresh extract.

These three read `data/` and exit non-zero without it. The builder-output name is not the figure
number, which is why `FIGURES.md` exists:

```bash
python -m src.ccs.fig_brca1_composite     # Additional file 1: Figure S6
python -m src.ccs.fig5_final              # Additional file 1: Figure S5
python -m src.ccs.fig6_final              # Additional file 1: Figure S3
```

`analyses/scripts/trust_layer_8192_fair.py` rebuilds the trust layer at the 8,192-bp readout against an
unpenalised two-parameter sigmoid, checks every value the manuscript and Additional file 1 print from
that rebuild, and writes `reports/trust_layer_8192_fair.json`. It reads the same Additional file 3 table
as `fig2_split`, located the same way. It exits 0 only when every published value agrees and every
deposited value it rebuilds is reproduced, and it never rewrites an unchanged record, so it also runs
from the read-only archive:

```bash
CCS_TABLES=<Additional file 3>/tables \
python analyses/scripts/trust_layer_8192_fair.py
```

### Tier 3: from raw data

`reports/DATA_MANIFEST.md` inventories every `data/` path the code opens, with the public source of
each; not every listed path is produced by the published analysis.

`analyses/scripts/maf_stratified_auroc.py` recomputes the allele-frequency-stratified and -matched eQTL
AUROCs of Additional file 1's Note S33 and checks every value printed from them. It reads the allele
frequencies that `analyses/scripts/build_piggtex_maf.py` computes into the undeposited `data/interim/`
from PigGTEx's genotype release, a 0.9 GB download whose URL and MD5 its docstring gives (extraction takes
about 7.1 GB). Without them it checks the printed values against the deposited
`analyses/results/eqtl_maf_matched_auroc.json` and exits 3; with them it never rewrites an unchanged
record, so it also runs from the read-only archive:

```bash
python analyses/scripts/build_piggtex_maf.py --geno PigGTEx_v0.ALL_Tissues_Genotype.tar.xz
python analyses/scripts/maf_stratified_auroc.py
```

Re-deriving the Evo 2 scores needs a GPU and the Evo 2 scoring stack; `container/` holds the recipe of
the container they were produced in.

## The recompute layer ships read-only

Every `reports/*.json`, `reports/*.parquet` and `analyses/results/*.json` in the
distributed archive is mode `0o444`.

Most deposited builders read `data/`, write a `reports/` artefact and carry no exit guard. Running
one without the raw tree would overwrite the very file the paper's numbers are checked against,
silently and at exit 0. The read-only bit turns that into a loud `PermissionError`.

To regenerate anything, take a writable copy first:

```bash
cp -r . /tmp/ccs && chmod -R u+w /tmp/ccs && cd /tmp/ccs
```

Without that step `python -m src.ccs.build_tables` prints `8 canonical values asserted, all match`
and then exits 1 on `PermissionError: reports/tables.json`. The assertion passing is the real
result; the write is what fails. From a writable copy the same command exits 0 and regenerates
`tables.json` byte-identically to the deposited file.

## What certifies the numbers

The numbers are certified by three things that run here:

- `build_tables.py`'s canonical-assertion gate, which fails the build on drift
- `tools/verify_from_data.py`, which re-derives from raw score and label files the number of
  values its own roll-up prints, and
  imports nothing from the analysis pipeline
- per-figure regeneration from the deposited JSONs

## The traceability table

`reports/traceability_table.json` is an artefact map: it gives the artefact and key that carry each of 211 values
from the analysis. Its section labels and printed forms are those of the reading map behind it, and some of its
values are printed in this submission in another form or not at all (17.6% is printed as 17.64%, for example), so
it ties values to artefacts, not to the submitted text, and
`python tools/check_traceability.py` re-resolves every row against those artefacts from this archive alone. Its
`_meta` block records how the table was made: `source` names the reading map it was built from
(`logs/number_traceability.json`, a working note in the authors' tree that is not deposited), `n_ok` counts the
rows that resolved to a file and key and agree with the published value, and `n_unresolved` counts the 333 map
entries that named no deposited file and key, which the table leaves out rather than guesses.
