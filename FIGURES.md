# Figure-to-script manifest

Each figure is rendered by exactly one **live builder**, and no number is typed into a figure script by
hand.

**Figures 1 to 8 and Additional file 1's Figures S8 to S12 are built from Additional file 3** rather than
from this repository, and their builders ship there, in its `figures/` directory. Eight of them have a
prep script beside them that recomputes what they draw from the deposited per-variant panels, and
Figures 3, 5 and S11 draw what Figure 2's prep script writes, Figure 5 with Additional file 3's tables and
Figure S11 with Figure 4's verdicts; Figure S8 reads Additional file 3's tables directly, and Figure 1 draws a
constructed toy panel and reads no data. They are listed below for completeness but they are not in `src/ccs/`,
and they need nothing from `data/`. The values Figures 1 to 8 and S10 to S12 draw are tabulated by figure and
panel in Additional file 3's `source_data/`, whose README maps every figure, main and supplementary, to its
builder and the files it reads.

The builders here read what the table below says. Three run from the archive as shipped
(`fig3_rebuild`, `fig4_trust` and `figS1_missing_panels`), and three more, `fig7_atlas` (Figure 9),
`fig8_readout` (Figure 10) and `fig2_split` (Additional file 1's Figure S7), need Additional file
3's tables in addition, `tables/fig1_atlas_pervariant.parquet`, `table2_cluster_aware_ci.tsv`,
`refseq_chromosome_map.csv` and, for Figure 10, `atlas_gerp_windows.parquet` — set
`CCS_TABLES` to point at them. They read deposited recompute-layer artefacts plus the image files
under `assets/`, all of which are the PhyloPic silhouettes, loaded by `fig2_style.load_silhouette`.
**Additional file 1's Figures S3, S5 and S6 additionally read the `data/` tree** --
`data/processed/scores/selection_evo2_40b.parquet`, `data/processed/conservation/*_gerp.parquet` and
`data/interim/brca1_windows.parquet` respectively — which is not deposited
(`reports/DATA_MANIFEST.md` gives the public source of every path), so those three exit non-zero
from the archive alone. The published values they draw are the deposited ones; the `data/` reads
supply the underlying per-variant panels, not the summaries.

Figure S6 also needs the four structure renders `reports/figures/brca1_*.png`. They are made in
three steps: `src/ccs/make_brca1_structure.py` writes a 3Dmol HTML page and prints
"-> save PNG as ..."; the PNG is saved from a browser by hand; and
`tools/recolour_brca1_structures.py` then re-maps it from RdBu_r onto the published TOL_DEL ramp,
fitting each pixel as k * RdBu_r(t) and re-emitting k * TOL_DEL(t).

Two of the three artefacts ship: the model-1 structures under `data/raw/structures/` and the
recoloured `brca1_*.png` the builder reads, under `reports/figures/`. The RdBu_r renders, the
full-frame browser saves at 3200x3040 RGBA, which trip the recolour script's idempotence guard so a
second run cannot double-map them, are not deposited, nor are the 3Dmol HTML pages. The two are not a pixelwise pair: `recolour()` re-maps RGB in
place at the same width and height, while the deposited panels are 706x390 and 707x759 to 707x760, so a crop
to the render's content box and a downscale of about 3.9x (BRCT) and 4.0x (RING), made by hand,
sit between them. `src/ccs/rebuild_brca1_structure_html.py` regenerates the HTML step and
reproduces the archived pages byte-identically when run with --ramp RdBu_r. The browser save is
manual and the page loads 3Dmol.js unversioned from a CDN, so that step is reproducible in method
rather than in bytes: a re-render matched BRCT within 0.1% on aspect and drew RING 1.5% wider.

The table below is authoritative for the figures it maps. Among the other `src/ccs/fig*` modules, `fig2_data`, `fig3_data`, `fig3_stats`, `fig4_asym`, `fig4_matrix`,
`fig4_reconcile`, `fig4_trust`, `fig5_stats` and `fig6_stats` are sources of published numbers,
`fig7_atlas.py` and `fig8_readout.py` read the atlas through `fig2_measure` and `fig2_style` and draw with
`style_main`, the style module shared with Figures 1 to 8, and `fig2_split.py` draws Figure S7 from
`fig2_measure`.

Run the builders from the deposit root:

```bash
python -m src.ccs.fig5_final
```

Use the `-m` form. The builders import sibling modules (`fig5_final` imports `fig5_stats`); most fall back to
a flat import when run as a file, but `fig4_trust` and `fig5_final` also use relative imports that only resolve
under `-m`, and `tools/stage_figures.py` runs every builder that way.

## Two numbering systems

Read this before matching any filename to any figure.

1. **Project numbering** — what the builder module is called and what it writes into
   `reports/figures/`. `fig3_rebuild.py` emits `Figure3_blindspot.pdf`, which reads as "Figure 3"
   and is Additional file 1's Figure S2, not Figure 3 of the submission.
2. **Submission numbering** — Figures 1 to 10 and Additional file 1's Figures S1 to S12.

Three builder output names carry their figure's number (`Figure2.pdf`, `figS1_missing_panels.pdf`
and `FigureS7_measurement_supp.pdf`) and the rest do not, so the table below is the map.
`tools/stage_figures.py` stages builder outputs to the names in its last column and must be edited
with this table, never separately. It agrees with Additional file 1's Note S58 on which builders
need the undeposited `data/` tree (Figures S3, S5 and S6).

| Figure | Builder output | Live builder | Recompute layer it draws from | Staged as |
|---|---|---|---|---|
| Figure 1 | `FigureConcept.pdf` | **Additional file 3** `figures/build_concept.py` | none: a constructed toy panel of 16 variants and three predictors, whose verdicts the builder asserts | `fig1.pdf` |
| Figure 2 | `Figure2.pdf` | **Additional file 3** `figures/build_fig2.py` | `figures/dbnsfp_49.csv`, `figures/fig2_raster.json`, from `tables/dbnsfp_reach_panel.parquet` and `tables/dbnsfp_predictors.csv` | `fig2.pdf` |
| Figure 3 | `FigureBounds.pdf` | **Additional file 3** `figures/build_fig_bounds.py` | `figures/dbnsfp_49.csv`, `figures/fig2_raster.json`, from `tables/dbnsfp_reach_panel.parquet` and `tables/dbnsfp_predictors.csv` | `fig3.pdf` |
| Figure 4 | `FigureDecide.pdf` | **Additional file 3** `figures/build_fig_decide.py` | `figures/fig_decide.json`, from `figures/dbnsfp_49.csv`, `dbnsfp_49_missense.csv`, `dbnsfp_49_provenance.csv` and three of Additional file 3's recompute scripts | `fig4.pdf` |
| Figure 5 | `FigureFrontier.pdf` | **Additional file 3** `figures/build_fig_frontier.py` | `tables/dbnsfp_breakdown_points.parquet` and `tables/dbnsfp_predictors.csv`, with `figures/dbnsfp_49.csv`, `dbnsfp_49_missense.csv` and `dbnsfp_49_provenance.csv` | `fig5.pdf` |
| Figure 6 | `Figure1.pdf` | **Additional file 3** `figures/build_fig1.py` | `figures/clinvar5.csv`, `f1_class.csv`, `f1_rich.json`, from `tables/clinvar_reach_panel.parquet` | `fig6.pdf` |
| Figure 7 | `FigureEvidence.pdf` | **Additional file 3** `figures/build_fig_evidence.py` | `figures/fig_evidence.json`, from `tables/clinvar_reach_panel.parquet` and `tables/erepo_clingen_labels.parquet` | `fig7.pdf` |
| Figure 8 | `Figure4.pdf` | **Additional file 3** `figures/build_fig4.py` | `figures/splice.json`, from `tables/table1_benchmark.parquet` and `table1_background.parquet` | `fig8.pdf` |
| Figure 9 | `Figure7_atlas.pdf` | `src/ccs/fig7_atlas.py` (data through `fig2_measure.py` and `fig2_style.py`) | `reports/fig2b_summary.json`, `fig2_pervariant.parquet`, `compiled_results.parquet`, and **Additional file 3** `tables/fig1_atlas_pervariant.parquet`, `table2_cluster_aware_ci.tsv` and `refseq_chromosome_map.csv` | `fig9.pdf` |
| Figure 10 | `Figure8_readout.pdf` | `src/ccs/fig8_readout.py` (data through `fig2_measure.py` and `fig2_style.py`) | `reports/fig2_data.json`, `readout_effect_fullpanel.json`, `readout_headtohead.json`, and **Additional file 3** `tables/fig1_atlas_pervariant.parquet`, `table2_cluster_aware_ci.tsv`, `refseq_chromosome_map.csv` and `atlas_gerp_windows.parquet` | `fig10.pdf` |
| Additional file 1: Figure S1 | `figS1_missing_panels.pdf` | `src/ccs/figS1_missing_panels.py` | `reports/fig4_pervariant.parquet` | `Additional_file_1_FigureS1.pdf` |
| Additional file 1: Figure S2 | `Figure3_blindspot.pdf` | `src/ccs/fig3_rebuild.py` | `reports/fig3_stats.json`, `eqtl_pervariant.parquet`, `eqtl_cluster_bca.json`, `eqtl_blindspot_probe.json`, `eqtl_tss_baseline.json` | `Additional_file_1_FigureS2.pdf` |
| Additional file 1: Figure S3 | `Figure6_curator.pdf` | `src/ccs/fig6_final.py` | `reports/fig6_free.json` **+ `data/`** | `Additional_file_1_FigureS3.pdf` |
| Additional file 1: Figure S4 | `Figure4_trust.pdf` | `src/ccs/fig4_trust.py` | `reports/fig4_reconciliation.json`, `fig4_pervariant.parquet`, `fig4_field.json` | `Additional_file_1_FigureS4.pdf` |
| Additional file 1: Figure S5 | `Figure5_reach.pdf` | `src/ccs/fig5_final.py` | `reports/fig5_reach.json` **+ `data/`** | `Additional_file_1_FigureS5.pdf` |
| Additional file 1: Figure S6 | `Figure_BRCA1.pdf` | `src/ccs/fig_brca1_composite.py` | `reports/brca1_residue_scores.parquet` **+ `data/`** | `Additional_file_1_FigureS6.pdf` |
| Additional file 1: Figure S7 | `FigureS7_measurement_supp.pdf` | `src/ccs/fig2_split.py` (panels from `src/ccs/fig2_measure.py`) | `reports/fig2_data.json`, and **Additional file 3** `tables/fig1_atlas_pervariant.parquet` | `Additional_file_1_FigureS7.pdf` |
| Additional file 1: Figure S8 | `FigureTransfer.pdf` | **Additional file 3** `figures/build_fig_transfer.py` | `tables/atlas_missense_transfer.parquet`, joined to `tables/fig1_atlas_pervariant.parquet` by `scripts/recompute_atlas_strata.py` | `Additional_file_1_FigureS8.pdf` |
| Additional file 1: Figure S9 | `FigurePlate.pdf` | **Additional file 3** `figures/build_fig_plate.py` | `figures/dbnsfp_49_pairspace.json`, from `tables/dbnsfp_reach_panel.parquet` and `tables/dbnsfp_predictors.csv` by `figures/prep_fig_plate.py` | `Additional_file_1_FigureS9.pdf` |
| Additional file 1: Figure S10 | `FigureMap.pdf` | **Additional file 3** `figures/build_fig_map.py` | `figures/study_map.json`, from Additional files 4 to 6 and Additional file 3's tables by `figures/prep_fig_map.py`, with `figures/literature_audit.json` | `Additional_file_1_FigureS10.pdf` |
| Additional file 1: Figure S11 | `FigureRanks.pdf` | **Additional file 3** `figures/build_fig_ranks.py` | `figures/dbnsfp_49.csv`, `dbnsfp_49_missense.csv`, `fig2_raster.json` and `fig_decide.json` | `Additional_file_1_FigureS11.pdf` |
| Additional file 1: Figure S12 | `FigureAudit.pdf` | **Additional file 3** `figures/build_fig_audit.py` | `figures/literature_audit.json`, from `tables/literature_audit_papers.csv` and `literature_audit_predictors.csv` by `figures/prep_fig_audit.py` through `scripts/recompute_identification.py` | `Additional_file_1_FigureS12.pdf` |

**Figures 9 and 10 each have their own builder, and Figure S7 comes from `fig2_split.py`.**
`fig7_atlas.py` draws the three panels of Figure 9 and `fig8_readout.py` the four of Figure 10. Both read the
atlas through `fig2_measure.py` (its loader, the AUROC, the Table 2 intervals, the per-species skill table
and, for Figure 10, the locus-bootstrap head-to-head and the t interval) and the species tree through
`fig2_style.py`, Figure 9 its silhouettes as well, and both draw with `style_main.py`. Figure S7's
two panels are `fig2_measure.py`'s `panel_c` and `panel_d`, placed by `fig2_split.py`. `fig2_measure.py`'s `panel_a`, `panel_b`,
`panel_headtohead` and `panel_composition` are called by no builder in the table.
All three builders read Additional file 3's tables, located as `fig2_measure._deposit_table()`
describes.

Of the rows whose builder output carries a number, every one but Figures 2, S1 and S7 disagrees
with it, which is why the table is written down rather than inferred: `fig6_final.py` writes `Figure6_curator.pdf`, which is Additional file 1's
Figure S3.

**`tools/stage_figures.py` applies the map in the table above for the figures built here.**
It holds the builder-to-deliverable mapping and renames builder outputs to the staged names. Its
`--check` mode reports any deliverable that is behind its builder without changing anything. It does
not handle Figures 1 to 8 or Figures S8 to S12: those are built in Additional file 3 and copied in, and that
archive's `figures/README.md` says how.

```bash
python tools/stage_figures.py       # renames builder outputs to the staged names
                                    # NEEDS CCS_TABLES and data/: it rebuilds rather than renames, so
                                    # it stops at the first builder whose input is absent (with only
                                    # Additional file 3's tables, at a missing conservation parquet)
```

`fig5_final.py` builds Additional file 1's Figure S5.

**Trust-layer figure note.** `fig4_asym_stats.py` and `fig4_matrix_stats.py` compute the
class-asymmetry plane and the operating-point intervals, emitting `reports/fig4_asym.json` and
`fig4_matrix.json`; `reports/fig4_leak.json`, the leakage control, is a deposited input with no
in-deposit writer (`reports/DATA_MANIFEST.md`). Those JSONs are quoted in the text rather than opened
by the renderer; `fig4_asym.py`, `fig4_matrix.py` and `fig4_leak.py` draw from them, and no submitted
figure uses their drawings. `fig4_reconcile.py` and
`fig4_field_stats.py` emit the artefacts the renderer does open. `fig4_trust.py` is the *renderer*.
