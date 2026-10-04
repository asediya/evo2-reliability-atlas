# Figure-to-script manifest

Each figure is rendered by exactly one **live builder**, and no number is typed into a figure script by
hand.

**Figures 1 to 8 and Additional file 1's Figure S8 are built from Additional file 3** rather than
from this repository, and their builders ship there, in its `figures/` directory. Five of them have a
prep script beside them that recomputes what they draw from the deposited per-variant panels, and
Figure 3 draws what Figure 2's prep script writes; Figures 5 and S8 read Additional file 3's tables
directly, and Figure 1 draws a constructed toy panel and reads no data. They are listed below for
completeness but they are not in `src/ccs/`, and they need nothing from `data/`.

The builders here read what the table below says. Three run from the archive as shipped
(`fig3_rebuild`, `fig4_trust` and `figS1_missing_panels`), and three more, `fig7_atlas` (Figure 9),
`fig8_readout` (Figure 10) and `fig2_split` (Additional file 1's Figure S7), need Additional file
3's tables in addition, `tables/fig1_atlas_pervariant.parquet`, `table2_cluster_aware_ci.tsv`,
`refseq_chromosome_map.csv` and, for Figure 10, `atlas_gerp_windows.parquet` — set
`CCS_TABLES` to point at them. They read deposited recompute-layer artefacts plus the image files
under `assets/`, all of which are the PhyloPic silhouettes, loaded by `fig2_style.load_silhouette`.
**Additional file 1's Figures S3, S5 and S6 additionally read the `data/` tree** --
`data/interim/brca1_windows.parquet`, `data/processed/scores/selection_evo2_40b.parquet` and
`data/processed/conservation/*_gerp.parquet` respectively — which is not deposited
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
place at the same width and height, while the deposited panels are 706x390 and 707x760, so a crop
to the render's content box and a downscale of about 3.9x (BRCT) and 4.0x (RING), made by hand,
sit between them. `src/ccs/rebuild_brca1_structure_html.py` regenerates the HTML step and
reproduces the archived pages byte-identically when run with --ramp RdBu_r. The browser save is
manual and the page loads 3Dmol.js unversioned from a CDN, so that step is reproducible in method
rather than in bytes: a re-render matched BRCT within 0.1% on aspect and drew RING 1.5% wider.

This manifest exists because the tree carries several exploratory `fig*` scripts; the ones in the
table below are authoritative for the figures they build. The other `src/ccs/fig*` modules are not
all exploratory: `fig2_data`, `fig3_data`, `fig3_stats`, `fig4_asym`, `fig4_matrix`,
`fig4_reconcile`, `fig4_trust`, `fig5_stats` and `fig6_stats` are sources of published numbers,
`fig7_atlas.py` and `fig8_readout.py` read the atlas through `fig2_measure` and `fig2_style` and draw with
`style_main`, the style module shared with Figures 1 to 8, and `fig2_split.py` draws Figure S7 from
`fig2_measure`. `fig2_hero`, `fig2_bc` and `fig2_defgh` draw panels that no submitted figure uses.

Run the builders from the deposit root, either way:

```bash
python -m src.ccs.fig5_final        # or: python src/ccs/fig5_final.py
```

Both work. The builders import sibling modules (`fig5_final` imports `fig5_stats`) but fall back to
a flat import when run as a file, so neither form raises `ModuleNotFoundError`.

## Two numbering systems

Read this before matching any filename to any figure.

1. **Project numbering** — what the builder module is called and what it writes into
   `reports/figures/`. `fig3_rebuild.py` emits `Figure3_blindspot.pdf`, which reads as "Figure 3"
   and is Additional file 1's Figure S2, not Figure 3 of the submission.
2. **Submission numbering** — Figures 1 to 10 and Additional file 1's Figures S1 to S8.

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

**Figures 9 and 10 each have their own builder, and Figure S7 comes from `fig2_split.py`.**
`fig7_atlas.py` draws the three panels of Figure 9 and `fig8_readout.py` the four of Figure 10. Both read the
atlas through `fig2_measure.py` (its loader, the AUROC, the Table 2 intervals, the per-species skill table
and, for Figure 10, the locus-bootstrap head-to-head and the t interval) and the species tree through
`fig2_style.py`, Figure 9 its silhouettes as well, and both draw with `style_main.py`. Figure S7's
two panels are `fig2_measure.py`'s `panel_c` and `panel_d`, placed by `fig2_split.py`. `fig2_hero.py`'s `build_dotplot`,
`fig2_bc.py`'s `panel_B`, `fig2_measure.py`'s `panel_a`, `panel_b`, `panel_headtohead` and
`panel_composition`, and `fig2_defgh.py`'s `panel_D` to `panel_H` are called by no builder in the table.
All three builders read Additional file 3's tables, located as `fig2_measure._deposit_table()`
describes.

Of the rows whose builder output carries a number, every one but Figures 2, S1 and S7 disagrees
with it, which is why the table is written down rather than inferred: `fig6_final.py` writes `Figure6_curator.pdf`, which is Additional file 1's
Figure S3.

**`tools/stage_figures.py` applies the map in the table above for the figures built here.**
It holds the builder-to-deliverable mapping and renames builder outputs to the staged names. Its
`--check` mode reports any deliverable that is behind its builder without changing anything. It does
not handle Figures 1 to 8 or Figure S8: those are built in Additional file 3 and copied in, and that
archive's `figures/README.md` says how.

```bash
python tools/stage_figures.py       # renames builder outputs to the staged names
                                    # NEEDS data/: it rebuilds rather than renames, so from the
                                    # archive alone it stops at a missing conservation parquet
```

**`src/ccs/fig1.py` is not the builder for any submitted figure.**
Neither is **`build_reach_penalty_figure.py` at this archive's root**: it is exploratory, it is
matched by no row and by no `src/ccs/fig*` exclusion above because it is not under `src/ccs/`,
and running it writes `Figure_reach_penalty.pdf`, `.png` and a 49-row `Table_S3_reach_penalty.tsv`
into the root — that TSV is not Additional file 1's Table S3, which is the baseline suite. Nothing
in the submission uses either file. `src/ccs/fig1.py` is an exploratory module and it writes
`reports/figures/Figure1.pdf` (canvas `figsize=(7.2, 7.4)` inches, 182.9 x 188.0 mm before the
tight-bbox trim, 175.3 x 186.9 mm after it). Figure 6 is built by `build_fig1.py` in Additional
file 3, which writes a file of the same name, `Figure1.pdf`; the collision is a trap, not a pointer. What
`fig5_final.py` builds is Additional file 1's Figure S5.

**Trust-layer figure note.** `fig4_asym_stats.py` and `fig4_matrix_stats.py` compute the
class-asymmetry plane and the operating-point intervals, emitting `reports/fig4_asym.json` and
`fig4_matrix.json`; `reports/fig4_leak.json`, the leakage control, is a deposited input with no
in-deposit writer (`reports/DATA_MANIFEST.md`). Those JSONs are quoted in the text rather than opened
by the renderer; `fig4_asym.py`, `fig4_matrix.py` and `fig4_leak.py` draw from them, and no submitted
figure uses their drawings. `fig4_reconcile.py` and
`fig4_field_stats.py` emit the artefacts the renderer does open. `fig4_trust.py` is the *renderer*.
`fig4_rich.py` is exploratory and not the submitted panel.

**Figure development.** The design-and-screening records from the figures' development
(`fig*_screen*.json`, `fig*_panelA_*.json`, `fig*_fleet_result.json`, `fig2_rework_spec.json`) are
not authoritative for the submitted figures and are not part of the code deposit.

## Canvas geometry — 170 × 173 mm

Every figure is one page **170 mm wide** (`W_MM` in `src/ccs/style_gb.py` and, for the ten main
figures, in `src/ccs/style_main.py`), and its height leaves
room for a legend set beneath the graphic within a 225 mm page: each legend is wrapped with real
Arial advances at 8 pt on 9.6 pt leading across the full 481.89 pt (170 mm) measure and the line
count multiplied by the leading. At 14 lines a legend sets to 47.4 mm, which caps the graphic at
177.6 mm; `H_MM_MAX`, **173 mm**, is that cap with a margin, and the builders here keep to it. The
main-figure legends as submitted set to six to nine lines (20.3 to 30.5 mm; each points to its
extended legend in Additional file 1's Note S59), and Additional file 1's figure legends to 9 to 13
lines (30.5 to 44.0 mm). Every main plate is at most 173.0 mm tall, which `style_main.Canvas` asserts,
and Additional file 1's run from 84.0 mm (Figure S1) to 173.0 mm, all under
177.6 mm; no graphic and its legend together exceed 217 mm.

### Layout rules that hold across a canvas change

A layout written against one canvas height breaks at another in three recurring ways, each a
distance authored in points against a box measured in millimetres, or the reverse. All three are
worth checking before any future canvas change:

- **Absolute millimetres from the top edge** (`fig3_rebuild.py`'s `Box`,
  `fig_brca1_composite.py`'s `TOP_PT`/`BOT_PT`, `fig6_final.py`'s band budget).
  These do not move when the page gets shorter, so the bottom of the stack leaves the paper.
- **A gap expressed as a fraction of a box** (`hspace`, a y-position at `0.68` of an
  axes, a label stagger at `1.07`/`1.21` of the ylim). The gap shrinks with the box
  while the type in it does not, so annotations close up and overlap.
- **A block pinned to a figure fraction sitting next to one pinned to an axes.**
  The two move by different amounts and collide.

Where a distance exists to clear a glyph, it is expressed in points and converted to the local
box's fraction at draw time. Where a band budget is authored in millimetres, the file says so at
the top.

### Colour: measure the greyscale and the dichromacies, do not eyeball them

Every pair of colours that carries a distinction is checked by WCAG relative luminance and under
simulated deuteranopia and protanopia; the two failure modes are independent.

- **RING vs BRCT** (Figure S6's domain track and key, the headers of its four structure panels, and
  its scatter) are navy `#0A2A5A` against ochre `#9F6B02`: 3.07:1 apart in greyscale, and 0.70,
  0.62 and 0.71 apart under the deuteranopia, protanopia and tritanopia matrices. Both clear WCAG
  as type (14.1:1 and 4.59:1 on white) and as marks, and the ochre sits 3.92:1 from the `#EDEDED`
  of unannotated positions, so the domain track reads as three distinct greys.
- **The deleteriousness ramp.** `RdBu_r` is ColorBrewer-safe for colour vision, but its two
  extremes both have relative luminance 0.030 (contrast 1.00), so in greyscale it cannot carry the
  quilt. `style_gb.TOL_DEL_NODES` keeps the diverging structure, the blue/red convention and the
  lightest anchor at t = 0.5 (so it lands on the TwoSlopeNorm centre), and ends the tolerated arm
  at a medium blue: ends contrast **4.00**, centre-vs-tolerated 3.23, centre-vs-deleterious 12.89,
  and dichromatic separation 0.675 and 0.705.
- **The three pair-coverage regimes** of all ten main figures — "answers the whole panel" and "q
  below one half" are the two poles of this paper's central claim — are navy `#23426B`, steel
  `#6F93BD` and crimson `#B23A48` in `style_main.py`. The two poles measure only 1.74:1 apart in
  greyscale, so every mark that carries a regime also carries its shape — circle, square,
  triangle — and every key shows each regime's own marker.
- **Additional file 1's Figure S5c supervised matrix** is sequential and monotone in lightness over
  AUROC 0.45–0.95; only one of its 24 cells is below chance. The stop positions are solved against
  the 24 values the panel holds: black-or-white ink on a mid-tone bottoms out at 4.16:1 (at cell
  luminance 0.202), and the ramp crosses that band where no datum sits. Worst numeral contrast
  **5.92:1**. Below-chance cells keep non-colour marks: a failure-coloured outline and a bold
  numeral.
- **Figure 8a's paired bars** separate the benchmark from the deployment background by lightness
  as well as hue, grey (L* 58) against blue (L* 46), and its dot matrix marks a declining predictor
  with a pale dot against the dark dot of one that answers. Figure 8b's reach ramp keeps its darker blue for the
  silent end, as in Figure 2.
- **Figure 9a's three markers** differ in shape as well as colour (a filled circle, a vertical bar
  and a cross), so Evo 2 and the two baselines separate in greyscale.
- **Figure S7b's catalogued alleles** are drawn over a log-count hexbin whose ramp is truncated to
  [0.08, 0.45], so the allele red never sits inside the hexbin's grey range: worst allele-vs-cell
  contrast 2.05, with a 2.26 range left to the hexbin.
- **Additional file 1's Figure S3a residual surface** is genuinely diverging about zero, where the
  centre is meaningful, so it keeps a diverging ramp, and it carries the sign redundantly as a hatch
  on every negative cell. (`hero_preview()` in the same module draws an unhatched version, but
  nothing calls it.)
- **Figure 6a's AUROC discs** use `style_main.AUC_CMAP` over 0.5 to 1.0, a sequential ramp that
  `style_main.py` checks to be monotone in lightness when it is imported, anchored at chance; a disc's
  area, not its colour, carries reach.
- **Figure 9b's densities** encode class by direction as well as colour, negatives above each
  species' line and positives below, and every positive is also drawn as a tick.

**The nine-species palette is a deliberate exception, and it is checked rather than assumed.**
Nine categorical hues cannot be made dichromat-safe — Okabe-Ito, the standard, stops at eight --
and nine cannot be separated in greyscale by hue at all. Measured, the worst pairs are dog vs
horse (0.136 deuteranopic, 0.110 protanopic, greyscale contrast 1.07) and sheep vs pig (greyscale
1.01); eight of the thirty-six pairs are at risk on one axis or another. That is acceptable here
only because species identity is never carried by colour alone: every figure that uses the palette
prints all nine species names on the canvas beside the marks they key — verified span-by-span on
Additional file 1's Figures S3, S4, S5 and S7 (Figure S2 is a single-species panel).
Colour reinforces a label; it does not replace one. If a future panel uses the palette without
naming the species, the fix is the label, not a new palette.

The four BRCA1 structure renders are recoloured at the pixel level by
`tools/recolour_brca1_structures.py`, which fits each pixel as `k * RdBu_r(t)` for a shade `k` and
a ramp position `t` and re-emits `k * TOL_DEL(t)`. The renders are shaded cartoons, so a flat
nearest-colour match catches only 16-42 % of content; the multiplicative fit reaches a median
residual of 0.5-0.7 sRGB units and remaps 95-98 %. It also asserts that autocrop's bounding box is
unchanged, because the renders sit on an opaque white background 15.1 units from RdBu_r's centre.

### The gate

`tools/pdfcheck.py` reports page size, minimum type size, span collisions at zero
area tolerance, off-page spans, and **fused numeric runs**. The last column is not
redundant. A collision audit compares Text objects pairwise, so it cannot see two
failures:

- Text that is **entirely** off the page does not appear in PyMuPDF's extraction at all.
- Labels that abut with no gap are extracted as **one span**, which collides with nothing
  (seven x-tick labels read as the single string `0.700.750.800.850.900.951.00`).

The fused-run column catches the second. The first is outside what any single-file check can
see; comparing the character multiset of each new figure against the previous build catches it,
and that comparison is the check to run after any geometry change: a figure that silently loses
text still passes every other test.
