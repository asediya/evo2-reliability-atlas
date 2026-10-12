# Data manifest — inputs required to regenerate the analysis from scratch

*This file, its path inventory and the prose around it, is written by `tools/build_data_manifest.py`, which greps the deposited code for paths under `data/` ending in a data-file extension, so the list follows what the scripts name rather than being typed by hand. It inventories what the code references; not every listed path is produced by the published analysis.*


## Why the raw data is not in the deposit

The `data/` tree for this study — nine genome FASTAs, per-species variant-window files, and the 8,192-bp Evo 2 score parquets — exceeds any reasonable supplementary-file limit, so it is **not** included in Additional file 2, apart from the two BRCA1 structure models and their README under `data/raw/structures/`.

**This does not block reproduction of the published numbers.** Every value in the manuscript, tables and figures is held in the recompute layer — the `reports/*.json` files and the small `reports/*.parquet` files, both shipped in Additional file 2 — which the table and figure builders read directly. Running `src/ccs/build_tables.py` against those deposited artefacts regenerates the deposit's own tables report, `reports/tables.md`, without touching `data/`, while `src/ccs/build_supplementary.py` does not produce the submitted supplementary table set, which ships as the export `reports/supplementary_tables.md`; `tools/verify_from_data.py`, by contrast, re-derives published values from the raw score and label files under `data/`. Most `fig*_stats.py` scripts are **not** in that class: they read per-variant panels under `data/` and exit non-zero without it; the three `fig4_*_stats.py` read only deposited `reports/` files and run here. Most deposited analysis scripts need `data/` too — 106 of the 173 under `src/` reference a `data/` path, including `build_phylop_crosscheck.py` and the other `fig*_stats.py` scripts, and `build_tost_equivalence.py` needs `data/` through its import of `fig5_stats`. What they emit — the recompute JSONs and small parquets — IS deposited, so every published value stays checkable where the script that produced it cannot be re-run here. The raw tree is needed to re-run Evo 2 scoring end-to-end (a GPU workload) and to rebuild Additional file 1's Figures S3, S5 and S6 (see FIGURES.md). Figures 9 and 10 and Additional file 1's Figures S1, S2, S4 and S7 rebuild from deposited artefacts, and Figures 1 to 8 and Additional file 1's Figures S8 to S12 are built in Additional file 3 and need nothing from `data/`.


## Deposited `reports/` artefacts with no in-deposit writer

Six artefacts in the deposit are inputs rather than outputs: deposited scripts read them, but no deposited script writes them. They are listed here so the provenance chain has no unexplained node.

- `reports/_recon_pervariant_trust.parquet` — per-variant trust-layer reconstruction (species, variant_id, label, score and the four calibration posteriors) for the 1,001-bp single-position readout on all 11,130 atlas variants. It is a deposited **input**, consumed by `build_emin.py`, `build_fig3_consequence.py`, `build_locus_clustered_ci.py`, `build_ablation_fp8.py`, `build_raw_ece.py`, `fig4_trust.py` and `analyses/scripts/calibration_error.py`. Produced by the trust-layer build from `data/processed/scores/<species>_evo2_40b_local_scores.parquet` (the input `src/ccs/fig4_reconcile.py` reads), which is not deposited; the file is shipped so those consumers run without it.
- `reports/ablation_fp8.parquet` — the FP8 batch-size control's per-variant scores at both batch sizes (variant_id, delta_b4, delta_b32; 600 rows). `build_ablation_fp8.py` reads it and regenerates `reports/ablation_fp8.json` from it, joining labels from `_recon_pervariant_trust.parquet`.
- `reports/fig4_leak.json` — the selective-layer confusion counts (missed positives, false alarms, caught, reaching the clinician); Additional file 1 prints the first three in Table S28, the refusal total in Table S27's notes and the share of errors refused (35.4%) in Note S52. Every stored count recomputes from the deposited `reports/fig4_pervariant.parquet` at the 15% cut.
- `reports/eqtl_pervariant.parquet` — the 20,000 pig cis-eQTL records (variant_id, chrom, pos, gene_id, pip, label, evo2_40b_score), read by the Figure S2 builder and the eQTL control scripts.
- `reports/gerp_pervariant.parquet` — GERP per atlas variant (11,130 rows; species, source, variant_id, label, gerp, gerp_is_finite), read by `src/ccs/gerp_consequence.py` and three `analyses/scripts/`.
- `reports/nt_vs_evo2.parquet` — the per-panel Nucleotide Transformer, Evo 2 and GERP AUROCs (10 rows), read by `src/ccs/fig3_data.py`.

Four further files are records that no deposited script reads or writes: `reports/eqtl_pervariant_meta.json` and `reports/gerp_pervariant_meta.json` describe the two parquets above, `reports/power_mde.json` holds the minimum-detectable-difference table, and `reports/supplementary_citation_map.json` maps a superseded reference numbering.


## Public sources of the raw inputs

Exact releases and assembly accessions are in the manuscript Methods and Availability of data and materials; the primary sources are:

- **Catalogued alleles (positives):** OMIA (Online Mendelian Inheritance in Animals, <https://omia.org>) for the eight non-human species; NCBI ClinVar (<https://www.ncbi.nlm.nih.gov/clinvar/>, release dated in Methods) for the human panel.
- **Negatives and genome assemblies:** population variants from Ensembl Variation for seven non-human species, the PigGTEx v0 genotype panel for pig, and ClinVar Benign/Likely_benign records for human; the Ensembl per-species releases and GenBank assembly accessions are listed in Methods.
- **Regulatory (eQTL) panel:** PigGTEx fine-mapped candidate causal *cis*-eQTLs (SuSiE-inf); their allele frequencies from PigGTEx's molQTL genotype release (Science Data Bank, doi:10.57760/sciencedb.08089), which `analyses/scripts/build_piggtex_maf.py` identifies by URL and MD5.
- **Conservation tracks:** GERP from the per-species Ensembl bigWig tracks (see `src/ccs/query_gerp.py`); phyloP from UCSC where a genuine track exists (human hg38 phyloP100way, chicken galGal6 phyloP77way) and, for cattle (bosTau9), from Zenodo record 13332541 (Roslin) over a Cactus 241-way alignment — a different alignment and species set, described in Additional file 1: Table S8.
- **Label-free panels:** the cattle population allele-frequency panel and the pilot bat cohort VCF described in Methods.


## Path inventory (176 distinct templates; `<var>` = a species, sample, model size, scorer or file stem)


### `analyses/data/` — Inputs of the analyses/scripts/ arms — splicing, ClinVar and strand-ladder scores and windows.

- `analyses/data/**/*w1001*.parquet`
- `analyses/data/**/*windows*.parquet`
- `analyses/data/clinvar/clinvar_evo2_1b_scores.parquet`
- `analyses/data/clinvar/clinvar_evo2_40b_scores.parquet`
- `analyses/data/clinvar/clinvar_evo2_7b_scores.parquet`
- `analyses/data/clinvar/clinvar_w1001.parquet`
- `analyses/data/clinvar/from40b/clinvar_scores.parquet`
- `analyses/data/mfass/S2.xlsx`
- `analyses/data/mfass/mfass_evo2_*_scores.parquet`
- `analyses/data/mfass/mfass_evo2_1b_scores.parquet`
- `analyses/data/mfass/mfass_evo2_40b_scores.parquet`
- `analyses/data/mfass/mfass_w1001.parquet`
- `analyses/data/mfass/random500k.parquet`
- `analyses/data/strand/from40b/strand40b_scores.parquet`
- `analyses/data/strand/strand_ladder_evo2_<var>_scores.parquet`
- `analyses/data/strand/strand_ladder_w8192.parquet`
- `analyses/data/strand/strand_subset_windows.parquet`
- `analyses/data/strand/upload40b/strand40b_w8192.parquet`
- `analyses/data/traitgym/complex_traits_test.parquet`
- `analyses/data/traitgym/mendelian_traits_test.parquet`

### `data/external/` — Third-party tracks and panels obtained externally.

- `data/external/GRCh38.fa`
- `data/external/GRCh38.primary_assembly.fa.gz`
- `data/external/dbnsfp/dbNSFP5.3.1a_grch38.gz`
- `data/external/evee/clean_shard_*.parquet`
- `data/external/evee/manifest.json`
- `data/external/human_panel/clinvar.vcf.gz`
- `data/external/human_panel/clinvar_panel_geneinfo.parquet`
- `data/external/human_panel/hg38.phyloP100way.bw`
- `data/external/human_panel/whole_genome_SNVs.tsv.gz`
- `data/external/proteingym/DMS_substitutions/*.parquet`

### `data/interim/` — Intermediate build products — variant windows, candidate tables, pilots.

- `data/interim/*_scoring_windows.parquet`
- `data/interim/*windows*.parquet`
- `data/interim/<var>.parquet`
- `data/interim/<var>_omia_pos.parquet`
- `data/interim/<var>_scoring_windows.parquet`
- `data/interim/ablation/eqtl_abl_8192.parquet`
- `data/interim/ablation/eqtl_abl_sample.parquet`
- `data/interim/ablation/eqtl_full_8192.parquet`
- `data/interim/ablation/eqtl_gerp_pilot.parquet`
- `data/interim/atlas8192/*_windows_8192.parquet`
- `data/interim/atlas8192/<var>_windows_8192.parquet`
- `data/interim/atlas8192/human_windows_8192.parquet`
- `data/interim/bat/bat.vcf.gz`
- `data/interim/bat/bat_af.tsv`
- `data/interim/bat_candidates.parquet`
- `data/interim/bat_windows.parquet`
- `data/interim/brca1_labels.parquet`
- `data/interim/brca1_short_windows.parquet`
- `data/interim/brca1_variant_residues.parquet`
- `data/interim/brca1_windows.parquet`
- `data/interim/cadd_clinvar_subset.tsv`
- `data/interim/clinvar_calib_subset.parquet`
- `data/interim/clinvar_evo2_windows.parquet`
- `data/interim/clinvar_gene.parquet`
- `data/interim/clinvar_keys.tsv`
- `data/interim/dbnsfp_clinvar_subset.tsv`
- `data/interim/dog_omia_pos.parquet`
- `data/interim/dog_scoring_windows.parquet`
- `data/interim/dog_snv_pool.tsv`
- `data/interim/dog_windows.parquet`
- `data/interim/eqtl_candidates.parquet`
- `data/interim/eqtl_emb.npz`
- `data/interim/eqtl_piggtex_maf.parquet`
- `data/interim/eqtl_piggtex_maf_meta.json`
- `data/interim/eqtl_windows.parquet`
- `data/interim/full_phylop.parquet`
- `data/interim/geno_pgen/*.psam`
- `data/interim/human_scoring_windows.parquet`
- `data/interim/omia_matched_panel.parquet`
- `data/interim/omia_multispecies_positives.parquet`
- `data/interim/pig_omia_pos.parquet`
- `data/interim/pig_scoring_windows.parquet`
- `data/interim/pig_scoring_windows_real.parquet`
- `data/interim/pig_windows.parquet`
- `data/interim/selection_candidates.parquet`
- `data/interim/selection_phylop.parquet`
- `data/interim/verify_table.parquet`

### `data/processed/` — Scored outputs — Evo 2 / NT / conservation scores per panel.

- `data/processed/abstention_risk_coverage.parquet`
- `data/processed/abstention_trust_budget.parquet`
- `data/processed/atlas_40b.json`
- `data/processed/atlas_40b.parquet`
- `data/processed/calibration_methods.parquet`
- `data/processed/calibration_rigor.parquet`
- `data/processed/calibration_robustness.parquet`
- `data/processed/calibration_size_curve.parquet`
- `data/processed/calibration_transfer.parquet`
- `data/processed/cattle_finemapped.parquet`
- `data/processed/clinvar_panel.parquet`
- `data/processed/conformal_per_species.parquet`
- `data/processed/conformal_robustness.parquet`
- `data/processed/conformal_sweep.parquet`
- `data/processed/consequence_<var>.parquet`
- `data/processed/consequence_human.parquet`
- `data/processed/conservation/*_gerp.parquet`
- `data/processed/conservation/<var>_gerp.parquet`
- `data/processed/conservation/cattle_ensvar_gerp.parquet`
- `data/processed/conservation/dog_cf3_gerp.parquet`
- `data/processed/conservation/human_gerp.parquet`
- `data/processed/conservation/omia_matched_phylop.parquet`
- `data/processed/dbnsfp_panel.parquet`
- `data/processed/decomposition.parquet`
- `data/processed/emb/<var>_emb.npz`
- `data/processed/eqtl_calibration.parquet`
- `data/processed/hardening_cis.parquet`
- `data/processed/human_scorers/<var>.parquet`
- `data/processed/human_scorers/alphamissense.parquet`
- `data/processed/nt_calibration_transfer.parquet`
- `data/processed/positives_consequence.parquet`
- `data/processed/prevalence_correction.parquet`
- `data/processed/prevalence_sweep.parquet`
- `data/processed/proteingym_esm2.parquet`
- `data/processed/scores/<var>_evo2_40b_local_scores.parquet`
- `data/processed/scores/<var>_nt_atlas.parquet`
- `data/processed/scores/bat_evo2_40b.parquet`
- `data/processed/scores/brca1_evo2_1b_meanll.parquet`
- `data/processed/scores/brca1_evo2_40b_delta.parquet`
- `data/processed/scores/brca1_evo2_40b_meanll_8192.parquet`
- `data/processed/scores/brca1_evo2_<var>_meanll.parquet`
- `data/processed/scores/brca1_evo2_{a.model_size}_meanll.parquet`
- `data/processed/scores/cattle_evo2_40b_local_scores.parquet`
- `data/processed/scores/clinvar_evo2_raw.parquet`
- `data/processed/scores/eqtl_abl_1002_ll.parquet`
- `data/processed/scores/eqtl_abl_2048_ll.parquet`
- `data/processed/scores/eqtl_abl_2048_sp.parquet`
- `data/processed/scores/eqtl_abl_4096_ll.parquet`
- `data/processed/scores/eqtl_abl_4096_sp.parquet`
- `data/processed/scores/eqtl_abl_{W}_ll.parquet`
- `data/processed/scores/eqtl_evo2_40b.parquet`
- `data/processed/scores/esm/*.parquet`
- `data/processed/scores/esm/<var>_esm.parquet`
- `data/processed/scores/esm/human_esm.parquet`
- `data/processed/scores/human_evo2_40b_local_scores.parquet`
- `data/processed/scores/nt/eqtl_nt.parquet`
- `data/processed/scores/omia_matched_evo2_scores.parquet`
- `data/processed/scores/pig_evo2_scores.parquet`
- `data/processed/scores/selection_evo2_40b.parquet`
- `data/processed/scores/selection_evo2_40b_api.parquet`
- `data/processed/scores/{e1001_stem}_evo2_40b_local_scores.parquet`
- `data/processed/scores_cloud/atlas8192_<var>_meanll_1b_8192.parquet`
- `data/processed/scores_cloud/atlas8192_<var>_meanll_7b_8192.parquet`
- `data/processed/scores_cloud/atlas8192_<var>_meanll_8192.parquet`
- `data/processed/scores_cloud/atlas8192_cattle_meanll_8192.parquet`
- `data/processed/scores_cloud/atlas8192_human_meanll_8192.parquet`
- `data/processed/scores_cloud/brca1_evo2_40b_meanll_8192.parquet`
- `data/processed/scores_cloud/eqtl_abl_8192_1b_ll.parquet`
- `data/processed/scores_cloud/eqtl_abl_8192_7b_ll.parquet`
- `data/processed/scores_cloud/eqtl_abl_8192_ll.parquet`

### `data/raw/` — Raw downloads — genome FASTAs, OMIA/ClinVar truth sets, source VCFs.

- `data/raw/clinvar/clinvar_GRCh38.vcf.gz`
- `data/raw/conservation/chromalias/<var>.chromAlias.txt`
- `data/raw/conservation/download/galGal6.phyloP77way.bw`
- `data/raw/conservation/download/hg38.phyloP100way.bw`
- `data/raw/conservation/pig/pig_gerp.Sscrofa11.1.bw`
- `data/raw/genomes/bat/bat.fa`
- `data/raw/genomes/cat/cat_fca126.fa`
- `data/raw/genomes/cattle/cattle.fa`
- `data/raw/genomes/chicken/chicken.fa`
- `data/raw/genomes/dog/dog_ROS_Cfam.fa`
- `data/raw/genomes/dog_cf3/dog_CanFam3.fa`
- `data/raw/genomes/goat/goat.fa`
- `data/raw/genomes/horse/horse.fa`
- `data/raw/genomes/human/human.fa`
- `data/raw/genomes/pig/Sus_scrofa.Sscrofa11.1.110.gtf.gz`
- `data/raw/genomes/pig/Sus_scrofa.Sscrofa11.1.112.gtf.gz`
- `data/raw/genomes/pig/pig.fa`
- `data/raw/genomes/pig/pig.fa.fai`
- `data/raw/genomes/sheep/sheep.fa`
- `data/raw/liftover/canFam3ToCanFam6.over.chain`
- `data/raw/snpeff/build.config`
- `data/raw/structures/*.pdb`
- `data/raw/structures/brca1_AF.pdb`
- `data/raw/structures/brca1_BRCT_1t29.pdb`
- `data/raw/structures/brca1_BRCT_1t29_model1.pdb`
- `data/raw/structures/brca1_RING_1jm7.pdb`
- `data/raw/structures/brca1_RING_1jm7_model1.pdb`
- `data/raw/truth/omia_all_variants.csv`
- `data/raw/truth/omia_cattle_snvs.csv`

