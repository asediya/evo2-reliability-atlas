# `structures/` — the two BRCA1 models Additional file 1's Figure S6b is rendered from

`src/ccs/make_brca1_structure.py` reads
`data/raw/structures/brca1_RING_1jm7.pdb` and `brca1_BRCT_1t29.pdb`, strips each to model 1,
and writes the stripped copy to `reports/figures/_brca1_model1.pdb` — **the same filename for
every domain**, so only the last domain rendered survives on disk.

The two stripped model-1 structures deposited here are the ones embedded, inline via
``v.addModel(`...`, 'pdb')``, in the 3Dmol HTML pages `src/ccs/rebuild_brca1_structure_html.py`
writes, which are not deposited; they are byte-for-byte the models the published renders were drawn from.

| file | contents | source entry |
|---|---|---|
| `brca1_RING_1jm7_model1.pdb` | chain A 1-103 (BRCA1), chain B 26-122 (BARD1), 4 Zn | PDB **1JM7**, model 1 of the 20-model NMR ensemble |
| `brca1_BRCT_1t29_model1.pdb` | chain A 1649-1859 (BRCA1 BRCT), chain B 1-12 (BACH1 phosphopeptide, SEP), 235 waters | PDB **1T29**, model 1 |

These are model-1 extracts, not the deposited PDB entries. For the originals see
RCSB 1JM7 and 1T29; `brca1_AF.pdb` (the AlphaFold full-length model, `--domain full`) is not
used by any submitted figure.
