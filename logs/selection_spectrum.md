# Idea 1: Evo2-40B deleteriousness vs purifying selection (cattle, 28k variants)

| freq bin | n | mean MAF | mean Evo2 deleteriousness |
|---|---|---|---|
| ultra-rare | 5607 | 0.0006 | 0.772 |
| rare | 5624 | 0.0050 | 0.511 |
| low | 5581 | 0.0244 | 0.319 |
| common | 5596 | 0.1214 | 0.201 |
| major | 5598 | 0.3899 | 0.178 |

Spearman(MAF, deleteriousness) = -0.099 (p=2.9e-62); expected NEGATIVE.
Monotone rise toward rare: True.
**PASS** — selection-spectrum validation holds: Evo2 captures deleteriousness genome-wide, label-free.
