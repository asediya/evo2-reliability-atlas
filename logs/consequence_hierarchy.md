# Do Evo2-40B scores respect the OMIA consequence hierarchy?

Mean 40B deleteriousness (evo2_40b_neg; higher = more damaging) per consequence, pooled:

| consequence | n | mean_40B | median |
|---|---|---|---|
| nonsense (stop-gain) | 112 | 6.863 | 7.180 |
| extension (stop-lost) | 1 | 2.977 | 2.977 |
| splicing | 70 | 3.774 | 2.010 |
| missense | 324 | 5.961 | 7.188 |
| regulatory | 18 | 0.282 | 0.086 |

**LoF** (nonsense/frameshift/splice/start-lost) n=182 mean=5.675
**missense** n=324 mean=5.961
**regulatory** n=18 mean=0.282

Hierarchy LoF > missense > regulatory holds: False
