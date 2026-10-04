# Is goat's n=9 calibration a fluke? (downsampling sampling distribution)

Label-rich donors ['cat', 'cattle', 'dog', 'human'] subsampled to n_pos positives at ~9% prevalence; LOSO-transferred isotonic calibrator applied; transfer-ECE bootstrapped B=4000/level (all 52 cores).

| positives (n_pos) | transfer-ECE median | 5-95% interval |
|---|---|---|
| 5 | 0.068 | [0.028, 0.109] |
| 9 | 0.065 | [0.034, 0.096] |
| 15 | 0.062 | [0.038, 0.088] |
| 25 | 0.06 | [0.039, 0.081] |
| 50 | 0.057 | [0.04, 0.074] |

**Goat's ACTUAL transferred ECE (n=9 positives) = 0.056** -> sits INSIDE the n_pos=9 sampling interval [0.034, 0.096]. Goat is NOT an outlier: its small-n calibration is exactly what the transfer produces at a 9-positive budget - the objection is answered with a controlled experiment on abundant data.

The interval NARROWS as n_pos grows (expected), so the label-poor species carry wider but well-characterized uncertainty - we report it rather than over-claiming per-species precision.
