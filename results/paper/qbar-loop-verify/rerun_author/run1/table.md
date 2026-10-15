### DESCRIPTIVE - prereg section 9 |Q̄| feedback-loop prediction (SafetyPointGoal1-v0, 2M steps)

Rows: lagu-tt interval rows with 300k < step <= 2M (34 per run). Final λ: `lambda` at 2M; denominator lagu_nogate-tt seeds 100-114 in both sets. 95% percentile bootstrap over runs (10,000 resamples, seed 0).

| seed set | lagu-tt runs | median \|qbar_avg\| (< 0.3?) | engagement (> 0.80?) | final-λ ratio lagu / lagu_nogate (> 3?) | antecedent met | prediction holds |
|---|---|---|---|---|---|---|
| primary (seeds 100-114) | 15 | 1.592 [0.802, 2.455] (no) | 0.919 [0.879, 0.948] (yes) | 0.0007 [0.0002, 2.3012] (no) | no | no |
| sensitivity (seeds 100-119) | 20 | 1.618 [0.903, 2.512] (no) | 0.907 [0.862, 0.945] (yes) | 0.0007 [0.0002, 2.4047] (no) | no | no |

Cell means of final λ: primary (seeds 100-114): lagu-tt 0.4009, lagu_nogate-tt 599.40; sensitivity (seeds 100-119): lagu-tt 0.4282, lagu_nogate-tt 599.40.

### DESCRIPTIVE - per run (no run excluded)

| seed | lagu-tt median \|qbar_avg\| after 300k | lagu-tt engagement after 300k | lagu-tt final λ | lagu-tt resumed | lagu_nogate-tt final λ | lagu_nogate-tt resumed |
|---|---|---|---|---|---|---|
| 100 | 0.512 | 0.698 | 0.6831 | no | 0.2139 | yes |
| 101 | 1.328 | 0.953 | 0.5337 | no | 0.4007 | yes |
| 102 | 2.138 | 0.940 | 1.245 | no | 0.44 | yes |
| 103 | 1.379 | 0.960 | 0.4242 | no | 8987 | yes |
| 104 | 10.891 | 0.944 | 0.02172 | no | 0.2216 | yes |
| 105 | 0.363 | 0.838 | 0.4581 | no | 0.294 | yes |
| 106 | 3.064 | 0.982 | 0.1308 | no | 0.1772 | no |
| 107 | 7.473 | 0.968 | 0.1785 | no | 0.3515 | no |
| 108 | 1.461 | 0.929 | 0.5619 | no | 0.2391 | no |
| 109 | 0.564 | 0.886 | 1.281 | yes | 0.2864 | no |
| 110 | 1.404 | 0.967 | 0.0002016 | yes | 0.1111 | no |
| 111 | 13.426 | 0.959 | 0.04018 | yes | 0.003579 | no |
| 112 | 2.498 | 0.937 | 3.06e-06 | yes | 0.1687 | no |
| 113 | 1.023 | 0.905 | 0.2119 | yes | 0.08199 | no |
| 114 | 11.727 | 0.919 | 0.2437 | yes | 0.8861 | no |
| 115 | 8.713 | 0.975 | 0.1809 | no | not run | - |
| 116 | 13.741 | 0.994 | 0.141 | no | not run | - |
| 117 | 0.572 | 0.700 | 1.588 | no | not run | - |
| 118 | 2.952 | 0.990 | 0.000328 | no | not run | - |
| 119 | 0.630 | 0.691 | 0.6407 | no | not run | - |
