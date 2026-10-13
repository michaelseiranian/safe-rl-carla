### DESCRIPTIVE - independent re-derivation, prereg section 9 |Q̄| feedback loop (SafetyPointGoal1-v0, 2M)

Pre-registered definitions; own code (csv + statistics; numpy bootstrap, 10,000 resamples, default_rng(0), percentile 95%).

| seed set | runs (lagu / nogate) | rows | median \|qbar_avg\| (< 0.3?) | engagement (> 0.8?) | final-λ ratio (> 3?) | antecedent | prediction holds |
|---|---|---|---|---|---|---|---|
| primary s100-114 | 15 / 15 | 510 | 1.5925 [0.802, 2.455] (no) | 0.9191 [0.879, 0.948] (yes) | 0.000669 [0.00017, 2.3012] (no) | no | no |
| sensitivity s100-119 | 20 / 15 | 680 | 1.6181 [0.903, 2.512] (no) | 0.9068 [0.862, 0.945] (yes) | 0.000714 [0.00019, 2.4047] (no) | no | no |

Cell means of final λ: primary s100-114: lagu-tt 0.400912, lagu_nogate-tt 599.3972; sensitivity s100-119: lagu-tt 0.428216, lagu_nogate-tt 599.3972

### DESCRIPTIVE - agreement with the author's qbar_loop.json (relative tolerance 1e-12)

| seed set | quantity | agrees | max rel. diff |
|---|---|---|---|
| primary s100-114 | median_abs_qbar | yes | 0.0e+00 |
| primary s100-114 | engagement | yes | 0.0e+00 |
| primary s100-114 | ratio | yes | 2.2e-19 |
| primary s100-114 | mean_final_lambda_lagu | yes | 5.6e-17 |
| primary s100-114 | mean_final_lambda_nogate | yes | 1.9e-16 |
| primary s100-114 | n_rows | yes | 0.0e+00 |
| primary s100-114 | median_ci95 | yes | 0.0e+00 |
| primary s100-114 | engagement_ci95 | yes | 0.0e+00 |
| primary s100-114 | ratio_ci95 | yes | 0.0e+00 |
| primary s100-114 | antecedent_met | yes | 0.0e+00 |
| primary s100-114 | engagement_gt_0.8 | yes | 0.0e+00 |
| primary s100-114 | ratio_gt_3 | yes | 0.0e+00 |
| primary s100-114 | prediction_holds | yes | 0.0e+00 |
| sensitivity s100-119 | median_abs_qbar | yes | 0.0e+00 |
| sensitivity s100-119 | engagement | yes | 0.0e+00 |
| sensitivity s100-119 | ratio | yes | 2.2e-19 |
| sensitivity s100-119 | mean_final_lambda_lagu | yes | 0.0e+00 |
| sensitivity s100-119 | mean_final_lambda_nogate | yes | 1.9e-16 |
| sensitivity s100-119 | n_rows | yes | 0.0e+00 |
| sensitivity s100-119 | median_ci95 | yes | 0.0e+00 |
| sensitivity s100-119 | engagement_ci95 | yes | 0.0e+00 |
| sensitivity s100-119 | ratio_ci95 | yes | 0.0e+00 |
| sensitivity s100-119 | antecedent_met | yes | 0.0e+00 |
| sensitivity s100-119 | engagement_gt_0.8 | yes | 0.0e+00 |
| sensitivity s100-119 | ratio_gt_3 | yes | 0.0e+00 |
| sensitivity s100-119 | prediction_holds | yes | 0.0e+00 |

Per-run final λ in author JSON agrees: yes. Author per-run table agrees at printed precision: yes. prereg_results.json secondary.qbar_loop agrees with the sensitivity row: yes.

### DESCRIPTIVE - claims in the author's summary/caveats

| claim | holds |
|---|---|
| no run's median |qbar_avg| < 0.3; lowest s105 = 0.363 | yes |
| highest per-run median s116 = 13.7 | yes |
| per-run engagement (s100-119) ranges 0.691-0.994 | yes |
| lagu_nogate-tt s103 final lambda 8987 | yes |
| other 14 lagu_nogate-tt final lambda within 0.0036-0.886 | yes |
| lagu-tt cell means 0.4009 / 0.4282 | yes |
| lagu_nogate-tt cell mean 599.40 | yes |
| resumed: lagu-tt s109-114, lagu_nogate-tt s100-105 (and no others) | yes |
| every CSV has 40 rows, unique sorted steps, last step 2M, 34 window rows | yes |
| lagu_nogate-tt seeds 115-119 were never run | yes |
| about 35% of ratio resamples exclude s103 | yes |
| both ratio upper bounds below 3 | yes |
| sensitivity row equals prereg_results.json secondary.qbar_loop | yes |

Bootstrap resamples whose lagu_nogate-tt draw excludes s103 (checks the caveat 'ratios near 1-2.4'):

- primary s100-114: fraction 0.3627; ratio min 0.412, p5 0.812, median 1.439, p95 2.423, max 4.593; fraction of ALL resamples with ratio > 3: 0.0029
- sensitivity s100-119: fraction 0.3566; ratio min 0.478, p5 0.919, median 1.562, p95 2.502, max 4.655; fraction of ALL resamples with ratio > 3: 0.0036

Across runs (DESCRIPTIVE context, not a test): Spearman rank correlation between a run's median |qbar_avg| and its engagement after 300k:
- primary s100-114: rho = +0.561; the 5 runs with the lowest median |qbar_avg| s105 (0.363 -> eng 0.838), s100 (0.512 -> eng 0.698), s109 (0.564 -> eng 0.886), s113 (1.023 -> eng 0.905), s101 (1.328 -> eng 0.953)
- sensitivity s100-119: rho = +0.708; the 5 runs with the lowest median |qbar_avg| s105 (0.363 -> eng 0.838), s100 (0.512 -> eng 0.698), s109 (0.564 -> eng 0.886), s117 (0.572 -> eng 0.700), s119 (0.630 -> eng 0.691)

### DESCRIPTIVE - alternative definitions (NOT pre-registered; robustness of the antecedent only)

| seed set | median, window step >= 300k | median of per-run medians | fraction of rows with \|qbar_avg\| < 0.3 | engagement, unweighted | ratio with lambda_avg at 2M |
|---|---|---|---|---|---|
| primary s100-114 | 1.563 | 1.461 | 0.098 | 0.919 | 0.000701 |
| sensitivity s100-119 | 1.590 | 1.800 | 0.093 | 0.907 | 0.000735 |

### DESCRIPTIVE - re-run of the author's make.py (copy, outputs in ./rerun_author/)

Copy identical to published script: yes. Return codes: 0, 0.

| file | published == re-run | re-run 1 == re-run 2 |
|---|---|---|
| qbar_loop.json | yes | yes |
| table.md | yes | yes |
| qbar_loop.png | yes | yes |
| qbar_loop.pdf | no | no |

PDF differs from the published one only in /CreationDate: yes.
