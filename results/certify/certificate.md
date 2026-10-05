# Gate-channel certificate: PASS

`/Users/michaelseiranian/Desktop/safe-rl-carla/results/certify`, 2026-09-29 23:38:57. Per-tensor SHA-256 of the final `.pt`; equal pairs also need identical `eval_*`, `lambda`, `qc_traj`, `mc_cost_traj` columns.

| A | B | seed | expect | tensors differing | CSV columns differing | sum n_risk | max engagement | max lambda | result |
|---|---|---|---|---|---|---|---|---|---|
| lagu-fix | lagu_nogate-fix | 0 | equal | 0 / 241 | none | 1.855e+04 / 1.855e+04 | 0.9703 | 0.1 / 0.1 | pass |
| lagu-fix | lagu_nogate-fix | 1 | equal | 0 / 241 | none | 1.855e+04 / 1.855e+04 | 0.9787 | 0.1 / 0.1 | pass |
| lagu-epi | lagu_nogate-epi | 0 | equal | 0 / 245 | none | 1.807e+04 / 1.807e+04 | 0.9996 | 0.1576 / 0.1576 | pass |
| lagu-epi | lagu_nogate-epi | 1 | equal | 0 / 245 | none | 1.823e+04 / 1.823e+04 | 0.9456 | 0.1531 / 0.1531 | pass |
| lagu-sw | lagu_nogate-sw | 0 | different | 193 / 241 | eval_cost, eval_return, eval_violation, lambda, qc_traj, mc_cost_traj | 1.855e+04 / 1.855e+04 | 0.9317 | 0.7586 / 0.4528 | pass |
| lagu-epigg | lagu_nogate-epi | 0 | different | 196 / 245 | eval_cost, eval_return, eval_violation, lambda, qc_traj, mc_cost_traj | 1.807e+04 / 1.807e+04 | 0.9952 | 0.1517 / 0.1576 | pass |

| run | digest (SHA-256 over the per-tensor hashes) |
|---|---|
| lagu-epi_SafetyPointGoal1-v0_s0 | `465f01042d92f77ec7bf14c34db955e10c508f3acef4ca79ce5fb096061f3d7b` |
| lagu-epi_SafetyPointGoal1-v0_s1 | `f992fbd6c3b46f2fed48dfe8dfd21548967f37c2aafc3bc0f055d01f986d59af` |
| lagu-epigg_SafetyPointGoal1-v0_s0 | `445c1f3f39c525103c3cc30ff3a9b61d5295b6a8fba8e9a4b102d04f52ae072b` |
| lagu-fix_SafetyPointGoal1-v0_s0 | `8cb8500ca7d40ce0fe6ccddc03c496505ed346d286eabc71abf075b59e073201` |
| lagu-fix_SafetyPointGoal1-v0_s1 | `63605f296dfae80982c5ddc8a6e67f78e58178f8aade8fc676f931108c56575b` |
| lagu-sw_SafetyPointGoal1-v0_s0 | `6b7d7404861e9d4d27476b8803bbb5d2a4a4d879b55a686f7fe6fa687a544a51` |
| lagu_nogate-epi_SafetyPointGoal1-v0_s0 | `465f01042d92f77ec7bf14c34db955e10c508f3acef4ca79ce5fb096061f3d7b` |
| lagu_nogate-epi_SafetyPointGoal1-v0_s1 | `f992fbd6c3b46f2fed48dfe8dfd21548967f37c2aafc3bc0f055d01f986d59af` |
| lagu_nogate-fix_SafetyPointGoal1-v0_s0 | `8cb8500ca7d40ce0fe6ccddc03c496505ed346d286eabc71abf075b59e073201` |
| lagu_nogate-fix_SafetyPointGoal1-v0_s1 | `63605f296dfae80982c5ddc8a6e67f78e58178f8aade8fc676f931108c56575b` |
| lagu_nogate-sw_SafetyPointGoal1-v0_s0 | `a204f50bdec8ef9a1c54d63abdce301f1687c6e7df1aa08efe5df4dcb53a2287` |
