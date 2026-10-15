# Verification of results/paper/e3_td3lag_e9 (DESCRIPTIVE; independent re-derivation, results/paper/e3-td3lag-verify/make.py)

## V1. E3 per task, lagu-tt minus lagu_nogate-tt, held-out endpoint (DESCRIPTIVE re-derivation; E3 NOT BLIND, first computed by an audit after G3)

| Task | Pairs | Return gate / no gate | Δreturn [95%] | Cost gate / no gate | Δcost [95%] | Collapsed (Fisher p) | Engagement 25k-300k / run | D (D/δ0) | Signs | MC spread of 95% bounds (20 RNG seeds) |
|---|---|---|---|---|---|---|---|---|---|---|
| PointGoal1 (E1 ref) | 15 | 5.42 / 15.59 | -10.17 [-16.08, -3.80] | 34.46 / 37.77 | -3.31 [-18.75, 16.70] | 9 / 3 (0.060) | 0.850 / 0.913 | 1.122 (0.449, δ0 2.5) | (-, -) | 1.29 |
| CarGoal1 | 10 | 2.10 / 10.15 | -8.05 [-15.78, -0.56] | 26.91 / 34.32 | -7.41 [-29.36, 19.26] | 8 / 5 (0.350) | 0.861 / 0.941 | 1.007 (0.403, δ0 2.5) | (-, -) | 1.72 |
| PointCircle1 | 10 | 36.88 / 33.37 | 3.51 [-7.57, 13.58] | 32.88 / 36.35 | -3.47 [-32.22, 24.39] | 0 / 0 (1.000) | 0.379 / 0.250 | 4.214 (0.843, δ0 5) | (+, -) | 1.15 |

- CarGoal1 section 9 sign check vs E1 (-, -): holds; resumed seeds none; diverged none; integrity problems none.
- PointCircle1 section 9 sign check vs E1 (-, -): does not hold; resumed seeds none; diverged none; integrity problems none.
- E1 reference: resumed seeds [100, 101, 102, 103, 104, 105, 109, 110, 111, 112, 113, 114]; without them (3 pairs) Δreturn -15.86 [-21.19, -11.17], Δcost 20.99 [-27.61, 98.79]; integrity problems none.
- E1 reference vs prereg.md section 15 (H3 entry): identical at 2 decimals; G2 entry D = 1.12158, engagement 0.8497 vs here 1.12158, 0.8497.
- CarGoal1 Δreturn decomposed by pair collapse status (n pairs, contribution to the mean): both 3 (+0.00); gate only 5 (-9.66); no-gate only 2 (+1.61).
- PointCircle1 Δreturn decomposed by pair collapse status (n pairs, contribution to the mean): neither 10 (+3.51).

Per-seed held-out values (gate / no gate):

| Task | Seed | Return | Cost | Collapsed |
|---|---|---|---|---|
| CarGoal1 | 100 | -0.05 / 19.35 | 0.69 / 45.35 | yes / no |
| CarGoal1 | 101 | 11.74 / 2.34 | 71.04 / 77.42 | no / yes |
| CarGoal1 | 102 | -0.87 / 2.32 | 13.49 / 65.66 | yes / yes |
| CarGoal1 | 103 | -0.01 / 19.22 | 2.98 / 42.53 | yes / no |
| CarGoal1 | 104 | 0.01 / 21.86 | 25.50 / 39.45 | yes / no |
| CarGoal1 | 105 | -0.25 / 27.10 | 3.32 / 38.66 | yes / no |
| CarGoal1 | 106 | 6.66 / -0.03 | 91.24 / 0.00 | no / yes |
| CarGoal1 | 107 | 1.17 / 9.94 | 23.33 / 23.67 | yes / no |
| CarGoal1 | 108 | 2.52 / -0.17 | 19.20 / 9.28 | yes / yes |
| CarGoal1 | 109 | 0.03 / -0.48 | 18.28 / 1.15 | yes / yes |
| PointCircle1 | 100 | 38.62 / 42.58 | 10.50 / 37.38 | no / no |
| PointCircle1 | 101 | 40.62 / 33.53 | 48.70 / 25.35 | no / no |
| PointCircle1 | 102 | 38.89 / 39.92 | 22.73 / 38.52 | no / no |
| PointCircle1 | 103 | 14.92 / 46.48 | 13.11 / 106.62 | no / no |
| PointCircle1 | 104 | 41.32 / 32.46 | 39.51 / 1.89 | no / no |
| PointCircle1 | 105 | 40.53 / 29.35 | 18.77 / 1.22 | no / no |
| PointCircle1 | 106 | 45.25 / 39.06 | 101.62 / 20.93 | no / no |
| PointCircle1 | 107 | 40.54 / 15.11 | 28.62 / 45.35 | no / no |
| PointCircle1 | 108 | 28.68 / 44.16 | 27.21 / 73.50 | no / no |
| PointCircle1 | 109 | 39.44 / 11.04 | 18.05 / 12.76 | no / no |

## V2. Lag-U vs TD3-Lag, held-out endpoint, unpaired IQM (DESCRIPTIVE only; CONFOUNDED: td3lag action noise 0.1, M = 1, no bonus; lagu action noise 0, M = 3, bonus)

| Dir / task | n | Return IQM Lag-U | Return IQM TD3-Lag | Diff | P(Lag-U return >) | Cost IQM Lag-U | Cost IQM TD3-Lag | Diff | P(Lag-U cost <) |
|---|---|---|---|---|---|---|---|---|---|
| gate PointGoal1 | 10 / 10 | 6.05 [1.28, 11.18] | 17.62 [13.26, 19.91] | -11.57 [-16.96, -5.03] | 0.12 [0.00, 0.32] | 22.50 [14.40, 46.00] | 44.48 [40.50, 56.55] | -21.98 [-36.00, 1.68] | 0.86 [0.64, 1.00] |
| gate_literal PointGoal1 | 10 / 10 | 27.17 [26.81, 27.55] | 26.39 [25.79, 26.83] | 0.77 [0.22, 1.49] | 0.83 [0.62, 0.97] | 49.27 [47.39, 52.60] | 48.49 [45.43, 52.82] | 0.78 [-3.88, 5.14] | 0.44 [0.18, 0.71] |
| gate_x CarGoal1 | 10 / 5 | 0.61 [-0.15, 4.17] | 17.54 [0.43, 29.08] | -16.93 [-28.45, 0.97] | 0.16 [0.00, 0.42] | 17.19 [7.66, 45.58] | 40.45 [13.95, 56.18] | -23.27 [-40.57, 13.54] | 0.70 [0.40, 1.00] |
| gate_x PointCircle1 | 10 / 5 | 39.77 [33.10, 41.22] | 39.43 [34.77, 42.06] | 0.34 [-7.00, 4.93] | 0.46 [0.14, 0.80] | 25.81 [17.34, 45.25] | 44.27 [16.03, 58.72] | -18.45 [-35.00, 13.20] | 0.66 [0.32, 0.96] |

- gate PointGoal1: cfg (action_noise, M, bonus, gate) {'lagu-tt': [(0.0, 3, True, True)], 'td3lag-tt': [(0.1, 1, False, False)]}; (actor_lr, critic_lr, dual, lambda_lr) {'lagu-tt': [(5e-06, 0.001, 'statewise', 2e-06)], 'td3lag-tt': [(5e-06, 0.001, 'statewise', 2e-06)]}; resumed ['lagu-tt_SafetyPointGoal1-v0_s109']; integrity problems none.
- gate_literal PointGoal1: cfg (action_noise, M, bonus, gate) {'lagu-lit': [(0.0, 3, True, True)], 'td3lag-lit': [(0.1, 1, False, False)]}; (actor_lr, critic_lr, dual, lambda_lr) {'lagu-lit': [(0.0, 0.0, 'statewise', 1e-05)], 'td3lag-lit': [(0.0, 0.0, 'statewise', 1e-05)]}; resumed none; integrity problems none.
- gate_x CarGoal1: cfg (action_noise, M, bonus, gate) {'lagu-tt': [(0.0, 3, True, True)], 'td3lag-tt': [(0.1, 1, False, False)]}; (actor_lr, critic_lr, dual, lambda_lr) {'lagu-tt': [(5e-06, 0.001, 'statewise', 2e-06)], 'td3lag-tt': [(5e-06, 0.001, 'statewise', 2e-06)]}; resumed none; integrity problems none.
- gate_x PointCircle1: cfg (action_noise, M, bonus, gate) {'lagu-tt': [(0.0, 3, True, True)], 'td3lag-tt': [(0.1, 1, False, False)]}; (actor_lr, critic_lr, dual, lambda_lr) {'lagu-tt': [(5e-06, 0.001, 'statewise', 2e-06)], 'td3lag-tt': [(5e-06, 0.001, 'statewise', 2e-06)]}; resumed none; integrity problems none.
- td3lag-lit held-out files present for seeds [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] (10 of 10).
- gate_x td3lag runs beyond s104: none.

## V3. E9 results/pareto seed 2 inventory and 3-seed in-training final window (DESCRIPTIVE)

| Variant | Recipe / dual / λ0 | noise, M | s2 files | rows / last | cols s2 (s0, s1) | snapshots (n, first, last, step) | resumed / diverged / heldout | args differing s0 vs s2 | cfg differing s0 vs s2 | cum_cost 1M |
|---|---|---|---|---|---|---|---|---|---|---|
| td3lag-fix005 | paper / fixed / 0.05 | 0.1, 1 | .ckpt.pt, .csv, .json, .log, .pt, .snap | 20 / 1000000 | 44 (31, 31) | 20, 50000, 1000000, [50000] | False / False / False | ['ckpt_every', 'drop_ckpt', 'seed'] | none | 87697 |
| td3lag-fix01 | paper / fixed / 0.1 | 0.1, 1 | .ckpt.pt, .csv, .json, .log, .pt, .snap | 20 / 1000000 | 44 (31, 31) | 20, 50000, 1000000, [50000] | False / False / False | ['ckpt_every', 'drop_ckpt', 'seed'] | none | 85153 |
| td3lag-fix02 | paper / fixed / 0.2 | 0.1, 1 | .ckpt.pt, .csv, .json, .log, .pt, .snap | 20 / 1000000 | 44 (31, 31) | 20, 50000, 1000000, [50000] | False / False / False | ['ckpt_every', 'drop_ckpt', 'seed'] | none | 80522 |
| td3lag-omni-fix03 | omnisafe / fixed / 0.3 | 0.1, 1 | .ckpt.pt, .csv, .json, .log, .pt, .snap | 20 / 1000000 | 44 (31, 31) | 20, 50000, 1000000, [50000] | False / False / False | ['ckpt_every', 'drop_ckpt', 'seed'] | none | 49610 |
| td3lag-omni-fix1 | omnisafe / fixed / 1 | 0.1, 1 | .ckpt.pt, .csv, .json, .log, .pt, .snap | 20 / 1000000 | 44 (31, 31) | 20, 50000, 1000000, [50000] | False / False / False | ['ckpt_every', 'drop_ckpt', 'seed'] | none | 20982 |
| td3lag-omni | omnisafe / episodic / 0 | 0.1, 1 | .ckpt.pt, .csv, .json, .log, .pt, .snap | 20 / 1000000 | 44 (31, 31) | 20, 50000, 1000000, [50000] | False / False / False | ['ckpt_every', 'drop_ckpt', 'seed'] | none | 50559 |

| Variant | window rows (s0, s1, s2) | Return s0 / s1 / s2 | Return mean ± sd | Cost s0 / s1 / s2 | Cost mean ± sd | Collapsed (window return < 5) | Cost ≤ 25 | λ at 1M | λ never decreases (all seeds) / mean λ rise per 50k over 750k-1M |
|---|---|---|---|---|---|---|---|---|---|
| td3lag-fix005 | [5, 5, 5] | -0.8 / -1.9 / -1.2 | -1.3 ± 0.6 | 10.2 / 64.1 / 133.0 | 69.1 ± 61.5 | 3/3 | 1/3 | 0.0500 / 0.0500 / 0.0500 | True / 0.0000 |
| td3lag-fix01 | [5, 5, 5] | -0.1 / -0.8 / -0.1 | -0.3 ± 0.4 | 71.8 / 63.3 / 110.7 | 82.0 ± 25.3 | 3/3 | 0/3 | 0.1000 / 0.1000 / 0.1000 | True / 0.0000 |
| td3lag-fix02 | [5, 5, 5] | -0.4 / -0.2 / -0.1 | -0.2 ± 0.2 | 73.7 / 48.3 / 107.8 | 76.6 ± 29.9 | 3/3 | 0/3 | 0.2000 / 0.2000 / 0.2000 | True / 0.0000 |
| td3lag-omni-fix03 | [5, 5, 5] | 0.1 / 3.3 / -0.9 | 0.8 ± 2.2 | 31.5 / 25.8 / 53.8 | 37.1 ± 14.8 | 3/3 | 0/3 | 0.3000 / 0.3000 / 0.3000 | True / 0.0000 |
| td3lag-omni-fix1 | [5, 5, 5] | -1.7 / -1.6 / -4.3 | -2.5 ± 1.5 | 0.4 / 0.0 / 0.4 | 0.3 ± 0.2 | 3/3 | 3/3 | 1.0000 / 1.0000 / 1.0000 | True / 0.0000 |
| td3lag-omni | [5, 5, 5] | 20.7 / 24.3 / 24.7 | 23.2 ± 2.2 | 36.0 / 39.4 / 48.0 | 41.1 ± 6.2 | 0/3 | 0/3 | 0.3987 / 0.3988 / 0.3988 | True / 0.0250 |

## V4. Agreement with the audited results.json

- E3 (Table 1, 1b, E1 reference and sensitivity): largest absolute difference 0 (agree)
- Lag-U vs TD3-Lag (Table 2): largest absolute difference 0 (agree)
- E9 (Tables 3-4): largest absolute difference 0 (agree)
- Section 7 quantities left in the audited results.json E3 block (prereg.md 9 says the margins are not applied to E3; they are not printed in tables.md): {'verdict': 0, 'tost_equivalent': 8}.
