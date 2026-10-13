# E3, Lag-U vs TD3-Lag, E9 pareto (DESCRIPTIVE; produced by results/paper/e3_td3lag_e9/make.py)

## Table 1. E3 per task: lagu-tt minus lagu_nogate-tt, held-out primary endpoint (DESCRIPTIVE, secondary prediction, prereg.md 9; NOT BLIND: E3 contrasts were first computed by an audit after G3)

Paired by seed; mean over seeds; A - B with 95% paired bootstrap interval (10,000 resamples). Collapsed = held-out return < 5 or diverged (raw task units, not rescaled), Fisher exact two-sided. Engagement and D are for the gated arm (lagu-tt), pooled over its seeds. The section 7 margins and labels are not applied to E3. The E1 row is the confirmatory H3 contrast, repeated for reference.

| Task | Pairs | Return gate | Return no gate | Δreturn [95%] | Cost gate | Cost no gate | Δcost [95%] | Collapsed gate / no gate (Fisher p) | Engagement 25k-300k | Engagement whole run | Mean budget D (D/δ0) | Signs (Δreturn, Δcost) | Same signs as E1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PointGoal1 (E1, reference) | 15 (s100-114) | 5.42 | 15.59 | -10.17 [-16.08, -3.80] | 34.46 | 37.77 | -3.31 [-18.75, 16.70] | 9 / 3 (0.060) | 0.850 | 0.913 | 1.122 (0.449, δ0 = 2.5) | (-, -) | reference |
| CarGoal1 (E3) | 10 (s100-109) | 2.10 | 10.15 | -8.05 [-15.78, -0.56] | 26.91 | 34.32 | -7.41 [-29.36, 19.26] | 8 / 5 (0.350) | 0.861 | 0.941 | 1.007 (0.403, δ0 = 2.5) | (-, -) | return yes, cost yes |
| PointCircle1 (E3) | 10 (s100-109) | 36.88 | 33.37 | 3.51 [-7.57, 13.58] | 32.88 | 36.35 | -3.47 [-32.22, 24.39] | 0 / 0 (1.000) | 0.379 | 0.250 | 4.214 (0.843, δ0 = 5) | (+, -) | return no, cost yes |

- CarGoal1: section 9 prediction (both mean paired differences have E1's signs, -/-): **holds** (Δreturn -, Δcost -).
- PointCircle1: section 9 prediction (both mean paired differences have E1's signs, -/-): **does not hold** (Δreturn +, Δcost -).
- PointGoal1 (E1): pairs with a resumed run: [100, 101, 102, 103, 104, 105, 109, 110, 111, 112, 113, 114]; without them (3 pairs) Δreturn -15.86 [-21.19, -11.17], Δcost 20.99 [-27.61, 98.79].
- CarGoal1: no resumed runs.
- PointCircle1: no resumed runs.

## Table 1b. E3 per seed: held-out return and cost (DESCRIPTIVE; NOT BLIND, see Table 1)

| Task | Seed | Return gate | Return no gate | Δreturn | Cost gate | Cost no gate | Δcost | Collapsed (gate, no gate) |
|---|---|---|---|---|---|---|---|---|
| CarGoal1 | 100 | -0.05 | 19.35 | -19.40 | 0.69 | 45.35 | -44.66 | (yes, no) |
| CarGoal1 | 101 | 11.74 | 2.34 | 9.40 | 71.04 | 77.42 | -6.38 | (no, yes) |
| CarGoal1 | 102 | -0.87 | 2.32 | -3.19 | 13.49 | 65.66 | -52.17 | (yes, yes) |
| CarGoal1 | 103 | -0.01 | 19.22 | -19.23 | 2.98 | 42.53 | -39.55 | (yes, no) |
| CarGoal1 | 104 | 0.01 | 21.86 | -21.85 | 25.50 | 39.45 | -13.95 | (yes, no) |
| CarGoal1 | 105 | -0.25 | 27.10 | -27.35 | 3.32 | 38.66 | -35.34 | (yes, no) |
| CarGoal1 | 106 | 6.66 | -0.03 | 6.69 | 91.24 | 0.00 | 91.24 | (no, yes) |
| CarGoal1 | 107 | 1.17 | 9.94 | -8.77 | 23.33 | 23.67 | -0.34 | (yes, no) |
| CarGoal1 | 108 | 2.52 | -0.17 | 2.69 | 19.20 | 9.28 | 9.92 | (yes, yes) |
| CarGoal1 | 109 | 0.03 | -0.48 | 0.51 | 18.28 | 1.15 | 17.13 | (yes, yes) |
| PointCircle1 | 100 | 38.62 | 42.58 | -3.96 | 10.50 | 37.38 | -26.88 | (no, no) |
| PointCircle1 | 101 | 40.62 | 33.53 | 7.09 | 48.70 | 25.35 | 23.35 | (no, no) |
| PointCircle1 | 102 | 38.89 | 39.92 | -1.03 | 22.73 | 38.52 | -15.79 | (no, no) |
| PointCircle1 | 103 | 14.92 | 46.48 | -31.56 | 13.11 | 106.62 | -93.51 | (no, no) |
| PointCircle1 | 104 | 41.32 | 32.46 | 8.87 | 39.51 | 1.89 | 37.62 | (no, no) |
| PointCircle1 | 105 | 40.53 | 29.35 | 11.18 | 18.77 | 1.22 | 17.55 | (no, no) |
| PointCircle1 | 106 | 45.25 | 39.06 | 6.20 | 101.62 | 20.93 | 80.69 | (no, no) |
| PointCircle1 | 107 | 40.54 | 15.11 | 25.43 | 28.62 | 45.35 | -16.73 | (no, no) |
| PointCircle1 | 108 | 28.68 | 44.16 | -15.48 | 27.21 | 73.50 | -46.29 | (no, no) |
| PointCircle1 | 109 | 39.44 | 11.04 | 28.41 | 18.05 | 12.76 | 5.29 | (no, no) |

## Table 2. Lag-U vs TD3-Lag, held-out primary endpoint, unpaired (DESCRIPTIVE only, prereg.md 6; CONFOUNDED: td3lag arms use action noise 0.1 and M = 1, lagu uses action noise 0 and M = 3 with the exploration bonus)

IQM over seeds; stratified (per-arm) bootstrap 95% intervals, 10,000 resamples. P(return) = P(Lag-U return > TD3-Lag return), P(cost) = P(Lag-U cost < TD3-Lag cost), over all seed pairs, ties counted 1/2.

| Directory | Task | n Lag-U / TD3-Lag | Return IQM Lag-U | Return IQM TD3-Lag | Return IQM diff | P(return) | Cost IQM Lag-U | Cost IQM TD3-Lag | Cost IQM diff | P(cost) |
|---|---|---|---|---|---|---|---|---|---|---|
| gate (E1 recipe, 2M): lagu-tt vs td3lag-tt | PointGoal1 | 10 / 10 | 6.05 [1.28, 11.18] | 17.62 [13.26, 19.91] | -11.57 [-16.96, -5.03] | 0.12 [0.00, 0.32] | 22.50 [14.40, 46.00] | 44.48 [40.50, 56.55] | -21.98 [-36.00, 1.68] | 0.86 [0.64, 1.00] |
| gate_literal (literal recipe, 1M): lagu-lit vs td3lag-lit | PointGoal1 | 10 / 10 | 27.17 [26.81, 27.55] | 26.39 [25.79, 26.83] | 0.77 [0.22, 1.49] | 0.83 [0.62, 0.97] | 49.27 [47.39, 52.60] | 48.49 [45.43, 52.82] | 0.78 [-3.88, 5.14] | 0.44 [0.18, 0.71] |
| gate_x (E1 recipe, 2M): lagu-tt vs td3lag-tt | CarGoal1 | 10 / 5 | 0.61 [-0.15, 4.17] | 17.54 [0.43, 29.08] | -16.93 [-28.45, 0.97] | 0.16 [0.00, 0.42] | 17.19 [7.66, 45.58] | 40.45 [13.95, 56.18] | -23.27 [-40.57, 13.54] | 0.70 [0.40, 1.00] |
| gate_x (E1 recipe, 2M): lagu-tt vs td3lag-tt | PointCircle1 | 10 / 5 | 39.77 [33.10, 41.22] | 39.43 [34.77, 42.06] | 0.34 [-7.00, 4.93] | 0.46 [0.14, 0.80] | 25.81 [17.34, 45.25] | 44.27 [16.03, 58.72] | -18.45 [-35.00, 13.20] | 0.66 [0.32, 0.96] |

- gate PointGoal1: Lag-U seeds s100-109, TD3-Lag seeds s100-109; held-out file for every run: yes. Config: lagu-tt: action_noise 0, M 3, bonus True, gate True; td3lag-tt: action_noise 0.1, M 1, bonus False, gate False.
- gate_literal PointGoal1: Lag-U seeds s100-109, TD3-Lag seeds s100-109; held-out file for every run: yes. Config: lagu-lit: action_noise 0, M 3, bonus True, gate True; td3lag-lit: action_noise 0.1, M 1, bonus False, gate False.
- gate_x CarGoal1: Lag-U seeds s100-109, TD3-Lag seeds s100-104; held-out file for every run: yes. Config: lagu-tt: action_noise 0, M 3, bonus True, gate True; td3lag-tt: action_noise 0.1, M 1, bonus False, gate False.
- gate_x PointCircle1: Lag-U seeds s100-109, TD3-Lag seeds s100-104; held-out file for every run: yes. Config: lagu-tt: action_noise 0, M 3, bonus True, gate True; td3lag-tt: action_noise 0.1, M 1, bonus False, gate False.
- td3lag-lit held-out check (prereg.md 16 item 5): 10 of 10 runs present (seeds [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]), missing held-out files: none.
- With 5 TD3-Lag seeds (gate_x) the IQM keeps the middle 3 runs, so its interval is coarse.

## Table 3. E9: what results/pareto holds for seed 2 (the pre-registered top-up; DESCRIPTIVE)

All runs SafetyPointGoal1-v0, td3lag (M = 1, action noise 0.1), 1M steps. `paper` recipe = literal learning rates (actor and critics 3e-4); `omnisafe` = actor 5e-6, critics 1e-3. No seed-2 run was resumed or diverged unless stated; no run in results/pareto has a held-out file.

| Variant | Recipe | Dual | λ (fixed or initial) | Rows / last step | Files | Snapshots | Columns (s2 / s0-s1) | cum_cost at 1M |
|---|---|---|---|---|---|---|---|---|
| td3lag-fix005 | paper | fixed | 0.05 | 20 / 1000000 | .csv, .json, .log, .pt, .ckpt.pt | 20 (50000-1000000, every 50k) | 44 / 31 | 87697 |
| td3lag-fix01 | paper | fixed | 0.1 | 20 / 1000000 | .csv, .json, .log, .pt, .ckpt.pt | 20 (50000-1000000, every 50k) | 44 / 31 | 85153 |
| td3lag-fix02 | paper | fixed | 0.2 | 20 / 1000000 | .csv, .json, .log, .pt, .ckpt.pt | 20 (50000-1000000, every 50k) | 44 / 31 | 80522 |
| td3lag-omni-fix03 | omnisafe | fixed | 0.3 | 20 / 1000000 | .csv, .json, .log, .pt, .ckpt.pt | 20 (50000-1000000, every 50k) | 44 / 31 | 49610 |
| td3lag-omni-fix1 | omnisafe | fixed | 1 | 20 / 1000000 | .csv, .json, .log, .pt, .ckpt.pt | 20 (50000-1000000, every 50k) | 44 / 31 | 20982 |
| td3lag-omni | omnisafe | episodic | 0 | 20 / 1000000 | .csv, .json, .log, .pt, .ckpt.pt | 20 (50000-1000000, every 50k) | 44 / 31 | 50559 |

- Seed-2 runs resumed, diverged or short of 1M: none. Seeds 0-1 were written with the older 31-column CSV schema (no cum_cost, lambda_avg, resumed); their args and cfg equal seed 2's except seed, ckpt_every and drop_ckpt.

## Table 4. Pareto table: TD3-Lag fixed-λ and OmniSafe-recipe variants, in-training final window, 3 seeds (DESCRIPTIVE; fixed-λ rows are evidence of stationary-λ exploitation only, prereg.md 13)

Final window = mean of eval_return / eval_cost over rows with step >= 800k (5 rows, 10 episodes each, in-training layouts EVAL_SEED + k). Mean ± sd (ddof 1) over seeds 0-2; collapsed = window return < 5; final λ at 1M per seed.

| Variant | Recipe | Dual | λ | Return s0 / s1 / s2 | Return mean ± sd (3 seeds) | Return mean s0-s1 | Cost s0 / s1 / s2 | Cost mean ± sd (3 seeds) | Cost mean s0-s1 | Collapsed | Cost ≤ 25 | Final λ s0 / s1 / s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| td3lag-fix005 | paper | fixed | 0.05 | -0.8 / -1.9 / -1.2 | -1.3 ± 0.6 | -1.3 | 10.2 / 64.1 / 133.0 | 69.1 ± 61.5 | 37.2 | 3/3 | 1/3 | 0.050 / 0.050 / 0.050 |
| td3lag-fix01 | paper | fixed | 0.1 | -0.1 / -0.8 / -0.1 | -0.3 ± 0.4 | -0.5 | 71.8 / 63.3 / 110.7 | 82.0 ± 25.3 | 67.6 | 3/3 | 0/3 | 0.100 / 0.100 / 0.100 |
| td3lag-fix02 | paper | fixed | 0.2 | -0.4 / -0.2 / -0.1 | -0.2 ± 0.2 | -0.3 | 73.7 / 48.3 / 107.8 | 76.6 ± 29.9 | 61.0 | 3/3 | 0/3 | 0.200 / 0.200 / 0.200 |
| td3lag-omni-fix03 | omnisafe | fixed | 0.3 | 0.1 / 3.3 / -0.9 | 0.8 ± 2.2 | 1.7 | 31.5 / 25.8 / 53.8 | 37.1 ± 14.8 | 28.7 | 3/3 | 0/3 | 0.300 / 0.300 / 0.300 |
| td3lag-omni-fix1 | omnisafe | fixed | 1 | -1.7 / -1.6 / -4.3 | -2.5 ± 1.5 | -1.7 | 0.4 / 0.0 / 0.4 | 0.3 ± 0.2 | 0.2 | 3/3 | 3/3 | 1.000 / 1.000 / 1.000 |
| td3lag-omni | omnisafe | episodic | 0 | 20.7 / 24.3 / 24.7 | 23.2 ± 2.2 | 22.5 | 36.0 / 39.4 / 48.0 | 41.1 ± 6.2 | 37.7 | 0/3 | 0/3 | 0.399 / 0.399 / 0.399 |

