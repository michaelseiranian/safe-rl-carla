# safe2 build report (final pre-v0 test pass)

Date: 2026-10-14. Worktree: `safe-rl-carla-method` (branch `method`). No git operations were made.
`lagu/agent.py`, `lagu/train.py`, `lagu/evaluate.py`, `lagu/analyze.py`, `lagu/certify.py` and `prereg.md`
are unchanged. Their mtimes are still the worktree checkout time (2026-10-11 08:34).

This file is in `safe2/`, so `analyze v0-hashes` hashes it with the code (the `safe2/*` glob). Any
edit after v0 is a deviation (see Open issue 2).

## 1. What was built

`safe2/` holds the shared code of the method study (SPEC.md, protocol `prereg_method.md`):

- `base.py`, `configs.py`, `train.py`: the shared off-policy trainer with every baseline config (R1, R2,
  B1-B9, B7b, B7c, SauteA/B, SHL-g{0.90,0.95}-{B4,B5}), PID/Adam duals, budget input, blinded/sealed
  logging, resume/restart/3M continuation rules.
- `bvs.py`: the BVS family (BVS-{SQ,SN}-k{0.02,0.1}, BVS-U, BVS-disc), shield, the projected margin dual
  (`m_tilde = max(-d/2, m_tilde + eta_m (J_e - d))`, amendment MF-3) and counters.
- `posthoc.py`: offline V and pi_b fits, entry recall (G0b, PHS). It now exists, and the PHS path runs
  (section 3).
- `evaluate.py`: the sealed held-out evaluator (deployed, shield_off, strict, s_near, PHS, R3).
- `seal.py`: hybrid RSA-OAEP/AES-256-CTR sealing via the openssl CLI, hash log, fixed-size padding.
- `seeds.py`, `stats.py`, `power.py`: seed/layout blocks, decision statistics, power.
- `probe.py`: the Stage-0 gates (g0a, g0b, s04) and `phs-quality`.
- `analyze.py`: selection, B*, candidates, kill rules (with M7), plan, cuts, v0 hashes, `--unblind`
  (with `--deadline`, M8).
- `launch.py`, `pause.sh`, `health.py`, `health.sh`: queue/launcher, power pausing, public health view.
- `SPEC.md`, `README.md`, `HOOKS.md`, and `tests/`.

Every entry of `analyze.V0_FILES` exists. The only file `v0-hashes` still lacks is
`results/method/keys/seal_pub.pem`.

## 2. Per-file test results (this pass, after the 2026-10-14 amendments)

Every file was run on its own (`python -m pytest -q <file>`, cwd = worktree, `.venv` of `safe-rl-carla`).

| File | Result |
|---|---|
| safe2/tests/test_base.py | 25 passed |
| safe2/tests/test_seal.py | 6 passed |
| safe2/tests/test_seeds.py | 7 passed |
| safe2/tests/test_stats.py | 18 passed |
| safe2/tests/test_train.py | 32 passed |
| safe2/tests/test_evaluate.py | 13 passed |
| safe2/tests/test_bvs.py | 38 passed |
| safe2/tests/test_analyze.py | 25 passed |
| safe2/tests/test_unblind.py | 23 passed |
| safe2/tests/test_blinding.py | 3 passed |
| safe2/tests/test_health.py | 3 passed |
| safe2/tests/test_launch.py | 31 passed |
| safe2/tests/test_probe.py | 41 passed |
| safe2/tests/test_posthoc.py | 9 passed |
| safe2/tests/test_smoke.py | 21 skipped (opt-in: `SAFE2_SMOKE=1`; 20k-30k steps) |
| safe2/tests/test_bvs_smoke.py | 8 skipped (opt-in, same reason) |
| safe2/tests/test_probe_smoke.py | 1 skipped (opt-in, 26k steps) |
| lagu/tests/test_launch.py | 7 passed |
| lagu/tests/test_train.py | 9 passed |
| lagu/tests/test_prereg.py | 5 passed |

In total, 295 tests passed (274 in safe2, 21 in lagu), 30 were skipped (all opt-in) and none failed. This pass changed no code
and no test.

## 3. Smoke runs (10k steps, --hidden 32, after the amendments)

- Driver: `.../method-build/v0prep/final/smoke.py`. It is the earlier `fix/integrator2/smoke.py` with
  the new output directory and a printout of the margin range.
- Setup: dev seed 953 on SafetyPointGoal1-v0, output in `.../method-build/v0prep/smoke/<arm>/`.
- Schedule, as before:
  - start_steps 2000;
  - B5 dual_start 4000;
  - BVS: shield_start 3000 and margin_start 4000;
  - BVS-U: u_start 5000;
  - eval every 5000 steps with 2 episodes, snapshots and checkpoints every 5000 steps.
- BVSU and BVSD were built from BVS-SQ-k0.1 (BVSU with a dev rho* = 0.1).
- Each run ran alone, with OMP_NUM_THREADS=1.

| Arm | steps | updates | steps/s | wall s | finite flags (out/net/loss/dual, both rows) | nonfinite actions | empty whitelisted cols | m_tilde range |
|---|---|---|---|---|---|---|---|---|
| BVS (BVS-SQ-k0.1) | 10000 | 8000 | 404 | 24.8 | all 1 | 0 | 0 | 9.1 .. 12.4 |
| BVS-U (BVSU) | 10000 | 8000 | 328 | 30.5 | all 1 | 0 | 0 | 26.9 .. 52.4 (acting m clipped at d = 25) |
| BVS-disc (BVSD) | 10000 | 8000 | 461 | 21.7 | all 1 | 0 | 0 | 4.6 .. 6.1 |
| B5 | 10000 | 8000 | 539 | 18.5 | all 1 | 0 | 0 | - |

- Steps/s are within noise of the earlier pass (360, 299, 435 and 514).
- The margin floor -d/2 did not bind in these short runs. The floor is covered by test_bvs.py (test 21
  and `test_margin_dual_projection_keeps_the_run_bound_and_bounds_every_window`).

### Post-hoc shield (PHS) on the B5 smoke run

1. **Fit.** `python -m safe2.posthoc fit --ckpt .../smoke/B5/B5_SafetyPointGoal1-v0_s953.ckpt.pt --episodes 1-9 --out .../smoke/phs/B5_SafetyPointGoal1-v0_s953.posthoc.pt --updates 2000`
   - Result: `fit done`, 3.2 s.
   - 2000 updates instead of 100k, to fit the smoke budget.
2. **Recall on held-out episode 10.** `python -m safe2.posthoc recall ... --episodes 10-10`
   - Result: entries 3, caught 3, false_alarm_steps 874, nonfinite_pred 0.
3. **Shielded rollout.** `.../v0prep/final/phs_smoke.py` built `evaluate.phs_policy(B5 snapshot 10000, fit, m = 5.0, rule SQ)` and ran
   `evaluate.rollout` on 2 `dev` layouts with `record_gap=True`.
   - Both episodes ran 1000 steps with no non-finite actions, and shield_stats were filled.
   - Interventions were 898 and 944, empty steps 898 and 944, entries 1 and 2, and 258 and 247 gaps
     were recorded.
   - The high intervention rate is expected from a 2k-update fit at hidden 32. It is a plumbing check,
     not a quality reading.

`evaluate --phs` (the CLI) was not run end to end. It needs `v1.json` and the 1M end-window
snapshots, and test_evaluate.py covers it.

## 4. Runtime per 1M steps (extrapolated; from the earlier pass)

These numbers come from the earlier pass's 10k-step smoke runs at **hidden 32**. This pass's steps/s
agree with them.

- The study uses hidden 256, so these are lower bounds.
- They ignore power-pause time and the eval load.

| Arm | naive h / 1M | shield-adjusted h / 1M |
|---|---|---|
| B1-B9, B7b, B7c, R1, R2, SauteA/B, SHL | 0.51-0.58 | - |
| BVS | 0.77 | ~0.86 |
| BVS-U | 0.93 | ~1.08 |
| BVS-disc | 0.64 | ~0.67 |

A timing run at hidden 256 is still needed before compute is planned.

## 5. Changes since the integration pass (amend step, 2026-10-14; not made by this pass)

- `bvs.py`: the projected margin update (MF-3).
- `analyze.py`:
  - the `approximately_saturated` label;
  - the per-50k window bound;
  - M7, the finite-seed pooling for 8.2, rho* and m_PHS;
  - M8, `--unblind --deadline`.
- Tests:
  - test_bvs.py: test 21 and the new window-bound test;
  - test_analyze.py: 2 new M7 tests;
  - test_unblind.py: the deadline test.
- `prereg_method.md`, `SPEC.md`, `README.md` and `HOOKS.md` were aligned with the code. The protocol has
  a new closing section, "Pre-v0 amendments (2026-10-14)", with 26 entries.
- Copies of the pre-amend files are in `.../method-build/v0prep/v0align/`.

The `analyze.py` edits were outside the amend step's given scope (`bvs.py`, `base.py`). **They need
the lead's sign-off.**

## 6. Open issues

**Blocking method-v0.**

1. **`results/method/keys/seal_pub.pem` does not exist.** The lead creates the study key pair. The
   public key is hashed at v0, and `analyze v0-hashes` refuses until it exists. It is the only missing
   input.
2. **This file is hashed at v0** (the `safe2/*` glob). The lead must choose one of two options:
   - keep it as is, in which case it is in the commit list below and any later edit is a
     deviations.json entry;
   - move it out of `safe2/` before running `v0-hashes`, in which case drop it from the list.
3. **Sign-off on the amend step's `analyze.py` edits** (section 5) and on the MF-3 behaviour change in
   `bvs.py`. Both were made before any BVS run.

**Not blocking, but read before tagging.**

4. **The evaluator does not record BVS-U triggers at deployment.** §3.5 therefore says only that the
   held-out intervention rate bounds the deployed trigger rate from below. A per-episode trigger count
   in `evaluate.rollout` would let it be reported directly. Adding it after v0 would be a deviation.
5. **`probe g0a|g0b --force` writes an incomplete `deviations.json` entry.** The entry has no file or
   function, and its text is `None`. `--unblind` refuses until the lead completes it. This is
   documented in SPEC §9 and the README. It is a trap, not a defect.
6. **Sauté (SF-1) is disclosed as a one-step-lag variant (§3.1), not fixed** (`train.py` was out of
   scope).
7. **The opt-in smoke files have not been run** (test_smoke.py, test_bvs_smoke.py, test_probe_smoke.py).
   They need 20k-30k steps, which is above this pass's 10k cap, and they include the BVS-U trig007_num
   assertion. Run each one alone with `SAFE2_SMOKE=1 SAFE2_SMOKE_OUT=<scratch>` before v0.
8. **Runtime at hidden 256 is unmeasured** (section 4).
9. **The integration pass's non-blocking items were not re-checked here**:
   - analysis M6 part 2 (`--unblind` does not re-derive B*/BVS*/SHL* with the v0 module);
   - no test of a deviation that flips the verdict;
   - `relogged_records` does not separate resume truncation;
   - stage0 F5 item 3 (`analyze.bstar` checks of the probe settings and hash);
   - s04 has not been run on the real legacy checkpoints.

## 7. File list for the method-v0 commit

All of these exist in the worktree. Exclude every `__pycache__/`.

```
prereg_method.md
safe2/__init__.py
safe2/analyze.py
safe2/base.py
safe2/bvs.py
safe2/configs.py
safe2/evaluate.py
safe2/health.py
safe2/health.sh
safe2/launch.py
safe2/pause.sh
safe2/posthoc.py
safe2/power.py
safe2/probe.py
safe2/seal.py
safe2/seeds.py
safe2/stats.py
safe2/train.py
safe2/SPEC.md
safe2/README.md
safe2/HOOKS.md
safe2/BUILD_REPORT.md
safe2/tests/__init__.py
safe2/tests/analyze_helpers.py
safe2/tests/fake_posthoc.py
safe2/tests/helpers.py
safe2/tests/test_analyze.py
safe2/tests/test_base.py
safe2/tests/test_blinding.py
safe2/tests/test_bvs.py
safe2/tests/test_bvs_smoke.py
safe2/tests/test_evaluate.py
safe2/tests/test_health.py
safe2/tests/test_launch.py
safe2/tests/test_posthoc.py
safe2/tests/test_probe.py
safe2/tests/test_probe_smoke.py
safe2/tests/test_seal.py
safe2/tests/test_seeds.py
safe2/tests/test_smoke.py
safe2/tests/test_stats.py
safe2/tests/test_train.py
safe2/tests/test_unblind.py
```

That is 43 files: `prereg_method.md`, 21 under `safe2/` and 21 under `safe2/tests/`. Drop
`safe2/BUILD_REPORT.md` if Open issue 2 is resolved by moving it.

These do not exist yet and must be added (force-added, since `results/` is git-ignored) before v0:

- `results/method/keys/seal_pub.pem`;
- `results/method/PREREG_SHA256.txt` and `results/method/FUNC_SHA256.json`. These are written by
  `python -m safe2.analyze v0-hashes` once the key exists, and after the final edit to any file in
  `safe2/`.
