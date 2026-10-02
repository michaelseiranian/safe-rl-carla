# Lag-U final experiment plan (synthesis of the three designs and both judge verdicts)

Both judges picked the "faithful" design, and it is the base here. I added the grafts both judges agreed on and fixed every fatal flaw either judge raised. Nothing has been launched.

## Checks I ran today (29 Sep, 21:45)

- **Pareto wave.** It is still running, at 850–950k of 1M, at 70–82 steps/s. Load average is 20–30 and swap is 2.58 of 4 GB. It should exit around 23:00.
- **Fixed λ = 0.3 at OmniSafe learning rates.** Both seeds have now collapsed (return 0.0 and 3.3 at 900k).
- **OmniSafe adaptive ramp.** At 850k, λ = 0.324 and return is 22.1 / 26.5. The cost critic reads 0.72 / 0.81 of the realised cost-to-go (pooled over the last 4 evals). The E0 calibration band below is set from these numbers.
- **Code facts:**
  - `lagu/` is untracked in git.
  - scipy is not installed in `.venv`.
  - `pi_stats` logs only the last policy update of each interval.
  - `engagement` is counted only when the gate is on.
- **Task facts:**

| Task | Horizon | Obs dim | δ0 |
|---|---|---|---|
| SafetyPointGoal1-v0 | 1000 | 60 | 2.5 |
| SafetyCarGoal1-v0 | 1000 | 72 | 2.5 |
| SafetyPointCircle1-v0 | 500 | 28 | 5.0 (set automatically) |

- **Bit-identity reasoning holds in `agent.py`.**
  - A detached δ enters only through `λ*(qc − δ)`. Its backward pass does not depend on δ.
  - RNG use (`rng.random`, `randn_like`) is the same with the gate on or off.
  - Under Eq. 18, `gap = mean(qc) − mean(δ_used)`, so the gate works exactly like a time-varying uniform budget.

## Fatal flaws and how the plan fixes them

| Flaw (who raised it) | Fix |
|---|---|
| E0's "binding" test is in critic units only (J1) | E0 now also needs a realised-cost condition (d) |
| PointButton1 has no OmniSafe reference recipe (J1, J2) | E3 uses CarGoal1 + PointCircle1; `cum_cost` is logged directly |
| ±5 / ±2.5 margins may be out of reach; Welch test used although seeds are shared (J1, J2) | Paired bootstrap and Wilcoxon; five-way verdict rule; one extension; report CIs as bounds; collapses analysed separately |
| "Identical CSVs" integration test would fail (J2) | Compare tensor hashes and `eval_*` columns only |
| Calibration of the "tt" cells is untested; the band is borderline (J1, J2) | Keep the omsw bridge cell; pooled band [0.65, 1.35]; C3 stated as a hypothesis |
| E0's window may be empty (J2) | Pre-stated outcome (iv), plus the E0-ext early-warning check |
| Taken from other designs | Matched budget D is weighted over **all** policy updates, not risk-only (minimal's E5 flaw); selection seeds 0–2 are disjoint from confirmatory seeds 100+ (minimal's overlap); the step budget is 1M or 2M by rule (minimal's hard cap); the endpoint is a 5-snapshot window, not one checkpoint (minimal); λ-lr is selected on lagu_nogate, not td3lag (reviewer); no raw ±10 pooling across tasks (reviewer) |
| Fixed-λ frontier used as a bound on adaptive duals (J1 guardrail, J2) | Stated as a rule. Fixed runs are "stationary-λ exploitation evidence" only, because of path dependence (fixed 0.05 collapsed where adaptive 0.05 kept 26.8; fixed 0.3 collapsed 2 of 2 where the ramp at 0.32 kept 22–26) |

**Grafts applied:** PointCircle1; shadow no-gate λ; replay-state E_D[Q_c] interval means; 150k SHA-256 certificate with positive controls; `--max-procs` queue; replay tensor views, `ckpt_every` 250k and `--drop-ckpt`; optional OmniSafe anchor; gradient-parity probe; analytic gate-grad test; 100-episode held-out evaluation; realised-cost E0 condition; the |Q̄| feedback-loop prediction; public pre-registration tags; E9 top-up; October freeze.

---

## (1) Research question and pre-registered hypotheses

**Research question.** Suppose Lag-U's critic-unit dual (Eq. 18) can bind on Safety-Gymnasium with a calibrated cost critic. Does the uncertainty-gated budget δ_ada (Eq. 16–17) then improve the return–cost trade-off compared with:
- (a) the same agent without the gate, and
- (b) the same agent given a uniform budget equal to the gate's average?

And what does the gate do under the literal published recipe?

**Primary endpoint for every contrast.** Per seed: mean return and mean cost over 100 held-out deterministic episodes. These are 5 snapshots, evenly spaced over the last 20% of training, × 20 layouts each. The layouts are EVAL_SEED + 1000 + 20i + k, disjoint from the in-training layouts and shared by all runs.

**Secondary endpoints:**
- mean of the in-training eval rows over the last 20%;
- cumulative training cost (the `cum_cost` column), over the whole run and at 500k;
- collapse (held-out return < 5);
- λ_avg, engagement and calibration.

**Verdict rule.** Contrasts are A − B, paired by seed. CIs are paired bootstrap with 10k resamples. The first label that matches applies.

| Label | Condition |
|---|---|
| Helps | (Δcost 95% upper bound < 0 and Δreturn 95% lower bound > −2.5), or (Δreturn lower bound > 0 and Δcost upper bound < +5) |
| Hurts | the mirror image of Helps, or A has ≥ 3 more collapsed seeds than B |
| Shift | both CIs exclude 0 with the same sign (movement along the trade-off) |
| Equivalent | both 90% CIs inside ±5 cost and ±2.5 return (TOST) |
| Inconclusive | none of the above. Allows exactly one extension of +5 seeds (115–119); after that, CIs are reported as effect bounds |

- Wilcoxon (exact) and Fisher exact on collapses are also reported.
- Holm correction runs over the primary p-values of H2–H5. H1 is deterministic.
- Power assumes no gain from pairing. With a between-seed SD of about 6 (Lag-U arms in the pilot), the 15-pair 95% half-width is about 4.7 cost. With an SD of 13 it is about 10, which means equivalence can only be claimed for true differences near 0.

**Hypotheses**

**H1 — channel (deterministic).**
- *Claim:* under the fixed and realised-cost episodic duals, lagu and lagu_nogate from the same seed have bit-identical network and optimizer tensors and identical `eval_*` columns. The positive controls (statewise with λ_init 0.1; gate_grad = 1) differ.
- *Confirm:* all 4 inert pairs hash-equal, both controls differ, and the liveness checks pass.
- *Refute:* any inert pair differs. That means a hidden channel or a bug; stop before E1.

**H2 — the gate acts through λ.**
- *Claim:* in E1 and E1b, lagu's engagement is above 10% on risk-sensitive samples over 25k–300k. Its time-averaged λ_avg exceeds lagu_nogate's (one-sided paired Wilcoxon, α = 0.05).
- *Confirm:* both hold.
- *Refute:* engagement below 10% (E1 is vacuous; E4's T-family replaces the matched arm), or λ_lagu ≤ λ_nogate.

**H3 — primary: the gate's effect on outcomes (E1, calibrated recipe, seeds 100–114).**
- *Prediction, from F1:* no Pareto improvement.
- *Confirm:* verdict is Equivalent, Shift or Hurts.
- *Refute (supports Lag-U's claim):* verdict is Helps.

**H4 — adaptivity vs uniform tightening (E2).**
- *Claim:* lagu-tt is equivalent to lagu_nogate-tt-match, where δ0 = D (defined in E2).
- *Confirm:* Equivalent, or Inconclusive with the CI containing 0.
- *Refute:* Helps (the timing of tightening matters), or Hurts (early over-tightening costs return).

**H5 — calibration precondition.**
- *Claim, literal recipe (E1b), both arms:*
  - held-out cost IQM 95% lower bound > 25;
  - |gap_avg| ≤ 0.25·δ0 over the last 20% (the constraint is met in critic units);
  - pooled qc_traj / mc_cost_traj < 0.65 (95% upper bound < 1).
- *Claim, E1 lagu_nogate:* pooled calibration within [0.65, 1.35].
- *Refute:* literal-recipe calibration ≥ 0.65 or literal-recipe cost ≤ 25.

**Pre-registered secondary predictions (outside the Holm family):**
- E3 reproduces the sign of the E1 differences on CarGoal1 and PointCircle1.
- |Q̄| feedback loop: if the median interval `qbar_avg` falls below 0.3 after 300k in lagu-tt, engagement exceeds 80% there and λ_lagu / λ_nogate exceeds 3 by the end.
- The frozen-λ return-halving point lies within a factor of 2 of λ* = ‖∇aQ̄‖ / ‖∇aQc‖.
- E4 shows a monotone dose-response in both knobs.

---

## (2) Waves, commands and gates

**Throughput assumed.** At 12-way on an idle machine: Lag-U 55 steps/s, TD3-Lag 110 steps/s. The tables give idle wall time and wall time at 0.6× throughput (the contention J1 measured). CarGoal1 is assumed 15% slower; its first run records the real rate.

| Run | Idle | 0.6× |
|---|---|---|
| Lag-U 1M | 5.1 h | 8.4 h |
| Lag-U 2M | 10.1 h | 16.8 h |
| TD3-Lag 1M | 2.5 h | 4.2 h |
| TD3-Lag 2M | 5.1 h | 8.4 h |

**Common setup for all waves:**
```
cd /Users/michaelseiranian/Desktop/safe-rl-carla; PY=.venv/bin/python; K=ckpt_every=250000,drop_ckpt=1
```

### Gate G0 (30 Sep – 1 Oct, no training)
All of the following must hold before Wave 1:
- Code changes 0–7 are done.
- `$PY -m pytest lagu/tests -q` is fully green.
- A 20k-step smoke run on each of the 3 tasks writes finite new columns, snapshots and `heldout.json`, and PointCircle1's JSON shows δ0 = 5.0. The smoke command is `$PY -m lagu.launch --max-procs 3 --out results/smoke --env <task> --arms lagu --seeds 0 --steps 20000 --eval-every 10000 --variant tt:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=2e-6`.
- The scheduler never exceeds its cap.
- `--dry-run` of every Wave-1 line prints the intended flags.
- The pareto wave has exited (`pgrep -f 'm lagu[.]train'` is empty).
- prereg.md, the E0 rule and the analysis code are committed and tagged `prereg-v1`.
- The machine is on AC with the lid open and load average below 3.

If any item fails, fix it and do not launch.

### Wave 1 — certificate W0, then E0 (launch evening of 1 Oct; ~6 h idle, ~10 h at 0.6×)
Certificate (11 runs × 150k, ~0.8 h):
```
$PY -m lagu.launch --max-procs 12 --out results/certify --arms lagu lagu_nogate --seeds 0 1 --steps 150000 --variant fix:dual=fixed,lambda_init=0.1,drop_ckpt=1 --variant epi:recipe=omnisafe,dual_start=30000,lambda_init=0.1,drop_ckpt=1
$PY -m lagu.launch --max-procs 12 --out results/certify --arms lagu lagu_nogate --seeds 0 --steps 150000 --variant sw:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=1e-5,lambda_init=0.1,drop_ckpt=1
$PY -m lagu.launch --max-procs 12 --out results/certify --arms lagu --seeds 0 --steps 150000 --variant epigg:recipe=omnisafe,dual_start=30000,lambda_init=0.1,gate_grad=1,drop_ckpt=1
```
E0: lagu_nogate only, selection seeds 0–2, 1M steps, 12 runs. There is no `drop_ckpt`, so E0-ext can resume these runs.
```
$PY -m lagu.launch --max-procs 12 --out results/e0_dual --arms lagu_nogate --seeds 0 1 2 --steps 1000000 \
 --variant tt-l1e-5:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=1e-5,ckpt_every=250000 \
 --variant tt-l2e-6:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=2e-6,ckpt_every=250000 \
 --variant tt-l4e-7:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=4e-7,ckpt_every=250000 \
 --variant omsw-l4e-7:recipe=omnisafe,dual=statewise,lambda_lr=4e-7,ckpt_every=250000
$PY -m lagu.certify results/certify; $PY -m lagu.analyze results/e0_dual --e0-rule
```

### Gate G1 (2 Oct)
**Certificate outcome:**
- Expectations met: proceed.
- An inert pair differs: STOP. Reproduce it in a unit test, fix, rerun W0.
- A positive control comes out equal: rerun W0 at 300k before proceeding.

**E0 rule.** It is mechanical and committed in advance. All conditions use the evals at 850k–1M; each seed contributes its last 4 rows.
- (a) **Calibration.** Pooled Σqc_traj / Σmc_cost_traj is in [0.65, 1.35]. Each seed is at least 0.5. No row has qc_traj < 0.
- (b) **No collapse.** The cell's mean eval return is at least 20 and no seed is below 10.
- (c) **Binding in critic units.** λ_avg ≥ 0.02, and either |gap_avg| ≤ 0.3 or λ_avg rose by more than 20% between 600k and 1M.
- (d) **Binding in realised cost.** Mean eval cost ≤ 35, or mean train_cost over 850k–1M is at least 10 below its 250–400k mean.

**Choice of recipe and step size:**
- (i) Pick the largest λ-lr among "tt" cells that pass all four conditions.
- (ii) If no tt cell passes but omsw passes, use omsw.
- (iii) If no cell meets (c) and (d), use the fastest cell that passes (a) and (b), at 2M steps.
- (iv) If fast cells collapse and slow cells never bind, use the fastest non-collapsing cell at 2M, and the paper leads with the negative result.
- (v) If no cell is calibrated, STOP and debug. Only E1b may run meanwhile.

**Step budget.** Use 1M if the chosen cell's λ_avg changes by less than 20% over 600k–1M and |gap_avg| ≤ 0.3. Otherwise use 2M.

**Record and tag.** Record `would_engage` and `qbar_avg` against the R2 and |Q̄| predictions. Commit the decision as `prereg-v1.1`; you then push both tags publicly.

### Wave 2 — E0-ext, E1, E1b, E3, TD3-Lag anchors, E9 (enqueued longest-first)

| Block | Runs | Idle | 0.6× |
|---|---|---|---|
| E0-ext | 3 | | |
| E1 | 40 | 30 h (done first) | 50 h |
| E1b | 30 | | |
| E3 | 50 | | |
| E9 | 6 | | |
| **Total at 2M** | **129 runs, ~1,000 process-hours** | **83 h** | **139 h** |

At 1M the wave needs about 45 h idle or 75 h at 0.6×.

```
LR=<G1>; STEPS=<G1>; V="tt:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=$LR,$K"   # if (ii): V="omsw:recipe=omnisafe,dual=statewise,lambda_lr=4e-7,$K"
# E0-ext, only if STEPS=2000000 (early warning on selection seeds):
$PY -m lagu.launch --max-procs 12 --out results/e0_dual --arms lagu_nogate --seeds 0 1 2 --steps 2000000 --resume --variant <chosen E0 variant string verbatim>
# E1 (primary; lagu first so D is available early)
$PY -m lagu.launch --max-procs 12 --out results/gate --arms lagu --seeds {100..114} --steps $STEPS --variant "$V"
$PY -m lagu.launch --max-procs 12 --out results/gate --arms lagu_nogate --seeds {100..114} --steps $STEPS --variant "$V"
$PY -m lagu.launch --max-procs 12 --out results/gate --arms td3lag --seeds {100..109} --steps $STEPS --variant "$V"
# E3 (frozen recipe, no re-tuning; PointCircle1 gets delta0=5.0 automatically)
for E in SafetyCarGoal1-v0 SafetyPointCircle1-v0; do
 $PY -m lagu.launch --max-procs 12 --out results/gate_x --env $E --arms lagu lagu_nogate --seeds {100..109} --steps $STEPS --variant "$V"
 $PY -m lagu.launch --max-procs 12 --out results/gate_x --env $E --arms td3lag --seeds {100..104} --steps $STEPS --variant "$V"; done
# E1b (literal published recipe, 1M)
$PY -m lagu.launch --max-procs 12 --out results/gate_literal --arms lagu lagu_nogate td3lag --seeds {100..109} --steps 1000000 --variant lit:$K
# E9 (pareto seed-2 top-up, last in queue)
$PY -m lagu.launch --max-procs 12 --out results/pareto --arms td3lag --seeds 2 --steps 1000000 --variant fix005:dual=fixed,lambda_init=0.05 --variant fix01:dual=fixed,lambda_init=0.1 --variant fix02:dual=fixed,lambda_init=0.2 --variant omni-fix03:recipe=omnisafe,dual=fixed,lambda_init=0.3 --variant omni-fix1:recipe=omnisafe,dual=fixed,lambda_init=1.0 --variant omni:recipe=omnisafe
```

### Gate G2 (once all 15 lagu-tt runs finish; ~Oct 3 idle, ~Oct 4 at 0.6×)
This gate reads lagu-only information. No contrast is looked at yet.
- Compute D with `D=$($PY -m lagu.analyze results/gate --budget-D lagu-tt)`, and the engagement over 25k–300k.
- **Engagement ≥ 10% and D ≤ 2.4:** enqueue E2 at the front of the queue.
- **D > 2.4:** skip the matched arm. Enqueue ggrad plus E4's T-family.
- **Engagement < 10%:** E1 is vacuous for H3 but is still reported. Skip the matched arm. Enqueue ggrad plus E4 at T ∈ {0.035, 0.0175}.
- **E0-ext check (2M only):** if at least 2 of the 3 selection seeds have mean eval return below 5 over 1.6–2.0M, the primary window for all PointGoal1 contrasts switches to 0.8–1.0M (`--end-step 1000000`). This is recorded before any E1 contrast is computed.

### Wave 3 — E2 mechanism (25 runs; 21 h idle, 35 h at 0.6×; overlaps Wave 2)
```
$PY -m lagu.launch --max-procs 12 --front --out results/gate --arms lagu_nogate --seeds {100..114} --steps $STEPS --variant "tt-match:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=$LR,delta0=$D,$K"
$PY -m lagu.launch --max-procs 12 --front --out results/gate --arms lagu --seeds {100..109} --steps $STEPS --variant "tt-ggrad:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=$LR,gate_grad=1,$K"
```

### Gate G3 (all E1 and E1b pairs complete)
```
$PY -m lagu.evaluate results/gate results/gate_literal results/gate_x --workers 12      # ~1 h total
$PY -m lagu.analyze results/gate --h2 lagu-tt lagu_nogate-tt --pair lagu-tt lagu_nogate-tt
$PY -m lagu.analyze results/gate_literal --pair lagu-lit lagu_nogate-lit --h5
```
If H3 is Inconclusive, run one extension (10 runs; 8 h idle, 14 h at 0.6×):
```
$PY -m lagu.launch --max-procs 12 --front --out results/gate --arms lagu lagu_nogate --seeds {115..119} --steps $STEPS --variant "$V"
```
H4 gets the same treatment, using lagu plus tt-match on seeds 115–119. Afterwards, `$PY -m lagu.analyze --prereg prereg.md results/gate results/gate_literal results/gate_x` produces `prereg_results.json` with the Holm correction.

### Gate G4 / Wave 4 — E4 frontiers (optional; 32 runs, 27 h idle, 45 h at 0.6×)
Run it only if the queue is empty by 14 Oct (it is mandatory in the vacuous-gate case).
```
$PY -m lagu.launch --max-procs 12 --out results/gate --arms lagu_nogate --seeds {100..107} --steps $STEPS --variant "tt-d175:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=$LR,delta0=1.75,$K" --variant "tt-d125:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=$LR,delta0=1.25,$K"
$PY -m lagu.launch --max-procs 12 --out results/gate --arms lagu --seeds {100..107} --steps $STEPS --variant "tt-T035:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=$LR,T=0.035,$K" --variant "tt-T14:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=$LR,T=0.14,$K"
```

### Optional OmniSafe anchor (needs your approval)
You would approve installing a separate venv from the local uv cache, and run it from your own terminal with `echo 9 > results/.queue.cap` while it runs. It is 3 seeds of native OmniSafe TD3Lag at 1M, compared at 500k–1M against td3lag-omni (pareto seeds 0–2).

### Budget and calendar

| Scope | Process-hours | Wall, idle | Wall, 0.6× |
|---|---|---|---|
| All except E4 | ~1,320 | 110 h | 184 h |
| Including E4 | | 137 h | 229 h |

- Wave 1: 1 Oct.
- Wave 2 + Wave 3 drain: about 6–7 Oct idle, 9–10 Oct at 0.6×.
- E4: by about 12–14 Oct.
- Launch freeze 16 Oct; results freeze 23 Oct.
- Held-out evaluation, gradient probe and analysis: 16–26 Oct.
- Writing from 5 Oct (sections 2–4 during compute). Full draft 13 Nov; adversarial review 14–20 Nov; TMLR submission 23–27 Nov; arXiv by 1 Dec.

**Cut list** (applied only for throughput, in this order): E4 → E9 → E3 CarGoal1 → E3 td3lag arms → E3 PointCircle1 from 10 to 6 seeds → E3 at 1M. Never cut E1, E1b or the E2 matched seed counts.

---

## (3) Code changes required before Wave 1

**CC0 — baseline commit.** Commit `lagu/` as it is now (it is untracked). Record SHA-256 hashes of the actor and critic parameters after 300 updates for each arm; this is the golden reference used in CC3(f).

**CC1 — logging** (`agent.py`, `train.py`).
- *Interval accumulators* (reset in `reset_counters`). They are Python floats computed after the optimizer step and never feed the loss or use RNG.
  - Over all policy updates: gap, mean Q_c(s, π(s)) on replay states, mean Q̄, mean √Q_eu, mean δ_used, and λ after the update.
  - Over risk-sensitive updates: mean δ_used, plus `would_hits` / `would_total` for ratio ≥ T. These are counted for every arm with M > 1, gate on or off.
- *Shadow λ* (`self.lmbda_shadow`, logging only):
  - Under the statewise dual: `clip(λs + lambda_lr·(mean(qc) − δ0))`.
  - Under other duals it equals λ.
  - Stored in `state_dict` as `lambda_shadow`; `load` falls back to λ if the key is missing.
- *CSV columns*, appended to the END of `FIELDS`: `gap_avg, qc_avg, qbar_avg, qstd_avg, delta_avg, delta_risk_avg, lambda_avg, lambda_shadow, would_engage, n_pi, cum_cost, cum_viol_eps, resumed`.
  - `cum_cost` is the running sum of per-step cost from step 1. It is checkpointed and needed because 50·train_cost is not exact for H = 500.
  - `cum_viol_eps` counts training episodes whose cost exceeds the limit.
  - `resumed` is 1 on rows written after a resume.
- *Resume guard:* if an existing CSV header differs from `FIELDS`, exit instead of resuming.

**CC2 — snapshots and held-out evaluation.**
- In `train.py`, at every eval step, save `<stem>.snap/<step>.pt` with the actor, members, cost critic, λ and step. This is about 2.4 MB each, or roughly 20 GB for the whole plan; 543 GB are free.
- `evaluate()` gets a `seed_base=EVAL_SEED` argument whose default leaves in-training evals unchanged.
- New `lagu/evaluate.py`: `python -m lagu.evaluate <dirs> [--end-step N] [--snapshots 5] [--episodes 20] [--workers 12]`.
  - Snapshots are at N·{0.8, 0.85, 0.9, 0.95, 1.0}, with N defaulting to the run's step count.
  - Snapshot i uses layouts EVAL_SEED + 1000 + 20i + k.
  - Output is `<stem>.heldout.json` with return, cost, violation and the four calibration quantities, per snapshot and pooled.

**CC3 — tests** (hidden size 32, `env_step` = 150k so both update branches occur).
- (a) Bit-identity: lagu vs lagu_nogate over 300 updates on the same batch stream with cost > 0, under the fixed dual (λ 0.3) and the episodic dual (same J_c sequence). Every tensor of every network, target and optimizer state must be `torch.equal`. Liveness asserts: `gate_hits > 0` and λ > 0.
- (b) Positive controls: statewise with λ_init 0.1, and gate_grad = 1, must both differ.
- (c) Analytic gate gradient: grad(gate_grad = 1) − grad(gate_grad = 0) equals the autograd of −λ·mean(δ_ada(std / |Q̄|.detach())), within atol 1e-6, and is non-zero.
- (d) Uniform-budget equivalence: one Eq. 18 update of lagu gives the same Δλ (within 1e-6) as lagu_nogate with δ0 set to that update's mean δ_used.
- (e) Shadow λ: it equals λ for nogate arms and is ≤ λ for lagu.
- (f) Golden regression: parameters match the CC0 hashes, so the logging is side-effect free.
- (g) `test_train`: the new columns are finite; `cum_cost` equals a manual sum; the schema guard refuses; replay tensor round-trip works (old numpy checkpoints still load); `drop_ckpt` deletes only after `.pt` is written; snapshots exist; the default `evaluate` output is unchanged.

**CC4 — `lagu/certify.py`.**
- Declared pairs:
  - Equal: fix s0 and s1; epi s0 and s1.
  - Different: sw s0, and lagu-epigg vs lagu_nogate-epi s0.
- For each final `.pt`, compute SHA-256 per tensor over actor, actor_targ, members, members_targ, cost, cost_targ, the three optimizer states and `lam_param`. Counters and cfg are excluded.
- For equal pairs, also require identical `eval_*`, `lambda`, `qc_traj` and `mc_cost_traj` columns.
- Liveness: Σn_risk > 0, lagu engagement > 0, and λ > 0 in some row.
- Output `certificate.json` plus a markdown table; exit nonzero if any expectation fails.

**CC5 — `analyze.py`** (numpy only; no scipy).
- `--e0-rule`: conditions (a)–(d), selection, step budget, outcome class; writes `e0_decision.json`.
- `--budget-D ARM`: D = Σ(delta_avg · n_pi) / Σ n_pi over post-warm-up rows, pooled over seeds.
- `--pair A B [--end-step N]`: paired primary and secondary endpoints; 10k paired bootstrap giving 95% and 90% CIs; exact Wilcoxon by sign-flip enumeration; TOST; Fisher exact on collapses; the verdict label; a sensitivity analysis dropping pairs with `resumed` rows.
- `--h2`, `--h5` modes.
- Unpaired lagu vs td3lag: IQM, stratified bootstrap and probability of improvement.
- `--prereg`: Holm correction and `prereg_results.json`.
- Figures: learning curves; λ vs shadow λ; engagement and would_engage; calibration decomposition (qc_avg on replay vs qc_traj on-policy vs mc_cost_traj); δ_eff(t).

**CC6 — queue and memory/disk.**
- `launch.py --max-procs N [--front]`:
  - Appends run specs to `results/.queue.jsonl` under an fcntl lock.
  - Ensures a single detached scheduler (`--scheduler`, `start_new_session`, `caffeinate -i`, pidfile `results/.queue.pid`).
  - Every 60 s the scheduler counts live `python -m lagu.train` processes (the `status.sh` regex) and re-reads the cap from `results/.queue.cap`.
  - It launches exactly as `launch.py` does today, skipping runs that are already live or complete unless `--resume` is given, and logs to `results/.queue.log`.
- `train.py --drop-ckpt 1`: delete `.ckpt.pt` after the final `.pt` is written.
- `Replay.state()` returns `torch.from_numpy` views, avoiding the pickle copy; `load()` calls `np.asarray` first.

**CC7 — prereg.md and README.**
- prereg.md holds: H1–H5 and the secondary predictions; endpoints, verdict rule and margins; seeds; the E0 rule, step budget, E0-ext rule, D definition and extension rule; no exclusions (crashed runs are resumed and flagged; NaN counts as collapse); no interim look at contrasts; the cut list; OmniSafe's published 3M numbers per task; the analysis commit hash.
- README gets a fidelity table covering the tt, omsw and literal recipes.

**Later, not before Wave 1:**
- `lagu/probe.py`: ‖∇aQc‖, ‖∇aQ̄‖, their cosine and λ* over 5,000 on-policy states, from snapshots and from healthy lr_sweep/E0 finals.
- An optional `ARMS['lagu_ens']`.

---

## (4) Paper title and outline

**Title:** "What Does Lag-U's Uncertainty Gate Do? A Pre-registered Multi-Seed Re-examination of Uncertainty-Adaptive Lagrangian Safe RL on Safety-Gymnasium"

1. **Introduction.** Lag-U's claims; there is no released code and no replication. Contributions: an open port; Proposition 1 with a certificate; the calibration precondition; a pre-registered paired head-to-head with a budget-matched control; three tasks.
2. **Background and related work.** CMDPs, TD3-Lag, Lag-U Eq. 9–18. The three dual families, including the finding that OmniSafe's dual is the deterministic ramp λ = 5e-7·(t − 202k) (**F4**). Related work: CAL, PID-Lagrangian, OmniSafe, SL-SAC, COX-Q, USC, Spoor et al.
3. **Where can the gate act?** Proposition 1 and Corollary (**F1**); unit tests and the W0 certificate table; under Eq. 18, a time-varying uniform budget; the gradient reading; T is not shift-invariant.
4. **Porting Lag-U.**
   - 4.1 Fidelity table and the δ0 mapping.
   - 4.2 The literal learning rate breaks App. A Step 1: **F2** (lr_sweep, and the pessimistic reversal in lr_sweep2); **F5** paper-lr half (stationary-λ exploitation, Q_c down to −440), with the path-dependence guardrail; **F3** (9× action-gradient ratio, frozen-λ degradation, (1+λ) is a no-op under Adam, gradient-parity λ*).
   - 4.3 Two-timescale step sizes from OmniSafe's per-task configs (**F4**; **F5** OmniSafe-lr half, calibration 0.72–0.81); E0's selection and its outcome.
5. **Head-to-head.**
   - 5.1 E1 (H2, H3; λ, shadow λ, engagement, calibration decomposition).
   - 5.2 E1b literal recipe (the **F6** pilot extended to 10 seeds; H5).
   - 5.3 Lag-U vs TD3-Lag, descriptive only, under both recipes.
6. **Mechanism.** E2 matched budget (H4), the differentiable gate, E4 frontiers, the |Q̄| loop.
7. **Generalisation.** E3 on CarGoal1 and PointCircle1 with engagement per task.
8. **Discussion.** Specify Eq. 17's gradient path; report calibration and engagement; realised vs critic-unit cost; transferring T.
9. **Limitations.** No MetaDrive; CPU only; 1–2M steps; 10–15 seeds; resumed runs are not bit-identical.

**Appendices:** hyperparameters, commands and compute; the literal-lr diagnosis in full (pilot, lr_sweep, lr_sweep2, pareto incl. E9; F2, F5); the F3 frozen-λ and parity probe; the **F7** MSc CARLA pilot (T = 3.0, 43× the paper's value, so the gate never fired); OmniSafe reference and anchor; the pre-registration and its deviations; per-seed tables.

---

## (5) Risks and fallback paper

**R1 — E0's window is empty (outcome iv).**
- *Evidence:* fixed λ 0.3 at OmniSafe learning rates collapsed 2 of 2 seeds; the adaptive ramp is still at cost ~43 at 850k.
- *Mitigation:* 2M steps and outcome (iv) framing. That result is itself CAL-consistent evidence that the critic-unit dual has no stable operating point.

**R2 — gate engagement is degenerate.**
- The gate may engage about 0% of the time, or about 100% of the time through the |Q̄| loop.
- *Mitigation:* `would_engage` is measured in E0; E1's manipulation check decides; E4's T-family covers it.

**R3 — throughput, memory and power.**
- *Throughput:* 12 processes on 5 P-cores plus 10 E-cores; swap already at 2.6 GB. Lower the cap to 10 if swap exceeds 3.5 GB, and keep other heavy jobs off the machine.
- *Power:* keep AC on and the lid open; a closed lid still sleeps the Mac. Any system sleep setting is your call.
- *Resumes:* runs resume from the last 250k checkpoint, are flagged, and get a sensitivity analysis.

**R4 — "not faithful because the learning rates changed."** Mitigated by the App. A / Borkar argument, OmniSafe's per-task values (not tuned by us), E1b reported in full, and the fidelity table.

**R5 — replay-state dual target differs from on-policy J_c.** It is measured (qc_avg vs qc_traj vs MC). Never claim constraint satisfaction unless held-out cost ≤ 25.

**R6 — power and bimodal outcomes.** Handled by the verdict rule, one extension, CIs reported as bounds, and collapses analysed separately.

**R7 — forking paths.** Selection seeds 0–2 vs confirmatory seeds 100+; tags `prereg-v1` and `v1.1` pushed publicly before E0 and E1 (you do the push); no interim contrasts.

**R8 — scope.** Safety-Gymnasium only, so the MetaDrive numbers cannot be refuted.

**R9 — gradient routing is our choice.** gate_grad routes through √Q_eu only, so the ggrad arm is labelled an extension, not the paper's method.

**R10 — logistics.** You need an arXiv endorser and an OpenReview profile in October.

**Fallback paper if the gate shows nothing** (H3 Equivalent or Shift, H4 Equivalent, or E1 vacuous):
- **Title:** "An Uncertainty Gate That Only Moves the Multiplier: A Pre-registered Negative Replication of Lag-U", used only if the data support it.
- **Structure:** Proposition 1 and the certificate (F1); the calibration precondition (F2, F5, H5); E1 and E3 nulls as bounded effect sizes; E2 matched-budget equivalence; recommendations.
- **Venue:** TMLR with a Reproducibility certification, plus a 4-page non-archival ICLR 2027 workshop version.

**If E0 fails (outcome iv):** lead with "critic-unit duals lack an operating point on PointGoal1 even with a calibrated critic", and read E1 through λ only.

**Minimum viable paper if compute collapses:** C1 with the certificate, E0, E1b and the diagnostics.

**Upgrade path:** if H3 or H4 comes out Helps and E3 agrees, extend to RLC 2027.

**Main paths:**
- lagu/agent.py, train.py, launch.py, analyze.py and tests/ in /Users/michaelseiranian/Desktop/safe-rl-carla/lagu/
- /Users/michaelseiranian/Desktop/safe-rl-carla/results/{pilot,lr_sweep,lr_sweep2,pareto}
- /private/tmp/claude-501/-Users-michaelseiranian-Desktop-safe-rl-carla/47ca4e66-e2f4-4455-bfea-c56f9a36fb6d/scratchpad/zhang2024.txt