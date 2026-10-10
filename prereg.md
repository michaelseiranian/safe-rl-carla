# Pre-registration: what does Lag-U's uncertainty gate do on Safety-Gymnasium?

**Status.** Written 29 Sep 2026. This file, the E0 rule in section 10.2 and the analysis code are
committed and tagged `prereg-v1` before Wave 1 (the certificate W0 and E0) starts. The E0 decision
is then entered in section 15 and committed as `prereg-v1.1` before any confirmatory run
(seeds 100 and up) starts. The author pushes both tags to a public remote. After `prereg-v1`,
sections 1-14 are not edited; every change goes into section 16 (Deviations) with its date and
reason. (The tag was in fact made later; see section 16, item 1.)

**Analysis code commit**: c515eb23709f9d60c41c9a00a20874374ca377e5 (full SHA; the `prereg-v1`
commit itself, entered 2026-10-02, see section 16). Analysis code means `lagu/analyze.py`,
`lagu/evaluate.py` and `lagu/certify.py`. They are committed first and that commit's SHA is entered
here; this file is then committed and tagged `prereg-v1` with the analysis code unchanged. (A
commit cannot contain its own SHA, so the tagged commit is not the one named here.)

**Paper under study.** Zhang, Liu, Li, Lin, Li, "Lag-U", *IEEE T-ITS* 25(10), 2024,
DOI 10.1109/TITS.2024.3397700. Equation numbers below are the paper's.

**Code.** The port in `lagu/`. Recipes are defined in `lagu/README.md` (section "Recipes for the
pre-registered study"). The launch commands for every wave are in `lagu/PLAN.md`, section (2).

---

## 1. Research question

Suppose Lag-U's critic-unit dual (Eq. 18) can bind on Safety-Gymnasium with a calibrated cost
critic. Does the uncertainty-gated budget δ_ada (Eq. 16-17) then improve the return-cost trade-off
compared with

- (a) the same agent without the gate (`lagu_nogate`), and
- (b) the same agent given a uniform budget equal to the gate's average (`lagu_nogate` with
  δ0 = D, section 10.4)?

And what does the gate do under the literal published recipe?

## 2. Setup

| Task | Horizon | Obs dim | δ0 | Role |
|---|---|---|---|---|
| SafetyPointGoal1-v0 | 1000 | 60 | 2.5 | primary (E0, E1, E1b, E2, E4) |
| SafetyCarGoal1-v0 | 1000 | 72 | 2.5 | generalisation (E3) |
| SafetyPointCircle1-v0 | 500 | 28 | 5.0 | generalisation (E3) |

δ0 = cost_limit / horizon / (1 - γ) with cost_limit 25 and γ = 0.99; `lagu.train` sets it from the
task's horizon.

**Arms.** `lagu` (ensemble M = 3, exploration bonus, gate), `lagu_nogate` (same, gate off),
`td3lag` (M = 1, action noise 0.1, no bonus, no gate). A run's stem is
`<arm>[-<tag>]_<env>_s<seed>`; the tag names the recipe or variant (e.g. `lagu-tt`,
`lagu_nogate-tt-match`, `lagu-lit`).

**Recipes.** `lit` = the literal published recipe. `tt` = the literal recipe with two-timescale
step sizes (actor 5e-6, critics 1e-3, λ-lr chosen by E0). `omsw` = OmniSafe's TD3Lag recipe with
the paper's state-wise dual at λ-lr 4e-7. "The chosen recipe" below is the E0 cell chosen by the
rule in section 10.2: `tt` at that cell's λ-lr, or `omsw` (outcome (ii), or outcome (iii)/(iv) when
no `tt` cell is healthy). If `omsw` is chosen, every `tt` variant string in E1-E4 is replaced by the
`omsw` string with the same added keys (`delta0`, `gate_grad`, `T`), and `tt` in every tag becomes
`omsw` (e.g. `lagu_nogate-omsw-match`); names below that contain `tt` then mean these runs.

## 3. Experiments

| ID | Purpose | Arms (tag) | Seeds | Steps | Directory |
|---|---|---|---|---|---|
| W0 | H1 certificate | `lagu`, `lagu_nogate` under fixed (`fix`), episodic (`epi`) and state-wise (`sw`) duals; `lagu-epigg` (gate_grad = 1) | 0-1 (`sw`, `epigg`: 0) | 150k | `results/certify` |
| E0 | pick the dual step size | `lagu_nogate`: `tt-l1e-5`, `tt-l2e-6`, `tt-l4e-7`, `omsw-l4e-7` | 0-2 | 1M | `results/e0_dual` |
| E0-ext | early warning for 2M runs | the chosen E0 cell, resumed | 0-2 | 2M | `results/e0_dual` |
| E1 | H2, H3 (primary) | `lagu`, `lagu_nogate`, `td3lag`, all with the chosen recipe | 100-114; `td3lag` 100-109 | 1M or 2M (10.2) | `results/gate` |
| E1b | H2, H5 | `lagu`, `lagu_nogate`, `td3lag` (`lit`) | 100-109 | 1M | `results/gate_literal` |
| E2 | H4 | `lagu_nogate-tt-match` (δ0 = D); `lagu-tt-ggrad` (gate_grad = 1, an extension, not the paper's method) | 100-114; ggrad 100-109 | as E1 | `results/gate` |
| E3 | secondary: other tasks | CarGoal1 and PointCircle1: `lagu`, `lagu_nogate`; `td3lag` | 100-109; `td3lag` 100-104 | as E1 | `results/gate_x` |
| E4 | secondary: dose-response | `lagu_nogate` δ0 ∈ {1.75, 1.25}; `lagu` T ∈ {0.035, 0.14} | 100-107 | as E1 | `results/gate` |
| E9 | descriptive: fixed-λ frontier, seed-2 top-up | `td3lag` fixed and OmniSafe variants | 2 | 1M | `results/pareto` |

E2's matched arm runs only under the third G2 branch in section 10.4; `lagu-tt-ggrad` is enqueued
under every branch. E4 runs only if the queue is empty by 14 Oct 2026, with two exceptions set at
G2 (section 10.4): when G2 skips the matched arm it enqueues E4's T-family, and when E1 is vacuous
E4 is mandatory and its T-family uses T ∈ {0.035, 0.0175}.
E3 uses the recipe chosen on PointGoal1 without re-tuning.

## 4. Seeds

- **Selection seeds 0-2.** Used only for W0 (seeds 0-1), E0 and E0-ext (and seed 2 for the E9
  top-up). They never enter a confirmatory contrast.
- **Confirmatory seeds 100-114.** Used for E1-E4 (subsets as listed in section 3). They are never
  used to choose a recipe, step size or step budget.
- **Extension seeds 115-119.** Used only by the single extension in section 10.5, for H3 and H4.

Contrasts pair runs by seed: the two runs of a pair share the seed and the evaluation layouts.

## 5. Endpoints

**Primary endpoint** (every contrast). Per run: mean return and mean cost over 100 held-out
deterministic episodes, produced by `lagu/evaluate.py`. The episodes come from 5 snapshots at
N × {0.8, 0.85, 0.9, 0.95, 1.0}, where N is the run's step count, 20 layouts each. Snapshot
i ∈ {0, ..., 4} plays layouts EVAL_SEED + 1000 + 20i + k, k = 0, ..., 19, with EVAL_SEED = 10000.
These layouts are disjoint from the in-training evaluation layouts (EVAL_SEED + k) and shared by
every run. The rule in section 10.3 may set N = 1M for PointGoal1 runs of 2M steps.

**Secondary endpoints.**
- Mean of the in-training evaluation rows (`eval_return`, `eval_cost`) with step ≥ 0.8N.
- Cumulative training cost (`cum_cost`) at the end of the run and at 500k.
- Collapse: held-out return < 5 (section 11 adds diverged runs).
- λ: `lambda_avg`, `lambda_shadow`; gate engagement (`engagement`, `would_engage`); critic
  calibration (`qc_avg` on replay states, `qc_traj` on-policy, `mc_cost_traj` realised).

**Summary statistics used below.**
- *Interval columns* (`*_avg`, `engagement`, `would_engage`, `n_pi`, `n_risk`) summarise the
  updates since the previous row: the row at step t covers (t - 50k, t]. A window "over the last
  20%" of an interval column therefore uses the rows with step > 0.8N. Evaluation columns
  (`eval_*`, `qc_traj`, `mc_cost_traj`) describe the policy at step t and use rows with step ≥ 0.8N.
- Every n_pi-weighted mean is taken over rows with n_pi > 0 (interval means are undefined
  otherwise).
- *Engagement over a window*: Σ(engagement × n_risk) / Σ n_risk over the rows in the window,
  pooled over seeds. This is the fraction of risk-sensitive samples on which the gate fired.
- *Time-averaged λ of a run*: Σ(lambda_avg × n_pi) / Σ n_pi over all rows of the run.
- *Pooled calibration*: Σ qc_traj / Σ mc_cost_traj, pooled as stated at each use.

## 6. Statistical analysis

Numpy only. Every bootstrap uses 10,000 resamples, percentile intervals and a fixed RNG seed.

- **Contrasts** are A - B on the primary endpoint, paired by seed, for return and for cost.
  A is the arm with the gate (or the manipulation), B the control.
- **Paired bootstrap**: resample seed pairs with replacement; the statistic is the mean paired
  difference. Report 95% and 90% intervals.
- **Exact Wilcoxon signed-rank** by enumerating all 2^n sign flips of the ranked non-zero paired
  differences (zero differences dropped, tied absolute values given average ranks). The p-value is
  the fraction of sign assignments whose signed-rank sum is ≥ the observed sum (one-sided) or ≥ it
  in absolute value (two-sided). If no difference is non-zero, p = 1.
- **TOST** is the 90% interval check inside the margins of section 7.
- **Fisher exact test** (two-sided) on the number of collapsed runs in A vs B.
- **Holm correction** over the following p-values. A test whose data are not collected (for
  example H4 when E2 does not run) is dropped from the family. H1 is deterministic and has no
  p-value.

| # | Hypothesis | Test |
|---|---|---|
| 1 | H2, E1 | one-sided exact Wilcoxon, time-averaged λ of `lagu` > `lagu_nogate`, seeds 100-114 |
| 2 | H2, E1b | same, `lagu-lit` vs `lagu_nogate-lit`, seeds 100-109 |
| 3 | H3 | two-sided exact Wilcoxon, Δreturn, E1 pairs |
| 4 | H3 | two-sided exact Wilcoxon, Δcost, E1 pairs |
| 5 | H4 | two-sided exact Wilcoxon, Δreturn, `lagu-tt` vs `lagu_nogate-tt-match` |
| 6 | H4 | two-sided exact Wilcoxon, Δcost, same pairs |
| 7 | H5 | one-sided exact Wilcoxon signed-rank, held-out cost - 25 > 0, `lagu-lit` |
| 8 | H5 | same, `lagu_nogate-lit` |

  The verdict labels in section 7 are based on bootstrap intervals and are not changed by Holm.
  Where a hypothesis's criterion uses a p-value (H2), the Holm-adjusted value is used.
- **Sensitivity analysis**: every contrast is repeated with the pairs that contain a resumed run
  (any row with `resumed` = 1) dropped.
- **Descriptive only** (no hypothesis): Lag-U vs TD3-Lag, unpaired, by IQM over seeds with
  stratified bootstrap intervals and the probability of improvement.
- **Power.** Assuming no gain from pairing: with a between-seed SD of about 6 (Lag-U arms in the
  pilot), the 95% half-width of a 15-pair difference is about 4.7 cost units; with an SD of 13 it
  is about 10. Equivalence can then only be shown for true differences near 0.

## 7. Verdict rule

Applied to a contrast A - B on the primary endpoint. The first label that matches applies. Bounds
are those of the paired bootstrap intervals.

| Label | Condition |
|---|---|
| Helps | (Δcost 95% upper bound < 0 and Δreturn 95% lower bound > -2.5), or (Δreturn 95% lower bound > 0 and Δcost 95% upper bound < +5) |
| Hurts | (Δcost 95% lower bound > 0 and Δreturn 95% upper bound < +2.5), or (Δreturn 95% upper bound < 0 and Δcost 95% lower bound > -5), or A has at least 3 more collapsed runs than B |
| Shift | the 95% intervals of Δreturn and Δcost both exclude 0 on the same side (both above 0, or both below 0): movement along the trade-off |
| Equivalent | the 90% interval of Δcost lies inside (-5, +5) and the 90% interval of Δreturn lies inside (-2.5, +2.5) |
| Inconclusive | none of the above |

Margins: 5 cost units and 2.5 return units, in PointGoal1 units. They are not pooled or rescaled
across tasks. An Inconclusive H3 or H4 triggers the single extension in section 10.5; after it,
the verdict is final and the intervals are reported as bounds on the effect.

## 8. Hypotheses

### H1 - channel (deterministic)

- *Claim.* Under the fixed and the realised-cost episodic duals, `lagu` and `lagu_nogate` from the
  same seed have bit-identical network and optimizer tensors and identical `eval_*` columns. The
  positive controls (state-wise dual with λ_init 0.1; gate_grad = 1) differ.
- *Data.* W0, via `lagu/certify.py`. Inert pairs: `fix` seeds 0 and 1, `epi` seeds 0 and 1.
  Controls: `sw` seed 0; `lagu-epigg` vs `lagu_nogate-epi` seed 0. For each final `.pt` a SHA-256
  is taken of every tensor of actor, actor_targ, members, members_targ, cost, cost_targ, the three
  optimizer states (and the episodic dual's Adam state), lam_param and the final λ (counters, RNG
  state and cfg excluded). Inert pairs must also match on the `eval_*`, `lambda`, `qc_traj` and
  `mc_cost_traj` columns. Liveness, for every run of every declared pair: Σ n_risk > 0 and λ > 0 in
  some row, and, if the run is gated, engagement > 0 in some row. The complete list of checks
  (including that the two runs of a pair differ in configuration only in the gate switches) is the
  docstring of `lagu/certify.py` at the analysis code commit.
- *Confirm.* All 4 inert pairs are hash-equal, both controls differ, and the liveness checks pass.
- *Refute.* Any inert pair differs. That indicates a hidden channel or a bug; stop before E1.
- A control that comes out equal is not a refutation; see section 10.1.

### H2 - the gate acts through λ

- *Claim.* In E1 and in E1b, `lagu`'s engagement over 25k-300k (rows with 25,000 < step ≤ 300,000)
  is above 10%, and its time-averaged λ exceeds `lagu_nogate`'s (one-sided paired exact Wilcoxon,
  Holm-adjusted α = 0.05).
- *Assessment.* Separately for E1 (seeds 100-114) and E1b (seeds 100-109).
- *Confirm.* Both conditions hold.
- *Refute.* Engagement below 10% (E1 is then vacuous for H3; E4's T-family replaces the matched
  arm), or the median paired difference in time-averaged λ (`lagu` - `lagu_nogate`) is ≤ 0.
- Otherwise (engagement ≥ 10%, median difference > 0, adjusted p ≥ 0.05): not confirmed.
- The E1 assessment decides the G2 branch (section 10.4) and whether H3 is vacuous.

### H3 - primary: the gate's effect on outcomes

- *Data.* E1, chosen recipe, `lagu` vs `lagu_nogate`, seeds 100-114 (plus 115-119 after an
  extension).
- *Prediction* (from Proposition 1: a detached gate acts only through λ): no Pareto improvement.
- *Confirm.* Final verdict Equivalent, Shift or Hurts.
- *Refute* (supports Lag-U's claim). Final verdict Helps.
- A final Inconclusive neither confirms nor refutes; the intervals are reported as effect bounds.
- If E1 is vacuous (H2's E1 engagement below 10%), the verdict is still computed and reported, and
  is labelled vacuous.

### H4 - adaptivity vs uniform tightening

- *Data.* E2: `lagu-tt` (E1) vs `lagu_nogate-tt-match` (δ0 = D), seeds 100-114 (plus 115-119
  after an extension). H4 is not tested when G2 skips the matched arm.
- *Claim.* The two arms are equivalent.
- *Confirm.* Final verdict Equivalent, or Inconclusive with both 95% intervals (Δreturn and Δcost)
  containing 0.
- *Refute.* Helps (the timing of tightening matters), Hurts (early over-tightening costs return),
  or Shift (the arms differ along the trade-off).
- A final Inconclusive with an interval that excludes 0: not confirmed; reported as bounds.

### H5 - calibration precondition

- *Claim, literal recipe (E1b, seeds 100-109), for `lagu-lit` and for `lagu_nogate-lit` each:*
  - the 95% bootstrap lower bound of the held-out cost IQM over seeds is > 25;
  - |gap| ≤ 0.25 × δ0 (= 0.625) over the last 20%, where gap is the mean over seeds of each run's
    n_pi-weighted mean of `gap_avg` over rows with step > 0.8N (the constraint is met in critic
    units);
  - pooled held-out calibration (Σ qc_traj / Σ mc_cost_traj over seeds and snapshots) is < 0.65,
    and its 95% upper bound (bootstrap over seeds) is < 1.
- *Claim, E1 `lagu_nogate` (chosen recipe):* pooled held-out calibration (seeds 100-114, all
  snapshots) lies in [0.65, 1.35].
- *Confirm.* Every part of the claim holds.
- *Refute.* Literal-recipe pooled calibration ≥ 0.65, or literal-recipe held-out cost IQM ≤ 25,
  in either arm.
- Otherwise: partially supported; the parts that fail are listed.

## 9. Secondary predictions (outside the Holm family)

- **E3.** On CarGoal1 and on PointCircle1, the mean paired differences Δreturn and Δcost
  (`lagu` - `lagu_nogate`) have the same signs as in E1. Intervals are reported; the margins of
  section 7 are not applied to these tasks.
- **|Q̄| feedback loop.** If the median of |`qbar_avg`| over `lagu-tt` rows after 300k (all
  seeds) falls below 0.3, then engagement over those rows exceeds 80% and the ratio of the cell
  means of final λ, `lagu` / `lagu_nogate`, exceeds 3.
- **Gradient parity.** The frozen-λ return-halving point lies within a factor of 2 of
  λ* = ‖∇a Q̄‖ / ‖∇a Qc‖, measured by `lagu/probe.py` (written after Wave 1).
- **E4.** Held-out cost and return are monotone in each knob (δ0 for `lagu_nogate`, T for `lagu`),
  judged on arm means.

## 10. Decision rules fixed in advance

### 10.1 Certificate outcome (gate G1)

- All expectations met: proceed.
- An inert pair differs: STOP. Reproduce it in a unit test, fix it, rerun W0.
- A positive control comes out equal: rerun W0 at 300k before proceeding.

### 10.2 E0 rule

Mechanical. Each E0 cell (one variant, seeds 0-2) is judged on its evaluation rows at 850k-1M;
each seed contributes its rows at steps 850k, 900k, 950k and 1M, 12 rows per cell. Rows are
selected by step, not as the last 4 rows, because E0-ext later appends rows up to 2M. A diverged
run (section 11) counts as a seed below 10 in (b), and a non-finite value that a condition reads
makes the cell fail that condition.

- **(a) Calibration.** Σ qc_traj / Σ mc_cost_traj over the 12 rows is in [0.65, 1.35]; each
  seed's ratio over its own 4 rows is ≥ 0.5; no row has qc_traj < 0.
- **(b) No collapse.** Mean `eval_return` over the 12 rows is ≥ 20, and no seed's mean over its 4
  rows is below 10.
- **(c) Binding in critic units.** Mean `lambda_avg` over the 12 rows is ≥ 0.02, and either
  |mean `gap_avg` over the 12 rows| ≤ 0.3, or λ rose by more than 20% between 600k and 1M:
  λ̄(1M) > 1.2 × λ̄(600k), where λ̄(t) is the mean over seeds of `lambda_avg` in the row at step t.
- **(d) Binding in realised cost.** Mean `eval_cost` over the 12 rows is ≤ 35, or the mean
  `train_cost` over rows 850k-1M is at least 10 below its mean over rows 250k-400k.

A cell is *healthy* if it passes (a) and (b), and *binds* if it passes (c) and (d). Speed order is
by λ-lr, largest first: `tt-l1e-5`, `tt-l2e-6`, `tt-l4e-7`, `omsw-l4e-7` (the tie at 4e-7 goes to
`tt`). The first outcome that matches applies:

- **(v)** No cell passes (a), or (iii)/(iv) would apply but no cell is healthy: STOP and debug.
  Only E1b may run meanwhile.
- **(i)** Some `tt` cell passes (a)-(d): choose the one with the largest λ-lr.
- **(ii)** Otherwise, if `omsw` passes (a)-(d): choose `omsw`.
- **(iii)** Otherwise, if no cell binds: choose the fastest healthy cell, at 2M steps.
- **(iv)** Otherwise (some cells bind, but none of them is healthy: fast cells collapse and slow
  cells never bind): choose the fastest healthy cell, at 2M steps; the paper leads with the
  negative result.

**Step budget** under (i) or (ii): 1M if the chosen cell's λ̄ changes by less than 20% over
600k-1M (|λ̄(1M) - λ̄(600k)| < 0.2 × λ̄(600k)) and |mean `gap_avg` over its 12 window rows| ≤ 0.3;
otherwise 2M. E1b always runs 1M.

**Record** `would_engage` and `qbar_avg` of the E0 cells against the degenerate-engagement risk
and the |Q̄| prediction (section 9), and enter the decision in section 15 before tagging
`prereg-v1.1`.

### 10.3 E0-ext rule (only when the step budget is 2M)

The chosen E0 cell's three selection-seed runs are resumed from 1M to 2M with the same variant
string. If at least 2 of the 3 seeds have mean `eval_return` below 5 over rows 1.6M-2.0M, the
primary window of every PointGoal1 contrast moves to 0.8-1.0M (N = 1M, `--end-step 1000000`).
This is decided and recorded (section 15) at G2, or when E0-ext finishes if that is later, and in
every case before any E1 contrast is computed. It does not apply to E3 or E1b.

### 10.4 Budget D and the matched-arm decision (gate G2)

**D** = Σ(`delta_avg` × `n_pi`) / Σ `n_pi` over all rows of the 15 `lagu-tt` runs (seeds
100-114) that contain policy updates (n_pi > 0), pooled over seeds. It is the gate's average
budget over **all** policy updates, not only risk-sensitive ones. It is computed with
`python -m lagu.analyze results/gate --budget-D lagu-tt` once all 15 runs have finished, and the
printed value is used verbatim as `delta0` of `lagu_nogate-tt-match`.

At G2, with E1 engagement as defined in H2 (25k-300k, `lagu-tt`, seeds 100-114), the first branch
that matches applies:

- Engagement < 10%: E1 is vacuous for H3 but is still reported. Skip the matched arm. Enqueue
  `lagu-tt-ggrad` and E4 at T ∈ {0.035, 0.0175}.
- D > 2.4: skip the matched arm. Enqueue `lagu-tt-ggrad` and E4's T-family (T ∈ {0.035, 0.14}).
- Otherwise (engagement ≥ 10% and D ≤ 2.4): enqueue E2 at the front of the queue.

G2 reads only `lagu` rows and the selection-seed E0-ext rows; no contrast is computed.

### 10.5 Extension rule

If the verdict of H3 or of H4 on its 15 pairs (H3 at gate G3, H4 when E2 completes) is
Inconclusive, that contrast gets exactly one extension: both of its arms are run on seeds 115-119 (H3: `lagu`, `lagu_nogate`; H4: `lagu`,
`lagu_nogate-tt-match`; the `lagu` runs are shared). The verdict rule is applied once more to all
20 pairs, and that verdict is final. H2 and H5 use the seeds listed in section 8 only.

### 10.6 Cut list and calendar

Cuts are made only for throughput, never in response to results, in this order: E4 → E9 → E3
CarGoal1 → E3 `td3lag` arms → E3 PointCircle1 from 10 to 6 seeds → E3 at 1M. E1, E1b and the E2
matched-arm seed counts are never cut. Each cut is recorded in section 16 with its date.
Launch freeze 16 Oct 2026; results freeze 23 Oct 2026. How anything unfinished at the results
freeze is handled will be recorded in section 16.

## 11. Exclusions, crashes and missing data

- **No run is excluded.**
- A crashed run is resumed from its last checkpoint (every 250k steps). Rows written after a
  resume carry `resumed` = 1. Resumed runs are seeded but not bit-identical to an uninterrupted
  run; they stay in the primary analysis, and the sensitivity analysis in section 6 drops their
  pairs.
- A run *diverges* if a value in its `eval_return`, `eval_cost`, `qc_traj`, `qbar_traj`, `q_loss`,
  `c_loss` or `lambda` column (rows after its random warm-up), or its held-out return or cost, is
  not finite. Values that are undefined by construction do not count: `jc` outside the episodic
  dual, `would_engage` when M = 1, and interval means over an interval without a policy update.
- A diverged run counts as collapsed. If a per-run quantity that a test or interval uses
  (held-out return or cost, time-averaged λ, gap, calibration) is not finite, the run (with its
  pair, in a paired contrast) is left out of that computation, the number left out is reported,
  and the run still counts in the collapse comparison (Fisher test and the Hurts clause).

## 12. No interim look at contrasts

No A - B comparison, and no per-arm aggregate of outcome columns on confirmatory seeds, is computed
or plotted before gate G3, when all E1 and E1b pairs are complete and the held-out evaluation has
run. Before G3 only these are read: run health (liveness, step counts, crashes, divergence;
`lagu/status.sh` prints each run's last log line and is used only for this), the `lagu`-only quantities at G2
(D and engagement), and the selection-seed E0 and E0-ext runs. H4 is analysed when E2 completes,
with the same rules.

## 13. Interpretation rules

- Fixed-λ runs (`results/pareto`, E9) are evidence of stationary-λ exploitation only. They are not
  used as a bound on adaptive duals, because outcomes are path-dependent: fixed λ = 0.05 collapsed
  where adaptive λ ≈ 0.05 kept return 26.8, and fixed λ = 0.3 collapsed 2 of 2 seeds where the
  OmniSafe ramp at λ ≈ 0.32 kept 22-26.
- Constraint satisfaction is claimed for an arm only if its held-out mean cost is ≤ 25.
- `lagu-tt-ggrad` routes the actor gradient through √Q_eu only; it is labelled an extension, not
  the paper's method.
- Lag-U vs TD3-Lag comparisons are descriptive.

## 14. External reference: OmniSafe's published numbers

OmniSafe's own TD3Lag and TD3PID results (repository file `benchmarks/off-policy/README.md`,
Table 2; cost_limit 25; 3M training steps for these navigation tasks, per the table caption).
Values are return and cost, ± as printed in the table. They are context only: our runs are 1-2M
steps in a different code base, and no hypothesis is tested against them.

| Task | TD3Lag return | TD3Lag cost | TD3PID return | TD3PID cost |
|---|---|---|---|---|
| SafetyPointGoal1-v0 | 25.27 ± 2.74 | 28.00 ± 15.75 | 18.76 ± 7.87 | 12.17 ± 9.39 |
| SafetyCarGoal1-v0 | 7.31 ± 5.34 | 33.83 ± 31.03 | 27.28 ± 4.50 | 9.50 ± 12.15 |
| SafetyPointCircle1-v0 | 83.07 ± 3.49 | 7.83 ± 15.79 | 70.95 ± 6.00 | 0.00 ± 0.00 |

## 15. Decision log

Filled in at the gates; each entry is committed before the next wave that depends on it.

- W0 certificate (G1): **PASS**, 2026-09-29 19:38:57 UTC (`results/certify/certificate.md`).
  Under the fixed λ (2 seeds) and the episodic dual (2 seeds), `lagu` and `lagu_nogate` are
  bit-identical (0 of 241 or 245 tensors differ, no CSV column differs). Both positive controls
  differ: the state-wise dual (193 of 241 tensors; λ 0.759 vs 0.453) and the gate gradient (196 of
  245).
- E0 outcome class (i)-(v), chosen cell, λ-lr (LR), step budget (STEPS): **(iv)**,
  `lagu_nogate-tt-l2e-6`, LR = 2e-6, STEPS = 2000000. Decided 2026-09-30 00:23:10 UTC by
  `lagu.analyze --e0-rule` (`results/e0_dual/e0_decision.json`); its hash entered
  `results/PREREG_SHA256.txt` at 00:23:52 UTC, before Wave 2. No tag `prereg-v1.1` was made
  (section 16, item 1).
- E0 `would_engage` and `qbar_avg` (from `e0_decision.json`):

  | cell | `would_engage` (run / 25k-300k) | `qbar_avg` | median \|`qbar_avg`\| after 300k |
  |---|---|---|---|
  | `lagu_nogate-tt-l1e-5` | 0.994 / 0.923 | -4.00 | 1.485 |
  | `lagu_nogate-tt-l2e-6` | 0.972 / 0.948 | -1.43 | 0.834 |
  | `lagu_nogate-tt-l4e-7` | 0.994 / 0.884 | 0.28 | 0.435 |
  | `lagu_nogate-omsw-l4e-7` | 0.967 / 0.808 | -78.68 | 0.706 |

- E0-ext window decision (G2, 2M only): **keep the 2M window**. The three selection seeds of
  `lagu_nogate-tt-l2e-6` have mean `eval_return` 24.51, 24.91 and 24.48 over 1.6M-2.0M, none
  below 5. Computed 2026-10-02 21:33:36 UTC by re-running `lagu.analyze --e0-rule`, which filled
  only the E0-ext block of `results/e0_dual/e0_decision.json` (every other entry is unchanged).
- D and E1 engagement over 25k-300k; G2 branch taken: **D = 1.12158** (the printed value; 1.121585
  to six decimals) and **engagement = 0.8497**, from the 15 finished `lagu-tt` runs (seeds
  100-114) with `lagu.analyze results/gate --budget-D lagu-tt`, 2026-10-02 21:33:30 UTC.
  Engagement is at least 10% and D is at most 2.4, so E2 is enqueued at the front of the queue:
  `lagu_nogate-tt-match` (delta0 = 1.12158, seeds 100-114) and `lagu-tt-ggrad` (seeds 100-109),
  2M steps, with the `lagu/PLAN.md` Wave 3 commands, checkpointing every 100k (section 16,
  item 3).
- Extensions run (G3): computed 2026-10-09 15:48 UTC with the `lagu/PLAN.md` G3 commands, after
  `lagu.evaluate` had written the held-out endpoint of every E1, E1b, E2 and E3 run. H3 on the 15 E1
  pairs is **Hurts** (only the collapse clause fires: 9 vs 3 collapsed runs), so H3 gets no
  extension. H4 on the 15 E2 pairs is **Inconclusive**, so section 10.5 gives it one extension:
  `lagu-tt` and `lagu_nogate-tt-match` (delta0 = 1.12158) on seeds 115-119, 2M steps (10 runs; the
  `lagu` runs are shared with the H3 arm but H3's verdict stays on its 15 pairs). The final H4
  verdict and the Holm family are computed when these finish.
- Final confirmatory analysis: 2026-10-10 00:55 UTC, `lagu.analyze --prereg prereg.md results/gate
  results/gate_literal results/gate_x` at commit 6b28b15, after checking that every run in every
  contrast had a held-out file (`prereg_results.json`; printout in `results/g3/prereg_final.txt`).
  Holm-adjusted p: H2 E1 1.0, H2 E1b 1.0, H3 return 0.040, H3 cost 1.0, H4 return 1.0, H4 cost 1.0,
  H5 `lagu-lit` 0.008, H5 `lagu_nogate-lit` 0.008.
  - H2: not confirmed (E1 and E1b).
  - H3 (15 pairs): Hurts, so confirmed by the section 8 rule. Return 5.42 vs 15.59, difference
    -10.17, 95% [-16.08, -3.80]; cost 34.46 vs 37.77, difference -3.31, 95% [-18.75, 16.70];
    collapsed runs 9 vs 3 (Fisher 0.06). Only the collapse clause fires.
  - H4 (20 pairs): Inconclusive with both 95% intervals containing 0, which the section 8 rule
    counts as confirmed. Return 4.24 vs 5.79, difference -1.55, 95% [-5.24, 1.93]; cost 32.85 vs
    39.87, difference -7.02, 95% [-23.18, 10.00]; collapsed runs 14 vs 12 (Fisher 0.74). Under
    section 16 item 5 it is reported as these bounds, not as equivalence.
  - H5: partially supported. Both literal-recipe arms meet every literal part (held-out cost IQM
    49.3 and 50.6, calibration 0.46 and 0.44, gap within a quarter of δ0); the E1 `lagu_nogate`
    calibration band fails (pooled 112.96, 95% [0.31, 336.9]; 0.50 without the one run whose
    critics blew up, which stays in under section 11).
  - Descriptive (section 16 item 5): every one of the 10 extension runs collapsed (`lagu-tt` and
    `lagu_nogate-tt-match`, seeds 115-119), against 9/15 and 7/15 on seeds 100-114. Code, config
    and launch arguments match the earlier seeds, none was resumed, and training reads the clock
    only to log speed; no systematic cause was found.

## 16. Deviations

**2026-10-02**

1. *Tag timing.* `prereg-v1` was made and pushed on 2026-10-02, not before Wave 1, and no
   `prereg-v1.1` was made. Wave 1 (2026-09-29 18:52:52 UTC) and Wave 2 (2026-09-30 00:24:02 UTC)
   were launched against a local hash record, `results/PREREG_SHA256.txt` (written 2026-09-29
   18:52:43 UTC; E0 decision appended 2026-09-30 00:23:52 UTC), which is published in the
   `prereg-v1` commit; every file it lists matches there. The analysis code commit named above is
   that commit itself, entered here afterwards. The `prereg-v1` tag message lists what had been
   seen by then.
2. *Resumed E1 runs.* A reboot on 2026-10-01 killed the 12 runs then in progress (frozen since
   2026-09-30 11:05 UTC at the author's request): `lagu-tt` s109-s114 and `lagu_nogate-tt`
   s100-s105. They resume from their checkpoints with `resumed` = 1 (section 11), so 12 of the 15
   E1 pairs contain a resumed run and the section 6 sensitivity analysis keeps seeds 106-108 only.
3. *Checkpoint period.* From 2026-10-02, resumed and queued runs checkpoint every 100k steps, not
   every 250k (section 11). Checkpointing does not change the training computation.
4. *Power-aware queue.* From 2026-10-02 the queue freezes trainers on battery power and unfreezes
   them on mains (SIGSTOP/SIGCONT). This does not change the training computation.

**2026-10-09** (written before any outcome of the H4 extension, E4 or X1 was read)

5. *Reporting commitments.*
   - H4 is reported as bounds (both 95% intervals, plus the time course) whatever its final label.
     E2 compares two budget schedules, the gate's falling mean budget against a constant equal to
     its whole-run average, so it cannot show that adaptive and uniform tightening are equivalent.
     The "E2 matched-budget equivalence" section planned in `lagu/PLAN.md` is withdrawn.
   - H3 stays at its 15 pairs. No `lagu_nogate-tt` run is made on seeds 115-119; the collapse count
     of `lagu-tt` s115-s119 is reported descriptively.
   - A contrast is computed only after checking that every run in it has a held-out file (the
     analysis falls back to in-training rows for the whole contrast when one is missing). Section
     15's G3 entry overstates the coverage: `td3lag-lit` s105-s109 finished after that evaluation
     pass and are evaluated before any descriptive TD3-Lag comparison.
   - The section 9 |Q̄| prediction is evaluated on `lagu-tt` seeds 100-114 (the E1 seeds), with
     seeds 100-119 as a sensitivity check.
   - A run not at its final step at the results freeze (23 Oct) is reported as unfinished and its
     pair is left out, with the count stated.
   - Every mechanism analysis (time courses, per-seed forensics, dose plots, other horizons) is
     exploratory, is shown as full curves rather than chosen windows, and is reported whichever way
     it comes out. `lagu/probe.py` is committed before it reads any snapshot.
6. *E4 runs as pre-registered* (the queue is empty; section 10.6 and `lagu/PLAN.md` Gate G4):
   `lagu_nogate` with δ0 ∈ {1.75, 1.25} and `lagu` with T ∈ {0.035, 0.14}, the E1 recipe, seeds
   100-107, 2M steps. Stated in advance: T = 0.14 lies below the median gate ratio of the E1 critics
   (about 0.2-0.4), so the T family varies how strongly the gate tightens rather than how selectively;
   with 8 seeds, monotonicity of arm means is reported descriptively.
7. *Exploratory add-on X1: the gate at the paper's dose, in the literal recipe.* In E1 the gate
   fired on 85-95% of samples and cut the mean budget to about 0.45 δ0, far below the paper's Fig.
   5(d) as we read it (about 0.76-0.88 δ0); in E1b it was nearly off after 300k (about 0.97 δ0). No arm has
   tested a moderate gate. Design: `lagu` with T = 0.03, literal recipe, seeds 100-109, 1M steps,
   tag `lit-T03`, in `results/gate_literal_x`. T was chosen from critic quantities only: on 18,000
   on-policy states of `lagu_nogate-lit` (seeds 100-105, snapshots at 300k, 600k and 1M), T = 0.03
   gives engagement 0.41 and a mean budget of 0.85 δ0; no outcome column was used. When those runs
   finish, `lagu_nogate` with δ0 = D* (the printed `--budget-D lagu-lit-T03` of that directory),
   same seeds and steps, tag `lit-match`. Contrasts, paired by seed against the existing E1b
   `lagu_nogate-lit`: gate vs ungated, gate vs matched, and matched vs ungated, with the section 6
   statistics and section 7 labels. Prediction (Proposition 1, with the literal dual nearly inert
   in E1b): no practically relevant difference in any contrast. No X1 outcome is read before all
   X1 runs finish; X1 is reported whichever way it comes out and is not part of the Holm family.

**2026-10-10**

8. *X1 matched budget.* With all 10 `lagu-lit-T03` runs finished, `--budget-D lagu-lit-T03` on
   `results/gate_literal_x` printed D* = 2.14758 (0.859 δ0, the declared dose of about 0.85 δ0);
   engagement over 25k-300k was 0.81, higher than the 0.41 estimated from the 300k-1M snapshots,
   as early uncertainty is larger. `lagu_nogate` with δ0 = 2.14758 (tag `lit-match`, seeds
   100-109, 1M) is queued at the front. No X1 outcome has been read.
