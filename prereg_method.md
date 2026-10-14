# Pre-registration: a learned budget-viability shield against a tuned realised-cost Lagrangian

**Status.** This is the draft of 2026-10-11 for tag `method-v0`. It was revised the same day after a
hostile integrity review; Appendix H gives the response to every review item. On 2026-10-14, before
`method-v0`, it was amended after a mathematical audit of BVS and the builders' code reviews; these are
amendments, not deviations, and the closing section "Pre-v0 amendments (2026-10-14)" lists them.
This file is separate from `prereg.md` (the Lag-U study), which it does not modify. Sections 0-20
and the appendices are frozen at `method-v0`. After that, only section 21 (decision log) and section 22 (deviations, dated,
with reasons) change. The implementation follows `safe2/SPEC.md`, which is frozen with this file.
Where the two disagree, this file wins, and the disagreement is logged in section 22.

**Scope in one paragraph.**

- **Primary comparison.** One primary candidate, the Budget-Viability Shield (BVS), is compared on
  held-out seeds and layouts with B*, the best of thirteen tuned TD3-family baseline configs.
- **Primary family.** It holds one hypothesis: BVS beats B* on an arm-level penalised return score.
- **Other candidates.** SHL-PID is run as an exploratory candidate. MC-TD3 was dropped at the design
  stage.
- **Controls.** Three controls ask which part of BVS does the work:
  - BVS-U: the same takeover machinery, triggered by Lag-U's reward-uncertainty ratio. This is the
    study's original takeover question.
  - BVS-disc: a discounted backup critic with the same effective horizon.
  - A post-hoc shield on B* (PHS), used at deployment only.

**Code.** New code lives in `safe2/` and imports `lagu` where useful. These files are not modified
and stay as hashed in `results/PREREG_SHA256.txt`: `prereg.md`, `lagu/agent.py`, `lagu/train.py`,
`lagu/evaluate.py`, `lagu/analyze.py` and `lagu/certify.py`. This study enters nothing into the
Lag-U study's 23 Oct results freeze. Results go to `results/method/`.

---

## 0. Integrity and process

- **Branch.** Work happens on branch `method` in the worktree `safe-rl-carla-method`.
- **Tag `method-v0`** holds all of the following, with every test passing:
  - this protocol and `safe2/SPEC.md`;
  - **all** study code, including `safe2/bvs.py` and its tests (see the last bullet of this list);
  - the analysis code (`safe2/stats.py`, `safe2/analyze.py`);
  - the Stage-0 code (`safe2/probe.py`, `safe2/posthoc.py`);
  - the seed registry (`safe2/seeds.py`);
  - the configs, trainer, evaluator, sealing tool, health tool, launcher and pause script
    (section 10);
  - the baseline code (`safe2/base.py`) and the power script (`safe2/power.py`);
  - the public sealing key `results/method/keys/seal_pub.pem`.

  It is **pushed to `origin` before any Stage-0 read and before any run of this study starts**. This
  avoids repeating deviation 1 of `prereg.md`. The push time goes into section 21.

  Every Appendix A choice is pinned at v0, so BVS is coded before any Stage-0 or pilot outcome
  exists. `bvs.py`, `posthoc.py` and their tests exist and pass before v0, so the earlier
  `method-v0.1` fallback (BVS code tagged after v0) is withdrawn: there is no `method-v0.1` tag, and
  `analyze --unblind` accepts none.
- **What is hashed.** `analyze v0-hashes` hashes every file in `safe2/` outside `safe2/tests/` (the
  code above, `SPEC.md`, `README.md`, `HOOKS.md`, `pause.sh`, `health.sh` and any other file kept
  there), this protocol and the public key.
- **Hash record.**
  - `results/method/PREREG_SHA256.txt` lists the SHA-256 of every file above at `method-v0`.
  - `results/method/FUNC_SHA256.json` lists the SHA-256 of the source of each decision function in
    `safe2/stats.py` (`stats.DECISION_FUNCTIONS`).
  - The tests in `safe2/tests/` at v0 are the pre-registered tests (section 16).
- **Tag `method-v0.5`** records, before any baseline confirmatory run starts:
  - B*, every baseline selection score and the extra-seed scores;
  - L* and Sauté*;
  - R_ref(PointGoal1);
  - the decisions of gates G0a and G0b (G0a's threshold needs B*'s pilot return; section 5).
- **Tag `method-v1`** freezes the candidates before any candidate confirmatory run. It records:
  - the BVS* and SHL* configs and the kill outcomes;
  - R_ref for all three tasks, BVS-U's matched rate ρ* and the post-hoc shield margin m_PHS;
  - the size of the primary family;
  - `results/method/plan.json` (every confirmatory run and evaluation that must finish before
    unblinding), an empty `results/method/cuts.json`, and the file hashes.
- **Firewall.**
  - The confirmatory analysis is produced only by `python -m safe2.analyze --unblind`.
  - It verifies the v0 hashes of every study code file and the v0 per-function hashes of the
    decision functions (section 10).
  - If a decision function changed after v0, the change must be a section 22 deviation, and the
    **v0 version decides**. The changed version is reported as a sensitivity analysis.
  - Any other number computed from seeds 300 and up, or from the held-out block, is exploratory and
    labelled as such.
- Every tag is pushed to `origin` when it is made. Every change after `method-v0` goes into section
  22 with its date, its reason and what had been seen by then.

## 1. Question, current evidence and novelty

### 1.1 Question

Can a learned, fixed-horizon viability critic that gates a reward-only TD3 proposer (BVS) give a
better held-out return-cost trade-off than the best tuned baseline, at the same 1M-step budget, on
Safety-Gymnasium Goal and Circle tasks? The baselines are realised-cost Lagrangian, Sauté and
literal-base TD3 configs. Secondary: what matters most?
- the takeover trigger (viability vs reward uncertainty);
- the critic (fixed-horizon undiscounted vs discounted at the same horizon);
- deployment-time shielding alone.

### 1.2 What we already know (corrected wording; Appendix E gives the sources)

- **Reward-ensemble uncertainty and cost.** This is exploratory, from PointGoal1 final snapshots with
  decoupled reward and cost.
  - Lag-U's reward-ensemble ratio is not better than the cost critic Q_c at predicting cost within
    10 steps in any arm.
  - It is significantly worse in 4 of 5 arms. It is below 0.5 (slightly anti-predictive) in all 3
    literal-recipe arms.
  - In `lagu-tt` it is nominally higher (AUROC 0.641 vs 0.598, difference +0.043, 95% CI [-0.087,
    +0.167]), in an arm where 12 of 15 runs have return below 5.
- **Headroom is suggested, not shown.**
  - The design-stage oracle filter was run in 20 configurations (17 filtered, 3 unfiltered), with
    scripted, privileged proposers on 10-30 episodes. Filtered returns range from 6.3 to 17.5.
  - The best filtered configuration at cost ≤ 25 reaches 15.9 return / 20.5 cost. The often-quoted
    17.5 / 25.2 is over the limit.
  - The best Lagrangian held-out run in this codebase reaches 14.6 / 24.96.
  - None of the 157 PointGoal1 held-out files (plus one 20k smoke run) reaches return ≥ 15 at cost
    ≤ 25. But none of those runs used a realised-cost dual, so this says nothing about B*.
  - "OmniSafe recipe about 20 / 29 at 1M" cannot be reproduced. No held-out file exists for
    `td3lag-omni`. Its final in-training evaluation (10 episodes, 3 seeds) gives 20.7 / 33.4, and its
    800k-1M rows give 23.2 / 41.1.
  - Gate G0a (section 5) tests the headroom on fresh layouts with learned actors.
- **Avoidability rests on a small scripted sample.**
  - The sample is 46 hazard entries in 10 PointGoal1 episodes of a scripted goal-seeker (CarGoal1: 30
    entries in 8 episodes).
  - The warning needed to avoid an entry has median 7 steps, 90th percentile 8 and maximum 10. So 90%
    of the sampled entries were avoidable with 8 steps of warning, and all with 10.
  - Exits finish within 19-26 steps. H = 32 rests on this sample.
- **Fixed λ.** 15 fixed-λ runs collapsed:
  - 9 are literal-base runs with λ fixed from step 10k (return about -1, evaluation cost 10-133, so
    not a passive collapse);
  - 6 are OmniSafe-base runs with λ fixed from 202k.
  Ramped OmniSafe runs passed λ 0.3-0.4 without collapsing. The data point to λ applied abruptly,
  not to "λ in the actor gradient" as such.
- **The published-recipe baseline is rate-limited.** The OmniSafe-recipe TD3-Lag dual reaches
  λ = 0.399 = 5e-7 × 798k at 1M in all 3 seeds, with training cost 45.6-46.7. Beating it would show
  dual speed, not method; hence the tuned baseline pool (section 3.1).

### 1.3 Prior work and the novelty claim

BVS combines known parts. Closest prior work:

- **Sim-to-Lab-to-Real** (Hsu et al., *Artificial Intelligence*, arXiv 2201.08355): a performance
  policy, a co-trained learned backup policy and a safety-critic shield used during training. This is
  BVS's architecture.
- **Conservative Safety Critics** (Bharadhwaj et al., ICLR 2021, arXiv 2010.14497): rejects sampled
  actions against a learned safety-critic threshold during training.
- **Recovery RL** (Thananjeyan et al., RA-L 2021) and **SAILR** (Wagener et al., ICML 2021): learned
  recovery and advantage-based intervention, with the proposer trained on the shielded MDP.
- **CPSS** (arXiv 2605.18841, May 2026): turns the remaining cumulative budget into a per-step
  threshold, which is BVS's τ_t = d - m - C_t.
- **Latent Q-Barrier shielding** (arXiv 2605.25267, May 2026): filters candidate actions by the
  remaining budget against predicted future cost.
- **Kalweit et al. 2020** (arXiv 2003.09398): truncated multi-step constraint heads (fixed-horizon
  constraint critics) with an argmax over the safe set, for discrete actions.
- **CAPS** (Chemingui et al., AAAI 2025, arXiv 2412.18946): an accumulated-cost filter with a
  discounted cost critic. **ProSh** (arXiv 2510.15720): TD3 with a backup critic and the budget in the
  state.
- **Anytime-Constrained RL** (McMahan & Zhu, AISTATS 2024, arXiv 2311.05511); **RL with almost sure
  constraints** (Castellano et al., L4DC 2022, arXiv 2112.05198).
- **Model predictive shielding** (Bastani, arXiv 1905.10691; Li & Bastani, ICRA 2020,
  arXiv 1910.10885) and **SEditor** (NeurIPS 2022, arXiv 2201.12427).
- **The margin dual** is Adaptive Conformal Inference (Gibbs & Candès, arXiv 2106.00170) or
  Conformal Decision Theory (arXiv 2310.05921, venue unverified). Its telescoping bound is theirs.
- **Fixed-horizon TD** (De Asis, Chan, Pitis, Sutton & Graves, AAAI 2020) supplies the critic.
- **Sauté RL** (Sootla et al., ICML 2022, arXiv 2202.06558) and **PID Lagrangian** (Stooke, Achiam
  & Abbeel, ICML 2020) supply two baselines.
- For the dropped MC-TD3: DVAC (IROS 2023), LP3 (CoRL 2021), A-CRL (arXiv 2102.11941) and CCPO (arXiv
  2310.03718, venue unverified).

**Novelty claim (incremental).** Each part has precedent:
- fixed-horizon multi-step constraint critics (Kalweit et al. 2020);
- a co-trained backup policy behind a shield (Sim-to-Lab-to-Real);
- budget-to-threshold filtering (CPSS, Latent Q-Barrier).

BVS is new **only in combination**:
- continuous actions, with the undiscounted fixed-horizon minimum-cost critic evaluated as a sampled
  minimum over a candidate set;
- an ACI-style margin calibrated on realised episode cost inside off-policy training.

We call the result a well-controlled, pre-registered test of that combination, not a new paradigm.

**Limits stated in advance** (from the mathematical audit of 2026-10-14; Appendix A.11 gives the
argument).
- **The shield is a per-step condition on a learned critic, not a guarantee.** What holds by
  construction is only this: on every shielded step with a non-empty allowed set, the executed
  action satisfies max_j V^j_H(s_t, a) ≤ τ_t for the learned V. Nothing holds on empty-set steps
  (best-effort exit), nor, for BVS-U, on untriggered steps.
  - The condition is not recursively feasible. Feasibility at t+1 would need all of: an exact
    critic; deterministic transitions (the lidar observation makes these tasks partially observed, so
    they are not); the minimising action among the candidates; m ≥ -ε; and a saturation gap
    V_H - V_{H-1} of about 0. A gap of about 0 is necessary but not sufficient.
  - Even under those ideal conditions with an allowed set that is never empty, the episode bound
    J ≤ d - max(m, 0) follows from V*_1 ≤ V*_H alone. The H-step lookahead's only role is to keep
    the allowed set non-empty.
  - The words "invariant", "recursively feasible" and "guarantee" are never used for it. It is called
    "the per-step budget condition"; the empty-set count and F measure its failures (section 14.4).
- **The margin bound is a conditional training-average bound** (Appendix A.6). It assumes A: every
  training episode whose acting margin is ≥ 25 has cost ≤ d. A is pathwise and concerns noisy training
  episodes run with the live V. With the margin dual active from 50k the bound is 26.24 on the Goal
  tasks (K = 950 episodes) and 25.36 on PointCircle1 (K = 1900, J_max = 500).
  - The strict-shield evaluation is evidence about A, not a check of it: it is deterministic, on
    held-out layouts, with a different candidate sampler.
  - With the projected integrator (A.6) A also bounds every window of W counted training episodes, by
    25 + 1350/W on the Goal tasks. Neither bound says anything about a single episode, a snapshot,
    in-training or held-out evaluation, or deployment.

**Scope of "best fair baseline".** B* is the best of the TD3-family baselines below, all run in our
code at the same budget.
- No SAC-Lagrangian, CVPO or CAL is run. CAL uses a SAC base, an update-to-data ratio of 10-20 and a
  legacy simulator; a TD3 port at one update per step would not be CAL.
- Any claim is "beats the best TD3-family baseline we tuned", never "beats the state of the art".
- If B* uses the OmniSafe base, the claim also says "with a different base learner" (section 12).

## 2. Setup

| Task | Horizon | Role |
|---|---|---|
| SafetyPointGoal1-v0 | 1000 | selection task (Stage 0, P1) and confirmatory |
| SafetyCarGoal1-v0 | 1000 | P2 and confirmatory |
| SafetyPointCircle1-v0 | 500 | P2 and confirmatory |

- The cost limit is d = 25 per episode on every task.
- Software: safety-gymnasium 1.0.0, mujoco 2.3.3, gymnasium 0.28.1, numpy 1.23.5, torch 2.7.1,
  Python 3.10, single-thread torch, macOS arm64 (`lagu/requirements.txt`). OpenSSL 3 is used for
  sealing.
- Every run is 1M environment steps. Every executed step counts toward the 1M: random warm-up,
  shielded, fallback and best-effort-exit steps.
- At most 12 concurrent runs. The queue is `safe2/launch.py`, a power-aware queue adapted from
  `lagu/launch.py`. It manages `safe2.train` runs and counts every live `safe2.train` study process
  (frozen ones included) in the 12-slot cap. It also reserves the Lag-U study's slots: per checkout,
  max(running, min(cap_L, alive + queued)), where alive and running count its live `lagu.train` runs
  (all, or not stopped) and cap_L and queued are read from that checkout's Lag-U queue files. So the
  method queue never takes a slot the Lag-U scheduler is about to refill, and a frozen Lag-U run keeps
  its slot until the Lag-U cap is set to 0 (SPEC §6.3).
- **Deployment information.**
  - BVS, BVS-U, BVS-disc, Sauté and the post-hoc shield observe the realised in-episode cost C_t at
    deployment (Safety-Gymnasium returns it every step).
  - The other B-family configs, SHL, R1 and R2 do not use it.
  This is stated wherever results are reported.

## 3. Arms

### 3.1 Baseline pool (the B grid and Sauté); B* is chosen from all thirteen configs

All baselines run in `safe2/base.py`; `safe2/SPEC.md` §2 gives every flag.

**OmniSafe TD3 base** (B1-B6, B8, B9):
- actor lr 5e-6, critics 1e-3, gradient clip 40;
- actor loss / (1 + λ) (pi_norm); cost target from the online actor (cost_target_online);
- 25k random warm-up, replay 1M, one twin reward critic (M = 1), action noise 0.1;
- cost-target clamp to [0, 1/(1 - γ)] (cost_clamp).

**Dual** (every Lagrangian config):
- It reads the mean cost of the last 50 training episodes that ended after the random warm-up.
  Warm-up episodes never enter this window, as in `lagu/train.py`.
- It is updated every environment step from step 200k (B8 and B7c: from 50k). λ is capped at 100.
- PID semantics follow OmniSafe's PIDLagrangian:
  - δ = J̄ - target, with target 25 (B9: 20);
  - I ← max(0, I + k_i δ);
  - P = k_p EMA_0.95(δ);
  - D = k_d max(0, EMA(J̄)_t - EMA(J̄)_{t-10});
  - λ = clip(P + I + D, 0, 100).
- **What these controllers are in practice** (stated with every result). The dual runs every
  environment step on a window mean that changes only at episode ends, so k_i is per environment step
  (per 1000-step episode I grows by 1000 k_i δ). The P-EMA catches up within about 100 steps of each
  window change, and D is non-zero only as a spike of about 100 steps after each change. With k_d = 0
  (every PID config but B3, whose k_d is 1e-8) the PID configs are PI controllers. The cap of 100
  departs from OmniSafe's `sum_norm` behaviour but cannot bind at these gains. The Adam dual moves λ by
  about its learning rate per step whatever the size of J̄ - target.

| Config | Base | Dual | Settings |
|---|---|---|---|
| B1 | OmniSafe | Adam | lr 5e-7 (OmniSafe TD3Lag rate) plus clamp |
| B2 | OmniSafe | Adam | lr 2e-6 |
| B3 | OmniSafe | PID | k_p 5e-7, k_i 1e-8, k_d 1e-8, D delay 10, EMA 0.95 (OmniSafe TD3PID.yaml gains) |
| B4 | OmniSafe | PID | k_p 0.005, k_i 3e-8, k_d 0 |
| B5 | OmniSafe | PID | k_p 0.005, k_i 1e-7, k_d 0 |
| B6 | OmniSafe | PID | k_p 0.02, k_i 1e-7, k_d 0 |
| **B7** | **literal TD3** | PID | B5 gains, plus pi_norm and cost_clamp |
| **B7b** | **literal TD3** | PID | B4 gains, plus pi_norm and cost_clamp |
| **B7c** | **literal TD3** | PID | B5 gains, plus pi_norm and cost_clamp, **dual from step 50k** |
| **B8** | OmniSafe | PID | B5 gains, **dual from step 50k** |
| **B9** | OmniSafe | PID | B5 gains, **dual target 20** (a lower operating point) |
| Sauté-a | literal TD3 | none | Sauté, n = 0.05 |
| Sauté-b | literal TD3 | none | Sauté, n = 0.2 |

- **Literal TD3 base** (B7 family, Sauté, R2, BVS):
  - lr 3e-4 for every network, batch 256, γ 0.99, policy delay 2;
  - target noise 0.2 clipped at 0.5, action noise 0.1;
  - M = 1, replay 1M, 10k uniform warm-up.
- **The B7 family** (literal base with a PID dual) gives the baseline side the same fast base learner
  as BVS. It has three configs because one config would be a weak test of that base, and the B7
  prediction of section 17 expects it to be fragile:
  - B7 uses the middle gain;
  - B7b uses the slower integral gain;
  - B7c starts the dual early.
  This reduces the base-learner confound but does not remove it. If B* uses the OmniSafe base, the
  primary contrast still compares different learners. The exploratory BVS* − L* contrast
  (section 14.1) and the wording rule of section 12 handle this.
- **L\*** is the best literal-base baseline config among B7, B7b, B7c, Sauté-a and Sauté-b. It is
  ranked by the selection rule on seeds 0-2 and recorded at `method-v0.5`.
- **Why B5's gains for B7, B7c, B8 and B9.** B5 is the middle of the PID grid and the gain shared
  with SHL. It was chosen before any run of this study. B8 and B7c start the dual at 50k, the step at
  which BVS's margin dual starts, so a baseline also has an early realised-cost controller.
- **B9** targets a mean training cost of 20.
  - A baseline with a lower operating point would make it a better B* if the endpoint still
    favoured a lower cost spend. The arm-level endpoint (section 11) favours it much less than a
    per-run score would, but not by zero (section 13).
  - B9 lets the baseline side choose such an operating point. BVS's deployed cost is not tied to its
    training target either.
- **Sauté** (Sootla et al. 2022, our port):
  - The state is augmented with b_t = clip((d - C_t)/d, -1, 1), the same budget input as BVS.
  - Once the remaining-budget fraction z_t = (d - C_t)/d is ≤ 0, the reward is replaced by -n.
  - **Disclosed variant.** The reward is shaped on the current safety state (C_t ≥ d), while the
    authors' wrapper shapes on the next one (C_t + c_t ≥ d). The step whose cost exhausts the budget
    keeps its reward here, a one-step lag per budget crossing; the process stays Markov in (o, b). The
    port is kept as coded and reported as "Sauté (one-step-lag variant)". Sauté's almost-sure
    satisfaction result (n → ∞) is not claimed for n = 0.05 or 0.2.
  - Both values of n exceed the per-step reward scale of about 0.026 (about 2× and 8×). They were set
    from that scale before any run.
- **Asymmetry.** The baseline pool gets 13 configs plus 3 extra seeds on its top 3 (section 6). BVS
  gets 4 configs. BVS's defaults were, however, shaped by design-stage oracle work (Appendix D).

### 3.2 Primary candidate: BVS (full specification in Appendix A)

- **Proposer.** A proposer π(s, b) is trained by TD3 on reward only, with b_t = clip((d - C_t)/d,
  -1, 1). It proposes a_π.
- **Viability critic.** A twin critic with H = 32 heads (head h = minimum achievable cost within h
  steps; fixed-horizon TD, undiscounted). It allows a iff max_j V^j_H(s_t, a) ≤ τ_t, where
  τ_t = max(ε, d - m_t - C_t), and τ_t = ε once the budget is spent.
- **Takeover.** If a_π is not allowed, 32 candidates are screened: 16 perturbations, 15 uniform
  actions and the backup actor π_b(s_t).
  - An allowed one is chosen by rule S-Q (argmax of min_j Q_r) or S-near (nearest to a_π).
  - If none is allowed, the best-effort exit (argmin of V_H) is executed.
- **Training.**
  - The proposer is trained Recovery-RL style on transitions relabelled with the proposed action and
    an intervention penalty κ, so it learns approximately the MDP that includes the shield. This is
    not unbiased for the deployed shielded MDP (Appendix A.5): the threshold depends on the margin m,
    which is not an input; the shield (V, π_b, and BVS-U's T) keeps changing; and the 1M replay
    evicts nothing, so Q_r fits a mixture of every shield since 25k plus 25k unshielded steps.
  - The margin m is a realised-cost dual, updated after every training episode that ends after step
    50k.
  - No λ and no cost term enter any actor gradient.
- **When the shield is on.** From step 25k, in training and at deployment. Appendix A pins every
  update schedule, random stream and margin definition.

### 3.3 Exploratory candidate: SHL-PID (Appendix B)

- **What it is.** The B-family base with a short-horizon cost critic: γ_c ∈ {0.90, 0.95}, a
  rate-normalised penalty s = (1 - γ_c)/(1 - γ), and PID gains B4 or B5.
- **Status.** It is **exploratory**: it is not in the primary family, gets no "beats" claim, and its
  p-values are reported unadjusted, as exploratory.
- **Expected effect.** +0 to +3 return. This is below what this design can detect (section 15).

### 3.4 Dropped at the design stage: MC-TD3

MC-TD3 (one TD3 network conditioned on a band of multiplier weights, with the operating point set by
a fast realised-cost dual) is not implemented or run.

- **Odds.** Its estimated odds of beating B* were 2-3%.
- **Why.**
  - Selecting a λ from a trained family cannot move the frontier the learner can reach.
  - Close prior work (DVAC, LP3, A-CRL) leaves a narrow delta.
  - Including it would enlarge the primary family.
- **Where its budget goes.** Its roughly 170 process-hours are reassigned:
  - about 92 fund 4 more confirmatory seeds per task for B* and BVS (16 instead of 12), with their
    PHS evaluations;
  - the remaining ~78, plus about 7 from slack, fund the baseline additions made after the review:
    B7b, B7c and B9 in P1 (about 23) and the L* confirmatory arm (up to about 62).
- **Reporting.** It is reported as dropped, with these reasons, whichever way the study comes out.

### 3.5 Controls (secondary contrasts; Appendix A.8-A.9)

- **BVS-U (the original takeover question).**
  - It is BVS* (same networks, shield machinery, backup actor, selection rule, relabel and margin
    dual) except for the **trigger**.
  - BVS intervenes when V rejects a_π. BVS-U enters the takeover branch when Lag-U's ratio u ≥ T_t.
    The ratio is u = sqrt(Var_m Q^m_1)/(|mean_m Q^m_1| + 1e-6) on (s, b, a_π), from a 3-member
    reward ensemble; the population variance (ddof 0, plus 1e-12) is exactly `lagu/agent.py`'s.
  - On a triggered step, the candidate set also contains a_π itself. The S-rule then picks among the
    V-allowed candidates; if none is allowed, the best-effort exit is executed.
  - One difference from BVS remains and is stated with every result. Under S-Q, a triggered step can
    replace a V-allowed a_π by an allowed candidate with a higher min_j Q_r, which BVS never does.
    Under S-near, a triggered step executes a_π whenever V allows it.
  - T_t is a running (1 - ρ*) quantile of u. It is set so that BVS-U's **trigger rate** 1[u_t ≥ T_t]
    equals ρ*, BVS*'s P1 intervention rate (frozen at `method-v1`; section 6). It matches the trigger
    rate, not the intervention rate: a triggered step can execute a_π, so BVS-U's intervention rate is
    at most its trigger rate. Both are reported.
  - **It is not Zhang et al.'s Eq. 16**, and is never called that. The ratio formula is exactly the
    Lag-U port's (`lagu/agent.py`), but:
    - it uses |Q̄| + 1e-6 (the port's documented deviation), not the signed Q̄, so a state with Q̄ near
      0 or negative can trigger; with the relabelled reward r - κI, Q̄ can be near 0 in
      intervention-heavy regions;
    - T is a running (1 - ρ*) quantile, not the fixed 0.07 (0.07 is used only from 25k to 35k, and
      its counterfactual trigger rate is logged);
    - u is evaluated online at the noisy training proposal a_π, on critics that also see b, and it
      gates an action override; in Eq. 16 it tightens the constraint threshold in the actor's loss at
      replay states;
    - T is frozen at a quantile of u over noisy training proposals, while deployment proposals are
      deterministic, so the deployed trigger rate is not ρ*. This is stated with the contrast, next to
      the held-out intervention rate (interventions per step), which bounds it from below.
    The paper calls BVS-U "Lag-U's ported uncertainty ratio used as a rate-matched shield trigger".
  - The contrast therefore asks which states an uncertainty trigger picks, at the same trigger
    budget. PointGoal1, seeds 300-311.
- **BVS-disc.**
  - BVS* with V replaced by a twin discounted minimum-cost critic Q_b, in the same budget test.
  - Its discount is γ_b = 1 - 1/32 = 0.96875, whose effective horizon equals H = 32, and its targets
    are clamped to [0, 32].
  - The contrast therefore asks whether a fixed-horizon undiscounted critic beats a discounted one at
    the same horizon. It does not test the ProSh/CAPS setting (γ 0.995, about 200 steps).
  - PointGoal1, seeds 300-311.
- **Post-hoc shield on B\* (PHS).**
  - For every B* confirmatory run, V and π_b are fitted offline on that run's replay (episodes 1-900;
    Appendix A.9). B*'s snapshots are then evaluated behind the BVS* shield (evaluation only, no
    training).
  - PHS − B* isolates "a shield at deployment".
  - BVS − PHS does **not** isolate training with the shield. It also changes:
    - an online V for an offline one;
    - the proposer (BVS's reward-only proposer vs B*'s Lagrangian actor);
    - the budget input of the proposer.
    It is reported as "BVS vs a deployment-only shield on B*".
  - Every PHS V gets the G0b quality metrics: entry recall and false alarms on the run's held-out
    replay episodes 901-1000, and the ranking Spearman (section 5). They are sealed with the run's
    outcomes.
  - All three tasks, seeds 300-315.

### 3.6 Reference arms (never primary comparators)

- **R1: OmniSafe-recipe TD3-Lag as published.** These are the settings of `lagu.train --recipe
  omnisafe`: no clamp, Adam 5e-7, dual from 202k, no λ cap. A golden test ties R1 to the audited
  port. Confirmatory seeds 300-307, three tasks.
- **R2: unconstrained literal TD3** (λ = 0). Selection seeds only (P1, P2). It sets R_ref.
- **R3: Lag-U literal.** These are the existing `lagu-lit` seeds 100-109 at 1M on PointGoal1,
  re-evaluated on the new held-out block (evaluation only, sealed).
  - Their snapshots exist only every 50k. Snapshot j ∈ {800k, 850k, 900k, 950k, 1M} (j = 0..4) plays
    held-out blocks i = 2j and 2j + 1, which is 40 episodes each and 200 in all.
  - R3 is descriptive: different seeds, and an untuned, over-limit method (cost about 50).
  - The study never headlines "beats Lag-U".

## 4. Seeds and layouts

| Integers | Use |
|---|---|
| training seeds 0, 1, 2 | selection: Stage 0 (through P1 B1), P1, P2, control health runs |
| training seeds 3, 4, 5 | extra selection seeds for the top-3 baseline configs only (unused before) |
| training seeds 300-315 | confirmatory: B* and BVS* all 16; BVS-U, BVS-disc, SHL* 300-311; R1, Sauté*, L* 300-307 (unused before) |
| training seeds 300-302 | 3M continuations (B*, BVS*, SHL*); 303-304 only with compute freed by a kill |
| training seeds 900-999 | development: smoke runs and unit tests (never analysed; not blinded, so the trainer never runs them beyond 50,000 steps) |
| layouts 10000-10019 | in-training evaluation, 20 episodes every 50k (same block as `lagu`; selection data only) |
| **layouts 70000 + 20i + k** | **confirmatory held-out block**, i = 0..9 (snapshot), k = 0..19 (episode): 70000-70199; every evaluation on it is sealed, whatever the training seed |
| layouts 90000-90029 | Stage 0, G0a (the same 30 layouts for each actor) |
| layouts 90100-90199 | Stage 0, G0b(ii) oracle-label rollouts; the same procedure for the PHS quality check |
| layouts 11000-11009 | S0-1 extensions only (reused `lagu` held-out layouts; exploratory, existing Lag-U runs) |
| layouts 90500-90999 | development: smoke runs and unit tests |

- **Abandoned block.** The shortlist's held-out block 20000 + 20i + k had already been played by six
  BVS design scripts (layouts 20000-20009 and more) and is abandoned.
- **Registry.** The registry of every range used before this study (Appendix C; `safe2/seeds.py`)
  was built by grepping every `reset(` with a seed, and every `seed_base`, in the repository, its
  worktree and the session scratchpad.
- **Run-time reset helper.** Every environment reset in `safe2` goes through one helper,
  `safe2.seeds.reset(env, block, i)`, which asserts at run time that the layout lies in the named
  block.
- **`safe2/tests/test_seeds.py`** asserts that:
  - the new blocks are disjoint from the earlier ones and from each other;
  - no study seed's first or resumed training layout (seed × 1000 + start) falls in an evaluation
    block;
  - seeds 3-5 and 300-315 were never used;
  - no file in `safe2/` other than `seeds.py` calls `.reset(`.
- **Use of seeds.** Seeds 0-5 never enter a confirmatory contrast. Seeds 300-315 are never used for
  any choice.
- **Pairing.** Contrasts are unpaired: different algorithms share a seed number but nothing else of
  value. Every arm plays the same held-out layouts.

## 5. Stage 0 (design gates; no candidate training)

Stage 0 starts only after `method-v0` is pushed. G0a and G0b need the P1 B1 runs (seeds 0-2, 1M,
checkpoints with replay kept), so they run once those finish. S0-1 and S0-4 can run at once.

**S0-1, the original question "does high uncertainty come just before cost?" (exploratory).**
- The existing result (`results/paper/uncertainty_anticipation`) is reported with the corrected
  wording of section 1.2, scoped to PointGoal1 final snapshots and decoupled reward and cost.
- Extensions, all **exploratory**, on the Lag-U study's existing runs and layouts 11000-11009:
  - predictors added next to the reward-ensemble ratio: the maximum hazard-lidar bin and Q_c(s, π(s)),
    at lead times k ∈ {5, 10, 25, 50};
  - mid-training snapshots: the saved snapshots nearest 0.3N and 0.6N, plus N;
  - CarGoal1 and PointCircle1 (`results/gate_x`);
  - event-aligned mean curves of each predictor from 50 steps before to 10 steps after each hazard
    entry;
  - AUROC and AUPRC, each AUPRC reported against its base rate (y_10 is 4.9-8.7%);
  - seed-cluster bootstrap intervals;
  - for `lagu-tt`, healthy runs only (held-out return ≥ 5) as well as all runs.
- S0-1 gates nothing. BVS-U (section 3.5) is the closed-loop test of the same question.

**S0-2, gate G0a: oracle ceiling (privileged; never reported as a method result).**
- **Proposer:** the deterministic 1M actor of each P1 B1 run (seeds 0, 1, 2). These actors are new;
  the `td3lag-omni` actors used at the design stage are not used.
- **Filter.** As in the design script `sceptic-bvs/sim3_select.py`, committed in `safe2/probe.py`.
  - Budget mode with B = 12 and H = 25, the configuration behind the design-stage 17.5 / 25.2. It is
    fixed here, not tuned on the B1 actors.
  - The filter acts once the episode cost reaches B and the agent is within 0.6 of a hazard. The
    proposed action is kept only if, after it, one of 8 constant evasive actions held for 25 steps
    avoids every hazard (MuJoCo state restore).
  - Otherwise, the viable candidate among 9 grid actions with the best privileged 15-step value is
    executed. The value holds the candidate for 5 steps, then follows the actor for 10 steps; the
    score is -goal distance, minus 10 if a hazard is entered.
  - If no candidate is viable, evasive action 0 is executed.
- **Episodes.** 30 per actor on layouts 90000-90029 (90 in all).
- **Decision.** It is taken at `method-v0.5`, because it needs B*'s pilot return R_B*(P1): the mean
  over B*'s six selection seeds of the selection-window return. **PROCEED iff** the mean over the 90
  episodes has return ≥ max(18, R_B*(P1) + 6) and cost ≤ 25. The +6 is the low end of the design's
  minimum detectable effect in PointGoal1 return units, so the gate passes only if a detectable gain
  is possible. Otherwise BVS is killed.

**S0-3, gate G0b: can V be learned from data nobody has studied?**
- **Data.** The 1M replays of P1 B1 seeds 0-2 (1000 episodes each).
  - V is fitted on episodes 201-900. Episodes 901-1000 are held out.
  - Episodes 1-200 are excluded from the fit for this reason: λ_init = 0 and the clamp touches only
    the cost target, so B1 behaves exactly like `td3lag-omni` until its dual starts at 200k. Its
    first 200k replay rows are expected to be bit-identical to the `td3lag-omni` s0-2 replays that
    were studied at design time. G0b verifies this by diffing rows 0-200k and reports the result
    (Appendix E).
  - The held-out episodes come from the same seeds and recipe on which the design-stage entry
    detector (AUROC 0.948) and these thresholds were developed. This is disclosed and accepted
    (Appendix D).
  - The `td3lag-omni` replays themselves are not used.
- **Fit.**
  - BVS's V (twin, H = 32 heads) and π_b as in Appendix A, with B1's 1M actor in place of the proposer
    in the target candidate set;
  - 100k updates, batch 256, lr 3e-4, torch seed = run seed.
- **(i) Entry recall.**
  - An entry is a step t_e of a held-out episode with cost > 0, no cost in the 12 steps before it,
    and t_e ≥ 12.
  - An alarm at t is max_j V^j_32(o_t, a_t) > 0.5.
  - An entry is caught if an alarm occurs at some t ∈ [t_e - 12, t_e - 7].
  - A false-alarm step is an alarm with no cost in steps t..t+31.
  - Pass iff pooled recall ≥ 0.80 and mean false-alarm steps per held-out episode ≤ 30.
- **(ii) Ranking.**
  - Each B1 actor plays deterministic episodes on layouts 90100, 90101, ..., collecting near-hazard
    states (hazard distance < 0.6). At most 20 states are taken per episode, at least 10 steps
    apart, in time order. An actor stops at 400 states or after 100 episodes.
  - For each state, the oracle label is the cost of the executed step plus the minimum, over the 8
    evasive constant actions held for the next 31 steps, of the cost incurred (state restore).
  - Pass iff at least 1000 states are pooled over the three seeds and the pooled Spearman correlation
    between max_j V^j_32 and the label is ≥ 0.7. Fewer than 1000 states is a fail.
- **PROCEED iff (i) and (ii) both pass.** Otherwise BVS is killed.

**Oracle mechanics and readings fixed at v0** (G0a, G0b(ii) and the PHS quality check; SPEC §8; no
study data were used to set them).
- **State restore.** A restore sets the saved state, runs `mj_forward`, and then puts the saved
  `qacc_warmstart` back, because MuJoCo 2.3's `mj_forward` overwrites it. Without this an episode that
  saved, explored and restored drifts from the plain episode at the 1e-14 level within a few steps;
  with it the episode stays bit-identical, so a G0a episode with no intervention is the actor's own.
- **Step order.** A privileged step is the physics step, then the cost, then the observation: the cost
  computation runs a second `mj_forward`, as `env.step` does, so this order reproduces `env.step` bit
  for bit, cost steps included.
- **G0a.** The 15-step value rollout stops at the first hazard entry and scores the goal distance
  there (score = -distance - 10). The candidate list holds (0, 0) twice, as the design script does; the
  duplicate never wins, since ties go to the first. The filter condition uses the episode cost before
  the step. The actor is the run's final 1M actor with the deterministic `clip(actor(o), -1, 1)`.
- **G0b.** The replay identity compares float32 bit patterns of rows 0-199,999 in insertion order and
  reports "unavailable" if they are no longer stored. Recall pools caught / entries over the three
  runs, false alarms pool steps / episodes. Collection stops as soon as a seed's 400th state is
  labelled. The label's early stops (a zero-cost evasive action; a rollout already above the best) are
  exact. A Spearman correlation with a constant side or fewer than 2 points is NaN, and NaN fails.
- **S0-1.** The `results/gate_x` arms take every run found at run time (stems recorded). The
  "healthy" subset (held-out return ≥ 5) applies to `lagu-tt` on PointGoal1 only, because 5 is a
  PointGoal1-scale threshold. The ensemble ratio is omitted for M = 1 arms and the lidar predictor on
  PointCircle1. AUPRC is step-wise average precision with tied scores as one threshold, bootstrapped
  like AUROC. The relative-risk table and figure of `make.py` are not reproduced.

If BVS is killed at Stage 0:
- BVS-U, BVS-disc and PHS are not run;
- the primary family is empty;
- the study reports the baseline pool, SHL (exploratory) and the gates.

**S0-4, SHL premise (descriptive).** The committed gradient probe (`safe2/probe.py`, adapted from
the design script `simplest/h3_grad.py`, 100 episodes per run) does two things:
- it re-measures the quiet-state share of penalty action-gradient mass on the runs where 0.80-0.87
  was measured;
- later, it measures SHL's own share (kill rule, section 8.4).

## 6. P1 pilot (PointGoal1, seeds 0-2, 1M) and selection

- **Runs.**
  - B1-B9, B7b, B7c, Sauté-a, Sauté-b and R2: 42 runs.
  - After the baseline configs are ranked: seeds 3-5 for the top 3 (9 runs).
  - BVS, 4 configs: 12 runs, after G0a and G0b pass.
  - SHL, 4 configs: 12 runs.
  - After BVS* is chosen: health runs of BVS-U and BVS-disc on seeds 0-2 (6 runs). They choose
    nothing.
- **Fixed tuning budget.** Each method's base recipe is fixed as specified. Only the listed grid is
  run. No seed is re-rolled.
- **Selection data.**
  - The in-training evaluation rows at 800k, 850k, 900k, 950k and 1M: 20 deterministic episodes each,
    on layouts 10000-10019.
  - BVS evaluates with the shield on, using the deployed margin m_dep: the mean acting margin of the
    last 20 training episodes. Its candidate sampler is seeded with the layout id. This is the same
    definition as at deployment (section 11).
  - A seed's R and C are the means over its five rows.
- **R_ref(task)** = the maximum, over R2 and the baseline configs run on that task, of the mean over
  seeds 0-2 of each seed's selection-window return R. Candidates never enter it.
  - PointGoal1's value is frozen at `method-v0.5`.
  - CarGoal1's and PointCircle1's come from P2 (R2 and B*) and are frozen at `method-v1`.
- **Collapse (selection).** A seed collapses if:
  - R / R_ref(task) < 0.2; or
  - any value on its arm's fixed whitelist of logged columns (section 10) is non-finite in any row:
    network outputs and losses, λ or m, return and cost.
  A seed whose selection R or C is non-finite gets R := 0 and C := the largest finite selection C
  among the configs ranked with it, and is collapsed.
- **Selection score** of a config = mean_seeds R - 1.0 × max(0, mean_seeds C - 25). The hinge is
  applied to the config's mean cost, as in the primary endpoint (section 11).
  - A config with collapses in at least two thirds of its seeds is ineligible.
  - Scores within 0.5 of the best are tied. Ties go to fewer collapses, then lower mean cost, then
    lower compute per run (`safe2.stats.rank`).
- **B\*.**
  1. Rank all thirteen baseline configs on seeds 0-2.
  2. Run seeds 3-5 on the top 3, where "top" means eligible configs first, then ineligible ones by
     the same rule if fewer than three are eligible.
  3. B* is the best of those three by the same rule (eligible first) on all six seeds. B* therefore
     always exists; an ineligible B* is reported as such.
  - Sauté* is the better of Sauté-a and Sauté-b by the same rule on seeds 0-2.
  - L* is the best of B7, B7b, B7c, Sauté-a and Sauté-b by the same rule on seeds 0-2.
  - If B* is a Sauté config, no separate Sauté* arm is run. If L* is B* or Sauté*, no separate L*
    arm is run.
  - All of these are recorded at `method-v0.5`.
- **Candidate configs.** BVS* and SHL* are the best of their own 4 configs on seeds 0-2, among
  eligible configs only. If none is eligible, the candidate is killed under rule 8.1(a).
- **Matched rate ρ\*** = Σ I_t / Σ steps over steps 35,001-1,000,000, pooled over BVS*'s three P1
  seeds. I_t = 1 when the executed action is not the proposal.
- **m_PHS(task)** = the mean, over BVS*'s selection-seed runs on that task, of m_dep at the 1M
  snapshot. These runs are P1 for PointGoal1 and P2 for the other two tasks.
- **A stopped or non-finite BVS* seed** (pre-v0 amendment). An eligible BVS* may have one diverged
  seed. ρ*, m_PHS and the 8.2 quantities are pooled over the selection seeds whose inputs exist and are
  finite: for ρ* and 8.2, the rows at 500k and 1M with finite counters (and, for 8.2(c), every
  evaluation row from 500k to 1M); for m_PHS, a 1M snapshot with a finite m_dep. A seed without them
  (stopped after diverging, or with a non-finite input) is left out and listed. If fewer than 2 of the
  3 seeds qualify, BVS* is killed: under 8.2 in P1, and under 8.1(c), as for collapses in 2 of 3
  seeds, on a P2 task without m_PHS. A non-finite pooled value is never recorded.

## 7. P2 (CarGoal1 and PointCircle1, seeds 0-2, frozen configs)

Arms: B*, R2, BVS* and SHL*. No retuning. The selection data and collapse rule are the same. P2 sets
R_ref for the two tasks, sets m_PHS, and applies kill rule (c).

## 8. Kill criteria (applied before the confirmatory stage; a killed candidate is reported, not run on)

8.1 **All candidates.**
- (a) In P1, collapses in at least 2 of 3 seeds of the selected config, or no eligible config.
- (b) The selected config's P1 score is more than 2 standard errors below B*'s:
  - kill iff score_k - score_B* < -2 SE, with the arm-level scores of section 6;
  - SE = sqrt(s_k²/3 + s_B*²/6), where s is the per-seed SD (ddof 1) of the linearised score
    R - w(C - 25), w = 1[config mean C > 25], each floored at 3.0 (`safe2.stats.kill_b`);
  - B* contributes its six seeds.
- (c) In P2, on either task, collapses in at least 2 of 3 seeds, or mean cost > 30 while B*'s mean
  cost ≤ 25.

8.2 **BVS.** BVS is killed if G0a or G0b fails. BVS* is also killed if, in P1 over steps 500k-1M,
**pooled over its three seeds and over the window**, any of these holds:
- the training intervention rate (Σ I_t / Σ shielded steps) exceeds 30%;
- mean F exceeds 3 per training episode (Σ F / Σ episodes; F = cost from the first empty allowed set
  of an episode to its end);
- more than 25% of its in-training evaluation episodes exceed cost 25. These are the rows at 500k,
  550k, ..., 1M: the deployed policy with the shield on, 220 episodes per seed.

The share of training episodes over 25 is reported, not used: the margin dual drives the mean
training episode cost to 25 by design. Seeds without complete finite inputs are handled as in
section 6 (fewer than 2 qualifying seeds kills BVS*).

8.3 **BVS controls.** They are not killed on outcome. They are dropped only if BVS is killed, or
(for throughput) by the cut list.

8.4 **SHL.** SHL* is killed if the quiet-state share of penalty action-gradient mass at 1M exceeds
0.6 in at least 2 of 3 seeds.

8.5 **No fallback.** A killed BVS* or SHL* is not replaced by its runner-up config. The candidate is
simply killed.

## 9. Confirmatory stage (fresh seeds 300-315, 1M steps)

| Arm | Tasks | Seeds | Runs |
|---|---|---|---|
| B* | 3 | 300-315 | 48 |
| BVS* (if not killed) | 3 | 300-315 | 48 |
| BVS-U | PointGoal1 | 300-311 | 12 |
| BVS-disc | PointGoal1 | 300-311 | 12 |
| PHS on B* (evaluation only, with V quality checks) | 3 | 300-315 | 48 evaluations |
| SHL* (exploratory, if not killed) | 3 | 300-311 | 36 |
| R1 | 3 | 300-307 | 24 |
| Sauté* (best Sauté config, if it is not B*) | 3 | 300-307 | 24 |
| L* (best literal-base baseline, if it is neither B* nor Sauté*) | 3 | 300-307 | 24 |
| 3M continuations (descriptive) | PointGoal1 | 300-302 | B*, BVS*, SHL*: 9 |

- Baseline confirmatory runs (B*, R1, Sauté*, L*) may start after `method-v0.5`. Candidate and
  control runs start after `method-v1`.
- B* runs keep their 1M checkpoint (with replay) until PHS is fitted.
- The 3M continuations resume from the 1M checkpoints only after the 1M held-out evaluation is
  written.
  - **Source rule** (pre-v0 amendment). A continuation starts only from a 1M run that finished, did
    not diverge and has its 1M held-out output. A continuation whose 1M source diverged or stopped
    unfinished is never started; its state is "not-run (source diverged|unfinished)", it does not
    block unblinding, and the continuation table lists it with that state.
  - A continuation is not a resume of its 1M run: its resume count starts at 0 and counts only its own
    resumes (the 3-resume limit of section 16 applies to those).
- **Arms that share a config.** Every contrast with Sauté* or L* uses the runs that played that
  config: the Sauté* (or L*) arm when it runs, the L* arm when L* is Sauté* and only L* runs, and B*'s
  runs when the config is B*.

## 10. Blinding and the analysis firewall

The tools are specified in `safe2/SPEC.md` §3-§6 and §9.

**Trainer (`safe2/train.py`).**
- For seeds ≥ 300 the console line, and so the `.log` file the launcher writes, prints only the step,
  steps per second and one finite flag. Uncaught errors print only their type.
- The public `<stem>.csv` holds only health columns:
  - step, episodes, updates, steps/s and wall time;
  - one finite flag per column group of the arm's fixed whitelist: outcomes, network outputs,
    losses, dual or margin;
  - the count of non-finite actions replaced by zeros, `resumed` and the number of resumes.
- Every outcome column goes to a sealed record file under `results/method/sealed/`. These columns
  are returns, costs, violation, cumulative cost, λ, margin, intervention statistics, misses, F,
  critic values and losses, and calibration.
- For seeds < 300 the trainer writes everything to the public CSV, as selection needs it. Columns
  that do not apply to an arm are empty, never NaN.

**Finite whitelist (per arm, fixed).**
- Every arm: return, cost, the reward-critic and actor losses, and the reward-critic value.
- Lagrangian arms add: the cost-critic loss and value, and λ.
- The BVS family adds: the V (or Q_b) and π_b losses, the mean V_H at executed actions, and the
  internal margin.
- BVS-U adds: T and the median u.

A non-finite whitelisted value marks the run as diverged (collapsed). It never makes the trainer
raise, and it never triggers a resume. The trainer replaces a non-finite action by zeros, counts it
and continues to 1M. The launcher never resumes a diverged run.

**Sealing.**
- Sealed outputs are encrypted with OpenSSL: RSA-4096-OAEP wraps a fresh AES-256 key per record or
  file. The public key is `results/method/keys/seal_pub.pem`, committed at v0.
- The private key is held only by the study lead, passphrase-protected and off the repository. Only
  `safe2.analyze --unblind` uses it.
- Every plaintext is padded to a fixed size before encryption (8192 bytes per outcome row, 1 MiB per
  whole file), so sealed sizes, which are public, do not depend on outcomes.
- The SHA-256 of every sealed record and file is appended to `results/method/HASHLOG.tsv`. That file
  is force-added, committed and pushed at least daily. `python -m safe2.seal verify` checks every
  sealed file against it.
- **Every evaluation on the held-out block 70000-70199 is sealed, whatever its training seed.** This
  includes R3's re-evaluation of seeds 100-109.
- **Remaining procedural limit.** Checkpoints, replays, snapshots and PHS weights stay unencrypted,
  because resuming, evaluating and fitting need them in-process. Only `safe2` tools open them, and
  none prints an outcome. This part of the blinding is procedural and is disclosed.
  - The final `.pt` of a blinded seed holds network weights only (no training-cost mean, λ, PID or
    margin state, or counter).
  - **Steps-per-second side channel.** The public steps/s and wall time (health tool, console, CSV)
    weakly track a BVS-family run's screening rate: the shield scores its 32+ candidates only on steps
    where a_π is rejected (or BVS-U triggers), and that rate is tied to cost behaviour. Base arms have
    fixed per-step compute. This channel is disclosed; nobody computes an outcome from it.

**Health tool (`safe2/health.py`).**
- It prints, per run, only its stem, state, last step, steps/s, resume count and finite flags. It
  never opens `sealed/`.
- Liveness comes from the step count advancing and from `safe2/pause.sh status`; crashes come from
  the queue log.
- `safe2/pause.sh` and `safe2/launch.py` match `safe2.train` processes. `--recover` rebuilds
  `safe2.train` commands, and the battery freeze applies to them; the tests below cover this.
- `lagu/status.sh` (which prints a log's last line) is not used on `results/method/`.

**Held-out evaluator (`safe2/evaluate.py`).** It seals its output and prints only "done" or the
error type.

**PHS fitting** reads B* replays and writes weights only. Its quality metrics are sealed.

**Before unblinding**, only run health may be read for seeds ≥ 300: liveness, step counts, crashes
and finite flags. No contrast, arm aggregate or plot of an outcome of seeds ≥ 300 or of the held-out
block is computed.

**Unblinding.** `python -m safe2.analyze --unblind --key <private key>` checks, in order:
1. The tags `method-v0`, `-v0.5` and `-v1` exist locally and on `origin`.
2. Every file hashed at `method-v0` matches its hash: every file of `safe2/` outside the tests
   (`stats.py`, `analyze.py`, `evaluate.py`, `train.py`, `base.py`, `bvs.py`, `seeds.py`,
   `configs.py`, `seal.py`, `posthoc.py`, `probe.py`, the ops files `launch.py`, `health.py`,
   `pause.sh` and `health.sh`, which decide what is resumed, left diverged or unfinished, and
   `SPEC.md`, `README.md` and `HOOKS.md`), the public key, and sections 0-20 and the appendices of this
   protocol. Any file that differs must be listed as a section 22 deviation.
3. The per-function hashes of `primary`, `guards`, `label`, `penalised`, `arm_score`, `hinge_diff`,
   `linearised`, `collapsed`, `impute_task`, `hinge_perm_test`, `strat_perm_test`, `welch_strat` and
   the selection and kill functions match `FUNC_SHA256.json`. If one differs, or if `stats.py` itself
   is a listed deviation, the v0 module decides (loaded from the tag), and the current one is reported
   as a sensitivity.
4. `plan.json` and the R_ref / ρ* / m_PHS record (`v1.json`) match their `method-v1` hashes (one entry
   each unless a listed deviation). `cuts.json`, `HASHLOG.tsv` and `deviations.json` (if any) are
   committed, clean and on `origin`.
5. Every run in `plan.json`, minus the pushed cuts, is finished, diverged, unfinished after 3
   resumes, or a continuation not run under the source rule (section 9), and every finished run has its
   held-out files.
6. The hash log verifies.

Only then does it write `results/method/UNBLINDED.txt` (UTC time, git SHA, and the SHA-256 of
`analyze.py`, `stats.py` and `deviations.json`), decrypt `sealed/`, and produce the confirmatory and
secondary tables in one pass.

**`deviations.json`** (`results/method/`) lists each post-v0 change of a hashed file or decision
function as `{"file" or "function", "entry"}`. Each `entry` must be text that occurs in section 22,
and the file must be committed and pushed. A malformed or unmatched entry refuses unblinding.

**Re-runs of `--unblind`** never replace the first output: a re-run writes `analysis/unblind.<n>`.
If its analysis code or its deviations differ from the first unblinding (by the hashes in
`UNBLINDED.txt`), its primary is labelled "re-run after unblinding (not confirmatory)", and the first
output stands.

**At the analysis deadline** (section 19, pre-v0 amendment). If a planned, non-cut run is still
queued, running or missing at the deadline, or a finished run lacks its held-out output, the primary
is computed anyway, by `--unblind --deadline`. Steps 1-4 and 6 still apply. Each such run is treated
as unfinished for imputation (section 11: the worst case in the candidate arm, B*'s median in B*'s
arm), listed by arm, and the output is labelled "computed at the analysis deadline". Runs that finish
later change nothing; a later re-run is a re-run as above.

**Required tests (at `method-v0`; the full list is `safe2/SPEC.md` §10).**
- A seed-300 smoke run's stdout, log and public CSV contain no outcome column or value, and its
  sealed records decrypt with a test key.
- The health tool's output contains only the allowed fields.
- Every held-out-block evaluation is sealed, whatever the seed.
- `--unblind` refuses when any of these holds:
  - a planned run or held-out file is missing;
  - a hash or decision-function hash differs without a deviation;
  - `plan.json` or `v1.json` differ;
  - `cuts.json` is uncommitted;
  - a tag is missing.
- The launcher, pause script and recovery handle `safe2.train` and skip diverged and 3-resume runs.

## 11. Endpoint (held-out)

- **Per run:** R and C are means over 200 deterministic episodes.
  - The 10 snapshots are at N × {0.82, 0.84, ..., 1.00} (N = 1M; snapshots are saved every 20k from
    0.8N). Snapshot i plays layouts 70000 + 20i + k, k = 0..19.
- **Deployed policy:** the deterministic actor.
  - Sauté tracks z_t.
  - BVS and its controls run with the shield on; b_t and C_t are tracked online. The margin is m_dep,
    the mean acting margin of the last 20 training episodes before the snapshot; this is the same
    definition as in in-training evaluation. The candidate sampler is seeded with the layout id.
    BVS-U's T is frozen at its snapshot value.
  - PHS runs as in Appendix A.9.
  - Secondary BVS modes: shield off, strict shield (τ_t = ε), and, if BVS* uses S-Q, the S-near rule.
- **Primary score per arm and task** (the hinge applies to the arm's means, not per run):
  S_t = [R̄_t - 1.0 × max(0, C̄_t - 25)] / R_ref(task), where R̄_t and C̄_t are the means over the
  arm's runs. The contrast is Δs = mean_t (S_t,BVS* - S_t,B*). Also reported per run: r = R / R_ref
  and c = C / 25.
- **Collapse:** r < 0.2, or a diverged run (a non-finite whitelisted logged value; section 10).
- **Imputation** (`safe2.stats.impute_task`; it always runs before any decision rule):
  - **Non-finite endpoint, either arm.** R := 0 and C := the worst finite per-run task-mean cost of
    that task in either arm. The run counts as collapsed. The same applies to a run that stopped
    after diverging.
  - **Candidate run unfinished after 3 resumes.** The same worst case.
  - **B\* run unfinished after 3 resumes.** R and C := the medians of B*'s finished runs on that task.
    It counts as collapsed only if that median does.
  - The decision code refuses non-finite input, so a non-finite value can never be scored as a pass
    (Appendix H, item 3).
- **Per-episode quantities** (secondary): the violation share (fraction of the 200 episodes with
  cost > 25) and CVaR10 (mean of the worst 20 episode costs).

## 12. Primary hypothesis and success rule

- **Family.** One hypothesis, H_BVS: Δs > 0. The family-wise level is one-sided α = 0.025 (Holm over
  a family of size 1). If BVS is killed, the family is empty and no confirmatory claim is made. SHL,
  the controls and the reference arms are outside the family.
- **Test.** A stratified permutation test of Δs (`safe2.stats.primary` / `hinge_perm_test`):
  - within each task the runs' (R, C) pairs are re-split at random between the arms, and the
    arm-mean hinge is recomputed for each split;
  - 100,000 permutations, RNG seed 0;
  - p = (1 + #{Δs* ≥ Δs}) / (1 + 100,000); unpaired.
- **Nominal level.** The test rejects at the nominal level **α_test = 0.016**. This keeps the size ≤
  0.025 under the unequal-shape nulls of Appendix G.1, such as a tight candidate against a B* with
  20-40% collapses and equal means. At a nominal 0.025 those nulls reached a size of 0.035.
- **Intervals and sensitivity.** All reported intervals for pooled contrasts are Welch-type
  stratified intervals (variance summed over tasks and arms, Satterthwaite df).
  - For Δs this is the delta method: the intervals use the linearised per-run scores
    [R - w(C - 25)] / R_ref, w = 1[arm mean C > 25], whose arm means equal S_t.
  - The Welch p is reported next to the permutation p and does not change the verdict.
  - Bootstrap percentile intervals are not used for inference.
- **Guards (point estimates, all required).**
  - G-r: no task has Δr_t = (mean R_BVS* - mean R_B*) / R_ref < -0.10.
  - G-c: in every task, BVS*'s held-out mean cost ≤ max(25, B*'s held-out mean cost) + 5.
    - The +5 tolerance is new. Without it, two arms with equal true cost at the limit pass the
      per-task test with probability about 5/8 per task, so (5/8)³ ≈ 0.24 over three tasks, for any
      number of seeds.
    - With it, the guard holds 0.67-0.92 of the time at a tie with 16 runs, rising towards 1 with
      more runs (Appendix G.6).
    - Cost above 25 is in any case charged one-for-one in the score.
  - G-col: BVS*'s pooled collapse count ≤ B*'s + 2.
- **Success rule.** "BVS beats the best fair baseline" iff p ≤ α_test and G-r, G-c and G-col all
  hold.
  - If p ≤ α_test and a guard fails, the verdict is "not confirmed (guard ...)".
  - Otherwise the contrast gets the first matching descriptive label from the Welch intervals:
    - Hurts: the Δs 95% upper bound is < 0, or BVS* has at least 3 more collapses.
    - Shift: the Δr and Δc 95% intervals both exclude 0 on the same side.
    - Equivalent: the Δr 90% interval lies inside ±0.10 and the Δc 90% interval inside ±0.20
      (±5 cost units).
    - Inconclusive.
  - At this design's spreads, the Equivalent label is practically unreachable: P ≤ 0.02 at exact
    equality (Appendix G.5). Non-significant results are reported as bounds, never as equivalence
    unless the Equivalent label applies.
- **The primary is always computed**, with the imputation of section 11. A missing run can never
  remove a "Hurts" label.
- **Different base learners.** If B* uses the OmniSafe base, every statement of the primary result
  says that BVS (literal TD3 learner) was compared with a baseline using a different base learner. The
  exploratory BVS* − L* contrast (section 14.1) is reported next to it.
- Constraint satisfaction is claimed only for task means ≤ 25 (as in `prereg.md` section 13).

## 13. Arm-level penalised score; no matched-cost safety hypothesis

**Why the hinge applies to arm means.** The draft applied the hinge per run.
- **The problem.** With a between-run cost SD of 15-25, an arm sitting at a mean cost of 25 then pays
  an expected 0.4·σ_C = 6-10 return units of penalty, because half its runs are above 25.
- **The consequence.** The rule preferred "same return, 10 fewer cost units" to "+5 return at equal
  cost", and it rewarded a lower cost spread. The draft's claim that "12 / 5 now loses to 14.5 / 25"
  was false: that candidate would have been declared better 7-11% of the time (Appendix H, item 1).
- **The fix.** The hinge is now applied to each arm's task mean (section 11).
- **The bias that remains.** For an arm whose true mean cost is at the limit, the plug-in penalty
  max(0, C̄ - 25) is still biased upward by about 0.4·σ_C/√n. That is 1.5-2.5 return units at
  n = 16, against 6-10 for the per-run hinge. A pure under-spender (15 / 15 against 15 / 25) is
  therefore still declared better 0.07-0.10 of the time under the full rule (per-run hinge:
  0.14-0.31).
  - A candidate with the same means and a third of B*'s cost spread is declared better 0.03-0.06
    of the time (per-run hinge: 0.18-0.25).
  - The review's 12 / 5 against 14.5 / 25 example is declared better 0.003-0.015 of the time
    (per-run: 0.08-0.10).
- **B9** gives the baseline side a lower operating point in case this residual favour matters. The
  seed-level penalised score (the per-run hinge) is reported as a secondary endpoint for seed-level
  constraint reliability (section 14.1).

**No matched-cost safety hypothesis.**

The shortlist's H^C (Δc < 0 at Δr > -0.10) rewarded spending less of the budget. Moving 4.7-9.7 cost
units down B*'s own frontier passes it with no improvement. A matched-cost replacement (lower
violation share or CVaR10 at |ΔC| ≤ 2.5) is **not** adopted, because it cannot be tested here
(section 15, Appendix G.4):

- With 16 runs per arm and task and a cost SD of 15-25, the SE of a per-task mean-cost difference is
  5.3-8.8 units. A 90% equivalence interval of the pooled ΔC then has a half-width of 5.0-8.4, above
  2.5: "matched cost" within 2.5 can never be shown.
- Read as a point-estimate gate, |ΔC_t| ≤ 2.5 in all three tasks holds with probability 0.011-0.048
  even when the true costs are equal. The hypothesis would then be tested on an outcome-selected
  subset of studies, with an undefined error rate.
- A second family member would split α (Holm's first step at half the level) and raise BVS's
  detectable effect by about 10%.

Violation share and CVaR10 are instead the **key secondary endpoints** (section 14.1). The words "at
matched cost" are used for them only if the pooled 90% interval of ΔC lies inside ±5 cost units.
This is the same margin as the Equivalent label's ±0.20 c. It is reachable only at the low end of the
assumed cost SD (half-width 5.0 at σ_C 15).

## 14. Secondary and exploratory analyses (descriptive; 95% Welch intervals; no multiplicity correction)

14.1 **Key secondary (BVS* vs B\*).**
- Per-task and pooled Δs, Δr and Δc.
- Violation share and CVaR10.
- The seed-level penalised score: the mean over runs of R - max(0, C - 25), the draft's per-run hinge,
  which reflects seed-level constraint reliability.
- Cumulative training cost at 1M **and over 200k-1M** (the window in which every 200k-start baseline
  dual is active).
- Collapses (Fisher exact test).
- **Exploratory:** BVS* − L*, the best literal-base baseline (seeds 300-307 of L*, unpaired, the
  primary computation with an unadjusted p). It is reported next to the primary whenever B* uses the
  OmniSafe base.

14.2 **Sensitivity.**
- The interquartile mean.
- Welch.
- Dropping resumed runs.
- Dropping imputed runs.
- The primary test repeated without CarGoal1 and without PointCircle1 (each reported, none
  decisive).

14.3 **Controls.**
- BVS* − BVS-U (PointGoal1, seeds 300-311): Δs, Δc, violation share, misses, the trigger rate and
  intervention rate achieved. BVS-U's counterfactual trigger rate at Lag-U's T = 0.07 is reported.
  With every result, the remaining S-Q difference of section 3.5 is stated.
- BVS* − BVS-disc (PointGoal1, seeds 300-311), worded as "fixed-horizon undiscounted vs discounted
  at the same effective horizon".
- PHS − B* ("a shield at deployment") and BVS* − PHS ("BVS vs a deployment-only shield on B*", not
  "training with the shield"; section 3.5), on three tasks, seeds 300-315. Each PHS V's recall, false
  alarms and Spearman are reported with it.
- BVS* shield-off, strict-shield and S-near evaluations against shield-on.
- Contrasts against R1, Sauté* and R3 (R3 is descriptive only).

14.4 **BVS mechanism.**
- Intervention rate; allowed-set size; empty-set count; misses (entries with no intervention in the
  12 steps before); F. These measure the failures of the per-step budget condition (section 1.3).
- Held-out empty-set steps split by the sign of the deployed margin m_dep. With m < -ε the threshold
  drops to ε at the budget crossing (A.4), the one step at which even an exact critic can lose
  feasibility.
- V calibration (V_H(s, a_exec) against the realised cost of the next H steps). This check is
  one-sided: the realised H-step cost of the executed policy bounds V*_H from above in expectation, so
  V_H above the realised cost is evidence of pessimism, but V_H below it is not evidence of optimism.
  The net sign of V's bias is not known in advance (the candidate restriction and the twin maximum bias
  it up, the minimum over noisy candidates biases it down, compounding over the heads), so V is called
  "conservative" only if this measurement shows it.
- The training-average margin bound against the realised training average, and the window bound
  (A.6) against the realised mean cost of each 50k-step interval from 50k (W = its episodes: 50 on the
  Goal tasks, bound 52; 100 on PointCircle1, bound 33.5). Both are conditional on assumption A.
- **The saturation gap** g = max_j(V_H - V_{H-1}) on near-hazard visited states at 1M, pooled over
  BVS* confirmatory runs. V is called "approximately saturated" only if mean g ≤ 0.05 and its 95th
  percentile ≤ 0.25 cost units. Saturation is necessary but not sufficient for recursive feasibility
  (section 1.3), so it never licenses "invariant".

14.5 **SHL (exploratory).**
- SHL* − B* with the primary computation, labelled exploratory, with an unadjusted p.
- Same-gain contrasts against B4 or B5 in P1.
- The gradient-mass share.

14.6 **Other.**
- Wall-clock time, updates and parameter counts.
- 3M continuations (seeds 300-302, PointGoal1, evaluated at N = 3M on the same held-out block).
- The full P1 and P2 tables.
- S0-1.

## 15. Power (recomputed for the final rules; `python -m safe2.power`, Appendix G)

The synthetic runs have a return SD of 6-9 and a cost SD of 15-25, in PointGoal1 units
(R_ref 26.3).

**Limit.** Only PointGoal1-scale SDs were simulated. The CarGoal1 and PointCircle1 SDs are assumed
to be similar. They were not taken from the Lag-U `gate_x` held-out files, because this revision
reads no outcome data beyond what was already cited.

- **Size.**
  - At the nominal α_test = 0.016, the stratified permutation test rejects 0.011-0.024 of studies
    with identical arms (n = 12 or 16 per arm and task, 0-40% collapses, Monte Carlo SE 0.003).
  - It rejects at most 0.025 of the unequal-shape nulls used to set α_test (8000 studies each).
  - The full success rule rejects at most 0.020. Welch on the linearised scores rejects 0.004-0.014.
- **Minimum detectable Δs** (80% power, one-sided 0.016, normal approximation, simulated SD of Δs):
  - 0.19-0.42 at 16 runs per arm and task, which is 4.9-11.0 PointGoal1 return units;
  - 0.22-0.48 at 12 runs.
  - The upper end is the case where both arms sit above 25 with a cost SD of 25. There, every cost
    unit enters the score.
- **Simulated power of the test** for a return gain at equal cost: +3 return 0.11-0.27, +5
  0.26-0.60, +8 0.59-0.93.
- **The guards cost less power than in the draft.**
  - At a tie at cost 25, G-c now holds 0.67-0.91 (draft: 0.23-0.39, capped near 0.24 for any n).
  - The full rule has power 0.10-0.26 at +3, 0.24-0.57 at +5 and 0.49-0.81 at +8 return (draft:
    0.06-0.12, 0.12-0.26 and 0.20-0.33).
  - When BVS sits at cost 20 and B* at 25 with +5 return, the full rule has power 0.49-0.79.
  - Type-I error of the full rule is ≤ 0.020.
- **Under-spending** (15 / 15 against 15 / 25, equal return) is declared "beats" 0.07-0.10 of the
  time, and lower cost spread alone (same means) 0.03-0.06. This is the residual bias of section 13;
  the per-run hinge gave 0.14-0.31 and 0.18-0.25.
- **Kill rule (b)** kills a candidate exactly as good as B* with probability 0.001-0.014 (per-seed
  score SD 3-12).
- **Overall odds.**
  - P(BVS passes G0a and G0b) ≈ 0.3, and P(survives the pilot | Stage 0) ≈ 0.5.
  - The full-rule power for a plausible +3 to +5 gain is 0.10-0.57.
  - So a confirmed "beats" has probability about 0.3 × 0.5 × (0.10 to 0.57) = 1.5-8.6%, **about
    2-9%**.
  - The most likely outcomes are "B* best or tied" and "BVS safer on the secondary endpoints, primary
    not confirmed".

## 16. Exclusions, crashes and bug reruns

- **No run is excluded.**
- **Crashes.**
  - A crashed run resumes from its last checkpoint (every 100k), with `resumed` = 1. The sensitivity
    analysis drops resumed runs.
  - A run is resumed at most 3 times. After that it is "unfinished" and imputed (section 11).
  - **Counting resumes** (pre-v0 amendment). Every resume counts, also when a run crashes again before
    its next checkpoint (the count is one more than the largest of the checkpoint's, the run json's
    count and its list of resumes). A run that crashed before its first checkpoint restarts from
    scratch only with `--restart`, which keeps its history, sets `resumed` = 1 and counts as a resume;
    a run that has started is never silently started again. A `--resume` of a finished run is a no-op
    and is not counted. A 3M continuation counts only its own resumes (section 9).
  - A diverged run (a non-finite whitelisted logged value, or a non-finite endpoint) counts as
    collapsed and is never resumed (section 10).
- **Bug reruns** are allowed only:
  - (a) when a pre-registered test fails, meaning a test in `safe2/tests/` committed before the
    affected runs started that **fails when run unmodified, at the run's commit, in the pinned
    environment**; or
  - (b) when a run crashes.

  Baselines have exactly the same right as candidates. After a fix, every run of the affected arm(s)
  at the affected stage is rerun, never only some seeds. A selected config is never patched in
  place.
- **A defect found any other way** (for example while looking at a disappointing result) is logged
  and reported, but the affected runs stand as run. The fixed code may be run only as a labelled
  exploratory arm. One exception applies, asymmetric by design.
  - **Scope.** It covers a defect in outcome measurement: the trainer's logging, the evaluator, the
    reconstruction of the deployed policy, sealing or the analysis. It also covers a defect that
    gives a candidate information or steps the protocol does not allow.
  - **Effect.** Such a defect **voids a "beats" verdict** unless the corrected exploratory rerun
    reproduces it. It can never void a negative verdict ("not confirmed", Hurts, Shift, Equivalent
    or Inconclusive).
- **Every bug hunt is logged** in section 22 when it starts: the date, what triggered it (test, crash
  or something seen), which outcome data had been seen by then, and what was changed.
- **The primary is always computed** at the analysis deadline (section 19), with the imputation of
  section 11, by `--unblind --deadline` if needed (section 10). Unfinished runs are listed.

## 17. Pre-declared predictions (descriptive; not tests)

- B*'s held-out PointGoal1 mean cost is ≤ 25, with return 13-19.
- B7 (literal base + PID) collapses in at least one P1 seed or ranks below B5 (λ-in-gradient
  fragility on a fast base). The same is predicted for B7b and B7c.
- BVS*'s cumulative training cost over 200k-1M is below B*'s. This is a prediction of direction only;
  the shortlist's 30% figure was an artefact of the 200k dual start and is withdrawn.
- BVS-U has a higher violation share than BVS* at the matched trigger rate.
- PHS lowers B*'s violation share. BVS* − PHS is smaller than PHS − B* on violation share.
- SHL*'s quiet-state gradient-mass share at 1M is ≤ 0.6.

## 18. Reported whichever way it comes out

- Every arm, config, seed and gate outcome, including killed and dropped candidates (MC-TD3) and why.
- The full P1 and P2 tables, and the tuning budget per method: baselines get 13 configs plus 3 extra
  seeds on the top 3; BVS and SHL get 4 configs each. Also the design freedom in Appendix D.
- All primary, secondary and control results, including negative ones, and S0-1.
- Imputed, diverged and unfinished runs, by arm.
- Wording rules:
  - "beats" only under the success rule of section 12;
  - "with a different base learner" when B* uses the OmniSafe base;
  - no "beats Lag-U" headline;
  - never "invariant", "recursively feasible" or "guarantee" for the shield's budget condition, and
    "approximately saturated" only under section 14.4;
  - the claims allowed about BVS's parts (Appendix A.11): executed actions meet the learned per-step
    condition except on reported empty-set steps; a training-average (and window) margin bound
    conditional on assumption A; a proposer trained Recovery-RL style on relabelled transitions;
    BVS-U as Lag-U's ported ratio used as a rate-matched trigger. Not allowed: any per-episode,
    late-training, evaluation or deployment guarantee; unbiased learning in the deployed shielded MDP;
    fidelity to Zhang et al.'s Eq. 16; that assumption A was verified;
  - the PID configs described as PI controllers updated every step (section 3.1), and Sauté as the
    one-step-lag variant;
  - BVS's need for online cost observation stated with every result;
  - BVS-U's and PHS's residual differences stated with their contrasts (section 3.5).
- Code, configs, CSVs (public and decrypted sealed), held-out files, the hash log and this protocol's
  git history.

## 19. Compute plan, calendar and cut list

**Capacity.** 10 plugged-in days × 12 h × 12 slots = 1440 process-hours (ph).
- Day 0 is the day `method-v0` is pushed. All code, BVS included, is written and tested before that,
  while the machine finishes the other job.
- The analysis deadline is day 12 (calendar). Unblinding happens no later than then.

**Per-run cost (1M steps, 12 in parallel).**
- OmniSafe base: 3.0 h (measured 2.2-2.8), ×1.07 averaged over tasks (CarGoal1 ×1.12).
- Literal base: 2.4 h.
- BVS: 4.0 h on PointGoal1 and 4.3 h averaged over tasks (per-update benchmark 4.44 ms vs Lag-U's
  4.52 ms).
- BVS-U: 5.0 h (two extra twin critics). BVS-disc: 4.0 h.
- PHS fit plus shielded evaluation: 0.2 h; its quality check: 0.1 h.
- Held-out evaluation: under 0.1 h per run.

| Stage | Runs | ph |
|---|---|---|
| Stage 0: G0a (about 6), G0b (about 2), S0-1 and S0-4 (about 4) | - | 12 |
| P1: B1-B6 54; B8, B9 18; B7, B7b, B7c 22; Sauté 14; R2 7; seeds 3-5 on top 3 27; BVS 48; SHL 36; BVS-U and BVS-disc health 27 | 81 | 253 |
| P2: B* 19; R2 15; BVS* 27; SHL* 19 | 24 | 80 |
| Confirmatory: B* 154; BVS* 206; BVS-U 60; BVS-disc 48; PHS with quality checks 15; SHL* 115; R1 77; Sauté* 60; L* up to 62 | 228 | 797 |
| Held-out evaluation (all arms, BVS secondary modes, R3) | - | 27 |
| 3M continuations (seeds 300-302, PointGoal1; B*, BVS*, SHL*) | 9 | 60 |
| **Total** | **342** | **about 1229 of 1440 (85%)** |

About 210 ph (15%) is slack for crashes, power interruptions and permitted reruns.

**Calendar (queue priority: the BVS path first, then B*, then the rest).**
- Before day 0: all code and tests, `bvs.py` included, are written; `method-v0` is pushed.
- Day 0-1:
  - launch the P1 baseline grid and R2 (42 runs) and SHL P1 (12 runs);
  - run S0-1 and S0-4.
- Day 1-2:
  - G0a and G0b once B1 finishes; seeds 3-5 on the top 3;
  - select B*, L* and Sauté*; take the gate decisions; push `method-v0.5`;
  - start the B*, R1, Sauté* and L* confirmatory runs and the baseline P2.
- Day 2-3: BVS P1, if both gates passed.
- Day 3-4:
  - select BVS*;
  - run BVS P2, the BVS-U and BVS-disc health runs and SHL P2;
  - apply the kill criteria; push `method-v1`.
- Day 4-8: BVS*, BVS-U, BVS-disc and SHL* confirmatory runs (about 430 ph). Baseline runs fill the
  gaps.
- Day 8-9: PHS fits and quality checks, held-out evaluation, 3M continuations.
- Day 10: unblind with `safe2.analyze --unblind`; write up whichever way it comes out.

**Queueing the plan.** `safe2.launch --from-plan` queues each planned run with the argv recorded in
`plan.json`, except that a relative `--out` is rewritten to the same directory as an absolute path, so
a run's json can differ from `plan.json` in that argument alone.

**Cut list.** Cuts are for throughput only, never in response to results, and are made in this
order. Each cut is a dated deviation, made before unblinding and recorded in `results/method/
cuts.json`. That file is committed and pushed before any planned run is skipped, and `--unblind`
verifies it.
1. 3M continuations.
2. SHL* confirmatory from 12 to 8 seeds per task, then to none.
3. R1, Sauté* and L* from 8 to 6 seeds per task.
4. BVS-U and BVS-disc health runs on seeds 0-2.
5. BVS-U and BVS-disc from 12 to 8 seeds.

**Never cut:**
- B* and BVS* at 16 seeds per task on all three tasks;
- the B grid, including the B7 family, B8 and B9;
- the extra seeds on the top 3;
- the Stage-0 gates, PHS with its quality checks, and held-out evaluation.

**Freed compute.** Compute freed by a kill goes only to 3M continuations for seeds 303-304 (in
addition to 300-302), never to new configs or arms.

**Optional anchor.** A native OmniSafe 3-seed anchor of B1 is optional, outside this budget. It is
run only if slack remains after every planned run, and is descriptive.

## 20. Code commitments and required tests

`safe2/SPEC.md` is the implementation specification. Every file below is committed, hashed and
pushed at `method-v0`.

| File | Content |
|---|---|
| `safe2/seeds.py` | registry, layout blocks, the run-time `reset` helper, `check()` |
| `safe2/stats.py` | arm-mean penalised score, imputation, hinge permutation test, Welch, guards, labels, selection, kill (b), decision-function hashes |
| `safe2/power.py` | section 15 and Appendix G simulations, α_test calibration |
| `safe2/configs.py` | every config with its flags and defaults |
| `safe2/base.py` | OmniSafe and literal bases, Adam and PID duals, dual start and target, cost_gamma and rate_norm, clamp, budget input, unconstrained |
| `safe2/bvs.py` | BVS, the BVS-U trigger and candidate set, the BVS-disc critic, shield, margin dual |
| `safe2/train.py` | shared trainer; Sauté; blinding; whitelist and finite flags; eval every 50k (20 episodes); snapshots every 20k from 0.8N; checkpoints every 100k; resume |
| `safe2/seal.py` | encryption, hash log, `verify` |
| `safe2/health.py`, `safe2/health.sh` | health tool |
| `safe2/launch.py`, `safe2/pause.sh` | method queue, power-aware freezing, recovery for `safe2.train` |
| `safe2/evaluate.py` | held-out block, 10 snapshots, BVS modes, PHS mode, R3 mapping, sealed output |
| `safe2/posthoc.py` | offline V and π_b fit, entry recall (G0b, PHS) |
| `safe2/probe.py` | G0a, G0b, S0-1 extensions, S0-4, PHS quality |
| `safe2/analyze.py` | selection, R_ref, gates, kill rules, plan, cuts, `--unblind` |
| `safe2/SPEC.md`, `safe2/README.md`, `safe2/HOOKS.md` | implementation specification, run book, agent hook interface |

The required tests are listed in `safe2/SPEC.md` §10, together with those of section 10. They
include:
- B1 and R1 against `lagu --recipe omnisafe` golden hashes, with and without the clamp;
- the PID reference sequence, the dual start and target, and warm-up episodes kept out of the window;
- the config-diff tests for the B7 family, B8, B9 and R1;
- the BVS tests of Appendix A.10, the BVS-U and BVS-disc tests, and the SHL tests of Appendix B.3;
- the snapshot round trip, resume and divergence handling;
- the blinding, sealing, health, launcher and `--unblind` tests;
- the Stage-0 oracle and sampling tests.

Every file above and every test in `safe2/tests/` exists before `method-v0`, and every test passes at
v0 (the opt-in smoke files run with `SAFE2_SMOKE=1`).

## 21. Decision log

Filled in at each gate. Every entry is committed and pushed before the stage that depends on it.

- `method-v0` pushed (UTC): 2026-10-14 10:14:53 (tag time; commit c0faadb; pushed with the branch `method` on 2026-10-14)
- Stage 0:
  - G0a mean return and cost;
  - G0b recall, false alarms and Spearman (number of states), and the replay identity result:
  `TBD`
- P1 baselines:
  - baseline scores, top 3, seeds 3-5 scores;
  - B*, L*, Sauté*, R_B*(P1), R_ref(PointGoal1);
  - the G0a and G0b decisions;
  - `method-v0.5` pushed:
  `TBD`
- Candidates:
  - the BVS* and SHL* configs, the P2 results and the kill outcomes;
  - ρ*, m_PHS and R_ref (all tasks);
  - `method-v1` pushed:
  `TBD`
- Hash-log pushes (daily): `TBD`
- Cuts (`cuts.json` commits) and reruns: `TBD`
- Unblinded (UTC, SHA): `TBD`

## 22. Deviations

None yet. (Changes made before `method-v0` are amendments, listed in the closing section "Pre-v0
amendments (2026-10-14)", not deviations.)

---

## Appendix A. BVS and its controls

Notation: d = 25, C_t = the realised cost so far in the episode, b_t = clip((d - C_t)/d, -1, 1).
`safe2/SPEC.md` §2.4 gives the code-level specification. Every choice below is pinned at
`method-v0`.

**A.1 Networks.**
- The proposer π(s, b) (tanh).
- Twin reward critics Q_r^{1,2}(s, b, a).
- Twin viability nets V^j(s, a), j = 1, 2, each with H heads V^j_h = h × sigmoid(f^j_h(s, a)) ∈
  [0, h].
- The backup actor π_b(s).
- Polyak targets (0.005) for all of them. Every network is an MLP 256 × 2, with Adam at 3e-4 and no
  gradient clipping.

**A.2 Viability TD (fixed-horizon TD).** For (s, a_exec, c, s'):
- y_1 = c;
- y_h = c + min over a' ∈ C'(s') of max_j V^{j,targ}_{h-1}(s', a'), for h = 2..H;
- C'(s') = {π_b^targ(s'), π^targ(s', b'), 4 uniform actions}. The uniform actions come from a
  dedicated torch generator, so the reward critics' random stream is the same as a TD3 base's.
- The loss is Σ_j Σ_h (V^j_h(s, a_exec) - y_h)², averaged over the batch.
- Bootstrap on truncation; drop it only on true termination. No head bootstraps from itself. V is
  never in an actor gradient.
- Costs are binary in the installed Safety-Gymnasium (every cost is replaced by its indicator), so
  V*_h ∈ [0, h] and the h × sigmoid range is exact. The fixed point is the minimum over policies that
  pick from C', which is at least V* (conservative). Its estimate's bias has no known sign (section
  14.4).

**A.3 Backup actor.** It minimises the mean over s of max_j V^j_H(s, π_b(s)), and steps only its own
parameters.

**A.4 Shield** (every step from 25k, in training and at deployment).
- τ_t = max(ε, d - m_t - C_t), and τ_t = ε once C_t ≥ d (also when m_t < 0). allowed(a) iff
  max_j V^j_H(s_t, a) ≤ τ_t.
- a_π = π(s_t, b_t), plus N(0, 0.1²) in training, clipped. If a_π is allowed, execute it.
- Otherwise build 32 candidates, in this order: 16 × clip(a_π + N(0, 0.3²)), 15 × U[-1, 1]², and
  π_b(s_t). A is the allowed subset.
  - If A is non-empty: S-Q takes the argmax over A of min_j Q_r^j(s_t, b_t, a); S-near takes the
    argmin over A of ‖a - a_π‖.
  - If A is empty: the argmin over the candidates of max_j V^j_H (best-effort exit).
- Ties go to the first index. I_t = 1[a_exec ≠ a_π], compared **by value**: a best-effort exit or a
  candidate equal to a_π (possible at the action bounds, where a clipped perturbation can duplicate a
  saturated a_π) gives I_t = 0. The same holds in every kind and for ρ*.
- With m_t < -ε, τ_t falls to ε at the step where C_t reaches d. That is the one step at which even
  an exact critic can lose feasibility (section 14.4 reports empty-set steps by the sign of m).
- **Random streams.** In training, the candidates come from the agent's own seeded generator. In
  in-training evaluation and at deployment, the sampler is `default_rng(layout id)`, created at
  each episode start.

**A.5 Proposer on the shielded MDP.**
- The replay stores (s, b, a_π, a_exec, r, c, s', b', I).
- y_r = r - κ I + γ min_j Q_r^{j,targ}(s', b', clip(π^targ(s', b') + clip(N(0, 0.2²), ±0.5))).
- Q_r is regressed at (s, b, a_π). The actor maximises Q_r^1(s, b, π(s, b)).
- V and π_b train on a_exec.
- **What this learns.** For a fixed shield, and given m, this is off-policy TD3 on the MDP whose
  action is a_π and whose transition includes the shield (as in Recovery RL). It is not unbiased for
  the shield actually deployed: τ depends on m, which is not in the input, so the process on (s, b) is
  not Markov across episodes; V, π_b (and BVS-U's T) change during training; and the replay (1M) is
  never evicted, so Q_r fits a mixture of every shield since 25k plus 25k unshielded steps. Q_r is
  also discontinuous in a_π at the boundary of the allowed set, where the deterministic policy
  gradient has no exact meaning. This is stated as a limitation.

**A.6 Margin dual (realised units).**
- The acting margin of an episode is m = clip(m̃, -d/2, d), with m̃ taken when the episode starts.
- After each training episode e that ends after step 50k: m̃ ← max(-d/2, m̃ + η_m (J_e - d)), with
  m̃_0 = 5 and η_m = 0.1. The projection onto [-d/2, ∞) (anti-windup; pre-v0 amendment) stops credit
  from safe early episodes from piling up below the acting margin's floor and paying for later
  violations. Without it, 475 episodes at J = 0 followed by 475 at J = 50 average exactly 25.
- **Deployed margin** m_dep = the mean acting margin of the last 20 completed training episodes (or
  clip(m̃_0) before any). It is the only margin used in in-training evaluation, in snapshots and at
  deployment.
- **Assumption A** (pathwise, about training episodes): every training episode whose acting margin
  is ≥ m_safe = 25 (so τ = ε throughout) has cost J ≤ d. It concerns the noisy training proposals, the
  training candidate sampler and the live V. The strict-shield evaluation (deterministic, held-out
  layouts, layout-seeded sampler) is evidence about A, not a check of it. Each episode that breaks A
  loosens the bound by (J_k - d)/K.
- **Training-average bound** (K = the training episodes that end after 50k). The projection only
  raises m̃, so Σ_k (J_k - d) ≤ (m̃_{K+1} - m̃_0)/η_m always (equality while it does not bind). Under
  A, m̃ ≤ m_safe + η_m (J_max - d), hence
  (1/K) Σ J_k ≤ d + (m_safe + η_m (J_max - d) - m̃_0)/(η_m K).
  - With the dual active from 50k: 26.24 on the Goal tasks (J_max = 1000, K = 950 episodes); 25.36 on
    PointCircle1 (J_max = 500, K = 1900). The bound is tight (worst case found numerically: 26.2367).
  - The draft's K = 975 corresponds to a 25k start.
- **Window bound** (from the projection). Under A, every window of W consecutive counted training
  episodes has mean cost ≤ d + (m_safe + η_m (J_max - d) + d/2)/(η_m W): 25 + 1350/W on the Goal
  tasks (38.5 at W = 100) and 25 + 850/W on PointCircle1.
- Both bounds apply to training episodes only and are conditional on A. They say nothing about a
  single episode, a snapshot, in-training or held-out evaluation (frozen m_dep, no dual update,
  deterministic proposer), or deployment.

**A.7 Defaults, grid and schedule.**
- Defaults: H = 32, K = 32 candidates, 6 target candidates, σ_c = 0.3, ε = 0.5. The literal TD3 base
  (section 3.1).
- **Update schedule.**
  - Steps 1-10,000: uniform random actions, shield off, no updates.
  - From step 10,001: one update per environment step, on one shared batch of 256. Q_r is updated,
    then V; on every 2nd update, the actor, then π_b, then the Polyak update of every target.
  - Steps 10,001-25,000: a_π is executed (shield off).
  - From step 25,001: the shield acts.
- P1 grid: rule ∈ {S-Q, S-near} × κ ∈ {0.02, 0.1}.

**A.8 BVS-disc.**
- V is replaced by a twin Q_b with target c + γ_b × min over C'(s') of max_j Q_b^targ, clamped to
  [0, 32], where γ_b = 1 - 1/32 = 0.96875.
- The test is max_j Q_b^j(s, a) ≤ τ_t (equivalently C_t + Q_b ≤ d - m).
- The shortlist wrote "C_t + Q_b ≤ τ_t". That counts C_t twice, because τ_t already subtracts it. The
  corrected form is the same budget condition as BVS, with the discounted critic.
- The shortlist's γ 0.995 (about 200 steps) is replaced by the matched horizon (Appendix H, item 8).
- Everything else is as in BVS*.

**A.9 BVS-U and PHS.**
- *BVS-U.*
  - **Reward critic.** 3 members, each a twin pair with its own target and independent smoothing
    noise. Each regresses on the same relabelled reward r - κI, bootstrapping from its own targets.
    Member 1 plays BVS's Q_r exactly (actor, target, S-Q). Members 2-3 feed only the trigger.
  - **Trigger quantity.** u_t = sqrt(Var_m Q^m_1(s_t, b_t, a_π) + 1e-12) / (|mean_m Q^m_1(s_t, b_t,
    a_π)| + 1e-6), with the population variance (ddof 0), as in `lagu/agent.py`.
  - **Trigger** iff u_t ≥ T_t. When triggered, the candidate set is A.4's 32 candidates plus a_π, and
    the S-rule (or the best-effort exit) picks a_exec. Otherwise a_π is executed even if V would
    reject it.
  - **Threshold.** T_t = 0.07 (Lag-U's value) from 25k to 35k. From step 35,000, every 1000 steps,
    it is the (1 - ρ*) empirical quantile (linear) of u over the last 10,000 executed steps. It is
    frozen at the snapshot value for deployment. ρ* = 0 means T = +∞, and ρ* = 1 means T = -∞, both
    only from step 35,000: between the shield start (25k) and 35k T is 0.07 whatever ρ* is.
- *PHS.* Per B* confirmatory run:
  - V (twin, H = 32) and π_b are fitted on the run's replay episodes 1-900, using A.2-A.3 with B*'s
    1M actor in C'(s'); 100k updates, batch 256, lr 3e-4, torch seed = run seed.
    - The networks have the run's hidden width (256 in the study); the learning rate is 3e-4 for both,
      whatever B*'s own critic rates are.
    - V and π_b train on the replay's executed action (`a_exec` when the run has one, else `a_pi`).
    - The fit refuses an episode range with a missing episode or one not stored from its first step.
    - The fit file records the checkpoint's SHA-256, the episodes, the updates and the seed. G0b and
      the PHS quality check refuse a fit, fresh or reused, whose record does not match the run.
  - Quality (`probe phs-quality`): G0b(i) on episodes 901-1000, and G0b(ii) with B*'s 1M actor (the
    run's final `.pt`) on layouts 90100+, with G0b's settings unchanged (100k updates; at most 400
    states or 100 episodes). The output is sealed to `sealed/phs/<stem>.quality.json.enc` and holds
    the recall too; nothing is printed but `<stem>: done`. These metrics are reported, not gated.
    - A non-finite V prediction makes recall and false alarms NaN; the count of non-finite
      predictions is reported.
    - PointCircle1 has no hazards, so G0b(ii) collects no state there; its Spearman is reported as
      "not applicable" (NaN), and only G0b(i) describes the PointCircle1 fits.
    - `posthoc recall --sealed` exists for a stand-alone sealed recall, but no plan record uses it.
  - Each of B*'s 10 snapshots is then evaluated behind A.4's shield, with:
    - proposer = B*'s deterministic actor;
    - the BVS* selection rule (S-Q uses B*'s twin-min Q_r);
    - C_t tracked online;
    - m = m_PHS(task) (section 6; frozen at `method-v1`);
    - the sampler seeded with the layout id.

**A.10 BVS tests.** These come from the shortlist, made exact in `safe2/SPEC.md` §10.
- **Golden equivalence.** With the shield off, κ = 0 and I = 0, BVS's actor and Q_r reproduce a
  literal TD3 base with the same budget input and λ ≡ 0 bit for bit (golden hashes, 300 synthetic
  updates).
- **Viability targets.** y_1 = c; y_h reads head h-1; outputs lie in [0, h].
- **Exact values** on a 1-D corridor MDP, within 0.1. This test runs the BVS update code at lr 1e-3,
  Polyak 0.05 and H = 6 (SPEC §10, test 15), not the study's values, so that it converges within a
  test's time.
- **Shield logic** with mocked V and Q_r, including τ = ε once the budget is spent.
- **Episode state** resets.
- **Relabel columns.**
- **Margin dual.** The projection onto [-d/2, ∞); the telescoping identity while the projection does
  not bind and the inequality when it does; the full-run and window bounds on adversarial streams that
  satisfy A; m_dep as the mean of the last 20 acting margins.
- **Update schedule and step accounting.**
- **Snapshot round trip.** Held-out evaluation reproduces in-training evaluation on a layout bit for
  bit, with the same m_dep and layout-seeded sampler.
- **Gradient isolation.** V's `.grad` is None after an actor step. BVS-U's members 2-3 never enter
  the actor or target computation.
- **BVS-U trigger.** ρ* = 0 never triggers, ρ* = 1 always does, and a triggered step under S-near
  with a_π allowed executes a_π.

**A.11 What the shield's budget condition does and does not give** (mathematical audit of
2026-10-14). Notation: V*_h(s, a) is the minimum expected cost within h steps; with c ∈ {0, 1},
V*_1 = E[c | s, a] ≤ V*_h ≤ V*_{h-1} + 1.
- **By construction** (learned V): on a shielded step with a non-empty allowed set,
  max_j V^j_H(s_t, a_t) ≤ τ_t. Nothing else holds unconditionally.
- **With an exact critic, per step:** E[c_t | s_t, a_t] ≤ τ_t. The floor ε = 0.5 still admits an
  expected cost of up to 0.5 per step after the budget is spent, so even then the shield is a soft
  per-step filter, not a hard budget.
- **Idealised episode bound.** If V = V*, transitions are deterministic, and the allowed set is never
  empty from t = 0, then a unit cost can occur only while τ_t ≥ 1, so J ≤ d - max(m, 0). This uses
  only V*_1 ≤ V*_H; the H-step lookahead serves only to keep the allowed set non-empty.
- **Conditional recursive feasibility.** A non-empty allowed set at t implies one at t+1 if (a)
  V_H = V*_H, (b) transitions are deterministic, (c) the minimiser over a' of V*_H(s_{t+1}, ·) is among
  the candidates (or is a_π), (d) m ≥ -ε, and (e) the saturation gap at s_{t+1} is covered by the
  slack τ_{t+1} - (τ_t - c_t) (in particular if V*_H = V*_{H-1} there). (a)-(c) fail in this study:
  V is learned, the lidar observation makes the process partially observed, and the candidate set is
  finite. With stochastic transitions the (H-1)-step condition at t+1 holds only in expectation. So
  this is a design rationale, not a guarantee, and section 1.3's wording applies.
- **Bias of V's target.** The finite candidate set and the twin maximum bias V upward
  (conservative); the minimum over noisy candidate estimates biases it downward (about -0.46σ per head
  where candidates tie, +0.56σ where one dominates, compounding over the heads; the sigmoid floor stops
  the downward drift at 0). The net sign is measured (section 14.4), not assumed.

## Appendix B. SHL-PID; MC-TD3

**B.1 SHL.**
- The B-family base and PID dual.
- Cost target y_c = c + γ_c Q_c^targ(s', π(s')), clamped to [0, 1/(1 - γ_c)].
- Actor loss [-Q_r^1(s, π(s)) + λ s Q_c(s, π(s))] / (1 + λ), with s = (1 - γ_c)/(1 - γ). The
  denominator is (1 + λ), not (1 + λ s), so relative to B4/B5 SHL is not a pure rescaling of the cost
  critic: the normalised cost weight is λ s/(1 + λ) and the reward weight 1/(1 + λ). The paper states
  this loss exactly.
- The level identity s Q_c^{γ_c} = Q_c^{γ} holds only for a constant cost rate; for bursty hazard
  costs s Q_c^{γ_c} is larger than Q_c^{γ} near hazards and smaller far from them. "Same gains as B4/B5"
  is a level match, not a gradient match.
- The reward critic keeps γ = 0.99. `mc_cost_traj` is logged at γ_c.

**B.2 Grid.** γ_c ∈ {0.90, 0.95} × PID gains {B4, B5}. B4 and B5 are its same-gain controls.

**B.3 Tests.**
- Bit-identity to the B config with the same gains when γ_c = γ.
- The cost target and clamp on a fixed batch.
- A finite-difference actor gradient equals the baseline's with Q_c scaled by s.
- The level identity on a synthetic chain, within 5%.
- The PID reference sequence; the dual is off before 200k; resume restores the PID state.

**B.4 MC-TD3 (dropped).** It is specified in the method-discovery shortlist: actor and critics
conditioned on w = λ/(1 + λ) ∈ [0, 0.8], band sampling, and an outer dual on w̄. It is not
implemented. Its odds were 2-3%.

## Appendix C. Seed and layout registry (grep of 2026-10-11)

**Sources.** `/Users/michaelseiranian/Desktop/safe-rl-carla` (the main checkout, including
`results/`), its worktree `safe-rl-carla-method`, and the session scratchpad.
- Every line containing `reset(` with a seed, or `seed_base`, was read.
- JSON `seed_base` values were collected per directory.
- Where a script takes its episode count from the command line, the whole thousand is reserved.
- The hostile review re-grepped `reset(` calls without a literal seed and found nothing missing.

| Layout ids | Used by |
|---|---|
| 0-2 | first training layout of seeds 0-2; `lookahead-filter` (`obs_layout.py`, `sim1.py`, `sim2_oracle.py` reset on 0 and 1); benches |
| 7, 9 | first training layout of scratchpad seeds 7 and 9 |
| 100-119 | first training layout of seeds 100-119 |
| 10000-10099 | `lagu` in-training evaluation (EVAL_SEED + k) |
| 11000-11099 | `lagu` held-out block 11000 + 20i + k (all `heldout.json` in `gate`, `gate_literal`, `gate_x`, `gate_literal_x`, `smoke`), and analyses on 11000-11029: `uncertainty_anticipation`, `uncertainty-before-cost-verify`, `gate_dose`, `dose-verify`, `probe-author`, `g3/forensics`, `governor-feas` |
| 12000-12999 | `probe-author-verify` state episodes |
| 15000-15999 | `feas-bandcond/shield_probe.py` |
| 20000-20999 | BVS design: `lookahead-filter/sim1.py`, `sim2_oracle.py`, `sim3_oraclefilter.py`, `sim4_exit.py`; `sceptic-bvs/sim3_select.py`, `sim3_vortex.py`; `g3/reviewer-forensics/probe.py` (20000-20002) |
| 30000-30999 | `diag-critic/a8_snap_probe.py` |
| 50100-50104 | `redteam/fidelity/a3.py` (50000 + seed, seeds 100-104) |
| 60100-60110 | `redteam/fidelity/a4.py` (60000 + seed, seeds 100-110) |
| 424242, 777777 | `redteam` single-layout probes |
| seed × 1000 + start | first layout after a resume (`lagu/train.py`; `g3/resume-audit` harness) |

**Training seeds used before:**
- 0, 1, 2 (`certify`, `e0_dual`, `pareto`, `pilot`, `lr_sweep*`, `smoke`);
- 7 and 9 (scratchpad);
- 100-119 (`gate*`).

The new blocks are listed in section 4. The S0-1 extensions reuse 11000-11009 on purpose, on the
Lag-U study's existing runs only. `safe2/seeds.py` encodes both tables, `safe2.seeds.check()` asserts
disjointness, and `safe2.seeds.reset()` enforces the blocks at run time.

## Appendix D. Design freedom used before this protocol

Everything below was seen before `method-v0` and shaped BVS's defaults or the gates. No BVS training
run exists and no learned V has been fitted.

- **Oracle filter, 20 configurations.** These are `sceptic-bvs/*.txt`, on layouts 20000+, with 10-30
  episodes each and H = 25 throughout.
  - A scripted radial goal-seeker in four parameter settings b0-b3 under strict filtering, and in
    settings b0-b2 under budget mode with B = 10 and with B = 15 (10 rows).
  - Vortex proposers in settings b1-b3: strict, budget B = 12, or unfiltered (7 rows).
  - A vortex proposer with value-aware selection, strict and budget B = 12 (2 rows).
  - A hazard-blind proposer with value-aware selection (1 row).
  - The script parameters varied were the proposer, mode, B, setting, β, τ and the selection rule
    (minimal intervention vs value-aware).
  - Further `lookahead-filter/sim3_oraclefilter.py` variants were run without kept outputs.
- **Avoidability and exit studies** (`lookahead-filter/sim2_oracle.py`, `sim4_exit.py`): 46
  PointGoal1 and 30 CarGoal1 entries of a scripted goal-seeker; exit horizons of 19-26 steps.
- **Replay studies on `td3lag-omni` s0-2** (`results/pareto`):
  - the entry detector (`sceptic-bvs/entry_detect*.py`, last 300 episodes, AUROC 0.948);
  - the per-episode cost structure (`replay_budget.py`);
  - hazard-lidar predictability (`lookahead-filter/replay_struct.py`);
  - the in-training curves of `pareto`, `e0_dual`, `lr_sweep*` and `pilot` (`sceptic-bvs/curves.py`).
  **Overlap with G0b.** G0b's B1 runs use the same seeds and recipe as these runs:
  - their first 200k replay rows are expected to be identical, and are excluded from G0b's fit
    (section 5);
  - G0b's held-out episodes 901-1000 are statistically similar to the detector's training data;
  - G0b's recall and false-alarm thresholds were set knowing what that detector achieved.
- **Timing** (`feas-bvs/bench_bvs.py`, random tensors).
- **Choices made after these:**
  - H = 32 (avoidability p90 plus the exit horizon);
  - the value-aware S-Q rule and the nearest-allowed S-near rule (from the oracle's two selection
    modes);
  - K = 32 candidates and σ_c = 0.3 (by analogy with the oracle's 25-action grid; untested);
  - ε = 0.5, m̃_0 = 5 and η_m = 0.1 (no data);
  - the κ grid (reward scale);
  - the shield start at 25k, and the budget in the state;
  - G0a's B = 12 and H = 25, and its absolute threshold 18;
  - G0b's thresholds.
  The update schedule, random streams and margin definitions of Appendix A were pinned without data.
- **Baselines' freedom.**
  - Dual gains and starts were informed by the Lag-U study's runs (λ trajectories of `td3lag-omni`,
    fixed-λ collapses).
  - Sauté's n comes from the reward scale.
  - B7b, B7c and B9 were added after the hostile review, without any new data.

## Appendix E. Corrections to the method-discovery shortlist

| Shortlist statement | Corrected statement |
|---|---|
| Held-out block 20000 + 20i + k is "new and disjoint" | It was played by six design scripts (20000-20009 and more); replaced by 70000 + 20i + k |
| G0b fits V on the `td3lag-omni` s0-2 replays, last 100 episodes held out | Those replays trained and tested the 0.948 entry detector, so G0b uses the new P1 B1 replays. Those share seeds and recipe with `td3lag-omni` and should be bit-identical to it for the first 200k steps (λ_init = 0, clamp on the cost target only); G0b fits on episodes 201-900 and reports the row diff |
| "A perfect filter reaches 17.5 / 25.2" | Over the limit; the best of 17 filtered configurations; the best at cost ≤ 25 is 15.9 / 20.5 |
| "0 of 158 held-out files reach ≥ 15 at ≤ 25" | 157 real files plus a 20k smoke run; none used a realised-cost dual, so silent about B* |
| "OmniSafe recipe ~20 / ~29 at 1M" | Not reproducible: no held-out file; last in-training evaluation 20.7 / 33.4; 800k-1M 23.2 / 41.1 |
| "Every entry avoidable if evasion starts ≥ 8 steps ahead (p90)" | 90% of 46 sampled entries with 8 steps, all with 10 (median 7, max 10), scripted proposer |
| Reward-ratio AUROC "below the cost critic's" | Not better in any arm; worse in 4 of 5; `lagu-tt` +0.043 [-0.087, +0.167] with 12 of 15 runs collapsed |
| "λ inside the actor gradient drives the collapses: 15 of 15" | 9 literal runs fixed from 10k, 6 Omni runs fixed from 202k; ramps passed 0.3-0.4; points to abrupt λ |
| "A per-episode budget invariant" | Not recursively feasible; a saturation gap of about 0 is necessary but not sufficient (exact critic, deterministic transitions, minimiser among the candidates and m ≥ -ε are also needed). Called "the per-step budget condition", never "invariant" (§1.3, A.11) |
| "BVS cumulative training cost ≥ 30% below B*'s" | Follows from the 200k dual start (about 23% of R1's cost accrues before it); B8, B7c and the 200k-1M window added |
| Novelty "moderate" | Incremental, new only in combination (section 1.3) |
| Margin bound "26.2 for K = 975" | 26.24 (Goal, K = 950) and 25.36 (Circle, K = 1900, J_max = 500) for the 50k start; a training average conditional on assumption A, with a window bound from the projected integrator (A.6) |
| BVS-disc "γ 0.995" | γ_b = 1 - 1/32, the matched horizon |

## Appendix F. Integrity critique (first round): where each fix is applied

| Item | Fix | Where |
|---|---|---|
| A1 | New held-out block, committed registry | §4, App. C, `safe2/seeds.py` + test |
| A2 | G0b on unseen replays, thresholds fixed now | §5 (narrowed in App. H, item 14) |
| A3 | Sealed outcomes, outcome-free health tool, committed at v0 | §10 (encrypted in App. H, item 5) |
| A4-A7 | Headroom, avoidability, uncertainty and λ wording | §1.2, App. E |
| B1 | B7 (literal base + PID + pi_norm + clamp) | §3.1 (extended to B7b and B7c in App. H, item 7) |
| B2 | Post-hoc shield on B*; S-near reported; online cost observation stated | §3.5, §11, §2 |
| B3 | B8 (dual from 50k) and the 200k-1M training-cost window | §3.1, §14.1, §17 |
| B4 | Design freedom disclosed | App. D |
| B5 | BVS-U; no "beats Lag-U" headline | §3.5, §18 |
| B6 | "Best fair baseline" scoped to the TD3 family | §1.3; native OmniSafe anchor optional (§19) |
| B7 | Three extra seeds on the top 3 before B* is frozen | §6 |
| C1 | H^C dropped, with reasons; violation share and CVaR10 key secondary | §13, §14.1 |
| C2 | Penalised score endpoint and the max(25, C_B*) guard | §11, §12 (arm-mean hinge and +5 tolerance in App. H, items 1-2) |
| C3 | Per-task return guard | §12 |
| C4 | Collapse r < 0.2 | §6, §11 |
| C5 | Reruns only for failed pre-registered tests or crashes; same right for baselines; hunts logged | §16 |
| D1 | Stratified permutation test; Welch sensitivity | §12 |
| D2 | Power recomputed | §15, App. G |
| D3 | Primary family BVS only; SHL exploratory; MC dropped; 16 seeds for B* and BVS | §3.3, §3.4, §9, §12 |
| D4 | Kill (b) at 2 SE | §8.1 |
| E | Prior work cited; novelty incremental; "invariant" rule; margin bound scoped | §1.3, §14.4, App. A.6 |
| F | S0-1 extensions, exploratory | §5 |
| G | v0 pushed before any Stage-0 read; firewall | §0, §10 |

**First-round fixes not applied, with reasons.**
- *B1, "and/or a BVS-on-Omni-base ablation".* Not added. The B7 family removes most of the confound
  from the baseline side, by giving the baseline the fast base. An Omni-base BVS would change BVS's
  own learner (its 3e-4 rates are part of the method) and would need its own tuning.
- *B6, native OmniSafe anchor.* Not budgeted. OmniSafe's dependency pins conflict with the study's
  environment, and the golden test already ties R1 to the audited port. It is optional if slack
  remains.
- *C1, the matched-cost replacement.* Dropped instead, with reasons (§13). The critique's own power
  numbers (minimum detectable Δc 10-20 units) support this.
- *C2* offered the penalised endpoint **or** the max(25, C_B*) guard. Both are applied, as decided
  by the study lead. After the hostile review, the endpoint's hinge moved to arm means and the guard
  gained a +5 tolerance (Appendix H).

## Appendix G. Simulations (`python -m safe2.power`; `--calibrate --reps 8000`)

Synthetic only; no study data are read.
- Each run's held-out means: R ~ N(μ_R, σ_R), C = max(0, μ_C + σ_C(ρ z_R + sqrt(1 - ρ²) z)).
- With probability p_col the run has collapsed instead: R ~ N(1.3, 0.5), C ~ |N(5, 5)|.
- Three tasks, in PointGoal1 units (R_ref 26.3).
- The statistic is the arm-mean penalised score, with the test at α_test = 0.016.
- Ranges are over the three SD settings (σ_R, σ_C, ρ) ∈ {(6, 15, 0), (9, 25, 0), (9, 25, 0.5)}.
- 2000 studies per cell and 1000 permutations unless stated.

**G.1 Size.**
- **Identical arms** (both 15 / 25), over σ_R ∈ {6, 9}, σ_C ∈ {15, 25}, ρ ∈ {0, 0.5}, p_col ∈ {0,
  0.2, 0.4}, n ∈ {12, 16}:
  - permutation test 0.011-0.024 (Monte Carlo SE 0.003; exact under exchangeability);
  - Welch on linearised scores 0.004-0.014;
  - full rule ≤ 0.020.
- **Unequal-shape nulls.** These set α_test. They are the review's score-space nulls, mapped to R
  with the costs well below the limit:
  - Setup: n = 16; equal mean score 0.45; a tight arm N(0.45, 0.15) against an arm that collapses
    (N(0.05, 0.02)) with probability 0, 0.2 or 0.4 and is otherwise wide (SD 0.4 or 0.6). Each arm
    is the candidate in turn, in 8000 studies.
  - At a nominal 0.025 the size reaches 0.035 (tight candidate, 40% collapses in B*).
  - The largest nominal level that keeps every size ≤ 0.025 is 0.0165, with a worst size of 0.0248.
    At 1000 permutations this rejects exactly the same studies as 0.016, which is adopted.

**G.2 Power at n = 16, p_col = 0.1.** Columns: the test alone, G-c holding, the full rule, the full
rule with the draft's strict G-c, and the draft's per-run-hinge test and full rule at 0.025.

| Candidate vs B* (return / cost) | Test | G-c holds | Full rule | Full, strict G-c | Draft test / full |
|---|---|---|---|---|---|
| 18 / 25 vs 15 / 25 (+3) | 0.11-0.27 | 0.67-0.91 | 0.10-0.26 | 0.06-0.16 | 0.12-0.25 / 0.06-0.14 |
| 20 / 25 vs 15 / 25 (+5) | 0.26-0.60 | 0.67-0.91 | 0.24-0.57 | 0.12-0.26 | 0.27-0.52 / 0.12-0.24 |
| 23 / 25 vs 15 / 25 (+8) | 0.59-0.93 | 0.67-0.91 | 0.49-0.81 | 0.21-0.34 | 0.57-0.89 / 0.20-0.33 |
| 20 / 20 vs 15 / 20 | 0.41-0.75 | 0.87-1.00 | 0.39-0.73 | 0.26-0.63 | 0.30-0.63 / 0.21-0.54 |
| 20 / 20 vs 15 / 25 | 0.52-0.84 | 0.91-1.00 | 0.49-0.79 | 0.37-0.71 | 0.52-0.88 / 0.37-0.73 |
| 14 / 25 vs 15 / 35 | 0.19-0.47 | 0.88-0.99 | 0.09-0.23 | 0.08-0.21 | 0.23-0.51 / 0.08-0.22 |
| 14 / 25 vs 15 / 45 | 0.78-1.00 | 0.98-1.00 | 0.24-0.39 | 0.24-0.39 | 0.78-1.00 / 0.24-0.39 |
| 15 / 15 vs 15 / 25 (pure under-spend) | 0.08-0.15 | 0.98-1.00 | 0.07-0.10 | 0.07-0.09 | 0.23-0.40 / 0.14-0.31 |
| 12 / 5 vs 14.5 / 25 (review example) | 0.005-0.04 | 1.00 | 0.003-0.015 | 0.003-0.015 | 0.25-0.39 / 0.08-0.10 |
| 15 / 25 vs 15 / 25, candidate cost SD one third | 0.036-0.089 | 1.00 | 0.034-0.062 | 0.025-0.044 | 0.49-0.77 / 0.22-0.25 |
| 15 / 25 vs 15 / 25, cost SD 5 vs 15 (σ_R 6) | 0.046 | 1.00 | 0.043 | 0.034 | 0.48 / 0.23 |
| 15 / 25 vs 15 / 25, cost SD 10 vs 25 (σ_R 9) | 0.057 | 0.99 | 0.044 | 0.035 | 0.44 / 0.18 |

In the last five rows the true arm-level contrast is zero (under-spending, lower cost spread) or
negative (12 / 5), so every entry there is a false rejection or false "beats" rate. What remains
above 0.025 is the plug-in bias of section 13. The B*-over-limit rows are limited by G-r: the candidate's per-task return deficit is
0.038 r.

**G.3 Minimum detectable Δs** (80%, one-sided 0.016, normal approximation, SD of Δs simulated under
the null, 20,000 studies, p_col ∈ {0, 0.2}):
- n = 16: 0.19-0.42 (4.9-11.0 PointGoal1 return units);
- n = 12: 0.22-0.48 (5.8-12.6).
The SD of Δs is largest when both arms sit above 25 with σ_C = 25, because w = 1 and every cost unit
enters the score.

**G.4 Matched cost** (equal true costs, n = 16).
- The SE of a per-task ΔC is 5.3 (σ_C 15) to 8.8 (σ_C 25).
- P(|ΔC_t| ≤ 2.5) is 0.36 / 0.22 per task, and 0.048 / 0.011 in all three.
- The pooled 90% half-width is 5.0 / 8.4.

**G.5 Equivalent label** (equal true means, n = 16, 4000 studies).
- P(Equivalent) is 0.019 at (6, 15, 0, no collapses), 0.003 with 10% collapses, and 0.000 at
  σ_C = 25.
- The median 90% half-widths are 0.077-0.122 for Δr and 0.195-0.293 for Δc, against margins of 0.10
  and 0.20.

**G.6 Cost guard at a tie** (both arms' true costs identical, μ_C 25, p_col 0.1, 20,000 studies;
probability that the guard holds):

| σ_C | n | Strict per task (draft) | Per task + 5 (adopted) | Pooled mean_t[C̄x − max(25, C̄y)] ≤ 0 |
|---|---|---|---|---|
| 15 | 16 | 0.389 | 0.916 | 0.856 |
| 15 | 64 | 0.578 | 1.000 | 0.952 |
| 15 | 256 | 0.888 | 1.000 | 0.999 |
| 25 | 16 | 0.256 | 0.681 | 0.729 |
| 25 | 64 | 0.251 | 0.936 | 0.722 |
| 25 | 256 | 0.254 | 1.000 | 0.730 |

At σ_C 25 the draws are clipped at 0, so the true mean cost is 27.1. Both arms are then above 25, and
the strict and pooled forms stay at a fixed hold probability for any n. The adopted form does not.

**G.7 Kill rule (b)** (candidate best of 4 on 3 seeds; B* the best 6-seed mean among the top 3 of
10; costs below the limit, so the linearised score is the return). P(kill) at a true advantage of
0 / +2 / +4:
- 0.001 / 0.000 / 0.000 (per-seed SD 3);
- 0.008 / 0.001 / 0.000 (SD 5);
- 0.013 / 0.005 / 0.001 (SD 8);
- 0.014 / 0.006 / 0.003 (SD 12).

The shortlist's rule (best of 4 vs best of 8, margin 1.0) gives 0.42-0.61 at an advantage of 0 and
0.07-0.48 at +2.

## Appendix H. Review responses (hostile integrity review of 2026-10-11)

The reviewer's scripts and outputs are in the session scratchpad, under
`method-build/hostile-reviewer/`. Numbers below marked "G" come from `safe2/power.py` with the final
rules (Appendix G). The verdicts are:
- **applied** (as proposed);
- **applied (variant)** (the problem is fixed in a different way, with the reason given);
- **partly applied**;
- **rejected** (with the reason given).

| # | Review point (short) | Verdict | Response and where |
|---|---|---|---|
| 1 | The per-run hinge rewards under-spending and low cost spread; "12/5 now loses" is false | applied (Fix A), plus B9 from Fix B | See note 1 below. |
| 2 | G-c is a fixed 24% lottery at the designed operating point | applied (variant: per-task +5 tolerance) | See note 2 below. |
| 3 | A NaN in B* yields "Beats B*"; `rank` and `kill_b` mishandle NaN | applied | Imputation pre-registered (§11): R := 0, C := worst finite task-mean cost in either arm, collapsed. `primary`, `hinge_perm_test`, `strat_perm_test`, `welch_strat`, `rank`, `selection_score` and `kill_b` raise on non-finite input. Selection imputation added (`impute_selection`). Tests: `test_non_finite_inputs_are_refused_not_scored`, `test_imputation_...`, `test_selection_imputation_...`. |
| 4 | Decision code can change after the pilot data are seen | applied | See note 4 below. |
| 5 | Blinding is procedural; the tooling cannot see `safe2` runs; R3 not sealed | partly applied | See note 5 below. |
| 6 | BVS is coded after Stage-0 and P1 outcomes are visible; schedule unpinned | applied | All code, `bvs.py` included, is at v0 (§0, §19). Appendix A pins the update order, start steps, shared batch, Polyak timing, random streams, τ when the budget is spent, candidate order, tie-breaking, I_t and m_dep. The fallback (`method-v0.1` before any read, with ambiguities logged in §22) is the reviewer's own. |
| 7 | B7 is a single config; "removes the confound" is overclaimed | applied | See note 7 below. |
| 8 | Controls overclaimed: BVS-disc horizon, BVS-U never executes allowed a_π, PHS wording, PHS V unchecked | applied | See note 8 below. |
| 9 | ρ* under-specified | applied | ρ* = Σ I_t / Σ steps over steps 35,001-1M pooled over BVS*'s three P1 seeds (§6). BVS-U matches its trigger rate 1[u ≥ T] (§3.5). u uses `lagu/agent.py`'s population variance (ddof 0, + 1e-12) and \|mean\| + 1e-6 (A.9). |
| 10 | The margin m is defined two ways; the A.10 round trip cannot pass | applied | One definition, m_dep = mean acting margin of the last 20 training episodes, in in-training evaluation, snapshots and deployment (§6, §11, A.6). The sampler is seeded by layout id in in-training evaluation too (A.4). |
| 11 | "Any logged value non-finite" is unusable and asymmetric; NaN could trigger a reseeded resume | applied | Fixed per-arm whitelist (§10, SPEC §3.7). Not-applicable columns are written empty, never NaN. The trainer never raises on non-finite values, and a diverged run is never resumed (§10, §16). |
| 12 | Size checked only under identical distributions | applied (pre-set nominal α) | Unequal-shape nulls added (G.1). α_test = 0.016 keeps their size ≤ 0.025 (8000 studies each). Identical-arm size 0.011-0.024. |
| 13 | Missing runs and found defects can only help the candidate; 16(a) vacuous | applied | Unfinished after 3 resumes: worst case for the candidate, B*'s median for B* (§11). The primary is always computed (§12, §16). An outcome-measurement defect voids a "beats" verdict unless an exploratory corrected rerun reproduces it, and never voids a negative verdict (§16). 16(a) reworded: "fails when run unmodified at the run's commit in the pinned environment". |
| 14 | G0b's data are not fully unseen | applied | Disclosed (§5, App. D, App. E). G0b fits on episodes 201-900 and diffs rows 0-200k against `td3lag-omni`. The held-out episodes' shared seeds and recipe with the detector are disclosed. Fresh seeds or CarGoal1 were not chosen: both cost extra runs, and the overlap is removed from the fit. |
| 15 | G0b(ii) sampling under-specified | applied | At most 20 states per episode, at least 10 steps apart, in time order; fewer than 1000 pooled states fails the gate (§5). |
| 16 | Several rules leave choices open | applied | §8.2 pooled over seeds and window. A killed BVS* or SHL* has no fallback (§8.5). No eligible candidate config means kill (a). Fewer than 3 eligible baselines: ineligible configs fill the top 3 in rank order, and B* always exists (§6). R_ref = max over configs of the mean over seeds 0-2 of each seed's selection-window return (§6). Cuts go into a pushed `cuts.json` that `--unblind` verifies (§19). |
| 17 | Novelty overstated | applied | §1.3 reworded: "new only in combination". Kalweit et al.'s fixed-horizon constraint heads and Sim-to-Lab-to-Real's co-trained backup policy are named. |
| 18 | Overall odds miscalculated | applied (recomputed) | With the final rules: 0.3 × 0.5 × (0.10 to 0.57) = 1.5-8.6%, written "about 2-9%" (§15). On the draft's numbers the reviewer's "about 1-4%" was right. |
| 19 | Margin bound uses the wrong K | applied | 26.24 (Goal, K = 950) and 25.36 (Circle, K = 1900, J_max = 500) (§1.3, A.6). |
| 20 | B8's rationale is false | applied (rationale replaced); premise partly rejected | The dual window excludes warm-up episodes (episodes ending at steps ≤ 25k), as in `lagu/train.py`, which B1 must reproduce. So at 50k the window holds 25 post-warm-up episodes and no warm-up ones. The draft's rationale was nonetheless unclear. B8 (and B7c) are now justified as "matching BVS's 50k margin-dual start" (§3.1), and the window rule is stated. |
| 21 | Sauté's n values do not bracket 0.026 | applied (reworded) | "Both exceed the per-step reward scale (about 2× and 8×)". The values are unchanged, because they were set before any run and changing them now would be arbitrary (§3.1). |
| 22 | MC-TD3's ~170 ph vs the 4 extra seeds | applied | About 92 go to the 4 extra seeds (with PHS). The remaining ~78, plus about 7 from slack, fund B7b, B7c and B9 (about 23) and L* (up to about 62) (§3.4, §19). |
| 23 | Continuation seeds inconsistent | applied | 300-302 always; 303-304 only with compute freed by a kill (§4, §19). `seeds.EXTRA_CONTINUATION_SEEDS = (303, 304)`. |
| 24 | R3 cannot follow the 10-snapshot protocol | applied | Snapshot j ∈ {800k, ..., 1M} plays held-out blocks 2j and 2j + 1 (§3.6, SPEC §5.3). |
| 25 | Power is PointGoal1-only | applied (limit stated) | §15 states the limit. The `gate_x` SDs were not read, because this revision reads no outcome data beyond what the protocol already cites. If the lead wants them, `safe2/power.py` can be re-run with other SD settings before v0, and the change logged. |
| 26 | Equivalence margins disagree; the label is nearly unreachable | applied | "At matched cost" now uses ±5 cost units, the same as the Equivalent label's ±0.20 c (§13). Unreachability is disclosed with numbers: P(Equivalent) ≤ 0.019 at exact equality; median 90% half-widths 0.077-0.122 (Δr) and 0.195-0.293 (Δc) (§12, G.5). The margins were not widened, because that would only make the label reachable, not more meaningful. |
| 27 | `test_seeds` greps only literal seed arguments | applied | `safe2.seeds.reset(env, block, i)` asserts block membership at run time. A test fails if any file in `safe2/` other than `seeds.py` contains `.reset(`. Tests: `test_reset_helper_...`, `test_every_reset_in_safe2_goes_through_the_helper`. |
| 28 | G0a's threshold of 18 is not tied to what can be detected | applied | PROCEED iff return ≥ max(18, R_B*(P1) + 6) and cost ≤ 25. The decision is taken at `method-v0.5` (§5). |

**Note 1 (item 1).**
- The hinge now applies to each arm's task means (§11, §13; `stats.arm_score`, `hinge_diff`,
  `hinge_perm_test`). The intervals use the delta method: Welch on the linearised per-run scores.
- Selection and kill (b) use the same arm-level score, so B* is chosen by the rule it is judged by.
- The "12/5 now loses" claim is deleted. Under the new rules (G), the 12/5 example is declared
  better 0.003-0.015 of the time, pure under-spending 0.07-0.10 and lower spread alone 0.03-0.06.
  Under the per-run hinge these were 0.08-0.10, 0.14-0.31 and 0.18-0.25.
- The residual plug-in bias (about 0.4·σ_C/√n, 1.5-2.5 return units) is disclosed in §13.
- B9 (B5 with dual target 20) is added, so a baseline can choose a lower operating point. Fix B's
  restatement of H_BVS was not adopted; the per-run score is instead reported as a secondary
  "seed-level" endpoint.
- The reviewer's scenarios are in `safe2/power.py` section B.

**Note 2 (item 2).**
- G-c is now: candidate task mean cost ≤ max(25, B*'s) + 5 in every task (§12).
- At a tie with 16 runs it holds 0.68-0.92, and it rises towards 1 with n (G.6). The strict form
  stays near 0.25 for any n at σ_C 25.
- The pooled form was not chosen, for two reasons. Its hold probability is also capped
  independently of n (0.72-0.73 at σ_C 25, G.6). And averaging over tasks lets one task's over-spend
  hide behind under-spends elsewhere.
- Dropping G-c was not chosen, because the guard was the study lead's decision, and it still blocks
  a candidate that buys return with cost well above B*'s.
- §15 states the guard's power cost.

**Note 4 (item 4).**
- `--unblind` verifies every study code file against its v0 hash, and the per-function source
  hashes of the decision functions (`stats.DECISION_FUNCTIONS`, `FUNC_SHA256.json`).
- Any post-v0 change is a §22 deviation. If a decision function changed, the v0 version decides.
- It also verifies `plan.json`, `v1.json` (R_ref, ρ*, m_PHS), `cuts.json` and the tags (§0, §10,
  SPEC §9).

**Note 5 (item 5).**
- `safe2/launch.py` and `safe2/pause.sh` match `safe2.train`, recover `safe2.train` commands and
  apply the battery freeze, with tests.
- Sealed outputs are encrypted to an RSA-4096 public key (private key held by the lead,
  passphrase-protected). Each sealed record's SHA-256 is appended to `HASHLOG.tsv`, which is pushed
  at least daily (§10, SPEC §4).
- Sealing is keyed on the held-out block, not the seed, so R3 and every other held-out evaluation is
  sealed.
- **Not applied:** checkpoints, replays, snapshots and PHS weights stay unencrypted, because
  resuming, evaluating and fitting need them in-process. This remaining procedural limit is stated
  in §10.

**Note 7 (item 7).**
- B7b (B4 gains) and B7c (B5, dual from 50k) are added to the pool (§3.1), for about 14 ph.
- The exploratory BVS* − L* contrast (best literal-base baseline) is pre-registered (§14.1), with an
  L* confirmatory arm at seeds 300-307.
- The wording "with a different base learner" is required when B* uses the OmniSafe base (§12,
  §18). The claim is now "reduces", not "removes", the confound.

**Note 8 (item 8).**
- **BVS-disc:** γ_b = 1 - 1/32, clamped to [0, 32], so the contrast is at the same horizon (§3.5,
  A.8).
- **BVS-U:** a_π is added to the candidate set on triggered steps. The remaining S-Q difference is
  stated with every result (§3.5, A.9).
- **PHS:** "BVS − PHS" is reworded to "BVS vs a deployment-only shield on B*", and the confounds
  (offline vs online V, proposer, budget input) are listed. Each PHS V gets the G0b(i)/(ii) metrics,
  fitted on episodes 1-900 and sealed (§3.5, A.9).

## Pre-v0 amendments (2026-10-14)

Made before `method-v0`, before any study run, Stage-0 read or BVS training run; no outcome data were
seen. They are part of the frozen protocol, not section 22 deviations. Sources: the mathematical audit
(MF = must fix, SF = should fix) and the builders' code reviews of `safe2/`.

| # | Amendment | Where |
|---|---|---|
| 1 | MF-2: the shield's budget condition is a per-step condition on a learned critic; recursive feasibility needs conditions that fail here; "invariant", "recursively feasible" and "guarantee" are banned; the 14.4 label becomes "approximately saturated" | §1.3, §14.4, §18, A.11, App. E |
| 2 | MF-1: the margin bound rests on a pathwise assumption A about training episodes; the strict-shield evaluation is evidence about A, not a check | §1.3, A.6 |
| 3 | MF-3: the margin integrator is projected onto [-d/2, ∞) (anti-windup). The full-run bound is unchanged (26.24 / 25.36); a window bound is added (25 + 1350/W Goal, 25 + 850/W Circle); neither covers single episodes, evaluation or deployment. Code: `bvs.py` `end_episode`; tests: test 21 and a new projection test | §1.3, A.6, A.10, §14.4 |
| 4 | MF-4: BVS-U is not Zhang et al.'s Eq. 16; its differences are listed; it matches the trigger rate, not the intervention rate | §3.5, §18 |
| 5 | MF-5: the proposer's relabel is not unbiased for the deployed shielded MDP (m not an input, non-stationary shield, no replay eviction) | §3.2, A.5 |
| 6 | SF-1: Sauté is disclosed as a one-step-lag variant | §3.1, §18 |
| 7 | SF-2, SF-5: empty-set steps split by the sign of m; V calibration is one-sided; V's bias sign is measured | §14.4, A.4, A.11 |
| 8 | SF-3, SF-4: the PID configs are PI controllers updated every step; SHL's exact loss | §3.1, B.1 |
| 9 | F1: I is defined by value | A.4 |
| 10 | F5: BVS-U's T is 0.07 between the shield start and 35k, whatever ρ*; ±∞ only from 35k | A.9 |
| 11 | Test 15 (corridor) runs at lr 1e-3, Polyak 0.05 and H = 6 | A.10 |
| 12 | F6: state restore puts `qacc_warmstart` back after `mj_forward`; the privileged step computes the cost before the observation; the Stage-0 readings | §5 |
| 13 | PHS fit and quality details: hidden width, executed action, refusals, fit-record checks, `phs-quality` output, PointCircle1 G0b(ii) not applicable, NaN guard | A.9 |
| 14 | M7: ρ*, m_PHS and 8.2 are pooled over seeds with complete finite inputs; fewer than 2 kills BVS* | §6, §8.2 |
| 15 | M3: continuation source rule ("not-run (source ...)"); L1/L8 (ops): a continuation counts only its own resumes | §9, §16 |
| 16 | Resume counting, `--restart`, and the no-op resume of a finished run | §16 |
| 17 | M4: the 12-slot cap reserves the Lag-U study's slots | §2 |
| 18 | L5: the steps/s side channel; the final `.pt` of a blinded seed holds weights only; fixed-size sealed plaintexts | §10 |
| 19 | L7: every non-test file of `safe2/` (ops files, SPEC, README, HOOKS included) is hashed at v0 and checked at `--unblind` | §0, §10 |
| 20 | The `deviations.json` requirements; `HASHLOG.tsv` pushed; the v0 module decides when `stats.py` is a listed deviation; `UNBLINDED.txt` records the code hashes; re-runs are labelled and never replace the first output | §10 |
| 21 | M8: `--unblind --deadline` computes the primary at the analysis deadline, imputing every incomplete run as unfinished | §10, §16 |
| 22 | L8 (ops): `--from-plan` rewrites a relative `--out` to an absolute path | §19 |
| 23 | Analysis L1: contrasts with Sauté* or L* use the runs that played that config | §9 |
| 24 | Dev seeds 900-999 are unblinded and capped at 50,000 steps | §4 |
| 25 | The `method-v0.1` fallback is withdrawn (`bvs.py` and `posthoc.py` exist at v0) | §0 |
| 26 | SPEC-level refinements, adopted as written in `safe2/SPEC.md`: the `seal.encrypt(plaintext, pub_path)` signature (§4); the evaluator's `--root` and `--r3-prefix` options and its `skipped` / `ERROR` lines (§5.3); the final `.pt` format (§3.5); the 3M output name `<stem>.<mode>.n<N>` (§5.3) | SPEC |
