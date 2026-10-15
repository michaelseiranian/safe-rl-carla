# Gradient-parity probe: protocol (prereg.md section 9)

**Label.** EXPLORATORY test of a pre-registered secondary prediction, outside the Holm family. All
other output is DESCRIPTIVE, including the pre-declared K = 3,000 sensitivity analysis (section 5).
Nothing here is confirmatory.

**Prediction (prereg.md section 9, verbatim).** "The frozen-λ return-halving point lies within a
factor of 2 of λ* = ‖∇a Q̄‖ / ‖∇a Qc‖, measured by `lagu/probe.py` (written after Wave 1)."

**Reading of the prediction.** The one-sentence prediction leaves several choices open. This
protocol fixes them as follows, before any study data is read: (a) halving is measured against R0,
the mean return of the retrained actor at λ = 0, not against the checkpoint's own return; (b) λ* is
the ratio of the mean action-gradient norms over 5,000 states; (c) "within a factor of 2" means the
95% interval of the median log2(ρ) lies inside [-1, 1]; (d) the prediction is supported only if it
is supported for both recipes (S1 and S2); (e) "frozen" means both λ and the critics are held fixed
while the actor is retrained.

**Script.** `results/paper/probe-author/probe.py`. It is to be copied unchanged to `lagu/probe.py`
and committed **before** it reads any study snapshot. While writing and revising it, no study
snapshot, CSV or held-out file was read; see section 7. Tests used only `--dry-run` and two scratch
models trained from scratch in the scratchpad (`make.py`, section 8).

## 1. Order of operations

1. Copy `probe.py` to `lagu/probe.py` byte for byte, then commit it together with a dated prereg.md
   section 16 entry (required) giving the probe's SHA-256, so readers can check that the script was
   committed before it read any snapshot; the commit itself is in the git history. The PLAN listed `lagu/probe.py` as "written after Wave 1"; it is in fact written
   after all training. Pick the commit SHA, then run.
2. Run from the repository root:
   `.venv/bin/python -m lagu.probe --prereg --out results/probe --workers 6`
   (or `--workers 3`), on mains power. The probe is not part of the power-aware training queue.
   **Cost.** Per checkpoint the run makes one 20-episode rollout for S, then:
   - at most 1 + 22 × 6 = 133 twenty-episode evaluations: R_ckpt once, then, at each of the 22
     grid points, 3 repetitions × 2 values of K. The earlier estimate counted R0 as an evaluation
     on top of the 22 grid points; R0 is the grid's first point, so at r = 1 the count was 1 + 22,
     not 2 + 22;
   - at most 22 × 3 × 10,000 = 660,000 actor updates (K = 3,000 is nested, so it costs only
     evaluations).

   Measured on the scratch models (section 8) with 3 concurrent processes: 13.1 s per 20-episode
   evaluation, 13.4 s per rollout, and 1.39 ms (M = 3) or 0.83 ms (M = 1) per actor update. One grid
   point therefore costs 120 s (M = 3) or 104 s (M = 1); evaluation is two thirds of it.

   | Scenario | CPU-hours | Wall, `--workers 3` | Wall, `--workers 6` |
   |---|---|---|---|
   | Upper bound: every checkpoint eligible at both K, full grid, no early stop | 32.4 | 10.8 h | about 6.3 h |
   | Lower bound: every checkpoint ineligible at both K (λ = 0 only) | 2.8 | 0.9 h | about 0.5 h |

   - Both rows include the 280 trajectory snapshots, about 1 CPU-hour.
   - The true cost lies between the bounds: ineligible checkpoints skip the grid, and the grid
     stops early.
   - The `--workers 6` figures were not measured with 6 concurrent processes (tests were limited
     to 3). They assume 5 workers at the measured speed and a sixth on one of this Mac's 10
     lower-tier cores. For that sixth worker they use the speed measured under
     `taskpolicy -c background`, 3.96× slower. That is pessimistic, since background QoS also
     lowers the clock; with all six at the measured speed it would be about 5.5 h.
3. Every job file records `probe_sha256`, the SHA-256 of the probe file that produced it. This
   must equal the SHA-256 of the committed `lagu/probe.py`. A cached job is reused only when the
   protocol hash, the snapshot's SHA-256 and `probe_sha256` all match, so output of an older script
   version is never reused.
4. Exit status: 2 if any declared file is missing, else 1 if any job failed, else 0. A missing file
   is listed as missing; it is never substituted. Labels are final only at exit 0. On exit 1, the
   same command is re-run once (cached jobs are reused); a job that fails again is reported as
   failed and counts as not eligible, with the count stated.
5. Report the result whichever way it comes out. `--summarise-only` re-prints the tables and
   figures from the job cache.

## 2. What it reads (and nothing else)

For each declared checkpoint the script reads two files:
- `<stem>.json`, used for `args.env`, `obs_dim`, `act_dim` and `horizon`;
- `<stem>.snap/<step>.pt`, used for actor, members, cost critic, λ and cfg.

It reads no CSV, no `heldout.json` and no other run. The SHA-256 of every snapshot file is
recorded. Stem = `<arm>_SafetyPointGoal1-v0_s<seed>`.

| Set | Role | Directory | Arm | Seeds | Main step | Trajectory steps |
|---|---|---|---|---|---|---|
| S1 | co-primary | `results/gate_literal` | `lagu_nogate-lit` (E1b, literal recipe) | 100-109 | 1,000,000 | 250k, 500k, 750k, 1M |
| S2 | co-primary | `results/gate` | `lagu_nogate-tt` (E1, chosen recipe) | 100-114 | 2,000,000 | 250k ... 2M (8) |
| S3a | descriptive | `results/gate_literal` | `td3lag-lit` | 100-109 | 1,000,000 | 250k ... 1M (4) |
| S3b | descriptive | `results/gate` | `td3lag-tt` | 100-109 | 2,000,000 | 250k ... 2M (8) |

That is 45 main checkpoints and 280 trajectory snapshots.
- No run is excluded by hand; `lagu_nogate-tt` s103 is in S2.
- A missing file is listed as missing and never substituted, and the run exits with status 2.
- Eligibility (section 3, step 5) is mechanical.

**Why these sets.**
- The ungated arms are used because, with δ detached, the frozen-λ actor objective has no gate
  term at all. The gated arms would add nothing to this question.
- S1 and S2 cover the two recipes the study tests: the paper's step sizes, and the two-timescale
  recipe chosen by E0.
- S3 is the TD3-Lag base on which the roughly 9x action-gradient ratio (PLAN F3) was first seen.
- The checkpoint is the run's last step N, the end of the primary-endpoint window.
- PLAN section 3 also names "healthy lr_sweep/E0 finals". These are dropped:
  - lr_sweep has no snapshots and only reproduces S3a's algorithm and step sizes on 2 selection
    seeds;
  - the chosen E0 cell is S2's recipe on 3 selection seeds.

**Table labels.** Every table carries its label from the set's role, in `--prereg` and `--stems`
mode alike: S1 and S2 are EXPLORATORY (the section 9 prediction); S3a, S3b and the trajectory
table are DESCRIPTIVE; the single table of `--stems` mode (hand-picked runs, not the declared sets)
is EXPLORATORY. The K = 3,000 section is DESCRIPTIVE for every set.

## 3. What it computes, per checkpoint

The constants are in `PROTOCOL` in the script, and the job cache is keyed on their hash.

1. **Probe states S.** Deterministic rollouts of the checkpoint's own policy (a = π(s)) on the
   **new layouts 12000-12019** (`EVAL_SEED + 2000 + k`, k = 0..19). S is every state at which the
   policy acted: 20,000 states on PointGoal1.
   - **Not the held-out layouts.** The brief that commissioned this script named the held-out
     layouts (11000 + j) as the source of the λ* states. The script does not use them, for three
     reasons:
     - S has two jobs: it is where λ* is measured, and it supplies every minibatch of the
       frozen-λ updates (step 3). The outcome R(λ) is scored on held-out layouts 11000-11019. Had S
       come from those layouts, the actor would be updated on the very episodes it is then scored
       on, and R(λ) would no longer be an out-of-sample outcome.
     - Layouts 11000-11099 are the study's primary endpoint (prereg section 5). Keeping the
       probe's input states off them keeps λ* independent of the endpoint episodes; only the
       probe's own outcome R(λ) reuses block 0 (11000-11019).
     - No file holds held-out states: `heldout.json` stores summary numbers only, and the probe
       reads no `heldout.json`. The states would have to be regenerated by rollouts anyway, and
       12000-12019 are disjoint from the in-training layouts (10000 + k) and from the held-out
       layouts (11000-11099).
   - The states are still on-policy states of that checkpoint, as PLAN section 3 specifies.
2. **λ\*.**
   - Draw 5,000 states from S without replacement (`numpy.random.default_rng(0)`), as PLAN section 3
     specifies.
   - At each state, with a = π(s):
     - g_r = ∇a Q̄(s, a), where Q̄ is the mean of the members' first heads (`agent.ensemble`, Eq. 10;
       the single head for td3lag);
     - g_c = ∇a Qc(s, a) (`agent.qc_pi`).
   - Euclidean norm over the action dimensions.
   - **λ\* = mean‖g_r‖ / mean‖g_c‖** (ratio of means over the 5,000 states).
   - Also recorded (DESCRIPTIVE): both mean norms, the mean cosine of (g_r, g_c), the median
     per-state ratio, and the fraction of action components with |a| > 0.99.
3. **Frozen-λ updates, r = 3 repetitions.** For each λ in the grid and each repetition i = 1, 2, 3,
   start from the checkpoint actor:
   - K Adam steps at lr 3e-4 (the paper's actor step size), fresh optimizer state, batch 256 drawn
     from S with replacement;
   - repetition i draws its batch sequence with `default_rng(seed_i)`, seed_i = 0, 1, 2; each
     repetition uses its own sequence at every λ;
   - loss = mean(−Q̄(s, π(s)) + λ·Qc(s, π(s))), Eq. 17 with λ frozen (δ is a constant and drops
     out of the gradient);
   - no bonus, no (1 + λ) scaling, no clipping;
   - **critics frozen**: only the actor changes;
   - primary K = 10,000. For the sensitivity K = 3,000 the actor after the first 3,000 of the same
     steps is used. That equals a separate 3,000-step run, because the shorter run's batch indices
     are a prefix of the longer run's (checked in `--dry-run`, with the nested actor compared bit
     for bit against a separate run).
4. **Return.**
   - R_i(λ) is the mean return of repetition i's updated actor over 20 deterministic episodes on
     the held-out layouts 11000 + k, k = 0..19 (the same 20 episodes for every repetition and every
     λ), played with `lagu.train.evaluate`. Cost is kept as DESCRIPTIVE.
   - **R(λ) = mean over the 3 repetitions of R_i(λ).** Every rule below uses this mean curve.
   - R_ckpt is the checkpoint's own actor on the same layouts, with no update (one evaluation; it
     involves no minibatch, so it has no repetitions).
   - R0 = R(0), the mean return after K updates at λ = 0.
5. **Eligibility**, per K, decided once R0 is known. A checkpoint is eligible when all of these
   hold; otherwise it is listed with its reason and plays no grid at that K.
   - R_ckpt ≥ 10 (twice the collapse line of prereg section 5);
   - R0 ≥ 10;
   - R0 ≥ R_ckpt / 2 (the update procedure alone did not halve the return);
   - λ* is finite and > 0.
6. **Grid.**
   - λ ∈ {0} ∪ {0.001·10^(j/4) : j = 0..20}: 0.001 to 100, 4 points per decade, 22 values.
   - Played in ascending order, each λ with all 3 repetitions, for every K still playing.
   - For each K, play stops after two consecutive points with R ≤ R0/4 (mean curve). Such a stop
     always comes after the first crossing, so it cannot change λ_half.
7. **Return-halving point, on the mean curve.**
   - λ_half is the first grid λ_j > 0 with R(λ_j) ≤ R0/2, interpolated linearly in log λ between
     λ_{j−1} and λ_j.
   - If that is the first nonzero point, λ_half ≤ 0.001 (left-censored). If no point qualifies,
     λ_half > 100 (right-censored).
   - **ρ = λ_half / λ\***.
   - **Spread across repetitions (DESCRIPTIVE)**, reported per checkpoint and as set medians:
     - `Rrange`: at each played λ (λ = 0 included), max_i R_i − min_i R_i; the table shows the
       median over the played λ, and `probe.json` also holds the maximum;
     - `l2rng`: each repetition's own halving point λ_half,i, found by the same rule on R_i against
       its own R_i(0), over the played λ only (right-censored at the last played λ when its curve
       has not crossed there), and the range max_i − min_i of log2 ρ_i (censored values imputed as
       in the rule).

**Trajectory (DESCRIPTIVE).** Steps 1-2 at every 250k snapshot, next to the training λ stored in
that snapshot.

## 4. Decision rule

Applied to S1 and S2 separately, at the primary K = 10,000 only.
- Statistic: the **median of log2(ρ)** over eligible checkpoints.
- A censored ρ is imputed one factor of 2 beyond the grid end (λ_half = 0.0005 or 200), counted and
  marked `*`. If λ* itself falls outside [0.002, 50], that checkpoint is flagged in the output,
  because imputation near the grid ends could then push ρ to the wrong side of the band.
- 95% interval: percentile bootstrap of the median of log2(ρ) over checkpoints, 10,000 resamples,
  `default_rng(0)`.
- Point estimate in ρ units: ρ̂ = 2^median(log2 ρ). With an even number of eligible checkpoints
  this is the geometric mean of the middle pair of ρ values, not their arithmetic mean (e.g. ρ =
  1, 2, 8, 9 gives ρ̂ = √(2·8) = 4, not 5). The interval is reported in both units.

| Label | Condition |
|---|---|
| SUPPORTED | the 95% interval of the median of log2(ρ) lies inside [−1, 1] (ρ inside [0.5, 2]) |
| REFUTED | that interval lies entirely above 1 or entirely below −1 |
| INCONCLUSIVE | otherwise |
| NOT EVALUABLE | fewer than 5 eligible checkpoints |

**Overall** (section 9 prediction), first matching line:
1. REFUTED if S1 or S2 is REFUTED;
2. SUPPORTED if S1 and S2 are both SUPPORTED;
3. NOT EVALUABLE if at least one of S1, S2 is NOT EVALUABLE;
4. otherwise INCONCLUSIVE.

| S1 \ S2 | SUPPORTED | REFUTED | INCONCLUSIVE | NOT EVALUABLE |
|---|---|---|---|---|
| SUPPORTED | SUPPORTED | REFUTED | INCONCLUSIVE | NOT EVALUABLE |
| REFUTED | REFUTED | REFUTED | REFUTED | REFUTED |
| INCONCLUSIVE | INCONCLUSIVE | REFUTED | INCONCLUSIVE | NOT EVALUABLE |
| NOT EVALUABLE | NOT EVALUABLE | REFUTED | NOT EVALUABLE | NOT EVALUABLE |

The prediction covers both recipes, so it cannot be supported when one set could not be evaluated;
a refutation in either set refutes it. The per-set labels are always printed next to the overall
label.

Always reported: the per-set labels, the median of log2(ρ) and ρ̂ with their intervals, the
fraction of eligible checkpoints with 0.5 ≤ ρ ≤ 2, the number censored, and the repetition spread.
Also reported, as DESCRIPTIVE: the Spearman correlation of λ_half with λ* (uncensored eligible
checkpoints, n ≥ 5). S3a and S3b get the same numbers labelled DESCRIPTIVE; they do not enter the
overall label.

## 5. Sensitivity analysis (pre-declared, DESCRIPTIVE)

Declared here before any study snapshot is read: steps 4-7 and the section 4 statistics are also
computed at **K = 3,000** frozen-λ updates, with their own R0, eligibility, stop and halving point,
on every checkpoint of every set. They are printed in their own section, labelled DESCRIPTIVE, with
the label the rule would give. They never change a per-set label or the overall label, which are
computed at K = 10,000 only.

Why K: the probe moves the actor at lr 3e-4 for every set, while the tt checkpoints trained their
actor at 5e-6 (section 6). K = 3,000 shows how much λ_half depends on how far the update is run.

## 6. Choices made here, and their limits

- **Frozen critics, fresh optimizer, no environment interaction during the updates.** The snapshot
  files hold only actor, members, cost critic(s), λ and cfg: no target networks, optimizer state or
  replay buffer. That is not true of every file of a run: the final `<stem>.pt`, written by
  `lagu.train.save_final` at the same step N as the main snapshot, holds `agent.state_dict()`,
  which includes the target networks and the training Adam states of actor and critics. Only the
  replay buffer is gone, with the dropped `.ckpt.pt`. So the fresh optimizer and the frozen
  critics are choices, not forced by the files:
  - a fresh Adam, because the stored actor moments were accumulated at the training actor lr
    (5e-6 under tt) and the probe applies one procedure at lr 3e-4 to every set;
  - frozen critics, because training them needs replay data, which no file holds, and because
    frozen critics make the updated objective exactly the one whose action gradients define λ*.
  The cost is that the critics do not co-adapt, as they would in training. The actor may also
  exploit critic errors off the data; the R0 eligibility conditions guard against the procedure
  itself destroying the policy.
- **K = 10,000 at lr 3e-4 for every set.** This follows the PLAN F3 pilot ("10k policy updates at
  frozen λ from a healthy checkpoint"). The tt checkpoints trained their actor at 5e-6, so for S2
  and S3b the probe moves the actor much faster than training did. The question is where the
  frozen-λ objective turns, not how fast the tt optimizer gets there. K = 3,000 is the one
  pre-declared sensitivity (section 5); no lr sensitivity is pre-declared.
- **λ* is a local, action-space quantity; λ_half is a global outcome.** That gap is what the
  prediction tests. The actor update acts on θ through tanh; saturation is reported.
- **Wide grid.** It spans 5 decades so that λ_half is located without knowing λ* in advance.
  Values outside it are censored and imputed, never dropped.
- **Three repetitions per λ, averaged.** One K-step update followed by one deterministic
  20-episode evaluation is numerically chaotic: on a scratch model, changing only the float
  summation order of the action gradient moved actor parameters by about 1e-7 after 200 steps, yet
  moved 2-episode returns by up to several units (verifier's note V6). R(λ) is therefore the mean
  of r = 3 repetitions that differ only in their minibatch sequence (seeds 0, 1, 2), on the same 20
  held-out episodes, and λ_half is read from the mean curve. The spread across repetitions is
  reported so a reader can see how much a single repetition would have moved λ_half. No smoothing
  or monotone fit is applied; the first-crossing rule and the R0/2 threshold are fixed here.

## 7. What the authors knew when writing this

Recorded so a reader can judge the choices above. No study snapshot, CSV or `heldout.json` was
opened while writing or revising the probe.

The author had read:
- prereg.md, PLAN.md and the README;
- the auto-memory notes for this project. These quote:
  - the F3 pilot, at paper step sizes on a healthy TD3-Lag checkpoint, 10k policy updates at
    frozen λ: return 27.8 / 23.6 / 19.2 / 13.4 / 4.9 at λ = 0 / 0.1 / 0.15 / 0.2 / 0.3, so the
    return halves near λ ≈ 0.19. Its script was not kept.
  - a cost/reward action-gradient ratio of about 9 (λ ≈ 0.11) on that class of checkpoint;
  - the confirmatory results already in prereg section 15-16, including collapse counts (E1
    `lagu_nogate-tt`: 3 of 15) and E1b/X1 time-averaged λ of about 0.03 for `lagu_nogate-lit`;
  - a red-team note that under tt the learner tolerated training λ up to about 0.4.

These informed two things:
- the grid range;
- the use of eligibility rules rather than hand-picked healthy runs.

No probe outcome was known. Earlier scratch probes by other agents (Q̄ sign, action saturation)
did read some study snapshots. They did not compute λ* or any frozen-λ return.

The revision (repetitions, K = 3,000 sensitivity, overall NOT EVALUABLE, exit status, cache key,
wording) was made after an independent verifier's review, before any study snapshot was read by
this script. The reviser read prereg.md, this directory, the verifier's notes in
`results/paper/probe-author-verify/` (scratch-model numbers and file metadata only; no study
outcome) and the `lagu/` source, and opened no other file under `results/`.

## 8. Tests run (all reproducible with `make.py`, output only in the scratchpad)

- `probe.py --dry-run`, on random untrained agents (M = 3 and M = 1, hidden 256); it builds no
  environment and reads no file. It checks:
  - gradient shapes (5000, 2) and finite λ*;
  - the fast Q̄ is bit-equal to `agent.ensemble`;
  - the critics stay unchanged by the actor updates, and the actor changes;
  - the nested K = 3,000 actor is bit-equal to a separate 3,000-step run, and different
    repetition seeds give different actors; the batch-index prefix property holds at the real
    sizes (10,000 and 3,000 x 256) for seeds 0, 1, 2;
  - the grid control (`play_grid`) on synthetic curves: per-K eligibility, per-K stop, a λ played
    only for the K still playing, the halving point on the mean curve, the repetition spread;
  - `halving_point` (interior crossing, left/right censoring, first crossing wins), `log2_rho`,
    `verdict` (all four labels; ρ̂ for even n is the geometric mean of the middle pair),
    `overall_label` (all 16 pairs, as in the table above) and `spearman` on synthetic numbers.
- Two scratch models trained from scratch (`lagu_nogate` and `td3lag`, hidden 256, 20k steps,
  seed 7), probed with `--quick`, which uses reduced constants and is labelled as a plumbing test:
  - `--stems` mode, its EXPLORATORY table label, and a snapshot and the final `.pt` of the same run
    giving identical numbers at both K (the probe is deterministic);
  - `--stems` with one missing stem exits with status 2; `--summarise-only` from the cache exits 0;
  - the `--prereg` code path with the sets pointed at the scratch runs: trajectory jobs, a missing
    file, cache reuse, overall NOT EVALUABLE with two NOT EVALUABLE co-primary sets, the
    DESCRIPTIVE label of S3a, the K = 3,000 section and both figures;
  - a job file whose `probe_sha256` differs from the current script is not reused.
- Run cost (section 1): `probe_checkpoint` itself at the full PROTOCOL constants, on the scratch
  models, with the grid cut short and eligibility forced so that every listed λ is played at both
  K: three concurrent processes (the `--workers 3` condition), then one process clamped to the
  machine's lower-tier cores. The per-component times (rollout, evaluation, actor update)
  reproduce each measured job to within 1 s (267, 234 and 268 s). The upper and lower bounds in
  section 1 are these times applied to the 45 declared checkpoints and 280 trajectory snapshots,
  with jobs handed to workers in declared order as `imap_unordered` does.
- Plumbing only, no bearing on the study models: on the untrained scratch models at the `--quick`
  constants (2 episodes, K = 200), the three repetitions' own halving points spanned a factor of
  2.6 to 3.6 (`l2rng` 1.38 to 1.86). This is why a single repetition is not used.
- Earlier, one cache bug was found and fixed: a cached result kept the set label of whichever job
  wrote it.
