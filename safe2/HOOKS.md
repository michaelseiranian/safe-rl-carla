# safe2 agent hooks: the interface between the trainer, the evaluator and an agent

`safe2/train.py` and `safe2/evaluate.py` call an agent only through the methods below
(`safe2.base.TrainerHooks`). `safe2.base.Agent` implements them for every `kind == "base"` config.
The BVS family implements them in `safe2/bvs.py` as a class `BVS`, which
`safe2.base.agent_class("bvs" | "bvs_u" | "bvs_disc")` resolves (`_AGENT_CLASSES` in `base.py`).
SPEC.md section 2.4 fixes what each BVS hook must compute; this file fixes when it is called, what it
receives and what it must return.

## Constructor

```python
BVS(obs_dim: int, act_dim: int, rc: RunConfig, horizon: int, seed: int = 0)
```

- `seed` is the agent seed the trainer derives:
  `SeedSequence(training_seed).spawn(2)[1].generate_state(1)[0]`.
- Before calling it, the trainer has already called `torch.manual_seed(seed)`,
  `np.random.seed(seed)` and `env.action_space.seed(seed)`.
- Networks must be built in SPEC 2.4's order: actor, members, V or Q_b, pib, each followed by its
  target.

## Attributes the trainer reads

| Attribute | Base | BVS family |
|---|---|---|
| `rc` | the `RunConfig` | the `RunConfig` |
| `uses_budget` (property) | `rc.budget_input` | `True` (the default of `TrainerHooks` for `kind != "base"`) |
| `replay_extras` | `{}` | `{"a_exec": ((act_dim,), np.float32), "I": ((), np.uint8)}` |
| `sample_keys` | the six or eight columns `learn` reads | `None` (every column) or a tuple |
| `eval_keep_traj` | `True` (calibration columns) | `False` unless `eval_columns` needs per-step data |
| `env_step` | set by the trainer every step | set by the trainer (may be ignored) |

## Call order of one training step

```
b = clip((25 - C) / 25, -1, 1);  x = cat(o, [b]) if agent.uses_budget else o
if step <= rc.start_steps:  a_pi = a_exec = env.action_space.sample();  I, info = 0, {}
else:                       a_pi, a_exec, I, info = agent.select_action(x, o, b, C, step)     # hook 1
non-finite a_exec or a_pi -> zeros (counted in nonfinite_actions)
o2, r, c, term, trunc = env.step(a_exec)
r_learn = -saute_n if saute_n > 0 and C >= 25 else r
replay.add(o, b, a_pi, a_exec, r_learn, c, o2, b2, d=term, ep, t, I, **info.get("replay", {}))   # hook 2
agent.on_step(step, t, c, I, info)
if term or trunc:
    windows and jc (jc only for episodes ending at step > start_steps)
    agent.end_episode(step, ep_ret, ep_cost)                                                     # hook 4
    episode counters; C = 0; t = 0; seeds.reset(env, "next")
    agent.begin_episode()
if step > rc.start_steps:
    agent.learn(replay.sample(rng_replay, rc.batch, agent.sample_keys), step)                  # hook 3
agent.post_update(step, jc)
if evaluation step:
    policy = agent.make_policy("deployed")
    eps = evaluate.rollout(env_eval, policy, "train_eval", range(rc.eval_episodes), keep_traj=...)
    row = trainer outcomes + agent.log_row() + agent.eval_columns(eps);  agent.reset_interval()
if snapshot step:   {step, config, obs_dim, act_dim, horizon, **agent.snapshot()}                # hook 5
if checkpoint step: {..., "agent": agent.state_dict(), ...}
```

`agent.begin_episode()` is also called once before the first step and once after a resume. A
resume abandons the interrupted episode: the new episode gets a fresh `ep` index.

## Hook 1: action selection

```python
select_action(x, o, b, C, step) -> (a_pi, a_exec, I, info)
```

- It is called only at steps > `rc.start_steps`; the trainer handles the uniform warm-up.
- `x` is `cat(o, [b])`. `o` is the raw observation (V, Q_b and π_b see `o` only). `b` and `C` are the
  budget input and the cost accumulated before this step.
- For the BVS family:
  - at steps ≤ `rc.shield_start`, return `(a_pi, a_pi, 0, {})` with `a_pi = propose(x, explore=True)`;
  - after that, `act(o, b, C, a_pi, m=self.m, rng=self.shield_rng, mode="on")`.
- Return float32 arrays of shape `(act_dim,)` in [-1, 1].
- `I` is 1 iff the executed action differs from `a_pi` by value (SPEC 2.4).
- `info` is passed unchanged to `on_step`. Use the keys of SPEC 2.4: `empty`, `n_allowed`,
  `triggered` and `vmax` (vmax(o, a_exec)). Under `"replay"` put values for any extra replay column
  other than `a_exec` and `I`. `n_allowed` must be present exactly on the steps that screen candidates:
  the evaluator counts those steps (`screen_steps`) and averages it (`n_allowed_mean`).

## Hook 2: extra replay columns

- `replay_extras` declares extra columns as `{name: (shape, numpy dtype)}`. The trainer allocates them
  and always passes `a_exec` and `I` to `replay.add`.
- A column that is not declared is not allocated, so base arms store `a_pi` only.
- Columns stored for every arm (`safe2.train.Replay`):

| Column | Shape and type |
|---|---|
| `o`, `o2` | (obs) float32 |
| `b`, `b2` | (1) float32 |
| `a_pi` | (act) float32 |
| `r` | float32, the learning reward (shaped for Saute) |
| `c`, `d` | float32 (`d` is true termination only) |
| `ep` | int32, the global episode index from 0 |
| `t` | int32, the step within the episode |

- `Replay.order()` gives the rows from the oldest to the newest stored transition (G0b, posthoc).

## Hook 3: every loss of one update

```python
learn(sample: dict, step: int) -> dict
```

- One call per step > `start_steps`. `sample` maps each column in `sample_keys` (every column if
  `None`) to numpy arrays of batch size `rc.batch`, drawn with one `rng_replay.integers` call.
- Every loss of the step uses this one batch, in SPEC 2.4's order:
  1. Q_r;
  2. V or Q_b;
  3. on every `policy_delay`-th update: the actor, π_b, then Polyak.
- Losses go into the agent's own stats for `log_row`. The return value is ignored by the trainer.
- `post_update(step, jc)` follows every step (after `learn`, when there is one). BVS-U updates T there
  (at `step == u_start` and every `u_every` steps after). The base agent runs its λ dual there.

## Hook 4: episode end and per-step bookkeeping

- `on_step(step, t, c, I, info)` runs after every executed step, warm-up included. Use it for the
  cumulative counters of SPEC 2.4 (steps > `shield_start`), the ρ* counters (steps > 35,000), the
  entry and miss tracking (the last 12 costs and I's), the first empty-set step for F, the
  `vH_exec_mean` and `u_p50` interval accumulators and the BVS-U window.
- `end_episode(step, ep_ret, ep_cost)` receives the raw return and the episode cost. It runs the
  margin dual: `margins.append(m)`, then, if `step > margin_start`,
  `m_tilde = max(-d/2, m_tilde + eta_m (ep_cost - d))` (projected anti-windup, protocol A.6). It closes
  the per-episode counters (F, `ep_cum`).
- `begin_episode()` fixes the acting margin `m = clip(m_tilde, -d/2, d)` for the episode and resets the
  per-episode trackers.

## Hook 5: snapshots and evaluation

- `snapshot() -> dict` returns the agent part of `<stem>.snap/<step>.pt`. For the BVS family:
  `actor`, `members`, `V` or `Qb`, `pib` (state dicts), `m_dep`, `T` (BVS-U) and anything
  `from_snapshot` needs. The file is read with `torch.load(weights_only=True)`, so it may hold only
  tensors, state dicts and Python scalars, strings, lists and dicts.
- `from_snapshot(snap) -> agent` (classmethod) rebuilds an agent that acts deterministically. It must
  not consume the caller's torch RNG: build it inside `torch.random.fork_rng()`, as
  `base.Agent.from_snapshot` does.
- `make_policy(mode) -> evaluate.Policy`. For the BVS family this is
  `evaluate.ShieldPolicy(self, mode)`. `evaluate.policy_from_agent` (in-training evaluation) and
  `evaluate.policy_from_snapshot` (held-out evaluation) both end here, so their episodes are
  identical.

**`ShieldPolicy` contract** (the "core" is the BVS agent):

- `core.rc` and `core.m_dep` (the deployed margin; also read in in-training evaluation).
- `core.propose(x, explore=False) -> a_pi`. It must be deterministic and draw nothing.
- ```python
  core.act(o, b, C, a_pi, m, rng, mode, rule) -> (a_exec, I, info)
  ```
  - `mode` is `"on"`, `"off"` or `"strict"`. The evaluation modes map as deployed → on,
    shield_off → off, strict → strict, s_near → on.
  - `rule` is `"SN"` in s_near and `rc.rule` otherwise.
  - `rng` is `default_rng(layout_id)`, created at each episode start.
  - BVS-U reads its own (frozen) T.
  - `info` must hold `empty` (bool) and `vmax` (max_j V_H(o, a_exec)).
- `core.saturation_gap(o, a) -> float` is the max_j (V^j_H − V^j_{H−1})(o, a). It is used only for
  `record_gap` (deployed BVS-family runs except BVS-disc, last held-out snapshot).
- `core.rc.kind == "bvs_disc"` makes the evaluator compute `v_real` as BVS-disc's critic quantity
  (`rc.gamma_b`, `rc.qb_max`).

`evaluate.rollout` derives the per-episode BVS statistics from `info`:

- `interventions`, `empty_steps` and `F`;
- `entries` and `misses` (`shield_stats`, with 12-step lookback);
- `v_pred` and `v_real` over steps with a full H-step future;
- `screen_steps` and `n_allowed_mean` (from `info["n_allowed"]`).

`evaluate.phs_policy(bstar_snapshot, posthoc_fit, m_phs, rule)` is
`ShieldPolicy(safe2.bvs.PHSCore(...), "deployed")`; `evaluate --phs` builds it per B* snapshot with the
fit `torch.load(<root>/phs/<stem>.posthoc.pt, weights_only=True)`, which must hold at least
`bvs.fit_offline`'s keys (`V`, `pib`, `obs_dim`, `act_dim`, `hidden`, `H`) and only weights-only
types (Appendix A.9).

## Logging, checkpoint and console

- `log_row() -> dict` gives the arm's columns of the current row. Keys must come from
  `train.OUTCOME_FIELDS` (others are dropped).
  - For the BVS family these are `qbar`, `vH_exec_mean`, `u_p50`, `q_loss`, `pi_loss`, `v_loss`,
    `pib_loss`, `m_tilde`, `m`, `m_dep`, `T` and the cumulative counters.
  - A value not computed yet is `""`. A diverged value stays NaN, because the finite flags read it.
  - Every column of `train.WHITELIST[kind]` must be present once learning has started, and once a
    whitelisted column has been written it must never be `""` again: the trainer counts such an empty
    cell as non-finite (divergence). BVS-U's `T` may be ±inf only when ρ* is 0 or 1.
    `tests/test_train.py::test_every_whitelisted_column_is_written_once_learning_runs` checks the
    base arms, `tests/test_bvs.py` the BVS kinds.
  - BVS-U also gives `trig007_num`, the cumulative count of steps > 35,000 with u ≥ 0.07 (Lag-U's T;
    protocol §14.3); `""` for the other kinds.
- `eval_columns(eps) -> dict` gives the in-training evaluation columns (`eval_int_steps`,
  `eval_empty_steps`; base arms give the calibration columns).
- `reset_interval()` runs after every row and restarts the interval accumulators.
- `console_dual() -> str` gives `"m=..."` for the console line of seeds < 300. Blinded runs never
  print it.
- `state_dict()` and `load_state_dict()` cover the complete agent state:
  - networks, targets and optimisers;
  - `rng`, `shield_rng` and the `v_gen` state;
  - the margin state and `margins`;
  - the BVS-U window and T;
  - every cumulative counter.

  A resume must continue exactly. `tests/test_train.py::test_resume_reproduces_an_uninterrupted_run_with_a_deterministic_env`
  shows the base check, which a BVS test can copy.

## RNG contract

| Owner | Stream | Used for |
|---|---|---|
| trainer | `env.action_space` (seeded with the training seed) | warm-up actions |
| trainer | `rng_replay = default_rng(SeedSequence(seed).spawn(2)[0])` | replay sampling only |
| trainer | torch global (seeded with the training seed, restored at resume) | network init; the reward critics' target smoothing |
| agent | `self.rng = default_rng(agent_seed)` | exploration noise (`LagU.act`; BVS `propose(explore=True)`) |
| agent (BVS) | `shield_rng`, `v_gen` from `SeedSequence(agent_seed).spawn(2)` | training candidates; V/Q_b uniform target candidates |
| evaluation | `default_rng(layout_id)` per episode | shield candidates at evaluation |

- Evaluation (in-training or held-out) consumes no training stream. The in-training evaluation runs
  on its own environment.
- Base arms consume the trainer's and the agent's streams in `lagu/train.py`'s order, which
  `tests/test_train.py::test_r1_and_b1_reproduce_the_lagu_train_trajectory` checks.

## Files the trainer writes

| File | Content |
|---|---|
| `<stem>.csv` | `HEALTH_FIELDS`, plus `OUTCOME_FIELDS` for unblinded seeds. A column that does not apply is `""` |
| `<stem>.json` | args, config, dims, `blind`, `resumes` and `n_resumes` |
| `<stem>.snap/<step>.pt` | from `snap_from × steps`, at multiples of `snap_every` and the last step; never overwritten (`<step>.r<k>.pt`) |
| `<stem>.ckpt.pt` | `config`, `env`, `seed`, `steps`, `agent`, `replay`, `step`, `episodes`, `ep_id`, `t`, `updates`, `ep_rets`, `ep_costs`, `jc`, `cum`, `nonfinite`, `wall_s`, `rng_replay`, `torch_rng`, `n_resumes`, `written` (whitelisted columns written so far). Atomic and fsynced |
| `<stem>.pt` | `{config, obs_dim, act_dim, horizon, step, agent: state_dict()}`; for blinded seeds `agent` holds only the state dicts of `snapshot()` (`actor`, `members`, `V`/`Qb`, `pib`). Atomic and fsynced |

For blinded seeds (300-899; the dev seeds 900-999 are not blinded) the outcome rows go to
`<root>/sealed/<stage>/<stem>.rows` (`safe2.seal.RecordFile`).
