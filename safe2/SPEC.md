# safe2 implementation specification

**Status.** Derived from `prereg_method.md` (revised 2026-10-11 after the hostile integrity review,
Appendix H, and amended on 2026-10-14 before v0: protocol section "Pre-v0 amendments"). This file is
part of tag `method-v0` and is hashed with it. Builders implement exactly
what is written here. Where this file and the protocol disagree, the protocol wins, and the
disagreement is logged in protocol §22 before any run that depends on it. Anything this file leaves
open is resolved by asking the study lead and logged in §22. It is never chosen silently.

Every file of §1 and every test of §10 is written and passes at `method-v0`.

---

## 0. Conventions

- Python 3.10 from `/Users/michaelseiranian/Desktop/safe-rl-carla/.venv`. Every entry point runs as
  `python -m safe2.<module>` from the repository root of the worktree and calls
  `torch.set_num_threads(1)` first. The launcher sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.
- Follow the style of `lagu/`:
  - a module docstring that states the purpose and the CLI;
  - `@dataclass` configs with one comment per field;
  - numpy and torch only; no new Python dependency.
  - The OpenSSL 3 command-line tool (`/opt/homebrew/bin/openssl`, or `$SAFE2_OPENSSL`) is the only
    external program.
- Never modify `lagu/agent.py`, `lagu/train.py`, `lagu/evaluate.py`, `lagu/analyze.py`,
  `lagu/certify.py` or `prereg.md`. Import from them freely:
  - from `lagu.agent`: `mlp`, `Actor`, `TwinQ`, `Config` and `LagU`;
  - from `lagu.analyze`: `fisher_exact`.
  `safe2` never calls `lagu.train.evaluate`; it uses its own rollout (§5), so every reset goes
  through `safe2.seeds.reset`.
- **Env resets.** `safe2.seeds.reset(env, block, i, start=None)` is the only call to `env.reset` in
  `safe2/`. A test enforces this (`test_every_reset_in_safe2_goes_through_the_helper`).
- **Steps** are 1-based global environment steps, as in `lagu/train.py`
  (`for step in range(start, steps + 1)`). "From step k" means steps > k unless stated with ≥.
- **Run stem**: `<config>_<env>_s<seed>`, e.g. `B5_SafetyPointGoal1-v0_s0`,
  `BVS-SQ-k0.1_SafetyCarGoal1-v0_s307`.
- **Directories** (all under `results/method/`; `results/` is git-ignored, so files that must be
  pushed are force-added):

| Path | Content |
|---|---|
| `p1/`, `p2/`, `conf/`, `cont3m/` | training runs of each stage: public CSV, json, snapshots, checkpoints, final `.pt`, launcher log |
| `stage0/` | G0a, G0b, S0-1 and S0-4 outputs (not sealed: selection seeds and existing runs only) |
| `phs/` | PHS fits (weights, not sealed). Their quality metrics go under `sealed/phs/` |
| `sealed/` | encrypted outcome records and files (§4) |
| `keys/seal_pub.pem` | public sealing key (committed at v0) |
| `HASHLOG.tsv` | SHA-256 of every sealed record and file (§4). Force-added and pushed daily |
| `PREREG_SHA256.txt`, `FUNC_SHA256.json` | v0 file hashes and per-function hashes (§11) |
| `v05.json`, `v1.json`, `plan.json`, `cuts.json`, `deviations.json` | decision records (§9) |
| `analysis/` | outputs of `safe2.analyze` |
| `.queue.*` | the method queue (§6.3) |

## 1. Files

| File | Purpose | Status |
|---|---|---|
| `seeds.py` | seed and layout registry, `layout()`, `reset()`, `check()` | written |
| `stats.py` | endpoint, imputation, tests, guards, labels, selection, kill (b), decision-function hashes | written |
| `power.py` | Appendix G simulations, `--calibrate` | written |
| `configs.py` | `RunConfig` dataclass and the `CONFIGS` registry (§2) | written |
| `base.py` | `Agent(LagU)` for every baseline and SHL, `PIDLagrangian` (§2.3) | written |
| `bvs.py` | `ViabilityNet`, `DiscCritic`, `BVS` agent (BVS, BVS-U, BVS-disc), shield, margin dual (§2.4) | written |
| `train.py` | shared trainer, replay, CSV schemas, blinding, snapshots, checkpoints, resume (§3) | written |
| `seal.py` | encryption of outcome records and files, hash log, `verify` (§4) | written |
| `evaluate.py` | policies, rollouts (in-training and held-out), modes, PHS, R3 (§5) | written |
| `health.py`, `health.sh` | outcome-free run health (§6.1) | written |
| `launch.py`, `pause.sh` | method queue, power-aware freezing, recovery (§6.2-6.3) | written |
| `posthoc.py` | offline V and π_b fit, entry recall (G0b, PHS) (§7) | written |
| `probe.py` | Stage 0: S0-1, G0a, G0b, S0-4; PHS quality (§8) | written |
| `analyze.py` | selection, gates, kill rules, plan, cuts, `--unblind` (§9) | written |
| `tests/` | §10 | written |

All of these files, `bvs.py`, `posthoc.py` and their tests included, are committed, hashed and
pushed at `method-v0`, together with this file, `README.md` (the run book) and `HOOKS.md` (the agent
hook interface). There is no `method-v0.1` fallback (protocol §0).

## 2. Configurations and agents

### 2.1 `RunConfig` (`safe2/configs.py`)

One frozen-at-v0 dataclass holds every setting of a run. Fields and defaults:

```python
@dataclass
class RunConfig:
    name: str = ""
    kind: str = "base"            # "base" | "bvs" | "bvs_u" | "bvs_disc"
    base: str = "literal"         # "omni" | "literal"
    # learner (mapped onto lagu.agent.Config for kind == "base")
    hidden: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    actor_lr: float = 0.0         # 0 = lr
    critic_lr: float = 0.0        # 0 = lr (reward and cost critics)
    max_grad_norm: float = 0.0    # 0 = no clipping
    pi_norm: bool = False         # actor loss / (1 + lambda)
    cost_target_online: bool = False
    cost_clamp: bool = False      # clamp cost targets to [0, 1 / (1 - cost_gamma)]
    batch: int = 256
    policy_delay: int = 2
    target_noise: float = 0.2
    target_clip: float = 0.5
    action_noise: float = 0.1
    cost_gamma: float = 0.0       # 0 = gamma (SHL: 0.90 or 0.95)
    rate_norm: bool = False       # SHL: penalty scaled by s = (1 - cost_gamma) / (1 - gamma)
    start_steps: int = 10_000     # uniform random warm-up steps
    replay: int = 1_000_000
    # dual
    dual: str = "none"            # "none" (lambda = 0) | "adam" | "pid"
    lambda_lr: float = 0.0        # adam
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    d_delay: int = 10
    ema_p: float = 0.95
    ema_d: float = 0.95
    lambda_max: float = 100.0     # 0 = no cap (R1 only)
    dual_start: int = 200_000     # the dual acts at steps >= dual_start
    dual_every: int = 1
    jc_window: int = 50           # post-warm-up training episodes in the dual's window
    dual_target: float = 25.0     # B9: 20
    # budget input and Saute
    budget_input: bool = False    # actor and critics see b_t = clip((25 - C_t) / 25, -1, 1)
    saute_n: float = 0.0          # > 0: reward replaced by -saute_n once C_t >= 25
    # BVS family
    H: int = 32
    n_perturb: int = 16
    n_uniform: int = 15           # + the backup action = 32 candidates
    n_targ_uniform: int = 4       # + pi_b_targ and pi_targ = 6 target candidates
    sigma_c: float = 0.3
    eps: float = 0.5
    rule: str = "SQ"              # "SQ" | "SN"
    kappa: float = 0.0
    shield_start: int = 25_000    # shield acts at steps > shield_start
    margin_start: int = 50_000    # margin dual updates after episodes ending at steps > margin_start
    m0: float = 5.0
    eta_m: float = 0.1
    m_window: int = 20            # deployed margin = mean acting margin of the last 20 training episodes
    # BVS-U
    M_u: int = 3
    rho_star: float = float("nan")
    T0: float = 0.07
    u_start: int = 35_000
    u_every: int = 1000
    u_window: int = 10_000
    # BVS-disc
    gamma_b: float = 1 - 1 / 32
    qb_max: float = 32.0
    # schedule
    cost_limit: float = 25.0
    eval_every: int = 50_000
    eval_episodes: int = 20
    snap_from: float = 0.8
    snap_every: int = 20_000
    ckpt_every: int = 100_000
```

`make(name, **overrides) -> RunConfig` returns a copy of the registry entry. Only the BVS-U and
BVS-disc entries are built from BVS* at `method-v1` (`from_bvs(bvs_star_name, kind, rho_star)`).

### 2.2 Registry `CONFIGS`

Templates:

- `OMNI` = `base="omni", actor_lr=5e-6, critic_lr=1e-3, max_grad_norm=40.0, pi_norm=True,
  cost_target_online=True, cost_clamp=True, start_steps=25_000`.
- `LIT` = `base="literal", lr=3e-4, start_steps=10_000` (every other learner field at its default:
  no clipping, no pi_norm, cost target from the target actor with smoothing noise, no clamp).

| Name | Built as | Notes |
|---|---|---|
| `R1` | `OMNI`, `cost_clamp=False`, `dual="adam"`, `lambda_lr=5e-7`, `dual_start=202_000`, `lambda_max=0` | the published OmniSafe recipe, identical to `lagu.train --arm td3lag --recipe omnisafe` |
| `B1` | `OMNI`, `dual="adam"`, `lambda_lr=5e-7` | |
| `B2` | `OMNI`, `dual="adam"`, `lambda_lr=2e-6` | |
| `B3` | `OMNI`, `dual="pid"`, `kp=5e-7, ki=1e-8, kd=1e-8` | |
| `B4` | `OMNI`, `dual="pid"`, `kp=0.005, ki=3e-8` | |
| `B5` | `OMNI`, `dual="pid"`, `kp=0.005, ki=1e-7` | |
| `B6` | `OMNI`, `dual="pid"`, `kp=0.02, ki=1e-7` | |
| `B8` | `B5`, `dual_start=50_000` | |
| `B9` | `B5`, `dual_target=20.0` | |
| `B7` | `LIT`, `pi_norm=True, cost_clamp=True`, `dual="pid"`, B5 gains | |
| `B7b` | `B7` with B4 gains | |
| `B7c` | `B7`, `dual_start=50_000` | |
| `SauteA` | `LIT`, `budget_input=True, saute_n=0.05` | |
| `SauteB` | `LIT`, `budget_input=True, saute_n=0.2` | |
| `R2` | `LIT` | |
| `SHL-g0.90-B4`, `SHL-g0.90-B5`, `SHL-g0.95-B4`, `SHL-g0.95-B5` | `B4` or `B5`, `cost_gamma=0.90` or `0.95`, `rate_norm=True` | |
| `BVS-SQ-k0.02`, `BVS-SQ-k0.1`, `BVS-SN-k0.02`, `BVS-SN-k0.1` | `LIT`, `kind="bvs"`, `budget_input=True`, `rule`, `kappa` | |
| `BVSU` | BVS*, `kind="bvs_u"`, `rho_star` from `v1.json` | |
| `BVSD` | BVS*, `kind="bvs_disc"` | |

Unlisted fields keep their defaults. Every PID config uses `d_delay=10` and `ema_p=ema_d=0.95`.
The baseline pool for B* is these 13 configs: B1-B9, B7b, B7c, SauteA and SauteB.

### 2.3 Baseline agent (`safe2/base.py`)

**`Agent(LagU)`.** It is built with
`lagu.agent.Config(M=1, use_bonus=False, use_gate=False, hidden, gamma, tau, lr, actor_lr, critic_lr,
max_grad_norm, pi_norm, cost_target_online, cost_clamp, cost_max=1.0, batch, policy_delay,
target_noise, target_clip, action_noise, delta0=25 / horizon / (1 - gamma), cost_limit=dual_target,
dual=("episodic" if dual == "adam" else "fixed"), lambda_lr, lambda_init=0.0, lambda_max)`.
With `budget_input`, `obs_dim + 1` is passed.

The agent overrides two methods:

- `update` changes only the cost target:
  `c_targ = c + cost_gamma (1 - d) Q_c^targ(o2, a2c)`, clamped to `[0, 1 / (1 - cost_gamma)]` when
  `cost_clamp` is set.
- `_policy_update` changes only the penalty:
  `loss = (-obj + lam_eff * (qc - delta)).mean()`, then `/ (1 + lambda)` when `pi_norm` is set,
  where `lam_eff = float(self.lmbda * s)` is computed as a Python float first and
  `s = (1 - cost_gamma) / (1 - gamma)` if `rate_norm` else `1.0`.

Every other statement, its order and every RNG draw are copied verbatim from `LagU`. The required
consequence: when `cost_gamma == gamma` and `s == 1.0`, the result is bit-identical to `LagU`
(golden tests, §10).

**Dual input.** The window `jc` holds the costs of the last `jc_window` training episodes that
ended at a step > `start_steps`. Warm-up episodes never enter it, as in `lagu/train.py`. At every
step with `step >= dual_start`, a non-empty window and `step % dual_every == 0`, the dual runs, after
the network update of that step:

- `adam`: `agent.update_dual_episodic(mean(jc))`. This is LagU's Adam on `-lambda (J - cost_limit)`,
  clipped to `[0, lambda_max]`.
- `pid`: `agent.lmbda = pid.update(mean(jc))`.
- `none`: λ stays 0 and nothing is called.

**`PIDLagrangian(kp, ki, kd, d_delay, ema_p, ema_d, lambda_max, target)`** follows OmniSafe's
`PIDLagrangian.pid_update` with `sum_norm=True` and `diff_norm=False`. Its state starts at
`I = 0.0`, `delta_p = 0.0`, `cost_d = 0.0` and `cost_ds = deque([0.0], maxlen=d_delay)`.

```python
def update(self, J):
    delta = J - self.target
    self.I = max(0.0, self.I + self.ki * delta)
    self.delta_p = self.ema_p * self.delta_p + (1 - self.ema_p) * delta
    self.cost_d = self.ema_d * self.cost_d + (1 - self.ema_d) * J
    D = max(0.0, self.cost_d - self.cost_ds[0])
    lam = max(0.0, self.kp * self.delta_p + self.I + self.kd * D)
    if self.lambda_max > 0:
        lam = min(lam, self.lambda_max)
    self.cost_ds.append(self.cost_d)
    return lam
```

`state_dict()` and `load_state_dict()` cover `I`, `delta_p`, `cost_d` and `cost_ds`.

**Saute** lives in the trainer, not in the agent (§3.2): the agent sees `obs_dim + 1` inputs and the
replayed reward is the shaped one.

### 2.4 BVS family (`safe2/bvs.py`)

Notation: d = 25, `C_t` is the cost accumulated before step t of the episode, and
`b_t = clip((d - C_t) / d, -1, 1)`. `x = cat(o, b)` is the input of the actor and the reward critics.
V, Q_b and π_b see `o` only.

**Networks**, constructed in this order (the order matters for the golden test):

1. `actor = lagu.agent.Actor(obs+1, act, hidden)`, then `actor_targ`.
2. `members = ModuleList([TwinQ(obs+1, act, hidden)] * M)`, with `M = 1`, or `M_u = 3` for BVS-U,
   then `members_targ`.
3. `V = ViabilityNet(obs, act, H, hidden)` for BVS and BVS-U, or `Qb = DiscCritic(obs, act, hidden)`
   for BVS-disc, then its target.
4. `pib = Actor(obs, act, hidden)`, then `pib_targ`.

Every target is a `deepcopy` with `requires_grad_(False)`. There are four Adam optimisers at lr
3e-4: actor, members (one optimiser over every member), V or Q_b, and pib. There is no gradient
clipping.

- `ViabilityNet`: two `mlp(obs + act, H, hidden)`. `forward(o, a)` returns a tensor of shape
  (2, B, H), with `V[j, :, h-1] = h * sigmoid(f_j(o, a))[..., h-1]` for h = 1..H.
  `vmax(o, a) = V[:, :, H-1].max(0)`.
- `DiscCritic`: two `mlp(obs + act, 1, hidden)`, with linear outputs.
  `vmax(o, a) = max_j Q_b^j(o, a)`.

**RNGs.**

- `self.rng`: a numpy Generator for exploration noise.
- `self.shield_rng`: a numpy Generator for training-time candidates.
- `self.v_gen`: a `torch.Generator` for the uniform target candidates of V or Q_b.

`seed` is the agent seed the trainer derives (as for `LagU`).

- `self.rng = np.random.default_rng(seed)`, as in `LagU`.
- `ss = np.random.SeedSequence(seed).spawn(2)` gives the other two:
  `shield_rng = default_rng(ss[0])` and
  `v_gen = torch.Generator().manual_seed(int(ss[1].generate_state(1)[0]))`.

The torch global RNG is used only for the reward critics' target smoothing. Every state goes into
`state_dict`.

**Proposal.** `a_pi = actor(x)`. In training, `N(0, 0.1²)` is added from `self.rng` and the result
is clipped to `[-1, 1]`. At deployment, `a_pi` is deterministic.

**Budget threshold.** `tau(C, m) = eps if C >= d else max(eps, d - m - C)`.

**Shield step** `act(o, b, C, a_pi, m, rng, mode)`, with `allowed(a) := vmax(o, a) <= tau(C, m)`:

```
mode "off":    return a_pi, I = 0
strict mode:   tau := eps for every C (otherwise as "on")
BVS, BVS-disc (mode "on"):
    if allowed(a_pi): return a_pi, I = 0
    cands = [clip(a_pi + N(0, sigma_c²)) for 16]  (rng.normal, shape (16, act))
          + [U[-1, 1]^act for 15]                   (rng.uniform, shape (15, act))
          + [pib(o)]                                (index 31)
    A = allowed(cands)
    if A.any():
        SQ: idx = argmax over A of min(q1, q2) of members[0] at (x, cand)
        SN: idx = argmin over A of ||cand - a_pi||_2
    else:
        idx = argmin over all cands of vmax                (best-effort exit, empty = 1)
    return cands[idx], I = 1[cands[idx] != a_pi]            (by value, as below)
BVS-U (mode "on"):
    triggered = u(x, a_pi) >= T
    if not triggered: return a_pi, I = 0
    cands = the 32 above + [a_pi]                           (index 32)
    choose idx as above (SQ, SN or best-effort)
    return cands[idx], I = 1[cands[idx] != a_pi]
```

- Ties go to the first index (`numpy.argmax` and `numpy.argmin`).
- **I is defined by value** (protocol A.4): `I = int(not numpy.array_equal(a_exec, a_pi))`, in every
  kind. A best-effort exit or a candidate equal to `a_pi` (possible at the action bounds, where a
  clipped perturbation can duplicate a saturated `a_pi`) gives I = 0.
- The step reports `I`, `n_allowed` (on intervention steps), `empty`, `triggered` and
  `vmax(o, a_exec)`.
- During training, `rng = self.shield_rng`. At deployment and in in-training evaluation,
  `rng = np.random.default_rng(layout_id)`, created once at the start of each episode.

**BVS-U trigger.**

- `u(x, a) = sqrt(var_m Q^m_1(x, a) + 1e-12) / (|mean_m Q^m_1(x, a)| + 1e-6)`. The variance is over
  the 3 members' first heads with `ddof=0` (population variance), exactly as `LagU.ensemble` and
  `LagU._policy_update` compute it.
- `T = T0 = 0.07` at steps `shield_start < step <= u_start`.
- At `step == u_start` and every `u_every` steps after it,
  `T = numpy.quantile(window, 1 - rho_star)` with the default linear method. `window` is a
  `deque(maxlen=u_window)` of `u(x_t, a_pi_t)` at every executed step > `shield_start`.
- Special cases: `rho_star == 0` gives `T = +inf`, and `rho_star == 1` gives `T = -inf`, both only from
  `step == u_start`; before it T is `T0` whatever `rho_star` is.
- At deployment, T is frozen at the snapshot's value.
- Members 2-3 enter only `u`. They never enter the actor loss, any target, or S-Q.

**Update** (one per environment step at steps > `start_steps`; one batch of 256 is shared by every
loss of that step). The batch holds `o, b, a_pi, a_exec, r, c, o2, b2, d, I`, where `d` marks true
termination only.

1. **Reward critics.**
   - `pi2 = actor_targ(x2)`.
   - For each member m, in order: `eps_m = (randn_like(a_pi) * 0.2).clamp(-0.5, 0.5)` and
     `t_m = (r - kappa * I) + gamma (1 - d) * min(members_targ[m](x2, clip(pi2 + eps_m, -1, 1)))`.
   - `loss_q = sum_m [mse(q1_m(x, a_pi), t_m) + mse(q2_m(x, a_pi), t_m)]`, followed by one Adam step.
2. **V (BVS, BVS-U).**
   - Target candidates: `C2 = [pib_targ(o2), pi2, U[-1, 1]^act x 4 from v_gen]`, shape (6, B, act).
   - `Vt = V_targ(o2, C2)` has shape (2, 6, B, H). `mn = Vt.max(0).values.min(0).values` has
     shape (B, H).
   - Targets: `y[:, 0] = c` and `y[:, h] = c + (1 - d) * mn[:, h - 1]` for h = 1..H-1. No head
     bootstraps from itself.
   - `loss_v = sum_j sum_h mean_B (V[j, :, h](o, a_exec) - y[:, h])²`, followed by one Adam step.
   **Q_b (BVS-disc).** `y = clamp(c + gamma_b (1 - d) * min over C2 of max_j Qb_targ^j(o2, ·), 0,
   qb_max)`. `loss = sum_j mean_B (Qb^j(o, a_exec) - y)²`.
3. **Delayed updates.** On every `policy_delay`-th update:
   - Actor: `loss = -members[0].q1(x, actor(x)).mean()`; Adam step on the actor only.
   - π_b: `loss = vmax(o, pib(o)).mean()`; Adam step on pib only. V's gradients from this loss are
     discarded: V's optimiser calls `zero_grad` before its own backward.
   - Polyak update (0.005) of `actor_targ`, `members_targ`, `V_targ` (or `Qb_targ`) and `pib_targ`.

**Margin dual.**

- `m_tilde = m0`. The acting margin is `m = clip(m_tilde, -d/2, d)`. It is fixed for a whole episode,
  being the value in effect when the episode starts.
- At each training episode end, `margins.append(m)` with `margins = deque(maxlen=m_window)`. Then,
  if the episode ended at a step > `margin_start`, `m_tilde = max(-d/2, m_tilde + eta_m * (J_e - d))`
  (projected anti-windup, protocol A.6) and m is recomputed for the next episode. The projection only
  raises `m_tilde`, so `sum (J_e - d) <= (m_tilde_K - m_tilde_0) / eta_m` always, with equality while
  it does not bind.
- **Deployed margin:** `m_dep = mean(margins)` if `margins` is non-empty, else `clip(m0, -d/2, d)`.
  It is the only margin used in in-training evaluation, in snapshots and at deployment.

**Episode-level bookkeeping** (training, steps > `shield_start`), with cumulative counters:

| Counter | Counts |
|---|---|
| `int_cum` | steps with I = 1 |
| `shield_steps_cum` | steps with the shield active |
| `empty_cum` | steps with an empty allowed set |
| `trig_cum` | BVS-U triggers |
| `F_cum` | cost F of each episode: the sum of costs from the first empty-set step, inclusive, to the episode end (0 if none) |
| `ep_cum` | episodes ended after `shield_start` |
| `entries_cum`, `misses_cum` | as defined next |

An **entry** is a step t_e with c > 0, no cost in the 12 steps before it, and t_e ≥ 12 (0-based within
the episode). It is a **miss** if no I = 1 occurs at steps t_e - 12 to t_e - 1.

The ρ* counters `rho_num` (I = 1) and `rho_den` (steps), and `trig_num` (BVS-U), count over steps
> 35,000 only.

## 3. Trainer (`safe2/train.py`)

### 3.1 CLI

```
python -m safe2.train --config NAME --env ENV --seed S --out DIR
                      [--steps 1000000] [--resume | --restart] [--hidden 256] [--drop-ckpt 0|1]
                      [--from-bvs NAME --rho-star X]      # BVSU / BVSD only
                      [--eval-every N --eval-episodes K --snap-every N --ckpt-every N]   # dev only
```

When `--out` lies under `results/method/`, the trainer refuses to start unless:

- `--steps` is 1,000,000 outside `cont3m/`; in `cont3m/` only 3,000,000 with `--resume` (continuations),
  on PointGoal1, seeds 300-304;
- `--hidden` is 256, and none of the dev-only flags is given;
- the seed is in `safe2.seeds.STUDY_TRAINING_SEEDS`.

Other directories accept everything, except that a dev seed (900-999; not blinded, §3.8) never trains
more than `DEV_MAX_STEPS` = 50,000 steps. `--resume` and `--restart` exclude each other.

### 3.2 Step loop

```
seeds.reset(env, "train", seed)               (or "resume", seed, start=step0 + 1)
C = 0; ep_ret = ep_cost = 0; t = 0
for step in range(start, steps + 1):
    b = clip((25 - C) / 25, -1, 1)
    x = cat(o, [b]) if (budget_input or kind != "base") else o
    if step <= start_steps:     a_pi = a_exec = env.action_space.sample(); I = 0
    elif kind == "base":        a_pi = a_exec = agent.act(x, explore=True); I = 0
    else:
        a_pi = agent.propose(x, explore=True)
        a_exec, I, info = agent.act(o, b, C, a_pi, m=agent.m, rng=agent.shield_rng, mode="on")
                          if step > shield_start else (a_pi, 0, {})
    a_exec := zeros if any entry of a_exec is non-finite (count nonfinite_actions); same for a_pi
    o2, r, c, term, trunc, _ = env.step(a_exec)
    r_learn = -saute_n if (saute_n > 0 and C >= 25) else r        # Saute: z_t <= 0 <=> C_t >= 25
                                                                  # (current state: one-step-lag variant, protocol 3.1)
    C2 = C + c;  b2 = clip((25 - C2) / 25, -1, 1)
    replay.add(o, b, a_pi, a_exec, r_learn, c, o2, b2, float(term), ep_id, t, I)
    bookkeeping (§2.4 counters; train return uses the raw r)
    o, C, t = o2, C2, t + 1
    if term or trunc: end-of-episode (windows, jc, margin dual, BVS-U window untouched), episodes += 1,
                      ep_id += 1, t = 0, C = 0, seeds.reset(env, "next")
    if step > start_steps: agent.update(replay.sample(rng_replay, batch))
    dual step (§2.3) for kind == "base"; BVS-U threshold update (§2.4)
    if step % eval_every == 0 or step == steps: in-training evaluation (§3.4), write the row
    if step >= snap_from * steps and (step % snap_every == 0 or step == steps): snapshot (§3.5)
    if step % ckpt_every == 0 or step == steps: checkpoint (§3.5)
save the final .pt (atomic and fsynced, as lagu.train.save_final); drop the checkpoint if --drop-ckpt 1
```

- `rng_replay = np.random.default_rng(SeedSequence(seed).spawn(2)[0])`. The agent seed is
  `SeedSequence(seed).spawn(2)[1].generate_state(1)[0]`, as in `lagu/train.py`.
  `torch.manual_seed(seed)`, `np.random.seed(seed)` and `env.action_space.seed(seed)` are called
  before the agent is built.
- Base agents act through `LagU.act` (exploration noise from `agent.rng`). For base arms the loop
  consumes every RNG (action-space sampling, `agent.rng`, `rng_replay`, the torch global RNG) in
  exactly the order of `lagu/train.py`. An R1 run therefore reproduces the training trajectory of
  `lagu.train --arm td3lag --recipe omnisafe` with the same seed, and B1 reproduces it up to step
  200k (the G0b replay diff relies on this). In-training evaluation uses a separate environment and
  consumes no training RNG.
- The trainer never raises on non-finite values. NaN weights, losses or observations do not stop
  the run, and a non-finite value never triggers a resume (§3.6).

### 3.3 Replay

`Replay(obs_dim, act_dim, size)` stores these columns as float32 unless noted:

- `o` (obs) and `b` (1);
- `a_pi` (act), the action column for base arms;
- `a_exec` (act; BVS family only, otherwise not allocated);
- `r` (learning reward: shaped for Saute, raw otherwise), `c`, `o2`, `b2`, `d`;
- `ep` (int32 global episode index from 0, warm-up included);
- `t` (int32 step within the episode);
- `I` (uint8; BVS family only).

`state()` and `load()` follow `lagu.train.Replay`.

### 3.4 In-training evaluation

- At each evaluation step, the deployed policy of the live agent (§5.1, built with
  `evaluate.policy_from_agent`) plays `eval_episodes` deterministic episodes on
  `seeds.reset(env_eval, "train_eval", k)`, k = 0..19. It uses the same `evaluate.rollout` code as
  held-out evaluation.
- BVS-family runs evaluate with the shield on, `m = m_dep`, their own rule and
  `rng = default_rng(layout_id)`. BVS-U uses the current T.
- Base arms also compute the calibration columns of `lagu.train.evaluate`, with the cost
  discounted at `cost_gamma`, so SHL's `mc_cost_traj` uses γ_c.

### 3.5 Snapshots, checkpoints and the final file

- **Snapshot** `DIR/<stem>.snap/<step>.pt` (weights only, `torch.save` of a dict):
  `step`, `config` (asdict), `obs_dim`, `act_dim`, `horizon`, `actor`, `members`, `cost` (base),
  `V` or `Qb`, `pib`, `m_dep`, `T`, `lambda`.
  - They are taken at every multiple of `snap_every` with `step >= 0.8 * steps`: 800k, 820k, ...,
    1M for a 1M run, and 2.4M to 3M for continuations.
  - Snapshots are never overwritten. A resumed run that reaches an existing snapshot step writes
    `<step>.r<k>.pt`, and the evaluator uses the newest resume's file.
- **Checkpoint** `DIR/<stem>.ckpt.pt`, written atomically (`.tmp` then `os.replace`), holds the
  complete state:
  - agent `state_dict`, including optimisers and every agent RNG;
  - the replay;
  - the step, the episode and `ep_id` counters, the 50-episode windows and `jc`;
  - the PID state, the margin state, `margins`, the BVS-U window and T;
  - every cumulative counter, `rng_replay`, the torch global RNG state and `n_resumes`.
- **Final file** `DIR/<stem>.pt` (atomic and fsynced): `config`, `obs_dim`, `act_dim`, `horizon`, `step`
  and `agent`, the agent's complete `state_dict` — for a blinded seed (§3.8) only the network weights
  of its snapshot (`agent.snapshot()`'s state dicts: `actor`, `members`, `cost` or `V`/`Qb` and `pib`),
  so no outcome scalar (the last training-cost mean, λ, the PID or margin state, counters) is in clear.
  The checkpoint also stores `written` (§3.6).
- **`--drop-ckpt 1`** deletes the checkpoint after the final `.pt` exists. `plan.json` sets it per
  run, and keeps the checkpoint for:
  - P1 B1 seeds 0-2 (G0a, G0b);
  - every B* confirmatory run (PHS) and every SHL* P1 run (S0-4);
  - PointGoal1 seeds 300-302 of B*, BVS* and SHL* (3M continuations).

### 3.6 Resume, crashes and divergence

- `--resume` loads the checkpoint and refuses on a config difference, as `lagu.train` does, and on a
  checkpoint of another run (its `env` or `seed` differ) or of a longer run.
- A `--resume` whose checkpoint is already at `--steps` (a finished run that kept its checkpoint) is a
  no-op: it writes only a missing final `.pt` (a crash between the last checkpoint and the `.pt`) and
  touches no CSV, json or sealed record.
- It truncates the public CSV and the sealed record file to steps ≤ the checkpoint step. Sealed
  records carry their step in clear, so no decryption is needed.
- It calls `seeds.reset(env, "resume", seed, start=ckpt_step + 1)`, sets `resumed = 1`, increments
  `n_resumes` and records the resume in the run's json. `n_resumes` is one more than the largest of the
  checkpoint's count, the json's count and the json's list of resumes, so a run that crashes again
  before its next checkpoint still counts every resume.
- **Fresh starts.** A run that has started before is never started from scratch silently: with a
  checkpoint the trainer refuses without `--resume`; without one (it crashed before its first
  checkpoint) it starts again only with `--restart`, which keeps the json's history, appends
  `{"from_step": 0, "restart": true, ...}` to its resumes, sets `resumed = 1` and counts as a resume.
  Snapshots stay monotone: a rewritten step gets `<step>.r<k>.pt` with k above every k already
  written for that step.
- **Continuations.** `--resume` from the final checkpoint of a shorter run (`ckpt["steps"] < --steps`,
  `ckpt["step"] == ckpt["steps"]`; the 1M checkpoint copied into `cont3m/`) is a continuation, not a
  resume: it is recorded in the json's `continuations` (`from_step`, the source's `previous_n_resumes`
  and `previous_resumes`), `resumed` stays 0 and the continuation's `n_resumes` counts only its own
  resumes from 0 (a restart from the same source checkpoint is one of them). Under `results/method/`
  a 3M `--resume` without a checkpoint is refused.
- **Divergence** means that some row has a whitelisted column (§3.7) that is non-finite.
  - A whitelisted cell counts as finite while it is empty only until the column has been written in
    some row of the run (`written`, kept in the checkpoint); an empty cell after that counts as
    non-finite. BVS-U's `T` may be ±∞ (never NaN) when ρ* is 0 or 1 (`train.inf_allowed`).
  - The run continues to its end, and its finite flags show it.
  - The launcher never resumes a run whose public CSV shows a finite flag of 0, and never resumes a
    run whose json records 3 resumes (§6.3).
  - A diverged run counts as collapsed (protocol §11). If it finished, its held-out endpoint is kept
    and is imputed (`stats.impute_task`: R := 0, C := worst, in either arm) only when that endpoint
    is non-finite; a run that stopped after a divergence is imputed in the same way.
  - A run that stopped without any divergence and is still unfinished after 3 resumes is
    "unfinished". It gets the worst case in the candidate arm and the median in B*'s arm.

### 3.7 CSV schemas

**Public CSV** `DIR/<stem>.csv`, one row per in-training evaluation, `HEALTH_FIELDS`:
`step, episodes, updates, sps, wall_s, finite_out, finite_net, finite_loss, finite_dual,
nonfinite_actions, resumed, n_resumes`.

- For seeds < 300 the public CSV also holds every `OUTCOME_FIELDS` column, after the health
  columns.
- For seeds ≥ 300 the outcome row goes only to the sealed record file (§4).
- A column that does not apply to an arm is written as the empty string, never as NaN.

**`OUTCOME_FIELDS`**:

| Group | Columns |
|---|---|
| outcomes | `train_return, train_cost` (last 50 training episodes), `eval_return, eval_cost, eval_violation` (20 in-training evaluation episodes), `cum_cost, cum_cost_200k` (training cost over steps > 200k), `cum_viol_eps` |
| network | `qbar` (batch mean min-twin Q_r(x, π(x)) at the last policy update), `qc` (batch mean Q_c(o, π(o)); base), `vH_exec_mean` (mean vmax(o, a_exec) over training steps in the interval; BVS family), `u_p50` (median u over the interval; BVS-U) |
| losses | `q_loss, pi_loss, c_loss` (base), `v_loss, pib_loss` (BVS family) |
| dual | `lambda, jc, pid_i, pid_p, pid_d` (base); `m_tilde, m, m_dep` (BVS family); `T` (BVS-U) |
| calibration (base) | `qc0, mc_cost0, qc_traj, mc_cost_traj, qbar_traj, mc_ret_traj` |
| BVS counters (cumulative) | `int_cum, shield_steps_cum, empty_cum, trig_cum, F_cum, ep_cum, entries_cum, misses_cum, rho_num, rho_den, trig_num, trig007_num` (BVS-U: steps > 35,000 with u ≥ 0.07, Lag-U's T; protocol §14.3), plus `eval_int_steps, eval_empty_steps` (in-training evaluation) |

**Finite whitelist** (the run diverges if any of these is non-finite in any row):

| Kind | Columns |
|---|---|
| base | `train_return, train_cost, eval_return, eval_cost, q_loss, pi_loss, qbar, c_loss, qc`, plus `lambda` when `dual != "none"` |
| bvs, bvs_disc | `train_return, train_cost, eval_return, eval_cost, q_loss, pi_loss, qbar, v_loss, pib_loss, vH_exec_mean, m_tilde` |
| bvs_u | the bvs list plus `T` and `u_p50` |

The flags group the whitelist as follows:

- `finite_out`: the four return and cost columns;
- `finite_net`: `qbar`, `qc`, `vH_exec_mean` and `u_p50`;
- `finite_loss`: every loss column;
- `finite_dual`: `lambda`, `m_tilde` and `T`.

### 3.8 Blinding

- Blinded seeds are the seeds ≥ 300 except the dev seeds 900-999, which never train more than 50,000
  steps (§3.1) and are never analysed.
- For seeds < 300, the console line is the `lagu.train` style
  `[stem] step= ret= cost= lam|m= sps=`.
- For seeds ≥ 300 it is `[stem] step=<step> sps=<sps> finite=<0|1>`. Nothing else is ever printed,
  warnings and tracebacks included: a traceback must not contain values, so the trainer catches
  exceptions at top level and prints only `type(e).__name__` and the step.
- The run json (args, config, dims) is public.

## 4. Sealing (`safe2/seal.py`)

- **Key.** Before v0 the study lead creates the key pair:
  `openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:4096 -aes-256-cbc -out <private path>`
  (passphrase-protected, kept off the repository), then `openssl pkey -in <private path> -pubout
  -out results/method/keys/seal_pub.pem`. The public key is force-added and hashed at v0. No `safe2`
  code ever reads the private key except `analyze --unblind --key`. The passphrase is typed by the
  lead into OpenSSL's own prompt.
- **`encrypt(plaintext: bytes, pub_path) -> bytes`:**
  - `k = os.urandom(32)` and `iv = os.urandom(16)`;
  - `wrapped = openssl pkeyutl -encrypt -pubin -inkey seal_pub.pem -pkeyopt rsa_padding_mode:oaep
    -pkeyopt rsa_oaep_md:sha256 -pkeyopt rsa_mgf1_md:sha256` with stdin `k + iv`;
  - `ct = openssl enc -aes-256-ctr -K hex(k) -iv hex(iv)` with the plaintext on stdin;
  - output `base64(wrapped) + b"." + base64(ct)`.
  - The key never touches disk, and `subprocess.run(..., input=..., capture_output=True,
    check=True)` is used for every call.
- **Fixed sizes.** AES-CTR keeps the plaintext length, and the files' sizes and the hash log's
  `n_bytes` are public, so every plaintext is padded with trailing spaces to a fixed size before
  `encrypt` (`seal.padded`): `PAD["record"]` = 8192 bytes per outcome row, `PAD["file"]` = 1 MiB per
  whole file. A longer plaintext raises (with no value in the message). Every sealed row and every
  sealed file therefore has the same size whatever its outcomes; the largest possible held-out output
  fits (a test checks it).
- **Record files** (outcome rows of seeds ≥ 300):
  `results/method/sealed/<stage>/<stem>.rows`. There is one line per row,
  `<step>\t<encrypt(padded(json.dumps(row)))>\n`, appended and fsynced.
- **Whole files** (held-out results, PHS quality metrics, R3): `sealed/<kind>/<name>.json.enc`
  holds `encrypt(padded(json bytes))`, written atomically.
- **Hash log** `results/method/HASHLOG.tsv`, appended under `fcntl.flock` and fsynced:
  `utc_iso \t relative_path \t record|file \t index \t sha256(encrypted bytes) \t n_bytes`.
  - The lead force-adds, commits and pushes it at least once a day; each push goes in §21.
  - `python -m safe2.seal verify` recomputes every sealed record's and file's hash against the
    log's last entry for it, and exits 1 on any mismatch or missing entry.
- **What is sealed:**
  - every outcome row of a seed ≥ 300;
  - every evaluation on the held-out block 70000-70199, whatever the seed (so R3 too);
  - PHS quality metrics.
- **What is not encrypted** (recorded in the protocol as a remaining procedural limit):
  - checkpoints, replays, snapshots and PHS weights, because resume, evaluation and fitting need
    them in-process; the final `.pt` (for blinded seeds network weights only, §3.5);
  - public CSVs, jsons and logs. Their timing columns (`sps`, `wall_s`) weakly track a BVS run's
    screening rate (the shield scores candidates only when it screens); this channel is disclosed.
  No tool prints an outcome computed from them before unblinding.
- **`decrypt(blob, key_path) -> bytes`** reverses `encrypt` with `openssl pkeyutl -decrypt -inkey
  key_path ...` and `openssl enc -d ...`. Only `analyze --unblind` calls it.

## 5. Evaluator (`safe2/evaluate.py`)

### 5.1 Policies

`Policy` objects expose `begin(layout_id)`, `act(o) -> a` and `observe(c)` (which tracks C and b),
plus per-step info. Every policy is deterministic; there is no exploration noise.

| Policy | Built from | Acts |
|---|---|---|
| `BasePolicy` | actor (+ `budget_input`) | `actor(x)` |
| `ShieldPolicy` | actor, members, V or Q_b, pib, m, T, rule, kind, mode | §2.4 shield with `rng = default_rng(layout_id)` from `begin` |
| `PHSPolicy` | B*'s snapshot actor and members[0] (proposer and S-Q critic), the run's PHS V and π_b, `m_PHS(task)`, BVS*'s rule | §2.4 shield with B*'s actor as proposer |
| `LaguPolicy` (R3) | a `lagu` snapshot via `lagu.evaluate.load_agent` | `agent.act(o, explore=False)` |

- `policy_from_agent(agent, mode)` and `policy_from_snapshot(path, mode)` must produce identical
  objects.
- **Modes:**
  - `deployed`: the shield on, with the config's rule;
  - `shield_off`;
  - `strict`: τ = ε;
  - `s_near`: the rule forced to SN; run only when BVS* uses SQ.

### 5.2 `rollout(env, policy, block, indices) -> list[dict]`

For each index:

1. `o, _ = seeds.reset(env, block, i)` and `policy.begin(layout_id)`.
2. Step until `term or trunc`.
3. Record the per-episode `return`, `cost` and `layout`. The BVS family also records `interventions`,
   `empty_steps`, `F`, `entries`, `misses`, `v_pred`, `v_real`, `screen_steps` and `n_allowed_mean`:
   - `v_pred` is the mean over steps of `vmax(o_t, a_t)`;
   - `v_real` is the mean over the same steps of the realised cost of steps t..t+H-1 (BVS-disc: the
     γ_b-discounted cost from t to the episode end, clamped at `qb_max`, its critic's quantity);
   - both use steps with a full H-step future only;
   - `screen_steps` counts the steps on which the shield scored its candidates (`info["n_allowed"]`
     present), and `n_allowed_mean` is the mean allowed-set size over them (NaN without one;
     protocol §14.4).
4. With `record_gap=True`, also record the saturation gaps `max_j (V^j_H - V^j_{H-1})(o_t, a_t)` at
   visited near-hazard states. A state is near-hazard when the task-state hazard distance
   `min_h dist_xy(h) - size` is below 0.6. This is used for logging only, and never for BVS-disc
   (Q_b has no heads).

### 5.3 CLI and outputs

```
python -m safe2.evaluate DIR [DIR ...] [--mode deployed|shield_off|strict|s_near] [--end-step N]
                         [--workers 12] [--force] [--root ROOT]
python -m safe2.evaluate --phs DIR [...]        # needs phs/<stem>.posthoc.pt and v1.json
python -m safe2.evaluate --r3 LAGU_DIR --seeds 100-109 [--r3-prefix lagu-lit]
```

- **Snapshots.** Snapshot i (i = 0..9) is the file at exactly `seeds.heldout_targets(N)[i]`, with
  N = `--end-step` or the run's steps. There is no nearest-step fallback; a missing file is an
  error. It plays `rollout(env, policy, "heldout", [20 i + k for k in range(20)])`. Under
  `results/method/`, N must be 1M, and 3M in `cont3m/` (otherwise `ERROR EndStepRefused`).
- **Secondary modes** skip the runs they do not apply to (every shield mode of a baseline; `s_near` of
  an S-N config) and print `<stem>: skipped (no <mode> mode)`.
- **R3.** Only the stems `<--r3-prefix>_*` (default `lagu-lit`) of LAGU_DIR are evaluated. The lagu
  snapshots at 800k, 850k, 900k, 950k and 1M are j = 0..4. Snapshot j plays held-out indices
  `20 (2j) + k` and `20 (2j + 1) + k`, for k = 0..19 (40 episodes per snapshot, 200 in all).
- **PHS** (`--phs`, protocol A.9). For each B* run of DIR (`v1.json` `b_star`; v1.json lies in the
  sealing root), each of its 10 snapshots at N = 1M is played behind `evaluate.phs_policy(snapshot,
  fit, m_PHS(task), BVS*'s rule)` = `ShieldPolicy(bvs.PHSCore(...), "deployed")`, with
  `fit = torch.load(<root>/phs/<stem>.posthoc.pt, weights_only=True)` (bvs.fit_offline's dict, saved by
  posthoc §7). Mode name `phs`; the output adds `phs: {fit_sha256, m, rule}`, and the side file also
  holds the fit's SHA-256. `--phs` refuses without v1.json or when BVS* is none (killed).
- **Output** (plaintext before sealing): `{stem, mode, end_step, snapshots: [{i, step, sha256,
  episodes: [...]}], pooled: {R, C}, gaps (deployed BVS-family runs except BVS-disc, last snapshot
  only)}`.
  - R is the mean of the 200 episode returns, and C the mean of the 200 episode costs.
  - Every output is on the held-out block, so it is always sealed to
    `sealed/heldout/<stem>.<mode>.json.enc`, or `<stem>.<mode>.n<N>.json.enc` when N ≠ 1M (a 3M
    continuation never takes its 1M run's name; `analyze.heldout_rel`). Plaintexts are padded to a
    fixed size before sealing (§4). The console prints only `<stem>: done`, `<stem>: skipped (...)`
    or `<stem>: ERROR <type or code>`.
- **Re-runs.** A run with an existing sealed output is skipped unless `--force`. The snapshot
  SHA-256 values are kept in clear in a side file, `sealed/heldout/<stem>.<mode>[.n<N>].sha`, so a
  changed snapshot is detected without decryption (`ERROR SnapshotsChanged`). The side file is written
  before the sealed file (a forced re-evaluation first removes the old pair), so no crash leaves a
  sealed file next to a stale side file; a sealed file without its side file is `ERROR ShaMissing`.
- Workers are processes (`spawn`), with one torch thread each.

## 6. Health, launcher and pause

### 6.1 `python -m safe2.health [DIR ...]`

- Prints one line per run:
  - the stem;
  - the state: `queued | running | frozen | finished | crashed | diverged | unfinished(3 resumes)`;
  - the last step, sps and `n_resumes`;
  - the finite flags, as all-ones or the first row with a zero.
- Reads only the public CSVs, run jsons, `launch.json`, the queue files and `ps`.
- It never opens a path under `sealed/` (a test patches `open`, `io.open` and `os.open` to fail on
  such paths).

### 6.2 `safe2/pause.sh pause | resume | status | power auto|always | cap N`

A copy of `lagu/pause.sh` with these changes:

- the queue files are `results/method/.queue.*`;
- `pids()` matches `-m safe2.train` with `--out` under `results/method/`;
- `resume` runs `python -m safe2.launch --recover`.

### 6.3 `python -m safe2.launch`

An adapted copy of `lagu/launch.py`; `lagu/launch.py` itself is not changed.

- `TRAINER = ^\S*python[0-9.]* -m safe2[.]train `. `managed()` is true for `safe2.train` writing
  under `results/method/`. Freezing, unfreezing, recovery and de-duplication act only on these runs.
- **Cap.** It counts every live `-m safe2.train` process (frozen included) whose `--out` lies under
  any `results/` directory, in either checkout, plus the Lag-U study's slots: per checkout,
  `max(running, min(cap_L, alive + queued))`, where alive and running are its live `lagu.train` runs
  (all, or not stopped), and cap_L and queued are read (only read) from that checkout's
  `results/.queue.cap` (12 when missing) and `results/.queue.jsonl`. The method queue therefore never
  takes a slot the Lag-U scheduler is about to refill; `lagu/pause.sh cap N` leaves 12 − N slots to
  the method study, and frozen Lag-U runs keep their slots unless cap_L is 0.
- **Queue files.** `results/method/.queue.{jsonl,cap,pid,log,paused,power}`. `scheduler_alive`
  checks for `safe2.launch` with `--scheduler`. The CLI always queues; `--max-procs N` only writes
  the cap file (when it does not exist yet, or always with `--set-cap`).
- **Markers** are `--config X `, `--env `, `--seed ` and `--out `. The stem is built as in §0.
  `spec_from_cmd` recognises `safe2.train`. The absolute `--out` is recorded in the spec's argv (a
  relative plan `--out` is resolved from the repository root), so every live run has one marker set.
- **Refusals** (`blocked()`, checked when queueing and again before each start): a cut run
  (`cuts.json`), a command `safe2.train.refusal` rejects (e.g. 3M outside `cont3m/`), a diverged
  run, a run with 3 resumes, a 3M continuation without its checkpoint. A finished run is final (also
  with `--resume`); a started run is never queued as a fresh start (only `--recover` restarts it, with
  `--restart`).
- **`--from-plan results/method/plan.json --stage STAGE`** queues every planned run of that stage
  with the exact argv recorded in the plan, minus the runs listed in `cuts.json` (queued runs that are
  now cut are unqueued; live ones are listed). A continuation's 1M checkpoint (`resume_from`) is
  copied into `cont3m/` only when the 1M run is finished and not diverged and its 1M held-out output
  exists; otherwise the run is skipped with `source diverged|unfinished|checkpoint missing`.
- **`--recover`** re-queues interrupted runs from their checkpoints (`--resume`), or restarts those
  that crashed before their first checkpoint (`--restart`). It never re-queues a run that `blocked()`
  refuses: it prints `diverged`, `unfinished`, `cut` or `refused: ...`.
- **`--retire STAGE/STEM ...`** (bug reruns, protocol §16) moves a run that is not alive out of the
  study: its run files, its sealed record file and its held-out outputs (3M ones for `cont3m/`, the
  others plus PHS fit and quality otherwise) go to `results/method/retired/<UTC time>/` (outside
  `sealed/`; their hashes stay in HASHLOG.tsv), and it leaves `launch.json` and the queue.

## 7. Offline fits (`safe2/posthoc.py`)

```
python -m safe2.posthoc fit --ckpt RUN.ckpt.pt --episodes A-B --out OUT.posthoc.pt [--updates 100000] [--seed S]
python -m safe2.posthoc recall --fit OUT.posthoc.pt --ckpt RUN.ckpt.pt --episodes A-B [--sealed]
```

- **Data.** These are the replay rows whose `ep` lies in `[A-1, B-1]` (episodes are 1-based and
  inclusive on the CLI), read in insertion order (`ptr`, `n`, as `lagu.train.Replay.load`).
  - G0b fits on episodes 201-900 and holds out 901-1000.
  - PHS fits on episodes 1-900 and holds out 901-1000.
  - Every requested episode must be stored whole (its `t` column is 0, 1, ..., len-1); a missing or
    partly stored episode makes `fit` and `recall` refuse.
  - The executed action is the replay's `a_exec` column when the run has one (BVS family), else
    `a_pi`.
- **Fit.**
  - `S` defaults to the run seed. `torch.manual_seed(S)`; batch indices come from `default_rng(S)`;
    `v_gen` is seeded with S.
  - V (twin, H = 32) and π_b have the run's hidden width (`rc.hidden`, 256 in the study) and are
    trained exactly as in §2.4, steps 2-3 (`bvs.fit_offline`): 100k updates, batch 256, Adam 3e-4 for
    both (whatever the run's own critic rates), π_b every 2nd update, Polyak 0.005.
  - The proposer is the checkpoint's own actor, frozen and deterministic. It takes the input
    `cat(o2, b2)` when the run has `budget_input`, and `o2` otherwise. It stands where `pi2` is in
    the target candidate set.
  - Output (`torch.save`, loadable with `weights_only=True`): `bvs.fit_offline`'s dict (`V`, `pib`,
    `obs_dim`, `act_dim`, `hidden`, `H`, `updates`, `batch`, `seed`, `losses`) plus `ckpt`, `ckpt_sha256`, `episodes`, `rows`, `env`,
    `config` and `run_seed`.
- **Python interface** (`probe.POSTHOC_API`): `fit(ckpt, episodes, out, updates, seed)`,
  `fit_meta(path) -> {ckpt_sha256, episodes, updates, seed}`, `load_fit(path)` (a `ViabilityNet`) and
  `recall(fit, ckpt, episodes)`. G0b and `probe phs-quality` refuse any fit, fresh or reused, whose
  `fit_meta` does not match the run (checkpoint SHA-256, episodes, updates, seed).
- **Recall** (G0b(i)), on the held-out episodes:
  - an alarm at t is `vmax(o_t, a_t) > 0.5`, with `a_t` the replay's executed action;
  - an entry (cost > 0, no cost in the 12 steps before it, t_e ≥ 12) is caught if an alarm occurs at
    some t in `[t_e - 12, t_e - 7]`;
  - a false-alarm step is an alarm with no cost in steps t..t+31 of the same episode, truncated at
    its end.
  - Output: `entries, caught, recall, false_alarm_steps, episodes, fa_per_episode, nonfinite_pred`.
    A non-finite `vmax` is never an alarm; any non-finite prediction makes `recall` and
    `fa_per_episode` NaN (a gate on them fails), and `probe.pool_recall` propagates it.
  - With `--sealed` (seeds ≥ 300) the output is sealed to `<root>/sealed/phs/<stem>.recall.json.enc`
    (root from the fit's location) and the console prints only `<stem>: done`. An unsealed recall of a
    blinded seed is refused. No plan record uses `recall --sealed`: `probe phs-quality` seals the
    recall with the ranking.

## 8. Stage 0 (`safe2/probe.py`)

**Oracle helpers.** These mirror `simulation_forward` of safety-gymnasium 1.0.0 without consuming
`task.random_generator`:

- `save(task)` and `load(task, s)` handle `qpos, qvel, act, qacc_warmstart` and `time`. `load` runs
  `mj_forward` and then puts the saved `qacc_warmstart` back, because MuJoCo 2.3's `mj_forward`
  overwrites it; a saved, explored and restored episode then stays bit-identical to the plain one.
- `step_physics(task, a)` runs `agent.apply_action(a, None)`, then 10 × (mocaps `move()`,
  `mj_step`), then `mj_forward`.
- `step_cost(task)` is `float(task.calculate_cost()["cost_sum"] > 0)` when
  `task.cost_conf.constrain_indicator`, else the raw sum.
- `hazard_distance(task)` is `min_h agent.dist_xy(h) - hazards.size`.
- Point actions: `EVASIVE` = (f, w) ∈ ((-1,0), (-1,1), (-1,-1), (0,1), (0,-1), (0,0), (0.5,1), (0.5,-1)).
  `GRID` = f ∈ (1, .5, 0, -.5, -1) × w ∈ (-1, -.5, 0, .5, 1) (25 actions, in this order).
  `VALUE_CANDS = GRID[::3] + [GRID[12]]`.
- `viable(task, s0, a, H)` applies a, fails if `hazard_distance < 0`, then tries each evasive action
  held H steps from the resulting state. It is true if one of them never enters a hazard.
- A test checks that `step_physics`, then `step_cost`, then `task.obs()` (in this order: the cost
  runs a second `mj_forward`, as `env.step` does) reproduce `env.step`'s next observation and cost
  bit for bit at every step up to the first `goal_met`, over at most 300
  steps on dev layout 90500. One environment steps and a second one on the same layout replays the
  same actions.

**CLI.**

```
python -m safe2.probe g0a [--runs results/method/p1] [--env ENV] [--out results/method/stage0] [--workers 3] [--force]
python -m safe2.probe g0b [--runs results/method/p1] [--pareto MAIN/results/pareto] [--env ENV]
                          [--out results/method/stage0] [--workers 3] [--reuse-fits] [--force]
python -m safe2.probe s01 [--root MAIN] [--out results/method/stage0/s01] [--workers 3]
python -m safe2.probe s04 [--runs CKPT ...] [--nep 100] [--out results/method/stage0] [--force]
python -m safe2.probe phs-quality RUN [--root ROOT] [--force]     # RUN = <dir>/<stem> of a B* run
```

`--force` replaces an existing output (for g0a and g0b, a rewrite is a deviation and is recorded in
`deviations.json`); without it a subcommand refuses to overwrite. `--reuse-fits` uses the fits already
in `<out>/g0b/`, which are still checked against their runs (§7).

**Readings fixed at v0** (protocol §5): G0a's value rollout stops at the first hazard entry and scores
the distance there; `VALUE_CANDS` holds (0, 0) twice and the duplicate never wins; the filter
condition uses the cost before the step; G0a and G0b(ii) use the run's final `.pt` actor with
`clip(actor(o), -1, 1)` in float32. G0b's replay identity compares float32 bit patterns in insertion
order ("unavailable" if rows 0..199,999 are no longer stored; the lagu checkpoint is loaded
`weights_only` with numpy scalars allow-listed). Recall pools caught / entries and false-alarm steps /
episodes; collection stops at the 400th labelled state; the label's early stops are exact; a Spearman
with a constant side or fewer than 2 points is NaN and fails.

**`probe g0a`.**

- **Input:** `results/method/p1/B1_SafetyPointGoal1-v0_s{0,1,2}.pt`, the final actors.
- For each actor, 30 episodes on `seeds.reset(env, "g0a", k)`, k = 0..29. At each step:
  1. `a = actor(o)`, deterministic.
  2. If `C >= 12` and `hazard_distance < 0.6`, save s0. If `viable(s0, a, 25)` fails:
     - score every viable candidate in `VALUE_CANDS` by a 15-step privileged rollout from s0: hold
       the candidate 5 steps, then follow the actor for 10 steps, with observations from
       `task.obs()`;
     - the score is `-||goal - agent||_xy - 10 · [a hazard was entered]`;
     - take the best (ties go to the first), or `EVASIVE[0]` if none is viable;
     - count an intervention, then `load(s0)`.
  3. `env.step(a)`.
- **Output** `stage0/g0a.json`: per-episode return, cost and interventions per actor, and the means
  over the 90 episodes.
- **Decision** (at v0.5, by `analyze bstar`): PROCEED iff mean return ≥ max(18, R_B*(P1) + 6) and
  mean cost ≤ 25. R_B*(P1) is the mean over B*'s six seeds of the selection-window return.

**`probe g0b`.**

- **Replay identity (disclosure).** Compare rows 0..199,999 (`o, a, r, c, o2, d`) of each B1 replay
  (seeds 0-2) with `/Users/michaelseiranian/Desktop/safe-rl-carla/results/pareto/
  td3lag-omni_SafetyPointGoal1-v0_s{0,1,2}.ckpt.pt`, read-only.
  - Report "identical" or the first differing row.
  - Replay rows are in insertion order: use `ptr` and `n` as in `lagu.train.Replay.load`.
- **Fits:** `posthoc fit` on episodes 201-900, seed = run seed, for each of the 3 replays.
- **(i):** `posthoc recall` on episodes 901-1000, pooled over the 3 runs. Pass iff recall ≥ 0.80 and
  false-alarm steps per held-out episode ≤ 30.
- **(ii) ranking.** For each seed, the B1 actor plays deterministic episodes on
  `seeds.reset(env, "g0b", k)`, k = 0, 1, ...
  - **Collecting a state.** At step t, collect the state if all of these hold: `hazard_distance <
    0.6`, fewer than 20 states taken in this episode, and t ≥ (the last collected t in this
    episode) + 10.
  - **Label.** At a collected state, with a_t the actor's action:
    `label = step_cost after step_physics(a_t) + min over EVASIVE of the sum of step_cost over 31
    steps holding it` (state restore). Then restore s0 and `env.step(a_t)`.
  - **Prediction** = `vmax(o_t, a_t)` of that seed's fit.
  - Each seed stops at 400 states or after 100 episodes.
  - Pooled Spearman: average ranks for ties, numpy only. Pass iff at least 1000 states are pooled
    and ρ ≥ 0.7. With fewer than 1000 states the gate fails.
- **Output:** `stage0/g0b.json`: the identity result, the recall, false-alarm and Spearman results,
  and the state counts.

**`probe s04 --runs R1 R2 ... --nep 100`.** This is `simplest/h3_grad.py` committed as a function.

- Gradients of λ s Q_c and of Q_r with respect to a at a = π(s).
- States are the first 500 steps of the last 100 complete episodes of each run's replay.
- Labels: quiet means no cost in t+1..t+30; imminent means cost in t+1..t+10.
- `share = sum |∇_a Q_c| over quiet states / sum over all states`.
- Runs:
  - the design runs: `pareto/td3lag-omni s0, s1, s2`, `pareto/td3lag-omni-fix03 s1`,
    `pareto/td3lag-fix005 s0`, and `e0_dual/lagu_nogate-{omsw-l4e-7, tt-l2e-6, tt-l4e-7} s0`, all
    from the main checkout, read-only;
  - after SHL P1, `p1/SHL*_SafetyPointGoal1-v0_s{0,1,2}.ckpt.pt`.
- **Output:** `stage0/s04.json`. Kill rule 8.4 uses it.

**`probe s01` (exploratory).** Adapts `results/paper/uncertainty_anticipation/make.py` of the main
checkout: its statistics, cluster bootstrap and definitions are copied into `probe.py` at v0.

Extensions:

- **Predictors:** the max hazard-lidar bin, with the slice computed from
  `task.obs_info.obs_space_dict` in flatten order, and Q_c(s, π(s)).
- **Leads:** k ∈ {5, 10, 25, 50}.
- **Snapshots:** the saved snapshots nearest 0.3N, 0.6N and N.
- **Runs:** the arms of `make.py`, plus the CarGoal1 and PointCircle1 arms in `results/gate_x`.
- **Layouts:** `seeds.reset(env, "s01", k)`, k = 0..9.
- **Statistics:** event-aligned mean curves from -50 to +10 steps around entries; AUROC and AUPRC
  with base rates; for `lagu-tt`, the subset of runs with held-out return ≥ 5 as well.
- **Output:** `stage0/s01/`. It gates nothing.

**`probe phs-quality RUN [--root ROOT] [--force]`** (RUN = `<dir>/<stem>` of a B* run; protocol
A.9).
- Inputs: `RUN.ckpt.pt` (replay), `RUN.pt` (B*'s 1M actor) and the fit `<root>/phs/<stem>.posthoc.pt`
  (root: the run's `results/method` unless `--root`). The fit must match the run (`fit_meta`:
  checkpoint SHA-256, episodes 1-900, 100k updates, seed = run seed), otherwise it refuses.
- G0b(i) on episodes 901-1000 and G0b(ii) with B*'s actor on the g0b layouts, with G0b's settings
  unchanged (at most 400 states or 100 episodes), reusing `g0b_rank_seed`, `pool_recall`, `spearman`
  and `vmax_fn`. On PointCircle1 (no hazards) (ii) collects no state and its Spearman is NaN, reported
  as "not applicable".
- Output: only `<root>/sealed/phs/<stem>.quality.json.enc` (recall and ranking); the console prints
  only `<stem>: done`. An existing output is replaced only with `--force`.

## 9. Analysis (`safe2/analyze.py`)

**Common selection data.**

- A seed's selection R and C are the means of `eval_return` and `eval_cost` over its public rows at
  steps 800k, 850k, 900k, 950k and 1M.
- A seed is **collapsed** if R / R_ref(task) < 0.2 or any row has a finite flag of 0.
- Non-finite selection values are imputed with `stats.impute_selection`.
- **R_ref(task)** = max over configs c of the mean over seeds 0-2 of R. The configs are R2 and every
  baseline config run on that task; candidates never count. It is computed and frozen by the
  subcommand that first needs it:
  - PointGoal1 by `p1-baselines` (recorded at v0.5);
  - CarGoal1 and PointCircle1 by `candidates` (recorded at v1, from P2's R2 and B* runs).

Subcommands. Each writes `analysis/<name>.txt` and a json, and refuses to run if its inputs are
incomplete.

1. **`p1-baselines`.** The 13 baseline configs and R2 on PointGoal1, seeds 0-2.
   - Computes R_ref(PointGoal1).
   - `rank(eligible_only=False)` of the 13 gives the top 3 (`p1/top3.json`).
2. **`bstar`.** Runs after seeds 3-5 of the top 3 and after G0a and G0b.
   - B* = first of `rank(top3 on seeds 0-5, eligible_only=False)`.
   - Sauté* = first of `rank([SauteA, SauteB] on 0-2, eligible_only=False)`.
   - L* = first of `rank([B7, B7b, B7c, SauteA, SauteB] on 0-2, eligible_only=False)`.
   - R_B*(P1) = B*'s mean R over seeds 0-5.
   - The G0a and G0b decisions.
   - Writes `v05.json` (tag `method-v0.5`).
3. **`candidates`.**
   - BVS* = first of `rank(4 BVS configs on 0-2)`, eligible only; if none is eligible, BVS is killed
     under (a). SHL* is chosen the same way.
   - **Kill rules:**
     - (a) collapses in at least 2 of 3 seeds of the selected config;
     - (b) `stats.kill_b(cand R, C on seeds 0-2, B* R, C on seeds 0-5)`;
     - (c) on P2;
     - 8.2 for BVS, pooled over seeds 0-2 and the window 500k-1M:
       - intervention rate = Δ`int_cum` / Δ`shield_steps_cum` between the 500k and 1M rows;
       - mean F = Δ`F_cum` / Δ`ep_cum`;
       - evaluation violation share = Σ(`eval_violation` × 20) / Σ 20 over the rows at 500k, 550k,
         ..., 1M.
     - 8.4 for SHL, from `stage0/s04.json`.
   - **ρ\*** = Σ `rho_num` / Σ `rho_den` at the 1M rows of BVS*'s seeds 0-2.
   - **m_PHS(task)** = the mean over BVS*'s selection-seed runs of `m_dep` in the 1M snapshot.
   - **Stopped or non-finite BVS* seeds** (protocol §6, M7). 8.2, ρ* and m_PHS pool only the seeds
     whose inputs exist and are finite (8.2 and ρ*: the 500k and 1M rows with finite counters, and for
     the evaluation share every row from 500k to 1M; m_PHS: a 1M snapshot with a finite `m_dep`). The
     others are listed in `p1/candidates.json` / `v1.json`. Fewer than 2 qualifying seeds kills BVS*
     (P1: rule 8.2; P2: rule (c) for that task). A non-finite pooled value is never written.
   - R_ref for CarGoal1 and PointCircle1.
   - A killed BVS* has no fallback config.
   - Writes `v1.json`.
4. **`plan`.**
   - Writes `plan.json`: every confirmatory run, with stage, config, env, seed, steps, out,
     drop_ckpt, cut group and the exact argv. It also lists every held-out evaluation with its mode,
     every PHS fit and quality check, and the R3 evaluations.
   - Writes `cuts.json = []`.
   - Records their SHA-256 in `PREREG_SHA256.txt` (tag `method-v1`).
5. **`cut ITEM`.**
   - Appends to `cuts.json` the date, the cut-list item (§19 of the protocol), the stems and the
     reason "throughput".
   - Refuses after `UNBLINDED.txt` exists. The lead commits and pushes the file before any planned
     run is skipped.
6. **`--unblind --key PRIVATE.pem`.** In order; any failure refuses and prints why.
   1. **Tags.** `method-v0`, `method-v0.5` and `method-v1` exist locally and on `origin`
      (`git ls-remote --tags origin`), pointing to the same objects.
   2. **Code.** The current SHA-256 of every file hashed at v0 must match `PREREG_SHA256.txt`
      (`V0_FILES` must all be there): every non-test file of `safe2/` (the `.py` modules including
      `launch.py`, `health.py`, `posthoc.py` and `probe.py`, `pause.sh`, `health.sh`, `SPEC.md`,
      `README.md`, `HOOKS.md` and any other file kept there), `keys/seal_pub.pem`, and
      `prereg_method.md` without sections 21-22 (`_frozen_part`, which requires exactly those two
      headings between "## 21." and "## Appendix A."). Any file that differs must be listed in
      `deviations.json` with the §22 entry.
   3. **Decision functions.** The per-function hashes of `stats.DECISION_FUNCTIONS` (from
      `stats.source_hashes()`) must match `FUNC_SHA256.json`. If one differs and is listed as a
      deviation, or `safe2/stats.py` itself is a listed deviated file, the verdict is computed with the
      method-v0 module, loaded from `git show method-v0:safe2/stats.py`. The current module's verdict
      is reported next to it as the deviation's sensitivity.
   4. **Records.** `v1.json` (R_ref, ρ*, m_PHS) and `plan.json` must match their v1 hashes, with one
      entry each in `PREREG_SHA256.txt` unless listed as deviations. `cuts.json`, `HASHLOG.tsv` and
      `deviations.json` (when present) must be committed and on `origin` (`git status --porcelain` is
      clean for each and its last commit is reachable from `origin`).
      **`deviations.json`** is a list of `{"file": path | "function": name, "entry": text}`; every
      `entry` must occur in the protocol's §22 text, otherwise (or for an entry without file/function
      or entry, such as the stub that `probe g0a|g0b --force` appends with the old file's SHA-256, until
      the lead completes it) unblinding refuses.
   5. **Completeness.** Every planned run, minus the cuts, is finished, diverged,
      `unfinished(3 resumes)`, or a continuation in state `not-run (source diverged|unfinished)`.
      Every finished run has its planned held-out outputs. `safe2.seal verify` passes.
      **`--deadline`** (protocol §10, M8): at the analysis deadline steps 1-4 and the hash-log check
      still apply, but every planned, non-cut run that is queued, running or missing, and every
      finished run without its held-out output, is imputed as unfinished (worst case in the candidate
      arm, B*'s median in B*'s arm), listed by arm, and the output is labelled "computed at the
      analysis deadline".
   6. Write `UNBLINDED.txt` (UTC time, `git rev-parse HEAD`, and the SHA-256 of `analyze.py`,
      `stats.py` and `deviations.json`). A re-run never overwrites the first output: it writes
      `analysis/unblind.<n>`, and its primary is labelled "re-run after unblinding (not
      confirmatory)" when the code or deviation hashes differ from the first line. Decrypt, impute
      (`stats.impute_task`, with diverged flags) and compute:
      - the primary: `stats.primary(BVS* vs B*, three tasks, diverged flags)`, with
        `n_perm=100_000`, `seed=0` and `alpha=stats.ALPHA_TEST`;
      - the secondary and exploratory tables of protocol §14. These are the per-task and pooled
        Δs, Δr and Δc; violation share; CVaR10; the seed-level penalised score (per-run hinge);
        cumulative training cost at 1M and over 200k-1M; collapses (Fisher); the IQM and Welch
        sensitivities; dropping resumed runs; the primary without CarGoal1 and without
        PointCircle1; the controls (a contrast with Sauté* or L* uses the runs that played that
        config: its own arm, the L* arm when L* is Sauté*, or B*'s runs when it is B*); BVS* - L*
        (exploratory); R1, Sauté* and R3; the mechanism metrics (including the training-average margin
        bound, the window bound on each 50k interval from 50k, and the saturation gap labelled
        "approximately saturated" when mean ≤ 0.05 and p95 ≤ 0.25, never "invariant"); SHL*; and the
        3M continuations;
      - every table in one pass, into `analysis/`.

## 10. Required tests (all in `safe2/tests/`, passing at `method-v0`)

`test_seeds.py` and `test_stats.py` hold the NaN regression, imputation, hinge permutation, tolerant
cost guard, per-function hashes and the `reset` helper tests. The numbered tests are below. Golden
tests use 300 synthetic updates on a fixed batch stream with
`hidden=32`, as in `lagu/tests/golden_make.py`.

**Baselines (`test_base.py`)**

1. `R1` reproduces `lagu.agent.LagU` built with the omnisafe recipe. Both run 300 updates and a
   scripted `update_dual_episodic` every update. Actor, member and cost-critic hashes and λ must be
   equal.
2. `B1` reproduces the same with `cost_clamp=True`.
3. SHL with `cost_gamma = gamma` and `rate_norm=False` is bit-identical to B4 and to B5.
4. SHL's cost target equals `clamp(c + γ_c (1-d) Q_c^targ, 0, 1/(1-γ_c))` on a fixed batch.
5. A finite-difference actor gradient equals the baseline's with Q_c scaled by s.
6. Level identity: on a constant-cost-rate chain, `(1-γ_c) Q_c` is within 5% of the rate for γ_c in
   {0.9, 0.95, 0.99}.
7. `PIDLagrangian` matches a hand-computed 30-step sequence: integral, EMA P, delayed D, anti-windup
   at 0 and the cap.
8. The dual is off before `dual_start` (200k; 50k for B8 and B7c). Its target is 20 for B9.
9. Warm-up episodes never enter `jc`.
10. Resume restores the PID state.
11. Config diff:
    - B7 differs from `LIT` only in `pi_norm`, `cost_clamp` and the dual;
    - B7b and B7c differ from B7 only in their gains or their dual start;
    - B8 and B9 differ from B5 only in `dual_start` or `dual_target`;
    - R1 differs from B1 only in `cost_clamp`, `dual_start` and `lambda_max`.
12. Saute: reward replacement at `C >= 25`, the b input and the episode reset.

**BVS (`test_bvs.py`)**

13. With the shield off, κ = 0 and `I = 0`, BVS reproduces `base.Agent(LIT, budget_input=True,
    dual="none", cost_target_online=True)` bit for bit. Actor and member hashes must be equal after
    300 updates.
14. Viability targets on a hand-built batch: `y_1 == c`; `y_h` reads target head h-1 (perturbing
    head h-1 changes `y_h` and nothing else); outputs lie in [0, h].
15. On a deterministic 1-D corridor MDP with one hazard cell and a stop action, the trained `V_h`
    equals the exact minimum cost within h steps, within 0.1, for every h and action. It runs the BVS
    update code at lr 1e-3, Polyak 0.05 and H = 6 so that it converges within a test's time.
16. Shield logic with mocked V and Q_r:
    - the executed action is allowed whenever A is non-empty;
    - it equals `a_pi` when `a_pi` is allowed;
    - it is the argmin of vmax when A is empty;
    - `tau = max(eps, d - m - C)`, and `tau = eps` once C ≥ d (including with a negative m);
    - SN returns the nearest allowed candidate;
    - candidate order and tie-breaking follow §2.4.
17. BVS-U:
    - ρ* = 0 never triggers and ρ* = 1 always does;
    - on a triggered step with `a_pi` allowed, SN returns `a_pi` with I = 0;
    - after an actor update, the `.grad` of members 2-3 is None (isolation);
    - S-Q and targets use member 1 only;
    - `u` equals LagU's ratio on the same members;
    - the T schedule (0.07 until 35k, then the quantile every 1000 steps).
18. BVS-disc: the test is `max_j Q_b <= tau`, γ_b = 1 - 1/32, and targets are clamped to [0, 32].
19. The episode state (b, C) resets at every episode start.
20. Relabel: the replay stores `a_pi` and `a_exec`, and Q_r's target uses `r - kappa * I`.
21. Margin dual:
    - `m_tilde` is projected onto [-12.5, ∞) and the acting margin is clipped to [-12.5, 25];
    - there are no updates for episodes ending at steps ≤ 50k;
    - a synthetic stream on which the projection never binds reproduces
      `(1/K) Σ J = d + (m_tilde_K - m_tilde_0) / (eta K)`;
    - `m_dep` is the mean of the last 20 acting margins;
    - with the projection, the audit's stream (475 episodes at J = 0, then J = 50) reaches the acting
      margin 25 after 15 over-budget episodes, and on adversarial streams satisfying assumption A
      (`test_margin_dual_projection_keeps_the_run_bound_and_bounds_every_window`)
      `Σ (J - d) <= (m_tilde_K - m_tilde_0) / eta`, the mean is ≤ 26.24 (K = 950) and every window of
      W ∈ {100, 200, 475} episodes averages ≤ 25 + 1350/W.
22. Step accounting: every executed step (warm-up, shielded, best-effort) increments the step
    counter and the replay.
23. Gradient isolation: after an actor step, V's `.grad` is None, and π_b's loss does not step V.
24. Update schedule: V and Q_r update every step from 10,001; the actor, π_b and every Polyak update
    happen every 2nd update; one shared batch per step.

**Trainer, evaluator, blinding (`test_train.py`, `test_evaluate.py`, `test_blinding.py`)**

25. Snapshot round trip. The held-out evaluator (`policy_from_snapshot`) on the train_eval block
    reproduces the in-training evaluation episode for episode, bit for bit, for a base arm, Saute
    and BVS (shield on, `m_dep`, layout-seeded sampler). This uses a 30k-step dev run with `--hidden
    64 --eval-every 10000 --snap-every 10000 --snap-from 0`.
26. Snapshot cadence (`heldout_targets`) and the R3 mapping (j → 2j, 2j+1).
27. Resume:
    - the CSV and sealed records are truncated to the checkpoint step;
    - the config check refuses a change, and a checkpoint of another run;
    - `n_resumes` is recorded, and counts every resume when a run crashes again before its next
      checkpoint;
    - a no-op `--resume` of a finished run changes no file;
    - a started run is never restarted without `--restart`, which keeps the history; snapshot names
      stay monotone;
    - a continuation counts only its own resumes;
    - the resume layout is `seed * 1000 + start`.
28. Non-finite values: a run whose actor weights are set to NaN at step k continues to the end,
    writes `finite_* = 0` and does not raise.
29. Seed-300 dev run (in a temporary directory with a temporary key):
    - stdout, the log and the public CSV contain no outcome column or value: the public CSV header
      equals `HEALTH_FIELDS`, and the console lines match the §3.8 pattern;
    - the sealed record file decrypts (with the temporary private key) to the outcome rows.
30. `seal` round trip; `verify` detects a modified record and a missing hash-log entry; sealed sizes
    do not depend on the outcomes.
31. Every evaluation on the held-out block is sealed whatever the seed, R3 (lagu-lit only) and PHS
    included; 3M outputs never take the 1M name; the side file is written first.
32. `health` output holds only the allowed fields and never opens `sealed/`.
33. `--unblind` refuses when:
    - a planned run or held-out file is missing;
    - a hashed file or decision function differs without a deviation entry;
    - `plan.json` or `v1.json` differ;
    - `cuts.json` is uncommitted;
    - a tag is missing.
    With a listed decision-function change, it decides with the v0 source. Also: `deviations.json`
    uncommitted or with an entry not in §22, `HASHLOG.tsv` unpushed, a rewritten record; a re-run
    writes `unblind.<n>` and is labelled a re-run.
33b. `candidates` with a stopped or non-finite BVS* seed pools over the qualifying seeds, lists the
    others, and kills BVS* when fewer than 2 qualify (protocol §6, M7); a non-finite m_PHS is never
    written.
33c. `--unblind --deadline` imputes queued, running and missing runs as unfinished, lists them and
    labels the output (protocol §10, M8).
34. Launcher:
    - `managed()` accepts `safe2.train` under `results/method/` and rejects others;
    - the cap counts `safe2.train` processes and reserves the Lag-U study's slots;
    - `--recover` rebuilds `safe2.train` commands with `--resume` (or `--restart`), and skips
      diverged, 3-resume, cut and refused runs;
    - `--from-plan` minus cuts, also for queued runs; continuations only from a finished,
      non-diverged 1M run;
    - a finished run is final; 3M runs only in `cont3m/`;
    - `spawn()`'s process settings; `--retire`;
    - `pause.sh` freezes only `safe2.train` study runs; `pause`, `resume`, `cap`, `power`, `status`.
35. The trainer refuses non-default flags and unregistered seeds under `results/method/`, 3M runs
    outside `cont3m/`, and dev seeds above 50,000 steps anywhere.

**Stage 0 and fits (`test_probe.py`, `test_posthoc.py`)**

36. `step_physics` and `step_cost` reproduce `env.step` up to the first `goal_met` (at most 300
    steps) on a dev layout.
37. `viable` on a constructed state.
38. G0b(ii) sampling: at most 20 states per episode, at least 10 steps apart; fewer than 1000 pooled
    states fails the gate.
39. Spearman with ties against a hand-computed case.
40. Recall and false alarms on a synthetic replay with known entries.
41. `posthoc fit` uses only the requested episodes.
42. The G0a decision uses max(18, R_B* + 6).

## 11. Tags and hash records

- **`method-v0`** (pushed before any study run and any Stage-0 read):
  - `results/method/PREREG_SHA256.txt` lists the SHA-256 of every file in `safe2/` outside
    `safe2/tests/` (the `.py` modules, `pause.sh`, `health.sh`, `SPEC.md`, `README.md`, `HOOKS.md`
    and any other file kept there), `prereg_method.md` and `results/method/keys/seal_pub.pem`, plus
    `openssl version` and `git rev-parse HEAD`;
  - `results/method/FUNC_SHA256.json` = `stats.source_hashes()`.
  Both are force-added.
- **`method-v0.5`:** `v05.json` and its hash appended to `PREREG_SHA256.txt`.
- **`method-v1`:** `v1.json` and `plan.json`, and their hashes appended.
- Every tag is pushed when it is made, and its push time goes in protocol §21.
- `HASHLOG.tsv` and `cuts.json` are pushed at least daily and before unblinding.
