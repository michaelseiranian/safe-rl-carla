# Lag-U on Safety-Gymnasium

A faithful port of Lag-U (Zhang, Liu, Li, Lin, Li, *IEEE T-ITS* 25(10), 2024,
DOI 10.1109/TITS.2024.3397700) on a TD3-Lagrangian base, with switches that isolate its
two ingredients so the question "does uncertainty-adaptive constraint tightening help?"
can be answered as a single-factor, multi-seed comparison.

The CARLA code in `safe_rl/` is the MSc pilot and is kept for reference only. Its
uncertainty threshold was 3.0 where the paper uses 0.07, so its gate never engaged.

## Install (macOS arm64, Python 3.10 only)

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r lagu/requirements.txt
.venv/bin/python -m pytest lagu/tests -q
```

## Run

```bash
# one run in the foreground
.venv/bin/python -m lagu.train --env SafetyPointGoal1-v0 --arm lagu --seed 0 --steps 1000000 --out results/pilot

# a grid of detached runs that survive the shell (caffeinate keeps the Mac awake; a closed lid still sleeps)
.venv/bin/python -m lagu.launch --out results/pilot --arms td3lag lagu_nogate lagu --seeds 0 1 2 --steps 1000000
lagu/status.sh results/pilot
.venv/bin/python -m lagu.launch --out results/pilot --arms td3lag lagu_nogate lagu --seeds 0 1 2 --steps 1000000 --resume
.venv/bin/python -m lagu.analyze results/pilot
```

Arms: `td3lag` (fixed constraint, action noise), `lagu` (ensemble + bonus + gate),
`lagu_nogate` (ensemble + bonus), `lagu_nobonus` (ensemble + gate). Variants pass extra
`lagu.train` flags, e.g. `--variant epi:dual=episodic,lambda_lr=0.035`. Each run writes
`<arm>[-variant]_<env>_s<seed>.{csv,json,ckpt.pt,pt}`; the CSV holds an evaluation row every
`--eval-every` steps with return, cost, violation rate, lambda, gate engagement rate, ratio
percentiles, and critic calibration (`qc0`, `qbar0` at the first state against the realised
Monte-Carlo `mc_cost0`, `mc_ret0`). `*.ckpt.pt` holds the replay buffer (~250 MB) and can be
deleted once a run is finished and analysed.

## Fidelity

| Quantity | Paper | This port |
|---|---|---|
| Ensemble | M = 3 members, each a TD3 twin pair with its own target (Eq. 12) | same, with independent target-smoothing noise per member; Q_bar and Q_eu from the members' first heads (Eq. 10-11) |
| Exploratory update | maximise Q_bar + sqrt(Q_eu) with prob c(x) = max(0.1, 1 - x/3e5), x = the vehicle's incremental training steps (Eq. 13, 15) | same, x = environment steps |
| Adaptive constraint | delta_ada = delta0 if sqrt(Q_eu)/Q_bar < T else delta0 T / ratio, T = 0.07 (Eq. 16) | same; the ratio uses abs(Q_bar) + 1e-6. `--gate-grad 1` lets the actor gradient flow through delta_ada via sqrt(Q_eu) only (the paper does not say; default detached, in which case delta_ada is a constant in Eq. 17 and the gate acts on the policy only through lambda) |
| Dual update | lambda += 1e-5 E[Q_c - delta] (Eq. 9, 18), in critic units | `--dual statewise` (default), batch mean, clipped at 0. `--dual episodic` is OmniSafe's convention (Adam on lambda (J_c - limit), J_c = mean realised cost of the last `--jc-window` training episodes). `--dual fixed` holds lambda at `--lambda-init` |
| Cost critic | single critic, Eq. 14 | single by default; `--cost-twin 1 --cost-clamp 1` gives a pessimistic max-of-twins critic (results/lr_sweep2: it overestimates and collapses the policy) |
| Networks | two hidden layers of 256, Adam 3e-4 for policy, value and cost, gamma 0.99, Glorot init | same, PyTorch default init (identical for every arm) |
| Action noise | cancelled for Lag-U | cancelled for Lag-U arms; sigma = 0.1 for td3lag |
| delta0 | 0.1 (MetaDrive) | cost_limit / horizon / (1 - gamma) = 2.5 for limit 25 over 1000 steps |
| Simulator | MetaDrive | Safety-Gymnasium 1.0.0 |
| Deployment intervention (rule-based takeover) | yes | not implemented; training study only |

Not in the paper, chosen here: policy delay 2, target smoothing 0.2/0.5, Polyak 0.005, batch 256,
10k random warm-up steps, replay 500k.

`--recipe omnisafe` reproduces OmniSafe 0.5.0's SafetyPointGoal1 TD3Lag settings, read from its
source: actor lr 5e-6, critic lr 1e-3, gradient clip 40, actor loss / (1 + lambda), cost target from
the online actor without smoothing noise, 25k random steps, 1M replay, and an Adam dual at 5e-7
stepped every env step on the last 50 training-episode costs, starting at step 202k. OmniSafe's
published TD3Lag numbers for this task (25.27 return, 28.00 cost) are 3M-step results and still
exceed the limit of 25; at 500k its own curve sits near cost 50, which the default arm reproduces.

## Recipes for the pre-registered study

The pre-registered study (`prereg.md` at the repository root) uses three recipes, passed to
`lagu.launch` as `--variant` strings. Anything not listed is the default described above. All three
use the paper's state-wise dual (Eq. 18: lambda += lambda_lr * batch mean of Q_c(s, pi(s)) - delta,
one step per policy update, clipped at 0) and delta0 = cost_limit / horizon / (1 - gamma), which is
2.5 on the Goal tasks and 5.0 on SafetyPointCircle1 (horizon 500).

| | literal (`lit`) | two-timescale (`tt`) | OmniSafe + state-wise dual (`omsw`) |
|---|---|---|---|
| `--variant` | `lit:` | `tt:actor_lr=5e-6,critic_lr=1e-3,lambda_lr=<LR>` | `omsw:recipe=omnisafe,dual=statewise,lambda_lr=4e-7` |
| Actor lr | 3e-4 (paper) | 5e-6 (OmniSafe) | 5e-6 (OmniSafe) |
| Reward-member and cost-critic lr | 3e-4 (paper) | 1e-3 (OmniSafe) | 1e-3 (OmniSafe) |
| lambda_lr | 1e-5 (paper) | chosen by E0 from {1e-5, 2e-6, 4e-7} on selection seeds 0-2 | 4e-7 |
| Gradient clip | none | none | 40 |
| Actor loss / (1 + lambda) | no | no | yes |
| Cost TD target | target actor + smoothing noise | target actor + smoothing noise | online actor, no noise |
| Random warm-up steps | 10k | 10k | 25k |
| Replay | 500k | 500k | 1M |
| Differs from the paper in | nothing | actor and critic step sizes; lambda_lr unless E0 picks 1e-5 | step sizes, clip, loss scaling, cost target, warm-up, replay |
| Used in | E1b (H2, H5) | E0; E1-E4 (H2-H4) | E0 bridge cell; E1-E4 only if E0 chooses it (outcome (ii), or (iii)/(iv) when no `tt` cell is healthy) |

* `tt` keeps everything of the literal recipe except the step sizes. The actor and critic step
  sizes are OmniSafe 0.5.0's SafetyPointGoal1 TD3Lag values (not tuned here); lambda_lr is chosen
  by E0. The same values are used unchanged on SafetyCarGoal1 and SafetyPointCircle1 (E3). `<LR>`
  is fixed at gate G1 by the E0 rule in `prereg.md` and recorded there before any confirmatory run.
* `omsw` is `--recipe omnisafe` with OmniSafe's episodic Adam dual replaced by the paper's
  state-wise dual. The recipe's `dual_start`, `dual_every` and `jc_window` act only on the episodic
  dual and have no effect here.
* The E1-E4 and E1b launch commands append `ckpt_every=250000,drop_ckpt=1`; E0 appends only
  `ckpt_every=250000`, so that E0-ext can resume it. These flags only control checkpoint files.

## Diagnosis so far (results/lr_sweep, results/lr_sweep2)

* The update is bit-identical to an independent TD3-Lag step; the replay buffer is intact.
* On SafetyPointGoal1 the cost critic's action gradient is about 9x the reward critic's, so any
  lambda above roughly 0.1-0.15 makes the actor minimise cost alone. It learns a near-stationary
  policy (median 0.5 mm per step), return falls to ~0, and the reward critic then diverges negative
  through TD3's min-of-twins bias. Dividing the loss by (1 + lambda) does not change this under Adam.
* The paper's state-wise dual compares a single off-policy cost critic, which reads about half the
  realised cost-to-go, with a critic-unit budget; it settles at "satisfied" while the realised cost
  is about twice the limit. A pessimistic twin critic reverses the bias and lambda runs away.
