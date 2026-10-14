"""Every run setting of the method study (prereg_method.md section 3; safe2/SPEC.md section 2).

One dataclass, RunConfig, holds every setting of a run; CONFIGS maps each registered config name to
its RunConfig. The trainer builds a run with make(name, **overrides) (overrides are the dev-only CLI
flags and the width); BVS-U and BVS-disc are built from the selected BVS config at method-v1 with
from_bvs(). The registry is frozen at tag method-v0: a change is a protocol deviation (section 22).

    python -m safe2.configs            # print every config and its non-default fields
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class RunConfig:
    name: str = ""                # registry name; the run stem is <name>_<env>_s<seed>
    kind: str = "base"            # "base" | "bvs" | "bvs_u" | "bvs_disc"
    base: str = "literal"         # "omni" | "literal" (documentation: the learner fields below decide)
    # learner (mapped onto lagu.agent.Config for kind == "base")
    hidden: int = 256             # width of the two hidden layers of every network
    gamma: float = 0.99           # reward discount
    tau: float = 0.005            # Polyak factor of every target network
    lr: float = 3e-4              # every network unless actor_lr / critic_lr is set
    actor_lr: float = 0.0         # 0 = lr
    critic_lr: float = 0.0        # 0 = lr (reward and cost critics)
    max_grad_norm: float = 0.0    # 0 = no clipping
    pi_norm: bool = False         # actor loss / (1 + lambda)
    cost_target_online: bool = False  # cost TD target from the online actor, no smoothing noise
    cost_clamp: bool = False      # clamp cost targets to [0, 1 / (1 - cost_gamma)]
    batch: int = 256              # one batch per environment step after the warm-up
    policy_delay: int = 2         # actor (and Polyak) update every policy_delay-th update
    target_noise: float = 0.2     # target-policy smoothing sd
    target_clip: float = 0.5      # target-policy smoothing clip
    action_noise: float = 0.1     # Gaussian exploration noise sd
    cost_gamma: float = 0.0       # 0 = gamma (SHL: 0.90 or 0.95)
    rate_norm: bool = False       # SHL: penalty scaled by s = (1 - cost_gamma) / (1 - gamma)
    start_steps: int = 10_000     # uniform random warm-up steps
    replay: int = 1_000_000       # replay capacity (min(replay, steps) is allocated)
    # dual
    dual: str = "none"            # "none" (lambda = 0) | "adam" | "pid"
    lambda_lr: float = 0.0        # adam: step size of the episodic Adam dual
    kp: float = 0.0               # pid: proportional gain (on the EMA of J - target)
    ki: float = 0.0               # pid: integral gain
    kd: float = 0.0               # pid: derivative gain (on the delayed rise of the EMA of J)
    d_delay: int = 10             # pid: derivative delay, in dual updates
    ema_p: float = 0.95           # pid: EMA factor of the proportional term
    ema_d: float = 0.95           # pid: EMA factor of the cost used by the derivative term
    lambda_max: float = 100.0     # 0 = no cap (R1 only)
    dual_start: int = 200_000     # the dual acts at steps >= dual_start
    dual_every: int = 1           # the dual acts at steps with step % dual_every == 0
    jc_window: int = 50           # post-warm-up training episodes in the dual's window
    dual_target: float = 25.0     # the dual's episodic cost target (B9: 20)
    # budget input and Saute
    budget_input: bool = False    # actor and critics see b_t = clip((25 - C_t) / 25, -1, 1)
    saute_n: float = 0.0          # > 0: reward replaced by -saute_n once C_t >= 25
    # BVS family (safe2/bvs.py; SPEC section 2.4)
    H: int = 32                   # viability heads (steps)
    n_perturb: int = 16           # perturbed candidates around a_pi
    n_uniform: int = 15           # uniform candidates (+ the backup action = 32 candidates)
    n_targ_uniform: int = 4       # uniform target candidates (+ pi_b_targ and pi_targ = 6)
    sigma_c: float = 0.3          # sd of the perturbed candidates
    eps: float = 0.5              # threshold floor tau >= eps
    rule: str = "SQ"              # "SQ" | "SN"
    kappa: float = 0.0            # intervention penalty in the reward critic's target
    shield_start: int = 25_000    # shield acts at steps > shield_start
    margin_start: int = 50_000    # margin dual updates after episodes ending at steps > margin_start
    m0: float = 5.0               # initial margin
    eta_m: float = 0.1            # margin dual step size
    m_window: int = 20            # deployed margin = mean acting margin of the last 20 training episodes
    # BVS-U
    M_u: int = 3                  # reward-critic members (member 1 is BVS's Q_r)
    rho_star: float = float("nan")  # target trigger rate, from v1.json
    T0: float = 0.07              # threshold before u_start
    u_start: int = 35_000         # first quantile update of T
    u_every: int = 1000           # T update period
    u_window: int = 10_000        # executed steps in T's quantile window
    # BVS-disc
    gamma_b: float = 1 - 1 / 32   # discount of Q_b (matched horizon 32)
    qb_max: float = 32.0          # Q_b targets clamped to [0, qb_max]
    # schedule
    cost_limit: float = 25.0      # episodic cost limit d
    eval_every: int = 50_000      # in-training evaluation period
    eval_episodes: int = 20       # in-training evaluation episodes (train_eval layouts 10000 + k)
    snap_from: float = 0.8        # snapshots at steps >= snap_from * steps ...
    snap_every: int = 20_000      # ... that are multiples of snap_every (and the last step)
    ckpt_every: int = 100_000     # checkpoint period (and the last step)


KINDS = ("base", "bvs", "bvs_u", "bvs_disc")
DUALS = ("none", "adam", "pid")
DEV_FIELDS = ("eval_every", "eval_episodes", "snap_every", "snap_from", "ckpt_every")   # dev-only CLI flags

# --- templates (SPEC 2.2) ----------------------------------------------------------------------
OMNI = dict(base="omni", actor_lr=5e-6, critic_lr=1e-3, max_grad_norm=40.0, pi_norm=True,
            cost_target_online=True, cost_clamp=True, start_steps=25_000)
LIT = dict(base="literal", lr=3e-4, start_steps=10_000)
B4_GAINS = dict(kp=0.005, ki=3e-8, kd=0.0)
B5_GAINS = dict(kp=0.005, ki=1e-7, kd=0.0)


def _build() -> dict:
    c: dict = {}

    def add(name: str, *parts: dict, **kw) -> None:
        merged: dict = {}
        for p in parts:
            merged.update(p)
        merged.update(kw)
        c[name] = RunConfig(name=name, **merged)

    add("R1", OMNI, cost_clamp=False, dual="adam", lambda_lr=5e-7, dual_start=202_000, lambda_max=0.0)
    add("B1", OMNI, dual="adam", lambda_lr=5e-7)
    add("B2", OMNI, dual="adam", lambda_lr=2e-6)
    add("B3", OMNI, dual="pid", kp=5e-7, ki=1e-8, kd=1e-8)
    add("B4", OMNI, dual="pid", **B4_GAINS)
    add("B5", OMNI, dual="pid", **B5_GAINS)
    add("B6", OMNI, dual="pid", kp=0.02, ki=1e-7, kd=0.0)
    add("B8", OMNI, dual="pid", **B5_GAINS, dual_start=50_000)
    add("B9", OMNI, dual="pid", **B5_GAINS, dual_target=20.0)
    add("B7", LIT, pi_norm=True, cost_clamp=True, dual="pid", **B5_GAINS)
    add("B7b", LIT, pi_norm=True, cost_clamp=True, dual="pid", **B4_GAINS)
    add("B7c", LIT, pi_norm=True, cost_clamp=True, dual="pid", **B5_GAINS, dual_start=50_000)
    add("SauteA", LIT, budget_input=True, saute_n=0.05)
    add("SauteB", LIT, budget_input=True, saute_n=0.2)
    add("R2", LIT)
    for g in ("0.90", "0.95"):
        for gains_name, gains in (("B4", B4_GAINS), ("B5", B5_GAINS)):
            add(f"SHL-g{g}-{gains_name}", OMNI, dual="pid", **gains, cost_gamma=float(g), rate_norm=True)
    for rule in ("SQ", "SN"):
        for k in ("0.02", "0.1"):
            add(f"BVS-{rule}-k{k}", LIT, kind="bvs", budget_input=True, rule=rule, kappa=float(k))
    return c


CONFIGS: dict[str, RunConfig] = _build()
BASELINE_POOL = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B7b", "B7c", "B8", "B9", "SauteA", "SauteB")
BVS_CONFIGS = ("BVS-SQ-k0.02", "BVS-SQ-k0.1", "BVS-SN-k0.02", "BVS-SN-k0.1")
SHL_CONFIGS = ("SHL-g0.90-B4", "SHL-g0.90-B5", "SHL-g0.95-B4", "SHL-g0.95-B5")
DERIVED = {"BVSU": "bvs_u", "BVSD": "bvs_disc"}   # built from BVS* at method-v1 (from_bvs)


def make(name: str, **overrides) -> RunConfig:
    """A copy of the registry entry `name` with `overrides` applied (unknown fields raise)."""
    if name in DERIVED:
        raise ValueError(f"{name} is built from BVS* with from_bvs(bvs_star_name, kind, rho_star)")
    if name not in CONFIGS:
        raise ValueError(f"unknown config {name!r}; registered: {', '.join(CONFIGS)}")
    return _check(replace(CONFIGS[name], **overrides))


def from_bvs(bvs_star_name: str, kind: str, rho_star: float = float("nan"), **overrides) -> RunConfig:
    """BVSU (kind "bvs_u", rho_star from v1.json) or BVSD (kind "bvs_disc") built from BVS*."""
    if bvs_star_name not in BVS_CONFIGS:
        raise ValueError(f"{bvs_star_name!r} is not a BVS config")
    name = {v: k for k, v in DERIVED.items()}.get(kind)
    if name is None:
        raise ValueError(f"kind must be one of {sorted(DERIVED.values())}")
    if kind == "bvs_u" and not (0.0 <= rho_star <= 1.0):
        raise ValueError("BVSU needs rho_star in [0, 1]")
    return _check(replace(CONFIGS[bvs_star_name], name=name, kind=kind,
                          rho_star=float(rho_star) if kind == "bvs_u" else float("nan"), **overrides))


def _check(rc: RunConfig) -> RunConfig:
    if rc.kind not in KINDS or rc.dual not in DUALS or rc.rule not in ("SQ", "SN"):
        raise ValueError(f"{rc.name}: bad kind / dual / rule ({rc.kind}, {rc.dual}, {rc.rule})")
    if rc.kind != "base" and (rc.dual != "none" or not rc.budget_input):
        raise ValueError(f"{rc.name}: the BVS family has no lambda dual and always sees the budget input")
    if rc.kind != "base" and not (rc.start_steps <= rc.shield_start and (rc.kind != "bvs_u" or rc.shield_start <= rc.u_start)):
        raise ValueError(f"{rc.name}: the BVS family needs start_steps <= shield_start (<= u_start for BVS-U): "
                         "the shield's per-step bookkeeping starts after the warm-up")
    if not (1 <= rc.eval_episodes <= 20):
        raise ValueError("eval_episodes must lie in 1..20 (the train_eval block holds 20 layouts)")
    return rc


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def config_diff(a: dict, b: dict) -> dict:
    """{field: (a, b)} for every field that differs (NaN equals NaN, so rho_star = nan compares)."""
    keys = list(a) + [k for k in b if k not in a]
    return {k: (a.get(k), b.get(k)) for k in keys if not _same(a.get(k), b.get(k))}


def non_default(rc: RunConfig) -> dict:
    """Fields of rc that differ from the RunConfig defaults (name excluded)."""
    return {k: v for k, v in config_diff(asdict(rc), asdict(RunConfig())).items() if k != "name"}


FIELD_NAMES = tuple(f.name for f in fields(RunConfig))

if __name__ == "__main__":
    for n, cfg in CONFIGS.items():
        print(n, {k: v[0] for k, v in non_default(cfg).items()})
