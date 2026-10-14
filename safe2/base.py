"""Baseline agent of the method study and the trainer's agent interface (safe2/SPEC.md section 2.3).

Agent(LagU) runs every kind == "base" config: R1, R2, B1-B9, B7b, B7c, SauteA/B and SHL. It is
lagu.agent.LagU with M = 1, no bonus and no gate (Lag-U's td3lag arm), built from a RunConfig, with
two overridden methods:

* update() changes only the cost target: c + cost_gamma (1 - d) Q_c^targ(o2, a2c), clamped to
  [0, 1 / (1 - cost_gamma)] when cost_clamp is set;
* _policy_update() changes only the penalty: lam_eff (Q_c - delta) with lam_eff = lambda * s, where
  s = (1 - cost_gamma) / (1 - gamma) with rate_norm and 1.0 otherwise.

Every other statement, its order and every RNG draw are copied from LagU, so with cost_gamma == gamma
and s == 1.0 the agent is bit-identical to LagU (tests/test_base.py golden tests). The dual is either
LagU's episodic Adam (dual "adam") or PIDLagrangian (dual "pid", OmniSafe semantics), fed with the
mean realised cost of the last 50 post-warm-up training episodes; dual "none" keeps lambda = 0.

TrainerHooks is the interface safe2/train.py and safe2/evaluate.py call on every agent; safe2/HOOKS.md
documents it for the BVS family (safe2/bvs.py). agent_class(kind) resolves the class of a kind.
"""
from __future__ import annotations

import importlib
from collections import deque
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from lagu.agent import Config, LagU
from safe2.configs import RunConfig

_AGENT_CLASSES = {"base": "safe2.base:Agent", "bvs": "safe2.bvs:BVS", "bvs_u": "safe2.bvs:BVS",
                  "bvs_disc": "safe2.bvs:BVS"}


def agent_class(kind: str):
    """The agent class of a RunConfig kind (imported lazily, so safe2.bvs is optional here)."""
    mod, name = _AGENT_CLASSES[kind].split(":")
    return getattr(importlib.import_module(mod), name)


def budget(C: float, d: float = 25.0) -> float:
    """b = clip((d - C) / d, -1, 1): the remaining-budget input of Saute and the BVS family."""
    return float(np.clip((d - C) / d, -1.0, 1.0))


class TrainerHooks:
    """The interface the trainer and the evaluator call on an agent (safe2/HOOKS.md).

    Defaults are those of a plain off-policy agent without a shield; base.Agent overrides what a
    Lagrangian baseline needs and safe2.bvs.BVS overrides the rest. Every hook must leave every RNG
    it does not own untouched (see HOOKS.md, "RNG contract")."""

    rc: RunConfig
    replay_extras: dict = {}        # extra replay columns {name: (shape tuple, numpy dtype)}
    sample_keys: tuple | None = None  # replay columns learn() reads (None = every column)
    eval_keep_traj: bool = False    # in-training rollouts keep per-step x, r, c for eval_columns()

    @property
    def uses_budget(self) -> bool:
        """The actor's input is cat(o, b) rather than o."""
        return bool(self.rc.budget_input or self.rc.kind != "base")

    def begin_episode(self) -> None:
        """A training episode starts (also the first one after a resume)."""

    def select_action(self, x: np.ndarray, o: np.ndarray, b: float, C: float, step: int):
        """Hook 1 (action selection) at a step > start_steps: returns (a_pi, a_exec, I, info).
        info may hold "replay": {column: value} for extra replay columns (hook 2)."""
        raise NotImplementedError

    def on_step(self, step: int, t: int, c: float, I: int, info: dict) -> None:
        """After env.step (every executed step, warm-up included): per-step bookkeeping."""

    def learn(self, sample: dict, step: int) -> dict:
        """Hook 3 (every loss of one update): one call per step > start_steps, one shared batch."""
        raise NotImplementedError

    def post_update(self, step: int, jc: deque) -> None:
        """After the update of a step (every step): duals and schedules (lambda, BVS-U's T)."""

    def end_episode(self, step: int, ep_ret: float, ep_cost: float) -> None:
        """Hook 4: a training episode ended at `step` with raw return ep_ret and cost ep_cost."""

    def log_row(self) -> dict:
        """The arm's outcome columns of the current CSV row (train.OUTCOME_FIELDS names only)."""
        return {}

    def eval_columns(self, episodes: list) -> dict:
        """Extra outcome columns computed from the in-training evaluation episodes."""
        return {}

    def reset_interval(self) -> None:
        """Called after each CSV row: restart interval accumulators."""

    def console_dual(self) -> str:
        """'lam=...' or 'm=...' for the console line of seeds < 300."""
        return ""

    def snapshot(self) -> dict:
        """Hook 5 (snapshot extras): the agent part of a snapshot (weights and deployed scalars)."""
        raise NotImplementedError

    @classmethod
    def from_snapshot(cls, snap: dict):
        """Rebuild an agent able to act deterministically from a snapshot dict."""
        raise NotImplementedError

    def make_policy(self, mode: str):
        """The deterministic evaluation policy (safe2.evaluate.Policy) in `mode`."""
        raise NotImplementedError


class PIDLagrangian:
    """OmniSafe's PIDLagrangian.pid_update with sum_norm=True and diff_norm=False, capped at lambda_max
    (0 = no cap). update(J) takes the mean realised episode cost of the dual's window."""

    def __init__(self, kp: float, ki: float, kd: float, d_delay: int, ema_p: float, ema_d: float,
                 lambda_max: float, target: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.d_delay, self.ema_p, self.ema_d = d_delay, ema_p, ema_d
        self.lambda_max, self.target = lambda_max, target
        self.I = 0.0
        self.delta_p = 0.0
        self.cost_d = 0.0
        self.cost_ds = deque([0.0], maxlen=d_delay)
        self.last_D = 0.0           # logging only

    def update(self, J: float) -> float:
        delta = J - self.target
        self.I = max(0.0, self.I + self.ki * delta)
        self.delta_p = self.ema_p * self.delta_p + (1 - self.ema_p) * delta
        self.cost_d = self.ema_d * self.cost_d + (1 - self.ema_d) * J
        D = max(0.0, self.cost_d - self.cost_ds[0])
        lam = max(0.0, self.kp * self.delta_p + self.I + self.kd * D)
        if self.lambda_max > 0:
            lam = min(lam, self.lambda_max)
        self.cost_ds.append(self.cost_d)
        self.last_D = D
        return lam

    def state_dict(self) -> dict:
        return {"I": self.I, "delta_p": self.delta_p, "cost_d": self.cost_d, "cost_ds": list(self.cost_ds),
                "last_D": self.last_D}

    def load_state_dict(self, s: dict) -> None:
        self.I, self.delta_p, self.cost_d = s["I"], s["delta_p"], s["cost_d"]
        self.cost_ds = deque(s["cost_ds"], maxlen=self.d_delay)
        self.last_D = s.get("last_D", 0.0)


def lagu_config(rc: RunConfig, horizon: int) -> Config:
    """The lagu.agent.Config of a base RunConfig (SPEC 2.3)."""
    return Config(M=1, use_bonus=False, use_gate=False, hidden=rc.hidden, gamma=rc.gamma, tau=rc.tau, lr=rc.lr,
                  actor_lr=rc.actor_lr, critic_lr=rc.critic_lr, max_grad_norm=rc.max_grad_norm,
                  pi_norm=rc.pi_norm, cost_target_online=rc.cost_target_online, cost_clamp=rc.cost_clamp,
                  cost_max=1.0, batch=rc.batch, policy_delay=rc.policy_delay, target_noise=rc.target_noise,
                  target_clip=rc.target_clip, action_noise=rc.action_noise,
                  delta0=25.0 / horizon / (1.0 - rc.gamma), cost_limit=rc.dual_target,
                  dual=("episodic" if rc.dual == "adam" else "fixed"), lambda_lr=rc.lambda_lr,
                  lambda_init=0.0, lambda_max=rc.lambda_max)


class Agent(LagU, TrainerHooks):
    """Every baseline (kind "base"): LagU's td3lag learner with cost_gamma, rate_norm and a PID dual."""

    eval_keep_traj = True

    def __init__(self, obs_dim: int, act_dim: int, rc: RunConfig, horizon: int, seed: int = 0,
                 device: str = "cpu"):
        if rc.kind != "base":
            raise ValueError(f"{rc.name}: base.Agent runs kind 'base' only")
        self.rc = rc
        self.obs_dim, self.act_dim, self.horizon = obs_dim, act_dim, horizon
        self.cost_gamma = rc.cost_gamma or rc.gamma
        self.rate_scale = (1.0 - self.cost_gamma) / (1.0 - rc.gamma) if rc.rate_norm else 1.0
        in_dim = obs_dim + 1 if rc.budget_input else obs_dim
        super().__init__(in_dim, act_dim, lagu_config(rc, horizon), device=device, seed=seed)
        self.pid = (PIDLagrangian(rc.kp, rc.ki, rc.kd, rc.d_delay, rc.ema_p, rc.ema_d, rc.lambda_max,
                                  rc.dual_target) if rc.dual == "pid" else None)
        self.sample_keys = ("o", "b", "a_pi", "r", "c", "o2", "b2", "d") if rc.budget_input else \
            ("o", "a_pi", "r", "c", "o2", "d")

    # ------------------------------------------------------------- learning (two overrides)
    def update(self, batch) -> dict:
        """LagU.update with the cost target at cost_gamma and its clamp at 1 / (1 - cost_gamma)."""
        cfg = self.cfg
        cg = self.cost_gamma
        o, a, r, c, o2, d = (torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch)
        with torch.no_grad():
            pi2 = self.actor_targ(o2)

            def smoothed():
                eps = (torch.randn_like(a) * cfg.target_noise).clamp(-cfg.target_clip, cfg.target_clip)
                return (pi2 + eps).clamp(-1.0, 1.0)
            targets = [r + cfg.gamma * (1.0 - d) * torch.min(*mt(o2, smoothed())) for mt in self.members_targ]
            a2c = self.actor(o2) if cfg.cost_target_online else smoothed()
            c_targ = c + cg * (1.0 - d) * self.qc_target(o2, a2c)
            if cfg.cost_clamp:
                c_targ = c_targ.clamp(0.0, cfg.cost_max / (1.0 - cg))
        member_losses = []
        for m, t in zip(self.members, targets):
            q1, q2 = m(o, a)
            member_losses.append(F.mse_loss(q1, t) + F.mse_loss(q2, t))
        q_loss = torch.stack(member_losses).sum()
        self._step(self.q_opt, q_loss, self.members.parameters())
        c_loss = sum(F.mse_loss(q, c_targ) for q in self.qc_heads(o, a))
        self._step(self.c_opt, c_loss, self.cost_params)
        self.n_updates += 1
        self.stats = {"q_loss": float(q_loss) / cfg.M, "c_loss": float(c_loss),
                      "member_losses": [float(x) for x in member_losses]}
        if self.n_updates % cfg.policy_delay == 0:
            self.pi_stats = self._policy_update(o)
            self._polyak()
        return self.stats

    def _policy_update(self, o: torch.Tensor) -> dict:
        """LagU._policy_update with the penalty lam_eff (Q_c - delta), lam_eff = float(lambda * s).
        Also logs the min-twin Q_r at (x, pi(x)) (no RNG, after the step, members unchanged)."""
        cfg = self.cfg
        a_pi = self.actor(o)
        qbar, var = self.ensemble(o, a_pi)
        std = torch.sqrt(var + 1e-12)
        qc = self.qc_pi(o, a_pi)
        ratio = std / (qbar.detach().abs() + cfg.ratio_eps)
        if not cfg.gate_grad:
            ratio = ratio.detach()
        delta = torch.full_like(qc, cfg.delta0)
        exploratory = self.rng.random() < self.explore_prob()
        if exploratory:
            obj = qbar + std if cfg.use_bonus else qbar
            self.n_explore += 1
        else:
            obj = qbar
            self.n_risk += 1
            if cfg.use_gate:
                delta = self.adaptive_delta(ratio)
                with torch.no_grad():
                    self.gate_hits += int((ratio >= cfg.T).sum())
                    self.gate_total += int(ratio.numel())
        lam_eff = float(self.lmbda * self.rate_scale)
        loss = (-obj + lam_eff * (qc - delta)).mean()
        if cfg.pi_norm:
            loss = loss / (1.0 + self.lmbda)
        self._step(self.pi_opt, loss, self.actor.parameters())
        gap = float((qc.detach() - delta.detach()).mean())
        qc_mean = float(qc.detach().mean())
        if cfg.dual == "statewise":
            self.lmbda = self._clip_lambda(self.lmbda + cfg.lambda_lr * gap)
            gap0 = float((qc.detach() - cfg.delta0).mean())
            self.lmbda_shadow = self._clip_lambda(self.lmbda_shadow + cfg.lambda_lr * gap0)
        else:
            self.lmbda_shadow = self.lmbda
        self.n_pi += 1
        r = ratio.detach()
        acc = self.acc
        acc["gap"] += gap
        acc["qc"] += qc_mean
        acc["qbar"] += float(qbar.detach().mean())
        acc["qstd"] += float(std.detach().mean())
        acc["delta"] += float(delta.detach().mean())
        acc["lambda"] += self.lmbda
        acc["n_pi"] += 1
        if not exploratory:
            acc["n_risk"] += 1
            acc["delta_risk"] += float(delta.detach().mean())
            if cfg.M > 1:
                acc["would_hits"] += int((r >= cfg.T).sum())
                acc["would_total"] += int(r.numel())
        with torch.no_grad():                       # logging only: members are not stepped here
            qmin = float(torch.min(*self.members[0](o, a_pi.detach())).mean())
        return {
            "pi_loss": float(loss), "lambda": self.lmbda,
            "branch": "explore" if exploratory else "risk", "explore_prob": self.explore_prob(),
            "qbar": float(qbar.mean()), "qstd": float(std.mean()), "qc": float(qc.mean()),
            "delta_mean": float(delta.mean()), "gap": gap,
            "ratio_p50": float(r.median()), "ratio_p90": float(torch.quantile(r, 0.9)),
            "qbar_min": qmin,
        }

    # ------------------------------------------------------------------ dual
    def dual_step(self, J: float) -> None:
        """One dual update on the window mean J (the trainer decides when; SPEC 2.3)."""
        if self.rc.dual == "adam":
            self.update_dual_episodic(J)
        elif self.rc.dual == "pid":
            self.lmbda = self.pid.update(J)
            self.lmbda_shadow = self.lmbda
            self.last_jc = float(J)

    # ------------------------------------------------------------- trainer hooks
    def select_action(self, x, o, b, C, step):
        a = self.act(x, explore=True)
        return a, a, 0, {}

    def learn(self, sample: dict, step: int) -> dict:
        if self.rc.budget_input:
            o = np.concatenate([sample["o"], sample["b"]], axis=1)
            o2 = np.concatenate([sample["o2"], sample["b2"]], axis=1)
        else:
            o, o2 = sample["o"], sample["o2"]
        return self.update((o, sample["a_pi"], sample["r"], sample["c"], o2, sample["d"]))

    def post_update(self, step: int, jc: deque) -> None:
        rc = self.rc
        if rc.dual != "none" and step >= rc.dual_start and jc and step % rc.dual_every == 0:
            self.dual_step(float(np.mean(jc)))

    def log_row(self) -> dict:
        ps, cs, pid = self.pi_stats, self.stats, self.pid
        row = {"qbar": ps.get("qbar_min", ""), "qc": ps.get("qc", ""), "pi_loss": ps.get("pi_loss", ""),
               "q_loss": cs.get("q_loss", ""), "c_loss": cs.get("c_loss", "")}
        if self.rc.dual != "none":
            row["lambda"] = self.lmbda
            row["jc"] = "" if np.isnan(self.last_jc) else self.last_jc
        if pid is not None:     # the three PID terms that sum to lambda (before the cap)
            row.update(pid_p=pid.kp * pid.delta_p, pid_i=pid.I, pid_d=pid.kd * pid.last_D)
        return row

    def eval_columns(self, episodes: list) -> dict:
        """Critic calibration as lagu.train.evaluate, with the cost discounted at cost_gamma."""
        gamma, cg, horizon = self.rc.gamma, self.cost_gamma, self.horizon
        qc0, mc_c, tq, tc, tmr, tmc = [], [], [], [], [], []
        for ep in episodes:
            xs, rs, cs = ep["traj"]["x"], np.asarray(ep["traj"]["r"]), np.asarray(ep["traj"]["c"])
            n = len(rs)
            g_r, g_c = np.zeros(n + 1), np.zeros(n + 1)
            for t in range(n - 1, -1, -1):
                g_r[t] = rs[t] + gamma * g_r[t + 1]
                g_c[t] = cs[t] + cg * g_c[t + 1]
            qc0.append(self.values_at(xs[0])[2])
            keep = max(1, min(len(xs), n - horizon // 2))
            qb_t, qc_t = self.values_batch(np.stack(xs[:keep]))
            tq.append(float(qb_t.mean())), tc.append(float(qc_t.mean()))
            tmr.append(float(g_r[:keep].mean())), tmc.append(float(g_c[:keep].mean()))
            mc_c.append(float(g_c[0]))
        m = lambda v: float(np.mean(v))
        return {"qc0": m(qc0), "mc_cost0": m(mc_c), "qc_traj": m(tc), "mc_cost_traj": m(tmc),
                "qbar_traj": m(tq), "mc_ret_traj": m(tmr)}

    def reset_interval(self) -> None:
        self.reset_counters()

    def console_dual(self) -> str:
        return f"lam={self.lmbda:.3g}"

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        d = super().state_dict()
        d["run_config"] = asdict(self.rc)
        d["pid"] = self.pid.state_dict() if self.pid is not None else None
        return d

    def load_state_dict(self, d: dict) -> None:
        super().load_state_dict(d)
        if self.pid is not None:
            self.pid.load_state_dict(d["pid"])

    def snapshot(self) -> dict:
        return {"actor": self.actor.state_dict(), "members": self.members.state_dict(),
                "cost": self.cost.state_dict(), "lambda": self.lmbda}

    @classmethod
    def from_snapshot(cls, snap: dict) -> "Agent":
        """Rebuild from a snapshot without touching the caller's torch RNG (fork_rng)."""
        rc = RunConfig(**snap["config"])
        with torch.random.fork_rng():
            agent = cls(snap["obs_dim"], snap["act_dim"], rc, snap["horizon"])
        for name in ("actor", "members", "cost"):
            getattr(agent, name).load_state_dict(snap[name])
        agent.lmbda = float(snap["lambda"])
        return agent

    def make_policy(self, mode: str):
        from safe2.evaluate import BasePolicy
        if mode != "deployed":
            raise ValueError(f"{self.rc.name}: a baseline has only the 'deployed' mode, not {mode!r}")
        return BasePolicy(self.actor, self.rc.budget_input, self.rc.cost_limit)
