"""Faithful port of Lag-U (Zhang, Liu, Li, Lin, Li; IEEE T-ITS 25(10):13653-13666, 2024,
DOI 10.1109/TITS.2024.3397700) on a TD3-Lagrangian base, with switches that isolate its
ingredients: the sqrt(Q_eu) exploration bonus (Eq. 15) and the uncertainty-gated constraint
delta_ada (Eq. 16). Equation numbers refer to the paper. Deviations are listed in README.md
under "Fidelity".
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(inp: int, out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class TwinQ(nn.Module):
    """One critic-ensemble member: a TD3 twin pair trained on its own target (Eq. 12)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


@dataclass
class Config:
    M: int = 3                   # ensemble members                          (paper: 3)
    hidden: int = 256            # width of the two hidden layers            (paper: 256)
    gamma: float = 0.99          #                                           (paper: 0.99)
    tau: float = 0.005           # Polyak factor for every target network
    lr: float = 3e-4             # actor, critics, cost critic               (paper: 3e-4)
    actor_lr: float = 0.0        # 0 = use lr. OmniSafe navigation recipe: actor 5e-6, critics 1e-3
    critic_lr: float = 0.0       # 0 = use lr (reward members and cost critic)
    max_grad_norm: float = 0.0   # 0 = no clipping. OmniSafe: 40 on actor and every critic
    pi_norm: bool = False        # divide the actor loss by (1 + lambda) (OmniSafe, Stooke et al. 2020)
    cost_target_online: bool = False  # cost TD target uses the online actor with no smoothing noise (OmniSafe)
    lambda_max: float = 0.0      # 0 = unbounded; otherwise clamp lambda to [0, lambda_max]
    lambda_lr: float = 1e-5      # alpha_lambda in Eq. 9 / Eq. 18            (paper: 1e-5, MetaDrive cost scale)
    lambda_init: float = 0.0
    dual: str = "statewise"      # "statewise": lambda += lr * E[Q_c - delta] (Eq. 9/18)
                                 # "fixed":     lambda stays at lambda_init (Pareto sweeps)
                                 # "episodic":  Adam ascent on lambda * (J_c - cost_limit) from realised
                                 #              episode costs (OmniSafe convention for TD3-Lag)
    cost_limit: float = 25.0     # episodic limit; only the episodic dual reads it
    gate_grad: bool = False      # let the actor gradient flow through delta_ada. The paper does not
                                 # say; detached (False) makes the gate act only through lambda's growth.
    cost_twin: bool = False      # pessimistic twin cost critic: target and actor use the MAX of two
                                 # heads (mirror of Eq. 12's min). Off = the paper's single critic.
    cost_clamp: bool = False     # clamp cost targets to [0, cost_max / (1 - gamma)]
    cost_max: float = 1.0        # largest per-step cost the task can emit
    batch: int = 256
    policy_delay: int = 2        # TD3-style delayed actor update
    target_noise: float = 0.2    # target-policy smoothing epsilon (Eq. 12)
    target_clip: float = 0.5
    delta0: float = 2.5          # constraint on Q_c(s, pi(s))               (paper: 0.1 on MetaDrive)
    T: float = 0.07              # uncertainty-ratio threshold (Eq. 16)      (paper: 0.07)
    c0: float = 0.1              # floor of the exploration probability      (paper: 0.1)
    Tc: int = 300_000            # exploration stop, in ENV steps x          (paper: 3e5, "incremental steps")
    use_bonus: bool = True       # + sqrt(Q_eu) in the exploratory update    (Eq. 15)
    use_gate: bool = True        # delta_ada in the risk-sensitive update    (Eq. 16-18)
    action_noise: float = 0.0    # Gaussian action noise; the paper cancels it for Lag-U
    ratio_eps: float = 1e-6      # guards sqrt(Q_eu) / |Q_bar| when Q_bar is near 0


class LagU:
    def __init__(self, obs_dim: int, act_dim: int, cfg: Config, device: str = "cpu", seed: int = 0):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.actor_targ = copy.deepcopy(self.actor)
        self.members = nn.ModuleList(
            [TwinQ(obs_dim, act_dim, cfg.hidden) for _ in range(cfg.M)]).to(self.device)
        self.members_targ = copy.deepcopy(self.members)
        self.cost = mlp(obs_dim + act_dim, 1, cfg.hidden).to(self.device)
        self.cost_targ = copy.deepcopy(self.cost)
        cost_params = list(self.cost.parameters())
        targets = [self.actor_targ, self.members_targ, self.cost_targ]
        if cfg.cost_twin:
            self.cost2 = mlp(obs_dim + act_dim, 1, cfg.hidden).to(self.device)
            self.cost2_targ = copy.deepcopy(self.cost2)
            cost_params += list(self.cost2.parameters())
            targets.append(self.cost2_targ)
        for net in targets:
            for p in net.parameters():
                p.requires_grad_(False)
        self.cost_params = cost_params
        a_lr, c_lr = (cfg.actor_lr or cfg.lr), (cfg.critic_lr or cfg.lr)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.q_opt = torch.optim.Adam(self.members.parameters(), lr=c_lr)
        self.c_opt = torch.optim.Adam(cost_params, lr=c_lr)
        self.lmbda = float(cfg.lambda_init)
        self.lmbda_shadow = float(cfg.lambda_init)   # logging only: Eq. 18 with delta0 in place of delta_ada
        if cfg.dual == "episodic":
            self._lam_param = nn.Parameter(torch.tensor(float(cfg.lambda_init)))
            self.lam_opt = torch.optim.Adam([self._lam_param], lr=cfg.lambda_lr)
        self.last_jc = float("nan")
        self.n_updates = 0
        self.n_pi = 0
        self.env_step = 0          # x in Eq. 13; set by the trainer every step
        self.rng = np.random.default_rng(seed)
        self.stats: dict = {}      # last critic update
        self.pi_stats: dict = {}   # last policy update
        self.reset_counters()

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o).squeeze(0).cpu().numpy()
        if explore and self.cfg.action_noise > 0:
            a = a + self.rng.normal(0.0, self.cfg.action_noise, size=a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    @torch.no_grad()
    def values_at(self, obs: np.ndarray):
        """(Q_bar, sqrt(Q_eu), Q_c) at (obs, pi(obs)); used for critic calibration at eval time."""
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o)
        qbar, var = self.ensemble(o, a)
        return float(qbar), float(torch.sqrt(var + 1e-12)), float(self.qc_pi(o, a))

    @torch.no_grad()
    def values_batch(self, obs: np.ndarray):
        """(Q_bar, Q_c) at (s, pi(s)) for a batch of states."""
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        a = self.actor(o)
        qbar, _ = self.ensemble(o, a)
        return qbar.cpu().numpy(), self.qc_pi(o, a).cpu().numpy()

    # ------------------------------------------------------------- mechanisms
    def ensemble(self, obs: torch.Tensor, act: torch.Tensor):
        """Q_bar (Eq. 10) and Q_eu (Eq. 11): mean and population variance over the
        members' first heads at (obs, act)."""
        qs = torch.stack([m(obs, act)[0] for m in self.members], dim=0)
        return qs.mean(0), qs.var(0, unbiased=False)

    def qc_heads(self, obs: torch.Tensor, act: torch.Tensor) -> list:
        x = torch.cat([obs, act], dim=-1)
        heads = [self.cost(x).squeeze(-1)]
        if self.cfg.cost_twin:
            heads.append(self.cost2(x).squeeze(-1))
        return heads

    def qc_pi(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Cost value the actor and the dual see: the pessimistic max over heads when twinned."""
        heads = self.qc_heads(obs, act)
        return torch.max(heads[0], heads[1]) if self.cfg.cost_twin else heads[0]

    @torch.no_grad()
    def qc_target(self, obs2: torch.Tensor, act2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs2, act2], dim=-1)
        v = self.cost_targ(x).squeeze(-1)
        if self.cfg.cost_twin:
            v = torch.max(v, self.cost2_targ(x).squeeze(-1))
        return v

    def explore_prob(self) -> float:
        """c(x) = max(c0, 1 - x / Tc), x = the agent's environment steps so far (Eq. 13)."""
        return max(self.cfg.c0, 1.0 - self.env_step / float(self.cfg.Tc))

    def _clip_lambda(self, lam: float) -> float:
        lam = max(0.0, lam)
        return min(lam, self.cfg.lambda_max) if self.cfg.lambda_max > 0 else lam

    def _step(self, opt, loss, params) -> None:
        opt.zero_grad()
        loss.backward()
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        opt.step()

    def adaptive_delta(self, ratio: torch.Tensor) -> torch.Tensor:
        """Eq. 16: delta0 below the threshold, delta0 * T / ratio above it."""
        d0, T = self.cfg.delta0, self.cfg.T
        return torch.where(ratio < T, torch.full_like(ratio, d0), d0 * T / ratio)

    def reset_counters(self) -> None:
        self.gate_hits = 0     # risk-sensitive samples with ratio >= T
        self.gate_total = 0    # risk-sensitive samples seen while the gate is on
        self.n_explore = 0     # exploratory policy updates
        self.n_risk = 0        # risk-sensitive policy updates
        # interval accumulators: plain floats computed after the optimizer step; they never touch
        # the loss or any RNG, so logging cannot change training (see tests/golden)
        self.acc = {k: 0.0 for k in ("gap", "qc", "qbar", "qstd", "delta", "lambda",
                                      "delta_risk", "n_pi", "n_risk", "would_hits", "would_total")}

    def interval_means(self) -> dict:
        a, nan = self.acc, float("nan")
        n, nr, nw = a["n_pi"], a["n_risk"], a["would_total"]
        out = {f"{k}_avg": (a[k] / n if n else nan) for k in ("gap", "qc", "qbar", "qstd", "delta", "lambda")}
        out["delta_risk_avg"] = a["delta_risk"] / nr if nr else nan
        out["would_engage"] = a["would_hits"] / nw if nw else nan
        out["n_pi"] = int(n)
        return out

    def engagement(self) -> float:
        """Fraction of risk-sensitive samples on which the gate tightened the constraint."""
        return self.gate_hits / self.gate_total if self.gate_total else 0.0

    def update_dual_episodic(self, jc: float) -> None:
        """OmniSafe-style dual step: gradient ascent (Adam) on lambda * (J_c - cost_limit)."""
        self.last_jc = float(jc)
        loss = -self._lam_param * (jc - self.cfg.cost_limit)
        self.lam_opt.zero_grad()
        loss.backward()
        self.lam_opt.step()
        with torch.no_grad():
            self._lam_param.fill_(self._clip_lambda(float(self._lam_param)))
        self.lmbda = float(self._lam_param)
        self.lmbda_shadow = self.lmbda

    # --------------------------------------------------------------- learning
    def update(self, batch) -> dict:
        cfg = self.cfg
        o, a, r, c, o2, d = (torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch)
        with torch.no_grad():
            pi2 = self.actor_targ(o2)

            def smoothed():
                eps = (torch.randn_like(a) * cfg.target_noise).clamp(-cfg.target_clip, cfg.target_clip)
                return (pi2 + eps).clamp(-1.0, 1.0)
            # each member bootstraps from the min of ITS OWN twin targets, with its own noise (Eq. 12)
            targets = [r + cfg.gamma * (1.0 - d) * torch.min(*mt(o2, smoothed())) for mt in self.members_targ]
            a2c = self.actor(o2) if cfg.cost_target_online else smoothed()
            c_targ = c + cfg.gamma * (1.0 - d) * self.qc_target(o2, a2c)
            if cfg.cost_clamp:
                c_targ = c_targ.clamp(0.0, cfg.cost_max / (1.0 - cfg.gamma))
        member_losses = []
        for m, t in zip(self.members, targets):
            q1, q2 = m(o, a)
            member_losses.append(F.mse_loss(q1, t) + F.mse_loss(q2, t))
        q_loss = torch.stack(member_losses).sum()
        self._step(self.q_opt, q_loss, self.members.parameters())
        c_loss = sum(F.mse_loss(q, c_targ) for q in self.qc_heads(o, a))  # Eq. 14 (per head)
        self._step(self.c_opt, c_loss, self.cost_params)
        self.n_updates += 1
        self.stats = {"q_loss": float(q_loss) / cfg.M, "c_loss": float(c_loss),
                      "member_losses": [float(x) for x in member_losses]}
        if self.n_updates % cfg.policy_delay == 0:
            self.pi_stats = self._policy_update(o)
            self._polyak()
        return self.stats

    def _policy_update(self, o: torch.Tensor) -> dict:
        cfg = self.cfg
        a_pi = self.actor(o)
        qbar, var = self.ensemble(o, a_pi)
        std = torch.sqrt(var + 1e-12)
        qc = self.qc_pi(o, a_pi)
        # sqrt(Q_eu) / Q_bar, Eq. 16. With gate_grad the actor gradient flows through sqrt(Q_eu) only:
        # through |Q_bar| it would flip sign when Q_bar < 0 and push the actor to LOWER the reward value.
        ratio = std / (qbar.detach().abs() + cfg.ratio_eps)
        if not cfg.gate_grad:
            ratio = ratio.detach()
        delta = torch.full_like(qc, cfg.delta0)
        exploratory = self.rng.random() < self.explore_prob()
        if exploratory:                                                 # Eq. 15
            obj = qbar + std if cfg.use_bonus else qbar
            self.n_explore += 1
        else:                                                           # Eq. 17
            obj = qbar
            self.n_risk += 1
            if cfg.use_gate:                                            # Eq. 16
                delta = self.adaptive_delta(ratio)
                with torch.no_grad():
                    self.gate_hits += int((ratio >= cfg.T).sum())
                    self.gate_total += int(ratio.numel())
        loss = (-obj + self.lmbda * (qc - delta)).mean()
        if cfg.pi_norm:
            loss = loss / (1.0 + self.lmbda)
        self._step(self.pi_opt, loss, self.actor.parameters())
        gap = float((qc.detach() - delta.detach()).mean())              # E[Q_c - delta]
        qc_mean = float(qc.detach().mean())
        if cfg.dual == "statewise":                                     # Eq. 9 / Eq. 18
            self.lmbda = self._clip_lambda(self.lmbda + cfg.lambda_lr * gap)
            # same float32 arithmetic as gap, so it equals lambda exactly whenever delta == delta0
            gap0 = float((qc.detach() - cfg.delta0).mean())
            self.lmbda_shadow = self._clip_lambda(self.lmbda_shadow + cfg.lambda_lr * gap0)
        else:
            self.lmbda_shadow = self.lmbda
        self.n_pi += 1
        r = ratio.detach()
        a = self.acc
        a["gap"] += gap
        a["qc"] += qc_mean
        a["qbar"] += float(qbar.detach().mean())
        a["qstd"] += float(std.detach().mean())
        a["delta"] += float(delta.detach().mean())
        a["lambda"] += self.lmbda
        a["n_pi"] += 1
        if not exploratory:
            a["n_risk"] += 1
            a["delta_risk"] += float(delta.detach().mean())
            if cfg.M > 1:                    # would the gate fire here? counted with the gate on or off
                a["would_hits"] += int((r >= cfg.T).sum())
                a["would_total"] += int(r.numel())
        return {
            "pi_loss": float(loss), "lambda": self.lmbda,
            "branch": "explore" if exploratory else "risk", "explore_prob": self.explore_prob(),
            "qbar": float(qbar.mean()), "qstd": float(std.mean()), "qc": float(qc.mean()),
            "delta_mean": float(delta.mean()), "gap": gap,
            "ratio_p50": float(r.median()), "ratio_p90": float(torch.quantile(r, 0.9)),
        }

    @torch.no_grad()
    def _polyak(self) -> None:
        tau = self.cfg.tau
        pairs = [(self.actor, self.actor_targ), (self.members, self.members_targ),
                 (self.cost, self.cost_targ)]
        if self.cfg.cost_twin:
            pairs.append((self.cost2, self.cost2_targ))
        for net, targ in pairs:
            for p, pt in zip(net.parameters(), targ.parameters()):
                pt.mul_(1.0 - tau).add_(tau * p)

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        d = {"actor": self.actor.state_dict(), "actor_targ": self.actor_targ.state_dict(),
             "members": self.members.state_dict(), "members_targ": self.members_targ.state_dict(),
             "cost": self.cost.state_dict(), "cost_targ": self.cost_targ.state_dict(),
             "pi_opt": self.pi_opt.state_dict(), "q_opt": self.q_opt.state_dict(),
             "c_opt": self.c_opt.state_dict(),
             "lambda": self.lmbda, "lambda_shadow": self.lmbda_shadow, "last_jc": self.last_jc,
             "n_updates": self.n_updates, "n_pi": self.n_pi, "env_step": self.env_step,
             "counters": (self.gate_hits, self.gate_total, self.n_explore, self.n_risk),
             "rng": self.rng.bit_generator.state, "cfg": asdict(self.cfg)}
        if self.cfg.dual == "episodic":
            d["lam_param"] = float(self._lam_param)
            d["lam_opt"] = self.lam_opt.state_dict()
        if self.cfg.cost_twin:
            d["cost2"], d["cost2_targ"] = self.cost2.state_dict(), self.cost2_targ.state_dict()
        return d

    def load_state_dict(self, d: dict) -> None:
        for name in ("actor", "actor_targ", "members", "members_targ", "cost", "cost_targ"):
            getattr(self, name).load_state_dict(d[name])
        for name in ("pi_opt", "q_opt", "c_opt"):
            getattr(self, name).load_state_dict(d[name])
        self.lmbda = d["lambda"]
        self.lmbda_shadow = d.get("lambda_shadow", self.lmbda)
        self.last_jc = d.get("last_jc", float("nan"))
        self.n_updates, self.n_pi = d["n_updates"], d["n_pi"]
        self.env_step = d.get("env_step", 0)
        self.gate_hits, self.gate_total, self.n_explore, self.n_risk = d["counters"]
        self.rng.bit_generator.state = d["rng"]
        if self.cfg.dual == "episodic":
            with torch.no_grad():
                self._lam_param.fill_(d["lam_param"])
            self.lam_opt.load_state_dict(d["lam_opt"])
        if self.cfg.cost_twin:
            self.cost2.load_state_dict(d["cost2"])
            self.cost2_targ.load_state_dict(d["cost2_targ"])
