"""The BVS family of the method study: the budget-viability shield (BVS), its controls BVS-U and
BVS-disc, and the post-hoc shield on a baseline (PHS) (safe2/SPEC.md section 2.4; prereg_method.md
Appendix A). No CLI: safe2/train.py runs the agents through the hooks of safe2/HOOKS.md,
safe2/evaluate.py wraps them in ShieldPolicy, and safe2/posthoc.py drives fit_offline().

BVS (kind "bvs") is a reward-only TD3 proposer pi(x), x = cat(o, b), b = clip((d - C)/d, -1, 1),
whose action passes through a shield from step shield_start on:

* V (ViabilityNet) is a twin critic with H = 32 heads. Head h estimates the minimum cost reachable
  within h steps by fixed-horizon, undiscounted TD: y_1 = c, y_h = c + (1 - d) min over 6 target
  candidates of max_j V^targ_{h-1}. An action is allowed iff max_j V^j_H(o, a) <= tau, with
  tau = max(eps, d - m - C), and tau = eps once C >= d (strict mode: always eps).
* If a_pi is not allowed, 32 candidates (16 perturbations of a_pi, 15 uniform actions and the backup
  actor pi_b(o)) are screened. Rule S-Q executes the allowed one with the highest min-twin Q_r, rule
  S-N the allowed one nearest a_pi; with none allowed the best-effort exit executes the argmin of
  vmax. Ties go to the first index.
* The proposer learns the shielded MDP (Recovery-RL relabel): Q_r is regressed at (x, a_pi) on
  r - kappa I; V and pi_b train on the executed action. No cost term enters any actor gradient.
* The margin m is a realised-cost dual: m_tilde = max(-d/2, m_tilde + eta_m (J_e - d)) after each
  training episode ending after margin_start (projected anti-windup, prereg A.6); the acting margin is
  clip(m_tilde, -d/2, d), fixed per episode; the deployed margin m_dep is the mean acting margin of
  the last 20 training episodes.

Controls (the same class; the RunConfig's kind decides):

* BVS-U ("bvs_u"): 3 reward-critic members, member 1 playing BVS's Q_r. The takeover branch is
  entered when Lag-U's ratio u(x, a_pi) >= T, T a running (1 - rho*) quantile of u; a_pi joins the
  candidates as index 32, and I = 1 iff the executed action differs from a_pi (by value, prereg A.4;
  a saturated a_pi can be duplicated by a clipped perturbation, which then gives I = 0, in every kind).
* BVS-disc ("bvs_disc"): V is replaced by a twin discounted minimum-cost critic Q_b
  (gamma_b = 1 - 1/32, targets clamped to [0, 32]) in the same budget test.

PHS: fit_offline() trains V and pi_b on a finished baseline run's replay with the same update code
(class Viability, SPEC 7), and PHSCore puts the same shield around the baseline's deterministic actor
at evaluation only.

Random streams (SPEC 2.4): self.rng = default_rng(seed) for exploration noise (as LagU);
shield_rng and v_gen from SeedSequence(seed).spawn(2) for the training candidates and V's uniform
target candidates; the torch global RNG only for the reward critics' target smoothing, so the reward
critics consume the same stream as a TD3 base. Evaluation passes default_rng(layout id) as the
candidate sampler and consumes no training stream.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lagu.agent import Actor, TwinQ, mlp
from safe2.base import TrainerHooks
from safe2.configs import RunConfig

KINDS = ("bvs", "bvs_u", "bvs_disc")
MODES = ("on", "off", "strict")
RHO_FROM = 35_000        # rho_num, rho_den and trig_num count training steps > 35,000 (prereg section 6)
ENTRY_LOOKBACK = 12      # an entry has no cost in the 12 steps before it, and t_e >= 12 (SPEC 2.4)
COUNTERS = ("int_cum", "shield_steps_cum", "empty_cum", "trig_cum", "F_cum", "ep_cum", "entries_cum",
            "misses_cum", "rho_num", "rho_den", "trig_num", "trig007_num")
U_ONLY = ("trig_cum", "trig_num", "trig007_num")    # BVS-U counters; empty CSV cells for the other kinds


def _frozen_copy(net: nn.Module) -> nn.Module:
    """A target network: a deepcopy with requires_grad_(False)."""
    targ = copy.deepcopy(net)
    targ.requires_grad_(False)
    return targ


def _step(opt: torch.optim.Optimizer, loss: torch.Tensor) -> None:
    """zero_grad, backward, step (no gradient clipping in the BVS family)."""
    opt.zero_grad()
    loss.backward()
    opt.step()


@torch.no_grad()
def _polyak(pairs, tau: float) -> None:
    for net, targ in pairs:
        for p, pt in zip(net.parameters(), targ.parameters()):
            pt.mul_(1.0 - tau).add_(tau * p)


# --- networks -----------------------------------------------------------------------------------------
class ViabilityNet(nn.Module):
    """Twin viability critic with H heads: forward(o, a) has shape (2, *batch, H), with
    V[j, ..., h-1] = h * sigmoid(f_j(o, a))[..., h-1] in [0, h] for h = 1..H."""

    def __init__(self, obs_dim: int, act_dim: int, H: int, hidden: int):
        super().__init__()
        self.f = nn.ModuleList([mlp(obs_dim + act_dim, H, hidden) for _ in range(2)])
        self.register_buffer("h", torch.arange(1, H + 1, dtype=torch.float32))

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o, a], dim=-1)
        return torch.stack([torch.sigmoid(f(x)) for f in self.f]) * self.h

    def vmax(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """max_j V^j_H(o, a), shape (*batch)."""
        return self.forward(o, a)[..., -1].max(0).values


class DiscCritic(nn.Module):
    """BVS-disc's twin discounted minimum-cost critic: forward(o, a) has shape (2, *batch), linear."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.f = nn.ModuleList([mlp(obs_dim + act_dim, 1, hidden) for _ in range(2)])

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o, a], dim=-1)
        return torch.stack([f(x).squeeze(-1) for f in self.f])

    def vmax(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """max_j Q_b^j(o, a), shape (*batch)."""
        return self.forward(o, a).max(0).values


class Viability:
    """V (or Q_b) and the backup actor pi_b, their targets and Adam optimisers: SPEC 2.4 update steps
    2-3. The BVS agent and the offline PHS fit (SPEC 7) both train through this class. Networks are
    built in the order V (or Q_b), its target, pi_b, its target. `gen` draws the uniform target
    candidates (BVS: v_gen; the offline fit: a generator seeded with its seed)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, H: int = 32, disc: bool = False,
                 lr: float = 3e-4, gen: torch.Generator | None = None, n_targ_uniform: int = 4,
                 gamma_b: float = 1 - 1 / 32, qb_max: float = 32.0):
        self.disc, self.H, self.act_dim = disc, H, act_dim
        self.n_targ_uniform, self.gamma_b, self.qb_max = n_targ_uniform, gamma_b, qb_max
        self.crit = DiscCritic(obs_dim, act_dim, hidden) if disc else ViabilityNet(obs_dim, act_dim, H, hidden)
        self.crit_targ = _frozen_copy(self.crit)
        self.pib = Actor(obs_dim, act_dim, hidden)
        self.pib_targ = _frozen_copy(self.pib)
        self.opt = torch.optim.Adam(self.crit.parameters(), lr=lr)
        self.pib_opt = torch.optim.Adam(self.pib.parameters(), lr=lr)
        self.gen = gen if gen is not None else torch.Generator()

    @torch.no_grad()
    def targets(self, o2: torch.Tensor, pi2: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """V: y (B, H) with y[:, 0] = c and y[:, h] = c + (1 - d) mn[:, h - 1], mn = min over the target
        candidates of max_j V^targ(o2, .); no head bootstraps from itself.
        Q_b: y (B,) = clamp(c + gamma_b (1 - d) min over candidates of max_j Q_b^targ(o2, .), 0, qb_max).
        Candidates C2 = [pi_b^targ(o2), pi2, n_targ_uniform x U[-1, 1]^act from self.gen], shape (6, B, act)."""
        B = pi2.shape[0]
        uni = torch.rand((self.n_targ_uniform, B, self.act_dim), generator=self.gen) * 2.0 - 1.0
        C2 = torch.cat([self.pib_targ(o2).unsqueeze(0), pi2.unsqueeze(0), uni])
        vt = self.crit_targ(o2.unsqueeze(0).expand(C2.shape[0], *o2.shape), C2)    # (2, 6, B[, H])
        mn = vt.max(0).values.min(0).values                                        # (B[, H])
        if self.disc:
            return (c + self.gamma_b * (1.0 - d) * mn).clamp(0.0, self.qb_max)
        y = torch.empty_like(mn)
        y[:, 0] = c
        y[:, 1:] = c.unsqueeze(1) + (1.0 - d).unsqueeze(1) * mn[:, :-1]
        return y

    def update(self, o: torch.Tensor, a: torch.Tensor, c: torch.Tensor, o2: torch.Tensor, d: torch.Tensor,
               pi2: torch.Tensor) -> float:
        """One Adam step of V (Q_b) at the executed action a: sum_j sum_h mean_B (V[j, :, h] - y[:, h])^2."""
        y = self.targets(o2, pi2, c, d)
        loss = ((self.crit(o, a) - y.unsqueeze(0)) ** 2).mean(1).sum()
        _step(self.opt, loss)
        return float(loss)

    def update_pib(self, o: torch.Tensor) -> float:
        """One Adam step of pi_b on mean_B vmax(o, pi_b(o)). Only pi_b steps; the gradient this loss
        leaves on V is discarded by V's zero_grad before its own next backward."""
        loss = self.crit.vmax(o, self.pib(o)).mean()
        _step(self.pib_opt, loss)
        return float(loss)

    def polyak(self, tau: float) -> None:
        _polyak(((self.crit, self.crit_targ), (self.pib, self.pib_targ)), tau)


# --- the shield --------------------------------------------------------------------------------------
class _Shield:
    """The SPEC 2.4 shield step, shared by BVS and PHSCore. A subclass provides rc, kind, act_dim,
    via (Viability: the critic and pi_b), sq_critic (the twin ranked by S-Q), q_input(x, o) and, for
    BVS-U, members (the trigger's ensemble) and T."""

    def tau(self, C: float, m: float, mode: str = "on") -> float:
        """tau = eps in strict mode or once C >= d (whatever m), else max(eps, d - m - C)."""
        d, eps = self.rc.cost_limit, self.rc.eps
        if mode == "strict" or C >= d:
            return eps
        return max(eps, d - m - C)

    def candidates(self, o: np.ndarray, a_pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The 32 candidates in SPEC order: n_perturb x clip(a_pi + N(0, sigma_c^2)) (rng.normal),
        n_uniform x U[-1, 1]^act (rng.uniform), then pi_b(o) (index 31)."""
        rc = self.rc
        pert = np.clip(a_pi + rng.normal(0.0, rc.sigma_c, size=(rc.n_perturb, self.act_dim)), -1.0, 1.0)
        uni = rng.uniform(-1.0, 1.0, size=(rc.n_uniform, self.act_dim))
        with torch.no_grad():
            pb = self.via.pib(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).numpy()
        return np.concatenate([pert, uni, pb]).astype(np.float32)

    @torch.no_grad()
    def _vmax(self, o: np.ndarray, A: np.ndarray) -> np.ndarray:
        """max_j V^j_H(o, a) (or max_j Q_b^j) for each row a of A."""
        At = torch.as_tensor(np.asarray(A), dtype=torch.float32)
        ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).expand(At.shape[0], -1)
        return self.via.crit.vmax(ot, At).numpy()

    @torch.no_grad()
    def _sq(self, x: np.ndarray, o: np.ndarray, A: np.ndarray) -> np.ndarray:
        """S-Q's ranking: min(q1, q2) of the S-Q critic at (input, a) for each row a of A."""
        At = torch.as_tensor(np.asarray(A), dtype=torch.float32)
        inp = torch.as_tensor(self.q_input(x, o), dtype=torch.float32).unsqueeze(0).expand(At.shape[0], -1)
        return torch.min(*self.sq_critic(inp, At)).numpy()

    @torch.no_grad()
    def u(self, x: np.ndarray, a: np.ndarray) -> float:
        """BVS-U's trigger quantity: sqrt(var_m Q^m_1 + 1e-12) / (|mean_m Q^m_1| + 1e-6) over the members'
        first heads at (x, a), population variance, exactly as LagU.ensemble and LagU._policy_update."""
        xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        at = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)
        qs = torch.stack([m(xt, at)[0] for m in self.members], dim=0)
        qbar, var = qs.mean(0), qs.var(0, unbiased=False)
        return float(torch.sqrt(var + 1e-12) / (qbar.abs() + 1e-6))

    def act(self, o: np.ndarray, b: float, C: float, a_pi: np.ndarray, m: float, rng: np.random.Generator,
            mode: str = "on", rule: str | None = None):
        """One shield step: (a_exec, I, info). info holds "empty" and "vmax" = vmax(o, a_exec), plus
        "n_allowed" on steps that screen candidates and "u", "triggered" for BVS-U (modes on and
        strict). rule defaults to rc.rule; rng draws the candidates (training: shield_rng; evaluation:
        default_rng(layout id)). Mode "off" executes a_pi; "strict" uses tau = eps."""
        if mode not in MODES:
            raise ValueError(f"unknown shield mode {mode!r}")
        rule = rule or self.rc.rule
        if rule not in ("SQ", "SN"):
            raise ValueError(f"unknown rule {rule!r}")
        a_pi = np.asarray(a_pi, dtype=np.float32)
        if mode == "off":
            return a_pi, 0, {"empty": False, "vmax": float(self._vmax(o, a_pi[None])[0])}
        tau = self.tau(C, m, mode)
        x = np.concatenate([o, [b]])
        info: dict = {"empty": False}
        if self.kind == "bvs_u":
            u = self.u(x, a_pi)
            info.update(u=u, triggered=bool(u >= self.T))
            if not info["triggered"]:
                info["vmax"] = float(self._vmax(o, a_pi[None])[0])
                return a_pi, 0, info
            cands = np.concatenate([self.candidates(o, a_pi, rng), a_pi[None]])   # a_pi is index 32
        else:
            v_pi = float(self._vmax(o, a_pi[None])[0])
            if v_pi <= tau:
                info["vmax"] = v_pi
                return a_pi, 0, info
            cands = self.candidates(o, a_pi, rng)
        vm = self._vmax(o, cands)
        allowed = np.flatnonzero(vm <= tau)
        if len(allowed):
            if rule == "SQ":
                idx = int(allowed[np.argmax(self._sq(x, o, cands)[allowed])])
            else:
                idx = int(allowed[np.argmin(np.linalg.norm(cands[allowed] - a_pi, axis=1))])
        else:                                       # best-effort exit
            idx = int(np.argmin(vm))
            info["empty"] = True
        info.update(n_allowed=int(len(allowed)), vmax=float(vm[idx]))
        I = int(not np.array_equal(cands[idx], a_pi))   # A.4: I = 1[a_exec != a_pi], by value
        return cands[idx], I, info

    @torch.no_grad()
    def saturation_gap(self, o: np.ndarray, a: np.ndarray) -> float:
        """max_j (V^j_H - V^j_{H-1})(o, a), logging only. Q_b has no heads: NaN for BVS-disc."""
        if self.via.disc:
            return float("nan")
        v = self.via.crit(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0),
                          torch.as_tensor(a, dtype=torch.float32).unsqueeze(0))
        return float((v[..., -1] - v[..., -2]).max())


# --- the agent ---------------------------------------------------------------------------------------
class BVS(_Shield, TrainerHooks):
    """BVS, BVS-U and BVS-disc (rc.kind), driven by safe2/train.py through the hooks of safe2/HOOKS.md."""

    def __init__(self, obs_dim: int, act_dim: int, rc: RunConfig, horizon: int, seed: int = 0):
        if rc.kind not in KINDS:
            raise ValueError(f"{rc.name}: safe2.bvs.BVS runs kinds {KINDS}, not {rc.kind!r}")
        self.rc, self.kind = rc, rc.kind
        self.obs_dim, self.act_dim, self.horizon = obs_dim, act_dim, horizon
        self.d = rc.cost_limit
        x_dim = obs_dim + 1
        # networks in SPEC order: actor, members, V or Q_b, pi_b, each followed by its target
        self.actor = Actor(x_dim, act_dim, rc.hidden)
        self.actor_targ = _frozen_copy(self.actor)
        M = rc.M_u if rc.kind == "bvs_u" else 1
        self.members = nn.ModuleList([TwinQ(x_dim, act_dim, rc.hidden) for _ in range(M)])
        self.members_targ = _frozen_copy(self.members)
        ss = np.random.SeedSequence(seed).spawn(2)
        self.v_gen = torch.Generator().manual_seed(int(ss[1].generate_state(1)[0]))
        self.via = Viability(obs_dim, act_dim, rc.hidden, rc.H, disc=rc.kind == "bvs_disc", lr=rc.lr, gen=self.v_gen,
                             n_targ_uniform=rc.n_targ_uniform, gamma_b=rc.gamma_b, qb_max=rc.qb_max)
        self.crit_name = "Qb" if rc.kind == "bvs_disc" else "V"
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=rc.actor_lr or rc.lr)
        self.q_opt = torch.optim.Adam(self.members.parameters(), lr=rc.critic_lr or rc.lr)
        self.sq_critic = self.members[0]               # S-Q, the actor and every target use member 1 only
        self.rng = np.random.default_rng(seed)         # exploration noise (as LagU)
        self.shield_rng = np.random.default_rng(ss[0])  # training-time candidates
        self.replay_extras = {"a_exec": ((act_dim,), np.float32), "I": ((), np.uint8)}
        self.sample_keys = ("o", "b", "a_pi", "a_exec", "r", "c", "o2", "b2", "d", "I")
        self.env_step = 0
        self.n_updates = 0
        self.stats: dict = {}                          # last losses and qbar (log_row)
        # margin dual
        self.m_tilde = float(rc.m0)
        self.m = self._clip_m(self.m_tilde)
        self.margins: deque = deque(maxlen=rc.m_window)
        self.m_dep = self._clip_m(rc.m0)
        # BVS-U threshold
        self.T = float(rc.T0)
        self.u_window: deque = deque(maxlen=rc.u_window)
        self.counters = {k: (0.0 if k == "F_cum" else 0) for k in COUNTERS}
        self.reset_interval()
        self.begin_episode()

    def _clip_m(self, m: float) -> float:
        return float(np.clip(m, -self.d / 2, self.d))

    def q_input(self, x: np.ndarray, o: np.ndarray) -> np.ndarray:
        return x

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def propose(self, x: np.ndarray, explore: bool = False) -> np.ndarray:
        """a_pi = actor(x), plus N(0, action_noise^2) from self.rng when exploring, clipped (as LagU.act).
        Deterministic and drawing nothing when explore is False."""
        a = self.actor(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()
        if explore and self.rc.action_noise > 0:
            a = a + self.rng.normal(0.0, self.rc.action_noise, size=a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def select_action(self, x, o, b, C, step):
        """Hook 1: a_pi from the noisy proposer; the shield acts at steps > shield_start with the
        episode's acting margin and shield_rng."""
        a_pi = self.propose(x, explore=True)
        if step <= self.rc.shield_start:
            return a_pi, a_pi, 0, {}
        a_exec, I, info = self.act(o, b, C, a_pi, m=self.m, rng=self.shield_rng, mode="on")
        return a_pi, a_exec, I, info

    # ---------------------------------------------------------------- learning
    def learn(self, sample: dict, step: int) -> dict:
        """Hook 3: one update on one shared batch: Q_r, then V (Q_b); on every policy_delay-th update the
        actor, then pi_b, then the Polyak update of every target."""
        T = lambda k: torch.as_tensor(sample[k], dtype=torch.float32)
        o, o2 = T("o"), T("o2")
        x, x2 = torch.cat([o, T("b")], dim=-1), torch.cat([o2, T("b2")], dim=-1)
        c, d = T("c"), T("d")
        pi2 = self._critic_update(x, T("a_pi"), T("r"), c, x2, d, T("I"))
        self.stats["v_loss"] = self.via.update(o, T("a_exec"), c, o2, d, pi2)
        self.n_updates += 1
        if self.n_updates % self.rc.policy_delay == 0:
            self._actor_update(x)
            self.stats["pib_loss"] = self.via.update_pib(o)
            self._polyak()
        return self.stats

    def _critic_update(self, x, a, r, c, x2, d, I) -> torch.Tensor:
        """Reward critics on the relabelled reward r - kappa I, each member bootstrapping from its own
        twin targets with its own smoothing noise (torch global RNG), as LagU.update. Returns
        pi2 = actor_targ(x2)."""
        rc = self.rc
        with torch.no_grad():
            pi2 = self.actor_targ(x2)

            def smoothed():
                eps = (torch.randn_like(a) * rc.target_noise).clamp(-rc.target_clip, rc.target_clip)
                return (pi2 + eps).clamp(-1.0, 1.0)
            r_rel = r - rc.kappa * I
            targets = [r_rel + rc.gamma * (1.0 - d) * torch.min(*mt(x2, smoothed())) for mt in self.members_targ]
        member_losses = []
        for m, t in zip(self.members, targets):
            q1, q2 = m(x, a)
            member_losses.append(F.mse_loss(q1, t) + F.mse_loss(q2, t))
        q_loss = torch.stack(member_losses).sum()
        _step(self.q_opt, q_loss)
        self.stats["q_loss"] = float(q_loss) / len(self.members)
        return pi2

    def _actor_update(self, x: torch.Tensor) -> None:
        """Actor: maximise Q_r^1 of member 1 at (x, actor(x)); Adam step on the actor only. Logs qbar,
        the batch-mean min-twin Q_r at the pre-step actions (no RNG; members are not stepped here).

        The hook makes the gradient reaching the actor's output contiguous. In base.Agent that gradient
        is the sum of the Q_r path and the (zero, at lambda = 0) cost path, a fresh contiguous tensor;
        here it would be a strided slice of TwinQ's cat backward, which sends the actor's weight
        gradients through a different BLAS kernel (differences of ~1e-11). The values are unchanged;
        the hook is what makes SPEC test 13 bit-identical."""
        a = self.actor(x)
        if a.requires_grad:
            a.register_hook(lambda g: g.contiguous())
        loss = (-self.members[0](x, a)[0]).mean()
        _step(self.pi_opt, loss)
        with torch.no_grad():
            qbar = float(torch.min(*self.members[0](x, a.detach())).mean())
        self.stats.update(pi_loss=float(loss), qbar=qbar)

    def _polyak(self) -> None:
        _polyak(((self.actor, self.actor_targ), (self.members, self.members_targ)), self.rc.tau)
        self.via.polyak(self.rc.tau)

    # ------------------------------------------------------- per-step and episode
    def begin_episode(self) -> None:
        """The acting margin m = clip(m_tilde, -d/2, d) is fixed for the episode; per-episode trackers
        (the 12-step cost and I lookbacks, the F accumulator) restart."""
        self.m = self._clip_m(self.m_tilde)
        self.ep = {"costs": deque(maxlen=ENTRY_LOOKBACK), "Is": deque(maxlen=ENTRY_LOOKBACK), "F_on": False,
                   "F": 0.0}

    def on_step(self, step: int, t: int, c: float, I: int, info: dict) -> None:
        """Cumulative counters (steps > shield_start), the rho* counters (steps > 35,000), entries and
        misses, F, the interval accumulators and BVS-U's u window."""
        k, ep = self.counters, self.ep
        I = int(I)
        if step > self.rc.shield_start:
            k["shield_steps_cum"] += 1
            k["int_cum"] += I
            if info.get("empty"):
                k["empty_cum"] += 1
                ep["F_on"] = True
            if ep["F_on"]:
                ep["F"] += c
            self.iv_vsum += info["vmax"]
            self.iv_n += 1
            if self.kind == "bvs_u":
                k["trig_cum"] += int(info["triggered"])
                self.u_window.append(info["u"])
                self.iv_u.append(info["u"])
            if t >= ENTRY_LOOKBACK and c > 0 and not any(x > 0 for x in ep["costs"]):
                k["entries_cum"] += 1
                k["misses_cum"] += int(not any(ep["Is"]))
        if step > RHO_FROM:
            k["rho_den"] += 1
            k["rho_num"] += I
            if self.kind == "bvs_u":
                k["trig_num"] += int(info.get("triggered", False))
                if "u" in info:                     # counterfactual trigger at Lag-U's T0 = 0.07 (prereg 14.3)
                    k["trig007_num"] += int(info["u"] >= self.rc.T0)
        ep["costs"].append(c)
        ep["Is"].append(I)

    def end_episode(self, step: int, ep_ret: float, ep_cost: float) -> None:
        """Hook 4: margins.append(m); m_tilde = max(-d/2, m_tilde + eta_m (J_e - d)) if the episode ended
        after margin_start (projected: no windup below the acting margin's floor, prereg A.6);
        m_dep = mean(margins); F and ep counters for episodes ending after shield_start."""
        rc = self.rc
        self.margins.append(self.m)
        if step > rc.margin_start:
            self.m_tilde = max(-self.d / 2, self.m_tilde + rc.eta_m * (ep_cost - self.d))
        self.m_dep = float(np.mean(self.margins))
        if step > rc.shield_start:
            self.counters["ep_cum"] += 1
            self.counters["F_cum"] += self.ep["F"]

    def post_update(self, step: int, jc: deque) -> None:
        """BVS-U: T = quantile(u window, 1 - rho*) (linear) at step u_start and every u_every steps after;
        rho* = 0 gives +inf (never triggers), rho* = 1 gives -inf (always triggers)."""
        rc = self.rc
        if self.kind != "bvs_u" or step < rc.u_start or (step - rc.u_start) % rc.u_every:
            return
        if rc.rho_star == 0.0:
            self.T = float("inf")
        elif rc.rho_star == 1.0:
            self.T = float("-inf")
        elif self.u_window:
            self.T = float(np.quantile(np.asarray(self.u_window), 1.0 - rc.rho_star))

    # ------------------------------------------------------------------ logging
    def log_row(self) -> dict:
        s = self.stats
        row = {key: s.get(key, "") for key in ("qbar", "q_loss", "pi_loss", "v_loss", "pib_loss")}
        row["vH_exec_mean"] = self.iv_vsum / self.iv_n if self.iv_n else ""
        row.update(m_tilde=self.m_tilde, m=self.m, m_dep=self.m_dep)
        row.update({k: v for k, v in self.counters.items() if self.kind == "bvs_u" or k not in U_ONLY})
        if self.kind == "bvs_u":
            row.update(T=self.T, u_p50=float(np.median(self.iv_u)) if self.iv_u else "")
        return row

    def eval_columns(self, episodes: list) -> dict:
        return {"eval_int_steps": int(sum(e["interventions"] for e in episodes)),
                "eval_empty_steps": int(sum(e["empty_steps"] for e in episodes))}

    def reset_interval(self) -> None:
        self.iv_vsum, self.iv_n = 0.0, 0     # vH_exec_mean
        self.iv_u: list = []                 # u_p50 (BVS-U)

    def console_dual(self) -> str:
        return f"m={self.m:.3g}"

    # ----------------------------------------------------------- snapshots
    def snapshot(self) -> dict:
        v = self.via
        snap = {"actor": self.actor.state_dict(), "members": self.members.state_dict(),
                self.crit_name: v.crit.state_dict(), "pib": v.pib.state_dict(), "m_dep": self.m_dep}
        if self.kind == "bvs_u":
            snap["T"] = self.T
        return snap

    @classmethod
    def from_snapshot(cls, snap: dict) -> "BVS":
        """Rebuild a deterministic agent (deployed margin and T frozen at the snapshot's values) without
        touching the caller's torch RNG."""
        rc = RunConfig(**snap["config"])
        with torch.random.fork_rng():
            agent = cls(snap["obs_dim"], snap["act_dim"], rc, snap["horizon"])
        agent.actor.load_state_dict(snap["actor"])
        agent.members.load_state_dict(snap["members"])
        agent.via.crit.load_state_dict(snap[agent.crit_name])
        agent.via.pib.load_state_dict(snap["pib"])
        agent.m_dep = float(snap["m_dep"])
        if rc.kind == "bvs_u":
            agent.T = float(snap["T"])
        return agent

    def make_policy(self, mode: str):
        from safe2.evaluate import ShieldPolicy
        return ShieldPolicy(self, mode)

    # --------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        v, n = self.via, self.crit_name
        return {"actor": self.actor.state_dict(), "actor_targ": self.actor_targ.state_dict(),
                "members": self.members.state_dict(), "members_targ": self.members_targ.state_dict(),
                n: v.crit.state_dict(), n + "_targ": v.crit_targ.state_dict(),
                "pib": v.pib.state_dict(), "pib_targ": v.pib_targ.state_dict(),
                "pi_opt": self.pi_opt.state_dict(), "q_opt": self.q_opt.state_dict(),
                "v_opt": v.opt.state_dict(), "pib_opt": v.pib_opt.state_dict(),
                "rng": self.rng.bit_generator.state, "shield_rng": self.shield_rng.bit_generator.state,
                "v_gen": self.v_gen.get_state(), "n_updates": self.n_updates, "stats": dict(self.stats),
                "m_tilde": self.m_tilde, "m": self.m, "margins": list(self.margins), "m_dep": self.m_dep,
                "T": self.T, "u_window": torch.tensor(list(self.u_window), dtype=torch.float64),
                "counters": dict(self.counters),
                "interval": {"vsum": self.iv_vsum, "n": self.iv_n,
                             "u": torch.tensor(self.iv_u, dtype=torch.float64)},
                "run_config": asdict(self.rc)}

    def load_state_dict(self, s: dict) -> None:
        v, n = self.via, self.crit_name
        for name, net in (("actor", self.actor), ("actor_targ", self.actor_targ), ("members", self.members),
                          ("members_targ", self.members_targ), (n, v.crit), (n + "_targ", v.crit_targ),
                          ("pib", v.pib), ("pib_targ", v.pib_targ), ("pi_opt", self.pi_opt), ("q_opt", self.q_opt),
                          ("v_opt", v.opt), ("pib_opt", v.pib_opt)):
            net.load_state_dict(s[name])
        self.rng.bit_generator.state = s["rng"]
        self.shield_rng.bit_generator.state = s["shield_rng"]
        self.v_gen.set_state(s["v_gen"])
        self.n_updates, self.stats = int(s["n_updates"]), dict(s["stats"])
        self.m_tilde, self.m, self.m_dep = float(s["m_tilde"]), float(s["m"]), float(s["m_dep"])
        self.margins = deque(s["margins"], maxlen=self.rc.m_window)
        self.T = float(s["T"])
        self.u_window = deque(s["u_window"].tolist(), maxlen=self.rc.u_window)
        self.counters = dict(s["counters"])
        self.iv_vsum, self.iv_n = float(s["interval"]["vsum"]), int(s["interval"]["n"])
        self.iv_u = s["interval"]["u"].tolist()


# --- the post-hoc shield (PHS) ----------------------------------------------------------------------
class PHSCore(_Shield):
    """The ShieldPolicy core of PHS (prereg A.9): the SPEC 2.4 shield of BVS (kind "bvs") around B*'s
    deterministic actor, at evaluation only.

    Proposer: the B* snapshot's actor, fed cat(o, b) when B* has the budget input (Saute) and o
    otherwise. S-Q critic: the snapshot's members[0] twin-min, on the same input. V and pi_b: the run's
    offline fit (fit_offline). m_dep = m_PHS(task). Rule: BVS*'s rule. rc is B*'s RunConfig with that
    rule and the fit's H; the shield's other settings (eps, sigma_c, candidate counts, d) are the
    RunConfig defaults, which every BVS config keeps."""

    kind = "bvs"

    def __init__(self, bstar_snapshot: dict, fit: dict, m_phs: float, rule: str):
        from safe2.base import Agent
        if rule not in ("SQ", "SN"):
            raise ValueError(f"unknown rule {rule!r}")
        base = Agent.from_snapshot(bstar_snapshot)
        if (fit["obs_dim"], fit["act_dim"]) != (base.obs_dim, base.act_dim):
            raise ValueError("the PHS fit and the B* snapshot have different dimensions")
        self.rc = replace(base.rc, rule=rule, H=int(fit["H"]))
        self.act_dim = base.act_dim
        self.budget_input = bool(base.rc.budget_input)
        self.actor, self.sq_critic = base.actor, base.members[0]
        with torch.random.fork_rng():
            self.via = Viability(fit["obs_dim"], fit["act_dim"], fit["hidden"], int(fit["H"]))
        self.via.crit.load_state_dict(fit["V"])
        self.via.pib.load_state_dict(fit["pib"])
        self.m_dep = float(m_phs)

    def q_input(self, x: np.ndarray, o: np.ndarray) -> np.ndarray:
        return x if self.budget_input else o

    @torch.no_grad()
    def propose(self, x: np.ndarray, explore: bool = False) -> np.ndarray:
        """B*'s deterministic action (BasePolicy arithmetic); x = cat(o, b) from ShieldPolicy."""
        if explore:
            raise ValueError("PHS is an evaluation-only shield")
        inp = x if self.budget_input else x[:-1]
        a = self.actor(torch.as_tensor(inp, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)


def fit_offline(data: dict, proposer: nn.Module, budget_input: bool, seed: int, updates: int = 100_000,
                batch: int = 256, hidden: int = 256, H: int = 32, lr: float = 3e-4, tau: float = 0.005,
                policy_delay: int = 2, n_targ_uniform: int = 4, keep_last: int = 1000) -> dict:
    """SPEC 7 fit: V (twin, H heads) and pi_b trained exactly as SPEC 2.4 steps 2-3 (class Viability)
    on fixed replay rows, for `updates` updates of `batch` rows; pi_b and the Polyak update of both
    targets on every policy_delay-th update.

    data: numpy arrays of the selected rows, "o", "a" (the executed action), "c", "o2", "b2", "d"
    (true termination). proposer: the run's own actor, frozen and deterministic; it takes cat(o2, b2)
    with budget_input and o2 otherwise, and stands where pi2 stands in the target candidates.
    Seeding: torch.manual_seed(seed) before the networks are built, batch indices from
    default_rng(seed), the target-candidate generator seeded with seed.

    Returns {"V", "pib" (state dicts), "obs_dim", "act_dim", "hidden", "H", "updates", "batch", "seed",
    "losses": {"v": [...], "pib": [...]} (the last keep_last of each)}. posthoc.py adds the metadata
    (checkpoint sha256, episodes)."""
    n, obs_dim = data["o"].shape
    act_dim = data["a"].shape[1]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    via = Viability(obs_dim, act_dim, hidden, H, lr=lr, gen=torch.Generator().manual_seed(seed),
                    n_targ_uniform=n_targ_uniform)
    proposer = copy.deepcopy(proposer).requires_grad_(False)
    v_losses, b_losses = deque(maxlen=keep_last), deque(maxlen=keep_last)
    for k in range(1, updates + 1):
        idx = rng.integers(0, n, batch)
        T = lambda key: torch.as_tensor(data[key][idx], dtype=torch.float32)
        o, a, c, o2, d = T("o"), T("a"), T("c"), T("o2"), T("d")
        with torch.no_grad():
            pi2 = proposer(torch.cat([o2, T("b2")], dim=-1) if budget_input else o2)
        v_losses.append(via.update(o, a, c, o2, d, pi2))
        if k % policy_delay == 0:
            b_losses.append(via.update_pib(o))
            via.polyak(tau)
    return {"V": via.crit.state_dict(), "pib": via.pib.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim,
            "hidden": hidden, "H": H, "updates": updates, "batch": batch, "seed": seed,
            "losses": {"v": list(v_losses), "pib": list(b_losses)}}
