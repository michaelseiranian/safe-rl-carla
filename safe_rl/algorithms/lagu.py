# safe-rl-carla\safe_rl\algorithms\lagu.py:
import torch, torch.nn as nn, torch.optim as optim
from safe_rl.utils.replay_buffer import ReplayBuffer
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
from typing import Optional
import warnings
import wandb
from safe_rl.env.carla_cmdp_env import MAX_ACTORS, ACTOR_FEATURES

# ---------------------------------------------------------------------------+
# Speed normalisation helper                                                 +
# (0 … 1.5) ↦ (‑1 … 1)                                                       +
# ---------------------------------------------------------------------------+
def _norm_speed(speed: torch.Tensor) -> torch.Tensor:
    """
    Speed is already normalized to [0, 2] in the env. Center at 1.0.
    """
    return torch.clamp(speed - 1.0, -1.0, 1.0)


def mlp(in_dim, out_dim, hidden=(256,256), act=nn.ReLU, out_act=nn.Identity):
    # smaller net = faster + less over-fit for 53-D input
    hidden=(128,128)
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim), out_act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, act_dim, max_action,
                 vec_dim=4,  speed_dim=1,
                 actor_flat_dim=MAX_ACTORS*ACTOR_FEATURES):
        super().__init__()
        in_dim = speed_dim + vec_dim + actor_flat_dim
        # vec_dim=4 supports (cte, heading, tl_flag, tl_dist) later (PlanT-style context).
        self.head = mlp(in_dim, act_dim)
        # Learned gate applied when actor tokens are all zeros (leaderboard eval)
        self.no_actor_gate = nn.Parameter(torch.zeros(in_dim))

        # Tiny forward bias on accel head to encourage motion during early exploration
        # (steer bias 0.0, accel bias +0.15)
        try:
            with torch.no_grad():
                b = getattr(self.head[-2], "bias", None)  # final Linear before out_act
                if b is not None and b.shape[0] >= 2:
                    b[:2] = torch.tensor([0.0, 0.15], device=b.device, dtype=b.dtype)
        except Exception:
            pass

        self.max_action = max_action

    def forward(self, speed, vec, actors_flat):
        z = torch.cat([_norm_speed(speed), vec, actors_flat], dim=-1)
        # If actor input is (near) all-zeros for a sample, mix in a small learned bias
        with torch.no_grad():
            is_zero = (actors_flat.abs().sum(dim=-1, keepdim=True) < 1e-6).float()
        z = z + is_zero * self.no_actor_gate
        return torch.tanh(self.head(z)) * self.max_action

class CriticEnsemble(nn.Module):
    def __init__(self, act_dim, M=3,
                 vec_dim=4, speed_dim=1,
                 actor_flat_dim=MAX_ACTORS*ACTOR_FEATURES):
        super().__init__()
        self.M = M
        in_dim = speed_dim + vec_dim + actor_flat_dim + act_dim
        self.Qs = nn.ModuleList([mlp(in_dim, 1) for _ in range(M)])

    def forward(self, speed, vec, actors_flat, a):
        za = torch.cat([_norm_speed(speed), vec, actors_flat, a], dim=-1)
        outs = torch.stack([Q(za) for Q in self.Qs], dim=0)   # (M,B,1)
        mean = outs.mean(0).squeeze(-1)                       # (B,)
        if outs.shape[0] < 2:
            var = torch.zeros_like(mean)
        else:
            # ε keeps √Var finite when Q gets large
            var = outs.var(0, unbiased=False).squeeze(-1) + 1e-8
        return mean, var

class CostCritic(nn.Module):
    def __init__(self, act_dim,
                 vec_dim=4, speed_dim=1, actor_flat_dim=MAX_ACTORS*ACTOR_FEATURES):
        super().__init__()
        in_dim = speed_dim + vec_dim + actor_flat_dim + act_dim
        self.Qc = mlp(in_dim, 1)

    def forward(self, speed, vec, actors_flat, a):
        return self.Qc(torch.cat([_norm_speed(speed), vec, actors_flat, a], dim=-1)).squeeze(-1)

# --------------------------------------------------------------------- #

@dataclass
class LagUHyper:
    gamma: float = 0.99
    tau: float = 0.01

    # Which baseline to run: "lagu" (uncertainty + adaptive budget),
    # "lag" (Lagrangian with fixed budget), or "td3" (reward-only).
    baseline: str = "lagu"

    # --- safer learning‑rates --------------------------------------------------
    #  • actor: *much* smaller – avoids wiping the throttle bias in a few steps
    #  • critic: standard TD3 setting
    lr_actor:   float = 5e-5
    lr_critic:  float = 3e-4
    batch:      int = 256
    M:          int = 3     # lighter ensemble; faster, less variance in targets

    # === Safe‑RL parameters (paper notation) ===
    # Tighter budget now that we do not down‑scale cost twice
    cost_limit_init: float = 0.05          # δ₀ per-step – tighter so λ can move with episodic one-shot costs
    c0:  float = 0.20                       # lower‑bound exploration probability (Eq. 13) :contentReference[oaicite:1]{index=1}
    Tc:  int   = 50_000                    # exploration stop threshold  (Eq. 13)
    T_thresh: float = 3.0                   # tighter uncertainty ratio threshold T (Eq. 16)

    # === misc ===
    device: str = "cuda"
    policy_delay: int = 2
    target_noise: float = 0.05
    target_noise_clip: float = 0.25
    cost_target_noise: float = 0.05
    cost_target_noise_clip: float = 0.10

    # λ update step—much smaller than before (paper uses 1e‑5 without √t scaling)
    lambda_alpha0: float = 5e-4            # calmer λ steps; grows ~1/√t
    # Optional per-stage ramp for λ step size:
    lambda_alpha0_ramp_after: int = 0      # policy updates after which to change alpha0
    lambda_alpha0_after: Optional[float] = None      # new alpha0 after ramp (None → no change)
    # Widen Δc clipping later in curriculum if desired:
    delta_clip_mult: float = 1.0           # tighter → Δc affects λ and π more
    seed: int = 0                    # 0 → keep global RNG (non‑deterministic)
    lambda_clip:   float = 10.0

    # scales √Var exploration bonus
    # bonus scaled down because replay targets are ×5 larger
    bonus_coef: float = 0.02      # retuned for unscaled replay
    # --- evaluation safety guard (item 9) ---
    enable_eval_guard: bool = True
    safe_guard_ratio_thresh: float = 1.2    # trip if std is big *relative* to |Q|
    safe_guard_abs_sqrtvar: float = 300.0   # raw std cap; avoids always-on braking

    # Exploration & robustness
    #random_steps: int = 800              # pure random actions at the start (env steps)
    actor_dropout_p: float = 0.5            # prob. to zero all actor tokens per sample

    # ── NEW: structural stabilizers ─────────────────────────────────────────
    # Use episodic λ updates (recommended). If False, minibatch λ updates remain
    # but are gated by movement and warmed-up Q scale.
    episodic_lambda: bool = True
    # Warm-up for adaptive budget: until policy has moved & Q scale is sane,
    # keep δ fixed to δ0 rather than shrinking adaptively.
    budget_adapt_warmup_pi_steps: int = 2_000
    # Floor for |Q| when forming ratio √Var / denom (prevents denom≈0 early).
    ratio_q_floor: float = 5.0
    # Gate λ updates unless mean batch speed exceeds this (avoid "park is safe").
    # Use a lower threshold so early-stage speeds (≈ 6–12 km/h → ~0.07–0.13 norm) can still update.
    lambda_update_min_speed: float = 0.08
    # Cap per-sample λ·Δc contribution inside policy loss to avoid single hot samples
    # from dominating gradients globally.
    lambda_term_cap: float = 0.4

    # Training warmup & early exploration
    warmup_steps: int = 10_000
    random_steps: int = 3_000

class LagUAgent:
    def __init__(self, obs_shape, act_dim, max_action, cfg:LagUHyper,
                 *, vec_dim:int = 4, replay_size:int = 200000, replay_scale:float = 0.1):
        self.cfg = cfg
        self.vec_dim = vec_dim
        self.actor = Actor(act_dim, max_action, vec_dim=vec_dim).to(cfg.device)
        self.critic = CriticEnsemble(act_dim, cfg.M, vec_dim=vec_dim).to(cfg.device)
        self.critic_targ = CriticEnsemble(act_dim, cfg.M, vec_dim=vec_dim).to(cfg.device)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        self.cost_critic = CostCritic(act_dim, vec_dim=vec_dim).to(cfg.device)
        self.cost_targ   = CostCritic(act_dim, vec_dim=vec_dim).to(cfg.device)
        self.cost_targ.load_state_dict(self.cost_critic.state_dict())

        self.pi_opt  = optim.Adam(self.actor.parameters(),  cfg.lr_actor)
        self.Q_opt   = optim.Adam(self.critic.parameters(), cfg.lr_critic)
        self.Qc_opt  = optim.Adam(self.cost_critic.parameters(), cfg.lr_critic)

        self.replay = ReplayBuffer(obs_shape, act_dim, size=replay_size,
                                   device=cfg.device, scale=replay_scale, vec_dim=vec_dim)
        self.lmbd   = torch.tensor([0.0], device=cfg.device, requires_grad=False)
        # Running EMAs for stabilization
        self.q_abs_ema     = torch.tensor([0.0], device=self.cfg.device)  # scale of |Q|
        self.ema_cost_gap  = torch.tensor([0.0], device=self.cfg.device)  # episodic cost gap EMA

        # reproducible stochasticity ---------------------------------------
        self.rng = torch.Generator(device=cfg.device)
        if cfg.seed:
            self.rng.manual_seed(cfg.seed)
            # also seed Python, NumPy, and torch globals
            import random
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

        self.step: int = 0 # counts gradient updates
        self.pi_step: int = 0 # counts policy updates (after delay)
        self.global_step: int = 0 # counts total environment steps
        # stage-relative counters for schedules (set/reset by trainer)
        self._stage_step0: int = 0
        self._eps_boost_steps: int = 10_000
        # NEW: per-stage wide random exploration period (env steps, relative to _stage_step0)
        self._stage_random_steps: int = 0

        # NEW: temporary learning-rate boost window set by the trainer after regime changes.
        # Decays inside train() and resets LR back to cfg.lr_actor when it reaches 0.
        self._lr_boost_steps: int = 0

        # ---- anti-stuck helpers (training-time only) -----------------------
        # Count consecutive near-zero-speed observations; when high, kick accel.
        self._zero_speed_streak: int = 0
        # When >0, enforce a minimum forward accel for a few steps.
        self._stuck_kick_remain: int = 0



    # --------------- interaction helpers ---------------- #

    def act(self, obs_spd, obs_vec, obs_actors, explore=True, safe_guard=True):
        with torch.no_grad():
            # local function to manage "stuck kick" schedule using *normalized* speed
            def _stuck_kick_should_fire(spd_norm: float) -> bool:
                # spd_norm≈0.05 ~ 4.5 kph (since speed_norm ≈ kph/90)
                threshold = 0.05
                if spd_norm < threshold:
                    self._zero_speed_streak += 1
                else:
                    self._zero_speed_streak = 0
                    self._stuck_kick_remain = 0
                # Arm the kick if we've been "stuck" for a short streak
                if self._stuck_kick_remain > 0:
                    self._stuck_kick_remain -= 1
                    return True
                if self._zero_speed_streak >= 10:
                    # Kick for the next ~20 control ticks
                    self._stuck_kick_remain = 20
                    self._zero_speed_streak = 0  # reset the detector
                    return True
                return False

            spd_norm_scalar = float(obs_spd)


            spd  = torch.tensor([[float(obs_spd)]], dtype=torch.float32, device=self.cfg.device)
            vec  = torch.as_tensor(obs_vec,    dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            actr = torch.as_tensor(obs_actors, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            a_t  = self.actor(spd, vec, actr)
            a = a_t.cpu().numpy()[0]
        if explore:
            # 1) Early: pure random with mild steering & forward bias
            if (self.global_step < self.cfg.random_steps) or (self.step < self.cfg.warmup_steps):
                rand_a = np.empty_like(a)
                rand_a[0] = np.random.uniform(-0.3, 0.3)   # gentler steer
                rand_a[1] = np.random.uniform(0.2, 0.5)    # consistent forward
                if _stuck_kick_should_fire(spd_norm_scalar):
                    rand_a[1] = max(rand_a[1], 0.30)
                return rand_a

            # --- NEW: wide pure-random window right after a stage switch ---
            try:
                t_rel = max(0, self.global_step - int(getattr(self, "_stage_step0", 0)))
                if t_rel < int(getattr(self, "_stage_random_steps", 0)):
                    rand_a = np.empty_like(a)
                    rand_a[0] = np.random.uniform(-0.3, 0.3)
                    rand_a[1] = np.random.uniform(0.2, 0.5)
                    if _stuck_kick_should_fire(spd_norm_scalar):
                        rand_a[1] = max(rand_a[1], 0.30)
                    return rand_a
            except Exception:
                pass

            # 2) Epsilon-greedy + moderate policy noise thereafter (stage-local if provided)
            t_rel = max(0, self.global_step - int(getattr(self, "_stage_step0", 0)))
            eps0 = float(getattr(self, "_eps_start", 0.35))
            decay = int(getattr(self, "_eps_decay_steps", 0))
            if decay > 0:
                eps = max(0.1, eps0 * max(0.0, 1.0 - t_rel / float(decay)))
            else:
                eps = max(0.1, 1.0 - max(0, self.global_step - self.cfg.random_steps) / 20000.0)
            if np.random.rand() < eps:
                # Random action with forward bias
                rand_a = np.empty_like(a)
                rand_a[0] = np.random.uniform(-0.5, 0.5)
                rand_a[1] = np.random.uniform(0.0, 0.7)
                if _stuck_kick_should_fire(spd_norm_scalar):
                    rand_a[1] = max(rand_a[1], 0.30)
                return rand_a
            else:
                a += np.random.normal(0, [0.1, 0.1])
                a = np.clip(a, -1.0, 1.0)
                if _stuck_kick_should_fire(spd_norm_scalar):
                    a[1] = max(a[1], 0.30)
                    a = np.clip(a, -1.0, 1.0)

        else:
            # --- evaluation-time safety guard (hard brake on high uncertainty) ---
            if safe_guard and self.cfg.enable_eval_guard:
                Qa, Va = self.critic(spd, vec, actr, a_t)
                std = torch.sqrt(Va + 1e-8)
                # Use stabilized |Q| scale like in training
                denom = max(float(self.q_abs_ema.item()), float(self.cfg.ratio_q_floor))
                ratio = (std / denom).item()
                if ratio >= self.cfg.safe_guard_ratio_thresh \
                   or std.item() >= self.cfg.safe_guard_abs_sqrtvar:
                    a[1] = -1.0   # accel full negative → throttle=0, brake=1
        return a

    def store(self, s_spd, s_vec, s_actr, a, r, c, s2_spd, s2_vec, s2_actr, d):
        self.replay.store(s_spd, s_vec, s_actr, a, r, c, s2_spd, s2_vec, s2_actr, d)
        # Reset anti-stuck helpers at episode boundaries
        try:
            if bool(d):
                self._zero_speed_streak = 0
                self._stuck_kick_remain = 0
        except Exception:
            pass

    # --------------- training step ---------------- #

    def train(self, log:bool=False):
        if self.replay.size < self.cfg.batch:
            return

        self.step += 1
        s_spd, s_vec, s_actr, a, r, c, s2_spd, s2_vec, s2_actr, d = self.replay.sample(self.cfg.batch)
        # --- actor-token dropout to mitigate train/eval domain shift (no actors at eval) ---
        if self.cfg.actor_dropout_p > 0.0:
            m = (torch.rand((s_actr.shape[0], 1), device=s_actr.device) < self.cfg.actor_dropout_p)
            s_actr  = torch.where(m, torch.zeros_like(s_actr),  s_actr)
            s2_actr = torch.where(m, torch.zeros_like(s2_actr), s2_actr)


        # ----- target actions with smoothing noise (TD3 style) -----
        with torch.no_grad():
            # ── reward branch (TD3 smoothing + min-Q target across ensemble) ──
            pi_targ_r = self.actor(s2_spd, s2_vec, s2_actr)
            if self.cfg.target_noise > 0.0:
                noise_r = (torch.randn_like(pi_targ_r) * self.cfg.target_noise).clamp(
                    -self.cfg.target_noise_clip, self.cfg.target_noise_clip)
                pi_targ_r = (pi_targ_r + noise_r).clamp(-1.0, 1.0)
            # Build concatenated input for each target head
            za_r = torch.cat([_norm_speed(s2_spd), s2_vec, s2_actr, pi_targ_r], dim=-1)
            Q2_all = torch.stack([Q(za_r) for Q in self.critic_targ.Qs], dim=0).squeeze(-1)  # (M,B)
            Q2_min = Q2_all.min(dim=0).values  # (B,)
            target_Q   = r + self.cfg.gamma * (1.0 - d) * Q2_min
            target_Q   = target_Q.clamp(-200.0, 200.0)

            # cost branch (optionally *less* aggressive smoothing)
            pi_targ_c = self.actor(s2_spd, s2_vec, s2_actr)
            if self.cfg.cost_target_noise > 0.0:
                noise_c = (torch.randn_like(pi_targ_c) * self.cfg.cost_target_noise).clamp(
                    -self.cfg.cost_target_noise_clip, self.cfg.cost_target_noise_clip)
                pi_targ_c = (pi_targ_c + noise_c).clamp(-1.0, 1.0)
            Q2c = self.cost_targ(s2_spd, s2_vec, s2_actr, pi_targ_c)
            target_Qc = (c + self.cfg.gamma * (1.0 - d) * Q2c).clamp(-200.0, 200.0)

        # ----- reward critic update -----
        Q_mean, Q_var = self.critic(s_spd, s_vec, s_actr, a)
        critic_loss = F.smooth_l1_loss(Q_mean, target_Q)
        self.Q_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.Q_opt.step()

        # ----- cost critic update -----
        Qc = self.cost_critic(s_spd, s_vec, s_actr, a)
        cost_loss = F.smooth_l1_loss(Qc, target_Qc)
        self.Qc_opt.zero_grad()
        cost_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), 1.0)
        self.Qc_opt.step()

        # -------- critics‑only warm‑up (Alg. 1, line 5) --------
        WARMUP_STEPS = int(getattr(self.cfg, "warmup_steps", 2000))
        if self.step < WARMUP_STEPS:
            # Keep targets tracking the online nets during warm-up
            with torch.no_grad():
                for p, p_targ in zip(self.critic.parameters(), self.critic_targ.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
                for p, p_targ in zip(self.cost_critic.parameters(), self.cost_targ.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            return

        # ----- delayed policy + λ update -----
        if self.step >= WARMUP_STEPS and self.step % self.cfg.policy_delay == 0:
            # ── Stage-local actor freeze: allow critics to adapt before moving π ──
            # Decrement a per-stage counter; if still positive, skip the actor update.
            pending = int(getattr(self, "_stage_actor_freeze_updates", 0))
            if pending > 0:
                self._stage_actor_freeze_updates = pending - 1
                # Keep target networks tracking while actor is frozen
                with torch.no_grad():
                    for p, p_targ in zip(self.critic.parameters(), self.critic_targ.parameters()):
                        p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
                    for p, p_targ in zip(self.cost_critic.parameters(), self.cost_targ.parameters()):
                        p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
                return

            self.pi_step += 1
            pi_actions = self.actor(s_spd, s_vec, s_actr)
            Q_pi, var_pi = self.critic(s_spd, s_vec, s_actr, pi_actions)
            cost_est     = self.cost_critic(s_spd, s_vec, s_actr, pi_actions)

            # ===== Baseline gating =====
            mode = (self.cfg.baseline or "lagu").lower()
            use_lagrangian        = mode in ("lagu", "lag")
            use_uncertainty_bonus = (mode == "lagu")
            use_adaptive_budget   = (mode == "lagu")
            # Keep a running scale of |Q| to stabilize ratio during early training
            with torch.no_grad():
                self.q_abs_ema.mul_(0.99).add_(0.01 * Q_pi.abs().mean())
            # Exploration probability (unchanged)
            explore_prob = max(self.cfg.c0, 1.0 - self.pi_step / float(self.cfg.Tc))

            # Compute readiness for adaptive-budget once
            adapt_ready = (self.pi_step >= self.cfg.budget_adapt_warmup_pi_steps) and \
                          (self.q_abs_ema.item() >= self.cfg.ratio_q_floor)

            # === Exploratory or Risk-sensitive branch ===
            if torch.rand((), generator=self.rng, device=s_spd.device) < explore_prob:
                # Exploratory: optional +√Var bonus (LagU only)
                if use_uncertainty_bonus:
                    bonus = torch.sqrt(var_pi.clamp(max=30.0) + 1e-8)
                    obj   = Q_pi + self.cfg.bonus_coef * bonus
                else:
                    obj   = Q_pi
                budget = torch.tensor(self.cfg.cost_limit_init, device=Q_pi.device)
            else:
                # Risk-sensitive: optional adaptive budget (LagU only)
                if use_adaptive_budget and adapt_ready:
                    denom = torch.clamp(self.q_abs_ema.detach(), min=self.cfg.ratio_q_floor)
                    ratio = torch.sqrt(var_pi + 1e-8) / denom
                    budget = torch.where(
                        ratio < self.cfg.T_thresh,
                        torch.tensor(self.cfg.cost_limit_init, device=ratio.device),
                        self.cfg.cost_limit_init * self.cfg.T_thresh / (ratio + 1e-8)
                    )
                else:
                    budget = torch.tensor(self.cfg.cost_limit_init, device=Q_pi.device)
                obj = Q_pi  # no bonus in risk phase

            # Always define Δc
            delta_c = cost_est - budget

            # Clip Δc unless truly in adaptive-budget risk phase
            if not (use_adaptive_budget and adapt_ready):
                delta_c = delta_c.clamp(-budget, +budget)


            # -------- safety-clip Δc to avoid λ explosions ------------
            clip = budget * getattr(self.cfg, "delta_clip_mult", 1.0)
            delta_c_clipped = delta_c.clamp(-clip, +clip)

            # TD3 baseline: no λ term; Lagrangian variants include it.
            lambda_term = (self.lmbd * delta_c_clipped) if use_lagrangian else 0.0
            # If the batch isn't moving, skip λ pressure in the actor loss to avoid "park is safe"
            if isinstance(lambda_term, torch.Tensor):
                with torch.no_grad():
                    moving = (s_spd.mean() > self.cfg.lambda_update_min_speed) | (s2_spd.mean() > self.cfg.lambda_update_min_speed)
                if not bool(moving):
                    lambda_term = 0.0
            # Local cap to avoid a single hot sample dominating the batch (symmetric)
            if isinstance(lambda_term, torch.Tensor):
                cap = float(self.cfg.lambda_term_cap)
                lambda_term = lambda_term.clamp(min=-cap, max=cap)
            policy_loss = -(obj - lambda_term).mean()
            # ---- NEW: keep actor away from tanh saturation (prevents dead policy at ±1) ----
            # simple action L2 regularizer; tiny weight
            policy_loss = policy_loss + 1e-3 * (pi_actions.pow(2).mean())
            # Regularize the no-actor input bias so it doesn't run away
            try:
                policy_loss = policy_loss + 1e-4 * (self.actor.no_actor_gate.pow(2).mean())
            except Exception:
                pass
            self.pi_opt.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.pi_opt.step()

            # --- NEW: temporary actor LR boost decay back to cfg.lr_actor ---
            try:
                if getattr(self, "_lr_boost_steps", 0) > 0 and self.pi_step % 10 == 0:
                    self._lr_boost_steps -= 10
                    if self._lr_boost_steps <= 0:
                        for g in self.pi_opt.param_groups:
                            g["lr"] = self.cfg.lr_actor
            except Exception:
                pass

            # ────────────────── extra monitoring ──────────────────
            try:
                wandb.log({
                    "safety/lambda": self.lmbd.item(),
                    "explore_prob":  explore_prob,
                    "actor/steer":   pi_actions[:, 0].mean(),
                    "actor/accel":pi_actions[:, 1].mean(),
                    "uncertainty/mean_var": var_pi.mean(),
                    "uncertainty/mean_std": torch.sqrt(var_pi + 1e-8).mean(),
                })
            except Exception as e:
                print("[LagU] wandb.log(extra) failed:", e, flush=True)

            # ---- λ update (minibatch) → optional & gated; prefer episodic in trainer ----
            lambda_delta = torch.tensor(0.0, device=self.lmbd.device)
            if use_lagrangian and not self.cfg.episodic_lambda:
                # Gate by movement to avoid "park is safe" fixed point
                moving = (s_spd.mean() > self.cfg.lambda_update_min_speed) | (s2_spd.mean() > self.cfg.lambda_update_min_speed)
                if bool(moving):
                    alpha0 = self.cfg.lambda_alpha0
                    ramp_after = int(getattr(self.cfg, "lambda_alpha0_ramp_after", 0) or 0)
                    alpha_after = getattr(self.cfg, "lambda_alpha0_after", None)
                    if alpha_after is not None and self.pi_step > ramp_after:
                        alpha0 = float(alpha_after)
                    alpha = alpha0 / np.sqrt(max(1, self.pi_step))
                    with torch.no_grad():
                        lambda_delta = alpha * delta_c_clipped.mean().detach()
                        self.lmbd.add_(lambda_delta)
                        self.lmbd.clamp_(0.0, self.cfg.lambda_clip)

            # soft target update
            with torch.no_grad():
                for p, p_targ in zip(self.critic.parameters(), self.critic_targ.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
                for p, p_targ in zip(self.cost_critic.parameters(), self.cost_targ.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

            if log:
                try:
                    den = float(max(self.q_abs_ema.detach().item(), self.cfg.ratio_q_floor))
                    guard_rate = float((
                        (torch.sqrt(var_pi + 1e-8) >= self.cfg.safe_guard_abs_sqrtvar) |
                        ((torch.sqrt(var_pi + 1e-8) / den) >= self.cfg.safe_guard_ratio_thresh)
                    ).float().mean().item())
                    wandb.log({
                        "loss/critic":     critic_loss.item(),
                        "loss/cost":       cost_loss.item(),
                        "loss/pi":         policy_loss.item(),
                        "safety/lambda":   self.lmbd.item(),
                        # use a scalar:
                        "safety/delta_cost": delta_c.mean().item(),
                        "safety/delta_cost_clipped": delta_c_clipped.mean().item(),
                        "uncertainty/mean_var_raw":  var_pi.mean().item(),
                        "uncertainty/mean_std_raw":  torch.sqrt(var_pi + 1e-8).mean().item(),
                        "uncertainty/mean_std_clamped": torch.sqrt(var_pi.clamp(max=10.0) + 1e-8).mean().item(),
                        "Q/mean_abs":      Q_pi.abs().mean().item(),
                        "Qc/mean_abs":     cost_est.abs().mean().item(),
                        "penalty/lambda_delta": float(lambda_delta.detach().cpu()),
                        # Match eval guard denominator (q_abs_ema with floor)
                        "guard/trigger_rate": guard_rate,
                    })
                except Exception as e:
                    print("[LagU] wandb.log failed:", e, flush=True)

    # -------------------------------------------------- #
    #  Checkpoint helpers
    # -------------------------------------------------- #
    def save_models(self, path: str):
        """
        Save actor, reward-critic, and cost-critic weights.
        """
        torch.save({
            "meta"       : {
                "baseline": getattr(self.cfg, "baseline", "lagu"),
                # persist input shape expectations for safer eval
                "vec_dim": int(getattr(self, "vec_dim", 4)),
                "actor_flat_dim": int(MAX_ACTORS * ACTOR_FEATURES),
            },
            "actor"       : self.actor.state_dict(),
            "critic"      : self.critic.state_dict(),
            "cost_critic" : self.cost_critic.state_dict(),
            "pi_opt"      : self.pi_opt.state_dict(),
            "Q_opt"       : self.Q_opt.state_dict(),
            "Qc_opt"      : self.Qc_opt.state_dict(),
            "lmbd"        : self.lmbd.detach().cpu().item(),
            "q_abs_ema"   : float(self.q_abs_ema.detach().cpu()),
            "step_counters": {
                "step"        : self.step,
                "pi_step"     : self.pi_step,
                "global_step" : self.global_step,
            },
        }, path)

    def load_models(self, path: str, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        # Friendly warning if checkpoint baseline != current baseline
        try:
            loaded_baseline = ckpt.get("meta", {}).get("baseline", "lagu")
            current_baseline = getattr(self.cfg, "baseline", "lagu")
            if str(loaded_baseline).lower() != str(current_baseline).lower():
                print(f"[load_models] WARNING: checkpoint baseline='{loaded_baseline}' "
                      f"!= current baseline='{current_baseline}'. Continuing anyway.", flush=True)
            # Shape expectations: warn loudly if vec/actor dims changed
            meta = ckpt.get("meta", {})
            exp_vec = int(meta.get("vec_dim", self.vec_dim))
            exp_actr = int(meta.get("actor_flat_dim", MAX_ACTORS * ACTOR_FEATURES))
            cur_actr = int(MAX_ACTORS * ACTOR_FEATURES)
            if (exp_vec != int(getattr(self, "vec_dim", exp_vec))) or (exp_actr != cur_actr):
                warnings.warn(f"[load_models] Checkpoint trained with vec_dim={exp_vec}, actor_flat_dim={exp_actr}; "
                              f"current build has vec_dim={getattr(self, 'vec_dim', exp_vec)}, actor_flat_dim={cur_actr}. "
                              "This can inflate uncertainty; consider adapting eval inputs or rebuilding the agent.", RuntimeWarning)
        except Exception:
            pass

        def _load_partial(module, state_dict, name):
            cur = module.state_dict()
            # keep only keys that exist **and** have identical shape
            filtered = {k: v for k, v in state_dict.items()
                        if (k in cur) and (cur[k].shape == v.shape)}
            missing = [k for k in cur.keys() if k not in filtered]
            skipped = [k for k in state_dict.keys() if k not in filtered]
            module.load_state_dict(filtered, strict=False)
            if missing:
                print(f"[load_models] {name}: left {len(missing)} param(s) at default (shape/key mismatch).")
            if skipped:
                print(f"[load_models] {name}: skipped {len(skipped)} param(s) from checkpoint (shape/key mismatch).")
            return (not missing and not skipped)  # True if perfect match

        ok_actor  = _load_partial(self.actor,       ckpt["actor"],       "actor")
        ok_critic = _load_partial(self.critic,      ckpt["critic"],      "critic")
        ok_qc     = _load_partial(self.cost_critic, ckpt["cost_critic"], "cost_critic")

        # re-sync targets from (possibly partially) loaded online nets
        self.critic_targ.load_state_dict(self.critic.state_dict())
        self.cost_targ.load_state_dict(self.cost_critic.state_dict())
        self.lmbd = torch.tensor([ckpt.get("lmbd", 0.0)],
                                 device=self.cfg.device,
                                 requires_grad=False)
        # Restore stabilized |Q| scale if available
        try:
            self.q_abs_ema = torch.tensor([float(ckpt.get("q_abs_ema", 0.0))],
                                          device=self.cfg.device)
        except Exception:
            pass
        # ─ optimiser & counters (optional) ─
        # Only load optimizers if the model loads were exact matches.
        if "pi_opt" in ckpt and ok_actor and ok_critic and ok_qc:
            try:
                self.pi_opt.load_state_dict(ckpt["pi_opt"])
                self.Q_opt.load_state_dict(ckpt["Q_opt"])
                self.Qc_opt.load_state_dict(ckpt["Qc_opt"])
            except Exception as e:
                print(f"[load_models] Skipping optimizer states (incompatible): {e}")
        else:
            print("[load_models] Skipping optimizer states (models changed).")
        ctrs = ckpt.get("step_counters", {})
        self.step        = ctrs.get("step", 0)
        self.pi_step     = ctrs.get("pi_step", 0)
        self.global_step = ctrs.get("global_step", 0)
        # If we had to do partial loads, uncertainty will be high. Soften eval guard.
        degraded = not (ok_actor and ok_critic and ok_qc)
        if degraded:
            try:
                print("[load_models] Partial parameter load detected → disabling eval guard for this session.")
                self.cfg.enable_eval_guard = False
            except Exception:
                pass

# ───────────────── weight‑synchronisation helpers ─────────────────
    def get_actor_weights(self):
        """Return a CPU copy of the actor weights (dict of tensors)."""
        return {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}

    def set_actor_weights(self, state_dict):
        """Load weights (CPU tensors) into the learner’s actor."""
        self.actor.load_state_dict(state_dict)
