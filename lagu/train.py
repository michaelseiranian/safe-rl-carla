"""Train one run of one arm on a Safety-Gymnasium task and log a CSV.

    python -m lagu.train --env SafetyPointGoal1-v0 --arm lagu --seed 0 --steps 1000000 --out results/pilot

Arms: td3lag | lagu | lagu_nogate | lagu_nobonus (see ARMS). A checkpoint with the replay buffer
is written at every evaluation; --resume continues from it (the simulator's own RNG cannot be
restored, so a resumed run is seeded, not bit-identical). Prefer lagu/launch.py to start runs.

--recipe omnisafe applies OmniSafe 0.5.0's SafetyPointGoal1 TD3Lag settings (actor 5e-6, critics 1e-3,
grad clip 40, loss / (1 + lambda), cost target from the online actor, 25k random steps, 1M replay,
episodic Adam dual at 5e-7 every env step on the last 50 training-episode costs after 202k steps).
Explicit flags override the recipe.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict

import numpy as np
import safety_gymnasium
import torch

from lagu.agent import Config, LagU

ARMS = {
    # base: single twin critic, Gaussian action noise, fixed constraint delta0 (Eq. 8-9)
    "td3lag":       dict(M=1, use_bonus=False, use_gate=False, action_noise=0.1),
    # full Lag-U (Eq. 13-18): ensemble, sqrt(Q_eu) bonus, uncertainty-gated constraint
    "lagu":         dict(M=3, use_bonus=True,  use_gate=True),
    # single-factor controls
    "lagu_nogate":  dict(M=3, use_bonus=True,  use_gate=False),
    "lagu_nobonus": dict(M=3, use_bonus=False, use_gate=True),
}

FIELDS = ["step", "episodes", "train_return", "train_cost", "eval_return", "eval_cost",
          "eval_violation", "mc_cost0", "qc0", "mc_ret0", "qbar0", "qstd0",
          "qc_traj", "mc_cost_traj", "qbar_traj", "mc_ret_traj", "lambda", "jc",
          "engagement", "n_explore", "n_risk", "explore_prob", "qbar", "qstd", "qc", "delta_mean",
          "ratio_p50", "ratio_p90", "q_loss", "c_loss", "sps",
          # interval means over every policy update since the previous row (CC1)
          "gap_avg", "qc_avg", "qbar_avg", "qstd_avg", "delta_avg", "delta_risk_avg", "lambda_avg",
          "lambda_shadow", "would_engage", "n_pi", "cum_cost", "cum_viol_eps", "resumed"]

RECIPES = {
    "paper": {},
    "omnisafe": dict(actor_lr=5e-6, critic_lr=1e-3, max_grad_norm=40.0, pi_norm=1, cost_target_online=1,
                     start_steps=25_000, replay=1_000_000, dual="episodic", lambda_lr=5e-7,
                     dual_every=1, jc_window=50, dual_start=202_000),
}
EVAL_SEED = 10_000   # every run evaluates on the same layouts, so seeds compare pairwise


class Replay:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.c = np.zeros(size, np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.ptr, self.n, self.size = 0, 0, size

    def add(self, o, a, r, c, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.c[i], self.o2[i], self.d[i] = o, a, r, c, o2, d
        self.ptr = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, rng: np.random.Generator, batch: int):
        idx = rng.integers(0, self.n, batch)
        return self.o[idx], self.a[idx], self.r[idx], self.c[idx], self.o2[idx], self.d[idx]

    def state(self) -> dict:
        return {**{k: torch.from_numpy(getattr(self, k)) for k in ("o", "a", "r", "c", "o2", "d")},
                "ptr": self.ptr, "n": self.n}

    def load(self, s: dict) -> None:
        """Restore a saved buffer; if the sizes differ (e.g. --steps was raised), keep the
        newest transitions that fit. Sampling is uniform, so their order does not matter."""
        s = {k: (np.asarray(v) if k not in ("ptr", "n") else v) for k, v in s.items()}   # tensors or arrays
        old, n, ptr = int(s["o"].shape[0]), int(s["n"]), int(s["ptr"])
        order = np.arange(n) if n < old else (np.arange(ptr, ptr + old) % old)   # oldest -> newest
        keep = order[-min(n, self.size):]
        k = len(keep)
        for name in ("o", "a", "r", "c", "o2", "d"):
            getattr(self, name)[:k] = s[name][keep]
        self.ptr, self.n = k % self.size, k


def evaluate(env, agent: LagU, episodes: int, cost_limit: float, gamma: float, horizon: int,
             seed_base: int = EVAL_SEED) -> dict:
    """Deterministic rollouts on the shared layouts EVAL_SEED + k. Critic calibration compares the
    critics with the Monte-Carlo discounted return and cost-to-go actually realised, at the first
    state (…0) and averaged over every state with at least horizon/2 steps left (…_traj)."""
    rets, costs, mc_c, mc_r, qb0, qs0, qc0 = ([] for _ in range(7))
    tq, tc, tmr, tmc = [], [], [], []
    for k in range(episodes):
        o, _ = env.reset(seed=seed_base + k)
        qb, qs, qc = agent.values_at(o)
        obs, rs, cs = [o], [], []
        done = False
        while not done:
            o, r, c, term, trunc, _ = env.step(agent.act(o, explore=False))
            rs.append(float(r))
            cs.append(float(c))
            done = term or trunc
            if not done:
                obs.append(o)
        n = len(rs)
        rs_a, cs_a = np.asarray(rs), np.asarray(cs)
        g_r, g_c = np.zeros(n + 1), np.zeros(n + 1)
        for t in range(n - 1, -1, -1):          # discounted return-to-go and cost-to-go from every t
            g_r[t] = rs_a[t] + gamma * g_r[t + 1]
            g_c[t] = cs_a[t] + gamma * g_c[t + 1]
        keep = max(1, min(len(obs), n - horizon // 2))
        qb_t, qc_t = agent.values_batch(np.stack(obs[:keep]))
        tq.append(float(qb_t.mean())), tc.append(float(qc_t.mean()))
        tmr.append(float(g_r[:keep].mean())), tmc.append(float(g_c[:keep].mean()))
        mc_c.append(float(g_c[0])), mc_r.append(float(g_r[0]))
        qb0.append(qb), qs0.append(qs), qc0.append(qc)
        rets.append(float(rs_a.sum())), costs.append(float(cs_a.sum()))
    costs = np.asarray(costs)
    m = lambda x: float(np.mean(x))
    return {"eval_return": m(rets), "eval_cost": float(costs.mean()),
            "eval_violation": float((costs > cost_limit).mean()),
            "mc_cost0": m(mc_c), "qc0": m(qc0), "mc_ret0": m(mc_r), "qbar0": m(qb0), "qstd0": m(qs0),
            "qc_traj": m(tc), "mc_cost_traj": m(tmc), "qbar_traj": m(tq), "mc_ret_traj": m(tmr)}


def save_ckpt(path, agent, replay, step, episodes, ep_rets, ep_costs, jc_costs, rngs, cum) -> None:
    tmp = path + ".tmp"
    torch.save({"agent": agent.state_dict(), "replay": replay.state(), "step": step,
                "episodes": episodes, "ep_rets": list(ep_rets), "ep_costs": list(ep_costs),
                "jc_costs": list(jc_costs), "rng": rngs.bit_generator.state,
                "torch_rng": torch.get_rng_state(), "cum": dict(cum)}, tmp)
    os.replace(tmp, path)


def save_snapshot(dirpath: str, agent: LagU, step: int) -> None:
    """Weights only (~2 MB) at every evaluation, for the held-out evaluation in lagu.evaluate."""
    os.makedirs(dirpath, exist_ok=True)
    torch.save({"step": step, "actor": agent.actor.state_dict(), "members": agent.members.state_dict(),
                "cost": agent.cost.state_dict(),
                "cost2": agent.cost2.state_dict() if agent.cfg.cost_twin else None,
                "lambda": agent.lmbda, "cfg": asdict(agent.cfg)}, os.path.join(dirpath, f"{step}.pt"))


def save_final(path: str, agent: LagU) -> None:
    """Atomic, fsynced write: a crash can never leave a truncated final .pt that looks finished."""
    tmp = path + ".tmp"
    torch.save(agent.state_dict(), tmp)
    fd = os.open(tmp, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    os.replace(tmp, path)


def truncate_csv(path: str, last_step: int) -> None:
    """Drop rows written after the checkpoint so a resumed run never duplicates steps. Refuse to
    resume into a CSV written with a different column schema."""
    with open(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise SystemExit(f"{path}: CSV columns differ from this trainer's FIELDS; refusing to resume")
        rows = list(reader)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", restval="")
        w.writeheader()
        for r in rows:
            if float(r["step"]) <= last_step:
                w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="SafetyPointGoal1-v0")
    p.add_argument("--arm", choices=sorted(ARMS), default="lagu")
    p.add_argument("--recipe", choices=sorted(RECIPES), default="paper")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--start-steps", type=int, default=None, help="uniform random actions first (10k)")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=0, help="0 = same as --eval-every")
    p.add_argument("--cost-limit", type=float, default=25.0, help="episodic cost limit (OmniSafe convention)")
    p.add_argument("--replay", type=int, default=None, help="replay capacity (500k)")
    p.add_argument("--T", type=float, default=None, help="override the uncertainty threshold")
    p.add_argument("--delta0", type=float, default=None,
                   help="override the Q_c constraint; default = cost_limit / horizon / (1 - gamma)")
    p.add_argument("--dual", choices=["statewise", "episodic", "fixed"], default=None)
    p.add_argument("--lambda-lr", type=float, default=None, help="override the dual step size")
    p.add_argument("--lambda-init", type=float, default=None)
    p.add_argument("--lambda-max", type=float, default=None, help="0 = unbounded")
    p.add_argument("--dual-every", type=int, default=None, help="episodic dual period in env steps (2000)")
    p.add_argument("--epoch-steps", type=int, default=None, help="alias of --dual-every")
    p.add_argument("--jc-window", type=int, default=None, help="episodes in the episodic Jc window (50)")
    p.add_argument("--dual-start", type=int, default=None, help="first env step of the episodic dual")
    p.add_argument("--actor-lr", type=float, default=None)
    p.add_argument("--critic-lr", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--pi-norm", type=int, default=None, help="1 = actor loss / (1 + lambda)")
    p.add_argument("--cost-target-online", type=int, default=None)
    p.add_argument("--gate-grad", type=int, default=0, help="1 = differentiate through delta_ada (via sqrt(Q_eu))")
    p.add_argument("--cost-twin", type=int, default=0, help="1 = pessimistic twin cost critic (max target)")
    p.add_argument("--cost-clamp", type=int, default=0, help="1 = clamp cost targets to [0, c_max/(1-gamma)]")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--tag", default="", help="variant name appended to the arm in file names")
    p.add_argument("--out", default="results/pilot")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--drop-ckpt", type=int, default=0, help="1 = delete the .ckpt.pt once the final .pt exists")
    args = p.parse_args()

    defaults = dict(start_steps=10_000, replay=500_000, dual="statewise", dual_every=2000, jc_window=50,
                    dual_start=None, lambda_lr=None, actor_lr=None, critic_lr=None, max_grad_norm=None,
                    pi_norm=None, cost_target_online=None, lambda_init=None, lambda_max=None)
    if args.epoch_steps is not None and args.dual_every is None:
        args.dual_every = args.epoch_steps
    for k, v in {**defaults, **RECIPES[args.recipe]}.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if args.dual_start is None:
        args.dual_start = args.start_steps

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(2)[0])   # replay sampling

    env = safety_gymnasium.make(args.env)
    eval_env = safety_gymnasium.make(args.env)
    env.action_space.seed(args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    horizon = int(getattr(getattr(env, "spec", None), "max_episode_steps", None) or 1000)
    gamma = 0.99
    # An episodic limit L over an H-step episode is a per-step budget L/H; its discounted
    # cost-to-go under gamma is the state-wise constraint Lag-U uses.
    delta0 = args.delta0 if args.delta0 is not None else args.cost_limit / horizon / (1.0 - gamma)
    cfg = Config(hidden=args.hidden, gamma=gamma, delta0=delta0, dual=args.dual,
                 cost_limit=args.cost_limit, gate_grad=bool(args.gate_grad),
                 cost_twin=bool(args.cost_twin), cost_clamp=bool(args.cost_clamp), **ARMS[args.arm])
    for k in ("T", "lambda_lr", "lambda_init", "lambda_max", "actor_lr", "critic_lr", "max_grad_norm"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    for k in ("pi_norm", "cost_target_online"):
        if getattr(args, k) is not None:
            setattr(cfg, k, bool(getattr(args, k)))
    agent_seed = int(np.random.SeedSequence(args.seed).spawn(2)[1].generate_state(1)[0])
    agent = LagU(obs_dim, act_dim, cfg, seed=agent_seed)
    replay = Replay(obs_dim, act_dim, min(args.replay, args.steps))
    ckpt_every = args.ckpt_every or args.eval_every

    os.makedirs(args.out, exist_ok=True)
    stem = f"{args.arm}{'-' + args.tag if args.tag else ''}_{args.env}_s{args.seed}"
    path = {ext: os.path.join(args.out, stem + ext) for ext in (".csv", ".json", ".pt", ".ckpt.pt")}
    snap_dir = os.path.join(args.out, stem + ".snap")
    ckpt = None
    if args.resume and os.path.exists(path[".ckpt.pt"]):
        ckpt = torch.load(path[".ckpt.pt"], weights_only=False)
    elif args.resume and os.path.exists(path[".pt"]):
        # finished and the checkpoint was dropped: never silently retrain from scratch
        last = 0
        if os.path.exists(path[".csv"]):
            with open(path[".csv"]) as f:
                steps_done = [float(r["step"]) for r in csv.DictReader(f) if r.get("step")]
            last = int(max(steps_done)) if steps_done else 0
        if last >= args.steps:
            print(f"[{stem}] already complete", flush=True)
            return
        raise SystemExit(f"[{stem}] finished at step {last} with its checkpoint dropped; "
                         f"cannot extend to {args.steps}")

    start, episodes = 1, 0
    cum = {"cost": 0.0, "viol_eps": 0}                          # whole-run training cost, violating episodes
    resumed = 0
    ep_rets, ep_costs = deque(maxlen=50), deque(maxlen=50)      # OmniSafe-style 50-episode windows
    jc_costs = deque(maxlen=args.jc_window)                      # episodes that feed the episodic dual
    if ckpt is not None:
        saved = ckpt["agent"]["cfg"]
        if saved != asdict(cfg):
            diff = {k: (saved.get(k), v) for k, v in asdict(cfg).items() if saved.get(k) != v}
            raise SystemExit(f"[{stem}] config differs from the checkpoint {diff}; refusing to resume")
        agent.load_state_dict(ckpt["agent"])
        replay.load(ckpt["replay"])
        rng.bit_generator.state = ckpt["rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        start, episodes = int(ckpt["step"]) + 1, int(ckpt["episodes"])
        ep_rets.extend(ckpt["ep_rets"]), ep_costs.extend(ckpt["ep_costs"]), jc_costs.extend(ckpt["jc_costs"])
        cum.update(ckpt.get("cum", {}))
        resumed = 1
        ckpt = None                                               # release the replay copy held by the checkpoint
        if os.path.exists(path[".json"]):                         # record the resume in the run's metadata
            with open(path[".json"]) as f:
                meta = json.load(f)
            meta.setdefault("resumes", []).append({"from_step": start - 1, "steps": args.steps, "argv": sys.argv})
            meta["args"]["steps"] = args.steps
            with open(path[".json"] + ".tmp", "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(path[".json"] + ".tmp", path[".json"])
        truncate_csv(path[".csv"], start - 1)
        csv_f = open(path[".csv"], "a", newline="")
        writer = csv.DictWriter(csv_f, fieldnames=FIELDS)
        o, _ = env.reset(seed=args.seed * 1000 + start)
        print(f"[{stem}] resumed after step {start - 1}", flush=True)
    else:
        with open(path[".json"], "w") as f:
            json.dump({"args": vars(args), "cfg": asdict(cfg), "horizon": horizon,
                       "obs_dim": obs_dim, "act_dim": act_dim}, f, indent=2)
        csv_f = open(path[".csv"], "w", newline="")
        writer = csv.DictWriter(csv_f, fieldnames=FIELDS)
        writer.writeheader()
        o, _ = env.reset(seed=args.seed)
    if start > args.steps:
        if not os.path.exists(path[".pt"]):
            save_final(path[".pt"], agent)
        print(f"[{stem}] already complete", flush=True)
        return

    ep_ret = ep_cost = 0.0
    t_last, step_last = time.time(), start - 1
    nan = float("nan")
    for step in range(start, args.steps + 1):
        agent.env_step = step                                    # x in Eq. 13
        a = env.action_space.sample() if step <= args.start_steps else agent.act(o, explore=True)
        o2, r, c, term, trunc, _ = env.step(a)
        replay.add(o, a, r, c, o2, float(term))
        ep_ret += r
        ep_cost += c
        cum["cost"] += float(c)
        o = o2
        if term or trunc:
            ep_rets.append(ep_ret)
            ep_costs.append(ep_cost)
            if step > args.start_steps:                          # warm-up episodes never feed the dual
                jc_costs.append(ep_cost)
            cum["viol_eps"] += int(ep_cost > args.cost_limit)
            episodes += 1
            ep_ret = ep_cost = 0.0
            o, _ = env.reset()
        if step > args.start_steps:
            agent.update(replay.sample(rng, cfg.batch))
            if (cfg.dual == "episodic" and step >= args.dual_start and jc_costs
                    and step % args.dual_every == 0):
                agent.update_dual_episodic(float(np.mean(jc_costs)))
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(eval_env, agent, args.eval_episodes, args.cost_limit, gamma, horizon)
            now = time.time()
            sps = (step - step_last) / max(now - t_last, 1e-9)
            t_last, step_last = now, step
            ps, cs = agent.pi_stats, agent.stats
            row = {
                "step": step, "episodes": episodes,
                "train_return": float(np.mean(ep_rets)) if ep_rets else nan,
                "train_cost": float(np.mean(ep_costs)) if ep_costs else nan,
                **ev,
                "lambda": agent.lmbda, "jc": agent.last_jc, "engagement": agent.engagement(),
                "n_explore": agent.n_explore, "n_risk": agent.n_risk, "explore_prob": agent.explore_prob(),
                "qbar": ps.get("qbar", nan), "qstd": ps.get("qstd", nan), "qc": ps.get("qc", nan),
                "delta_mean": ps.get("delta_mean", nan),
                "ratio_p50": ps.get("ratio_p50", nan), "ratio_p90": ps.get("ratio_p90", nan),
                "q_loss": cs.get("q_loss", nan), "c_loss": cs.get("c_loss", nan), "sps": sps,
                **agent.interval_means(), "lambda_shadow": agent.lmbda_shadow,
                "cum_cost": cum["cost"], "cum_viol_eps": cum["viol_eps"], "resumed": resumed,
            }
            writer.writerow(row)
            csv_f.flush()
            agent.reset_counters()
            print(f"[{stem}] step={step} ret={ev['eval_return']:.1f} cost={ev['eval_cost']:.1f} "
                  f"viol={ev['eval_violation']:.2f} lam={agent.lmbda:.3g} eng={row['engagement']:.3f} "
                  f"qc/mc={ev['qc_traj']:.2f}/{ev['mc_cost_traj']:.2f} sps={sps:.0f}", flush=True)
            save_snapshot(snap_dir, agent, step)
            if step % ckpt_every == 0 or step == args.steps:
                save_ckpt(path[".ckpt.pt"], agent, replay, step, episodes, ep_rets, ep_costs, jc_costs, rng, cum)
    save_final(path[".pt"], agent)
    csv_f.close()
    if args.drop_ckpt and os.path.exists(path[".pt"]) and os.path.exists(path[".ckpt.pt"]):
        os.remove(path[".ckpt.pt"])


if __name__ == "__main__":
    main()
