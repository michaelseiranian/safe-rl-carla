"""Gradient-parity probe: prereg.md section 9, "Gradient parity".

    python -m lagu.probe --dry-run                      # random untrained agent, shapes only, reads no file
    python -m lagu.probe --prereg [--out results/probe] [--workers 3]
    python -m lagu.probe --prereg --summarise-only      # re-print tables and figures from the job cache
    python -m lagu.probe --stems DIR/STEM@STEP ... --out DIR [--quick]   # any run, labelled EXPLORATORY

Label: EXPLORATORY test of a pre-registered secondary prediction, outside the Holm family. The
prediction is in prereg.md section 9; the operational protocol below was fixed in this file, and
the file was committed, before it read any study snapshot. Everything else it prints is DESCRIPTIVE,
including the pre-declared K = 3,000 sensitivity analysis.

PREDICTION (prereg.md section 9, verbatim): "The frozen-lambda return-halving point lies within a
factor of 2 of lambda* = ||grad_a Q_bar|| / ||grad_a Q_c||, measured by lagu/probe.py."

WHAT IS READ. For each declared checkpoint, only <stem>.json (env, obs_dim, act_dim, horizon) and
the snapshot file <stem>.snap/<step>.pt (actor, members, cost critic, lambda, cfg). No CSV, no
heldout.json, no other run. Each file's SHA-256 is recorded.

CHECKPOINT SETS (stem = <arm>_SafetyPointGoal1-v0_s<seed>; step = the run's last step N):
    S1  co-primary   results/gate_literal  lagu_nogate-lit  seeds 100-109  step 1,000,000  (E1b, literal recipe)
    S2  co-primary   results/gate          lagu_nogate-tt   seeds 100-114  step 2,000,000  (E1, chosen recipe)
    S3a descriptive  results/gate_literal  td3lag-lit       seeds 100-109  step 1,000,000
    S3b descriptive  results/gate          td3lag-tt        seeds 100-109  step 2,000,000
Every listed checkpoint is probed; none is excluded by hand. A missing file is reported as missing,
and the run then exits with status 2.

PROTOCOL, per checkpoint (constants in PROTOCOL below):
 1. Probe states S. Deterministic rollouts of the checkpoint's own policy (a = pi(s), no noise) on
    the new layouts PROBE_SEED + k, k = 0..19, PROBE_SEED = EVAL_SEED + 2000 = 12000, not on the
    held-out layouts: S also supplies the update minibatches of step 3, and R(lambda) is scored on
    held-out layouts, so S must not come from the episodes it is scored on (PROTOCOL.md section 3).
    These layouts are disjoint from the in-training layouts (EVAL_SEED + k) and the held-out layouts
    (EVAL_SEED + 1000 + j, j < 100). S is every state at which the policy acted: 20,000 states.
 2. lambda*. From S draw 5,000 states uniformly without replacement (numpy default_rng(0)). At each
    state s, a = pi(s); g_r = grad_a Q_bar(s, a), with Q_bar the mean of the M members' first heads
    (agent.ensemble, Eq. 10; M = 1 for td3lag), and g_c = grad_a Q_c(s, a) (agent.qc_pi). Norms are
    Euclidean over the action dimensions.
        lambda* = mean_s ||g_r(s)|| / mean_s ||g_c(s)||     (ratio of means over the 5,000 states)
    Also recorded (DESCRIPTIVE): the two mean norms, the mean cosine of g_r and g_c, the median of the
    per-state ratios ||g_r|| / ||g_c||, and the fraction of action components with |a| > 0.99.
 3. Frozen-lambda updates. For each lambda in GRID (0 first, then ascending) and each repetition
    i = 1..r (r = 3), start from the checkpoint's actor and make K Adam steps (lr 3e-4, the paper's
    actor step size, default betas, fresh optimizer state) on the risk-sensitive actor loss of
    Eq. 17 with lambda frozen:
        loss = mean_batch( -Q_bar(s, pi(s)) + lambda * Q_c(s, pi(s)) )
    (delta is a constant and drops out of the gradient; no exploration bonus; no (1 + lambda)
    scaling; no gradient clipping). Batches of 256 states are drawn from S with replacement by
    numpy default_rng(seed_i), seed_i = 0, 1, 2 for the three repetitions; repetition i uses the
    same batch sequence at every lambda. The critics are frozen: only the actor changes.
    Primary K = 10,000. Sensitivity K = 3,000 (DESCRIPTIVE, pre-declared): the actor after the first
    3,000 of the same steps, which equals a separate 3,000-step run because the batch indices of the
    shorter run are a prefix of the longer run's (checked in --dry-run).
    Snapshot files hold no target networks, optimizer state or replay buffer. The run's final
    <stem>.pt (written at the same step N) does hold target networks and the training Adam states;
    only the replay buffer is gone (the study runs dropped their .ckpt.pt). Neither is used: a fresh
    Adam because the stored actor moments were accumulated at the training actor lr (5e-6 under tt),
    and frozen critics because training them needs replay data, which no file holds, and because
    frozen critics make the objective exactly the one whose action gradients define lambda*.
 4. Return. Each updated actor plays the 20 held-out layouts EVAL_SEED + 1000 + k, k = 0..19
    (lagu.train.evaluate, deterministic): R_i(lambda). The curve used by every rule below is the
    mean over the r repetitions, R(lambda) = mean_i R_i(lambda); the mean cost is kept
    (DESCRIPTIVE). R_ckpt is the checkpoint's own actor on the same layouts with no update (one
    evaluation; it involves no minibatch). R0 = R(0), the mean return after K updates at lambda = 0.
 5. Eligibility, per K, decided once R0 is known: R_ckpt >= 10, R0 >= 10, R0 >= R_ckpt / 2, and
    lambda* finite and > 0. 10 is twice the collapse line of prereg.md section 5, and R0 >= R_ckpt/2
    means the update procedure alone did not halve the return. An ineligible checkpoint is listed
    with its reason and plays no grid at that K.
 6. Grid: lambda in {0} U {0.001 * 10^(j/4) : j = 0..20} (0.001 to 100, 4 points per decade). For
    each K it is played in ascending order and stops after two consecutive points with
    R <= R0 / 4 (mean curve); the first crossing below, the only quantity the rule uses, always lies
    before such a stop. A lambda is played, with all r repetitions, for every K still playing.
 7. Return-halving point, on the mean curve. lambda_half: the first grid point lambda_j > 0 with
    R(lambda_j) <= R0 / 2, interpolated linearly in log(lambda) between lambda_{j-1} and lambda_j.
    If that is the first nonzero point, lambda_half <= 0.001 (left-censored); if no point
    qualifies, lambda_half > 100 (right-censored). rho = lambda_half / lambda*.
    Spread across repetitions (DESCRIPTIVE): the range max_i R_i - min_i R_i at each played lambda
    (median and max over the played lambdas, lambda = 0 included), and lambda_half_i found by the
    same rule on each repetition's own curve against its own R_i(0), over the played lambdas only
    (right-censored at the last played lambda), with the range of log2(rho_i) over i.

DECISION RULE, per co-primary set (S1 and S2 separately), at the primary K = 10,000 only. The
statistic is the median of log2(rho) over the eligible checkpoints; a censored rho is imputed one
factor of 2 beyond the grid end (lambda_half = 0.0005 or 200) and counted. Its 95% interval is the
percentile bootstrap of that median over checkpoints (10,000 resamples, numpy default_rng(0)). The
point estimate reported in rho units is 2^median(log2 rho): with an even number of checkpoints it
is the geometric mean of the middle pair of rho, not their arithmetic mean.
    SUPPORTED     the 95% interval of the median of log2(rho) lies inside [log2 0.5, log2 2] = [-1, 1]
    REFUTED       that interval lies entirely above 1 or entirely below -1
    INCONCLUSIVE  otherwise
    NOT EVALUABLE fewer than 5 eligible checkpoints
Overall: REFUTED if S1 or S2 is REFUTED; otherwise SUPPORTED if both are SUPPORTED; otherwise NOT
EVALUABLE if at least one is NOT EVALUABLE; otherwise INCONCLUSIVE. The per-set labels, the point
estimate and the fraction of eligible checkpoints with 0.5 <= rho <= 2 are always reported. S3a
and S3b get the same numbers, labelled DESCRIPTIVE, and do not enter the overall label.

SENSITIVITY (pre-declared, DESCRIPTIVE): every number of steps 5-7 and of the decision rule is also
computed at K = 3,000, with its own R0, eligibility, stop and halving point, and printed in its own
section. It never changes a per-set label or the overall label.

DESCRIPTIVE extras: lambda* at every snapshot whose step is a multiple of 250k (up to N) of every
run in every set, next to the training lambda stored in that snapshot (steps 1-2 above only); and
the Spearman correlation of lambda_half with lambda* within a set (uncensored eligible checkpoints,
n >= 5).

Outputs in --out: jobs/<kind>/<stem>@<step>.json (one per probed file, reused only when the
protocol hash, the file's SHA-256 and the SHA-256 of this script all match), probe.json,
probe_tables.txt, probe_curves.{pdf,png} and probe_trajectory.{pdf,png}. Jobs run in --workers
processes with one torch thread each. Exit status: 2 if any declared file is missing, else 1 if
any job failed (or was not run under --summarise-only), else 0.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

from lagu.agent import Config, LagU

EVAL_SEED = 10_000           # == lagu.train.EVAL_SEED (checked in --dry-run against the import)
PROTOCOL = {
    "probe_seed": EVAL_SEED + 2000,         # layouts of the probe states S
    "state_episodes": 20,                    # deterministic episodes that make S
    "parity_states": 5000,                   # states drawn from S for lambda*
    "heldout_seed": EVAL_SEED + 1000,        # layouts of the return evaluation (block 0)
    "eval_episodes": 20,
    "K": 10_000,                             # frozen-lambda actor updates (primary)
    "K_sens": 3_000,                         # pre-declared sensitivity, DESCRIPTIVE
    "lr": 3e-4,
    "batch": 256,
    "rep_seeds": [0, 1, 2],                  # minibatch-sequence seeds of the r = 3 repetitions
    "rng_seed": 0,                           # state subsample, bootstrap
    "grid": [0.0] + [round(0.001 * 10 ** (j / 4), 10) for j in range(21)],
    "stop_frac": 0.25,                       # stop after 2 consecutive points with R <= stop_frac * R0
    "min_return": 10.0,
    "min_frac_of_ckpt": 0.5,
    "factor": 2.0,
    "n_boot": 10_000,
    "min_eligible": 5,
    "traj_every": 250_000,
}
QUICK = {**PROTOCOL, "state_episodes": 2, "parity_states": 500, "eval_episodes": 2, "K": 200, "K_sens": 60,
         "grid": [0.0, 0.01, 0.1, 1.0, 10.0], "min_return": -1e9, "min_frac_of_ckpt": -1e9, "n_boot": 500}
ENV = "SafetyPointGoal1-v0"
SETS = [  # (id, role, directory, arm tag, seeds, step)
    ("S1", "co-primary", "results/gate_literal", "lagu_nogate-lit", range(100, 110), 1_000_000),
    ("S2", "co-primary", "results/gate", "lagu_nogate-tt", range(100, 115), 2_000_000),
    ("S3a", "descriptive", "results/gate_literal", "td3lag-lit", range(100, 110), 1_000_000),
    ("S3b", "descriptive", "results/gate", "td3lag-tt", range(100, 110), 2_000_000),
]
TAG = {"co-primary": "EXPLORATORY", "exploratory": "EXPLORATORY", "descriptive": "DESCRIPTIVE"}
NAN = float("nan")


# ------------------------------------------------------------------ loading
def digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def ckpt_path(stem: str, step) -> str:
    return f"{stem}.pt" if step == "final" else os.path.join(stem + ".snap", f"{step}.pt")


def load_agent(stem: str, step):
    """Actor, members and cost critic(s) of a snapshot (or of a final <stem>.pt); never a random init."""
    with open(stem + ".json") as f:
        meta = json.load(f)
    d = torch.load(ckpt_path(stem, step), weights_only=True)
    names = {f.name for f in dataclasses.fields(Config)}
    agent = LagU(meta["obs_dim"], meta["act_dim"], Config(**{k: v for k, v in d["cfg"].items() if k in names}))
    for name in ("actor", "members", "cost") + (("cost2",) if agent.cfg.cost_twin else ()):
        getattr(agent, name).load_state_dict(d[name])
    return agent, meta, float(d.get("lambda", NAN))


# ---------------------------------------------------------------- the probe
def qbar_fast(agent: LagU, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Q_bar of Eq. 10 from the members' first heads only; equals agent.ensemble(o, a)[0]."""
    x = torch.cat([o, a], dim=-1)
    return torch.stack([m.q1(x).squeeze(-1) for m in agent.members], dim=0).mean(0)


@torch.no_grad()
def collect_states(agent: LagU, env, episodes: int, seed_base: int) -> np.ndarray:
    states = []
    for k in range(episodes):
        o, _ = env.reset(seed=seed_base + k)
        done = False
        while not done:
            states.append(o)
            o, _, _, term, trunc, _ = env.step(agent.act(o, explore=False))
            done = term or trunc
    return np.asarray(states, dtype=np.float32)


def parity(agent: LagU, states: np.ndarray) -> dict:
    o = torch.as_tensor(states)
    with torch.no_grad():
        a0 = agent.actor(o)
    a = a0.clone().requires_grad_(True)
    g_r, = torch.autograd.grad(agent.ensemble(o, a)[0].sum(), a)
    g_c, = torch.autograd.grad(agent.qc_pi(o, a).sum(), a)
    nr, nc = g_r.norm(dim=1).double(), g_c.norm(dim=1).double()
    cos = (g_r * g_c).sum(1).double() / (nr * nc)
    ok = nc > 0
    return {"lam_star": float(nr.mean() / nc.mean()), "g_r_mean": float(nr.mean()), "g_c_mean": float(nc.mean()),
            "cos_mean": float(cos[ok & (nr > 0)].mean()), "ratio_median": float((nr[ok] / nc[ok]).median()),
            "sat_frac": float((a0.abs() > 0.99).double().mean()), "n": int(len(states)),
            "shapes": [list(g_r.shape), list(g_c.shape)]}


def frozen_lambda_actors(agent: LagU, states: np.ndarray, lam: float, P: dict, seed: int, Ks) -> dict:
    """{K: actor after K Adam steps} for each K in Ks, from ONE run of max(Ks) steps on the batch
    sequence default_rng(seed). A shorter run's batch indices are a prefix of a longer run's
    (checked in --dry-run), so each actor equals that of a separate K-step run."""
    Ks = sorted(set(Ks))
    actor = copy.deepcopy(agent.actor)
    opt = torch.optim.Adam(actor.parameters(), lr=P["lr"])
    S = torch.as_tensor(states)
    idx = np.random.default_rng(seed).integers(0, len(S), size=(Ks[-1], P["batch"]))
    out = {}
    for k in range(Ks[-1]):
        o = S[idx[k]]
        a = actor(o)
        loss = (-qbar_fast(agent, o, a) + lam * agent.qc_pi(o, a)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if k + 1 in Ks:
            out[k + 1] = copy.deepcopy(actor) if k + 1 < Ks[-1] else actor
    return out


def play(agent: LagU, actor, env, meta: dict, P: dict) -> dict:
    from lagu.train import evaluate
    probe = copy.copy(agent)            # shares the frozen critics; only the actor differs
    probe.actor = actor
    ev = evaluate(env, probe, P["eval_episodes"], agent.cfg.cost_limit, agent.cfg.gamma, meta["horizon"],
                  seed_base=P["heldout_seed"])
    return {"return": ev["eval_return"], "cost": ev["eval_cost"]}


def halving_point(lams, rets, r0: float):
    """(lambda_half, censor) from the first grid point lambda_j > 0 with R <= R0 / 2, interpolated in
    log(lambda); censor is 'left' (lambda_half <= lams[0]), 'right' (> lams[-1]) or ''."""
    h, prev = r0 / 2.0, None
    for lam, r in zip(lams, rets):
        if r is None:
            break
        if r <= h:
            if prev is None:
                return lam, "left"
            (l0, r_0), t = prev, (prev[1] - h) / (prev[1] - r)
            return math.exp(math.log(l0) + t * (math.log(lam) - math.log(l0))), ""
        prev = (lam, r)
    return lams[-1], "right"


def log2_rho(lam_half: float, censor: str, lam_star: float) -> float:
    """log2(lambda_half / lambda*); a censored lambda_half is imputed one factor of 2 beyond the grid end."""
    return math.log2(lam_half / lam_star) + {"left": -1.0, "right": 1.0, "": 0.0}[censor]


def rep_spread(p0: dict, grid: list[dict], lam_star: float) -> dict:
    """DESCRIPTIVE spread across repetitions: the range of R_i at each played lambda (0 included), and
    each repetition's own halving point over the played lambdas (right-censored at the last one)."""
    ranges = [max(p["returns"]) - min(p["returns"]) for p in [p0] + grid]
    lams, reps = [g["lam"] for g in grid], []
    for i, r0 in enumerate(p0["returns"]):
        lh, c = halving_point(lams, [g["returns"][i] for g in grid], r0)
        reps.append({"lam_half": lh, "censor": c, "log2_rho": log2_rho(lh, c, lam_star)})
    l2 = [x["log2_rho"] for x in reps]
    return {"R_range_median": float(np.median(ranges)), "R_range_max": float(max(ranges)), "reps": reps,
            "log2_rho_range": float(max(l2) - min(l2)), "n_rep_censored": sum(bool(x["censor"]) for x in reps)}


def play_grid(point, r_ckpt: float, lam_star: float, P: dict) -> dict:
    """Steps 5-7 for K and K_sens. point(lam, Ks) -> {K: [{'return', 'cost'} per repetition]}. Each K has
    its own R0, eligibility and stop; a lambda is played, with every repetition, for the Ks still playing."""
    Ks = sorted({P["K"], P["K_sens"]})
    pts, why, done = {K: [] for K in Ks}, {K: "" for K in Ks}, {K: False for K in Ks}
    for lam in P["grid"]:
        active = [K for K in Ks if not done[K]]
        if not active:
            break
        res = point(lam, active)
        for K in active:
            rets, costs = [x["return"] for x in res[K]], [x["cost"] for x in res[K]]
            pts[K].append({"lam": lam, "return": float(np.mean(rets)), "cost": float(np.mean(costs)),
                           "returns": rets, "costs": costs})
            r0 = pts[K][0]["return"]
            if lam == 0.0:
                why[K] = "; ".join(w for w, bad in (
                    ("R_ckpt < min_return", not r_ckpt >= P["min_return"]),
                    ("R0 < min_return", not r0 >= P["min_return"]),
                    ("R0 < R_ckpt / 2", not r0 >= P["min_frac_of_ckpt"] * r_ckpt),
                    ("lambda* not finite and > 0", not (math.isfinite(lam_star) and lam_star > 0))) if bad)
                done[K] = bool(why[K])
            elif len(pts[K]) >= 3 and all(p["return"] <= P["stop_frac"] * r0 for p in pts[K][-2:]):
                done[K] = True
    out, lams = {}, P["grid"][1:]
    for K in Ks:
        p0, grid = pts[K][0], pts[K][1:]
        b = {"K": K, "R0": {k: p0[k] for k in ("return", "cost", "returns", "costs")},
             "eligible": not why[K], "why": why[K], "grid": grid}
        if b["eligible"]:
            lh, c = halving_point(lams, [g["return"] for g in grid] + [None] * (len(lams) - len(grid)), p0["return"])
            b.update(lam_half=lh, censor=c, log2_rho=log2_rho(lh, c, lam_star), spread=rep_spread(p0, grid, lam_star))
        out[str(K)] = b
    return out


def probe_checkpoint(agent: LagU, meta: dict, env, P: dict) -> dict:
    S = collect_states(agent, env, P["state_episodes"], P["probe_seed"])
    sub = np.random.default_rng(P["rng_seed"]).choice(len(S), size=min(P["parity_states"], len(S)), replace=False)
    par = parity(agent, S[np.sort(sub)])
    for net in (agent.members, agent.cost) + ((agent.cost2,) if agent.cfg.cost_twin else ()):
        net.requires_grad_(False)
    out = {"n_states": int(len(S)), "parity": par, "R_ckpt": play(agent, agent.actor, env, meta, P)}

    def point(lam, Ks):
        res = {K: [] for K in Ks}
        for seed in P["rep_seeds"]:
            actors = frozen_lambda_actors(agent, S, lam, P, seed, Ks)
            for K in Ks:
                res[K].append(play(agent, actors[K], env, meta, P))
        return res

    out["byK"] = play_grid(point, out["R_ckpt"]["return"], par["lam_star"], P)
    return out


# -------------------------------------------------------------------- jobs
def job_file(out: str, job: dict) -> str:
    return os.path.join(out, "jobs", job["kind"], f"{os.path.basename(job['stem'])}@{job['step']}.json")


def run_job(args: tuple) -> dict:
    job, P = args
    torch.set_num_threads(1)
    torch.manual_seed(P["rng_seed"])
    t0 = time.time()
    try:
        import safety_gymnasium
        agent, meta, lam_train = load_agent(job["stem"], job["step"])
        env = safety_gymnasium.make(meta["args"]["env"])
        if job["kind"] == "main":
            res = probe_checkpoint(agent, meta, env, P)
        else:
            S = collect_states(agent, env, P["state_episodes"], P["probe_seed"])
            sub = np.random.default_rng(P["rng_seed"]).choice(len(S), size=min(P["parity_states"], len(S)), replace=False)
            res = {"n_states": int(len(S)), "parity": parity(agent, S[np.sort(sub)])}
        env.close()
        return {**job, **res, "lambda_train": lam_train, "seconds": time.time() - t0}
    except Exception as e:   # one bad file must not abort the others
        return {**job, "error": f"{type(e).__name__}: {e}"}


def declared_jobs(root: str, P: dict) -> list[dict]:
    jobs = []
    for sid, role, d, arm, seeds, step in SETS:
        for s in seeds:
            stem = os.path.join(root, d, f"{arm}_{ENV}_s{s}")
            jobs.append({"kind": "main", "set": sid, "role": role, "stem": stem, "step": step})
    for sid, role, d, arm, seeds, step in SETS:
        for s in seeds:
            stem = os.path.join(root, d, f"{arm}_{ENV}_s{s}")
            for t in range(P["traj_every"], step + 1, P["traj_every"]):
                jobs.append({"kind": "traj", "set": sid, "role": role, "stem": stem, "step": t})
    return jobs


def run_all(jobs: list[dict], P: dict, out: str, workers: int, summarise_only: bool) -> list[dict]:
    phash = hashlib.sha256(json.dumps(P, sort_keys=True).encode()).hexdigest()
    me = digest(os.path.abspath(__file__))
    done, todo = [], []
    for job in jobs:
        f, path = job_file(out, job), ckpt_path(job["stem"], job["step"])
        absent = [p for p in (job["stem"] + ".json", path) if not os.path.exists(p)]
        if absent:
            done.append({**job, "error": "missing", "missing_files": absent})
            continue
        job = {**job, "file": path, "sha256": digest(path)}
        if os.path.exists(f):
            with open(f) as fh:
                prev = json.load(fh)
            if (prev.get("protocol_sha256") == phash and prev.get("sha256") == job["sha256"]
                    and prev.get("probe_sha256") == me):
                done.append({**prev, "set": job["set"], "role": job["role"]})   # the cache is per file, not per set
                continue
        (done.append({**job, "error": "not run (--summarise-only)"}) if summarise_only else todo.append(job))
    if todo:
        print(f"probing {len(todo)} files with {workers} workers", flush=True)
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(min(workers, len(todo))) if workers > 1 and len(todo) > 1 else None
        it = pool.imap_unordered(run_job, [(j, P) for j in todo]) if pool else map(run_job, [(j, P) for j in todo])
        for res in it:
            if "error" not in res:
                res.update(protocol=P, protocol_sha256=phash, probe_sha256=me,
                           written=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
                os.makedirs(os.path.dirname(job_file(out, res)), exist_ok=True)
                with open(job_file(out, res) + ".tmp", "w") as fh:
                    json.dump(res, fh, indent=1)
                os.replace(job_file(out, res) + ".tmp", job_file(out, res))
            print(f"  {res['kind']} {os.path.basename(res['stem'])}@{res['step']}: "
                  f"{res.get('error') or 'lambda*=%.4g (%.0fs)' % (res['parity']['lam_star'], res['seconds'])}",
                  flush=True)
            done.append(res)
        if pool:
            pool.close()
            pool.join()
    return done


# ---------------------------------------------------------------- summary
def verdict(x: np.ndarray, P: dict) -> dict:
    """The rule on log2(rho): the median of log2(rho) and its percentile-bootstrap 95% interval."""
    x = np.asarray(x, float)
    b = math.log2(P["factor"])
    out = {"n_eligible": int(len(x)), "frac_within": float(np.mean(np.abs(x) <= b)) if len(x) else NAN}
    if len(x) < P["min_eligible"]:
        return {**out, "label": "NOT EVALUABLE", "median_log2_rho": NAN, "ci95_log2_rho": [NAN, NAN],
                "rho_hat": NAN, "ci95_rho": [NAN, NAN]}
    rng = np.random.default_rng(P["rng_seed"])
    meds = np.median(x[rng.integers(0, len(x), size=(P["n_boot"], len(x)))], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    label = ("SUPPORTED" if -b <= lo and hi <= b else "REFUTED" if lo > b or hi < -b else "INCONCLUSIVE")
    m = float(np.median(x))
    return {**out, "label": label, "median_log2_rho": m, "ci95_log2_rho": [float(lo), float(hi)],
            "rho_hat": float(2 ** m), "ci95_rho": [float(2 ** lo), float(2 ** hi)]}


def overall_label(labels: list[str]) -> str:
    if any(x == "REFUTED" for x in labels):
        return "REFUTED"
    if all(x == "SUPPORTED" for x in labels):
        return "SUPPORTED"
    if any(x == "NOT EVALUABLE" for x in labels):
        return "NOT EVALUABLE"
    return "INCONCLUSIVE"


def spearman(x, y) -> float:
    def ranks(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for u in np.unique(v):        # average ranks for ties
            r[v == u] = r[v == u].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(rx) >= 5 and rx.std() > 0 and ry.std() > 0 else NAN


def set_title(sid: str, role: str) -> str:
    what = {"co-primary": " (prereg section 9 prediction)",
            "exploratory": " (hand-picked runs, not the pre-registered sets)"}.get(role, "")
    return f"{sid} [{TAG[role]}{what}]"


def summarise(results: list[dict], P: dict, sets: list[tuple], header: str) -> tuple[dict, str]:
    g = lambda x, f="%.4g": "nan" if x is None or not np.isfinite(x) else f % x
    K, Ks = str(P["K"]), str(P["K_sens"])
    lines = [header, "", f"Primary K = {P['K']:,} frozen-lambda updates; R = mean over r = {len(P['rep_seeds'])} "
             f"repetitions (minibatch seeds {P['rep_seeds']}) on the same {P['eval_episodes']} held-out episodes.", ""]
    per_set, sens, labels = {}, {}, {}
    half = lambda b: ((("<=" if b["censor"] == "left" else ">" if b["censor"] == "right" else "") + g(b["lam_half"]))
                      if b["eligible"] else "-")
    rho = lambda b: g(2 ** b["log2_rho"]) + ("*" if b["censor"] else "") if b["eligible"] else "-"
    for sid, role, *_ in sets:
        main = sorted((r for r in results if r["kind"] == "main" and r["set"] == sid), key=lambda r: r["stem"])
        lines += [f"Table {set_title(sid, role)}: per checkpoint, {role}, K = {P['K']:,}", "  " + " ".join([
            f"{'run@step':<46}", f"{'lam_train':>9}", f"{'|gQbar|':>8}", f"{'|gQc|':>8}", f"{'cos':>6}",
            f"{'lam*':>8}", f"{'lam*med':>8}", f"{'sat':>5}", f"{'R_ckpt':>7}", f"{'R0':>7}",
            f"{'lam_half':>10}", f"{'rho':>8}", f"{'Rrange':>6}", f"{'l2rng':>5}", "eligible / note"])]
        el = []
        for r in main:
            name = f"{os.path.basename(r['stem'])}@{r['step']}"
            if "error" in r:
                lines.append(f"  {name:<46} {r['error']}" + (f": {', '.join(r['missing_files'])}"
                                                            if r.get("missing_files") else ""))
                continue
            p, b = r["parity"], r["byK"][K]
            sp = b.get("spread", {})
            lines.append("  " + " ".join([f"{name:<46}", f"{g(r['lambda_train']):>9}", f"{g(p['g_r_mean']):>8}",
                         f"{g(p['g_c_mean']):>8}", f"{g(p['cos_mean'], '%.2f'):>6}", f"{g(p['lam_star']):>8}",
                         f"{g(p['ratio_median']):>8}", f"{g(p['sat_frac'], '%.2f'):>5}",
                         f"{g(r['R_ckpt']['return'], '%.2f'):>7}", f"{g(b['R0']['return'], '%.2f'):>7}",
                         f"{half(b):>10}", f"{rho(b):>8}", f"{g(sp.get('R_range_median'), '%.2f'):>6}",
                         f"{g(sp.get('log2_rho_range'), '%.2f'):>5}", "yes" if b["eligible"] else "no: " + b["why"]]))
            if b["eligible"]:
                el.append(r)
        v = verdict([r["byK"][K]["log2_rho"] for r in el], P)
        unc = [r for r in el if not r["byK"][K]["censor"]]
        ok = [r for r in main if "error" not in r]
        v.update(n_listed=len(main), n_missing=sum(r.get("error") == "missing" for r in main),
                 n_failed=sum("error" in r and r["error"] != "missing" for r in main),
                 n_censored=sum(bool(r["byK"][K]["censor"]) for r in el),
                 spearman_lamhalf_lamstar=spearman([r["parity"]["lam_star"] for r in unc],
                                                   [r["byK"][K]["lam_half"] for r in unc]),
                 median_lam_star=float(np.median([r["parity"]["lam_star"] for r in ok])) if ok else NAN,
                 median_R_range=float(np.median([r["byK"][K]["spread"]["R_range_median"] for r in el])) if el else NAN,
                 median_log2_rho_range=float(np.median([r["byK"][K]["spread"]["log2_rho_range"] for r in el]))
                 if el else NAN)
        per_set[sid] = {"role": role, **v}
        if role == "co-primary":
            labels[sid] = v["label"]
        lines += [f"  -> {sid}: {v['n_eligible']} of {v['n_listed']} eligible ({v['n_missing']} missing, "
                  f"{v['n_failed']} failed, {v['n_censored']} censored, * = imputed one factor 2 beyond the grid); "
                  f"median log2(rho) {g(v['median_log2_rho'], '%.3f')}, 95% [{g(v['ci95_log2_rho'][0], '%.3f')}, "
                  f"{g(v['ci95_log2_rho'][1], '%.3f')}], i.e. rho_hat = 2^median {g(v['rho_hat'])}, 95% "
                  f"[{g(v['ci95_rho'][0])}, {g(v['ci95_rho'][1])}]; within x2: {g(v['frac_within'], '%.2f')}; "
                  f"Spearman(lam_half, lam*) {g(v['spearman_lamhalf_lamstar'], '%.2f')} [DESCRIPTIVE]; "
                  f"repetition spread: median Rrange {g(v['median_R_range'], '%.2f')}, median l2rng "
                  f"{g(v['median_log2_rho_range'], '%.2f')} [DESCRIPTIVE]; label {v['label']} [{TAG[role]}]", ""]
    if labels:
        overall = overall_label(list(labels.values()))
        lines += [f"OVERALL gradient-parity prediction [EXPLORATORY, outside the Holm family]: {overall} "
                  f"({', '.join(f'{k} {v}' for k, v in labels.items())})", ""]
    else:
        overall = None
    lines += ["Rrange = median over the played lambdas (0 included) of max - min of R_i over the repetitions; "
              "l2rng = max - min over the repetitions of log2(rho_i), each repetition's own halving point "
              "(censored values imputed as in the rule). Both DESCRIPTIVE.", "",
              f"SENSITIVITY K = {P['K_sens']:,} [DESCRIPTIVE, pre-declared; never changes a label above]"]
    for sid, role, *_ in sets:
        main = sorted((r for r in results if r["kind"] == "main" and r["set"] == sid and "error" not in r),
                      key=lambda r: r["stem"])
        lines += [f"  Table {sid}, K = {P['K_sens']:,}", "    " + " ".join([
            f"{'run@step':<46}", f"{'lam*':>8}", f"{'R0':>7}", f"{'lam_half':>10}", f"{'rho':>8}", f"{'Rrange':>6}",
            f"{'l2rng':>5}", "eligible / note"])]
        el = []
        for r in main:
            b = r["byK"][Ks]
            sp = b.get("spread", {})
            lines.append("    " + " ".join([f"{os.path.basename(r['stem']) + '@' + str(r['step']):<46}",
                         f"{g(r['parity']['lam_star']):>8}", f"{g(b['R0']['return'], '%.2f'):>7}", f"{half(b):>10}",
                         f"{rho(b):>8}", f"{g(sp.get('R_range_median'), '%.2f'):>6}",
                         f"{g(sp.get('log2_rho_range'), '%.2f'):>5}", "yes" if b["eligible"] else "no: " + b["why"]]))
            if b["eligible"]:
                el.append(r)
        v = verdict([r["byK"][Ks]["log2_rho"] for r in el], P)
        v["n_censored"] = sum(bool(r["byK"][Ks]["censor"]) for r in el)
        sens[sid] = {"role": role, **v}
        lines += [f"    -> {sid}, K = {P['K_sens']:,}: {v['n_eligible']} eligible, {v['n_censored']} censored; median "
                  f"log2(rho) {g(v['median_log2_rho'], '%.3f')}, 95% [{g(v['ci95_log2_rho'][0], '%.3f')}, "
                  f"{g(v['ci95_log2_rho'][1], '%.3f')}], rho_hat {g(v['rho_hat'])}; within x2: "
                  f"{g(v['frac_within'], '%.2f')}; the rule would read {v['label']} [DESCRIPTIVE]", ""]
    traj = sorted((r for r in results if r["kind"] == "traj" and "error" not in r), key=lambda r: (r["set"], r["step"]))
    if traj:
        lines += ["Table T [DESCRIPTIVE]: lambda* over training (median over runs; lam_train = lambda stored in the snapshot)",
                  f"  {'set':<5} {'step':>9} {'n':>3} {'lam*':>9} {'lam_train':>10} {'lam_train/lam*':>15} {'n(lam_train>lam*)':>18}"]
        for sid, *_ in sets:
            for t in sorted({r["step"] for r in traj if r["set"] == sid}):
                rs = [r for r in traj if r["set"] == sid and r["step"] == t]
                ls_ = np.array([r["parity"]["lam_star"] for r in rs])
                lt = np.array([r["lambda_train"] for r in rs])
                lines.append(f"  {sid:<5} {t:>9} {len(rs):>3} {g(np.median(ls_)):>9} {g(np.median(lt)):>10} "
                             f"{g(np.median(lt / ls_)):>15} {int(np.sum(lt > ls_)):>18}")
    return {"sets": per_set, "sensitivity": sens, "overall": overall}, "\n".join(lines).rstrip()


def figures(results: list[dict], sets: list[tuple], out: str, title: str, P: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf, ink, mute, grey, blue, orange = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29c", "#2a78d6", "#eb6834"
    plt.rcParams.update({"figure.facecolor": surf, "axes.facecolor": surf, "savefig.facecolor": surf,
                         "text.color": ink, "axes.labelcolor": ink, "xtick.color": mute, "ytick.color": mute,
                         "axes.edgecolor": mute, "font.size": 9})
    K = str(P["K"])
    n = len(sets)
    cols = min(2, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows), squeeze=False)
    for ax, (sid, role, d, arm, seeds, step) in zip(axes.flat, sets):
        el = [r for r in results if r["kind"] == "main" and r["set"] == sid and "error" not in r
              and r["byK"][K]["eligible"]]
        ax.axvspan(0.5, 2.0, color=blue, alpha=0.10, lw=0, label="within a factor 2 of $\\lambda^*$")
        ax.axhline(0.5, color=mute, lw=1, ls="--", label="half of $R_0$")
        for r in el:
            b, ls = r["byK"][K], r["parity"]["lam_star"]
            r0 = b["R0"]["return"]
            ax.plot([p["lam"] / ls for p in b["grid"]], [p["return"] / r0 for p in b["grid"]], color=grey, lw=1)
            if not b["censor"]:
                ax.plot([b["lam_half"] / ls], [0.5], "o", ms=5, color=orange, mec=surf, mew=1)
        ax.plot([], [], color=grey, lw=1, label=f"one checkpoint (n = {len(el)} eligible)")
        ax.plot([], [], "o", ms=5, color=orange, label="return-halving point")
        ax.set_xscale("log")
        ax.set_title(f"{sid} ({role}): {arm} @ {step:,}" if step else f"{sid} ({role})", fontsize=9)
        ax.set_xlabel("frozen $\\lambda$ / $\\lambda^*$ (log scale)")
        ax.set_ylabel("held-out return / $R_0$")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title + f"\nreturn after {P['K']:,} frozen-$\\lambda$ actor updates (mean of {len(P['rep_seeds'])} "
                 "repetitions), relative to $\\lambda = 0$ ($R_0$)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"probe_curves.{ext}"), dpi=200)
    plt.close(fig)

    traj = [r for r in results if r["kind"] == "traj" and "error" not in r]
    if not traj:
        return
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), squeeze=False)
    for ax, (sid, role, d, arm, seeds, step) in zip(axes.flat, sets):
        rs = [r for r in traj if r["set"] == sid]
        for stem in sorted({r["stem"] for r in rs}):
            pts = sorted((r["step"], r["parity"]["lam_star"], r["lambda_train"]) for r in rs if r["stem"] == stem)
            t = [p[0] / 1e6 for p in pts]
            ax.plot(t, [p[1] for p in pts], color=blue, lw=0.8, alpha=0.45)
            ax.plot(t, [max(p[2], 1e-6) for p in pts], color=orange, lw=0.8, alpha=0.45, ls="--")
        ax.plot([], [], color=blue, lw=1.5, label="$\\lambda^*$ (gradient parity)")
        ax.plot([], [], color=orange, lw=1.5, ls="--", label="training $\\lambda$ (floored at 1e-6)")
        ax.set_yscale("log")
        ax.set_title(f"{sid}: {arm} (one line per seed)", fontsize=9)
        ax.set_xlabel("training step (millions)")
        ax.set_ylabel("$\\lambda$ (unitless multiplier, log scale)")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.legend(fontsize=7, frameon=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("DESCRIPTIVE: gradient parity $\\lambda^*$ and the training multiplier over training", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"probe_trajectory.{ext}"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- dry run
def dry_run() -> int:
    """Random untrained agents only: shapes, finiteness, the fast Q_bar, the update, the nested K, the
    grid control and the rule on synthetic numbers. Reads no file and builds no environment."""
    from lagu.train import EVAL_SEED as TRAIN_EVAL_SEED
    assert TRAIN_EVAL_SEED == EVAL_SEED
    torch.manual_seed(0)
    torch.set_num_threads(1)
    P = {**PROTOCOL, "K": 50, "K_sens": 20, "parity_states": 5000}
    rng = np.random.default_rng(0)
    for arm, kw in (("lagu_nogate", dict(M=3, use_bonus=True, use_gate=False)),
                    ("td3lag", dict(M=1, use_bonus=False, use_gate=False, action_noise=0.1))):
        agent = LagU(60, 2, Config(hidden=256, **kw))
        S = rng.standard_normal((20_000, 60)).astype(np.float32)
        sub = np.random.default_rng(0).choice(len(S), size=P["parity_states"], replace=False)
        par = parity(agent, S[np.sort(sub)])
        assert par["shapes"] == [[5000, 2], [5000, 2]], par["shapes"]
        assert all(np.isfinite(par[k]) for k in ("lam_star", "g_r_mean", "g_c_mean", "cos_mean", "ratio_median"))
        o = torch.as_tensor(S[:256])
        a = agent.actor(o)
        assert torch.equal(qbar_fast(agent, o, a), agent.ensemble(o, a)[0])
        for net in (agent.members, agent.cost):
            net.requires_grad_(False)
        before = [p.detach().clone() for p in list(agent.members.parameters()) + list(agent.cost.parameters())]
        t0 = time.time()
        a0 = frozen_lambda_actors(agent, S, 0.0, P, 0, [P["K"]])[P["K"]]
        nest = frozen_lambda_actors(agent, S, 1.0, P, 0, [P["K_sens"], P["K"]])
        ms = (time.time() - t0) / (2 * P["K"]) * 1e3
        sep = frozen_lambda_actors(agent, S, 1.0, P, 0, [P["K_sens"]])[P["K_sens"]]
        rep1 = frozen_lambda_actors(agent, S, 1.0, P, 1, [P["K"]])[P["K"]]
        after = list(agent.members.parameters()) + list(agent.cost.parameters())
        assert all(torch.equal(x, y) for x, y in zip(before, after)), "critics changed"
        a1 = nest[P["K"]]
        same = lambda m1, m2: all(torch.equal(p, q) for p, q in zip(m1.parameters(), m2.parameters()))
        assert same(nest[P["K_sens"]], sep), "the nested K_sens actor differs from a separate K_sens-step run"
        assert not same(nest[P["K_sens"]], a1) and not same(rep1, a1), "K or the repetition seed had no effect"
        d0 = max(float((p - q).abs().max()) for p, q in zip(a0.parameters(), agent.actor.parameters()))
        d01 = max(float((p - q).abs().max()) for p, q in zip(a0.parameters(), a1.parameters()))
        assert d0 > 0 and d01 > 0
        with torch.no_grad():
            assert agent.actor(o).shape == a0(o).shape == (256, 2)
        print(f"{arm:<12} M={agent.cfg.M}: lambda*={par['lam_star']:.4g} |gQbar|={par['g_r_mean']:.3g} "
              f"|gQc|={par['g_c_mean']:.3g} cos={par['cos_mean']:.2f} sat={par['sat_frac']:.2f}; actor update "
              f"{ms:.2f} ms; critics frozen, nested K_sens actor == separate run, repetition seeds differ: OK")
    for s in PROTOCOL["rep_seeds"]:    # the prefix property at the real sizes
        long = np.random.default_rng(s).integers(0, 20_000, size=(PROTOCOL["K"], PROTOCOL["batch"]))
        short = np.random.default_rng(s).integers(0, 20_000, size=(PROTOCOL["K_sens"], PROTOCOL["batch"]))
        assert np.array_equal(long[:PROTOCOL["K_sens"]], short)
    # grid control on synthetic curves: R_i(lam) = 20 / (1 + (lam / h_K)^2) + small repetition noise
    Pg = {**PROTOCOL}
    h = {PROTOCOL["K"]: 0.2, PROTOCOL["K_sens"]: 2.0}
    calls = []

    def point(lam, Ks):
        calls.append((lam, tuple(Ks)))
        return {K: [{"return": 20.0 / (1 + (lam / h[K]) ** 2) + 0.1 * i, "cost": 1.0} for i in range(3)] for K in Ks}

    out = play_grid(point, 20.0, 0.2, Pg)
    bK, bS = out[str(PROTOCOL["K"])], out[str(PROTOCOL["K_sens"])]
    assert bK["eligible"] and bS["eligible"] and not bK["censor"] and not bS["censor"]
    assert abs(bK["lam_half"] / 0.2 - 1) < 0.25 and abs(bS["lam_half"] / 2.0 - 1) < 0.25, (bK["lam_half"], bS["lam_half"])
    assert len(bK["grid"]) < len(bS["grid"]) <= 21 and all(len(c[1]) == 1 for c in calls[len(bK["grid"]) + 1:])
    assert abs(bK["R0"]["return"] - 20.1) < 1e-12 and bK["spread"]["R_range_median"] > 0.19
    out = play_grid(lambda lam, Ks: {K: [{"return": 5.0, "cost": 0.0}] * 3 for K in Ks}, 20.0, 0.2, Pg)
    assert all(not b["eligible"] and b["grid"] == [] and "R0 < min_return" in b["why"] for b in out.values())
    # the rule on synthetic numbers
    lams = PROTOCOL["grid"][1:]
    lh, c = halving_point([0.1, 1.0], [8.0, 2.0], 10.0)        # R reaches 5 half-way from 8 to 2
    assert c == "" and abs(lh - 10 ** -0.5) < 1e-12, lh
    assert halving_point(lams, [4.0] + [None] * 20, 10.0) == (lams[0], "left")
    assert halving_point(lams, [9.0] * 21, 10.0) == (lams[-1], "right")
    lh, c = halving_point([0.1, 0.2, 0.4], [9.0, 4.0, 9.0], 10.0)  # the first crossing counts, not a later recovery
    assert c == "" and 0.1 < lh < 0.2, lh
    assert abs(log2_rho(0.2, "", 0.1) - 1.0) < 1e-12 and log2_rho(100.0, "right", 10.0) > math.log2(10) + 0.99
    for xs, want in (([0.1] * 10, "SUPPORTED"), ([3.0] * 10, "REFUTED"), ([-3.0] * 10, "REFUTED"),
                     ([0.0, 0.0, 0.0, 0.0, 0.0, 3, 3, 3, 3, 3], "INCONCLUSIVE"), ([0.0] * 4, "NOT EVALUABLE")):
        got = verdict(xs, PROTOCOL)["label"]
        assert got == want, (xs, got, want)
    v = verdict(np.log2([1.0, 2.0, 8.0, 9.0, 16.0, 32.0]), PROTOCOL)   # even n: geometric mean of the middle pair
    assert abs(v["rho_hat"] - math.sqrt(8.0 * 9.0)) < 1e-9 and abs(v["median_log2_rho"] - math.log2(72) / 2) < 1e-12
    L = ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "NOT EVALUABLE")
    want = {("SUPPORTED", "SUPPORTED"): "SUPPORTED", ("INCONCLUSIVE", "INCONCLUSIVE"): "INCONCLUSIVE",
            ("SUPPORTED", "INCONCLUSIVE"): "INCONCLUSIVE", ("INCONCLUSIVE", "SUPPORTED"): "INCONCLUSIVE"}
    for x in L:
        for y in L:
            exp = ("REFUTED" if "REFUTED" in (x, y) else want.get((x, y), "NOT EVALUABLE"))
            assert overall_label([x, y]) == exp, (x, y, overall_label([x, y]))
    assert (set_title("S1", "co-primary").startswith("S1 [EXPLORATORY ") and set_title("S3a", "descriptive") ==
            "S3a [DESCRIPTIVE]" and set_title("X", "exploratory").startswith("X [EXPLORATORY "))
    assert abs(spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) - 1.0) < 1e-12
    assert len(PROTOCOL["grid"]) == 22 and PROTOCOL["grid"][1] == 0.001 and PROTOCOL["grid"][-1] == 100.0
    print(f"grid: {len(PROTOCOL['grid'])} values, 0 then {PROTOCOL['grid'][1]} .. {PROTOCOL['grid'][-1]}; batch "
          f"prefix property at K={PROTOCOL['K']:,} / {PROTOCOL['K_sens']:,} for seeds {PROTOCOL['rep_seeds']}; "
          f"play_grid, halving_point, log2_rho, verdict, overall_label and spearman pass their synthetic checks")
    print("dry run OK (no file read, no environment built)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--prereg", action="store_true", help="the declared sets and protocol")
    m.add_argument("--stems", nargs="+", help="DIR/STEM@STEP (STEP = a snapshot step or 'final'); EXPLORATORY")
    p.add_argument("--out", default="results/probe")
    p.add_argument("--root", default=".", help="repository root for the declared sets")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="tiny constants for plumbing tests; never with --prereg")
    p.add_argument("--summarise-only", action="store_true")
    a = p.parse_args(argv)
    if a.dry_run:
        return dry_run()
    if a.prereg and a.quick:
        p.error("--quick changes the protocol and cannot be used with --prereg")
    P = QUICK if a.quick else PROTOCOL
    if a.prereg:
        sets, jobs = SETS, declared_jobs(a.root, P)
        header = ("EXPLORATORY: gradient-parity probe (prereg.md section 9; outside the Holm family; protocol "
                  "fixed in lagu/probe.py before any study snapshot was read)")
    else:
        sets = [("X", "exploratory", "", "--stems", (), 0)]
        jobs = []
        for s in a.stems:
            stem, step = s.rsplit("@", 1)
            jobs.append({"kind": "main", "set": "X", "role": "exploratory", "stem": stem,
                         "step": step if step == "final" else int(step)})
        header = "EXPLORATORY: gradient-parity probe on hand-picked runs (--stems; not the pre-registered sets)"
    if a.quick:
        header = "QUICK PLUMBING TEST - NOT THE PRE-REGISTERED PROTOCOL\n" + header
    os.makedirs(a.out, exist_ok=True)
    results = run_all(jobs, P, a.out, a.workers, a.summarise_only)
    summary, text = summarise(results, P, sets, header)
    missing = [r for r in results if r.get("error") == "missing"]
    failed = [r for r in results if "error" in r and r["error"] != "missing"]
    status = 2 if missing else 1 if failed else 0
    text += (f"\n\nSTATUS: {len(missing)} declared files missing, {len(failed)} jobs failed or not run; "
             f"exit status {status}")
    print(text)
    with open(os.path.join(a.out, "probe_tables.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(a.out, "probe.json"), "w") as f:
        json.dump({"protocol": P, "summary": summary, "results": results, "exit_status": status}, f, indent=1,
                  default=str)
    figures(results, sets, a.out, header.split("\n")[0], P)
    return status


if __name__ == "__main__":
    sys.exit(main())
