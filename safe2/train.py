"""Shared trainer of the method study: every baseline, SHL and the BVS family (safe2/SPEC.md section 3).

    python -m safe2.train --config NAME --env ENV --seed S --out DIR
                          [--steps 1000000] [--resume | --restart] [--hidden 256] [--drop-ckpt 0|1]
                          [--from-bvs NAME --rho-star X]      # BVSU / BVSD only
                          [--eval-every N --eval-episodes K --snap-every N --snap-from F --ckpt-every N]  # dev only

The run stem is <config>_<env>_s<seed>. Under results/method/ the trainer refuses to start unless
--steps is 1,000,000 (outside cont3m/) or 3,000,000 with --resume (only in cont3m/, PointGoal1, seeds
300-304), --hidden is 256, no dev-only flag is given and the seed is a study training seed. Outside
results/method/ (development) everything is accepted, except that a dev seed (900-999, not blinded)
never trains more than DEV_MAX_STEPS steps.

Files in DIR: <stem>.csv (public, one row per in-training evaluation), <stem>.json (args, config,
dims, resumes), <stem>.snap/<step>.pt (weights-only snapshots from 0.8 N), <stem>.ckpt.pt (complete
state, atomic, every 100k), <stem>.pt (final file, atomic and fsynced: the complete agent state, or
for a blinded seed the network weights only).

Starts, resumes and continuations (SPEC 3.6).
- A run that has started before (its json, checkpoint or snapshots exist) is never started from
  scratch silently: with a checkpoint it is continued with --resume; without one (it crashed before
  its first checkpoint) it is started again only with --restart, which keeps its history and counts as
  a resume.
- --resume checks that the checkpoint is this run's (config, env, seed) and is a no-op on a run whose
  checkpoint is already at --steps (nothing is written).
- n_resumes counts every resume and restart of the run, even when it crashes again before its next
  checkpoint; the launcher never resumes a run with 3.
- A continuation (--resume from the final checkpoint of a shorter run, e.g. a 1M checkpoint copied
  into cont3m/ and run with --steps 3000000) is recorded in the json's "continuations"; it is not a
  resume, and the continuation's n_resumes counts only its own resumes.

Blinding (prereg_method.md section 10). For a confirmatory training seed (>= 300, outside the dev
range 900-999) the public CSV holds only HEALTH_FIELDS, every outcome row is sealed to
<root>/sealed/<stage>/<stem>.rows (safe2.seal), the console prints only
"[stem] step=<step> sps=<sps> finite=<0|1>", and an uncaught error prints only its type and the step.
Through the CLI the process's own stdout and stderr are pointed at /dev/null for such runs, so
library warnings never reach the launcher's log.

Base arms consume every RNG (action-space sampling, agent.rng, rng_replay, the torch global RNG) in
the order of lagu/train.py, so R1 reproduces `lagu.train --arm td3lag --recipe omnisafe` with the
same seed (tests/test_base.py). The trainer calls agents only through base.TrainerHooks
(safe2/HOOKS.md). It never raises on non-finite values: a non-finite action is replaced by zeros
and counted, and a non-finite whitelisted value sets the row's finite flags to 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import warnings
from collections import deque
from dataclasses import asdict

import numpy as np
import torch

from safe2 import configs, evaluate, seal, seeds
from safe2.base import agent_class, budget

HEALTH_FIELDS = ["step", "episodes", "updates", "sps", "wall_s", "finite_out", "finite_net", "finite_loss",
                 "finite_dual", "nonfinite_actions", "resumed", "n_resumes"]
OUTCOME_FIELDS = [
    # outcomes
    "train_return", "train_cost", "eval_return", "eval_cost", "eval_violation", "cum_cost", "cum_cost_200k",
    "cum_viol_eps",
    # network
    "qbar", "qc", "vH_exec_mean", "u_p50",
    # losses
    "q_loss", "pi_loss", "c_loss", "v_loss", "pib_loss",
    # dual
    "lambda", "jc", "pid_i", "pid_p", "pid_d", "m_tilde", "m", "m_dep", "T",
    # calibration (base)
    "qc0", "mc_cost0", "qc_traj", "mc_cost_traj", "qbar_traj", "mc_ret_traj",
    # BVS counters (cumulative) and in-training evaluation counts
    "int_cum", "shield_steps_cum", "empty_cum", "trig_cum", "F_cum", "ep_cum", "entries_cum", "misses_cum",
    "rho_num", "rho_den", "trig_num", "trig007_num", "eval_int_steps", "eval_empty_steps",
]
_BVS_WL = ["train_return", "train_cost", "eval_return", "eval_cost", "q_loss", "pi_loss", "qbar", "v_loss",
           "pib_loss", "vH_exec_mean", "m_tilde"]
WHITELIST = {
    "base": ["train_return", "train_cost", "eval_return", "eval_cost", "q_loss", "pi_loss", "qbar", "c_loss", "qc"],
    "bvs": _BVS_WL, "bvs_disc": _BVS_WL, "bvs_u": _BVS_WL + ["T", "u_p50"],
}
GROUPS = {
    "finite_out": ("train_return", "train_cost", "eval_return", "eval_cost"),
    "finite_net": ("qbar", "qc", "vH_exec_mean", "u_p50"),
    "finite_loss": ("q_loss", "pi_loss", "c_loss", "v_loss", "pib_loss"),
    "finite_dual": ("lambda", "m_tilde", "T"),
}
STUDY_STEPS = 1_000_000
CONTINUATION_STEPS = 3_000_000
CONTINUATION_STAGE = "cont3m"                  # the only directory under results/method/ that runs 3M
CONTINUATION_ENV = "SafetyPointGoal1-v0"       # prereg_method.md section 9: continuations on PointGoal1 only
DEV_MAX_STEPS = 50_000                         # dev seeds are not blinded, so they stay smoke runs


def whitelist(rc: configs.RunConfig) -> list[str]:
    """The finite whitelist of an arm (SPEC 3.7): a non-finite value in any row marks divergence."""
    wl = list(WHITELIST[rc.kind])
    if rc.kind == "base" and rc.dual != "none":
        wl.append("lambda")
    return wl


def inf_allowed(rc: configs.RunConfig) -> tuple:
    """Whitelisted columns that may legitimately be infinite: BVS-U's T is +inf for rho* = 0 (never
    triggers) and -inf for rho* = 1 (always triggers), prereg A.9. NaN never is."""
    return ("T",) if rc.kind == "bvs_u" and rc.rho_star in (0.0, 1.0) else ()


def _finite(v, written_before: bool = False, inf_ok: bool = False) -> bool:
    """'' (not computed yet, e.g. before the first update) counts as finite only while the column has
    never been written in this run: once written, an empty cell is a regression and counts as
    non-finite. Anything else must be a finite number (or +-inf when inf_ok)."""
    if v == "" or v is None:
        return not written_before
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(x)) or (inf_ok and bool(np.isinf(x)))


def finite_flags(row: dict, wl: list[str], written: set | frozenset = frozenset(), inf_ok: tuple = ()) -> dict:
    """1 per group when every whitelisted column of the group is finite in this row (a group with no
    whitelisted column is 1). `written` holds the whitelisted columns some earlier row of the run
    filled; `inf_ok` the columns that may be +-inf (inf_allowed)."""
    return {g: int(all(_finite(row.get(c, ""), c in written, c in inf_ok) for c in cols if c in wl))
            for g, cols in GROUPS.items()}


def blinded(seed: int) -> bool:
    """Outcomes are sealed for training seeds >= 300 (BLIND_FROM), except the dev seeds 900-999."""
    return seed >= seeds.BLIND_FROM and seed not in seeds.DEV_TRAINING_SEEDS


def under_method(path: str) -> bool:
    parts = os.path.realpath(path).split(os.sep)
    return any(parts[i] == "results" and parts[i + 1] == "method" for i in range(len(parts) - 1))


def stem_of(name: str, env_id: str, seed: int) -> str:
    return f"{name}_{env_id}_s{seed}"


# --- replay -----------------------------------------------------------------------------------------
class Replay:
    """Columns (SPEC 3.3): o, b, a_pi, r (learning reward), c, o2, b2, d (true termination), ep (global
    episode index from 0), t (step within the episode), plus the agent's replay_extras (a_exec and I
    for the BVS family). Sampling draws one rng.integers(0, n, batch) per call, as lagu.train.Replay."""

    def __init__(self, obs_dim: int, act_dim: int, size: int, extras: dict | None = None):
        cols = {"o": ((obs_dim,), np.float32), "b": ((1,), np.float32), "a_pi": ((act_dim,), np.float32),
                "r": ((), np.float32), "c": ((), np.float32), "o2": ((obs_dim,), np.float32),
                "b2": ((1,), np.float32), "d": ((), np.float32), "ep": ((), np.int32), "t": ((), np.int32)}
        for k, (shape, dtype) in (extras or {}).items():
            cols[k] = (tuple(shape), dtype)
        self.data = {k: np.zeros((size,) + shape, dtype) for k, (shape, dtype) in cols.items()}
        self.ptr, self.n, self.size = 0, 0, size

    def add(self, **row) -> None:
        """Every allocated column must be given; keys that are not allocated are ignored."""
        i = self.ptr
        for k, arr in self.data.items():
            arr[i] = row[k]
        self.ptr = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, rng: np.random.Generator, batch: int, keys=None) -> dict:
        idx = rng.integers(0, self.n, batch)
        return {k: self.data[k][idx] for k in (keys or self.data)}

    def order(self) -> np.ndarray:
        """Row indices from the oldest to the newest stored transition."""
        return np.arange(self.n) if self.n < self.size else (np.arange(self.ptr, self.ptr + self.size) % self.size)

    def state(self) -> dict:
        return {**{k: torch.from_numpy(v) for k, v in self.data.items()}, "ptr": self.ptr, "n": self.n}

    def load(self, s: dict) -> None:
        """Restore a saved buffer (tensors or arrays); if the sizes differ keep the newest transitions
        that fit (as lagu.train.Replay.load)."""
        s = {k: (np.asarray(v) if k not in ("ptr", "n") else v) for k, v in s.items()}
        old, n, ptr = int(s["o"].shape[0]), int(s["n"]), int(s["ptr"])
        order = np.arange(n) if n < old else (np.arange(ptr, ptr + old) % old)
        keep = order[-min(n, self.size):]
        k = len(keep)
        for name, arr in self.data.items():
            arr[:k] = s[name][keep]
        self.ptr, self.n = k % self.size, k


# --- files ------------------------------------------------------------------------------------------
def _fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_atomic(obj, path: str, fsync: bool = True) -> None:
    """torch.save to path.tmp, fsync, os.replace: a crash never leaves a truncated file at path."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    if fsync:
        _fsync_file(tmp)
    os.replace(tmp, path)


def write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def read_rows(path: str, fieldnames: list[str]) -> list[dict]:
    """Rows of a public CSV with the expected header; a torn last line from a crash is dropped."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fieldnames:
            raise SystemExit(f"{path}: CSV columns differ from this trainer's schema; refusing to resume")
        rows = []
        for r in reader:
            if None in r.values() or None in r or not r.get("step", "").isdigit():
                continue
            rows.append(r)
    return rows


def truncate_csv(path: str, last_step: int, fieldnames: list[str]) -> None:
    rows = [r for r in read_rows(path, fieldnames) if int(r["step"]) <= last_step]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def snapshot_path(snap_dir: str, step: int, n_resumes: int) -> str:
    """<step>.pt, or <step>.r<k>.pt when that step already exists, with k = max(n_resumes, 1 + every k
    already written for that step): snapshots are never overwritten, and the newest file always has
    the largest k (evaluate.snapshot_file picks it)."""
    path = os.path.join(snap_dir, f"{step}.pt")
    if not os.path.exists(path):
        return path
    pat = re.compile(rf"^{step}\.r(\d+)\.pt$")
    ks = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(snap_dir)) if m]
    return os.path.join(snap_dir, f"{step}.r{max([n_resumes] + [k + 1 for k in ks])}.pt")


def read_meta(path: str) -> dict | None:
    """The run json, or None when it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def prior_resumes(meta: dict | None, ckpt_count: int = 0) -> int:
    """Resumes and restarts recorded so far: the largest of the checkpoint's count, the json's count and
    the json's list (a run that crashes again before its next checkpoint still counts every resume)."""
    meta = meta or {}
    return max(int(ckpt_count), int(meta.get("n_resumes", 0)), len(meta.get("resumes", [])))


# --- the run ----------------------------------------------------------------------------------------
def _make_env(env_id: str):
    import safety_gymnasium        # imported here so a blinded run's import warnings go to /dev/null
    return safety_gymnasium.make(env_id)


def run(rc: configs.RunConfig, env_id: str, seed: int, out: str, steps: int = STUDY_STEPS, resume: bool = False,
        drop_ckpt: bool = False, make_env=None, console=None, on_eval=None, progress: dict | None = None,
        argv: list | None = None, restart: bool = False) -> str:
    """Train one run; returns its stem. make_env(env_id) builds both environments (tests pass fakes);
    console(line) prints a console line; on_eval(step, episodes) sees every in-training evaluation's
    episodes (tests); progress["step"] tracks the current step for error reports. restart allows a run
    that started before but has no checkpoint to start again from scratch (see the module docstring)."""
    if seed in seeds.DEV_TRAINING_SEEDS and steps > DEV_MAX_STEPS:
        raise SystemExit(f"dev seed {seed} with {steps} steps: dev seeds are not blinded and stay below "
                         f"{DEV_MAX_STEPS} steps")
    torch.set_num_threads(1)
    make_env = make_env or _make_env
    console = console or (lambda line: print(line, flush=True))
    progress = progress if progress is not None else {}
    blind = blinded(seed)
    status = (lambda line: None) if blind else console     # blinded console: step lines only (SPEC 3.8)
    stem = stem_of(rc.name, env_id, seed)
    os.makedirs(out, exist_ok=True)
    path = {ext: os.path.join(out, stem + ext) for ext in (".csv", ".json", ".pt", ".ckpt.pt")}
    snap_dir = os.path.join(out, stem + ".snap")
    fields = HEALTH_FIELDS if blind else HEALTH_FIELDS + OUTCOME_FIELDS
    records = None
    if blind:
        root = seal.find_root(out)
        records = seal.RecordFile(root, os.path.relpath(os.path.realpath(out), os.path.realpath(root)), stem)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])    # replay sampling
    env = make_env(env_id)
    env_eval = make_env(env_id)
    env.action_space.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    horizon = int(getattr(getattr(env, "spec", None), "max_episode_steps", None) or 1000)
    agent_seed = int(np.random.SeedSequence(seed).spawn(2)[1].generate_state(1)[0])
    agent = agent_class(rc.kind)(obs_dim, act_dim, rc, horizon, seed=agent_seed)
    replay = Replay(obs_dim, act_dim, min(rc.replay, steps), extras=agent.replay_extras)
    wl, inf_ok = whitelist(rc), inf_allowed(rc)

    ckpt = None
    meta_old = read_meta(path[".json"])
    started = meta_old is not None or bool(os.path.isdir(snap_dir) and os.listdir(snap_dir))
    if resume and os.path.exists(path[".ckpt.pt"]):
        ckpt = torch.load(path[".ckpt.pt"], weights_only=True)
        diff = configs.config_diff(ckpt["config"], asdict(rc))
        if diff:
            raise SystemExit(f"[{stem}] config differs from the checkpoint {diff}; refusing to resume")
        if ckpt.get("env") != env_id or int(ckpt.get("seed", -1)) != seed:
            raise SystemExit(f"[{stem}] the checkpoint belongs to {ckpt.get('env')} seed {ckpt.get('seed')}; "
                             "refusing to resume")
        if int(ckpt["steps"]) > steps:
            raise SystemExit(f"[{stem}] the checkpoint is of a {int(ckpt['steps'])}-step run; refusing to resume "
                             f"it as a {steps}-step run")
        if int(ckpt["step"]) >= steps:                 # finished and kept its checkpoint: write nothing
            if not os.path.exists(path[".pt"]):        # (it crashed between its last checkpoint and the .pt)
                agent.load_state_dict(ckpt["agent"])
                save_atomic(_final(agent, rc, obs_dim, act_dim, horizon, steps, blind), path[".pt"])
            if drop_ckpt:
                os.remove(path[".ckpt.pt"])
            status(f"[{stem}] already complete")
            return stem
        if int(ckpt["steps"]) < steps and int(ckpt["step"]) != int(ckpt["steps"]):
            raise SystemExit(f"[{stem}] a continuation starts only from the final checkpoint of its run "
                             f"(this one is at step {int(ckpt['step'])} of {int(ckpt['steps'])})")
    elif resume and os.path.exists(path[".pt"]):
        last = 0
        if os.path.exists(path[".csv"]):
            done = [int(r["step"]) for r in read_rows(path[".csv"], fields)]
            last = max(done) if done else 0
        if last >= steps:
            status(f"[{stem}] already complete")
            return stem
        raise SystemExit(f"[{stem}] finished at step {last} with its checkpoint dropped; cannot extend to {steps}")
    elif resume and steps == CONTINUATION_STEPS and under_method(out):
        raise SystemExit(f"[{stem}] no checkpoint to continue from in {out}")
    if ckpt is None:
        if os.path.exists(path[".pt"]):
            raise SystemExit(f"[{stem}] a finished run exists in {out}; refusing to overwrite it")
        if os.path.exists(path[".ckpt.pt"]):
            raise SystemExit(f"[{stem}] a checkpoint exists in {out}; continue the run with --resume")
        if started and not restart:
            raise SystemExit(f"[{stem}] this run started before in {out} (json or snapshots exist) and has no "
                             "checkpoint; start it again from scratch only with --restart")

    start, episodes, ep_id, updates, nonfinite, wall_prev = 1, 0, 0, 0, 0, 0.0
    cum = {"cost": 0.0, "cost_200k": 0.0, "viol_eps": 0}
    resumed, n_resumes = 0, 0
    ep_rets, ep_costs = deque(maxlen=50), deque(maxlen=50)
    jc = deque(maxlen=rc.jc_window)
    written: set = set()                               # whitelisted columns some row has filled
    if ckpt is not None:
        agent.load_state_dict(ckpt["agent"])
        replay.load(ckpt["replay"])
        rng.bit_generator.state = ckpt["rng_replay"]
        torch.set_rng_state(ckpt["torch_rng"])
        start = int(ckpt["step"]) + 1
        episodes, ep_id, updates = int(ckpt["episodes"]), int(ckpt["ep_id"]), int(ckpt["updates"])
        ep_id += int(ckpt["t"] > 0)                   # the interrupted episode is abandoned
        nonfinite, wall_prev = int(ckpt["nonfinite"]), float(ckpt["wall_s"])
        ep_rets.extend(ckpt["ep_rets"]), ep_costs.extend(ckpt["ep_costs"]), jc.extend(ckpt["jc"])
        cum.update(ckpt["cum"])
        written = set(ckpt.get("written", []))
        meta = meta_old or _meta(rc, env_id, seed, steps, out, drop_ckpt, argv, obs_dim, act_dim, horizon, blind)
        continuing = int(ckpt["steps"]) < steps       # started from a shorter run's final checkpoint
        if continuing and not any(c["from_step"] == start - 1 for c in meta.get("continuations", [])):
            meta.setdefault("continuations", []).append({"from_step": start - 1, "steps": steps, "argv": argv,
                          "previous_n_resumes": prior_resumes(meta_old, int(ckpt["n_resumes"])),
                          "previous_resumes": meta.get("resumes", [])})
            meta["resumes"] = []
        else:                                          # a resume (counted, whatever the checkpoint says)
            n_resumes = prior_resumes(meta, 0 if continuing else int(ckpt["n_resumes"])) + 1
            resumed = 1
            meta.setdefault("resumes", []).append({"from_step": start - 1, "steps": steps, "argv": argv})
        ckpt = None                                    # release the replay copy
        if os.path.exists(path[".csv"]):
            truncate_csv(path[".csv"], start - 1, fields)        # refuses a CSV of another schema
        else:
            with open(path[".csv"], "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
        if records is not None:
            records.truncate(start - 1)
        meta["n_resumes"] = n_resumes
        meta["args"]["steps"] = steps
        write_json(path[".json"], meta)
        o, _ = seeds.reset(env, "resume", seed, start=start)
        status(f"[{stem}] {'resumed' if resumed else 'continued'} after step {start - 1}")
    else:
        meta = _meta(rc, env_id, seed, steps, out, drop_ckpt, argv, obs_dim, act_dim, horizon, blind)
        if started:                                    # --restart: the history is kept; it counts as a resume
            n_resumes, resumed = prior_resumes(meta_old) + 1, 1
            meta["resumes"] = (meta_old or {}).get("resumes", []) + [
                {"from_step": 0, "restart": True, "steps": steps, "argv": argv}]
            meta["n_resumes"] = n_resumes
        write_json(path[".json"], meta)
        with open(path[".csv"], "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
        if records is not None:
            records.truncate(0)
        o, _ = seeds.reset(env, "train", seed)
    csv_f = open(path[".csv"], "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fields, restval="", extrasaction="ignore")

    agent.begin_episode()
    C = ep_ret = ep_cost = 0.0
    t = 0
    t0 = t_last = time.time()
    step_last = start - 1
    zeros = np.zeros(act_dim, np.float32)
    d = rc.cost_limit
    for step in range(start, steps + 1):
        progress["step"] = step
        agent.env_step = step
        b = budget(C, d)
        x = np.concatenate([o, [b]]) if agent.uses_budget else o
        if step <= rc.start_steps:
            a_pi = a_exec = env.action_space.sample()
            I, info = 0, {}
        else:
            a_pi, a_exec, I, info = agent.select_action(x, o, b, C, step)
        bad = False
        if not np.all(np.isfinite(a_exec)):
            a_exec, bad = zeros, True
        if not np.all(np.isfinite(a_pi)):
            a_pi, bad = zeros, True
        nonfinite += int(bad)
        o2, r, c, term, trunc, _ = env.step(a_exec)
        r, c = float(r), float(c)
        r_learn = -rc.saute_n if (rc.saute_n > 0 and C >= d) else r      # Saute: z_t <= 0 <=> C_t >= d
        C2 = C + c
        replay.add(o=o, b=b, a_pi=a_pi, a_exec=a_exec, r=r_learn, c=c, o2=o2, b2=budget(C2, d), d=float(term),
                   ep=ep_id, t=t, I=I, **info.get("replay", {}))
        agent.on_step(step, t, c, I, info)
        ep_ret += r
        ep_cost += c
        cum["cost"] += c
        if step > 200_000:
            cum["cost_200k"] += c
        o, C, t = o2, C2, t + 1
        if term or trunc:
            ep_rets.append(ep_ret)
            ep_costs.append(ep_cost)
            if step > rc.start_steps:                  # warm-up episodes never feed the dual
                jc.append(ep_cost)
            cum["viol_eps"] += int(ep_cost > d)
            agent.end_episode(step, ep_ret, ep_cost)
            episodes += 1
            ep_id += 1
            C = ep_ret = ep_cost = 0.0
            t = 0
            o, _ = seeds.reset(env, "next")
            agent.begin_episode()
        if step > rc.start_steps:
            agent.learn(replay.sample(rng, rc.batch, agent.sample_keys), step)
            updates += 1
        agent.post_update(step, jc)

        if step % rc.eval_every == 0 or step == steps:
            policy = evaluate.policy_from_agent(agent, "deployed")
            eps = evaluate.rollout(env_eval, policy, "train_eval", range(rc.eval_episodes),
                                   keep_traj=agent.eval_keep_traj)
            if on_eval is not None:
                on_eval(step, eps)
            now = time.time()
            sps = (step - step_last) / max(now - t_last, 1e-9)
            t_last, step_last = now, step
            costs = np.asarray([e["cost"] for e in eps])
            res = {"train_return": float(np.mean(ep_rets)) if ep_rets else "",
                   "train_cost": float(np.mean(ep_costs)) if ep_costs else "",
                   "eval_return": float(np.mean([e["return"] for e in eps])), "eval_cost": float(costs.mean()),
                   "eval_violation": float((costs > d).mean()), "cum_cost": cum["cost"],
                   "cum_cost_200k": cum["cost_200k"], "cum_viol_eps": cum["viol_eps"]}
            res.update(agent.log_row())
            res.update(agent.eval_columns(eps))
            res = {k: res.get(k, "") for k in OUTCOME_FIELDS}
            flags = finite_flags(res, wl, written, inf_ok)
            written |= {c for c in wl if res.get(c, "") != ""}
            row = {"step": step, "episodes": episodes, "updates": updates, "sps": sps,
                   "wall_s": wall_prev + now - t0, **flags, "nonfinite_actions": nonfinite,
                   "resumed": resumed, "n_resumes": n_resumes}
            if records is not None:
                records.append(step, {"step": step, **res})
            else:
                row.update(res)
            writer.writerow(row)
            csv_f.flush()
            os.fsync(csv_f.fileno())
            agent.reset_interval()
            if blind:
                console(f"[{stem}] step={step} sps={sps:.0f} finite={int(all(flags.values()))}")
            else:
                console(f"[{stem}] step={step} ret={res['eval_return']:.1f} cost={res['eval_cost']:.1f} "
                        f"{agent.console_dual()} sps={sps:.0f}")
        if step >= rc.snap_from * steps and (step % rc.snap_every == 0 or step == steps):
            os.makedirs(snap_dir, exist_ok=True)
            snap = {"step": step, "config": asdict(rc), "obs_dim": obs_dim, "act_dim": act_dim,
                    "horizon": horizon, **agent.snapshot()}
            save_atomic(snap, snapshot_path(snap_dir, step, n_resumes))
        if step % rc.ckpt_every == 0 or step == steps:
            save_atomic({"config": asdict(rc), "env": env_id, "seed": seed, "steps": steps,
                         "agent": agent.state_dict(), "replay": replay.state(), "step": step,
                         "episodes": episodes, "ep_id": ep_id, "t": t, "updates": updates,
                         "ep_rets": list(ep_rets), "ep_costs": list(ep_costs), "jc": list(jc), "cum": dict(cum),
                         "nonfinite": nonfinite, "wall_s": wall_prev + time.time() - t0,
                         "rng_replay": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                         "n_resumes": n_resumes, "written": sorted(written)}, path[".ckpt.pt"])
    save_atomic(_final(agent, rc, obs_dim, act_dim, horizon, steps, blind), path[".pt"])
    csv_f.close()
    if drop_ckpt and os.path.exists(path[".pt"]) and os.path.exists(path[".ckpt.pt"]):
        os.remove(path[".ckpt.pt"])
    return stem


def _final(agent, rc, obs_dim, act_dim, horizon, steps, blind: bool) -> dict:
    """The final <stem>.pt: the config and dims with the agent's complete state, or, for a blinded seed,
    only the network weights of its snapshot ("agent": {"actor": ..., "members": ..., ...}). The
    complete state holds outcome scalars in clear (the last training-cost mean, lambda, the PID or
    margin state, the BVS counters), and nothing needs it once the run is finished: resume reads the
    checkpoint, evaluation the snapshots."""
    state = agent.state_dict()
    if blind:
        state = {k: v for k, v in agent.snapshot().items() if isinstance(v, dict)}
    return {"config": asdict(rc), "obs_dim": obs_dim, "act_dim": act_dim, "horizon": horizon, "step": steps,
            "agent": state}


def _meta(rc, env_id, seed, steps, out, drop_ckpt, argv, obs_dim, act_dim, horizon, blind) -> dict:
    return {"args": {"config": rc.name, "env": env_id, "seed": seed, "steps": steps, "out": out,
                     "drop_ckpt": int(drop_ckpt), "argv": argv},
            "config": asdict(rc), "horizon": horizon, "obs_dim": obs_dim, "act_dim": act_dim, "blind": blind,
            "resumes": [], "n_resumes": 0}


# --- CLI ----------------------------------------------------------------------------------------------
def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=STUDY_STEPS)
    p.add_argument("--resume", action="store_true", help="continue from <stem>.ckpt.pt")
    p.add_argument("--restart", action="store_true",
                   help="start a run that started before but has no checkpoint again from scratch (counts as a "
                        "resume; safe2.launch --recover adds it)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--drop-ckpt", type=int, choices=[0, 1], default=0)
    p.add_argument("--from-bvs", default=None, help="BVSU / BVSD: the selected BVS config (BVS*)")
    p.add_argument("--rho-star", type=float, default=None, help="BVSU: the target trigger rate from v1.json")
    for f in ("eval_every", "eval_episodes", "snap_every", "ckpt_every"):
        p.add_argument("--" + f.replace("_", "-"), type=int, default=None, help="dev only")
    p.add_argument("--snap-from", type=float, default=None, help="dev only")
    return p.parse_args(argv)


def build_config(a: argparse.Namespace) -> configs.RunConfig:
    dev = {f: getattr(a, f) for f in configs.DEV_FIELDS if getattr(a, f) is not None}
    if a.config in configs.DERIVED:
        if a.from_bvs is None:
            raise SystemExit(f"{a.config} needs --from-bvs NAME (and --rho-star X for BVSU)")
        kind = configs.DERIVED[a.config]
        rho = a.rho_star if a.rho_star is not None else float("nan")
        return configs.from_bvs(a.from_bvs, kind, rho, hidden=a.hidden, **dev)
    if a.from_bvs is not None or a.rho_star is not None:
        raise SystemExit("--from-bvs / --rho-star are for BVSU and BVSD only")
    return configs.make(a.config, hidden=a.hidden, **dev)


def refusal(a: argparse.Namespace) -> str | None:
    """Why a run must not start (None = allowed): the study guard under results/method/ (SPEC 3.1), and
    anywhere a dev seed longer than DEV_MAX_STEPS (dev seeds are not blinded; a long dev run of a study
    config would be an unregistered look at outcomes)."""
    if a.resume and a.restart:
        return "--resume and --restart exclude each other"
    if not under_method(a.out):
        if a.seed in seeds.DEV_TRAINING_SEEDS and a.steps > DEV_MAX_STEPS:
            return f"dev seed {a.seed} with --steps above {DEV_MAX_STEPS}: dev seeds are for smoke runs only"
        return None
    if os.path.basename(os.path.realpath(a.out)) == CONTINUATION_STAGE:
        if not (a.steps == CONTINUATION_STEPS and a.resume):
            return f"{CONTINUATION_STAGE}/ holds only 3M continuations: --steps {CONTINUATION_STEPS} --resume"
        if a.seed not in seeds.CONTINUATION_SEEDS + seeds.EXTRA_CONTINUATION_SEEDS or a.env != CONTINUATION_ENV:
            return f"seed {a.seed} on {a.env} is not a 3M continuation ({CONTINUATION_ENV}, seeds 300-304)"
    elif a.steps != STUDY_STEPS:
        return (f"--steps must be {STUDY_STEPS} under results/method ({CONTINUATION_STEPS} only with --resume in "
                f"{CONTINUATION_STAGE}/)")
    if a.hidden != 256:
        return "--hidden must be 256 under results/method"
    used = [f for f in configs.DEV_FIELDS if getattr(a, f) is not None]
    if used:
        return f"dev-only flags under results/method: {used}"
    if a.seed not in seeds.STUDY_TRAINING_SEEDS:
        return f"seed {a.seed} is not a study training seed"
    return None


def _quiet_fds():
    """Point fds 1 and 2 at /dev/null and return a private line writer on the original stdout."""
    sys.stdout.flush()
    sys.stderr.flush()
    out = os.fdopen(os.dup(1), "w", buffering=1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    os.close(null)
    return lambda line: (out.write(line + "\n"), out.flush())


def main(argv: list[str] | None = None, quiet_fds: bool = False) -> int:
    a = parse(argv)
    why = refusal(a)
    if why:
        raise SystemExit(f"refusing to start: {why}")
    rc = build_config(a)
    stem = stem_of(rc.name, a.env, a.seed)
    blind = blinded(a.seed)
    console = None
    if blind:
        warnings.simplefilter("ignore")
        if quiet_fds:
            console = _quiet_fds()
    console = console or (lambda line: print(line, flush=True))
    progress = {"step": 0}
    try:
        run(rc, a.env, a.seed, a.out, steps=a.steps, resume=a.resume, drop_ckpt=bool(a.drop_ckpt),
            console=console, progress=progress, argv=list(sys.argv if argv is None else argv), restart=a.restart)
    except SystemExit as e:
        if blind and not isinstance(e.code, (int, type(None))):
            console(str(e.code))
            return 1
        raise
    except BaseException as e:      # a traceback could hold values: print the type and the step only
        if not blind:
            raise
        console(f"[{stem}] ERROR {type(e).__name__} step={progress['step']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(quiet_fds=True))
