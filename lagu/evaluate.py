"""Held-out evaluation of saved snapshots (the primary endpoint).

    python -m lagu.evaluate results/gate results/gate_literal [--end-step N] [--snapshots 5] \
        [--episodes 20] [--workers 12] [--force]

Every run in the given directories with a <stem>.json and a <stem>.snap/ is evaluated at the saved
steps nearest N * {0.8, 0.85, 0.9, 0.95, 1.0} (--snapshots points evenly over the last 20%), where
N is --end-step or the run's --steps. Snapshot i plays deterministic episodes on the layouts
EVAL_SEED + 1000 + 20 i + k, k < episodes: disjoint from the in-training layouts (EVAL_SEED + k) and
shared by every run, so seeds compare pairwise. Runs that have not reached N are skipped.

Writes <stem>.heldout.json with train.evaluate()'s output (return, cost, violation and critic
calibration) for each snapshot and pooled; every snapshot plays the same number of episodes, so the
pooled value is the mean over all of them, and each snapshot records the SHA-256 of its file. A
run whose heldout.json exists is skipped unless --force; if that file was written with other
settings or from snapshot files that have changed since, it is left alone, reported, and the exit
status is 1. So is a run whose snapshots go past the --steps in its json (extended with --resume):
pass --end-step for it. Runs are spread over --workers processes with one torch thread each.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import safety_gymnasium
import torch

from lagu.agent import Config, LagU
from lagu.train import EVAL_SEED, evaluate

HELDOUT_SEED = EVAL_SEED + 1000
STRIDE = 20   # layouts reserved per snapshot, so the blocks never overlap


def snapshot_steps(snap_dir: str) -> list[int]:
    names = (os.path.basename(p)[:-3] for p in glob.glob(os.path.join(snap_dir, "*.pt")))
    return sorted(int(n) for n in names if n.isdigit())


def pick(steps: list[int], end: int, n: int) -> list[tuple[float, int]]:
    """(target, saved step nearest it) for each of end * linspace(0.8, 1.0, n); ties go earlier."""
    targets = np.linspace(0.8, 1.0, n) * end if n > 1 else [float(end)]
    return [(float(t), min(steps, key=lambda s: (abs(s - t), s))) for t in targets]


def digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_agent(meta: dict, snap: dict) -> LagU:
    """Every network evaluate() reads must come from the snapshot, never from a random init."""
    agent = LagU(meta["obs_dim"], meta["act_dim"], Config(**snap["cfg"]))
    for name in ("actor", "members", "cost") + (("cost2",) if agent.cfg.cost_twin else ()):
        getattr(agent, name).load_state_dict(snap[name])
    return agent


def _init_worker() -> None:
    torch.set_num_threads(1)


def _evaluate_run(job: tuple) -> tuple[str, str]:
    stem, end, chosen, episodes = job
    name = os.path.basename(stem)
    try:
        with open(stem + ".json") as f:
            meta = json.load(f)
        env = safety_gymnasium.make(meta["args"]["env"])
        per = []
        for i, (target, step) in enumerate(chosen):
            path = os.path.join(stem + ".snap", f"{step}.pt")
            agent = load_agent(meta, torch.load(path, weights_only=True))
            base = HELDOUT_SEED + STRIDE * i
            ev = evaluate(env, agent, episodes, agent.cfg.cost_limit, agent.cfg.gamma, meta["horizon"],
                          seed_base=base)
            per.append({"index": i, "target": target, "step": step, "seed_base": base, "sha256": digest(path),
                        **ev})
        env.close()
        pooled = {k: float(np.mean([s[k] for s in per])) for k in ev}
        out = {"stem": name, "end_step": end, "snapshots": len(chosen), "episodes": episodes,
               "per_snapshot": per, "pooled": pooled}
        tmp = stem + ".heldout.json.tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, stem + ".heldout.json")
        steps = [s for _, s in chosen]
        note = "  (repeated snapshot steps)" if len(set(steps)) < len(steps) else ""
        return name, (f"ret={pooled['eval_return']:.1f} cost={pooled['eval_cost']:.1f} "
                       f"viol={pooled['eval_violation']:.2f} qc/mc={pooled['qc_traj']:.2f}/"
                       f"{pooled['mc_cost_traj']:.2f} steps={steps}{note}")
    except Exception as e:   # one bad run must not abort the others
        return name, f"ERROR {type(e).__name__}: {e}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="+")
    p.add_argument("--end-step", type=int, default=None, help="N; default = each run's --steps")
    p.add_argument("--snapshots", type=int, default=5)
    p.add_argument("--episodes", type=int, default=20, help=f"per snapshot, at most {STRIDE}")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--force", action="store_true", help="re-evaluate runs that have a heldout.json")
    a = p.parse_args(argv)
    if not (1 <= a.episodes <= STRIDE and a.snapshots >= 1):
        p.error(f"need 1 <= --episodes <= {STRIDE} (layout blocks would overlap) and --snapshots >= 1")

    jobs, failed = [], 0
    for d in a.dirs:
        if not os.path.isdir(d):
            failed += 1
            print(f"{d}: no such directory")
            continue
        snap_dirs = [s for s in sorted(glob.glob(os.path.join(d, "*.snap"))) if os.path.exists(s[:-len(".snap")] + ".json")]
        if not snap_dirs:
            print(f"{d}: no runs with snapshots")
        for snap_dir in snap_dirs:
            stem = snap_dir[:-len(".snap")]
            name = os.path.basename(stem)
            with open(stem + ".json") as f:
                end = a.end_step or int(json.load(f)["args"]["steps"])
            steps = snapshot_steps(snap_dir)
            if not steps or steps[-1] < end:
                print(f"{name}: skipped, last snapshot {steps[-1] if steps else None} < end step {end}")
                continue
            if a.end_step is None and steps[-1] > end:
                failed += 1
                print(f"{name}: snapshots go to {steps[-1]}, past the {end} steps in its json (extended with "
                      f"--resume?); pass --end-step")
                continue
            chosen = pick(steps, end, a.snapshots)
            out = stem + ".heldout.json"
            if os.path.exists(out) and not a.force:
                with open(out) as f:
                    prev = json.load(f)
                if (prev["end_step"], prev["snapshots"], prev["episodes"]) != (end, a.snapshots, a.episodes):
                    why = f"end_step={prev['end_step']} snapshots={prev['snapshots']} episodes={prev['episodes']}"
                elif [s.get("sha256") for s in prev["per_snapshot"]] != [
                        digest(os.path.join(snap_dir, f"{s}.pt")) for _, s in chosen]:
                    why = "snapshot files that have changed since"
                else:
                    print(f"{name}: done already")
                    continue
                failed += 1
                print(f"{name}: heldout.json was made from {why}; left alone (use --force to overwrite)")
                continue
            jobs.append((stem, end, chosen, a.episodes))

    if a.workers > 1 and len(jobs) > 1:
        pool = mp.get_context("spawn").Pool(min(a.workers, len(jobs)), initializer=_init_worker)
        results = pool.imap_unordered(_evaluate_run, jobs)
    else:
        pool = None
        _init_worker()
        results = map(_evaluate_run, jobs)
    for name, msg in results:
        failed += msg.startswith("ERROR")
        print(f"{name}: {msg}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
