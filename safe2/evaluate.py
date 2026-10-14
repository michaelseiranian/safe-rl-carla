"""Policies, rollouts and the sealed held-out evaluation of the method study (safe2/SPEC.md section 5).

    python -m safe2.evaluate DIR [DIR ...] [--mode deployed|shield_off|strict|s_near] [--end-step N]
                             [--workers 12] [--force] [--root ROOT]
    python -m safe2.evaluate --phs DIR [...] [--root ROOT]        # PHS on the B* runs of DIR (conf/)
    python -m safe2.evaluate --r3 LAGU_DIR --seeds 100-109 [--r3-prefix lagu-lit] [--root ROOT]

Every run in the given directories with a <stem>.json and a <stem>.snap/ is evaluated on the held-out
block 70000-70199: snapshot i (i = 0..9) is the file at exactly seeds.heldout_targets(N)[i] (the newest
resume's <step>.r<k>.pt when a resumed run rewrote it; no nearest-step fallback), N = --end-step or the
run's steps, and plays deterministic episodes on held-out indices 20 i + k, k < 20. Under
results/method/, N must be 1M (3M in cont3m/). The secondary modes skip the runs they do not apply to
(baselines; s_near on an S-N config). R3 replays the Lag-U study's lagu-lit snapshots (stems
<prefix>_*) at 800k-1M (j = 0..4) on indices 20 (2j) + k and 20 (2j + 1) + k. PHS (prereg A.9) plays
each of a B* run's 10 snapshots behind the BVS* shield with the run's offline fit
<root>/phs/<stem>.posthoc.pt and m = m_PHS(task) from <root>/v1.json.

Every output is on the held-out block, so it is ALWAYS sealed (safe2.seal) to
<root>/sealed/heldout/<stem>.<mode>.json.enc (.<mode>.n<N>.json.enc when N != 1M, so a 3M
continuation never takes its 1M run's name) whatever the training seed; the snapshot hashes (and the
PHS fit's) go in clear to the side file <stem>.<mode>[.n<N>].sha, written before the sealed file, so a
changed snapshot is detected without decryption. The console prints only "<stem>: done",
"<stem>: skipped (...)" or "<stem>: ERROR <type or code>". The sealing root is the enclosing
results/method of each run (or a directory holding keys/seal_pub.pem, for development); --root
overrides it.

The same rollout() serves in-training evaluation (safe2.train) and held-out evaluation, and
policy_from_snapshot() builds its policy through policy_from_agent(), so both produce identical
episodes (tests/test_train.py snapshot round trip).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys

import numpy as np
import torch

from safe2 import configs, seal, seeds
from safe2.base import agent_class, budget
from safe2.configs import RunConfig

MODES = ("deployed", "shield_off", "strict", "s_near")
SHIELD_MODE = {"deployed": "on", "shield_off": "off", "strict": "strict", "s_near": "on"}
NEAR_HAZARD = 0.6          # hazard distance below which saturation gaps are recorded (logging only)
ENTRY_LOOKBACK = 12        # an entry has no cost in the 12 steps before it (SPEC 2.4)
R3_STEPS = (800_000, 850_000, 900_000, 950_000, 1_000_000)
R3_PREFIX = "lagu-lit"     # R3 is the Lag-U study's lagu-lit arm only (prereg section 3.6)
STUDY_STEPS, CONTINUATION_STEPS, CONTINUATION_STAGE = 1_000_000, 3_000_000, "cont3m"   # = safe2.train's (tests)


class Refused(RuntimeError):
    """An evaluation refused for a reason printed as its fixed code (never a value)."""


# --- policies ---------------------------------------------------------------------------------------
class Policy:
    """Deterministic evaluation policy: begin(layout_id) at each episode start, act(o) per step, then
    observe(c) with the step's cost (tracks C and the budget input b). `info` holds the last step's
    details (always "x", the actor input; shielded policies add "I", "empty", "vmax")."""

    shielded = False

    def __init__(self, cost_limit: float = 25.0):
        self.d = cost_limit
        self.C = 0.0
        self.info: dict = {}

    def begin(self, layout_id: int) -> None:
        self.C = 0.0
        self.layout = layout_id
        self.info = {}

    @property
    def b(self) -> float:
        return budget(self.C, self.d)

    def act(self, o: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def observe(self, c: float) -> None:
        self.C += float(c)


class BasePolicy(Policy):
    """actor(x), x = cat(o, b) with the budget input (Saute), else o. Same arithmetic as LagU.act."""

    def __init__(self, actor: torch.nn.Module, budget_input: bool, cost_limit: float = 25.0):
        super().__init__(cost_limit)
        self.actor, self.budget_input = actor, budget_input

    @torch.no_grad()
    def act(self, o):
        x = np.concatenate([o, [self.b]]) if self.budget_input else o
        t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        a = np.clip(self.actor(t).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
        self.info = {"x": x}
        return a


class ShieldPolicy(Policy):
    """The section 2.4 shield around a BVS-family agent (`core`, from safe2.bvs).

    Per step: a_pi = core.propose(x, explore=False), then
    core.act(o, b, C, a_pi, m=core.m_dep, rng=self.rng, mode=SHIELD_MODE[mode], rule=rule) ->
    (a_exec, I, info), with rule = "SN" in mode s_near and core.rc.rule otherwise, and
    rng = default_rng(layout_id) created at begin(). BVS-U reads its own current T inside act().
    info must hold "empty" (bool) and "vmax" (max_j V_H at a_exec); see safe2/HOOKS.md."""

    shielded = True

    def __init__(self, core, mode: str):
        super().__init__(core.rc.cost_limit)
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "s_near" and core.rc.rule != "SQ":
            raise ValueError("mode s_near is run only for an S-Q config")
        self.core, self.mode = core, mode
        self.rule = "SN" if mode == "s_near" else core.rc.rule
        self.H = core.rc.H
        # BVS-disc's critic is a discounted cost-to-go clamped at qb_max: v_real is computed the same way
        self.disc = (core.rc.gamma_b, core.rc.qb_max) if core.rc.kind == "bvs_disc" else None
        self.rng = np.random.default_rng(0)

    def begin(self, layout_id: int) -> None:
        super().begin(layout_id)
        self.rng = np.random.default_rng(layout_id)

    @torch.no_grad()
    def act(self, o):
        b, C = self.b, self.C
        x = np.concatenate([o, [b]])
        a_pi = self.core.propose(x, explore=False)
        a, I, info = self.core.act(o, b, C, a_pi, m=self.core.m_dep, rng=self.rng,
                                   mode=SHIELD_MODE[self.mode], rule=self.rule)
        self.info = dict(info, I=int(I), x=x)
        return a

    @property
    def records_gap(self) -> bool:
        """Saturation gaps exist only for a headed V (BVS-disc's Q_b has no heads)."""
        return self.disc is None

    def gap(self, o, a) -> float:
        """Saturation gap max_j (V^j_H - V^j_{H-1})(o, a), provided by the core (logging only)."""
        return float(self.core.saturation_gap(o, a))


def phs_policy(bstar_snapshot: dict, posthoc_fit: dict, m_phs: float, rule: str) -> ShieldPolicy:
    """PHS (Appendix A.9): A.4's shield with B*'s deterministic actor as proposer, B*'s members[0] as
    the S-Q critic, the run's PHS V and pi_b (posthoc_fit: safe2.bvs.fit_offline's dict, as
    safe2.posthoc saves it), m = m_PHS(task), BVS*'s rule; deployed mode."""
    from safe2.bvs import PHSCore
    return ShieldPolicy(PHSCore(bstar_snapshot, posthoc_fit, m_phs, rule), "deployed")


class LaguPolicy(Policy):
    """R3: a Lag-U study snapshot (lagu.evaluate.load_agent), acting with agent.act(o, explore=False)."""

    def __init__(self, agent):
        super().__init__(agent.cfg.cost_limit)
        self.agent = agent

    def act(self, o):
        self.info = {"x": o}
        return self.agent.act(o, explore=False)


def policy_from_agent(agent, mode: str = "deployed") -> Policy:
    return agent.make_policy(mode)


def load_snapshot(path: str) -> dict:
    return torch.load(path, weights_only=True)


def policy_from_snapshot(path: str, mode: str = "deployed") -> Policy:
    """The same policy object policy_from_agent() builds from the live agent that wrote the snapshot."""
    snap = load_snapshot(path)
    rc = RunConfig(**snap["config"])
    agent = agent_class(rc.kind).from_snapshot(snap)
    return policy_from_agent(agent, mode)


# --- rollouts -----------------------------------------------------------------------------------------
def hazard_distance(env) -> float:
    """min over hazards of agent.dist_xy(h) - hazards.size (Safety-Gymnasium 1.0.0 task state)."""
    task = env.unwrapped.task
    hz = getattr(task, "hazards", None)
    if hz is None:                     # PointCircle1 has no hazards: no near-hazard state
        return float("inf")
    return float(min(task.agent.dist_xy(p) for p in hz.pos) - hz.size)


def shield_stats(I, empty, vmax, costs, H: int, n_allowed=(), disc: tuple | None = None) -> dict:
    """Per-episode BVS statistics (SPEC 2.4 and 5.2) from the per-step I, empty-set flag, vmax(o, a)
    and cost of one episode, plus the allowed-set sizes of its screening steps (the steps on which the
    shield scored its candidates; prereg 14.4): screen_steps and n_allowed_mean (NaN without one).
    v_real is the realised cost of steps t..t+H-1; for BVS-disc (disc = (gamma_b, qb_max)) it is the
    gamma_b-discounted cost from t to the episode end, clamped at qb_max, like its critic's targets."""
    I, empty, costs = np.asarray(I, int), np.asarray(empty, bool), np.asarray(costs, float)
    n_allowed = np.asarray(n_allowed, float)
    n = len(costs)
    first = int(np.argmax(empty)) if empty.any() else n
    entries = misses = 0
    for t in range(ENTRY_LOOKBACK, n):
        if costs[t] > 0 and not costs[t - ENTRY_LOOKBACK:t].any():
            entries += 1
            misses += int(not I[t - ENTRY_LOOKBACK:t].any())
    full = n - H + 1                                         # steps t with a full H-step future
    if full > 0:
        v_pred = float(np.mean(np.asarray(vmax, float)[:full]))
        if disc is None:
            csum = np.concatenate([[0.0], np.cumsum(costs)])
            v_real = float(np.mean(csum[H:H + full] - csum[:full]))
        else:
            gamma_b, qb_max = disc
            togo = np.zeros(n + 1)
            for t in range(n - 1, -1, -1):
                togo[t] = costs[t] + gamma_b * togo[t + 1]
            v_real = float(np.mean(np.minimum(togo[:full], qb_max)))
    else:
        v_pred = v_real = float("nan")
    return {"interventions": int(I.sum()), "empty_steps": int(empty.sum()), "F": float(costs[first:].sum()),
            "entries": entries, "misses": misses, "v_pred": v_pred, "v_real": v_real,
            "screen_steps": int(len(n_allowed)),
            "n_allowed_mean": float(n_allowed.mean()) if len(n_allowed) else float("nan")}


def rollout(env, policy: Policy, block: str, indices, record_gap: bool = False,
            keep_traj: bool = False) -> list[dict]:
    """Deterministic episodes on seeds.layout(block, i) for each i. Each episode records return, cost,
    layout, length and replaced non-finite actions; shielded policies add shield_stats(); keep_traj
    keeps the per-step actor inputs, rewards and costs ("traj"); record_gap records saturation gaps at
    near-hazard states ("gaps")."""
    out = []
    for i in indices:
        o, _ = seeds.reset(env, block, i)
        lid = seeds.layout(block, i)
        policy.begin(lid)
        rs, cs, xs, Is, empties, vmaxs, gaps, allowed = [], [], [], [], [], [], [], []
        nonfinite = 0
        done = False
        while not done:
            a = policy.act(o)
            if not np.all(np.isfinite(a)):
                a = np.zeros_like(a)
                nonfinite += 1
            if record_gap and hazard_distance(env) < NEAR_HAZARD:
                gaps.append(policy.gap(o, a))
            if keep_traj:
                xs.append(policy.info["x"])
            if policy.shielded:
                Is.append(policy.info["I"])
                empties.append(bool(policy.info.get("empty", False)))
                vmaxs.append(float(policy.info["vmax"]))
                if "n_allowed" in policy.info:
                    allowed.append(int(policy.info["n_allowed"]))
            o, r, c, term, trunc, _ = env.step(a)
            policy.observe(c)
            rs.append(float(r))
            cs.append(float(c))
            done = term or trunc
        ep = {"layout": lid, "return": float(np.asarray(rs).sum()), "cost": float(np.asarray(cs).sum()),
              "length": len(rs), "nonfinite_actions": nonfinite}
        if policy.shielded:
            ep.update(shield_stats(Is, empties, vmaxs, cs, policy.H, allowed, policy.disc))
        if record_gap:
            ep["gaps"] = gaps
        if keep_traj:
            ep["traj"] = {"x": xs, "r": rs, "c": cs}
        out.append(ep)
    return out


# --- held-out evaluation ----------------------------------------------------------------------------
def digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def snapshot_file(snap_dir: str, step: int) -> str:
    """<step>.pt, or the newest resume's <step>.r<k>.pt when a resumed run rewrote that step."""
    pat = re.compile(rf"^{step}\.r(\d+)\.pt$")
    ks = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(snap_dir)) if m]
    path = os.path.join(snap_dir, f"{step}.r{max(ks)}.pt" if ks else f"{step}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no snapshot at step {step} in {snap_dir}")
    return path


def heldout_plan(stem_path: str, end_step: int) -> list[tuple[int, int, str, list[int]]]:
    """(i, step, file, held-out indices) of the 10 snapshots of a run's held-out endpoint."""
    snap_dir = stem_path + ".snap"
    return [(i, s, snapshot_file(snap_dir, s), [seeds.HELDOUT_STRIDE * i + k for k in range(seeds.HELDOUT_EPISODES)])
            for i, s in enumerate(seeds.heldout_targets(end_step))]


def r3_plan(stem_path: str) -> list[tuple[int, int, str, list[int]]]:
    """R3: lagu snapshot j (800k, 850k, ..., 1M) plays held-out indices 20 (2j) + k and 20 (2j + 1) + k."""
    out = []
    for j, s in enumerate(R3_STEPS):
        path = os.path.join(stem_path + ".snap", f"{s}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        idx = [seeds.HELDOUT_STRIDE * (2 * j) + k for k in range(seeds.HELDOUT_EPISODES)] + \
              [seeds.HELDOUT_STRIDE * (2 * j + 1) + k for k in range(seeds.HELDOUT_EPISODES)]
        out.append((j, s, path, idx))
    return out


def output_paths(root: str, name: str, mode: str, end_step: int = STUDY_STEPS) -> tuple[str, str, str]:
    """(rel for seal.write_file, path of the sealed file, path of the clear .sha side file). An end step
    other than 1M adds ".n<N>" after the mode (safe2.analyze.heldout_rel)."""
    tag = f"{name}.{mode}" + ("" if end_step == STUDY_STEPS else f".n{end_step}")
    rel = os.path.join("heldout", f"{tag}.json")
    return rel, os.path.join(root, "sealed", rel + ".enc"), os.path.join(root, "sealed", "heldout", f"{tag}.sha")


def _make_env(env_id: str):
    import safety_gymnasium
    return safety_gymnasium.make(env_id)


def _init_worker() -> None:
    torch.set_num_threads(1)


def _write_sha(path: str, digests: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(digests, f)
    os.replace(tmp, path)


def _evaluate_job(job: dict) -> tuple[str, str]:
    """One run in one mode: play every planned snapshot, seal the output. Returns (name, message)."""
    name = job["name"]
    try:
        env = _make_env(job["env"])
        phs = job.get("phs")
        fit = torch.load(phs["fit"], weights_only=True) if phs else None
        snaps, eps_all = [], []
        for n, (i, step, path, idx) in enumerate(job["plan"]):
            if job["mode"] == "r3":
                from lagu.evaluate import load_agent
                policy = LaguPolicy(load_agent(job["meta"], torch.load(path, weights_only=True)))
            elif phs:
                policy = phs_policy(load_snapshot(path), fit, phs["m"], phs["rule"])
            else:
                policy = policy_from_snapshot(path, job["mode"])
            gap = job["mode"] == "deployed" and policy.shielded and policy.records_gap and n == len(job["plan"]) - 1
            eps = rollout(env, policy, "heldout", idx, record_gap=gap)
            snaps.append({"i": i, "step": step, "sha256": digest(path), "episodes": eps})
            eps_all += eps
        env.close()
        out = {"stem": name, "mode": job["mode"], "end_step": job["end_step"], "snapshots": snaps,
               "pooled": {"R": float(np.mean([e["return"] for e in eps_all])),
                          "C": float(np.mean([e["cost"] for e in eps_all]))}}
        if phs:
            out["phs"] = {"fit_sha256": digest(phs["fit"]), "m": phs["m"], "rule": phs["rule"]}
        if any("gaps" in e for e in snaps[-1]["episodes"]):
            out["gaps"] = [g for e in snaps[-1]["episodes"] for g in e.pop("gaps")]
        rel, enc, sha_path = output_paths(job["root"], name, job["mode"], job["end_step"])
        for p in (enc, sha_path):          # a forced re-evaluation removes the old pair first, then writes
            if os.path.exists(p):          # the .sha before the sealed file: no crash leaves a stale pair
                os.remove(p)
        _write_sha(sha_path, job["digests"])
        seal.write_file(job["root"], rel, out)
        return name, "done"
    except Exception as e:      # one bad run must not abort the others; print the type only
        return name, f"ERROR {type(e).__name__}"


def _parse_seeds(s: str) -> set[int]:
    lo, _, hi = s.partition("-")
    return set(range(int(lo), int(hi or lo) + 1))


def method_stage(d: str) -> str | None:
    """The stage directory name of d when d lies under results/method/ (None elsewhere)."""
    parts = os.path.realpath(d).split(os.sep)
    hits = [i for i in range(len(parts) - 1) if parts[i] == "results" and parts[i + 1] == "method"]
    return parts[-1] if hits and hits[-1] + 1 < len(parts) - 1 else None


def _runs(d: str):
    """(stem path, meta) of every run of directory d (a <stem>.json with a <stem>.snap/)."""
    for jpath in sorted(glob.glob(os.path.join(d, "*.json"))):
        stem = jpath[:-len(".json")]
        if os.path.isdir(stem + ".snap"):
            with open(jpath) as f:
                yield stem, json.load(f)


def not_applicable(meta: dict, mode: str) -> bool:
    """A secondary mode the run has no policy for: every shield mode of a baseline, s_near of S-N."""
    cfg = meta.get("config", {})
    return (mode != "deployed" and cfg.get("kind", "base") == "base") or (mode == "s_near" and cfg.get("rule") != "SQ")


def collect_jobs(a) -> tuple[list[dict], int]:
    jobs, failed = [], 0
    if a.r3:
        root = a.root or seal.DEFAULT_ROOT
        wanted = _parse_seeds(a.seeds)
        for stem, meta in _runs(a.r3):
            if not os.path.basename(stem).startswith(a.r3_prefix + "_") or int(meta["args"]["seed"]) not in wanted:
                continue
            jobs.append({"name": os.path.basename(stem), "stem": stem, "mode": "r3", "env": meta["args"]["env"],
                         "meta": meta, "end_step": R3_STEPS[-1], "root": root, "plan_fn": r3_plan})
        return jobs, failed
    for d in a.dirs:
        if not os.path.isdir(d):
            print(f"{d}: ERROR NotADirectoryError")
            failed += 1
            continue
        stage = method_stage(d)
        v1 = None
        if a.phs:
            root = a.root or seal.find_root(d)
            v1_path = os.path.join(root, "v1.json")
            if not os.path.exists(v1_path):
                raise SystemExit(f"--phs needs {v1_path} (B*, BVS* and m_PHS)")
            with open(v1_path) as f:
                v1 = json.load(f)
            if not v1.get("bvs_star"):
                raise SystemExit("--phs: v1.json has no BVS* (BVS was killed), so PHS is not run")
        for stem, meta in _runs(d):
            name = os.path.basename(stem)
            if v1 is not None:
                if meta["args"]["config"] != v1["b_star"]:
                    continue
                env = meta["args"]["env"]
                jobs.append({"name": name, "stem": stem, "mode": "phs", "env": env, "meta": meta,
                             "end_step": STUDY_STEPS, "root": root,
                             "phs": {"fit": os.path.join(root, "phs", name + ".posthoc.pt"),
                                     "m": float(v1["m_phs"][env]), "rule": configs.CONFIGS[v1["bvs_star"]].rule},
                             "plan_fn": lambda s: heldout_plan(s, STUDY_STEPS)})
                continue
            if not_applicable(meta, a.mode):
                print(f"{name}: skipped (no {a.mode} mode)")
                continue
            end = a.end_step or int(meta["args"]["steps"])
            if stage is not None and end != (CONTINUATION_STEPS if stage == CONTINUATION_STAGE else STUDY_STEPS):
                print(f"{name}: ERROR EndStepRefused")      # 1M in every stage, 3M in cont3m/ only
                failed += 1
                continue
            jobs.append({"name": name, "stem": stem, "mode": a.mode, "env": meta["args"]["env"],
                         "meta": meta, "end_step": end, "root": a.root,
                         "plan_fn": lambda s, end=end: heldout_plan(s, end)})
    return jobs, failed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="*")
    p.add_argument("--mode", choices=MODES, default="deployed")
    p.add_argument("--end-step", type=int, default=None, help="N; default = each run's --steps")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--force", action="store_true", help="re-evaluate runs that have a sealed output")
    p.add_argument("--phs", action="store_true", help="PHS on the B* runs of DIR (needs phs/<stem>.posthoc.pt, v1.json)")
    p.add_argument("--r3", default=None, metavar="LAGU_DIR")
    p.add_argument("--seeds", default="100-109", help="R3 training seeds")
    p.add_argument("--r3-prefix", default=R3_PREFIX, help="R3: evaluate only the stems <prefix>_*")
    p.add_argument("--root", default=None, help="sealing root (default: the run's results/method)")
    a = p.parse_args(argv)
    if not a.r3 and not a.dirs:
        p.error("give run directories, or --r3 LAGU_DIR")
    if a.phs and (a.r3 or a.mode != "deployed" or a.end_step):
        p.error("--phs takes run directories only (PHS is deployed mode at 1M)")
    torch.set_num_threads(1)
    jobs, failed = collect_jobs(a)
    ready = []
    for job in jobs:
        try:
            job["plan"] = job.pop("plan_fn")(job["stem"])
            job["root"] = job["root"] or seal.find_root(os.path.dirname(job["stem"]))
            job["digests"] = [digest(path) for _, _, path, _ in job["plan"]] + \
                ([digest(job["phs"]["fit"])] if job.get("phs") else [])
            _, enc, sha = output_paths(job["root"], job["name"], job["mode"], job["end_step"])
            if os.path.exists(enc) and not a.force:
                if not os.path.exists(sha):
                    raise Refused("ShaMissing")
                with open(sha) as f:
                    old = json.load(f)
                if old != job["digests"]:
                    raise Refused("SnapshotsChanged")
                print(f"{job['name']}: skipped (sealed output exists)")
                continue
            seal.pub_key(job["root"])
            ready.append(job)
        except Exception as e:
            failed += 1
            print(f"{job['name']}: ERROR {e if isinstance(e, Refused) else type(e).__name__}")
    if a.workers > 1 and len(ready) > 1:
        pool = mp.get_context("spawn").Pool(min(a.workers, len(ready)), initializer=_init_worker)
        results = pool.imap_unordered(_evaluate_job, ready)
    else:
        pool = None
        results = map(_evaluate_job, ready)
    for name, msg in results:
        failed += msg.startswith("ERROR")
        print(f"{name}: {msg}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
