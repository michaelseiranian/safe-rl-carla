"""Stage 0 of the method study: gates G0a and G0b and the S0-1 extensions (safe2/SPEC.md section 8;
prereg_method.md section 5).

    python -m safe2.probe g0a [--runs results/method/p1] [--env ENV] [--out results/method/stage0] [--workers 3]
    python -m safe2.probe g0b [--runs results/method/p1] [--pareto MAIN/results/pareto] [--env ENV]
                              [--out results/method/stage0] [--workers 3] [--reuse-fits]
    python -m safe2.probe s01 [--root MAIN] [--out results/method/stage0/s01] [--workers 3]
    python -m safe2.probe s04 [--runs CKPT ...] [--nep 100] [--out results/method/stage0] [--force]
    python -m safe2.probe phs-quality RUN [--root ROOT] [--force]     # RUN = <dir>/<stem> of a B* run

MAIN is the main checkout, /Users/michaelseiranian/Desktop/safe-rl-carla (read-only here).

* G0a (privileged oracle ceiling; never a method result). The deterministic final actor of each P1 B1
  run (<runs>/B1_<env>_s{0,1,2}.pt) plays 30 episodes on layouts 90000-90029 behind the MuJoCo
  state-restore viability filter of the design script sceptic-bvs/sim3_select.py (budget mode,
  B = 12, H = 25, value-aware selection). Output <out>/g0a.json. The PROCEED rule needs B*'s pilot
  return and is applied at method-v0.5 by g0a_decision() (called by safe2.analyze bstar).
* G0b (can V be learned offline?). (0) Replay identity: rows 0..199,999 of each B1 replay against the
  td3lag-omni replay of the same seed (disclosure only). (1) safe2.posthoc fits V and pi_b on episodes
  201-900 of each B1 replay. (i) Entry recall and false alarms on episodes 901-1000, pooled.
  (ii) Each B1 actor plays layouts 90100+; near-hazard states get an oracle label (the executed
  step's cost plus the best evasive 31-step cost, state restore); pooled Spearman between
  max_j V^j_32 and the label. PROCEED iff (i) and (ii) pass. Output <out>/g0b.json and the fits in
  <out>/g0b/<stem>.posthoc.pt.
* S0-1 (EXPLORATORY; gates nothing). The statistics of the main checkout's
  results/paper/uncertainty_anticipation/make.py (copied below), extended with the max hazard-lidar bin
  as a predictor, leads k in {5, 10, 25, 50}, the snapshots nearest 0.3N, 0.6N and N, the CarGoal1 and
  PointCircle1 arms of results/gate_x, event-aligned curves around entries, AUPRC with base rates, and
  lagu-tt's healthy runs (held-out return >= 5). Layouts 11000-11009. Output <out>/ (s01.json, s01.txt,
  s01_scores.csv, s01_curves.csv, traces/).
* S0-4 (descriptive; kill rule 8.4 for SHL*). The design script simplest/h3_grad.py as a function: on the
  first 500 steps of the last 100 complete episodes of each run's replay, |grad_a Q_c| and |grad_a Q_bar|
  at a = pi(s); share = the quiet states' part of the Q_c gradient mass. Default runs: the design runs in
  MAIN/results (read-only); after SHL P1, the SHL* p1 checkpoints. Output <out>/s04.json (merged by stem).

* PHS quality (prereg A.9; reported, not gated). For one B* confirmatory run: G0b(i) of its PHS fit
  (<root>/phs/<stem>.posthoc.pt, fitted on episodes 1-900) on episodes 901-1000, and G0b(ii) with B*'s 1M
  actor on the g0b layouts. Sealed to <root>/sealed/phs/<stem>.quality.json.enc; the console prints only
  `<stem>: done`.

Every output records the SHA-256 of every input file it read. Stage-0 outputs are not sealed (selection
seeds and existing runs only). G0b needs safe2.posthoc with the Python interface named in
POSTHOC_API below.

Oracle helpers mirror Safety-Gymnasium 1.0.0's Underlying.simulation_forward without touching
task.random_generator, so a filter that saves, explores and restores the state leaves the episode
exactly as it would have been (tests/test_probe.py checks this bit for bit).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import sys
from dataclasses import asdict, dataclass

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one BLAS thread per process (spawned workers inherit it), as the launcher

import numpy as np
import torch

from safe2 import seeds

MAIN_CHECKOUT = "/Users/michaelseiranian/Desktop/safe-rl-carla"
DEFAULT_RUNS = os.path.join("results", "method", "p1")
DEFAULT_OUT = os.path.join("results", "method", "stage0")
G0_ENV = "SafetyPointGoal1-v0"
G0_SEEDS = (0, 1, 2)
POSTHOC_API = """safe2.posthoc must expose (episodes are 1-based and inclusive, as on its CLI):
  fit(ckpt, episodes, out, updates=100_000, seed=0) -> dict          writes the fit to `out`
  fit_meta(path) -> dict with ckpt_sha256, episodes [A, B], updates, seed   (checked by g0b and phs-quality)
  recall(fit, ckpt, episodes) -> dict with entries, caught, recall, false_alarm_steps, episodes,
                                 fa_per_episode (and nonfinite_pred: any non-finite prediction fails (i))
  load_fit(path) -> object with vmax(o, a) for float32 torch tensors o (B, obs_dim) and a (B, act_dim),
                    returning a tensor or array of shape (B,) (a safe2.bvs.ViabilityNet qualifies)"""


# --- small shared helpers -------------------------------------------------------------------------
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes(*modules: str) -> dict:
    """SHA-256 of the safe2 source files that produced an output (provenance)."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for m in modules:
        p = os.path.join(here, m + ".py")
        out[f"safe2/{m}.py"] = sha256(p) if os.path.exists(p) else None
    return out


def write_json(path: str, obj) -> None:
    """Atomic json write; numpy scalars and arrays (e.g. from safe2.posthoc's outputs) become plain values."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, path)


def _method_root(path: str) -> str | None:
    """The results/method directory that `path` lies under (resolved), or None."""
    parts = os.path.realpath(path).split(os.sep)
    for i in range(len(parts) - 1):
        if parts[i] == "results" and parts[i + 1] == "method":
            return os.sep.join(parts[:i + 2])
    return None


def guard_gate(out: str, name: str, cfg, run_seeds, force: bool = False) -> None:
    """Refuse a Stage-0 gate that would not be the preregistered one or would overwrite an earlier result.
    Under results/method/: the default settings (G0aConfig() / G0bConfig()) and selection seeds only.
    Anywhere: an existing <out>/<name>.json is kept unless force; a forced overwrite under results/method/
    appends the old file's SHA-256 to results/method/deviations.json (SPEC 3.1, test 35, for the probe)."""
    root = _method_root(out)
    if root is not None:
        if asdict(cfg) != asdict(type(cfg)()):
            raise SystemExit(f"{name}: non-default settings under results/method: {asdict(cfg)}")
        if not run_seeds or not set(run_seeds) <= set(seeds.SELECTION_SEEDS):
            raise SystemExit(f"{name}: run seeds {tuple(run_seeds)} under results/method; only {seeds.SELECTION_SEEDS}")
    path = os.path.join(out, name + ".json")
    if not os.path.exists(path):
        return
    if not force:
        raise SystemExit(f"{name}: {path} exists; a re-run needs --force (recorded in deviations.json)")
    if root is not None:
        dev = os.path.join(root, "deviations.json")
        entries = []
        if os.path.exists(dev):
            with open(dev) as f:
                entries = json.load(f)
        entries.append({"overwritten": os.path.relpath(os.path.realpath(path), os.path.dirname(os.path.dirname(root))),
                        "old_sha256": sha256(path), "by": f"safe2.probe {name} --force", "entry": None})
        write_json(dev, entries)


def make_env(env_id: str):
    import warnings
    warnings.filterwarnings("ignore")
    import safety_gymnasium
    return safety_gymnasium.make(env_id)


def _init_worker() -> None:
    torch.set_num_threads(1)


def _map(fn, jobs: list, workers: int):
    """fn over jobs, in `workers` spawned processes (one torch thread each; at most 3, as make.py) or
    in-process."""
    workers = max(1, min(3, workers))
    if workers > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(jobs)), initializer=_init_worker) as pool:
            return pool.map(fn, jobs)
    return [fn(j) for j in jobs]


def load_policy(pt_path: str):
    """(evaluate.BasePolicy, RunConfig) of the actor in a safe2 final <stem>.pt
    ({config, obs_dim, act_dim, horizon, step, agent}). Deterministic, the arithmetic of LagU.act."""
    from lagu.agent import Actor
    from safe2.configs import RunConfig
    from safe2.evaluate import BasePolicy
    pt = torch.load(pt_path, weights_only=True)
    rc = RunConfig(**pt["config"])
    in_dim = int(pt["obs_dim"]) + (1 if rc.budget_input else 0)
    with torch.random.fork_rng():
        actor = Actor(in_dim, int(pt["act_dim"]), rc.hidden)
    actor.load_state_dict(pt["agent"]["actor"])
    return BasePolicy(actor, rc.budget_input, rc.cost_limit), rc


# --- oracle helpers (privileged MuJoCo state) -----------------------------------------------------
# (forward, turn) pairs. Point actions are (f, w) themselves; Car maps them onto its wheels as the design
# script did: (f - w, f + w), clipped.
EVASIVE = ((-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1), (0, 0), (0.5, 1), (0.5, -1))
GRID = tuple((f, w) for f in (1, 0.5, 0, -0.5, -1) for w in (-1, -0.5, 0, 0.5, 1))
VALUE_CANDS = GRID[::3] + (GRID[12],)   # as sim3_select.py: GRID[12] = (0, 0) is listed twice (inert)


def action(env_id: str, fw) -> np.ndarray:
    f, w = fw
    a = np.array([f - w, f + w]) if "Car" in env_id else np.array([f, w], dtype=float)
    return np.clip(a, -1.0, 1.0)


class Actions:
    """EVASIVE and VALUE_CANDS as action vectors of one task."""

    def __init__(self, env_id: str):
        self.evasive = [action(env_id, x) for x in EVASIVE]
        self.value_cands = [action(env_id, x) for x in VALUE_CANDS]


def save(task) -> tuple:
    d = task.data
    return d.qpos.copy(), d.qvel.copy(), d.act.copy(), d.qacc_warmstart.copy(), d.time


def load(task, s: tuple) -> None:
    """Restore a saved state and recompute positions and sensors with mj_forward. MuJoCo 2.3's mj_forward
    writes its solution into qacc_warmstart, so the saved warm start is put back afterwards: the next
    env.step then continues bit for bit as if nothing had been explored (without this the episode
    drifts by ~1e-14 within a few steps)."""
    import mujoco
    m, d = task.model, task.data
    d.qpos[:], d.qvel[:], d.act[:], d.qacc_warmstart[:] = s[0], s[1], s[2], s[3]
    d.time = s[4]
    mujoco.mj_forward(m, d)
    d.qacc_warmstart[:] = s[3]


def step_physics(task, a) -> None:
    """One environment step of physics, as Underlying.simulation_forward with no action noise and the
    deterministic frameskip (binomial(n, 1) = n), without drawing from task.random_generator."""
    import mujoco
    assert not task.action_noise and task.sim_conf.frameskip_binom_p == 1.0, "non-deterministic physics step"
    task.agent.apply_action(np.asarray(a), None)
    for _ in range(task.sim_conf.frameskip_binom_n):
        for mocap in task._mocaps.values():
            mocap.move()
        mujoco.mj_step(task.model, task.data)
    mujoco.mj_forward(task.model, task.data)


def step_cost(task) -> float:
    """The cost env.step would report for the current state (Builder._cost). Like env.step it runs an
    mj_forward (inside calculate_cost), so step_physics then step_cost then task.obs() is env.step."""
    total = task.calculate_cost()["cost_sum"]
    return float(total > 0.0) if task.cost_conf.constrain_indicator else float(total)


def hazard_distance(task) -> float:
    """min_h agent.dist_xy(h) - hazards.size; +inf on a task without hazards (PointCircle1)."""
    hz = getattr(task, "hazards", None)
    if hz is None:
        return float("inf")
    return float(min(task.agent.dist_xy(p) for p in hz.pos) - hz.size)


def viable(task, s0: tuple, a, H: int, evasive) -> bool:
    """From s0, apply a; False if that enters a hazard. Otherwise True iff one evasive action held for H
    steps never enters a hazard. Leaves the simulator in a modified state: the caller restores s0."""
    load(task, s0)
    step_physics(task, a)
    if hazard_distance(task) < 0:
        return False
    s1 = save(task)
    for e in evasive:
        load(task, s1)
        for _ in range(H):
            step_physics(task, e)
            if hazard_distance(task) < 0:
                break
        else:
            return True
    return False


# --- G0a: oracle ceiling ---------------------------------------------------------------------------
@dataclass
class G0aConfig:
    env: str = G0_ENV
    block: str = "g0a"            # layout block (tests use "dev")
    episodes: int = 30            # per actor, layouts 90000 + k
    B: float = 12.0               # the filter acts once the episode cost reaches B ...
    near: float = 0.6             # ... and the hazard distance is below this
    H: int = 25                   # evasive horizon of viable()
    hold: int = 5                 # value rollout: hold the candidate this many steps ...
    follow: int = 10              # ... then follow the actor this many steps
    hit_penalty: float = 10.0     # score = -goal distance - hit_penalty * [a hazard was entered]


def value_score(task, s0: tuple, cand, policy, cfg: G0aConfig) -> float:
    """Privileged 15-step value of a candidate (sim3_select.py SEL=1): from s0 hold it `hold` steps, then
    follow the actor on task.obs() for `follow` steps; stop at the first hazard entry. Score =
    -||goal - agent||_xy at the end - hit_penalty if a hazard was entered."""
    load(task, s0)
    entered = False
    for j in range(cfg.hold + cfg.follow):
        step_physics(task, cand if j < cfg.hold else policy.act(task.obs()))
        if hazard_distance(task) < 0:
            entered = True
            break
    dist = float(np.linalg.norm(task.goal.pos[:2] - task.agent.pos[:2]))
    return -dist - (cfg.hit_penalty if entered else 0.0)


def oracle_filter(task, policy, a, cfg: G0aConfig, acts: Actions) -> tuple:
    """(a_exec, intervened, fallback) at a step where the filter is active. The simulator is restored to
    the current state before returning, so env.step(a_exec) continues the true episode."""
    s0 = save(task)
    out = (a, False, False)
    if not viable(task, s0, a, cfg.H, acts.evasive):
        best, pick = -math.inf, None
        for c in acts.value_cands:
            if not viable(task, s0, c, cfg.H, acts.evasive):
                continue
            score = value_score(task, s0, c, policy, cfg)
            if score > best:                    # ties go to the first candidate
                best, pick = score, c
        out = (acts.evasive[0], True, True) if pick is None else (pick, True, False)
    load(task, s0)
    return out


def g0a_episode(env, policy, i: int, cfg: G0aConfig, acts: Actions) -> dict:
    """One episode of the filtered actor on seeds.layout(cfg.block, i)."""
    task = env.unwrapped.task
    o, _ = seeds.reset(env, cfg.block, i)
    lid = seeds.layout(cfg.block, i)
    policy.begin(lid)
    rs, cs = [], []
    C = 0.0
    n_int = n_fb = 0
    done = False
    while not done:
        a = policy.act(o)
        if C >= cfg.B and hazard_distance(task) < cfg.near:
            a, hit, fb = oracle_filter(task, policy, a, cfg, acts)
            n_int += int(hit)
            n_fb += int(fb)
        o, r, c, term, trunc, _ = env.step(a)
        policy.observe(c)
        rs.append(float(r))
        cs.append(float(c))
        C += float(c)
        done = term or trunc
    return {"layout": lid, "return": float(np.asarray(rs).sum()), "cost": float(np.asarray(cs).sum()),
            "interventions": n_int, "fallbacks": n_fb, "length": len(rs)}   # sums as evaluate.rollout


def _g0a_job(job: dict) -> dict:
    cfg = G0aConfig(**job["cfg"])
    policy, rc = load_policy(job["pt"])
    env = make_env(cfg.env)
    eps = [g0a_episode(env, policy, i, cfg, Actions(cfg.env)) for i in range(cfg.episodes)]
    env.close()
    return {"stem": job["stem"], "config": rc.name, "episodes": eps,
            "mean_return": float(np.mean([e["return"] for e in eps])),
            "mean_cost": float(np.mean([e["cost"] for e in eps])),
            "mean_interventions": float(np.mean([e["interventions"] for e in eps]))}


def g0a(runs: str, out: str, cfg: G0aConfig | None = None, run_seeds=G0_SEEDS, workers: int = 3,
        config_name: str = "B1", force: bool = False) -> dict:
    """Run G0a and write <out>/g0a.json. Returns the output dict (no decision: see g0a_decision)."""
    cfg = cfg or G0aConfig()
    guard_gate(out, "g0a", cfg, run_seeds, force)
    stems = [f"{config_name}_{cfg.env}_s{s}" for s in run_seeds]
    pts = [os.path.join(runs, st + ".pt") for st in stems]
    for p in pts:
        if not os.path.exists(p):
            raise SystemExit(f"g0a: missing input {p}")
    inputs = {p: sha256(p) for p in pts}
    actors = _map(_g0a_job, [{"stem": st, "pt": p, "cfg": asdict(cfg)} for st, p in zip(stems, pts)], workers)
    eps = [e for a in actors for e in a["episodes"]]
    res = {"probe": "g0a", "settings": asdict(cfg), "inputs": inputs, "code": code_hashes("probe"),
           "actors": actors, "n_episodes": len(eps),
           "mean_return": float(np.mean([e["return"] for e in eps])),
           "mean_cost": float(np.mean([e["cost"] for e in eps])),
           "mean_interventions": float(np.mean([e["interventions"] for e in eps])),
           "decision": "taken at method-v0.5 by g0a_decision(mean_return, mean_cost, R_B*(P1))",
           "label": "PRIVILEGED oracle ceiling; never a method result"}
    write_json(os.path.join(out, "g0a.json"), res)
    return res


def g0a_decision(mean_return: float, mean_cost: float, r_bstar_p1: float, cost_limit: float = 25.0) -> dict:
    """PROCEED iff mean return >= max(18, R_B*(P1) + 6) and mean cost <= 25 (prereg_method.md section 5)."""
    thr = max(18.0, float(r_bstar_p1) + 6.0)
    ok = bool(np.isfinite(mean_return) and np.isfinite(mean_cost) and mean_return >= thr and mean_cost <= cost_limit)
    return {"return_threshold": thr, "cost_limit": cost_limit, "mean_return": float(mean_return),
            "mean_cost": float(mean_cost), "r_bstar_p1": float(r_bstar_p1), "proceed": ok}


# --- G0b: offline V ----------------------------------------------------------------------------------
@dataclass
class G0bConfig:
    env: str = G0_ENV
    block: str = "g0b"            # layout block of (ii) (tests use "dev")
    fit_episodes: tuple = (201, 900)        # 1-based, inclusive (the first 200 replay episodes are excluded)
    holdout_episodes: tuple = (901, 1000)   # (i) entry recall and false alarms
    updates: int = 100_000        # posthoc fit updates
    identity_rows: int = 200_000  # (0) rows compared with the td3lag-omni replay
    recall_min: float = 0.80      # (i) pooled recall >= this ...
    fa_max: float = 30.0          # ... and false-alarm steps per held-out episode <= this
    near: float = 0.6             # (ii) collect states with hazard distance below this ...
    per_episode: int = 20         # ... at most this many per episode ...
    gap: int = 10                 # ... at least this many steps apart
    states_per_seed: int = 400    # a seed stops at this many states ...
    max_episodes: int = 100       # ... or after this many episodes
    label_steps: int = 31         # evasive steps of the oracle label (executed step + 31 = 32 = H)
    min_states: int = 1000        # (ii) fails with fewer pooled states
    rho_min: float = 0.7          # (ii) pooled Spearman >= this


REPLAY_COLUMNS = (("o", "o"), ("a_pi", "a"), ("r", "r"), ("c", "c"), ("o2", "o2"), ("d", "d"))  # (safe2, lagu)
# lagu checkpoints hold numpy scalars (ep_rets); the older ones (pareto s0, s1) also hold numpy replay columns
NUMPY_SCALARS = [np.core.multiarray.scalar, np.core.multiarray._reconstruct, np.ndarray, np.dtype] + [
    type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32", "uint8", "bool")]


def _np(x) -> np.ndarray:
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def insertion_index(rep: dict, total: int, lo: int, hi: int):
    """Buffer rows holding the transitions inserted lo..hi-1 (0-based, one per environment step) of a
    replay state {columns, ptr, n} after `total` insertions; None if some were overwritten or never
    inserted. Rows are in insertion order as in lagu.train.Replay.load."""
    n, ptr, size = int(rep["n"]), int(rep["ptr"]), int(_np(rep["o"]).shape[0])
    first = total - n                                    # insertion index of the oldest stored row
    if lo < first or hi > total or lo >= hi:
        return None
    order = np.arange(n) if n < size else (np.arange(ptr, ptr + size) % size)
    return order[lo - first:hi - first]


def replay_identity(b1_ckpt: str, lagu_ckpt: str, rows: int = 200_000) -> dict:
    """Compare rows 0..rows-1 (o, a, r, c, o2, d) of a safe2 checkpoint's replay with a lagu.train
    checkpoint's, bit for bit. Both files are read-only (memory-mapped)."""
    A = torch.load(b1_ckpt, weights_only=True, mmap=True)
    with torch.serialization.safe_globals(NUMPY_SCALARS):      # still a weights-only unpickler
        B = torch.load(lagu_ckpt, weights_only=True, mmap=True)
    ia = insertion_index(A["replay"], int(A["step"]), 0, rows)
    ib = insertion_index(B["replay"], int(B["step"]), 0, rows)
    if ia is None or ib is None:
        return {"result": "unavailable", "rows": rows,
                "why": "first rows overwritten or fewer rows stored" + (" (B1)" if ia is None else " (td3lag-omni)")}
    first, cols = None, []
    for ca, cb in REPLAY_COLUMNS:
        xa, xb = _np(A["replay"][ca])[ia], _np(B["replay"][cb])[ib]
        if xa.dtype != xb.dtype or xa.shape != xb.shape:
            bad = 0
        else:
            neq = (xa.view(np.uint8).reshape(rows, -1) != xb.view(np.uint8).reshape(rows, -1)).any(1)
            bad = int(np.argmax(neq)) if neq.any() else None
        if bad is not None:
            if first is None or bad < first:
                first, cols = bad, [cb]
            elif bad == first:
                cols.append(cb)
    if first is None:
        return {"result": "identical", "rows": rows}
    return {"result": "differs", "rows": rows, "first_row": first, "columns": cols}


class StateSampler:
    """G0b(ii) collection rule inside one episode: take step t iff the hazard distance is below `near`,
    fewer than `per_episode` states were taken in this episode, and t >= the last taken t + `gap`."""

    def __init__(self, near: float = 0.6, per_episode: int = 20, gap: int = 10):
        self.near, self.per_episode, self.gap = near, per_episode, gap
        self.begin()

    def begin(self) -> None:
        self.n, self.last = 0, None

    def take(self, t: int, hd: float) -> bool:
        if hd < self.near and self.n < self.per_episode and (self.last is None or t >= self.last + self.gap):
            self.n += 1
            self.last = t
            return True
        return False


def oracle_label(task, s0: tuple, a, steps: int, evasive) -> float:
    """Cost of applying a at s0 plus the minimum over evasive actions held `steps` steps of the cost they
    incur (state restore). The simulator is left modified: the caller restores s0."""
    load(task, s0)
    step_physics(task, a)
    c0 = step_cost(task)
    s1 = save(task)
    best = math.inf
    for e in evasive:
        load(task, s1)
        tot = 0.0
        for _ in range(steps):
            step_physics(task, e)
            tot += step_cost(task)
            if tot >= best:                     # cannot improve on the best (exact shortcut)
                break
        best = min(best, tot)
        if best == 0.0:
            break
    return c0 + best


def g0b_rank_seed(env, policy, vmax, cfg: G0bConfig, acts: Actions) -> dict:
    """G0b(ii) for one actor: deterministic episodes on layouts seeds.layout(cfg.block, k), k = 0, 1, ...;
    oracle label and prediction vmax(o_t, a_t) at each collected state."""
    task = env.unwrapped.task
    sampler = StateSampler(cfg.near, cfg.per_episode, cfg.gap)
    rec = {"layout": [], "t": [], "pred": [], "label": []}
    episodes = 0
    while episodes < cfg.max_episodes and len(rec["pred"]) < cfg.states_per_seed:
        o, _ = seeds.reset(env, cfg.block, episodes)
        lid = seeds.layout(cfg.block, episodes)
        policy.begin(lid)
        sampler.begin()
        episodes += 1
        t, done = 0, False
        while not done:
            a = policy.act(o)
            if sampler.take(t, hazard_distance(task)):
                s0 = save(task)
                label = oracle_label(task, s0, a, cfg.label_steps, acts.evasive)
                load(task, s0)
                rec["pred"].append(float(vmax(o[None], np.asarray(a)[None])[0]))
                rec["label"].append(label)
                rec["layout"].append(lid)
                rec["t"].append(t)
                if len(rec["pred"]) >= cfg.states_per_seed:
                    break
            o, r, c, term, trunc, _ = env.step(a)
            policy.observe(c)
            t += 1
            done = term or trunc
    return {**rec, "episodes": episodes, "states": len(rec["pred"])}


def _backend(name: str):
    mod = importlib.import_module(name)
    missing = [f for f in ("fit", "fit_meta", "recall", "load_fit") if not hasattr(mod, f)]
    if missing:
        raise SystemExit(f"{name} lacks {missing}. {POSTHOC_API}")
    return mod


def vmax_fn(fit):
    """numpy (o, a) batches -> numpy max_j V^j_32(o, a) through a fit's torch vmax (no gradient)."""
    @torch.no_grad()
    def f(o, a) -> np.ndarray:
        v = fit.vmax(torch.as_tensor(np.asarray(o), dtype=torch.float32), torch.as_tensor(np.asarray(a), dtype=torch.float32))
        return np.asarray(v.cpu() if isinstance(v, torch.Tensor) else v, dtype=float).reshape(-1)
    return f


def _g0b_fit_job(job: dict) -> dict:
    be = _backend(job["backend"])
    return be.fit(job["ckpt"], tuple(job["episodes"]), job["out"], updates=job["updates"], seed=job["seed"])


def _g0b_rank_job(job: dict) -> dict:
    cfg = G0bConfig(**job["cfg"])
    be = _backend(job["backend"])
    fit = be.load_fit(job["fit"])
    policy, _ = load_policy(job["pt"])
    env = make_env(cfg.env)
    res = g0b_rank_seed(env, policy, vmax_fn(fit), cfg, Actions(cfg.env))
    env.close()
    return {"stem": job["stem"], **res}


def avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with tied values given their average rank (make.py)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, first, counts = np.unique(xs, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(avg, counts)
    return ranks


def spearman(x, y) -> float:
    """Spearman correlation: Pearson correlation of average ranks (ties averaged). nan if a side is
    constant, a value is not finite, or there are fewer than 2 points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    rx, ry = avg_ranks(x), avg_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def pool_recall(per_run: list[dict]) -> dict:
    """Pooled G0b(i) over runs: recall = sum caught / sum entries, false alarms per held-out episode =
    sum false-alarm steps / sum episodes. Both are NaN (so (i) fails) when any run had a non-finite
    prediction (a NaN vmax is never an alarm, so it would silently lower the counts)."""
    ent = sum(int(r["entries"]) for r in per_run)
    cau = sum(int(r["caught"]) for r in per_run)
    fa = sum(float(r["false_alarm_steps"]) for r in per_run)
    eps = sum(int(r["episodes"]) for r in per_run)
    bad = sum(int(r.get("nonfinite_pred", 0)) for r in per_run)
    return {"entries": ent, "caught": cau, "recall": cau / ent if (ent and not bad) else float("nan"),
            "false_alarm_steps": fa, "episodes": eps,
            "fa_per_episode": fa / eps if (eps and not bad) else float("nan"), "nonfinite_pred": bad}


def check_fit(be, path: str, ckpt_sha256: str, episodes, updates: int, seed: int, who: str) -> None:
    """Refuse a fit that was not made on this checkpoint, these episodes, this many updates and this seed
    (review F4: a dev or copied fit must not decide a gate)."""
    want = {"ckpt_sha256": ckpt_sha256, "episodes": [int(x) for x in episodes], "updates": int(updates),
            "seed": int(seed)}
    got = be.fit_meta(path)
    got = {k: ([int(x) for x in got[k]] if k == "episodes" else got[k]) for k in want}
    bad = [k for k in want if got[k] != want[k]]
    if bad:
        raise SystemExit(f"{who}: the fit {path} does not match its run ({', '.join(bad)})")


def g0b_decision(recall: dict, n_states: int, rho: float, cfg: G0bConfig | None = None,
                 nonfinite_pred: int = 0) -> dict:
    """(i) pooled recall >= 0.80 and false alarms per episode <= 30; (ii) at least 1000 pooled states and
    Spearman >= 0.7 (fewer states fails). PROCEED iff both. Non-finite values, including any non-finite
    prediction, fail."""
    cfg = cfg or G0bConfig()
    rec, fa = float(recall["recall"]), float(recall["fa_per_episode"])
    ok_i = bool(np.isfinite(rec) and np.isfinite(fa) and rec >= cfg.recall_min and fa <= cfg.fa_max)
    ok_ii = bool(n_states >= cfg.min_states and np.isfinite(rho) and rho >= cfg.rho_min and nonfinite_pred == 0)
    return {"pass_i": ok_i, "pass_ii": ok_ii, "proceed": ok_i and ok_ii}


def g0b(runs: str, pareto: str, out: str, cfg: G0bConfig | None = None, run_seeds=G0_SEEDS, workers: int = 3,
        backend: str = "safe2.posthoc", reuse_fits: bool = False, config_name: str = "B1",
        lagu_stem: str = "td3lag-omni", force: bool = False) -> dict:
    """Run G0b and write <out>/g0b.json (with the decision). Fits go to <out>/g0b/<stem>.posthoc.pt."""
    cfg = cfg or G0bConfig()
    guard_gate(out, "g0b", cfg, run_seeds, force)
    be = _backend(backend)
    stems = [f"{config_name}_{cfg.env}_s{s}" for s in run_seeds]
    pt = {st: os.path.join(runs, st + ".pt") for st in stems}
    ck = {st: os.path.join(runs, st + ".ckpt.pt") for st in stems}
    lg = {st: os.path.join(pareto, f"{lagu_stem}_{cfg.env}_s{s}.ckpt.pt") for st, s in zip(stems, run_seeds)}
    fits = {st: os.path.join(out, "g0b", st + ".posthoc.pt") for st in stems}
    needed = list(pt.values()) + list(ck.values()) + list(lg.values()) + (list(fits.values()) if reuse_fits else [])
    for p in needed:
        if not os.path.exists(p):
            raise SystemExit(f"g0b: missing input {p}")
    inputs = {p: sha256(p) for p in list(pt.values()) + list(ck.values()) + list(lg.values())}

    identity = {}
    for st in stems:                    # disclosure only: an unreadable checkpoint must not abort the gate
        try:
            identity[st] = replay_identity(ck[st], lg[st], cfg.identity_rows)
        except Exception as e:
            identity[st] = {"result": "unavailable", "rows": cfg.identity_rows, "why": repr(e)}
    os.makedirs(os.path.join(out, "g0b"), exist_ok=True)
    if not reuse_fits:
        _map(_g0b_fit_job, [{"backend": backend, "ckpt": ck[st], "episodes": list(cfg.fit_episodes), "out": fits[st],
                             "updates": cfg.updates, "seed": s} for st, s in zip(stems, run_seeds)], workers)
    for st, s in zip(stems, run_seeds):
        check_fit(be, fits[st], inputs[ck[st]], cfg.fit_episodes, cfg.updates, s, "g0b")
    fit_hashes = {fits[st]: sha256(fits[st]) for st in stems}
    recalls = {st: be.recall(fits[st], ck[st], tuple(cfg.holdout_episodes)) for st in stems}
    pooled = pool_recall(list(recalls.values()))
    ranks = _map(_g0b_rank_job, [{"backend": backend, "stem": st, "pt": pt[st], "fit": fits[st], "cfg": asdict(cfg)}
                                 for st in stems], workers)
    pred = np.concatenate([np.asarray(r["pred"], float) for r in ranks]) if ranks else np.zeros(0)
    label = np.concatenate([np.asarray(r["label"], float) for r in ranks]) if ranks else np.zeros(0)
    rho = spearman(pred, label)
    nonfinite = int((~np.isfinite(pred)).sum())
    decision = g0b_decision(pooled, len(pred), rho, cfg, nonfinite)
    res = {"probe": "g0b", "settings": asdict(cfg), "inputs": inputs, "fits": fit_hashes,
           "code": code_hashes("probe", "posthoc"), "identity": identity,
           "recall": {"per_run": recalls, "pooled": pooled},
           "ranking": {"per_run": ranks, "states": int(len(pred)), "spearman": rho,
                       "nonfinite_pred": nonfinite},
           "decision": decision}
    write_json(os.path.join(out, "g0b.json"), res)
    return res


# --- PHS quality (prereg A.9): G0b(i) and (ii) for one B* run's PHS fit; sealed, reported, not gated ------
PHS_FIT_EPISODES = (1, 900)


def phs_quality(run: str, root: str | None = None, cfg: G0bConfig | None = None, backend: str = "safe2.posthoc",
                force: bool = False) -> str:
    """G0b(i) on episodes 901-1000 of RUN.ckpt.pt and G0b(ii) with B*'s 1M actor (RUN.pt) on the g0b
    layouts, for the fit <root>/phs/<stem>.posthoc.pt (checked against the run: checkpoint SHA-256,
    episodes 1-900, updates, seed = run seed). Sealed to <root>/sealed/phs/<stem>.quality.json.enc; returns
    its path. Nothing is printed here (the CLI prints `<stem>: done`)."""
    from safe2 import seal
    stem = os.path.basename(run)
    ck, pt = run + ".ckpt.pt", run + ".pt"
    root = root or seal.find_root(os.path.dirname(os.path.abspath(run)))
    fit_path = os.path.join(root, "phs", stem + ".posthoc.pt")
    rel = os.path.join("phs", stem + ".quality.json")
    out_path = os.path.join(root, "sealed", rel + ".enc")
    for p in (ck, pt, fit_path):
        if not os.path.exists(p):
            raise SystemExit(f"{stem}: missing input {p}")
    if os.path.exists(out_path) and not force:
        raise SystemExit(f"{stem}: {out_path} exists; a re-run needs --force")
    seal.pub_key(root)                                  # refuse before any work without the key
    meta = torch.load(ck, weights_only=True, mmap=True)
    run_seed, env_id = int(meta["seed"]), str(meta["env"])
    del meta
    cfg = cfg or G0bConfig(env=env_id, fit_episodes=PHS_FIT_EPISODES)
    if cfg.env != env_id:
        raise SystemExit(f"{stem}: settings for {cfg.env}, run on {env_id}")
    be = _backend(backend)
    inputs = {p: sha256(p) for p in (ck, pt, fit_path)}
    check_fit(be, fit_path, inputs[ck], cfg.fit_episodes, cfg.updates, run_seed, "phs-quality")
    rec = be.recall(fit_path, ck, tuple(cfg.holdout_episodes))
    policy, _ = load_policy(pt)
    env = make_env(env_id)
    try:
        rank = g0b_rank_seed(env, policy, vmax_fn(be.load_fit(fit_path)), cfg, Actions(env_id))
    finally:
        env.close()
    pred, label = np.asarray(rank["pred"], float), np.asarray(rank["label"], float)
    res = {"probe": "phs-quality", "stem": stem, "settings": asdict(cfg), "inputs": inputs,
           "code": code_hashes("probe", "posthoc"),
           "recall": {"run": rec, "pooled": pool_recall([rec])},
           "ranking": {**rank, "spearman": spearman(pred, label), "nonfinite_pred": int((~np.isfinite(pred)).sum())}}
    res = json.loads(json.dumps(res, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    return seal.write_file(root, rel, res)


# --- S0-1 (EXPLORATORY): statistics copied from results/paper/uncertainty_anticipation/make.py ----------
S01_ENV = "SafetyPointGoal1-v0"
S01_ARMS = [  # (label, directory under --root, env, seeds; None = every run of the label found there)
    ("lagu-tt", "results/gate", S01_ENV, tuple(range(100, 115))),
    ("lagu_nogate-tt", "results/gate", S01_ENV, tuple(range(100, 115))),
    ("lagu-lit", "results/gate_literal", S01_ENV, tuple(range(100, 110))),
    ("lagu_nogate-lit", "results/gate_literal", S01_ENV, tuple(range(100, 110))),
    ("lagu-lit-T03", "results/gate_literal_x", S01_ENV, tuple(range(100, 110))),
] + [(lab, "results/gate_x", env, None) for env in ("SafetyCarGoal1-v0", "SafetyPointCircle1-v0")
     for lab in ("lagu-tt", "lagu_nogate-tt", "td3lag-tt")]
S01_HEALTHY = ("lagu-tt", S01_ENV, 5.0)   # (label, env, held-out return threshold): extra "healthy" arm
S01_EPISODES = 10
S01_FRACS = (0.3, 0.6, 1.0)               # snapshots nearest these fractions of N
S01_SNAP_TOL = 25_000                     # a snapshot must lie within this many steps of frac * N (exactly N at 1.0)
LEADS = (5, 10, 25, 50)
PREDICTORS = [("ratio", "ratio = sqrt(Q_eu)/|Q_bar|"), ("std", "sqrt(Q_eu)"), ("qc", "Q_c"),
              ("negqbar", "-Q_bar"), ("lidar", "max hazard-lidar bin")]
ENSEMBLE_ONLY = ("ratio", "std")          # undefined for a single critic (td3lag, M = 1)
CURVE = (-50, 10)                         # event-aligned curve offsets around an entry (inclusive)
ENTRY_LOOKBACK = 12                       # an entry: cost > 0, no cost in the 12 steps before, t >= 12
B = 10_000                                # cluster-bootstrap resamples (make.py)


def weights(R: int, n_boot: int = B) -> np.ndarray:
    """Run multiplicities of the cluster-bootstrap resamples (fresh default_rng(0)), as make.py."""
    idx = np.random.default_rng(0).integers(0, R, (n_boot, R))
    W = np.zeros((n_boot, R))
    np.add.at(W, (np.arange(n_boot)[:, None], idx), 1.0)
    return W


def auroc(x: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC via the rank statistic; nan if a class is empty (make.py)."""
    p = int(y.sum())
    n = len(y) - p
    if p == 0 or n == 0:
        return float("nan")
    r = avg_ranks(x)
    return float((r[y].sum() - p * (p + 1) / 2.0) / (p * n))


def pair_matrix(xs: list, ys: list) -> tuple:
    """U[r, s] = #(pos of run r > neg of run s) + 0.5 #(equal); P[r], N[r] class counts (make.py)."""
    R = len(xs)
    negs = [np.sort(x[~y]) for x, y in zip(xs, ys)]
    poss = [x[y] for x, y in zip(xs, ys)]
    U = np.zeros((R, R))
    for r in range(R):
        for s in range(R):
            lo = np.searchsorted(negs[s], poss[r], side="left")
            hi = np.searchsorted(negs[s], poss[r], side="right")
            U[r, s] = lo.sum() + 0.5 * (hi - lo).sum()
    P = np.array([len(p) for p in poss], float)
    N = np.array([len(n) for n in negs], float)
    return U, P, N


def boot_auroc(U, P, N, W) -> np.ndarray:
    num = np.einsum("br,rs,bs->b", W, U, W)
    den = (W @ P) * (W @ N)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def pct(v: np.ndarray) -> tuple:
    """95% percentile interval; nan resamples dropped, +-inf kept (make.py)."""
    v = v[~np.isnan(v)]
    if not len(v):
        return (np.nan, np.nan)
    q = np.percentile(np.where(np.isinf(v), np.sign(v) * 1e300, v), [2.5, 97.5])
    return tuple(float(np.sign(x) * np.inf) if abs(x) >= 1e299 else float(x) for x in q)


def targets(cost: np.ndarray, ep: np.ndarray, k: int) -> tuple:
    """(usable mask, y_k): y_k(t) = 1 if any cost in t+1..t+k; t needs a full k-step future (make.py)."""
    use = np.zeros(len(cost), bool)
    y = np.zeros(len(cost), bool)
    for e in np.unique(ep):
        ix = np.flatnonzero(ep == e)
        c = cost[ix] > 0
        n = len(ix)
        cs = np.concatenate([[0], np.cumsum(c)])
        t = np.arange(max(n - k, 0))
        use[ix[t]] = True
        y[ix[t]] = (cs[t + k + 1] - cs[t + 1]) > 0
    return use, y


# --- S0-1 extensions: AUPRC, its exact cluster bootstrap, entries and aligned curves ------------------
def average_precision(x: np.ndarray, y: np.ndarray) -> float:
    """Step-wise AUPRC: sum over score thresholds (tied scores form one threshold) of the recall gain
    times the precision at that threshold; nan without positives."""
    P = int(y.sum())
    if P == 0:
        return float("nan")
    order = np.argsort(-x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    last = np.r_[np.flatnonzero(np.diff(xs) != 0), len(xs) - 1]
    tp = np.cumsum(ys)[last]
    fp = (last + 1) - tp
    dtp = np.diff(np.r_[0.0, tp])
    return float(np.sum(dtp * tp / (tp + fp)) / P)


def ap_counts(xs: list, ys: list) -> tuple:
    """Per threshold group g (scores pooled over runs, descending) and run r: positives p[g, r] and
    negatives q[g, r]. The pooled AUPRC of a resample with run multiplicities w is computed exactly from
    these (boot_ap), identical to recomputing it on the duplicated data."""
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    run = np.concatenate([np.full(len(v), r) for r, v in enumerate(xs)])
    order = np.argsort(-x, kind="mergesort")
    gid = np.r_[0, np.cumsum(np.diff(x[order]) != 0)]
    G, R = int(gid[-1]) + 1 if len(gid) else 0, len(xs)
    p, q = np.zeros((G, R)), np.zeros((G, R))
    np.add.at(p, (gid, run[order]), y[order].astype(float))
    np.add.at(q, (gid, run[order]), (~y[order]).astype(float))
    return p, q


def boot_ap(p: np.ndarray, q: np.ndarray, W: np.ndarray, chunk: int = 256) -> np.ndarray:
    """AUPRC of every resample (rows of W: run multiplicities); nan where a resample has no positive."""
    cp, cn = np.cumsum(p, 0), np.cumsum(q, 0)
    keep = p.sum(1) > 0
    Pr = p.sum(0)
    p, cp, cn = p[keep], cp[keep], cn[keep]
    out = np.empty(len(W))
    for s in range(0, len(W), chunk):
        w = W[s:s + chunk].T
        dtp, tp, fp = p @ w, cp @ w, cn @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(dtp > 0, tp / (tp + fp), 0.0)
            den = Pr @ w
            out[s:s + chunk] = np.where(den > 0, (dtp * prec).sum(0) / den, np.nan)
    return out


def entry_steps(cost: np.ndarray, lookback: int = ENTRY_LOOKBACK) -> list[int]:
    """Episode-local entries: t >= lookback with cost > 0 and no cost in the lookback steps before."""
    c = np.asarray(cost) > 0
    return [t for t in range(lookback, len(c)) if c[t] and not c[t - lookback:t].any()]


def aligned_curve(runs: list[dict], key: str, lo: int = CURVE[0], hi: int = CURVE[1]) -> dict:
    """Mean of runs' `key` at offsets lo..hi around every entry (pooled over runs and episodes); offsets
    outside the episode are skipped and counted out of n."""
    offs = np.arange(lo, hi + 1)
    s, n = np.zeros(len(offs)), np.zeros(len(offs), int)
    entries = 0
    for r in runs:
        for e in np.unique(r["ep"]):
            ix = np.flatnonzero(r["ep"] == e)
            v, c = r[key][ix], r["cost"][ix]
            for te in entry_steps(c):
                entries += 1
                t = te + offs
                ok = (t >= 0) & (t < len(ix))
                s[ok] += v[t[ok]]
                n[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return {"offsets": offs.tolist(), "mean": mean.tolist(), "n": n.tolist(), "entries": entries}


def lidar_slice(task, key: str = "hazards_lidar"):
    """The slice of `key` in the flat observation (task.obs_info.obs_space_dict in flatten order), or None."""
    from gymnasium.spaces.utils import flatdim
    off = 0
    for k, space in task.obs_info.obs_space_dict.spaces.items():
        n = flatdim(space)
        if k == key:
            return slice(off, off + n)
        off += n
    return None


def snapshot_steps(snap_dir: str) -> list[int]:
    return sorted(int(n[:-3]) for n in os.listdir(snap_dir) if n.endswith(".pt") and n[:-3].isdigit())


def nearest(steps: list[int], target: float) -> int:
    """The saved step nearest target; ties go to the earlier step (lagu.evaluate.pick)."""
    return min(steps, key=lambda s: (abs(s - target), s))


def _s01_rollout(job: dict) -> str:
    """Play `episodes` deterministic episodes with one lagu snapshot; cache per-step values in an npz
    keyed by the snapshot's SHA-256, the block and the episode count."""
    from lagu.evaluate import load_agent
    path, out = job["snap"], job["trace"]
    digest = sha256(path)
    if os.path.exists(out):
        with np.load(out) as z:
            if str(z["sha256"]) == digest and int(z["episodes"]) == job["episodes"] and str(z["block"]) == job["block"]:
                return out
    with open(job["json"]) as f:
        meta = json.load(f)
    agent = load_agent(meta, torch.load(path, weights_only=True))
    cfg = agent.cfg
    env = make_env(meta["args"]["env"])
    sl = lidar_slice(env.unwrapped.task)
    cols = {k: [] for k in ("ep", "t", "qbar", "std", "ratio", "qc", "lidar", "cost", "reward")}
    for k in range(job["episodes"]):
        o, _ = seeds.reset(env, job["block"], k)
        obs, cs, rs = [], [], []
        done = False
        while not done:
            obs.append(o)
            o, r, c, term, trunc, _ = env.step(agent.act(o, explore=False))
            cs.append(float(c))
            rs.append(float(r))
            done = term or trunc
        ob_np = np.stack(obs)
        with torch.no_grad():                       # one batched critic call per episode (make.py)
            ob = torch.as_tensor(ob_np, dtype=torch.float32)
            a = agent.actor(ob)
            qbar, var = agent.ensemble(ob, a)
            std = torch.sqrt(var + 1e-12)
            ratio = std / (qbar.abs() + cfg.ratio_eps)
            qc = agent.qc_pi(ob, a)
        n = len(cs)
        cols["ep"].append(np.full(n, k, np.int16))
        cols["t"].append(np.arange(n, dtype=np.int16))
        for name, v in (("qbar", qbar), ("std", std), ("ratio", ratio), ("qc", qc)):
            cols[name].append(v.numpy().astype(np.float32))
        cols["lidar"].append(ob_np[:, sl].max(1).astype(np.float32) if sl is not None else np.full(n, np.nan, np.float32))
        cols["cost"].append(np.asarray(cs, np.float32))
        cols["reward"].append(np.asarray(rs, np.float32))
    env.close()
    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, sha256=digest, episodes=job["episodes"], block=job["block"], step=job["step"],
                        M=cfg.M, has_lidar=sl is not None, **arrays)
    os.replace(tmp, out)
    return out


def s01_plan(root: str, arms=None) -> tuple[list[dict], dict]:
    """(runs, inputs): every run of every arm with its three snapshots, and the SHA-256 of each json,
    snapshot and (healthy arm) heldout.json read."""
    arms = S01_ARMS if arms is None else arms
    runs, inputs = [], {}
    for label, d, env, seed_list in arms:
        base = os.path.join(root, d)
        if seed_list is None:
            pre = f"{label}_{env}_s"
            seed_list = sorted(int(n[len(pre):-5]) for n in os.listdir(base)
                               if n.startswith(pre) and n.endswith(".json") and n[len(pre):-5].isdigit())
        if not seed_list:
            raise SystemExit(f"s01: no runs of {label} on {env} in {base}")
        for s in seed_list:
            stem = os.path.join(base, f"{label}_{env}_s{s}")
            with open(stem + ".json") as f:
                N = int(json.load(f)["args"]["steps"])
            inputs[stem + ".json"] = sha256(stem + ".json")
            steps = snapshot_steps(stem + ".snap")
            snaps = []
            for frac in S01_FRACS:
                st = nearest(steps, frac * N) if steps else None
                if st is None or (st != N if frac == 1.0 else abs(st - frac * N) > S01_SNAP_TOL):
                    raise SystemExit(f"s01: {stem}.snap has no snapshot at {frac:g}N = {frac * N:g} "
                                     f"(nearest {st}; tolerance {0 if frac == 1.0 else S01_SNAP_TOL})")
                p = os.path.join(stem + ".snap", f"{st}.pt")
                inputs[p] = sha256(p)
                snaps.append({"frac": frac, "step": st, "path": p})
            run = {"arm": f"{label}@{env}", "label": label, "env": env, "seed": s, "stem": stem, "N": N, "snaps": snaps}
            if (label, env) == S01_HEALTHY[:2]:
                hp = stem + ".heldout.json"
                with open(hp) as f:
                    run["heldout_return"] = float(json.load(f)["pooled"]["eval_return"])
                inputs[hp] = sha256(hp)
            runs.append(run)
    names = [os.path.basename(r["stem"]) for r in runs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"s01: run names repeat across directories (traces are keyed by name): {dup}")
    return runs, inputs


def _trace_name(run: dict, frac: float) -> str:
    return f"{os.path.basename(run['stem'])}.f{frac:.1f}.npz"


def s01_score(runs: list[dict], k: int, predictors: list[str], W: np.ndarray) -> dict:
    """make.py's AUROC table (pooled, cluster bootstrap, per-run median, minus Q_c) plus pooled AUPRC with
    its cluster bootstrap and the base rate, for every predictor at lead k."""
    masks = [targets(r["cost"], r["ep"], k) for r in runs]
    ys = [y[u] for u, y in masks]
    y_all = np.concatenate(ys)
    base = float(y_all.mean()) if len(y_all) else float("nan")
    P_r = np.array([y.sum() for y in ys], float)
    N_r = np.array([len(y) for y in ys], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        base_boot = (W @ P_r) / (W @ N_r)
    out, boots = {}, {}
    for p in predictors:
        xs = [r[p][u].astype(np.float64) for r, (u, _) in zip(runs, masks)]
        x_all = np.concatenate(xs)
        auc = auroc(x_all, y_all)
        U, P, N = pair_matrix(xs, ys)
        bt = boot_auroc(U, P, N, W)
        boots[p] = bt
        per = np.array([auroc(x, y) for x, y in zip(xs, ys)])
        ok = np.isfinite(per)
        ap = average_precision(x_all, y_all)
        pq = ap_counts(xs, ys)
        bap = boot_ap(*pq, W)
        out[p] = {"auroc": auc, "auroc_ci": pct(bt), "per_run_median": float(np.median(per[ok])) if ok.any() else float("nan"),
                  "per_run_defined": int(ok.sum()), "auprc": ap, "auprc_ci": pct(bap), "base_rate": base,
                  "base_rate_ci": pct(base_boot), "auprc_over_base": ap / base if base > 0 else float("nan"),
                  "undefined_resamples": int((~np.isfinite(bt)).sum())}
    for p in predictors:
        if p != "qc" and "qc" in boots:
            out[p]["minus_qc"] = out[p]["auroc"] - out["qc"]["auroc"]
            out[p]["minus_qc_ci"] = pct(boots[p] - boots["qc"])
    return out


def s01(root: str = MAIN_CHECKOUT, out: str = os.path.join(DEFAULT_OUT, "s01"), workers: int = 3, arms=None,
        block: str = "s01", episodes: int = S01_EPISODES, n_boot: int = B) -> dict:
    """Run the S0-1 extensions (EXPLORATORY) and write <out>/s01.json, s01.txt, s01_scores.csv, s01_curves.csv."""
    runs, inputs = s01_plan(root, arms)
    jobs = []
    for r in runs:
        for sn in r["snaps"]:
            sn["trace"] = os.path.join(out, "traces", _trace_name(r, sn["frac"]))
            jobs.append({"snap": sn["path"], "json": r["stem"] + ".json", "trace": sn["trace"], "block": block,
                         "episodes": episodes, "step": sn["step"]})
    _map(_s01_rollout, jobs, workers)

    groups: dict = {}
    healthy = []
    for r in runs:
        groups.setdefault(r["arm"], []).append(r)
        if (r["label"], r["env"]) == S01_HEALTHY[:2] and r["heldout_return"] >= S01_HEALTHY[2]:
            groups.setdefault(f"{r['arm']} (held-out return >= {S01_HEALTHY[2]:g})", []).append(r)
            healthy.append(os.path.basename(r["stem"]))
    n_candidates = sum((r["label"], r["env"]) == S01_HEALTHY[:2] for r in runs)
    lines = []
    say = lambda s="": (lines.append(s), print(s))
    say("EXPLORATORY - S0-1 extensions: does a signal anticipate cost? (gates nothing; not confirmatory)")
    say(f"Deterministic episodes on layouts {seeds.layout(block, 0)}-{seeds.layout(block, episodes - 1)}, "
        f"snapshots nearest {', '.join(f'{f:g}N' for f in S01_FRACS)}; y_k(t) = 1 if any cost in t+1..t+k. "
        f"95% seed-cluster bootstrap ({n_boot:,} resamples, rng 0).")
    say(f"Healthy {S01_HEALTHY[0]}@{S01_HEALTHY[1]} runs (held-out return >= {S01_HEALTHY[2]:g}): "
        f"{len(healthy)} of {n_candidates}" + ("" if len(healthy) >= 2 else " (fewer than 2: no bootstrap is meaningful)"))
    scores, curves, table = [], [], {}
    for arm, members in groups.items():
        W = weights(len(members), n_boot)
        for fi, frac in enumerate(S01_FRACS):
            data = []
            for r in members:
                with np.load(r["snaps"][fi]["trace"]) as z:
                    d = {k: z[k] for k in ("ep", "t", "qbar", "std", "ratio", "qc", "lidar", "cost", "reward")}
                    d["negqbar"] = -d["qbar"]
                    d["M"], d["has_lidar"] = int(z["M"]), bool(z["has_lidar"])
                data.append(d)
            preds = [p for p, _ in PREDICTORS
                     if not (p in ENSEMBLE_ONLY and min(d["M"] for d in data) < 2)
                     and not (p == "lidar" and not all(d["has_lidar"] for d in data))]
            for d in data:
                for p in preds:
                    if not np.all(np.isfinite(d[p])):
                        raise SystemExit(f"s01 {arm}: non-finite {p}; the copied metric has no rule for it")
            ret = float(np.mean([d["reward"].sum() / episodes for d in data]))
            cst = float(np.mean([d["cost"].sum() / episodes for d in data]))
            say()
            say(f"== EXPLORATORY {arm}, snapshot {frac:g}N ({len(data)} runs): return {ret:.1f}, cost {cst:.1f}")
            say(f"  {'k':>3s} {'predictor':24s} {'AUROC [95% CI]':>23s} {'per-run med':>11s} "
                f"{'minus Q_c [95% CI]':>25s} {'AUPRC [95% CI]':>23s} {'base rate':>9s}")
            for k in LEADS:
                res = s01_score(data, k, preds, W)
                table[f"{arm}|{frac:g}N|k={k}"] = res
                for p, plabel in PREDICTORS:
                    if p not in res:
                        continue
                    c = res[p]
                    mq = (f"{c['minus_qc']:+8.3f} [{c['minus_qc_ci'][0]:+.3f}, {c['minus_qc_ci'][1]:+.3f}]"
                          if "minus_qc" in c else f"{'(reference)':>25s}")
                    say(f"  {k:3d} {plabel:24s} {c['auroc']:7.3f} [{c['auroc_ci'][0]:.3f}, {c['auroc_ci'][1]:.3f}] "
                        f"{c['per_run_median']:11.3f} {mq} {c['auprc']:7.3f} [{c['auprc_ci'][0]:.3f}, "
                        f"{c['auprc_ci'][1]:.3f}] {c['base_rate']:9.3f}")
                    scores.append({"analysis": "EXPLORATORY", "arm": arm, "snapshot": f"{frac:g}N", "k": k,
                                   "predictor": p, "runs": len(data), "auroc": c["auroc"],
                                   "auroc_lo": c["auroc_ci"][0], "auroc_hi": c["auroc_ci"][1],
                                   "per_run_median": c["per_run_median"], "per_run_defined": c["per_run_defined"],
                                   "minus_qc": c.get("minus_qc", ""),
                                   "minus_qc_lo": c.get("minus_qc_ci", ("", ""))[0],
                                   "minus_qc_hi": c.get("minus_qc_ci", ("", ""))[1],
                                   "auprc": c["auprc"], "auprc_lo": c["auprc_ci"][0], "auprc_hi": c["auprc_ci"][1],
                                   "base_rate": c["base_rate"], "auprc_over_base": c["auprc_over_base"],
                                   "undefined_resamples": c["undefined_resamples"]})
            for p in preds:
                cv = aligned_curve(data, p)
                for o, mval, n in zip(cv["offsets"], cv["mean"], cv["n"]):
                    curves.append({"analysis": "EXPLORATORY", "arm": arm, "snapshot": f"{frac:g}N", "predictor": p,
                                   "offset": o, "mean": mval, "n": n, "entries": cv["entries"]})
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "s01.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for name, rows in (("s01_scores.csv", scores), ("s01_curves.csv", curves)):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["analysis"])
            w.writeheader()
            w.writerows(rows)
    res = {"probe": "s01", "analysis": "EXPLORATORY", "block": block, "episodes": episodes, "fracs": list(S01_FRACS),
           "leads": list(LEADS), "n_boot": n_boot, "healthy": list(S01_HEALTHY),
           "healthy_runs": {"stems": healthy, "count": len(healthy), "of": n_candidates}, "inputs": inputs,
           "code": code_hashes("probe"),
           "runs": [{k: v for k, v in r.items() if k != "snaps"} | {"snapshots": [{"frac": s["frac"], "step": s["step"]}
                                                                               for s in r["snaps"]]} for r in runs],
           "arms": {a: [os.path.basename(r["stem"]) for r in m] for a, m in groups.items()}, "scores": table}
    write_json(os.path.join(out, "s01.json"), res)
    return res


# --- S0-4 (descriptive; feeds kill rule 8.4): the design script simplest/h3_grad.py as a function -------
S04_DESIGN_RUNS = tuple(os.path.join(MAIN_CHECKOUT, "results", d, f"{st}_SafetyPointGoal1-v0_s{s}.ckpt.pt")
                        for d, st, s in (("pareto", "td3lag-omni", 0), ("pareto", "td3lag-omni", 1),
                                         ("pareto", "td3lag-omni", 2), ("pareto", "td3lag-omni-fix03", 1),
                                         ("pareto", "td3lag-fix005", 0), ("e0_dual", "lagu_nogate-omsw-l4e-7", 0),
                                         ("e0_dual", "lagu_nogate-tt-l2e-6", 0), ("e0_dual", "lagu_nogate-tt-l4e-7", 0)))
S04_EPISODE = 1000            # only complete episodes of this length are used (h3_grad.py)
S04_STATES = 500              # the first 500 steps of each
S04_QUIET, S04_IMMINENT = 30, 10      # quiet: no cost in t+1..t+30; imminent: cost in t+1..t+10


def s04_labels(c: np.ndarray) -> np.ndarray:
    """h3_grad.py's labels of the first S04_STATES steps of one episode's costs: 2 in a hazard (cost at t),
    1 imminent, 0 quiet, -1 other."""
    cc = np.concatenate([np.asarray(c, np.float64), np.zeros(40)])
    lab = np.full(S04_STATES, -1, int)
    for t in range(S04_STATES):
        if cc[t] > 0:
            lab[t] = 2
        elif cc[t + 1:t + 1 + S04_IMMINENT].sum() > 0:
            lab[t] = 1
        elif cc[t + 1:t + 1 + S04_QUIET].sum() == 0:
            lab[t] = 0
    return lab


def s04_states(rep: dict, nep: int) -> tuple:
    """(rows, labels): the first S04_STATES rows of the last `nep` complete S04_EPISODE-step episodes of a
    replay state {o, o2, c, ptr, n} (episode breaks where o2[i] != o[i + 1], as h3_grad.py)."""
    o, o2, c = _np(rep["o"]), _np(rep["o2"]), _np(rep["c"])
    n, ptr, size = int(rep["n"]), int(rep["ptr"]), o.shape[0]
    order = np.arange(n) if n < size else (np.arange(ptr, ptr + size) % size)
    idx = order[-(nep + 2) * S04_EPISODE:]
    brk = np.where(np.any(o2[idx[:-1]] != o[idx[1:]], axis=1))[0]
    eps = [(a, b) for a, b in zip(brk[:-1] + 1, brk[1:]) if b - a + 1 == S04_EPISODE][-nep:]
    rows = [idx[a:a + S04_STATES] for a, _ in eps]
    labs = [s04_labels(c[idx[a:b + 1]]) for a, b in eps]
    return (np.concatenate(rows) if rows else np.zeros(0, int)), (np.concatenate(labs) if labs else np.zeros(0, int))


def s04_agent(ck: dict, obs_dim: int, act_dim: int):
    """(agent, lambda, s) of a checkpoint: a safe2 ckpt ({config, agent, ...}; s = the SHL rate scale) or a
    lagu.train ckpt ({agent: {cfg, lambda, ...}}; s = 1)."""
    if "config" in ck:
        from safe2.base import Agent
        from safe2.configs import RunConfig
        rc = RunConfig(**ck["config"])
        with torch.random.fork_rng():
            agent = Agent(obs_dim, act_dim, rc, S04_EPISODE)
        agent.load_state_dict(ck["agent"])
        return agent, float(agent.lmbda), float(agent.rate_scale)
    from lagu.agent import Config, LagU
    with torch.random.fork_rng():
        agent = LagU(obs_dim, act_dim, Config(**ck["agent"]["cfg"]))
    agent.load_state_dict(ck["agent"])
    return agent, float(ck["agent"]["lambda"]), 1.0


def s04_run(ckpt: str, nep: int = 100) -> dict:
    """Action gradients the actor sees at a = pi(s) on replay states: |grad_a lambda s Q_c| and |grad_a Q_bar|.
    share = sum of |grad_a Q_c| over quiet states / sum over all states (lambda s is one constant per run)."""
    with torch.serialization.safe_globals(NUMPY_SCALARS):
        ck = torch.load(ckpt, weights_only=True, mmap=True)
    rep = ck["replay"]
    rows, lab = s04_states(rep, nep)
    if len(lab) == 0:
        raise SystemExit(f"s04: no complete {S04_EPISODE}-step episode in the replay of {ckpt}")
    o = _np(rep["o"])[rows].astype(np.float32)
    if "config" in ck and ck["config"].get("budget_input"):
        o = np.concatenate([o, _np(rep["b"])[rows].reshape(-1, 1).astype(np.float32)], 1)
        obs_dim = o.shape[1] - 1
    else:
        obs_dim = o.shape[1]
    act_dim = _np(rep["a_pi" if "a_pi" in rep else "a"]).shape[1]
    agent, lam, scale = s04_agent(ck, obs_dim, act_dim)
    del ck, rep
    ot = torch.as_tensor(o)
    with torch.no_grad():
        api = agent.actor(ot)
    at = api.clone().requires_grad_(True)
    gc = torch.autograd.grad(agent.qc_pi(ot, at).sum(), at)[0].numpy()
    at2 = api.clone().requires_grad_(True)
    gr = torch.autograd.grad(agent.ensemble(ot, at2)[0].sum(), at2)[0].numpy()
    nc, nr = np.linalg.norm(gc, axis=1), np.linalg.norm(gr, axis=1)
    pen = lam * scale * nc
    res = {"ckpt": ckpt, "lambda": lam, "rate_scale": scale, "episodes": int(len(lab) // S04_STATES),
           "n": int(len(lab)), "share": float(nc[lab == 0].sum() / nc.sum()) if nc.sum() > 0 else float("nan"),
           "share_states_quiet": float((lab == 0).mean())}
    for name, k in (("quiet", lab == 0), ("imminent", lab == 1), ("in_hazard", lab == 2), ("all", lab > -9)):
        res[name] = {"frac": float(k.mean()), "n": int(k.sum())} | ({
            "grad_qc_median": float(np.median(nc[k])), "grad_qbar_median": float(np.median(nr[k])),
            "penalty_over_reward_median": float(np.median(pen[k] / (nr[k] + 1e-12))),
            "frac_penalty_dominates": float((pen[k] > nr[k]).mean())} if k.any() else {})
    return res


def s04(ckpts=S04_DESIGN_RUNS, out: str = DEFAULT_OUT, nep: int = 100, force: bool = False) -> dict:
    """S0-4: s04_run on each checkpoint, merged into <out>/s04.json under its stem (the file name without
    .ckpt.pt), which kill rule 8.4 reads. A stem already recorded is refused unless force."""
    for p in ckpts:
        if not os.path.exists(p):
            raise SystemExit(f"s04: missing input {p}")
    path = os.path.join(out, "s04.json")
    res = {"probe": "s04", "analysis": "descriptive (S0-4; kill rule 8.4 reads 'share')", "runs": {}, "inputs": {}}
    if os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
    stems = [os.path.basename(p)[:-len(".ckpt.pt")] for p in ckpts]
    again = sorted(set(stems) & set(res["runs"]))
    if again and not force:
        raise SystemExit(f"s04: {again} already in {path}; a re-run needs --force")
    res["settings"] = {"nep": nep, "episode": S04_EPISODE, "states": S04_STATES, "quiet": S04_QUIET,
                       "imminent": S04_IMMINENT}
    res["code"] = code_hashes("probe")
    for st, p in zip(stems, ckpts):
        res["inputs"][p] = sha256(p)
        res["runs"][st] = s04_run(p, nep)
        print(f"s04 {st}: share {res['runs'][st]['share']:.3f} ({res['runs'][st]['episodes']} episodes)")
        write_json(path, res)
    return res


# --- CLI ----------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("g0a", help="oracle ceiling (privileged)")
    a.add_argument("--runs", default=DEFAULT_RUNS)
    a.add_argument("--env", default=G0_ENV)
    a.add_argument("--out", default=DEFAULT_OUT)
    a.add_argument("--workers", type=int, default=3)
    a.add_argument("--force", action="store_true", help="overwrite g0a.json (recorded in deviations.json)")
    b = sub.add_parser("g0b", help="offline V: replay identity, recall, ranking")
    b.add_argument("--runs", default=DEFAULT_RUNS)
    b.add_argument("--pareto", default=os.path.join(MAIN_CHECKOUT, "results", "pareto"))
    b.add_argument("--env", default=G0_ENV)
    b.add_argument("--out", default=DEFAULT_OUT)
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--reuse-fits", action="store_true", help="use the fits already in <out>/g0b/")
    b.add_argument("--force", action="store_true", help="overwrite g0b.json (recorded in deviations.json)")
    s = sub.add_parser("s01", help="EXPLORATORY S0-1 extensions")
    s.add_argument("--root", default=MAIN_CHECKOUT)
    s.add_argument("--out", default=os.path.join(DEFAULT_OUT, "s01"))
    s.add_argument("--workers", type=int, default=3)
    h = sub.add_parser("phs-quality", help="PHS fit quality of one B* run (sealed; prints no number)")
    h.add_argument("run", help="the run path without extension (conf/<stem>)")
    h.add_argument("--root", default=None, help="sealing root (default: the run's results/method)")
    h.add_argument("--force", action="store_true", help="replace an existing sealed output")
    q = sub.add_parser("s04", help="S0-4 gradient probe (descriptive; kill rule 8.4)")
    q.add_argument("--runs", nargs="+", default=list(S04_DESIGN_RUNS), help="*.ckpt.pt files (default: design runs)")
    q.add_argument("--nep", type=int, default=100)
    q.add_argument("--out", default=DEFAULT_OUT)
    q.add_argument("--force", action="store_true", help="replace runs already in s04.json")
    args = p.parse_args(argv)
    if args.cmd == "g0a":
        r = g0a(args.runs, args.out, G0aConfig(env=args.env), workers=args.workers, force=args.force)
        for x in r["actors"]:
            print(f"g0a {x['stem']}: return {x['mean_return']:.2f} cost {x['mean_cost']:.2f} "
                  f"interventions/ep {x['mean_interventions']:.1f}")
        print(f"g0a: {r['n_episodes']} episodes, return {r['mean_return']:.2f} cost {r['mean_cost']:.2f} "
              f"-> {os.path.join(args.out, 'g0a.json')} (decision at method-v0.5)")
    elif args.cmd == "g0b":
        r = g0b(args.runs, args.pareto, args.out, G0bConfig(env=args.env), workers=args.workers,
                reuse_fits=args.reuse_fits, force=args.force)
        for st, idn in r["identity"].items():
            print(f"g0b identity {st}: {idn['result']}" + (f" (first row {idn['first_row']}: {idn['columns']})"
                                                         if idn["result"] == "differs" else ""))
        pr, rk, dc = r["recall"]["pooled"], r["ranking"], r["decision"]
        print(f"g0b (i): recall {pr['recall']:.3f} ({pr['caught']}/{pr['entries']}), false alarms/episode "
              f"{pr['fa_per_episode']:.2f} -> {'pass' if dc['pass_i'] else 'FAIL'}")
        print(f"g0b (ii): {rk['states']} states, Spearman {rk['spearman']:.3f} -> {'pass' if dc['pass_ii'] else 'FAIL'}")
        print(f"g0b: {'PROCEED' if dc['proceed'] else 'KILL'} -> {os.path.join(args.out, 'g0b.json')}")
    elif args.cmd == "phs-quality":
        phs_quality(args.run, args.root, force=args.force)
        print(f"{os.path.basename(args.run)}: done")
    elif args.cmd == "s04":
        s04(args.runs, args.out, args.nep, args.force)
    else:
        s01(args.root, args.out, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
