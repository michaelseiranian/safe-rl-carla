"""DESCRIPTIVE verification of results/paper/probe-author (the gradient-parity probe of prereg.md
section 9). Nothing here is confirmatory, and nothing here computes any probe outcome on study data.

    cd /Users/michaelseiranian/Desktop/safe-rl-carla && .venv/bin/python results/paper/probe-author-verify/make.py

READS NO STUDY SNAPSHOT, CSV, heldout.json OR STUDY .json CONTENT. For the declared study files it
only calls os.stat (existence and access time), which does not change them. Everything that is
probed is a scratch model trained from scratch here (hidden 32, 20k steps, seed 7) or a random
untrained agent. All writes go to the scratchpad, except verify_tables.txt next to this script.

Sections (every table is DESCRIPTIVE):
  V1  hashes of the three delivered files (probe.py must equal the reported SHA-256)
  V2  static audit of probe.py: every file-reading call, every csv/heldout string, PROTOCOL constants
  V3  declared sets: own enumeration from prereg.md section 3 / 15 vs probe.declared_jobs; existence
      and access time of the declared files (os.stat only)
  V4  --dry-run run as `python -m lagu.probe` from a scratch mirror package (lagu/agent.py, train.py
      symlinked; probe.py copied byte for byte) under a file-open audit hook; own numpy-float64
      lambda* (manual backprop) on the same random agents
  V5  the author's make.py, copied with only its scratch path redirected to this task's scratchpad;
      it retrains the scratch models; its outputs are compared with the author's saved outputs
  V6  own re-derivation (independent code) of lambda*, R_ckpt, R0 and R(lambda) on the scratch models
      with the probe's --quick constants, against the probe's job files; for the M = 3 model also with
      only the floating-point summation order of the action gradient changed (numerical sensitivity)
  V7  `python -m lagu.probe --stems ... --quick --workers 2` (spawned workers re-import lagu.probe)
  V8  --prereg end to end with the FULL protocol on a fake repository root holding two scratch
      checkpoints under declared names (everything else 'missing'), workers 1, under the audit hook;
      own re-derivation of lambda*, R_ckpt, R0 with the full constants
  V9  decision-rule functions against own re-implementations on random inputs; the grid stop rule
      against the full grid; the overall label for every pair of per-set labels
  V10 own timing and the run-cost upper bound with the per-checkpoint evaluation count corrected
"""
from __future__ import annotations

import ast
import datetime as dt
import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = "/Users/michaelseiranian/Desktop/safe-rl-carla"
AUTH = os.path.join(REPO, "results/paper/probe-author")
HERE = os.path.join(REPO, "results/paper/probe-author-verify")
SCR_ROOT = "/private/tmp/claude-501/-Users-michaelseiranian-Desktop-safe-rl-carla/47ca4e66-e2f4-4455-bfea-c56f9a36fb6d/scratchpad/paper"
AUTH_SCR = os.path.join(SCR_ROOT, "probe-author")
V = os.path.join(SCR_ROOT, "probe-author-verify")
PY = os.path.join(REPO, ".venv/bin/python")
ENVV = {**os.environ, "PYTHONPATH": REPO, "PYTHONDONTWRITEBYTECODE": "1"}
REPORTED = {"probe_sha256": "715afadee7184d6dafae70c1a6463fab72428f19404545024ce8d756c80e187e",
            "dry_lam_star": {"lagu_nogate": 0.863, "td3lag": 0.714}, "scratch_td3lag_lam_star": 0.02013,
            "n_main": 45, "n_traj": 280, "upper_cpu_h": 9.2}
OUT: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    OUT.append(s)


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sh(args, cwd=REPO, env=None, check=True) -> str:
    r = subprocess.run(args, cwd=cwd, env=env or ENVV, capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "AdroitHand" not in l)
    if check and r.returncode:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(args)}\n{out}")
    return out


def load_probe():
    spec = importlib.util.spec_from_file_location("probe_under_test", os.path.join(AUTH, "probe.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(mod)
    return mod


AUDIT_RUNNER = '''"""Run a module as `python -m <mod>` would (cwd first on sys.path) under an audit hook that
records every file path opened in this process."""
import json, os, runpy, sys
log, done = [], [False]
def hook(ev, args):
    if ev == "open" and not done[0] and args and isinstance(args[0], (str, bytes)):
        log.append([args[0] if isinstance(args[0], str) else args[0].decode(), str(args[1]) if len(args) > 1 else ""])
sys.addaudithook(hook)
out, mod = sys.argv[1], sys.argv[2]
sys.argv = [mod] + sys.argv[3:]
sys.path[0] = os.getcwd()
code = 0
try:
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
except SystemExit as e:
    code = e.code or 0
except BaseException as e:
    code = f"{type(e).__name__}: {e}"
    raise
finally:
    done[0] = True
    with open(out, "w") as f:
        json.dump({"exit": code, "opened": log}, f)
sys.exit(code if isinstance(code, int) else 1)
'''


# ------------------------------------------------------------------ independent re-derivation code
def mlp_np(sd: dict, prefix: str):
    return [(sd[f"{prefix}{i}.weight"].double().numpy(), sd[f"{prefix}{i}.bias"].double().numpy()) for i in (0, 2, 4)]


def np_forward_grad(layers, x: np.ndarray, cols):
    """ReLU MLP value and d value / d x[:, cols], float64 numpy, manual backprop."""
    (W1, b1), (W2, b2), (W3, b3) = layers
    h1 = x @ W1.T + b1
    m1 = h1 > 0
    h2 = (h1 * m1) @ W2.T + b2
    m2 = h2 > 0
    y = (h2 * m2) @ W3.T + b3
    g = ((W3[0][None, :] * m2) @ W2 * m1) @ W1
    return y[:, 0], g[:, cols]


def np_lam_star(actor_sd, members_sd, M, cost_sd, o64: np.ndarray, act_dim: int) -> dict:
    (W1, b1), (W2, b2), (W3, b3) = mlp_np(actor_sd, "net.")
    a = np.tanh(np.maximum(np.maximum(o64 @ W1.T + b1, 0) @ W2.T + b2, 0) @ W3.T + b3)
    x = np.concatenate([o64, a], 1)
    cols = list(range(o64.shape[1], o64.shape[1] + act_dim))
    gr = np.mean([np_forward_grad(mlp_np(members_sd, f"{i}.q1."), x, cols)[1] for i in range(M)], 0)
    gc = np_forward_grad(mlp_np({"c." + k: v for k, v in cost_sd.items()}, "c."), x, cols)[1]
    nr, nc = np.linalg.norm(gr, axis=1), np.linalg.norm(gc, axis=1)
    return {"lam_star": float(nr.mean() / nc.mean()), "g_r": float(nr.mean()), "g_c": float(nc.mean()),
            "cos": float(np.mean((gr * gc).sum(1) / (nr * nc))), "ratio_median": float(np.median(nr / nc)),
            "sat": float((np.abs(a) > 0.99).mean())}


def own_probe(stem: str, step, C: dict, grid=None, share_x: bool = True) -> dict:
    """Independent implementation (own nets, own rollouts, own update loop) of the probe's steps on a
    scratch checkpoint. C: state_episodes, parity_states, eval_episodes, K. share_x: the M reward
    heads read one concatenated (s, a) tensor (as the probe does); False gives each head its own copy,
    which changes only the floating-point order in which autograd sums the action gradients."""
    import safety_gymnasium
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    path = f"{stem}.pt" if step == "final" else os.path.join(stem + ".snap", f"{step}.pt")
    d = torch.load(path, weights_only=True)
    meta = json.load(open(stem + ".json"))

    def build(sd, prefix):
        W = [sd[f"{prefix}{i}.weight"] for i in (0, 2, 4)]
        net = nn.Sequential(nn.Linear(W[0].shape[1], W[0].shape[0]), nn.ReLU(), nn.Linear(W[1].shape[1], W[1].shape[0]),
                            nn.ReLU(), nn.Linear(W[2].shape[1], W[2].shape[0]))
        net.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)})
        return net

    M = d["cfg"]["M"]
    q1s = [build(d["members"], f"{i}.q1.").requires_grad_(False) for i in range(M)]
    qc = build(d["cost"], "").requires_grad_(False)
    def qbar(o, a):
        if share_x:
            x = torch.cat([o, a], -1)
            return torch.stack([q(x)[:, 0] for q in q1s]).mean(0)
        return torch.stack([q(torch.cat([o, a], -1))[:, 0] for q in q1s]).mean(0)
    qcost = lambda o, a: qc(torch.cat([o, a], -1))[:, 0]
    env = safety_gymnasium.make(meta["args"]["env"])

    def rollout(actor, base, n):
        S, rets, costs = [], [], []
        for k in range(n):
            o, _ = env.reset(seed=base + k)
            R = Cc = 0.0
            done = False
            while not done:
                S.append(o)
                with torch.no_grad():
                    a = torch.tanh(actor(torch.as_tensor(o, dtype=torch.float32)[None]))[0].numpy()
                o, r, c, te, tr, _ = env.step(np.clip(a, -1, 1).astype(np.float32))
                R += r
                Cc += c
                done = te or tr
            rets.append(R)
            costs.append(Cc)
        return np.asarray(S, np.float32), float(np.mean(rets)), float(np.mean(costs))

    actor0 = build(d["actor"], "net.")
    S, _, _ = rollout(actor0, 12000, C["state_episodes"])
    sub = np.sort(np.random.default_rng(0).choice(len(S), size=min(C["parity_states"], len(S)), replace=False))
    par = np_lam_star(d["actor"], d["members"], M, d["cost"], S[sub].astype(np.float64), meta["act_dim"])
    _, r_ck, _ = rollout(actor0, 11000, C["eval_episodes"])
    idx = np.random.default_rng(0).integers(0, len(S), size=(C["K"], 256))
    St = torch.as_tensor(S)
    rets = {}
    for lam in ([0.0] + list(grid or [])):
        act = build(d["actor"], "net.")
        opt = torch.optim.Adam(act.parameters(), lr=3e-4)
        for k in range(C["K"]):
            ob = St[idx[k]]
            ab = torch.tanh(act(ob))
            loss = (-qbar(ob, ab) + lam * qcost(ob, ab)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        rets[lam] = rollout(act, 11000, C["eval_episodes"])[1]
    env.close()
    return {"n_states": len(S), "parity": par, "R_ckpt": r_ck, "R": rets, "snap_keys": sorted(d.keys())}


# ------------------------------------------------------------------ own decision-rule code
def my_halving(lams, rets, r0):
    h = r0 / 2
    for j, r in enumerate(rets):
        if r is None:
            break
        if r <= h:
            if j == 0:
                return lams[0], "left"
            l0, l1, ra, rb = lams[j - 1], lams[j], rets[j - 1], r
            t = (ra - h) / (ra - rb)
            return 10 ** (math.log10(l0) + t * (math.log10(l1) - math.log10(l0))), ""
    return lams[-1], "right"


def my_stop(rets, r0):
    out, below = [], 0
    for r in rets:
        out.append(r)
        below = below + 1 if r <= 0.25 * r0 else 0
        if below == 2:
            break
    return out + [None] * (len(rets) - len(out))


def my_verdict(x, n_boot=10_000):
    x = np.asarray(x, float)
    if len(x) < 5:
        return "NOT EVALUABLE", math.nan, (math.nan, math.nan)
    rng = np.random.default_rng(0)
    draws = x[rng.integers(0, len(x), size=(n_boot, len(x)))]
    s = np.sort(draws, axis=1)
    n = len(x)
    meds = s[:, n // 2] if n % 2 else 0.5 * (s[:, n // 2 - 1] + s[:, n // 2])
    q = np.sort(meds)
    def pct(p):  # linear interpolation, as numpy's default
        pos = p / 100 * (len(q) - 1)
        lo = int(math.floor(pos))
        return q[lo] + (pos - lo) * (q[min(lo + 1, len(q) - 1)] - q[lo])
    lo, hi = pct(2.5), pct(97.5)
    lab = "SUPPORTED" if (lo >= -1 and hi <= 1) else "REFUTED" if (lo > 1 or hi < -1) else "INCONCLUSIVE"
    return lab, 2 ** float(np.median(x)), (2 ** lo, 2 ** hi)


def my_spearman(x, y):
    def avg_rank(v):
        v = np.asarray(v, float)
        r = np.empty(len(v))
        order = np.argsort(v)
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = avg_rank(x), avg_rank(y)
    if len(rx) < 5 or rx.std() == 0 or ry.std() == 0:
        return math.nan
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / math.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


# ===================================================================================== sections
def v1() -> None:
    say("== V1 [DESCRIPTIVE] file hashes")
    for f in ("probe.py", "PROTOCOL.md", "make.py"):
        say(f"  {f:<12} {sha(os.path.join(AUTH, f))}")
    ok = sha(os.path.join(AUTH, "probe.py")) == REPORTED["probe_sha256"]
    say(f"  probe.py equals the reported SHA-256 715afade...: {ok}")
    say(f"  lagu/probe.py exists in the repository (should not, until committed): {os.path.exists(os.path.join(REPO, 'lagu/probe.py'))}")


def v2(probe) -> None:
    say("\n== V2 [DESCRIPTIVE] static audit of probe.py")
    src = open(os.path.join(AUTH, "probe.py")).read()
    tree = ast.parse(src)
    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in ("open", "torch.load", "json.load", "np.load", "np.loadtxt", "csv.reader", "csv.DictReader"):
                reads.append((node.lineno, ast.unparse(node)[:110]))
    say("  every file-reading call (line: call):")
    for ln, s in sorted(reads):
        say(f"    {ln:>4}: {s}")
    hits = [(i + 1, l.strip()[:100]) for i, l in enumerate(src.splitlines()) if re.search(r"\.csv|heldout\.json|read_csv", l)]
    say(f"  lines mentioning '.csv', 'heldout.json' or 'read_csv': {hits if hits else 'none'}")
    P = probe.PROTOCOL
    g = P["grid"]
    checks = {
        "probe_seed == 12000": P["probe_seed"] == 12000, "state_episodes == 20": P["state_episodes"] == 20,
        "parity_states == 5000": P["parity_states"] == 5000, "heldout_seed == 11000": P["heldout_seed"] == 11000,
        "eval_episodes == 20": P["eval_episodes"] == 20, "K == 10000": P["K"] == 10_000, "lr == 3e-4": P["lr"] == 3e-4,
        "batch == 256": P["batch"] == 256, "rng_seed == 0": P["rng_seed"] == 0,
        "grid: 22 values, 0 then 1e-3..1e2, 4/decade": len(g) == 22 and g[0] == 0 and g[1] == 0.001 and g[-1] == 100.0
        and np.allclose(np.diff(np.log10(g[1:])), 0.25),
        "stop_frac == 0.25": P["stop_frac"] == 0.25, "min_return == 10": P["min_return"] == 10.0,
        "min_frac_of_ckpt == 0.5": P["min_frac_of_ckpt"] == 0.5, "factor == 2": P["factor"] == 2.0,
        "n_boot == 10000": P["n_boot"] == 10_000, "min_eligible == 5": P["min_eligible"] == 5,
        "traj_every == 250000": P["traj_every"] == 250_000,
        "EVAL_SEED == lagu.train.EVAL_SEED": probe.EVAL_SEED == __import__("lagu.train", fromlist=["x"]).EVAL_SEED,
    }
    say("  PROTOCOL constants against the report: " + ("all match" if all(checks.values()) else
        "MISMATCH " + str([k for k, v in checks.items() if not v])))


def v3(probe) -> None:
    say("\n== V3 [DESCRIPTIVE] declared checkpoint sets (own enumeration from prereg sections 3, 15, 16; os.stat only)")
    own = []   # (set, dir, arm, seeds, N): S1 E1b lagu_nogate-lit; S2 E1 lagu_nogate-tt (s100-114, 2M per the
    #            E0 rule and the E0-ext keep-2M decision); S3a E1b td3lag-lit; S3b E1 td3lag-tt
    for sid, d, arm, seeds, N in (("S1", "results/gate_literal", "lagu_nogate-lit", range(100, 110), 1_000_000),
                                  ("S2", "results/gate", "lagu_nogate-tt", range(100, 115), 2_000_000),
                                  ("S3a", "results/gate_literal", "td3lag-lit", range(100, 110), 1_000_000),
                                  ("S3b", "results/gate", "td3lag-tt", range(100, 110), 2_000_000)):
        for s in seeds:
            stem = os.path.join(REPO, d, f"{arm}_SafetyPointGoal1-v0_s{s}")
            own.append(("main", sid, stem, N))
            own += [("traj", sid, stem, t) for t in range(250_000, N + 1, 250_000)]
    theirs = [(j["kind"], j["set"], os.path.normpath(j["stem"]), j["step"]) for j in probe.declared_jobs(REPO, probe.PROTOCOL)]
    n_main = sum(k == "main" for k, *_ in own)
    say(f"  own: {n_main} main + {len(own) - n_main} trajectory jobs; probe.declared_jobs: "
        f"{sum(k == 'main' for k, *_ in theirs)} + {sum(k == 'traj' for k, *_ in theirs)}; identical sets: {sorted(own) == sorted(theirs)}")
    pairs = {(st, ss) for _, ss, st, _ in own}
    multi = sorted({st for st, _ in pairs if sum(1 for x, _ in pairs if x == st) > 1})
    say(f"  runs declared in more than one set (would expose the cache-label bug the author fixed): {len(multi)}")
    today = dt.datetime(2026, 10, 11).timestamp()
    rows = []
    for sid in ("S1", "S2", "S3a", "S3b"):
        stems = sorted({s for _, ss, s, _ in own if ss == sid})
        snaps = [os.path.join(s + ".snap", f"{t}.pt") for k, ss, s, t in own if ss == sid]
        snaps = sorted(set(snaps))
        def stat_group(paths):
            ex = [p for p in paths if os.path.exists(p)]
            at = [os.stat(p).st_atime for p in ex]
            return len(ex), len(paths), sum(a >= today for a in at), (dt.datetime.fromtimestamp(max(at)).strftime("%m-%d %H:%M") if at else "-")
        rows.append((sid, stat_group(snaps), stat_group([s + ".json" for s in stems]),
                     stat_group([s + ".csv" for s in stems]), stat_group([s + ".heldout.json" for s in stems]),
                     sum(os.path.exists(s + ".ckpt.pt") for s in stems)))
    say("  existence and last access (os.stat; 'today' = accessed on 2026-10-11 local, by anyone):")
    say(f"    {'set':<4} {'snapshots exist/declared, today, latest':<42} {'json':<26} {'csv':<26} {'heldout.json':<26} ckpt.pt")
    for sid, a, b, c, e, k in rows:
        f = lambda t: f"{t[0]}/{t[1]}, {t[2]} today, {t[3]}"
        say(f"    {sid:<4} {f(a):<42} {f(b):<26} {f(c):<26} {f(e):<26} {k}")
    main_today = [(ss, os.path.basename(s), t) for k, ss, s, t in own if k == "main"
                  and os.path.exists(p := os.path.join(s + ".snap", f"{t}.pt")) and os.stat(p).st_atime >= today]
    traj_today = sorted({(ss, os.path.basename(s)[-4:], t) for k, ss, s, t in own if k == "traj"
                         and os.path.exists(p := os.path.join(s + ".snap", f"{t}.pt")) and os.stat(p).st_atime >= today})
    say(f"  declared MAIN checkpoints accessed today: {len(main_today)}; declared trajectory snapshots accessed today: "
        f"{len(traj_today)} -> {sorted({(a, c) for a, _, c in traj_today})} seeds {[b for _, b, _ in traj_today]}")
    for sid, d, arm in (("S2", "results/gate", "lagu_nogate-tt"), ("S1", "results/gate_literal", "lagu_nogate-lit")):
        steps = {}
        for p in glob.glob(os.path.join(REPO, d, f"{arm}_SafetyPointGoal1-v0_s1*.snap", "*.pt")):
            if os.stat(p).st_atime >= today:
                st = int(os.path.basename(p)[:-3])
                steps[st] = steps.get(st, 0) + 1
        say(f"  {arm}: every snapshot step read today (step: n runs), declared or not: {dict(sorted(steps.items()))}")
    # does any other analysis script compute action gradients (lambda*) or actor updates?
    roots = [os.path.join(REPO, "results", "paper"), SCR_ROOT]
    hits = []
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "*.py"), recursive=True):
            if "probe-author" in p or f"{os.sep}lagu{os.sep}" in p:   # skip this probe and copies of the lagu package
                continue
            txt = open(p, errors="ignore").read()
            if re.search(r"autograd\.grad|\.backward\(|requires_grad_\(True\)", txt):
                hits.append(os.path.relpath(p, r))
    say(f"  other analysis scripts under results/paper and the scratchpad (copies of the lagu package excluded) that compute gradients or actor updates: "
        f"{hits if hits else 'none'}")


def v4() -> dict:
    say("\n== V4 [DESCRIPTIVE] --dry-run as `python -m lagu.probe` from a scratch mirror, under a file-open audit hook")
    mirror = os.path.join(V, "mirror")
    shutil.rmtree(mirror, ignore_errors=True)
    os.makedirs(os.path.join(mirror, "lagu"))
    for f in ("__init__.py", "agent.py", "train.py"):
        os.symlink(os.path.join(REPO, "lagu", f), os.path.join(mirror, "lagu", f))
    shutil.copyfile(os.path.join(AUTH, "probe.py"), os.path.join(mirror, "lagu", "probe.py"))
    with open(os.path.join(V, "audit_run.py"), "w") as f:
        f.write(AUDIT_RUNNER)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    log = os.path.join(V, "audit_dryrun.json")
    out = sh([PY, os.path.join(V, "audit_run.py"), log, "lagu.probe", "--dry-run"], cwd=mirror, env=env)
    for l in out.splitlines():
        say("    | " + l)
    a = json.load(open(log))
    res_opens = sorted({p for p, _ in a["opened"] if "/safe-rl-carla/results" in p})
    say(f"  exit {a['exit']}; files opened: {len(a['opened'])}; under {REPO}/results: {res_opens if res_opens else 'none'}")
    m = dict(re.findall(r"^(\w+)\s+M=\d: lambda\*=([0-9.]+)", out, re.M))
    # own lambda* on the same random agents: same torch / numpy seeds and construction order as dry_run()
    import torch
    from lagu.agent import Config, LagU
    torch.manual_seed(0)
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    own = {}
    for arm, kw in (("lagu_nogate", dict(M=3, use_bonus=True, use_gate=False)),
                    ("td3lag", dict(M=1, use_bonus=False, use_gate=False, action_noise=0.1))):
        ag = LagU(60, 2, Config(hidden=256, **kw))
        S = rng.standard_normal((20_000, 60)).astype(np.float32)
        sub = np.sort(np.random.default_rng(0).choice(20_000, 5000, replace=False))
        own[arm] = np_lam_star(ag.actor.state_dict(), ag.members.state_dict(), ag.cfg.M, ag.cost.state_dict(),
                               S[sub].astype(np.float64), 2)
    say(f"  {'arm':<12} {'dry-run lambda*':>16} {'own float64 lambda*':>20} {'reported':>9}")
    for arm in own:
        say(f"  {arm:<12} {m.get(arm, 'nan'):>16} {own[arm]['lam_star']:>20.6f} {REPORTED['dry_lam_star'][arm]:>9}")
    return own


def v5() -> None:
    say("\n== V5 [DESCRIPTIVE] the author's make.py, scratch path redirected to this task's scratchpad")
    src = open(os.path.join(AUTH, "make.py")).read()
    old = '"47ca4e66-e2f4-4455-bfea-c56f9a36fb6d/scratchpad/paper/probe-author")'
    assert src.count(old) == 1
    os.makedirs(os.path.join(V, "author_make"), exist_ok=True)
    red = os.path.join(V, "author_make", "make_redirected.py")
    with open(red, "w") as f:
        f.write(src.replace(old, '"47ca4e66-e2f4-4455-bfea-c56f9a36fb6d/scratchpad/paper/probe-author-verify/author_make")'))
    for d in ("quick_stems", "quick_prereg_path"):        # force a fresh probe pass (models are reused if present)
        shutil.rmtree(os.path.join(V, "author_make", d), ignore_errors=True)
    t0 = time.time()
    out = sh([PY, red])
    with open(os.path.join(V, "author_make", "stdout.txt"), "w") as f:
        f.write(out)
    say(f"  ran in {time.time() - t0:.0f} s (one line differs from the original: the scratch path)")
    for l in out.splitlines():
        if l.startswith(("lagu_nogate ", "td3lag ", "  20-episode", "  upper bound", "OVERALL")):
            say("    | " + l.strip())
    mine, theirs = os.path.join(V, "author_make"), AUTH_SCR
    same = {}
    for p in sorted(glob.glob(os.path.join(theirs, "run", "*.snap", "*.pt"))):
        q = p.replace(theirs, mine)
        same[os.path.relpath(p, theirs)] = os.path.exists(q) and sha(p) == sha(q)
    say(f"  retrained scratch snapshots bit-identical to the author's: {all(same.values())} ({len(same)} files)")
    a, b = open(os.path.join(theirs, "quick_stems", "probe_tables.txt")).read(), open(os.path.join(mine, "quick_stems", "probe_tables.txt")).read()
    say(f"  quick_stems/probe_tables.txt identical to the author's: {a == b}")
    diffs = []
    for p in sorted(glob.glob(os.path.join(theirs, "quick_prereg_path", "jobs", "*", "*.json"))):
        q = p.replace(theirs, mine)
        x, y = json.load(open(p)), json.load(open(q))
        for k in ("parity", "R_ckpt", "R0", "grid", "lam_half", "censor", "log2_rho", "eligible", "lambda_train", "protocol_sha256", "probe_sha256"):
            if x.get(k) != y.get(k):
                diffs.append((os.path.basename(p), k))
    say(f"  quick_prereg_path job files: numeric fields identical to the author's: {not diffs} {diffs if diffs else ''}")


def v6() -> None:
    say("\n== V6 [DESCRIPTIVE] own re-derivation on the scratch models, --quick constants (plumbing only, not the study)")
    C = {"state_episodes": 2, "parity_states": 500, "eval_episodes": 2, "K": 200}
    grid = [0.01, 0.1, 1.0, 10.0]
    run = os.path.join(V, "author_make", "run")
    say(f"  {'checkpoint':<44} {'quantity':<10} {'probe':>12} {'own':>12} {'|diff|':>9}")
    for arm, step in (("td3lag", 20000), ("td3lag", "final"), ("lagu_nogate", 20000)):
        stem = os.path.join(run, f"{arm}-scr_SafetyPointGoal1-v0_s7")
        jf = os.path.join(V, "author_make", "quick_stems", "jobs", "main", f"{os.path.basename(stem)}@{step}.json")
        if not os.path.exists(jf):
            jf = os.path.join(V, "author_make", "quick_prereg_path", "jobs", "main", f"{os.path.basename(stem)}@{step}.json")
        pj = json.load(open(jf))
        o = own_probe(stem, step, C, grid)
        pairs = [("lambda*", pj["parity"]["lam_star"], o["parity"]["lam_star"]), ("cos", pj["parity"]["cos_mean"], o["parity"]["cos"]),
                 ("R_ckpt", pj["R_ckpt"]["return"], o["R_ckpt"]), ("R0", pj["R0"]["return"], o["R"][0.0])]
        pairs += [(f"R({g['lam']})", g["return"], o["R"][g["lam"]]) for g in pj["grid"]]
        for q, a, b in pairs:
            say(f"  {os.path.basename(stem) + '@' + str(step):<44} {q:<10} {a:>12.6g} {b:>12.6g} {abs(a - b):>9.1e}")
        if step == 20000 and arm == "td3lag":
            say(f"  snapshot keys: {o['snap_keys']} (no replay, target networks or optimizer state)")
        if arm == "lagu_nogate":
            o2 = own_probe(stem, step, C, grid, share_x=False)
            for g in pj["grid"]:
                say(f"  {'  same, heads read separate (s, a) copies':<44} {'R(' + str(g['lam']) + ')':<10} {g['return']:>12.6g} "
                    f"{o2['R'][g['lam']]:>12.6g} {abs(g['return'] - o2['R'][g['lam']]):>9.1e}")
            say("  (separate copies change only the float summation order of the action gradient: actor parameters differ by"
                " ~1e-7 after 200 steps, yet 2-episode returns move by up to several units on this untrained model)")
    import torch
    for arm in ("td3lag", "lagu_nogate"):
        stem = os.path.join(run, f"{arm}-scr_SafetyPointGoal1-v0_s7")
        s, f = torch.load(os.path.join(stem + ".snap", "20000.pt"), weights_only=True), torch.load(stem + ".pt", weights_only=True)
        eq = all(torch.equal(s[n][k], f[n][k]) for n in ("actor", "members", "cost") for k in s[n])
        say(f"  {arm}: snapshot 20000.pt and final .pt hold identical actor/members/cost tensors: {eq}")


def v7() -> None:
    say("\n== V7 [DESCRIPTIVE] `python -m lagu.probe --stems ... --quick --workers 2` (spawned workers)")
    mirror, run, out = os.path.join(V, "mirror"), os.path.join(V, "author_make", "run"), os.path.join(V, "spawn_out")
    shutil.rmtree(out, ignore_errors=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    sh([PY, "-m", "lagu.probe", "--quick", "--workers", "2", "--out", out, "--stems",
        os.path.join(run, "td3lag-scr_SafetyPointGoal1-v0_s7@20000"), os.path.join(run, "lagu_nogate-scr_SafetyPointGoal1-v0_s7@final")],
       cwd=mirror, env=env)
    js = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "jobs", "main", "*.json")))]
    say(f"  jobs written: {len(js)}; probe_sha256 recorded = mirrored lagu/probe.py = reported: "
        f"{all(j['probe_sha256'] == sha(os.path.join(mirror, 'lagu', 'probe.py')) == REPORTED['probe_sha256'] for j in js)}")
    for j in js:
        say(f"    {os.path.basename(j['stem'])}@{j['step']}: lambda* {j['parity']['lam_star']:.6g}")


def v8() -> None:
    say("\n== V8 [DESCRIPTIVE] --prereg end to end, FULL protocol, fake root with two scratch checkpoints under declared names")
    run, fr, out = os.path.join(V, "author_make", "run"), os.path.join(V, "fakeroot"), os.path.join(V, "fake_out")
    shutil.rmtree(fr, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(fr, "results", "gate_literal"))
    for src, dst in (("lagu_nogate", "lagu_nogate-lit"), ("td3lag", "td3lag-lit")):
        s = os.path.join(run, f"{src}-scr_SafetyPointGoal1-v0_s7")
        t = os.path.join(fr, "results", "gate_literal", f"{dst}_SafetyPointGoal1-v0_s100")
        shutil.copyfile(s + ".json", t + ".json")
        os.makedirs(t + ".snap")
        for st in (250_000, 500_000, 750_000, 1_000_000):
            shutil.copyfile(os.path.join(s + ".snap", "20000.pt"), os.path.join(t + ".snap", f"{st}.pt"))
    log = os.path.join(V, "audit_fakeroot.json")
    t0 = time.time()
    txt = sh([PY, os.path.join(V, "audit_run.py"), log, "lagu.probe", "--prereg", "--root", fr, "--out", out, "--workers", "1"],
             cwd=os.path.join(V, "mirror"), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    a = json.load(open(log))
    say(f"  ran in {time.time() - t0:.0f} s, exit {a['exit']}")
    for l in txt.splitlines():
        if l.startswith(("  ->", "OVERALL", "Table S", "  lagu_nogate-lit", "  td3lag-lit")) and "missing" not in l:
            say("    | " + l)
    pj_all = json.load(open(os.path.join(out, "probe.json")))["results"]
    say(f"    | (plus {sum(r.get('error') == 'missing' and r['kind'] == 'main' for r in pj_all)} main and "
        f"{sum(r.get('error') == 'missing' and r['kind'] == 'traj' for r in pj_all)} trajectory jobs reported 'missing'; exit status "
        f"{a['exit']} although every co-primary checkpoint is missing)")
    opened = sorted({p for p, _ in a["opened"] if p.startswith(fr)})
    kinds = sorted({re.sub(r"\.snap/\d+\.pt$", ".snap/<step>.pt", os.path.relpath(p, fr)) for p in opened})
    say(f"  files opened under the fake root: {len(opened)} -> {kinds}")
    say(f"  files opened under {REPO}/results: {sorted({p for p, _ in a['opened'] if '/safe-rl-carla/results' in p}) or 'none'}")
    C = {"state_episodes": 20, "parity_states": 5000, "eval_episodes": 20, "K": 10_000}
    say(f"  {'checkpoint':<46} {'quantity':<9} {'probe':>12} {'own':>12} {'|diff|':>9}")
    for dst in ("lagu_nogate-lit", "td3lag-lit"):
        stem = os.path.join(fr, "results", "gate_literal", f"{dst}_SafetyPointGoal1-v0_s100")
        pj = json.load(open(os.path.join(out, "jobs", "main", f"{os.path.basename(stem)}@1000000.json")))
        o = own_probe(stem, 1_000_000, C)
        for q, x, y in (("n_states", pj["n_states"], o["n_states"]), ("lambda*", pj["parity"]["lam_star"], o["parity"]["lam_star"]),
                        ("R_ckpt", pj["R_ckpt"]["return"], o["R_ckpt"]), ("R0", pj["R0"]["return"], o["R"][0.0])):
            say(f"  {os.path.basename(stem) + '@1M':<46} {q:<9} {x:>12.6g} {y:>12.6g} {abs(x - y):>9.1e}")
        elig = o["R_ckpt"] >= 10 and o["R"][0.0] >= 10 and o["R"][0.0] >= o["R_ckpt"] / 2
        say(f"  {'':<46} eligible: probe {pj['eligible']} ({pj['why']}), own rule {elig}")


def v9(probe) -> None:
    say("\n== V9 [DESCRIPTIVE] decision-rule functions against own re-implementations")
    rng = np.random.default_rng(12345)
    lams = probe.PROTOCOL["grid"][1:]
    n_h = n_stop = 0
    bad_h = bad_stop = 0
    for i in range(20_000):
        r0 = float(rng.uniform(10, 40))
        kind = i % 3
        if kind == 0:     # noisy decreasing
            mid = rng.uniform(-3.5, 2.5)
            rets = r0 / (1 + np.exp((np.log10(lams) - mid) * rng.uniform(0.5, 6))) + rng.normal(0, rng.uniform(0, 3), len(lams))
        elif kind == 1:   # random walk
            rets = r0 + np.cumsum(rng.normal(-1, 3, len(lams)))
        else:             # mostly flat, sometimes never crossing
            rets = r0 * rng.uniform(0.3, 1.2) + rng.normal(0, 1, len(lams))
        rets = [float(x) for x in rets]
        full_p, full_m = probe.halving_point(lams, rets, r0), my_halving(lams, rets, r0)
        n_h += 1
        bad_h += not (full_p[1] == full_m[1] and abs(full_p[0] - full_m[0]) <= 1e-9 * max(1, full_m[0]))
        trunc = my_stop(rets, r0)
        n_stop += 1
        bad_stop += probe.halving_point(lams, trunc, r0) != full_p
    say(f"  halving_point vs own (20,000 random curves on the 21-point grid): {bad_h} mismatches")
    say(f"  stop rule (two consecutive R <= R0/4) never changes lambda_half (same 20,000 curves): {bad_stop} changes")
    bad = 0
    for _ in range(2000):
        lh, ls = 10 ** rng.uniform(-3, 2), 10 ** rng.uniform(-3, 1)
        c = ["", "left", "right"][int(rng.integers(3))]
        want = math.log2(({"left": 0.0005, "right": 200.0}.get(c, lh)) / ls) if c else math.log2(lh / ls)
        got = probe.log2_rho({"left": 0.001, "right": 100.0}.get(c, lh), c, ls)
        bad += abs(got - want) > 1e-9
    say(f"  log2_rho with censoring imputed at 0.0005 / 200 (2,000 cases): {bad} mismatches")
    bad = n = 0
    labels = {}
    for i in range(300):
        k = int(rng.integers(0, 16))
        x = rng.normal(rng.uniform(-3, 3), rng.uniform(0.1, 2.5), k)
        v = probe.verdict(x, probe.PROTOCOL)
        lab, med, ci = my_verdict(x)
        labels[lab] = labels.get(lab, 0) + 1
        n += 1
        same = v["label"] == lab and (k < 5 or (abs(v["median_rho"] - med) < 1e-9 and np.allclose(v["ci95_rho"], ci, rtol=1e-9)))
        bad += not same
    say(f"  verdict vs own bootstrap (300 random sets, n 0-15): {bad} mismatches; labels seen {labels}")
    bad = 0
    for _ in range(500):
        k = int(rng.integers(3, 15))
        x, y = rng.integers(0, 6, k).astype(float), rng.normal(size=k)
        a, b = probe.spearman(x, y), my_spearman(x, y)
        bad += not ((math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9)
    say(f"  spearman (average ranks for ties) vs own (500 cases): {bad} mismatches")
    xs = [1.0, 2.0, 8.0, 9.0]
    say(f"  even-n example rho = {xs}: code reports 2**median(log2 rho) = {2 ** np.median(np.log2(xs)):.3f}; "
        f"arithmetic median of rho = {np.median(xs):.3f} (the docstring says median of log2 rho; the summary says median of rho)")
    # overall label for every pair of per-set labels, through summarise() on synthetic job records
    proto = {"SUPPORTED": [0.1] * 10, "REFUTED": [3.0] * 10, "INCONCLUSIVE": [0.0] * 5 + [3.0] * 5, "NOT EVALUABLE": [0.0] * 3}

    def recs(sid, xs):
        return [{"kind": "main", "set": sid, "stem": f"x/{sid}_{i}", "step": 1, "lambda_train": 0.1, "eligible": True, "why": "",
                 "censor": "", "lam_half": 0.1 * 2 ** x, "log2_rho": x, "R_ckpt": {"return": 20.0}, "R0": {"return": 20.0},
                 "grid": [], "parity": {"lam_star": 0.1, "g_r_mean": 1, "g_c_mean": 1, "cos_mean": 0, "ratio_median": 1, "sat_frac": 0}}
                for i, x in enumerate(xs)]
    sets = [("S1", "co-primary"), ("S2", "co-primary")]
    say("  overall label from summarise() for each (S1, S2) pair of per-set labels:")
    corner = "S1 \\ S2"
    say(f"    {corner:<14} " + " ".join(f"{k:>14}" for k in proto))
    for a in proto:
        row = []
        for b in proto:
            summ, _ = probe.summarise(recs("S1", proto[a]) + recs("S2", proto[b]), probe.PROTOCOL, sets, "synthetic")
            assert summ["sets"]["S1"]["label"] == a and summ["sets"]["S2"]["label"] == b
            row.append(summ["overall"])
        say(f"    {a:<14} " + " ".join(f"{r:>14}" for r in row))


def v10(probe) -> None:
    say("\n== V10 [DESCRIPTIVE] own timing and the upper bound of the --prereg run")
    import safety_gymnasium
    import torch
    from lagu.agent import Config, LagU
    torch.set_num_threads(1)
    torch.manual_seed(1)
    ag = LagU(60, 2, Config(hidden=256))
    for net in (ag.members, ag.cost):
        net.requires_grad_(False)
    S = torch.randn(20_000, 60)
    actor = __import__("copy").deepcopy(ag.actor)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    idx = np.random.default_rng(0).integers(0, 20_000, size=(400, 256))
    t0 = time.time()
    for k in range(400):
        o = S[idx[k]]
        a = actor(o)
        x = torch.cat([o, a], -1)
        loss = (-torch.stack([m.q1(x)[:, 0] for m in ag.members]).mean(0) + 0.1 * ag.cost(x)[:, 0]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    t_upd = (time.time() - t0) / 400
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    t0 = time.time()
    for k in range(20):
        o, _ = env.reset(seed=12_000 + k)
        done = False
        while not done:
            with torch.no_grad():
                a = ag.actor(torch.as_tensor(o, dtype=torch.float32)[None])[0].numpy()
            o, _, _, te, tr, _ = env.step(a)
            done = te or tr
    t_roll = time.time() - t0
    K, G = probe.PROTOCOL["K"], len(probe.PROTOCOL["grid"])
    per_lam = K * t_upd + t_roll
    n_main, n_traj = REPORTED["n_main"], REPORTED["n_traj"]
    as_coded = (n_main * (t_roll + 2 * t_roll + G * per_lam) + n_traj * t_roll) / 3600
    corrected = (n_main * (t_roll + t_roll + G * per_lam) + n_traj * t_roll) / 3600
    say(f"  hidden-256, M=3 actor update {t_upd * 1e3:.2f} ms; 20-episode rollout of a hidden-256 actor {t_roll:.1f} s; "
        f"per lambda {per_lam:.0f} s")
    say(f"  upper bound: author's formula (2 evaluations + 22 lambdas per checkpoint) {as_coded:.1f} CPU-h; with the count "
        f"corrected (R_ckpt once; R0 is the first of the 22 grid values) {corrected:.1f} CPU-h; reported {REPORTED['upper_cpu_h']}")


def main() -> None:
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    os.makedirs(V, exist_ok=True)
    say("DESCRIPTIVE verification of results/paper/probe-author (no study snapshot, CSV, heldout.json or study .json is read)")
    say(f"run {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    probe = load_probe()
    v1()
    v2(probe)
    v3(probe)
    v4()
    v5()
    v6()
    v7()
    v8()
    v9(probe)
    v10(probe)
    with open(os.path.join(HERE, "verify_tables.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")
    say(f"\nwrote {os.path.join(HERE, 'verify_tables.txt')}")


if __name__ == "__main__":
    main()
