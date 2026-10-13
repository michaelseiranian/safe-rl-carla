"""DESCRIPTIVE. Independent verification of results/paper/calibration_feasibility (cost-critic
calibration, feasibility counts, E0 health rule (a)-(b) on the E1 lagu_nogate-tt seeds).

Nothing here is confirmatory and no hypothesis is tested. This script does NOT import lagu.analyze
or the verified make.py: it parses the raw <stem>.csv, <stem>.json and <stem>.heldout.json files
itself, recomputes every number in the verified tables, and compares them with the verified
per_run.csv and tables.md. It then runs a copy of the verified make.py in the scratch directory
(outputs written there, not into results/) and checks that the copy reproduces the verified
tables.md and per_run.csv byte for byte, and the PNG pixel for pixel.

Run from the repository root:

    .venv/bin/python results/paper/calibration-feasibility-verify/make.py

Pre-specified (from the task brief, nothing chosen after looking at outcomes):
  calibration  = sum(qc_traj) / sum(mc_cost_traj), held-out: over the 5 per_snapshot entries of
                 each run and over all seeds of the arm; 95% percentile bootstrap over seeds,
                 10,000 resamples, numpy.random.default_rng(0) (fresh per row).
  in-training  = the single evaluation row at step 1,000,000 and the run's final row (2M), E1 arms.
  feasibility  = held-out pooled return/cost; cost <= 25; return >= 15 and cost <= 25;
                 collapse = return < 5 (or diverged, prereg 11), split by cost <= 25 / > 25.
  E0 (a)-(b)   = prereg 10.2 on rows 850k, 900k, 950k, 1M of lagu_nogate-tt seeds 100-114, blocks
                 100-102, 103-105, 106-108, 109-111, 112-114.
All listed seeds are used and no run is excluded; the 'without s103' value is the one extra view the
brief asks for (lagu_nogate-tt PointGoal1 only).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = "/Users/michaelseiranian/Desktop/safe-rl-carla"
VER = os.path.join(ROOT, "results/paper/calibration_feasibility")
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = ("/private/tmp/claude-501/-Users-michaelseiranian-Desktop-safe-rl-carla/"
           "47ca4e66-e2f4-4455-bfea-c56f9a36fb6d/scratchpad/paper/calibration-feasibility-verify")
PY = os.path.join(ROOT, ".venv/bin/python")
B, SEED = 10_000, 0
PG, CG, PC = "SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyPointCircle1-v0"
DIVERGE = ("eval_return", "eval_cost", "qc_traj", "qbar_traj", "q_loss", "c_loss", "lambda")
R = lambda a, b: list(range(a, b + 1))
SPEC = [  # (dir, env, label, seeds) -- the task's list
    ("gate", PG, "lagu-tt", R(100, 119)), ("gate", PG, "lagu_nogate-tt", R(100, 114)),
    ("gate", PG, "lagu_nogate-tt-match", R(100, 119)), ("gate", PG, "lagu-tt-ggrad", R(100, 109)),
    ("gate", PG, "td3lag-tt", R(100, 109)), ("gate", PG, "lagu_nogate-tt-d175", R(100, 107)),
    ("gate", PG, "lagu_nogate-tt-d125", R(100, 107)), ("gate", PG, "lagu-tt-T035", R(100, 107)),
    ("gate", PG, "lagu-tt-T14", R(100, 107)),
    ("gate_literal", PG, "lagu-lit", R(100, 109)), ("gate_literal", PG, "lagu_nogate-lit", R(100, 109)),
    ("gate_literal", PG, "td3lag-lit", R(100, 109)),
    ("gate_literal_x", PG, "lagu-lit-T03", R(100, 109)), ("gate_literal_x", PG, "lagu_nogate-lit-match", R(100, 109)),
    ("gate_x", CG, "lagu-tt", R(100, 109)), ("gate_x", CG, "lagu_nogate-tt", R(100, 109)), ("gate_x", CG, "td3lag-tt", R(100, 104)),
    ("gate_x", PC, "lagu-tt", R(100, 109)), ("gate_x", PC, "lagu_nogate-tt", R(100, 109)), ("gate_x", PC, "td3lag-tt", R(100, 104)),
]
E1 = R(100, 114)
FAILS: list[str] = []


def flt(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def check(ok: bool, what: str) -> None:
    if not ok:
        FAILS.append(what)
        print("  MISMATCH:", what)


# ---------------------------------------------------------------- raw loading (own code)

def load_dir(d: str) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", d, "*.csv"))):
        stem = p[:-4]
        if not os.path.exists(stem + ".json"):
            continue
        args = json.load(open(stem + ".json"))["args"]
        lab = args["arm"] + (f"-{args['tag']}" if args.get("tag") else "")
        with open(p) as fh:
            rows = [{k: flt(v) for k, v in r.items()} for r in csv.DictReader(fh)]
        held = json.load(open(stem + ".heldout.json")) if os.path.exists(stem + ".heldout.json") else None
        key = (d, args["env"], lab, int(args["seed"]))
        assert key not in out, key
        out[key] = {"rows": rows, "args": args, "held": held, "stem": os.path.basename(stem)}
    return out


RUNS = {}
for d in ("gate", "gate_literal", "gate_literal_x", "gate_x"):
    RUNS.update(load_dir(d))

print("=" * 100)
print("DESCRIPTIVE verification of results/paper/calibration_feasibility (independent re-derivation)")
print("=" * 100)

# coverage: exactly the listed arms and seeds, nothing extra
have = {}
for (d, env, lab, s) in RUNS:
    have.setdefault((d, env, lab), []).append(s)
spec_keys = {(d, e, l) for d, e, l, _ in SPEC}
for d, env, lab, seeds in SPEC:
    check(sorted(have.get((d, env, lab), [])) == seeds, f"coverage {d} {env} {lab}")
check(set(have) == spec_keys, f"unlisted arms present: {sorted(set(have) - spec_keys)}")
print(f"coverage: {len(RUNS)} runs, {len(SPEC)} arms; arm/seed sets equal the task list: "
      f"{all(sorted(have.get((d, e, l), [])) == s for d, e, l, s in SPEC) and set(have) == spec_keys}")


def per_run(run) -> dict:
    h, a, rows = run["held"], run["args"], run["rows"]
    N = max(int(a["steps"]), int(rows[-1]["step"]))
    snaps = h["per_snapshot"]
    qc = np.array([flt(s["qc_traj"]) for s in snaps])
    mc = np.array([flt(s["mc_cost_traj"]) for s in snaps])
    ret_s = np.array([flt(s["eval_return"]) for s in snaps])
    cost_s = np.array([flt(s["eval_cost"]) for s in snaps])
    lo = int(a.get("start_steps") or 0)
    div_csv = any(k in r and not np.isfinite(r[k]) for r in rows if lo < r["step"] <= N for k in DIVERGE)
    ret, cost = flt(h["pooled"]["eval_return"]), flt(h["pooled"]["eval_cost"])
    div = div_csv or not (np.isfinite(ret) and np.isfinite(cost))
    return {"N": N, "end_ok": int(h["end_step"]) == N and int(rows[-1]["step"]) == N,
            "n_snap": len(snaps), "snap_steps": [int(s["step"]) for s in snaps],
            "snap_seed_base": [int(s["seed_base"]) for s in snaps],
            "episodes": int(h["episodes"]),
            "qc": float(qc.sum()), "mc": float(mc.sum()), "qc_finite": bool(np.all(np.isfinite(qc)) and np.all(np.isfinite(mc))),
            "calib": float(qc.sum() / mc.sum()) if mc.sum() != 0 else math.nan,
            "ret": ret, "cost": cost, "ret_snapmean": float(ret_s.mean()), "cost_snapmean": float(cost_s.mean()),
            "div": div, "resumed": any(r.get("resumed") == 1 for r in rows)}


P = {k: per_run(r) for k, r in RUNS.items()}
# integrity of the held-out files
print("held-out files: all at final step:", all(p["end_ok"] for p in P.values()),
      "| 5 snapshots each:", all(p["n_snap"] == 5 for p in P.values()),
      "| 20 episodes:", all(p["episodes"] == 20 for p in P.values()),
      "| finite qc/mc:", all(p["qc_finite"] for p in P.values()),
      "| diverged runs:", sum(p["div"] for p in P.values()))
check(all(p["end_ok"] and p["n_snap"] == 5 and p["episodes"] == 20 for p in P.values()), "held-out integrity")
bad_snap = [k for k, p in P.items() if p["snap_steps"] != [round(p["N"] * f) for f in (0.8, 0.85, 0.9, 0.95, 1.0)]
            or p["snap_seed_base"] != [10000 + 1000 + 20 * i for i in range(5)]]
print("snapshot steps = N x {0.8,...,1.0} and seed bases = 11000 + 20i for every run:", not bad_snap, bad_snap[:3])
gap = max(max(abs(p["ret"] - p["ret_snapmean"]), abs(p["cost"] - p["cost_snapmean"])) for p in P.values())
print(f"pooled held-out return/cost equal the mean of the 5 snapshot means (max abs diff {gap:.2e})")


# ---------------------------------------------------------------- statistics (own code)

def boot_ratio(qc, mc):
    qc, mc = np.asarray(qc, float), np.asarray(mc, float)
    r = qc.sum() / mc.sum()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(qc), size=(B, len(qc)))
    with np.errstate(divide="ignore", invalid="ignore"):
        b = qc[idx].sum(1) / mc[idx].sum(1)
    b = b[np.isfinite(b)]
    return float(r), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def g(x):
    if not np.isfinite(x):
        return "n/a"
    a = abs(x)
    return f"{x:.1f}" if a >= 100 else f"{x:.2f}" if a >= 10 else f"{x:.3f}"


def gci(t):
    return f"{g(t[0])} [{g(t[1])}, {g(t[2])}]"


def show(title, header, rows):
    print()
    print(title)
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    print("  " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


# ---------------------------------------------------------------- verified outputs, for comparison

ver_csv = {}
with open(os.path.join(VER, "per_run.csv")) as fh:
    for r in csv.DictReader(fh):
        d = r["dir"].replace("results/", "")
        ver_csv[(d, r["env"], r["arm"], int(r["seed"]))] = r
check(set(ver_csv) == set(P), "per_run.csv run set")
mx = 0.0
for k, p in P.items():
    v = ver_csv[k]
    for mine, col in ((p["ret"], "heldout_return"), (p["cost"], "heldout_cost"), (p["qc"], "sum_qc_traj"),
                      (p["mc"], "sum_mc_cost_traj"), (p["calib"], "calibration")):
        vv = flt(v[col])
        if np.isfinite(mine) or np.isfinite(vv):
            mx = max(mx, abs(mine - vv) / max(1.0, abs(mine)))
    check(int(v["collapsed"]) == int(p["div"] or p["ret"] < 5), f"collapsed flag {k}")
    check(int(v["resumed"]) == int(p["resumed"]), f"resumed flag {k}")
print(f"per_run.csv vs own recomputation: {len(P)} runs, max relative difference over return, cost, "
      f"Σqc, Σmc, calibration = {mx:.1e}")
check(mx < 1e-9, "per_run.csv values")

ver_md = open(os.path.join(VER, "tables.md")).read()


def md_rows(table_prefix):
    sec = ver_md.split(table_prefix, 1)[1].split("###", 1)[0]
    return [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in sec.splitlines()
            if ln.startswith("|") and not ln.startswith("|---")][1:]


T1v, T1bv, T2v, T3v = (md_rows(f"### Table {t}.") for t in ("1", "1b", "2", "3"))

# ---------------------------------------------------------------- (1) held-out calibration

short = lambda e: e.replace("Safety", "").replace("-v0", "")
rows, i_v = [], 0
for d, env, lab, seeds in SPEC:
    variants = [("all listed", seeds)] + ([("E1 seeds only", E1)] if (d, env, lab) == ("gate", PG, "lagu-tt") else [])
    for name, ss in variants:
        E = [P[(d, env, lab, s)] for s in ss]
        pooled = boot_ratio([e["qc"] for e in E], [e["mc"] for e in E])
        wo = "-"
        if (d, env, lab) == ("gate", PG, "lagu_nogate-tt"):
            E2 = [P[(d, env, lab, s)] for s in ss if s != 103]
            wo = gci(boot_ratio([e["qc"] for e in E2], [e["mc"] for e in E2]))
        c = np.array([e["calib"] for e in E])
        med = float(np.median(c[np.isfinite(c)]))
        nund = int((~np.isfinite(c)).sum())
        nneg = sum(e["qc"] < 0 for e in E)
        rng_s = f"{g(np.nanmin(c))} to {g(np.nanmax(c))}"
        mine = [gci(pooled), wo, g(med) + (f" ({nund} undefined)" if nund else ""), rng_s, str(nneg)]
        v = T1v[i_v]
        i_v += 1
        same = mine == [v[5], v[6], v[7], v[8], v[9]] and v[1] == f"`{lab}`" and v[4] == str(len(ss))
        check(same, f"Table 1 row {short(env)} {lab} {name}: mine {mine} vs {v[5:10]}")
        rows.append([short(env), d, lab, name, len(ss), *mine, "match" if same else "DIFFERS"])
show("TABLE 1 (DESCRIPTIVE). Held-out calibration Σqc_traj/Σmc_cost_traj, own recomputation vs verified table",
     ["env", "dir", "arm", "seeds", "n", "pooled [95% CI]", "without s103", "median per-run", "range", "Σqc<0", "vs verified"],
     rows)

# ---------------------------------------------------------------- (1b) in-training rows at 1M and final

rows, i_v = [], 0
for lab in ("lagu-tt", "lagu_nogate-tt"):
    for name, step in (("1M", 1_000_000), ("final (2M)", None)):
        qc, mc, per, sel_steps = [], [], [], []
        for s in E1:
            rr = RUNS[("gate", PG, lab, s)]["rows"]
            pick = [r for r in rr if r["step"] == step] if step else [rr[-1]]
            assert len(pick) == 1
            sel_steps.append(int(pick[0]["step"]))
            qc.append(pick[0]["qc_traj"]), mc.append(pick[0]["mc_cost_traj"])
            per.append(pick[0]["qc_traj"] / pick[0]["mc_cost_traj"] if pick[0]["mc_cost_traj"] != 0 else math.nan)
        assert step or set(sel_steps) == {2_000_000}
        per = np.array(per)
        wo = "-"
        if lab == "lagu_nogate-tt":
            keep = [i for i, s in enumerate(E1) if s != 103]
            wo = gci(boot_ratio([qc[i] for i in keep], [mc[i] for i in keep]))
        nund = int((~np.isfinite(per)).sum())
        mine = [gci(boot_ratio(qc, mc)), wo, g(float(np.median(per[np.isfinite(per)]))) + (f" ({nund} undefined)" if nund else ""),
                f"{g(np.nanmin(per))} to {g(np.nanmax(per))}", str(sum(q < 0 for q in qc))]
        v = T1bv[i_v]
        i_v += 1
        same = mine == v[4:9]
        check(same, f"Table 1b {lab} {name}: mine {mine} vs {v[4:9]}")
        rows.append([lab, name, *mine, "match" if same else "DIFFERS"])
show("TABLE 1b (DESCRIPTIVE). In-training calibration, E1 arms, seeds 100-114",
     ["arm", "row", "pooled [95% CI]", "without s103", "median per-run", "range", "qc<0", "vs verified"], rows)

# ---------------------------------------------------------------- (2) feasibility

rows, i_v = [], 0
for d, env, lab, seeds in SPEC:
    variants = [("all listed", seeds)] + ([("E1 seeds only", E1)] if (d, env, lab) == ("gate", PG, "lagu-tt") else [])
    for name, ss in variants:
        E = [P[(d, env, lab, s)] for s in ss]
        ret = np.array([e["ret"] for e in E]); cost = np.array([e["cost"] for e in E])
        col = np.array([e["div"] or e["ret"] < 5 for e in E])
        mine = [str(len(ss)), str(int((cost <= 25).sum())), str(int(((ret >= 15) & (cost <= 25)).sum())),
                str(int(col.sum())), str(int((col & (cost <= 25)).sum())), str(int((col & (cost > 25)).sum()))]
        v = T2v[i_v]
        i_v += 1
        same = mine == v[4:10]
        check(same, f"Table 2 {short(env)} {lab} {name}: mine {mine} vs {v[4:10]}")
        rows.append([short(env), d, lab, name, *mine, "match" if same else "DIFFERS"])
show("TABLE 2 (DESCRIPTIVE). Feasibility counts on the held-out endpoint",
     ["env", "dir", "arm", "seeds", "n", "cost<=25", "ret>=15&cost<=25", "collapsed", "safe stalls", "collapsed cost>25", "vs verified"],
     rows)

# ---------------------------------------------------------------- (3) E0 rule (a)-(b)

def e0_ab(runs: list[dict]) -> dict:
    """prereg 10.2 (a) and (b) on rows at 850k, 900k, 950k, 1M (selected by step)."""
    W = []
    for run in runs:
        w = [r for r in run["rows"] if r["step"] in (850_000, 900_000, 950_000, 1_000_000)]
        assert [int(r["step"]) for r in w] == [850_000, 900_000, 950_000, 1_000_000], run["stem"]
        W.append(w)
    allr = [r for w in W for r in w]
    qc = np.array([r["qc_traj"] for r in allr]); mc = np.array([r["mc_cost_traj"] for r in allr])
    ret = np.array([r["eval_return"] for r in allr])
    pooled = qc.sum() / mc.sum()
    per = [sum(r["qc_traj"] for r in w) / sum(r["mc_cost_traj"] for r in w) for w in W]
    neg = [int(run["args"]["seed"]) for run, w in zip(runs, W) if any(r["qc_traj"] < 0 for r in w)]
    div = []
    for run in runs:
        lo = int(run["args"].get("start_steps") or 0)
        if any(k in r and not np.isfinite(r[k]) for r in run["rows"] if lo < r["step"] <= 1_000_000 for k in DIVERGE):
            div.append(int(run["args"]["seed"]))
    rmean = float(ret.mean()); rseed = [float(np.mean([r["eval_return"] for r in w])) for w in W]
    a = bool(np.all(np.isfinite(qc)) and np.all(np.isfinite(mc)) and 0.65 <= pooled <= 1.35
             and all(x >= 0.5 for x in per) and not neg)
    b = bool(np.all(np.isfinite(ret)) and not div and rmean >= 20 and all(x >= 10 for x in rseed))
    return {"pooled": float(pooled), "per": per, "neg": neg, "a": a, "ret": rmean, "rseed": rseed, "div": div, "b": b,
            "nrows": len(allr)}


e0runs = load_dir("e0_dual")
ref = e0_ab([e0runs[("e0_dual", PG, "lagu_nogate-tt-l2e-6", s)] for s in (0, 1, 2)])
dec = json.load(open(os.path.join(ROOT, "results/e0_dual/e0_decision.json")))["cells"]["lagu_nogate-tt-l2e-6"]
print()
print(f"E0 reference cell recomputed: calib {ref['pooled']:.6f} vs e0_decision.json {dec['calib']:.6f}; "
      f"(a) {ref['a']} vs {dec['a']}; (b) {ref['b']} vs {dec['b']}; return {ref['ret']:.4f} vs {dec['return']:.4f}")
check(abs(ref["pooled"] - dec["calib"]) < 1e-9 and ref["a"] == dec["a"] and ref["b"] == dec["b"], "E0 reference vs e0_decision.json")
blocks = [("reference l2e-6 s0-2", None)] + [(f"block {s}-{s + 2}", R(s, s + 2)) for s in range(100, 115, 3)] + [("all 15", E1)]
rows = []
for i, (name, ss) in enumerate(blocks):
    c = ref if ss is None else e0_ab([RUNS[("gate", PG, "lagu_nogate-tt", s)] for s in ss])
    mine = [g(c["pooled"]), ", ".join(g(x) for x in c["per"]), ", ".join(map(str, c["neg"])) or "none",
            "pass" if c["a"] else "FAIL", g(c["ret"]), ", ".join(f"{x:.1f}" for x in c["rseed"]),
            ", ".join(map(str, c["div"])) or "none", "pass" if c["b"] else "FAIL"]
    v = T3v[i]
    same = mine == v[3:11] and v[2] == str(c["nrows"])
    check(same, f"Table 3 {name}: mine {mine} vs {v[3:11]}")
    why_a = []
    if not (0.65 <= c["pooled"] <= 1.35): why_a.append("pooled out of band")
    if any(x < 0.5 for x in c["per"]): why_a.append("a seed ratio < 0.5")
    if c["neg"]: why_a.append("row qc<0")
    why_b = []
    if c["ret"] < 20: why_b.append("mean < 20")
    if any(x < 10 for x in c["rseed"]): why_b.append("a seed < 10")
    rows.append([name, c["nrows"], *mine, "; ".join(why_a) or "-", "; ".join(why_b) or "-",
                 "healthy" if c["a"] and c["b"] else "not healthy", "match" if same else "DIFFERS"])
show("TABLE 3 (DESCRIPTIVE). E0 rule (a)-(b), lagu_nogate-tt PointGoal1, rows 850k-1M",
     ["cell", "rows", "(a) pooled", "(a) per seed", "qc<0 seeds", "(a)", "(b) mean ret", "(b) per seed", "div", "(b)",
      "(a) fails because", "(b) fails because", "healthy", "vs verified"], rows)

# ---------------------------------------------------------------- claims in the summary

print()
print("CLAIM CHECKS (summary statements)")
med_tt = {}
for d, env, lab, seeds in SPEC:
    if "lit" in lab:
        continue
    c = np.array([P[(d, env, lab, s)]["calib"] for s in seeds])
    med_tt[f"{short(env)} {lab}"] = float(np.median(c[np.isfinite(c)]))
outside = {k: round(v, 3) for k, v in med_tt.items() if not 0.45 <= v <= 0.69}
print(f"  'Medians [of the 2M tt arms] stay around 0.45-0.69': medians outside [0.45, 0.69]: {outside}")
lit = [boot_ratio([P[(d, e, l, s)]["qc"] for s in ss], [P[(d, e, l, s)]["mc"] for s in ss])[0]
       for d, e, l, ss in SPEC if "lit" in l]
print(f"  literal-recipe arms pooled calibration range: {min(lit):.3f}-{max(lit):.3f}")
fin = [(e, l) for (d, e, l, s), p in P.items() if e == PG and p["ret"] >= 15 and p["cost"] <= 25]
print(f"  PointGoal1 runs with held-out return >= 15 and cost <= 25 (any arm): {len(fin)}")
print("  safe stalls / collapses by arm (gated vs ungated):")
for d, env, lab, seeds in SPEC:
    E = [P[(d, env, lab, s)] for s in seeds]
    col = [e["div"] or e["ret"] < 5 for e in E]
    ss = sum(c and e["cost"] <= 25 for c, e in zip(col, E))
    if sum(col):
        print(f"    {short(env):12s} {lab:24s} safe stalls {ss} of {sum(col)} collapses (n={len(E)})")
x1 = [p["cost"] for (d, e, l, s), p in P.items() if d == "gate_literal_x"]
print(f"  X1 min held-out cost: {min(x1):.2f} (no run <= 25: {min(x1) > 25})")
neg2m = sum(RUNS[("gate", PG, "lagu-tt", s)]["rows"][-1]["qc_traj"] < 0 for s in E1)
print(f"  lagu-tt E1 seeds with qc_traj < 0 in their final (2M) in-training row: {neg2m} of 15")
# the same policies (2M) on the held-out layouts: last per_snapshot entry (index 4, step 2M)
last = [RUNS[("gate", PG, "lagu-tt", s)]["held"]["per_snapshot"][4] for s in E1]
qL, mL = np.array([x["qc_traj"] for x in last]), np.array([x["mc_cost_traj"] for x in last])
perL = np.where(mL != 0, qL / np.where(mL != 0, mL, 1), np.nan)
print(f"  same lagu-tt policies at 2M on held-out layouts (snapshot 5): pooled {qL.sum() / mL.sum():.3f}, "
      f"median per-run {np.nanmedian(perL):.3f}, runs with qc_traj < 0: {int((qL < 0).sum())} of 15 "
      "(claim 'critic falls toward and below zero by 2M' rests on the single in-training row)")
s103 = P[("gate", PG, "lagu_nogate-tt", 103)]
print(f"  lagu_nogate-tt s103 held-out per-run calibration {s103['calib']:.1f}, diverged (prereg 11) = {s103['div']}")

# ---------------------------------------------------------------- run a copy of the verified make.py

print()
print("REPRODUCTION of the verified make.py (copy run in scratch; results/ untouched)")
os.makedirs(SCRATCH, exist_ok=True)
rep = os.path.join(SCRATCH, "rerun")
shutil.rmtree(rep, ignore_errors=True)
os.makedirs(rep)
src = open(os.path.join(VER, "make.py")).read()
old = 'ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))'
assert src.count(old) == 1
open(os.path.join(rep, "make.py"), "w").write(src.replace(old, f'ROOT = "{ROOT}"'))
p = subprocess.run([PY, os.path.join(rep, "make.py")], cwd=ROOT, capture_output=True, text=True)
print(f"  exit code {p.returncode}; stderr: {p.stderr.strip()[-300:] or 'empty'}")
check(p.returncode == 0, "verified make.py copy runs")
for fn in ("tables.md", "per_run.csv"):
    same = open(os.path.join(rep, fn), "rb").read() == open(os.path.join(VER, fn), "rb").read()
    print(f"  {fn}: byte-identical to the verified file: {same}")
    check(same, f"rerun {fn} identical")
import matplotlib.image as mpimg  # noqa: E402
a, b = mpimg.imread(os.path.join(rep, "calibration_dots.png")), mpimg.imread(os.path.join(VER, "calibration_dots.png"))
same = a.shape == b.shape and bool(np.array_equal(a, b))
print(f"  calibration_dots.png: pixel-identical {same} (shape {a.shape})")
check(same, "rerun png identical")
printed = "\n".join(p.stdout.rstrip("\n").splitlines()[:-1]).rstrip()   # drop the final 'wrote ...' line
print(f"  printed tables equal tables.md: {printed == open(os.path.join(VER, 'tables.md')).read().rstrip()}")

print()
print(f"RESULT: {len(FAILS)} mismatches" + (": " + "; ".join(FAILS) if FAILS else " - every recomputed number equals the verified tables"))
