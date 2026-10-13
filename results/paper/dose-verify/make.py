"""DESCRIPTIVE / EXPLORATORY verification of results/paper/gate_dose (the gate's actual dose).
Never confirmatory; outside the Holm family (prereg section 16 item 5).

Run from the repository root:

    .venv/bin/python results/paper/dose-verify/make.py [--fresh] [--skip-author]

Independent re-derivation (own code; the author's make.py is NOT imported):

A. (DESCRIPTIVE) Part 1 from the raw <stem>.csv/.json files: per arm and window (rows with
   lo < step <= hi), engagement = sum(engagement*n_risk)/sum(n_risk), budget(all) =
   sum(delta_avg*n_pi)/sum(n_pi)/delta0 (rows with n_pi > 0), budget(risk) =
   sum(delta_risk_avg*n_risk)/sum(n_risk)/delta0, pooled over seeds; 95% percentile bootstrap over
   seeds (10,000 resamples, numpy default_rng(0)). Row-grid / delta0 / T / use_gate checks, values
   already recorded in prereg.md, and a band-membership table (0.76-0.88, as we read Zhang et al.
   Fig. 5(d)). Compared with the author's part1_dose_windows.csv.
B. (EXPLORATORY) Part 2 with own rollouts: every ungated snapshot (lagu_nogate-tt s100-114 at
   600k/1.2M/2M, lagu_nogate-lit s100-109 at 600k/1M) plays deterministic episodes on layouts
   11000-11004; Q_bar and sqrt(Q_eu) are computed here from the members' first heads (Eq. 10-11).
   |Q_bar| gate: sd/(|Q|+1e-6) >= T; signed gate (Eq. 16 as written): Q+1e-6 > 0 and
   sd/(Q+1e-6) >= T; budget = T/ratio where engaged, else 1. Compared state-by-state with the
   author's cache and seed-by-seed with part2_per_seed.csv.
C. (EXPLORATORY) Extra diagnostics for claims in the author's summary: where the |Q_bar| gate
   fires (Q_bar <= 0 vs > 0), the signed budget on Q_bar > 0 states only, the per-seed spread of
   the signed budget, near-never seeds, and the logged sign of qbar_avg in the gated E1 runs.
D. Re-runs a byte-identical copy of the author's make.py in a scratch tree whose results/ and
   lagu/ are symlinks to the repository (outputs copied to ./rerun_author/), once with a copy of
   the author's cache and once with --fresh, and diffs every CSV and tables.txt.

Outputs (this directory): verify_tables.txt, verify.json, fig_signed_mixture.{pdf,png},
cache/ (own rollouts, keyed by snapshot SHA-256), rerun_author/{cached,fresh}/.
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np

ROOT = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
AUTH = os.path.join(ROOT, "results", "paper", "gate_dose")
SCRATCH = ("/private/tmp/claude-501/-Users-michaelseiranian-Desktop-safe-rl-carla/"
           "47ca4e66-e2f4-4455-bfea-c56f9a36fb6d/scratchpad/paper/dose-verify")
PG = "SafetyPointGoal1-v0"
B = 10_000
BAND = (0.76, 0.88)
WIN = {"25k-300k": (25_000, 300_000), "300k-1M": (300_000, 1_000_000), "1M-2M": (1_000_000, 2_000_000),
       "whole run": (0, 10**12)}

# arms of the brief (+ the author's extras, checked too); (key, dir, tag, env, seeds, expected delta0, T)
ARMS1 = [
    ("E1", "results/gate", "lagu-tt", PG, list(range(100, 115)), 2.5, 0.07),
    ("E1all", "results/gate", "lagu-tt", PG, list(range(100, 120)), 2.5, 0.07),
    ("E1e4", "results/gate", "lagu-tt", PG, list(range(100, 108)), 2.5, 0.07),
    ("T035", "results/gate", "lagu-tt-T035", PG, list(range(100, 108)), 2.5, 0.035),
    ("T14", "results/gate", "lagu-tt-T14", PG, list(range(100, 108)), 2.5, 0.14),
    ("ggrad", "results/gate", "lagu-tt-ggrad", PG, list(range(100, 110)), 2.5, 0.07),
    ("lit", "results/gate_literal", "lagu-lit", PG, list(range(100, 110)), 2.5, 0.07),
    ("litT03", "results/gate_literal_x", "lagu-lit-T03", PG, list(range(100, 110)), 2.5, 0.03),
    ("car", "results/gate_x", "lagu-tt", "SafetyCarGoal1-v0", list(range(100, 110)), 2.5, 0.07),
    ("circle", "results/gate_x", "lagu-tt", "SafetyPointCircle1-v0", list(range(100, 110)), 5.0, 0.07),
]
ARMS2 = [("tt", "results/gate", "lagu_nogate-tt", list(range(100, 115)), (600_000, 1_200_000, 2_000_000)),
         ("lit", "results/gate_literal", "lagu_nogate-lit", list(range(100, 110)), (600_000, 1_000_000))]
TS = (0.07, 0.03)
EPS = 1e-6
LAYOUTS = [11000 + k for k in range(5)]


def stem(d, tag, env, s):
    return os.path.join(ROOT, d, f"{tag}_{env}_s{s}")


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def boot(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(0).integers(0, len(num), size=(B, len(num)))
    s = num[idx].sum(1) / den[idx].sum(1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ------------------------------------------------------------------------------------------ A
def part_a(out):
    print("=" * 110)
    print("A. DESCRIPTIVE - Part 1 re-derived from raw CSV/JSON (own code)")
    print("=" * 110)
    res, checks = [], []
    for key, d, tag, env, seeds, d0x, Tx in ARMS1:
        per = []
        for s in seeds:
            st = stem(d, tag, env, s)
            meta = json.load(open(st + ".json"))
            cfg, N = meta["cfg"], int(meta["args"]["steps"])
            rr = rows_of(st + ".csv")
            steps = [int(fl(r["step"])) for r in rr]
            ok_grid = steps == list(range(50_000, N + 1, 50_000))
            if not (cfg["use_gate"] and abs(cfg["delta0"] - d0x) < 1e-9 and abs(cfg["T"] - Tx) < 1e-12 and ok_grid):
                checks.append(f"FAIL {os.path.basename(st)}: use_gate={cfg['use_gate']} d0={cfg['delta0']} "
                              f"T={cfg['T']} grid_ok={ok_grid}")
            # interval rows with n_risk > 0 must have finite engagement / delta_risk_avg
            for r in rr:
                if fl(r["n_risk"]) > 0 and not (np.isfinite(fl(r["engagement"])) and np.isfinite(fl(r["delta_risk_avg"]))):
                    checks.append(f"NaN with n_risk>0: {os.path.basename(st)} step {r['step']}")
                if fl(r["n_pi"]) <= 0:
                    checks.append(f"n_pi=0 row: {os.path.basename(st)} step {r['step']}")
            per.append({"seed": s, "N": N, "d0": cfg["delta0"], "rows": rr,
                        "resumed": int(any(fl(r["resumed"]) == 1 for r in rr))})
        N = per[0]["N"]
        for wn, (lo, hi) in WIN.items():
            if lo >= N:
                continue
            S = {"eng": [], "all": [], "risk": []}
            nrows = set()
            for p in per:
                w = [r for r in p["rows"] if lo < fl(r["step"]) <= hi]
                nrows.add(len(w))
                e_n = sum(fl(r["engagement"]) * fl(r["n_risk"]) for r in w if fl(r["n_risk"]) > 0)
                e_d = sum(fl(r["n_risk"]) for r in w if fl(r["n_risk"]) > 0)
                a_n = sum(fl(r["delta_avg"]) / p["d0"] * fl(r["n_pi"]) for r in w if fl(r["n_pi"]) > 0)
                a_d = sum(fl(r["n_pi"]) for r in w if fl(r["n_pi"]) > 0)
                r_n = sum(fl(r["delta_risk_avg"]) / p["d0"] * fl(r["n_risk"]) for r in w if fl(r["n_risk"]) > 0)
                S["eng"].append((e_n, e_d)); S["all"].append((a_n, a_d)); S["risk"].append((r_n, e_d))
            row = {"key": key, "window": wn, "n_seeds": len(per), "rows_per_seed": sorted(nrows),
                   "n_resumed": sum(p["resumed"] for p in per)}
            for k, v in S.items():
                v = np.array(v)
                row[k] = float(v[:, 0].sum() / v[:, 1].sum())
                row[k + "_lo"], row[k + "_hi"] = boot(v[:, 0], v[:, 1])
            res.append(row)
    print("grid/config checks:", "all pass" if not checks else f"{len(checks)} problems")
    for c in checks[:20]:
        print("  ", c)

    # compare with author's CSV
    auth = {(r["key"], r["window"]): r for r in rows_of(os.path.join(AUTH, "part1_dose_windows.csv"))}
    dmax_pt, dmax_ci, missing = 0.0, 0.0, []
    for r in res:
        a = auth.get((r["key"], r["window"]))
        if a is None:
            missing.append((r["key"], r["window"]))
            continue
        for k in ("eng", "all", "risk"):
            dmax_pt = max(dmax_pt, abs(r[k] - float(a[k])))
            dmax_ci = max(dmax_ci, abs(r[k + "_lo"] - float(a[k + "_lo"])), abs(r[k + "_hi"] - float(a[k + "_hi"])))
    extra_auth = sorted(set(auth) - {(r["key"], r["window"]) for r in res})
    print(f"vs author part1_dose_windows.csv: {len(res)} cells; max |diff| point = {dmax_pt:.2e}, "
          f"CI bounds = {dmax_ci:.2e} (author CSV rounded to 6 s.f.); missing in author: {missing}; "
          f"author-only cells: {extra_auth}")
    print()
    print(f"{'arm':8s} {'window':10s} {'rows/seed':>9s} {'resumed':>7s} {'engagement':>22s} {'budget(all)':>22s} "
          f"{'budget(risk)':>22s}  band(all/risk)")
    def band(x):
        return "below" if x < BAND[0] else "INSIDE" if x <= BAND[1] else "above"
    for r in res:
        f = lambda k: f"{r[k]:.3f} [{r[k+'_lo']:.3f},{r[k+'_hi']:.3f}]"
        print(f"{r['key']:8s} {r['window']:10s} {str(r['rows_per_seed']):>9s} {r['n_resumed']:>7d} {f('eng'):>22s} "
              f"{f('all'):>22s} {f('risk'):>22s}  {band(r['all'])}/{band(r['risk'])}")
    g = {(r["key"], r["window"]): r for r in res}
    rec = [("E1 D = whole-run budget(all)*2.5 vs 1.12158 (prereg 15)", g[("E1", "whole run")]["all"] * 2.5, 1.12158),
           ("X1 D* = whole-run budget(all)*2.5 vs 2.14758 (prereg 16.8)", g[("litT03", "whole run")]["all"] * 2.5, 2.14758),
           ("E1 engagement 25k-300k vs 0.8497 (prereg 15)", g[("E1", "25k-300k")]["eng"], 0.8497),
           ("X1 engagement 25k-300k vs 0.81 (prereg 16.8)", g[("litT03", "25k-300k")]["eng"], 0.81),
           ("E4 T=0.035 whole-run budget vs 0.33 (prereg 16.10)", g[("T035", "whole run")]["all"], 0.33),
           ("E4 T=0.07 (s100-107) whole-run budget vs 0.44", g[("E1e4", "whole run")]["all"], 0.44),
           ("E4 T=0.14 whole-run budget vs 0.57", g[("T14", "whole run")]["all"], 0.57),
           ("E1b 300k-1M budget vs 'about 0.97' (prereg 16.7)", g[("lit", "300k-1M")]["all"], 0.97)]
    print("\nValues already recorded in prereg.md:")
    for name, v, ref in rec:
        print(f"  {name}: {v:.5f} (ref {ref})")
    out["part1"] = res
    out["part1_checks"] = checks
    out["part1_vs_author"] = {"max_abs_diff_point": dmax_pt, "max_abs_diff_ci": dmax_ci, "missing": missing,
                              "author_only": [list(x) for x in extra_auth]}
    out["recorded"] = [{"name": n, "value": v, "ref": r} for n, v, r in rec]
    return g


# ------------------------------------------------------------------------------------------ B
def _job(args):
    st, step, cache = args
    import torch
    import safety_gymnasium
    from lagu.agent import Config, LagU
    torch.set_num_threads(1)
    path = os.path.join(st + ".snap", f"{step}.pt")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if os.path.exists(cache):
        z = np.load(cache)
        if str(z["sha256"]) == sha:
            return st, step, z["q"], z["sd"], z["ret"], "cache"
    meta = json.load(open(st + ".json"))
    snap = torch.load(path, weights_only=True)
    ag = LagU(meta["obs_dim"], meta["act_dim"], Config(**snap["cfg"]))
    ag.actor.load_state_dict(snap["actor"])
    ag.members.load_state_dict(snap["members"])
    env = safety_gymnasium.make(meta["args"]["env"])
    Q, SD, RET = [], [], []
    for lay in LAYOUTS:
        o, _ = env.reset(seed=lay)
        states, ret, done = [], 0.0, False
        while not done:
            states.append(o)
            o, r, c, term, trunc, _ = env.step(ag.act(o, explore=False))
            ret += float(r)
            done = term or trunc
        with torch.no_grad():
            ob = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            act = ag.actor(ob)
            x = torch.cat([ob, act], -1)
            heads = torch.stack([m.q1(x).squeeze(-1) for m in ag.members], 0)   # members' first heads
            q = heads.mean(0)
            sd = torch.sqrt(((heads - q) ** 2).mean(0) + 1e-12)                  # population variance
        Q.append(q.double().numpy()); SD.append(sd.double().numpy()); RET.append(ret)
    env.close()
    Q, SD, RET = np.concatenate(Q), np.concatenate(SD), np.array(RET)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache + ".tmp.npz", q=Q, sd=SD, ret=RET, sha256=sha)
    os.replace(cache + ".tmp.npz", cache)
    return st, step, Q, SD, RET, "computed"


def gate(q, sd, T, signed):
    if signed:
        den = q + EPS
        r = np.where(den > 0, sd / np.where(den > 0, den, 1.0), -np.inf)
    else:
        r = sd / (np.abs(q) + EPS)
    on = r >= T
    bud = np.where(on, T / np.where(on, r, 1.0), 1.0)
    return on, bud


def part_b(out, fresh):
    print("\n" + "=" * 110)
    print("B. EXPLORATORY - Part 2 re-derived with own rollouts (layouts 11000-11004, deterministic policy)")
    print("=" * 110)
    jobs = []
    for key, d, tag, seeds, steps in ARMS2:
        for s in seeds:
            st = stem(d, tag, PG, s)
            assert not json.load(open(st + ".json"))["cfg"]["use_gate"]
            for step in steps:
                c = os.path.join(HERE, "cache", f"{os.path.basename(st)}_{step}.npz")
                if fresh and os.path.exists(c):
                    os.remove(c)
                jobs.append((st, step, c))
    with mp.get_context("spawn").Pool(3) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    print(f"rollouts: {len(res)} ({sum(r[-1] == 'computed' for r in res)} computed, "
          f"{sum(r[-1] == 'cache' for r in res)} from own cache)")
    R = {(os.path.basename(r[0]), r[1]): r for r in res}

    # state-by-state comparison with the author's cache
    dq = dsd = 0.0
    dret = 0.0
    nlen = 0
    for (name, step), r in R.items():
        z = np.load(os.path.join(AUTH, "cache", f"{name}_{step}.npz"))
        if len(z["qbar"]) != len(r[2]):
            nlen += 1
            continue
        dq = max(dq, float(np.max(np.abs(z["qbar"] - r[2]) / np.maximum(1.0, np.abs(z["qbar"])))))
        dsd = max(dsd, float(np.max(np.abs(z["std"] - r[3]) / np.maximum(1.0, np.abs(z["std"])))))
        dret = max(dret, float(np.max(np.abs(z["ep_return"] - r[4]))))
    print(f"vs author cache: length mismatches {nlen}; max rel|dQ_bar| {dq:.2e}, max rel|d sqrt(Q_eu)| {dsd:.2e}, "
          f"max |d episode return| {dret:.2e}")

    auth_seed = {(r["key"], int(r["step"]), int(r["seed"])): r for r in rows_of(os.path.join(AUTH, "part2_per_seed.csv"))}
    auth_sum = {(r["key"], int(r["step"])): r for r in rows_of(os.path.join(AUTH, "part2_signed_counterfactual.csv"))}
    per_seed, summ = [], []
    dmax_seed, dmax_sum = 0.0, 0.0
    for key, d, tag, seeds, steps in ARMS2:
        for step in steps:
            vals = []
            for s in seeds:
                name = f"{tag}_{PG}_s{s}"
                q, sd = R[(name, step)][2], R[(name, step)][3]
                v = {"key": key, "step": step, "seed": s, "n": len(q), "frac_le0": float((q + EPS <= 0).mean()),
                     "med_q": float(np.median(q)), "ret": float(R[(name, step)][4].mean())}
                pos = q + EPS > 0
                for T in TS:
                    for sg in (False, True):
                        on, bud = gate(q, sd, T, sg)
                        nm = "sgn" if sg else "abs"
                        v[f"eng_{nm}_{T}"], v[f"bud_{nm}_{T}"] = float(on.mean()), float(bud.mean())
                    on_a, bud_a = gate(q, sd, T, False)
                    v[f"absfire_le0_{T}"] = float(on_a[~pos].mean()) if (~pos).any() else float("nan")
                    v[f"absfire_pos_{T}"] = float(on_a[pos].mean()) if pos.any() else float("nan")
                    v[f"bud_on_pos_{T}"] = float(bud_a[pos].mean()) if pos.any() else float("nan")
                    v[f"absbud_le0_{T}"] = float(bud_a[~pos].mean()) if (~pos).any() else float("nan")
                # logged would_engage at the snapshot row
                rr = rows_of(stem(d, tag, PG, s) + ".csv")
                lr = [r for r in rr if int(fl(r["step"])) == step][0]
                v["logged_we"] = fl(lr["would_engage"])
                a = auth_seed[(key, step, s)]
                for mine, theirs in [("frac_le0", "frac_qbar_le0"), ("eng_abs_0.07", "eng_abs_T0.07"),
                                     ("eng_sgn_0.07", "eng_sgn_T0.07"), ("bud_abs_0.07", "budget_abs_T0.07"),
                                     ("bud_sgn_0.07", "budget_sgn_T0.07"), ("eng_abs_0.03", "eng_abs_T0.03"),
                                     ("eng_sgn_0.03", "eng_sgn_T0.03"), ("bud_abs_0.03", "budget_abs_T0.03"),
                                     ("bud_sgn_0.03", "budget_sgn_T0.03"), ("logged_we", "logged_would_engage_T007")]:
                    dmax_seed = max(dmax_seed, abs(v[mine] - float(a[theirs])))
                vals.append(v)
            per_seed += vals
            row = {"key": key, "step": step, "n_seeds": len(vals)}
            for k in ["frac_le0", "logged_we"] + [f"{m}_{g}_{T}" for T in TS for m in ("eng", "bud") for g in ("abs", "sgn")]:
                x = np.array([v[k] for v in vals])
                row[k] = float(x.mean())
                row[k + "_lo"], row[k + "_hi"] = boot(x, np.ones_like(x))
            for T in TS:
                row[f"never_{T}"] = int(sum(v[f"eng_sgn_{T}"] == 0.0 for v in vals))
                row[f"near_never_{T}"] = int(sum(v[f"eng_sgn_{T}"] <= 0.01 for v in vals))
                fired = [v[f"bud_sgn_{T}"] for v in vals if v[f"eng_sgn_{T}"] > 0]
                row[f"sgn_bud_fired_min_{T}"] = float(min(fired)) if fired else float("nan")
                row[f"sgn_bud_fired_max_{T}"] = float(max(fired)) if fired else float("nan")
                row[f"sgn_bud_fired_n_ge_076_{T}"] = int(sum(b >= BAND[0] for b in fired))
                row[f"n_fired_{T}"] = len(fired)
                for k in (f"absfire_le0_{T}", f"absfire_pos_{T}", f"bud_on_pos_{T}", f"absbud_le0_{T}"):
                    x = np.array([v[k] for v in vals])
                    row[k] = float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
                # pooled over every state of every seed
                Qall = np.concatenate([R[(f"{tag}_{PG}_s{s}", step)][2] for s in seeds])
                Sall = np.concatenate([R[(f"{tag}_{PG}_s{s}", step)][3] for s in seeds])
                pos = Qall + EPS > 0
                on, bud = gate(Qall, Sall, T, False)
                row[f"pooled_absfire_le0_{T}"] = float(on[~pos].mean()) if (~pos).any() else float("nan")
                row[f"pooled_absfire_pos_{T}"] = float(on[pos].mean())
                row[f"pooled_bud_on_pos_{T}"] = float(bud[pos].mean())
            x, y = np.array([v["eng_abs_0.07"] for v in vals]), np.array([v["logged_we"] for v in vals])
            row["seed_corr_logged_vs_onpolicy"] = float(np.corrcoef(x, y)[0, 1])
            row["seed_mad_logged_vs_onpolicy"] = float(np.mean(np.abs(x - y)))
            fr, bs = np.array([v["frac_le0"] for v in vals]), np.array([v["bud_sgn_0.07"] for v in vals])
            row["seed_corr_fracle0_vs_sgnbud"] = float(np.corrcoef(fr, bs)[0, 1]) if fr.std() > 0 else float("nan")
            a = auth_sum[(key, step)]
            for mine, theirs in [("frac_le0", "frac_qbar_le0"), ("eng_abs_0.07", "eng_abs_T0.07"),
                                 ("eng_sgn_0.07", "eng_sgn_T0.07"), ("bud_abs_0.07", "budget_abs_T0.07"),
                                 ("bud_sgn_0.07", "budget_sgn_T0.07"), ("bud_abs_0.03", "budget_abs_T0.03"),
                                 ("bud_sgn_0.03", "budget_sgn_T0.03"), ("bud_sgn_0.07_lo", "budget_sgn_T0.07_lo"),
                                 ("bud_sgn_0.07_hi", "budget_sgn_T0.07_hi"), ("never_0.07", "n_seeds_sgn_never_T0.07")]:
                dmax_sum = max(dmax_sum, abs(row[mine] - float(a[theirs])))
            summ.append(row)
    print(f"vs author part2_per_seed.csv: max |diff| over every per-seed value = {dmax_seed:.2e}")
    print(f"vs author part2_signed_counterfactual.csv: max |diff| over summary values = {dmax_sum:.2e}")
    print()
    print(f"{'critics':5s} {'step':>8s} {'fracQ<=0':>8s} {'logged':>7s} | T=0.07: {'eng|Q|':>6s} {'engSg':>6s} "
          f"{'bud|Q|':>6s} {'budSg [95%]':>20s} never/near | T=0.03: {'bud|Q|':>6s} {'budSg':>6s} never")
    for r in summ:
        print(f"{r['key']:5s} {r['step']:>8d} {r['frac_le0']:8.3f} {r['logged_we']:7.3f} |        "
              f"{r['eng_abs_0.07']:6.3f} {r['eng_sgn_0.07']:6.3f} {r['bud_abs_0.07']:6.3f} "
              f"{r['bud_sgn_0.07']:6.3f} [{r['bud_sgn_0.07_lo']:.3f},{r['bud_sgn_0.07_hi']:.3f}] "
              f"{r['never_0.07']:>2d}/{r['near_never_0.07']:<2d}      |        {r['bud_abs_0.03']:6.3f} "
              f"{r['bud_sgn_0.03']:6.3f} {r['never_0.03']:>2d}")
    out["part2_vs_author"] = {"state_rel_dq": dq, "state_rel_dsd": dsd, "ret": dret, "len_mismatch": nlen,
                              "per_seed_max_diff": dmax_seed, "summary_max_diff": dmax_sum}
    out["part2_summary"] = summ
    out["part2_per_seed"] = per_seed
    return summ, per_seed


# ------------------------------------------------------------------------------------------ C
def part_c(out, summ, per_seed, g1):
    print("\n" + "=" * 110)
    print("C. EXPLORATORY - diagnostics for statements in the author's summary")
    print("=" * 110)
    print("C1. Where does the port's |Q_bar| gate fire? (T = 0.07; mean over seeds of per-seed fractions; "
          "'pooled' = over all states)")
    print(f"{'critics':5s} {'step':>8s} {'P(fire|Q<=0)':>12s} {'P(fire|Q>0)':>11s} {'pooled Q<=0':>11s} "
          f"{'pooled Q>0':>10s} | budget on Q>0 states only (= signed gate there): seed-mean / pooled")
    for r in summ:
        T = 0.07
        print(f"{r['key']:5s} {r['step']:>8d} {r[f'absfire_le0_{T}']:12.3f} {r[f'absfire_pos_{T}']:11.3f} "
              f"{r[f'pooled_absfire_le0_{T}']:11.3f} {r[f'pooled_absfire_pos_{T}']:10.3f} | "
              f"{r[f'bud_on_pos_{T}']:.3f} / {r[f'pooled_bud_on_pos_{T}']:.3f}")
    print("  -> signed mean budget = P(Q<=0)*1 + P(Q>0)*budget_on_Q>0; it lands near the band by mixing un-gated "
          "states (budget 1) with heavily tightened ones.")
    print("\nC2. Per-seed signed budget (T = 0.07) on seeds where the signed gate fires at all; near-never = "
          "engagement <= 0.01")
    for r in summ:
        if r["key"] != "tt":
            continue
        fired = sorted(v["bud_sgn_0.07"] for v in per_seed if v["key"] == "tt" and v["step"] == r["step"]
                       and v["eng_sgn_0.07"] > 0)
        print(f"  tt {r['step']:>8d}: never {r['never_0.07']}/15, near-never {r['near_never_0.07']}/15; fired seeds "
              f"(n={len(fired)}) budget range {min(fired):.3f}-{max(fired):.3f}, {r['sgn_bud_fired_n_ge_076_0.07']} "
              f"of them >= 0.76; values {[round(x, 2) for x in fired]}")
        print(f"     seed-level corr(frac Q<=0, signed budget) = {r['seed_corr_fracle0_vs_sgnbud']:.3f}")
    print("\nC3. Logged would_engage (replay, |Q|, T=0.07) vs on-policy |Q| engagement, seed level")
    for r in summ:
        print(f"  {r['key']:4s} {r['step']:>8d}: means {r['logged_we']:.3f} vs {r['eng_abs_0.07']:.3f}; "
              f"corr {r['seed_corr_logged_vs_onpolicy']:.3f}; mean |diff| {r['seed_mad_logged_vs_onpolicy']:.3f}")
    print("\nC4. Specific per-seed statements")
    pick = {(v["key"], v["step"], v["seed"]): v for v in per_seed}
    for (k, st, s) in [("tt", 600_000, 103), ("tt", 1_200_000, 103), ("tt", 2_000_000, 103), ("tt", 2_000_000, 108)]:
        v = pick[(k, st, s)]
        print(f"  {k} s{s} @{st}: median Q_bar {v['med_q']:.3f}, mean return (5 ep) {v['ret']:.1f}, "
              f"eng|Q| {v['eng_abs_0.07']:.3f}, eng signed {v['eng_sgn_0.07']:.3f}")

    # C5: logged sign of qbar_avg in gated E1 (lagu-tt s100-114) after 300k, and engagement by sign
    print("\nC5. DESCRIPTIVE - gated E1 lagu-tt s100-114, interval rows with step > 300k: logged qbar_avg sign and "
          "engagement")
    for tag, lab in (("lagu-tt", "gated E1"), ("lagu_nogate-tt", "ungated E1")):
        q, e, n, we = [], [], [], []
        for s in range(100, 115):
            for r in rows_of(stem("results/gate", tag, PG, s) + ".csv"):
                if fl(r["step"]) > 300_000:
                    q.append(fl(r["qbar_avg"])); e.append(fl(r["engagement"])); n.append(fl(r["n_risk"]))
                    we.append(fl(r["would_engage"]))
        q, e, n, we = map(np.array, (q, e, n, we))
        neg = q <= 0
        col = e if tag == "lagu-tt" else we
        nm = "engagement" if tag == "lagu-tt" else "would_engage"
        f = lambda m: float((col[m] * n[m]).sum() / n[m].sum()) if m.any() else float("nan")
        print(f"  {lab:10s}: rows {len(q)}, qbar_avg <= 0 in {neg.mean():.3f}; median |qbar_avg| "
              f"{np.median(np.abs(q)):.3f}; n_risk-weighted {nm}: rows with qbar_avg<=0 {f(neg):.3f}, "
              f"rows with qbar_avg>0 {f(~neg):.3f}")
        out[f"C5_{tag}"] = {"rows": int(len(q)), "frac_neg": float(neg.mean()), "median_abs": float(np.median(np.abs(q))),
                            "eng_neg": f(neg), "eng_pos": f(~neg)}


# ------------------------------------------------------------------------------------------ figure
def figure(summ, per_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                         "font.size": 9, "axes.grid": True, "grid.color": "#e4e3df", "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    cols = {600_000: ("#2a78d6", "o"), 1_200_000: ("#eb6834", "s"), 2_000_000: ("#1baf7a", "^")}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    a.axhspan(*BAND, color="#9a9890", alpha=0.25, lw=0, label="0.76-0.88: as we read Zhang et al. Fig. 5(d)")
    xx = np.linspace(0, 1, 50)
    for st, (c, m) in cols.items():
        v = [p for p in per_seed if p["key"] == "tt" and p["step"] == st]
        a.plot([p["frac_le0"] for p in v], [p["bud_sgn_0.07"] for p in v], m, color=c, ms=6, alpha=0.85,
               label=f"tt critics @ {st/1e6:g}M (one marker per seed)")
        bpos = [r for r in summ if r["key"] == "tt" and r["step"] == st][0]["pooled_bud_on_pos_0.07"]
        a.plot(xx, xx + (1 - xx) * bpos, color=c, lw=1, ls="--")
    a.set_xlabel("fraction of visited states with Q_bar <= 0 (per seed)")
    a.set_ylabel("signed-gate mean budget\ndelta_ada / delta0 (T = 0.07)")
    a.set_title("(a) signed budget per seed tracks the share of Q_bar <= 0 states\n"
                "dashed: f + (1 - f) x pooled budget on Q_bar > 0 states", loc="left", fontsize=9)
    a.set_xlim(-0.03, 1.03); a.set_ylim(0, 1.05)
    a.legend(fontsize=7.5, loc="lower right")
    b = ax[1]
    ks = [r for r in summ if r["key"] == "tt"]
    x = np.arange(len(ks))
    w = 0.26
    b.axhspan(*BAND, color="#9a9890", alpha=0.25, lw=0)
    b.bar(x - w, [r["bud_abs_0.07"] for r in ks], w, color="#2a78d6", label="|Q_bar| gate, all states")
    b.bar(x, [r["bud_sgn_0.07"] for r in ks], w, color="#eb6834", label="signed gate, all states")
    b.bar(x + w, [r["bud_on_pos_0.07"] for r in ks], w, color="#4a3aa7", hatch="//", edgecolor="#fcfcfb",
          label="either gate, Q_bar > 0 states only")
    b.set_xticks(x, [f"tt {r['step']/1e6:g}M" for r in ks])
    b.set_ylabel("mean budget delta_ada / delta0\n(T = 0.07; mean over seeds)")
    b.set_xlabel("ungated critics (lagu_nogate-tt s100-114) and snapshot step")
    b.set_ylim(0, 1.05)
    b.set_title("(b) where the signed gate fires it tightens about as hard as |Q_bar|", loc="left", fontsize=9)
    b.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("EXPLORATORY (verification): the signed-gate average is a mixture of ungated states and heavily "
                 "tightened states\n5 held-out layouts x 1000 on-policy states per snapshot; static counterfactual "
                 "on ungated critics", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0.01, 0, 1, 0.88))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_signed_mixture.{ext}"), dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ------------------------------------------------------------------------------------------ D
def part_d(out, fresh_also=True):
    print("\n" + "=" * 110)
    print("D. Re-run of a byte-identical copy of the author's make.py (scratch tree, symlinked results/ and lagu/)")
    print("=" * 110)
    res = {}
    for mode in ("cached", "fresh"):
        if mode == "fresh" and not fresh_also:
            continue
        root = os.path.join(SCRATCH, f"root_{mode}")
        if os.path.exists(root):
            shutil.rmtree(root)
        gd = os.path.join(root, "results", "paper", "gate_dose")
        os.makedirs(gd)
        for sub in ("gate", "gate_literal", "gate_literal_x", "gate_x"):
            os.symlink(os.path.join(ROOT, "results", sub), os.path.join(root, "results", sub))
        os.symlink(os.path.join(ROOT, "lagu"), os.path.join(root, "lagu"))
        shutil.copy2(os.path.join(AUTH, "make.py"), os.path.join(gd, "make.py"))
        if mode == "cached":
            shutil.copytree(os.path.join(AUTH, "cache"), os.path.join(gd, "cache"))
        same_src = filecmp.cmp(os.path.join(AUTH, "make.py"), os.path.join(gd, "make.py"), shallow=False)
        cmd = [sys.executable, os.path.join(gd, "make.py"), "--workers", "3"] + (["--fresh"] if mode == "fresh" else [])
        p = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        dest = os.path.join(HERE, "rerun_author", mode)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.makedirs(dest)
        cmp = {}
        for fn in ("tables.txt", "part1_dose_windows.csv", "part2_signed_counterfactual.csv", "part2_per_seed.csv",
                   "fig_budget_vs_step.png", "fig_signed_counterfactual.png", "fig_budget_vs_step.pdf",
                   "fig_signed_counterfactual.pdf"):
            src = os.path.join(gd, fn)
            if not os.path.exists(src):
                cmp[fn] = "MISSING"
                continue
            shutil.copy2(src, os.path.join(dest, fn))
            if filecmp.cmp(src, os.path.join(AUTH, fn), shallow=False):
                cmp[fn] = "byte-identical"
            elif fn.endswith(".txt"):
                a = open(os.path.join(AUTH, fn)).read().splitlines()
                b = open(src).read().splitlines()
                diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
                cmp[fn] = f"differs on {len(diff)} line(s) (+{abs(len(a) - len(b))} length): " + \
                          "; ".join(f"L{i+1}: '{x.strip()[-60:]}' -> '{y.strip()[-60:]}'" for i, x, y in diff[:3])
            elif fn.endswith(".png"):
                import matplotlib.image as mpimg
                A, Bm = mpimg.imread(os.path.join(AUTH, fn)), mpimg.imread(src)
                cmp[fn] = ("pixel-identical (bytes differ)" if A.shape == Bm.shape and np.array_equal(A, Bm)
                           else f"pixels differ (shape {A.shape} vs {Bm.shape}, max {np.max(np.abs(A - Bm)) if A.shape == Bm.shape else 'n/a'})")
            else:
                cmp[fn] = "bytes differ (PDF metadata expected)"
        n_cache = len([f for f in os.listdir(os.path.join(gd, "cache"))]) if os.path.isdir(os.path.join(gd, "cache")) else 0
        res[mode] = {"returncode": p.returncode, "same_source": same_src, "files": cmp, "cache_files": n_cache,
                     "stderr_tail": p.stderr[-800:]}
        with open(os.path.join(dest, "stdout.txt"), "w") as f:
            f.write(p.stdout)
        print(f"[{mode}] returncode {p.returncode}; copy byte-identical to author's make.py: {same_src}; "
              f"cache files after run: {n_cache}")
        for fn, v in cmp.items():
            print(f"    {fn}: {v}")
        if p.returncode != 0:
            print("    stderr:", p.stderr[-800:])
    out["rerun_author"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="recompute own rollouts")
    ap.add_argument("--skip-author", action="store_true", help="skip part D")
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    out = {}
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("DESCRIPTIVE / EXPLORATORY verification of results/paper/gate_dose (not confirmatory; outside the "
              "Holm family)\n")
        g1 = part_a(out)
        summ, per_seed = part_b(out, a.fresh)
        part_c(out, summ, per_seed, g1)
    text = buf.getvalue()
    print(text, end="", flush=True)
    figure(summ, per_seed)
    if not a.skip_author:
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            part_d(out)
        print(buf2.getvalue(), end="")
        text += buf2.getvalue()
    with open(os.path.join(HERE, "verify_tables.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(HERE, "verify.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
