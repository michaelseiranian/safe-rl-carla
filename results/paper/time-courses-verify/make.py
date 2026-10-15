"""DESCRIPTIVE / EXPLORATORY verification of results/paper/timecourse (independent re-derivation).

Nothing here is confirmatory. This script checks another agent's exploratory time courses
(results/paper/timecourse/make.py and its outputs) by recomputing every number from the raw run
files with its own loader (the csv module; lagu.analyze is not imported), its own quantile
function and its own paired bootstrap, then comparing.

    cd /Users/michaelseiranian/Desktop/safe-rl-carla
    .venv/bin/python results/paper/time-courses-verify/make.py

Steps (all printed; the printout is also saved as verify_output.txt next to this script):
  A  raw-file audit: 50k grid, duplicates, non-finite values, n_pi > 0, resumed flags, config.
  B  every value in their timecourse_data.csv recomputed (median / IQR over seeds per 50k row,
     collapse count = runs with eval_return < 5, n_pi-weighted mean of delta_avg / 2.5).
  C  the EXPLORATORY horizon table (E1, window 0.8N <= step <= N, N = 0.5, 1, 1.5, 2M) with an
     own paired bootstrap (10,000 resamples of seed pairs, percentile, numpy default_rng(0)).
  D  the 2M held-out reference rows from <stem>.heldout.json (pooled), against prereg.md 15.
  E  every numeric / qualitative statement of their report, checked over the full curves.
  F  figure checks: per-arm marker offsets (their staggering rule) for coinciding markers.
  G  their make.py re-run twice from a scratch copy (outputs written to scratch only), and the
     outputs compared with the files in results/paper/timecourse/.
Figures (DESCRIPTIVE): verify_overlay.{pdf,png} (own recomputation as lines, their CSV values
as markers) and verify_collapse_raster.{pdf,png} (per-seed eval_return < 5 at every 50k row).
Numpy + matplotlib only. Writes only next to this script and under the scratch directory below.
"""
from __future__ import annotations

import csv
import filecmp
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/michaelseiranian/Desktop/safe-rl-carla"
THEIRS = os.path.join(ROOT, "results", "paper", "timecourse")
SCRATCH = ("/private/tmp/claude-501/-Users-michaelseiranian-Desktop-safe-rl-carla/"
           "47ca4e66-e2f4-4455-bfea-c56f9a36fb6d/scratchpad/paper/time-courses-verify")
PY = os.path.join(ROOT, ".venv", "bin", "python")
ENV = "SafetyPointGoal1-v0"
DELTA0, LIMIT, COLLAPSE, D_E2 = 2.5, 25.0, 5.0, 1.12158
EXT = set(range(115, 120))
S10, S15, S20 = list(range(100, 110)), list(range(100, 115)), list(range(100, 120))

# Defined from the brief (not copied from their script): experiment -> [(dir, arm, seeds)], steps
EXPS = {
    "E1": ([("gate", "lagu-tt", S15), ("gate", "lagu_nogate-tt", S15)], 2_000_000),
    "E2": ([("gate", "lagu-tt", S20), ("gate", "lagu_nogate-tt", S20),
            ("gate", "lagu_nogate-tt-match", S20)], 2_000_000),
    "E1b": ([("gate_literal", "lagu-lit", S10), ("gate_literal", "lagu_nogate-lit", S10)], 1_000_000),
    "X1": ([("gate_literal_x", "lagu-lit-T03", S10), ("gate_literal_x", "lagu_nogate-lit-match", S10),
            ("gate_literal", "lagu_nogate-lit", S10)], 1_000_000),
}
COLS = ("eval_return", "eval_cost", "lambda_avg", "qc_avg", "qstd_avg", "delta_avg", "n_pi", "resumed")


# ----------------------------------------------------------------------------------------------
# loading and statistics (own code)
def f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


_RUNS: dict = {}


def run(d: str, arm: str, seed: int):
    """One run's columns as arrays, or None if the CSV does not exist."""
    key = (d, arm, seed)
    if key not in _RUNS:
        stem = os.path.join(ROOT, "results", d, f"{arm}_{ENV}_s{seed}")
        if not os.path.exists(stem + ".csv"):
            _RUNS[key] = None
        else:
            with open(stem + ".csv") as fh:
                rows = list(csv.DictReader(fh))
            with open(stem + ".json") as fh:
                meta = json.load(fh)
            held = None
            if os.path.exists(stem + ".heldout.json"):
                with open(stem + ".heldout.json") as fh:
                    held = json.load(fh)
            r = {c: np.array([f(x.get(c)) for x in rows]) for c in COLS}
            r["step"] = np.array([f(x["step"]) for x in rows])
            r.update(meta=meta, held=held, seed=seed, arm=arm, dir=d)
            _RUNS[key] = r
    return _RUNS[key]


def runs(d, arm, seeds):
    out = [run(d, arm, s) for s in seeds]
    return [r for r in out if r is not None]


def q7(x, p):
    """Type-7 (linear) sample quantile, written out by hand; NaNs dropped."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    if len(x) == 0:
        return math.nan
    h = (len(x) - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, len(x) - 1)
    return float(x[lo] + (h - lo) * (x[hi] - x[lo]))


def M(rs, col):
    return np.vstack([r[col] for r in rs])


def per_step(rs):
    """Everything plotted, per 50k row, from the seeds in rs."""
    out = {}
    for c in ("eval_return", "eval_cost", "lambda_avg", "qc_avg", "qstd_avg"):
        m = M(rs, c)
        out[c] = np.array([[q7(m[:, i], p) for i in range(m.shape[1])] for p in (0.25, 0.5, 0.75)])
    ret = M(rs, "eval_return")
    out["collapsed"] = np.array([sum(1 for v in ret[:, i] if not v >= COLLAPSE) for i in range(ret.shape[1])])
    dl, npi = M(rs, "delta_avg"), M(rs, "n_pi")
    wm = []
    for i in range(dl.shape[1]):
        ok = np.isfinite(dl[:, i]) & np.isfinite(npi[:, i]) & (npi[:, i] > 0)
        wm.append(float((dl[ok, i] * npi[ok, i]).sum() / npi[ok, i].sum()) / DELTA0)
    out["budget_wmean"] = np.array(wm)
    out["budget_median"] = np.array([q7(dl[:, i] / DELTA0, 0.5) for i in range(dl.shape[1])])
    out["budget"] = np.array([[q7(dl[:, i] / DELTA0, p) for i in range(dl.shape[1])] for p in (0.25, 0.5, 0.75)])
    return out


def boot95(d, n=10_000, seed=0):
    """Own paired bootstrap: resample seed pairs (rows of d) with replacement, mean, percentile."""
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)                      # (n, k)
    return np.percentile(means, [2.5, 97.5], axis=0).T


def steps_of(N):
    return np.arange(50_000, N + 1, 50_000, dtype=float)


def at(x, steps, t):
    return x[int(np.where(steps == t)[0][0])]


# ----------------------------------------------------------------------------------------------
def section_A():
    print("A. RAW-FILE AUDIT (DESCRIPTIVE)")
    problems, resumed = [], {}
    for key, (arms, N) in EXPS.items():
        for d, arm, seeds in arms:
            for s in seeds:
                r = run(d, arm, s)
                if r is None:
                    if not (key == "E2" and arm == "lagu_nogate-tt" and s in EXT):
                        problems.append(f"missing {d}/{arm} s{s}")
                    continue
                if not np.array_equal(r["step"], steps_of(N)):
                    problems.append(f"{arm} s{s}: steps not the 50k grid to {N}")
                for c in COLS:
                    if not np.all(np.isfinite(r[c])):
                        problems.append(f"{arm} s{s}: non-finite {c}")
                if np.any(r["n_pi"] <= 0):
                    problems.append(f"{arm} s{s}: n_pi <= 0")
                if np.any(r["resumed"] == 1):
                    resumed.setdefault(arm, set()).add(s)
    missing_ng = [s for s in EXT if run("gate", "lagu_nogate-tt", s) is None]
    print(f"  lagu_nogate-tt s115-119 on disk: {5 - len(missing_ng)} of 5 (so E2 uses s100-114 for that arm)")
    print(f"  problems: {problems or 'none'} (every run on the full 50k grid, all plotted columns finite, n_pi > 0)")
    for arm, ss in sorted(resumed.items()):
        print(f"  resumed runs: {arm} s{min(ss)}-s{max(ss)} ({len(ss)} runs)")
    cfg = {(arm): (run(d, arm, seeds[0])["meta"]["cfg"].get("delta0"), run(d, arm, seeds[0])["meta"]["cfg"].get("T"),
                   run(d, arm, seeds[0])["meta"]["cfg"].get("use_gate"))
           for arms, _ in EXPS.values() for d, arm, seeds in arms}
    for arm, (d0, T, g) in cfg.items():
        print(f"  cfg {arm:24s} delta0={d0:.6g} T={T:g} use_gate={g}")
    print()
    return resumed


# ----------------------------------------------------------------------------------------------
def section_B():
    print("B. EVERY VALUE OF THEIR timecourse_data.csv RECOMPUTED (DESCRIPTIVE)")
    with open(os.path.join(THEIRS, "timecourse_data.csv")) as fh:
        theirs = list(csv.DictReader(fh))
    T = {(r["experiment"], r["arm"], r["seed_set"], int(r["step"]), r["metric"]): r for r in theirs}
    mine, S = {}, {}
    for key, (arms, N) in EXPS.items():
        st = steps_of(N)
        for d, arm, seeds in arms:
            rs = runs(d, arm, seeds)
            sets = {"all": rs}
            if key == "E2" and any(r["seed"] in EXT for r in rs):
                sets["s100-114"] = [r for r in rs if r["seed"] not in EXT]
                sets["s115-119"] = [r for r in rs if r["seed"] in EXT]
            for name, sub in sets.items():
                P = per_step(sub)
                S[(key, arm, name)] = (P, len(sub), st)
                for i, t in enumerate(st):
                    if name != "s115-119":
                        for c in ("eval_return", "eval_cost", "lambda_avg", "qc_avg", "qstd_avg"):
                            mine[(key, arm, name, int(t), c)] = (P[c][1, i], P[c][0, i], P[c][2, i], math.nan)
                    if name == "all":
                        mine[(key, arm, name, int(t), "budget")] = (P["budget"][1, i], P["budget"][0, i],
                                                                    P["budget"][2, i], P["budget_wmean"][i])
                    mine[(key, arm, name, int(t), "collapsed_count")] = (float(P["collapsed"][i]), math.nan,
                                                                         math.nan, math.nan)
    only_mine, only_theirs = set(mine) - set(T), set(T) - set(mine)
    worst, nval, bad = 0.0, 0, []
    for k in set(mine) & set(T):
        for j, fld in enumerate(("median", "q25", "q75", "npi_weighted_mean")):
            a, b = mine[k][j], f(T[k][fld])
            if math.isnan(a) and math.isnan(b):
                continue
            nval += 1
            rel = abs(a - b) / max(1e-12, abs(a), abs(b)) if abs(a - b) > 1e-12 else 0.0
            ok = abs(a - b) <= 1e-9 or rel <= 6e-6       # their CSV is printed with 6 significant digits
            worst = max(worst, rel)
            if not ok:
                bad.append((k, fld, a, b))
    print(f"  keys: mine {len(mine)}, theirs {len(T)}, only mine {len(only_mine)}, only theirs {len(only_theirs)}")
    print(f"  values compared: {nval}; mismatches beyond 6-significant-digit rounding: {len(bad)}; "
          f"worst relative difference {worst:.2e}")
    for b in bad[:10]:
        print("   MISMATCH", b)
    print()
    return S, T


# ----------------------------------------------------------------------------------------------
def section_C():
    print("C. EXPLORATORY HORIZON TABLE, E1 (seeds 100-114, 15 pairs), recomputed with own bootstrap")
    print("   window = in-training rows with 0.8N <= step <= N; collapsed = window-mean return < 5")
    with open(os.path.join(THEIRS, "horizon_table.csv")) as fh:
        th = {(int(r["N"]), r["arm"]): r for r in csv.DictReader(fh)}
    G, U = runs("gate", "lagu-tt", S15), runs("gate", "lagu_nogate-tt", S15)
    hdr = f"  {'N':>5} {'rows':>4} | {'arm':15s} {'ret mean':>8} {'ret med':>8} {'cost mean':>9} {'cost med':>8} {'collapsed':>9} | match theirs"
    print(hdr)
    out, maxdiff = [], 0.0
    for N in (500_000, 1_000_000, 1_500_000, 2_000_000):
        W = {}
        for name, rs in (("lagu-tt", G), ("lagu_nogate-tt", U)):
            m = (rs[0]["step"] >= 0.8 * N) & (rs[0]["step"] <= N)
            ret = np.array([r["eval_return"][m].mean() for r in rs])
            cost = np.array([r["eval_cost"][m].mean() for r in rs])
            W[name] = (ret, cost, int(np.sum(~(ret >= COLLAPSE))), int(m.sum()))
        d = np.column_stack([W["lagu-tt"][0] - W["lagu_nogate-tt"][0], W["lagu-tt"][1] - W["lagu_nogate-tt"][1]])
        ci = boot95(d)
        for name in ("lagu-tt", "lagu_nogate-tt"):
            ret, cost, col, nrow = W[name]
            vals = [ret.mean(), np.median(ret), cost.mean(), np.median(cost), col]
            tv = th[(N, name)]
            their = [f(tv["return_mean"]), f(tv["return_median"]), f(tv["cost_mean"]), f(tv["cost_median"]),
                     f(tv["collapsed"])]
            dd = max(abs(a - b) for a, b in zip(vals, their))
            if name == "lagu-tt":
                mine_d = [d[:, 0].mean(), *ci[0], d[:, 1].mean(), *ci[1]]
                their_d = [f(tv[k]) for k in ("d_return_mean", "d_return_lo95", "d_return_hi95",
                                              "d_cost_mean", "d_cost_lo95", "d_cost_hi95")]
                dd = max(dd, max(abs(a - b) for a, b in zip(mine_d, their_d)))
            maxdiff = max(maxdiff, dd)
            print(f"  {N / 1e6:4.1f}M {nrow:4d} | {name:15s} {vals[0]:8.2f} {vals[1]:8.2f} {vals[2]:9.2f} "
                  f"{vals[3]:8.2f} {col:5d}/15 | max |diff| {dd:.1e}")
            out.append((N, name, *vals))
        print(f"  {'':>11}   d return {d[:, 0].mean():+6.2f} [{ci[0][0]:+6.2f}, {ci[0][1]:+6.2f}]   "
              f"d cost {d[:, 1].mean():+6.2f} [{ci[1][0]:+6.2f}, {ci[1][1]:+6.2f}]")
    print(f"  largest absolute difference from their horizon_table.csv: {maxdiff:.1e}")
    print()
    return out


def section_D():
    print("D. 2M HELD-OUT REFERENCE ROWS (primary endpoint, pooled 5 snapshots x 20 layouts), own read")
    res = {}
    for name in ("lagu-tt", "lagu_nogate-tt"):
        rs = runs("gate", name, S15)
        ret = np.array([f(r["held"]["pooled"]["eval_return"]) for r in rs])
        cost = np.array([f(r["held"]["pooled"]["eval_cost"]) for r in rs])
        ends = {r["held"]["end_step"] for r in rs}
        col = int(np.sum(~(ret >= COLLAPSE)))
        res[name] = (ret.mean(), cost.mean(), col)
        print(f"  {name:15s} return {ret.mean():6.2f} (median {np.median(ret):5.2f})  cost {cost.mean():6.2f} "
              f"(median {np.median(cost):5.2f})  collapsed {col}/15  end_step {sorted(ends)}")
    ok = (round(res["lagu-tt"][0], 2), round(res["lagu_nogate-tt"][0], 2), round(res["lagu-tt"][1], 2),
          round(res["lagu_nogate-tt"][1], 2), res["lagu-tt"][2], res["lagu_nogate-tt"][2]) == (
        5.42, 15.59, 34.46, 37.77, 9, 3)
    print(f"  prereg.md 15 (5.42 vs 15.59, 34.46 vs 37.77, 9 vs 3) reproduced: {ok}")
    print()


# ----------------------------------------------------------------------------------------------
CHECKS = []


def check(label, mine, claim, ok):
    if isinstance(ok, (bool, np.bool_)):
        ok = bool(ok)
    CHECKS.append((label, mine, claim, ok))


def section_E(S):
    print("E. STATEMENTS IN THEIR REPORT, CHECKED ON THE FULL CURVES (DESCRIPTIVE)")
    g, u = S[("E1", "lagu-tt", "all")], S[("E1", "lagu_nogate-tt", "all")]
    st = g[2]
    P, Q = g[0], u[0]
    bw = P["budget_wmean"]
    # budget
    vals = [at(bw, st, t) for t in (50_000, 300_000, 500_000, 1_000_000, 2_000_000)]
    check("E1 lagu-tt budget (n_pi-wtd mean) at 50k/0.3M/0.5M/1M/2M", " ".join(f"{v:.3f}" for v in vals),
          "0.93 0.50 0.58 0.36 0.31",
          all(abs(a - b) < 0.006 for a, b in zip(vals, (0.93, 0.50, 0.58, 0.36, 0.31))))
    bm = P["budget_median"]
    vals_m = [at(bm, st, t) for t in (50_000, 300_000, 500_000, 1_000_000, 2_000_000)]
    check("  same, MEDIAN over seeds (summary calls it 'median budget')", " ".join(f"{v:.3f}" for v in vals_m),
          "summary wording 'median'", "note: the plotted/quoted value is the mean, not the median")
    below = bw < D_E2 / DELTA0
    first = st[np.argmax(below)]
    stays = bool(np.all(below[st >= first]))
    check("E1 lagu-tt budget first below 0.449 / stays below after", f"{first / 1e6:.2f}M / {stays}",
          "from 0.7M", first == 700_000 and stays)
    check("E1 lagu-tt budget minimum", f"{bw.min():.3f} at {st[np.argmin(bw)] / 1e6:.2f}M", "0.306",
          abs(bw.min() - 0.306) < 0.0006)
    # collapse counts
    pairs = [(int(at(P["collapsed"], st, t)), int(at(Q["collapsed"], st, t)))
             for t in (250_000, 500_000, 1_000_000, 1_500_000, 2_000_000)]
    check("E1 runs < 5, gate vs nogate at 0.25/0.5/1/1.5/2M", " ".join(f"{a}v{b}" for a, b in pairs),
          "5v4 9v2 5v3 13v3 12v4", pairs == [(5, 4), (9, 2), (5, 3), (13, 3), (12, 4)])
    c = P["collapsed"]
    early = st <= 1_000_000
    pk1 = int(c[early].max())
    pk1s = st[early][c[early] == pk1] / 1e6
    pk2 = int(c.max())
    pk2s = st[c == pk2] / 1e6
    check("E1 lagu-tt collapse peaks", f"{pk1}/15 at {pk1s.tolist()}M; {pk2}/15 at {pk2s.tolist()}M",
          "11/15 at 0.4-0.45M; 13/15 at 1.5M", pk1 == 11 and list(pk1s) == [0.4, 0.45] and pk2 == 13
          and list(pk2s) == [1.5])
    lo = (st >= 600_000) & (st <= 900_000)
    check("E1 lagu-tt 'partly recovers near 0.75M': min count / max median return in 0.6-0.9M",
          f"{int(c[lo].min())}/15 at {(st[lo][c[lo] == c[lo].min()] / 1e6).tolist()}M; median return "
          f"{P['eval_return'][1][lo].max():.2f} (nogate {Q['eval_return'][1][lo].max():.2f})",
          "partly recovers", True)
    # median return, lambda, qc, qstd
    for col, ts, claim in (("eval_return", (500_000, 1_000_000, 2_000_000), "2.37v13.49 8.62v21.78 1.23v19.38"),
                           ("lambda_avg", (500_000, 1_000_000, 2_000_000), "0.34v0.21 0.22v0.25 0.40v0.28"),
                           ("qc_avg", (500_000, 1_000_000), "0.34v2.55 0.95v2.64"),
                           ("qstd_avg", (500_000, 2_000_000), "0.19v0.14 2.64v0.52")):
        got = " ".join(f"{at(P[col][1], st, t):.2f}v{at(Q[col][1], st, t):.2f}" for t in ts)
        check(f"E1 median {col}, gate v nogate", got, claim, got == claim)
    # qualitative ranges over the full curve
    late = st >= 300_000
    lam_g, lam_u = P["lambda_avg"][1], Q["lambda_avg"][1]
    check("E1 median lambda range (all rows) gate | nogate",
          f"{lam_g.min():.3f}-{lam_g.max():.3f} | {lam_u.min():.3f}-{lam_u.max():.3f}; rows >= 0.3M: "
          f"{lam_g[late].min():.2f}-{lam_g[late].max():.2f} | {lam_u[late].min():.2f}-{lam_u[late].max():.2f}",
          "'similar ... about 0.2-0.45'", "approx.: holds after ~0.15M; both start near 0 at 50k; gate "
          "above nogate by ~0.1 over 0.25-0.45M")
    lam_diff = lam_g - lam_u
    check("E1 median lambda gate - nogate, largest gap", f"{lam_diff.max():+.3f} at {st[np.argmax(lam_diff)] / 1e6:.2f}M, "
          f"{lam_diff.min():+.3f} at {st[np.argmin(lam_diff)] / 1e6:.2f}M", "similar", True)
    qg, qu = P["qc_avg"][1], Q["qc_avg"][1]
    check("E1 median Q_c range gate | nogate (all rows; rows >= 0.3M)",
          f"{qg.min():.2f}-{qg.max():.2f} | {qu.min():.2f}-{qu.max():.2f}; "
          f"{qg[late].min():.2f}-{qg[late].max():.2f} | {qu[late].min():.2f}-{qu[late].max():.2f}",
          "'about 0-1.6 against 2.3-3.0'", "holds only from ~0.3M; before that both arms 2.5-4.6")
    sg = P["qstd_avg"][1]
    check("E1 median qstd lagu-tt: max before 1M / after 1M", f"{sg[st < 1e6].max():.2f} / {sg[st >= 1e6].max():.2f}",
          "grows after 1M", bool(sg[st >= 1e6].max() > 2 * sg[st < 1e6].max()))
    # E2
    e2 = {a: S[("E2", a, "all")] for a in ("lagu-tt", "lagu_nogate-tt", "lagu_nogate-tt-match")}
    st2 = e2["lagu-tt"][2]
    got = " | ".join(" ".join(f"{int(at(e2[a][0]['collapsed'], st2, t))}/{e2[a][1]}" for a in e2)
                     for t in (500_000, 2_000_000))
    check("E2 runs < 5 (gate / nogate / match) at 0.5M | 2M", got, "12/20 2/15 17/20 | 17/20 4/15 15/20",
          got == "12/20 2/15 17/20 | 17/20 4/15 15/20")
    for a in ("lagu-tt", "lagu_nogate-tt-match"):
        c = e2[a][0]["collapsed"] / e2[a][1]
        half = st2[np.argmax(c >= 0.5)] / 1e6
        peak = st2[np.argmax(e2[a][0]["collapsed"])] / 1e6
        check(f"E2 {a}: first row with >= half collapsed; first peak row", f"{half:.2f}M; {peak:.2f}M "
              f"({int(e2[a][0]['collapsed'].max())}/{e2[a][1]})", "'matched collapses even earlier'",
              "compare the two rows: match crosses half one row (50k) before lagu-tt and peaks higher")
    for a, claim in (("lagu-tt", 0.9), ("lagu_nogate-tt-match", 0.35)):
        P5, n5, _ = S[("E2", a, "s115-119")]
        allfive = P5["collapsed"] == 5
        first5 = st2[np.argmax(allfive)] / 1e6
        stays = bool(np.all(allfive[st2 >= first5 * 1e6]))
        n_all5 = int(allfive.sum())
        n_after = int(allfive[st2 >= first5 * 1e6].sum())
        n_rows_after = int((st2 >= first5 * 1e6).sum())
        check(f"E2 {a} s115-119: at 2M / first row all 5 / all 5 at every later row",
              f"{int(P5['collapsed'][-1])}/5 / {first5:.2f}M / {stays} ({n_after} of the {n_rows_after} rows from "
              f"there on have all 5 below 5)",
              f"5/5 at 2M, first all 5 at {claim}M", int(P5["collapsed"][-1]) == 5 and abs(first5 - claim) < 1e-9)
    # E1b, X1
    for key, gate_arm, claims in (("E1b", "lagu-lit", (0.843, 0.2)), ("X1", "lagu-lit-T03", (0.689, 0.25))):
        P1, _, st1 = S[(key, gate_arm, "all")]
        bw1 = P1["budget_wmean"]
        check(f"{key} {gate_arm} budget minimum", f"{bw1.min():.3f} at {st1[np.argmin(bw1)] / 1e6:.2f}M",
              f"{claims[0]} at {claims[1]}M", abs(bw1.min() - claims[0]) < 0.0006 and st1[np.argmin(bw1)] / 1e6 == claims[1])
        if key == "E1b":
            check("E1b lagu-lit budget min over rows >= 0.4M / >= 0.6M / at 1M",
                  f"{bw1[st1 >= 4e5].min():.3f} / {bw1[st1 >= 6e5].min():.3f} / {bw1[-1]:.3f}",
                  ">= 0.95 from 0.4M; 'about 0.99 after 0.6M'; 0.990", bw1[st1 >= 4e5].min() >= 0.95 and
                  abs(bw1[-1] - 0.990) < 0.0006)
        else:
            check("X1 lagu-lit-T03 budget at 0.5M / 1M", f"{at(bw1, st1, 5e5):.3f} / {bw1[-1]:.3f}", "0.866 / 0.945",
                  abs(at(bw1, st1, 5e5) - 0.866) < 0.0006 and abs(bw1[-1] - 0.945) < 0.0006)
        arms = [a for (k, a, nm) in S if k == key and nm == "all"]
        for a in arms:
            Pa, na, _ = S[(key, a, "all")]
            r_, c_ = Pa["eval_return"][1], Pa["eval_cost"][1]
            grid = (st1 % 250_000 == 0)
            check(f"{key} {a}: median return range all rows | 250k grid",
                  f"{r_.min():.2f}-{r_.max():.2f} | {r_[grid].min():.2f}-{r_[grid].max():.2f}",
                  "E1b 26.6-28.1 / X1 26.6-27.8", "range holds on the 250k digest grid only; "
                  f"{int(np.sum(r_ < 26.5))} of {len(st1)} rows below 26.5")
            check(f"{key} {a}: median cost range all rows | 250k grid (x limit 25, all rows)",
                  f"{c_.min():.2f}-{c_.max():.2f} | {c_[grid].min():.2f}-{c_[grid].max():.2f} "
                  f"({c_.min() / LIMIT:.2f}-{c_.max() / LIMIT:.2f} x limit)",
                  "E1b 'about 45-56'; X1 'about twice the limit throughout'",
                  "E1b range holds on the 250k grid; all rows wider" if key == "E1b" else
                  "'about twice' = 1.6-2.4x the limit")
            check(f"{key} {a}: run-rows with eval_return < 5", f"{int(Pa['collapsed'].sum())}", "none",
                  int(Pa["collapsed"].sum()) == 0)
    Px = {a: S[("X1", a, "all")][0] for a in ("lagu-lit-T03", "lagu_nogate-lit-match", "lagu_nogate-lit")}
    got = " ".join(f"{Px[a]['eval_cost'][1][-1]:.2f}" for a in Px)
    check("X1 median cost at 1M (T03 / match / ungated)", got, "56.75 56.50 49.85", got == "56.75 56.50 49.85")
    # E1b Q_c overlap
    Pg, Pu = S[("E1b", "lagu-lit", "all")][0], S[("E1b", "lagu_nogate-lit", "all")][0]
    st1 = S[("E1b", "lagu-lit", "all")][2]
    sep = (Pg["qc_avg"][2] < Pu["qc_avg"][0])
    check("E1b 'curves overlap': rows where lagu-lit Q_c IQR lies wholly below nogate's",
          f"{int(sep.sum())} rows: {(st1[sep] / 1e6).tolist()}M", "two arms' curves overlap",
          "holds for return/cost/lambda/qstd; Q_c separates early" if sep.any() else True)
    w = max(len(c[0]) for c in CHECKS)
    for lab, mine, claim, ok in CHECKS:
        status = ("OK" if ok is True else "FAIL" if ok is False else "NOTE")
        print(f"  [{status:4s}] {lab}")
        print(f"         recomputed: {mine}")
        print(f"         their claim: {claim}" + (f"   ({ok})" if isinstance(ok, str) else ""))
    print()


# ----------------------------------------------------------------------------------------------
def section_F():
    print("F. FIGURE CHECK: marker offsets under their staggering rule (markevery=(j*max(1,every//k), every))")
    for key, (arms, N) in EXPS.items():
        n_rows = len(steps_of(N))
        every = max(1, n_rows // 8)
        k = len(arms)
        idx = []
        for j in range(k):
            off = j * max(1, every // k)
            idx.append(set(range(off, n_rows, every)))
        coll = [(arms[a][1], arms[b][1], len(idx[a] & idx[b])) for a in range(k) for b in range(a + 1, k)
                if idx[a] & idx[b]]
        print(f"  {key}: every={every}, offsets={[j * max(1, every // k) for j in range(k)]}; "
              f"coinciding marker positions: {coll or 'none'}")
    print("  (where markers coincide and the lines are equal, as in X1 panel (c), the earlier-drawn arm is hidden)")
    print()


# ----------------------------------------------------------------------------------------------
def png_equal(a, b):
    import matplotlib.image as mpimg
    A, B = mpimg.imread(a), mpimg.imread(b)
    return A.shape == B.shape and bool(np.array_equal(A, B))


def pdf_equal_ignoring_dates(a, b):
    strip = lambda p: re.sub(rb"/CreationDate \([^)]*\)", b"", open(p, "rb").read())
    return strip(a) == strip(b)


def section_G():
    print("G. THEIR make.py RE-RUN TWICE FROM A SCRATCH COPY (outputs to scratch only)")
    outs = []
    for k in (1, 2):
        d = os.path.join(SCRATCH, f"rerun{k}")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        src = open(os.path.join(THEIRS, "make.py")).read()
        patched = src.replace('ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))', f'ROOT = "{ROOT}"')
        assert patched != src, "ROOT line not found"
        with open(os.path.join(d, "make.py"), "w") as fh:
            fh.write(patched)
        t0 = time.time()
        p = subprocess.run([PY, os.path.join(d, "make.py")], cwd=ROOT, capture_output=True, text=True)
        dt = time.time() - t0
        assert p.returncode == 0, p.stderr
        with open(os.path.join(d, "stdout.txt"), "w") as fh:
            fh.write(p.stdout)
        outs.append((d, p.stdout, dt))
    (d1, o1, t1), (d2, o2, t2) = outs
    print(f"  run times {t1:.1f}s and {t2:.1f}s; stdout identical across runs: {o1 == o2}")
    names = sorted(x for x in os.listdir(THEIRS) if x != "make.py")
    for nm in names:
        a, b1, b2 = os.path.join(THEIRS, nm), os.path.join(d1, nm), os.path.join(d2, nm)
        if not os.path.exists(b1):
            print(f"  {nm:22s} NOT PRODUCED by the re-run")
            continue
        if nm.endswith(".png"):
            same, rep = png_equal(a, b1), png_equal(b1, b2)
            how = "pixel-identical"
        elif nm.endswith(".pdf"):
            same, rep = pdf_equal_ignoring_dates(a, b1), pdf_equal_ignoring_dates(b1, b2)
            how = "identical except /CreationDate"
        else:
            same, rep = filecmp.cmp(a, b1, shallow=False), filecmp.cmp(b1, b2, shallow=False)
            how = "byte-identical"
        print(f"  {nm:22s} {how} to results/paper/timecourse: {same}; across the two re-runs: {rep}")
    extra = sorted(set(os.listdir(d1)) - set(names) - {"make.py", "stdout.txt"})
    print(f"  files the re-run makes that are not in results/paper/timecourse: {extra or 'none'}")
    hz = open(os.path.join(THEIRS, "horizon_table.txt")).read()
    print(f"  horizon_table.txt appears verbatim in their stdout: {hz.strip() in o1}")
    for word in ("EXPLORATORY", "DESCRIPTIVE", "confirmatory"):
        print(f"  '{word}' occurrences in their stdout: {o1.count(word)}")
    print()
    return o1


# ----------------------------------------------------------------------------------------------
def figures(S, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    BLUE, VERM, GREEN, INK = "#0072B2", "#D55E00", "#009E73", "#222222"
    col = {"lagu-tt": BLUE, "lagu_nogate-tt": VERM, "lagu_nogate-tt-match": GREEN}

    # overlay: own recomputation (lines) vs their CSV (markers)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), facecolor="white")
    spec = [("E1", "eval_return", "median", "(a) E1 median eval return", "return per episode"),
            ("E1", "collapsed_count", "median", "(b) E1 runs with eval return < 5", "number of runs (of 15)"),
            ("E1", "budget", "npi_weighted_mean", "(c) E1 mean budget, n_pi-weighted", r"$\delta$ / 2.5"),
            ("E2", "eval_return", "median", "(d) E2 median eval return (all seeds)", "return per episode"),
            ("E2", "collapsed_count", "median", "(e) E2 runs with eval return < 5", "number of runs"),
            ("E2", "qstd_avg", "median", r"(f) E2 median ensemble std $\sqrt{Q_{eu}}$", "discounted return")]
    for ax, (key, metric, fld, title, ylab) in zip(axes.ravel(), spec):
        for arm, c in col.items():
            if ("E2" if key == "E2" else "E1", arm, "all") not in S:
                continue
            P, n, st = S[(key, arm, "all")]
            y = (P["collapsed"] if metric == "collapsed_count" else P["budget_wmean"] if metric == "budget"
                 else P[metric][1])
            ax.plot(st / 1e6, y, color=c, lw=2, label=f"{arm}, own recomputation")
            ty = [f(T[(key, arm, "all", int(t), metric)][fld]) for t in st]
            ax.plot(st / 1e6, ty, ls="none", marker="o", mfc="white", mec=INK, ms=4.5, mew=0.9)
        if metric == "collapsed_count":
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if metric == "budget":
            ax.axhline(D_E2 / DELTA0, color=INK, ls="-.", lw=1)
            ax.axhspan(0.76, 0.88, color="#999999", alpha=0.25, lw=0)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlabel("environment steps (millions)", fontsize=9)
        ax.grid(True, color="#dddddd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[1, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([], [], ls="none", marker="o", mfc="white", mec=INK, ms=5))
    l.append("value in their timecourse_data.csv")
    h.append(Line2D([], [], color="#999999", lw=6, alpha=0.5))
    l.append("(c) 0.76-0.88, paper's dose as we read Zhang et al. Fig. 5(d); dash-dot: 1.12158 / 2.5")
    fig.text(0.01, 0.012, "Seeds: E1 panels s100-114 for both arms (n = 15); E2 panels lagu-tt and "
             "lagu_nogate-tt-match s100-119 (n = 20), lagu_nogate-tt s100-114 (n = 15).", fontsize=8.5)
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("DESCRIPTIVE verification: own recomputation from raw CSVs (lines) vs their plotted values "
                 "(open circles); SafetyPointGoal1-v0, in-training eval rows every 50k", fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"verify_overlay.{ext}"), dpi=170, facecolor="white")
    plt.close(fig)

    # per-seed raster of eval_return < 5
    panels = [("E1 / E2: lagu-tt (gate on, T = 0.07)", "gate", "lagu-tt", S20),
              ("E1 / E2: lagu_nogate-tt (gate off, $\\delta_0$ = 2.5)", "gate", "lagu_nogate-tt", S15),
              ("E2: lagu_nogate-tt-match (gate off, $\\delta_0$ = 1.12158)", "gate", "lagu_nogate-tt-match", S20)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="white", sharey=True)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#eeeeee", VERM])
    for ax, (title, d, arm, seeds) in zip(axes, panels):
        rs = runs(d, arm, seeds)
        m = np.vstack([~(r["eval_return"] >= COLLAPSE) for r in rs]).astype(float)
        st = rs[0]["step"]
        ax.imshow(m, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                  extent=(st[0] / 1e6 - 0.025, st[-1] / 1e6 + 0.025, len(rs) - 0.5, -0.5))
        ax.set_yticks(range(len(rs)))
        ax.set_yticklabels([f"s{r['seed']}" + (" (ext.)" if r["seed"] in EXT else "") for r in rs], fontsize=7.5)
        if any(r["seed"] in EXT for r in rs):
            ax.axhline(14.5, color=INK, lw=1.2)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("environment steps (millions)", fontsize=9)
    axes[0].set_ylabel("seed", fontsize=9)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=VERM), Patch(color="#eeeeee", ec="#999999")],
               ["eval_return < 5 at that row (10 in-training episodes)", "eval_return >= 5"],
               loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("DESCRIPTIVE: per-seed collapse raster, SafetyPointGoal1-v0, every 50k row (no run excluded; "
                 "lagu_nogate-tt has no s115-119)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"verify_collapse_raster.{ext}"), dpi=170, facecolor="white")
    plt.close(fig)
    print("Figures: verify_overlay.{pdf,png}, verify_collapse_raster.{pdf,png} (DESCRIPTIVE)")


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)

        def flush(self):
            sys.__stdout__.flush()

    with redirect_stdout(Tee()):
        print("DESCRIPTIVE / EXPLORATORY verification of results/paper/timecourse. Not confirmatory.\n")
        section_A()
        S, T = section_B()
        section_C()
        section_D()
        section_E(S)
        section_F()
        section_G()
        figures(S, T)
    with open(os.path.join(HERE, "verify_output.txt"), "w") as fh:
        fh.write(buf.getvalue())


if __name__ == "__main__":
    main()
