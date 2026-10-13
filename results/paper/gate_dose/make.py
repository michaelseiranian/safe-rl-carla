"""EXPLORATORY / DESCRIPTIVE analysis: the Lag-U gate's actual dose. Never confirmatory.

Outside the Holm family. Under prereg section 16 item 5, every mechanism analysis is exploratory
and is reported whichever way it comes out. The windows, seeds, thresholds and snapshot steps below
were fixed by the task statement before any output of this script was read. No run is excluded.

Run from the repository root (one script makes every output in this directory):

    .venv/bin/python results/paper/gate_dose/make.py [--fresh] [--workers 3]

PART 1 (DESCRIPTIVE, from logged interval columns; no new computation on networks)
    For every gated arm, over the windows 25k-300k, 300k-1M and 1M-2M (rows with lo < step <= hi;
    the row at step t covers the updates in (t - 50k, t]) and over the whole run:
      engagement   = sum(engagement * n_risk) / sum(n_risk), pooled over seeds (prereg section 5);
      budget (all) = sum(delta_avg * n_pi) / sum(n_pi) / delta0, pooled over seeds, rows with
                     n_pi > 0. This is the mean delta over every policy update; exploratory updates
                     (Eq. 15) always use delta0, so before Tc = 300k it is diluted towards 1;
      budget (risk)= sum(delta_risk_avg * n_risk) / sum(n_risk) / delta0: the mean delta_ada over
                     risk-sensitive updates only (Eq. 16-17).
    Intervals: 95% percentile bootstrap over seeds (10,000 resamples of whole runs; the statistic is
    recomputed as the pooled ratio; numpy.random.default_rng(0) fresh for each interval).
    Arms (as listed in the task): lagu-tt E1 (PointGoal1, seeds 100-114), lagu-lit E1b (100-109),
    lagu-lit-T03 X1 (100-109), lagu-tt on SafetyCarGoal1 and on SafetyPointCircle1 E3 (100-109;
    PointCircle1 delta0 = 5.0), lagu-tt-T035 and lagu-tt-T14 E4 (100-107). Extra rows, marked as
    such and not in the task's list: lagu-tt on seeds 100-119 (all lagu-tt seeds, adds the H4
    extension seeds) and on 100-107 (the E4 seed set), and lagu-tt-ggrad E2 (100-109, gate_grad = 1;
    an extension, not the paper's method). delta0 is read from each run's json.
    Figure fig_budget_vs_step: per-row pooled budget vs step, both definitions, with the reference
    band 0.76-0.88 delta0 labelled "as we read Zhang et al. Fig. 5(d)".

PART 2 (EXPLORATORY): offline signed-ratio counterfactual.
    The port computes ratio = sqrt(Q_eu) / (|Q_bar| + 1e-6) (lagu/agent.py). Eq. 16 is written with
    the signed Q_bar: the gate tightens only when the signed ratio is >= T, which never happens when
    Q_bar <= 0. On the UNGATED critics (lagu_nogate-tt PointGoal1 seeds 100-114, snapshots 600k,
    1.2M and 2M; lagu_nogate-lit seeds 100-109, snapshots 600k and 1M), each snapshot plays 5
    deterministic episodes on the held-out layouts HELDOUT_SEED + k, k = 0..4 (HELDOUT_SEED = EVAL_SEED
    + 1000 = 11000, as in lagu/evaluate.py; the same 5 layouts for every snapshot and seed). At every
    visited state s, with a = pi(s), we take Q_bar (Eq. 10) and sqrt(Q_eu) (Eq. 11) from the
    snapshot's networks and compute, for T in {0.07, 0.03}:
      |Q_bar| gate:  engaged iff sqrt(Q_eu) / (|Q_bar| + eps) >= T            (the port)
      signed gate:   engaged iff Q_bar + eps > 0 and sqrt(Q_eu) / (Q_bar + eps) >= T   (Eq. 16 as written)
      budget delta_ada / delta0 = 1 if not engaged, else T / ratio (Eq. 16).
    Engagement = fraction of states engaged; mean budget = mean of delta_ada / delta0 over states.
    Per snapshot both are means over all visited states (5 x 1000); the arm value is the mean over
    seeds (equal state counts) with a 95% bootstrap interval over seeds (as in part 1). Because the
    signed gate is off wherever Q_bar <= 0, its arm mean mixes seeds at exactly delta0 with seeds
    that are tightened; the table counts the seeds on which it never fires and the figure shows
    every seed as a dot.
    These are on-policy states of the deterministic policy, not the replay batches the gate sees in
    training; the logged would_engage of the same ungated run (replay batches, |Q_bar|, T = 0.07,
    interval ending at the snapshot step) is printed beside it as a cross-check.
    Gated-arm critics are NOT used: the gate itself shifted them (through lambda and the actor),
    so they would not show what the gate would do to an unshifted learner.
    Rollouts are cached in cache/ keyed by the snapshot's SHA-256; --fresh recomputes them.

Outputs (this directory): tables.txt (the printout), part1_dose_windows.csv,
part2_signed_counterfactual.csv, part2_per_seed.csv, fig_budget_vs_step.{pdf,png},
fig_signed_counterfactual.{pdf,png}.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing as mp
import os
import sys
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

N_BOOT = 10_000
WINDOWS = [("25k-300k", 25_000, 300_000), ("300k-1M", 300_000, 1_000_000),
           ("1M-2M", 1_000_000, 2_000_000), ("whole run", 0, 10**12)]
REF_BAND = (0.76, 0.88)          # mean budget / delta0, as we read Zhang et al. Fig. 5(d)
PG = "SafetyPointGoal1-v0"

# (key, label, directory, tag, env, seeds, extra?)
P1_ARMS = [
    ("E1",      "lagu-tt T=0.07 (E1)",                 "results/gate",           "lagu-tt",      PG, range(100, 115), False),
    ("E1all",   "lagu-tt T=0.07, s100-119 [extra]",    "results/gate",           "lagu-tt",      PG, range(100, 120), True),
    ("E1e4",    "lagu-tt T=0.07, s100-107 [extra]",    "results/gate",           "lagu-tt",      PG, range(100, 108), True),
    ("T035",    "lagu-tt-T035 T=0.035 (E4)",           "results/gate",           "lagu-tt-T035", PG, range(100, 108), False),
    ("T14",     "lagu-tt-T14 T=0.14 (E4)",             "results/gate",           "lagu-tt-T14",  PG, range(100, 108), False),
    ("ggrad",   "lagu-tt-ggrad gate_grad=1 [extra]",   "results/gate",           "lagu-tt-ggrad", PG, range(100, 110), True),
    ("lit",     "lagu-lit T=0.07 (E1b)",               "results/gate_literal",   "lagu-lit",     PG, range(100, 110), False),
    ("litT03",  "lagu-lit-T03 T=0.03 (X1)",            "results/gate_literal_x", "lagu-lit-T03", PG, range(100, 110), False),
    ("car",     "lagu-tt CarGoal1 (E3)",               "results/gate_x",         "lagu-tt",      "SafetyCarGoal1-v0", range(100, 110), False),
    ("circle",  "lagu-tt PointCircle1 (E3)",           "results/gate_x",         "lagu-tt",      "SafetyPointCircle1-v0", range(100, 110), False),
]

# (key, label, directory, tag, seeds, snapshot steps)
P2_ARMS = [
    ("tt",  "lagu_nogate-tt (E1)",   "results/gate",         "lagu_nogate-tt",  range(100, 115), (600_000, 1_200_000, 2_000_000)),
    ("lit", "lagu_nogate-lit (E1b)", "results/gate_literal", "lagu_nogate-lit", range(100, 110), (600_000, 1_000_000)),
]
P2_T = (0.07, 0.03)
P2_EPISODES = 5
RATIO_EPS = 1e-6                 # Config.ratio_eps in lagu/agent.py

# colour-blind-safe categorical slots (validated: adjacent CVD dE >= 9.1, normal-vision >= 22.9);
# low-contrast slots are always paired with a line style and a direct label
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_VIOLET, C_MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ----------------------------------------------------------------------------- helpers
def stem(d, tag, env, s):
    return os.path.join(d, f"{tag}_{env}_s{s}")


def read_csv(path):
    rows = list(csv.DictReader(open(path)))
    cols = {}
    for k in rows[0]:
        cols[k] = np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    return cols


def boot_ratio(num, den):
    """95% percentile interval of sum(num[idx]) / sum(den[idx]) over 10,000 resamples of seeds."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(0).integers(0, len(num), (N_BOOT, len(num)))
    stat = num[idx].sum(-1) / den[idx].sum(-1)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def wsums(x, w, mask):
    ok = mask & np.isfinite(x) & (w > 0)
    return float((x[ok] * w[ok]).sum()), float(w[ok].sum())


# ----------------------------------------------------------------------------- part 1
def load_part1():
    data = {}
    for key, label, d, tag, env, seeds, extra in P1_ARMS:
        runs = []
        for s in seeds:
            st = stem(d, tag, env, s)
            meta = json.load(open(st + ".json"))
            assert meta["cfg"]["use_gate"], st
            c = read_csv(st + ".csv")
            runs.append({"seed": s, "delta0": float(meta["cfg"]["delta0"]), "T": float(meta["cfg"]["T"]),
                         "N": int(meta["args"]["steps"]), "c": c})
        data[key] = runs
    return data


def part1_table(data):
    out = []
    for key, label, d, tag, env, seeds, extra in P1_ARMS:
        runs = data[key]
        N = runs[0]["N"]
        for wname, lo, hi in WINDOWS:
            if lo >= N:
                continue
            sums = {"eng": [], "all": [], "risk": []}
            for r in runs:
                c, d0 = r["c"], r["delta0"]
                m = (c["step"] > lo) & (c["step"] <= hi)
                sums["eng"].append(wsums(c["engagement"], c["n_risk"], m))
                sums["all"].append(wsums(c["delta_avg"] / d0, c["n_pi"], m))
                sums["risk"].append(wsums(c["delta_risk_avg"] / d0, c["n_risk"], m))
            row = {"key": key, "arm": label, "dir": d, "tag": tag, "env": env, "extra": extra,
                   "seeds": f"{min(seeds)}-{max(seeds)}", "n_seeds": len(runs), "T": runs[0]["T"],
                   "delta0": runs[0]["delta0"], "window": wname}
            for k, v in sums.items():
                v = np.array(v)
                row[k] = float(v[:, 0].sum() / v[:, 1].sum())
                row[k + "_lo"], row[k + "_hi"] = boot_ratio(v[:, 0], v[:, 1])
            out.append(row)
    return out


def print_part1(rows):
    print("=" * 118)
    print("TABLE 1 (DESCRIPTIVE). Gate dose of every gated arm, from the logged interval columns.")
    print("engagement = sum(engagement*n_risk)/sum(n_risk); budget(all) = sum(delta_avg*n_pi)/sum(n_pi)/delta0;")
    print("budget(risk) = sum(delta_risk_avg*n_risk)/sum(n_risk)/delta0; pooled over seeds; [95% bootstrap over seeds].")
    print(f"Window lo-hi = rows with lo < step <= hi. Reference: {REF_BAND[0]}-{REF_BAND[1]} delta0 as we read Zhang et al. "
          "Fig. 5(d). [extra] = not in the task's list.")
    print("=" * 118)
    print(f"{'arm':36s} {'seeds':8s} {'d0':>4s} {'window':10s} {'engagement':>22s} {'budget(all)/d0':>22s} "
          f"{'budget(risk)/d0':>22s}")
    last = None
    for r in rows:
        if last is not None and r["key"] != last:
            print("-" * 118)
        last = r["key"]
        f = lambda k: f"{r[k]:.3f} [{r[k + '_lo']:.3f},{r[k + '_hi']:.3f}]"
        print(f"{r['arm']:36s} {r['seeds']:8s} {r['delta0']:4.1f} {r['window']:10s} {f('eng'):>22s} "
              f"{f('all'):>22s} {f('risk'):>22s}")
    print()


def part1_curves(runs, col, wcol):
    """Per-row pooled value over seeds and its 95% bootstrap band (resampling seeds)."""
    step = runs[0]["c"]["step"]
    for r in runs:
        assert np.array_equal(r["c"]["step"], step)
    x = np.stack([r["c"][col] / r["delta0"] for r in runs])
    w = np.stack([r["c"][wcol] for r in runs])
    ok = np.isfinite(x) & (w > 0)
    num, den = np.where(ok, x * w, 0.0), np.where(ok, w, 0.0)
    mean = num.sum(0) / np.maximum(den.sum(0), 1e-300)
    idx = np.random.default_rng(0).integers(0, len(runs), (N_BOOT, len(runs)))
    bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-300)
    return step, mean, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


# ----------------------------------------------------------------------------- part 2
def _rollout(job):
    st, step, cache = job
    import torch
    import safety_gymnasium
    from lagu.evaluate import HELDOUT_SEED, digest, load_agent
    torch.set_num_threads(1)
    path = os.path.join(st + ".snap", f"{step}.pt")
    sha = digest(path)
    layouts = np.array([HELDOUT_SEED + k for k in range(P2_EPISODES)])
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if str(z["sha256"]) == sha and np.array_equal(z["layouts"], layouts):
            return st, step, z["qbar"], z["std"], z["ep_return"], z["ep_cost"], "cache"
    meta = json.load(open(st + ".json"))
    env = safety_gymnasium.make(meta["args"]["env"])
    agent = load_agent(meta, torch.load(path, weights_only=True))
    qb, sd, er, ec = [], [], [], []
    for seed in layouts:
        o, _ = env.reset(seed=int(seed))
        obs, ret, cost, done = [o], 0.0, 0.0, False
        while not done:
            o, r, c, term, trunc, _ = env.step(agent.act(o, explore=False))
            ret, cost = ret + float(r), cost + float(c)
            done = term or trunc
            if not done:
                obs.append(o)
        with torch.no_grad():                         # Q_bar (Eq. 10), sqrt(Q_eu) (Eq. 11) at (s, pi(s))
            ot = torch.as_tensor(np.stack(obs), dtype=torch.float32)
            q, var = agent.ensemble(ot, agent.actor(ot))
        qb.append(q.numpy().astype(np.float64))
        sd.append(torch.sqrt(var + 1e-12).numpy().astype(np.float64))
        er.append(ret)
        ec.append(cost)
    env.close()
    qb, sd = np.concatenate(qb), np.concatenate(sd)
    er, ec = np.array(er), np.array(ec)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + ".tmp.npz"
        np.savez_compressed(tmp, qbar=qb, std=sd, ep_return=er, ep_cost=ec, sha256=sha, layouts=layouts)
        os.replace(tmp, cache)
    return st, step, qb, sd, er, ec, "computed"


def gate_stats(qb, sd, T):
    r_abs = sd / (np.abs(qb) + RATIO_EPS)
    den = qb + RATIO_EPS
    pos = den > 0
    r_sgn = np.where(pos, sd / np.where(pos, den, 1.0), -np.inf)
    out = {}
    for name, r in (("abs", r_abs), ("sgn", r_sgn)):
        eng = r >= T
        budget = np.where(eng, T / np.where(eng, r, 1.0), 1.0)
        out[name] = (float(eng.mean()), float(budget.mean()))
    return out


def run_part2(fresh, workers):
    jobs = []
    for key, label, d, tag, seeds, steps in P2_ARMS:
        for s in seeds:
            st = stem(d, tag, PG, s)
            assert not json.load(open(st + ".json"))["cfg"]["use_gate"], st
            for step in steps:
                cache = os.path.join(HERE, "cache", f"{os.path.basename(st)}_{step}.npz")
                if fresh and os.path.exists(cache):
                    os.remove(cache)
                jobs.append((st, step, cache))
    if workers > 1:
        with mp.get_context("spawn").Pool(min(workers, 3)) as pool:
            res = pool.map(_rollout, jobs, chunksize=1)
    else:
        res = list(map(_rollout, jobs))
    n_new = sum(r[-1] == "computed" for r in res)
    per = {(r[0], r[1]): r for r in res}

    seed_rows, summary = [], []
    for key, label, d, tag, seeds, steps in P2_ARMS:
        for step in steps:
            vals = []
            for s in seeds:
                st = stem(d, tag, PG, s)
                _, _, qb, sd, er, ec, _ = per[(st, step)]
                c = read_csv(st + ".csv")
                i = int(np.where(c["step"] == step)[0][0])
                row = {"arm": label, "key": key, "seed": s, "step": step, "states": len(qb),
                       "frac_qbar_le0": float((qb + RATIO_EPS <= 0).mean()), "median_qbar": float(np.median(qb)),
                       "median_ratio_abs": float(np.median(sd / (np.abs(qb) + RATIO_EPS))),
                       "logged_would_engage_T007": float(c["would_engage"][i]),
                       "logged_qbar_avg": float(c["qbar_avg"][i]),
                       "ep_return_mean": float(er.mean()), "ep_cost_mean": float(ec.mean())}
                for T in P2_T:
                    g = gate_stats(qb, sd, T)
                    for name in ("abs", "sgn"):
                        row[f"eng_{name}_T{T}"], row[f"budget_{name}_T{T}"] = g[name]
                vals.append(row)
            seed_rows += vals
            n = np.array([v["states"] for v in vals], float)
            srow = {"arm": label, "key": key, "step": step, "n_seeds": len(vals), "states": int(n.sum())}
            cols = ["frac_qbar_le0", "logged_would_engage_T007"] + [
                f"{m}_{g}_T{T}" for T in P2_T for m in ("eng", "budget") for g in ("abs", "sgn")]
            for k in cols:
                x = np.array([v[k] for v in vals])
                w = n if k != "logged_would_engage_T007" else np.ones_like(n)
                srow[k] = float((x * w).sum() / w.sum())
                srow[k + "_lo"], srow[k + "_hi"] = boot_ratio(x * w, w)
            srow["median_qbar_over_seeds"] = float(np.median([v["median_qbar"] for v in vals]))
            srow["median_ratio_abs_over_seeds"] = float(np.median([v["median_ratio_abs"] for v in vals]))
            for T in P2_T:   # seeds on which the signed gate fires on no visited state (budget exactly delta0)
                srow[f"n_seeds_sgn_never_T{T}"] = int(sum(v[f"eng_sgn_T{T}"] == 0.0 for v in vals))
            srow["per_seed"] = {k: [v[k] for v in vals] for k in cols if k != "logged_would_engage_T007"}
            summary.append(srow)
    return seed_rows, summary, n_new, len(jobs)


def print_part2(summary, n_new, n_jobs):
    print("=" * 118)
    print("TABLE 2 (EXPLORATORY). Offline counterfactual: the gate the UNGATED critics would have, |Q_bar| (port) vs")
    print("signed Q_bar (Eq. 16 as written). On-policy states of the deterministic policy, 5 held-out layouts")
    print(f"(11000-11004) per snapshot, 1000 states each; mean over seeds [95% bootstrap over seeds]. Rollouts: "
          f"{n_jobs - n_new} from cache, {n_new} computed.")
    print("Gated-arm critics are not used: the gate itself shifted them, so they cannot show what the gate would")
    print("do to an unshifted learner. logged w_e = the ungated run's logged would_engage (replay batches, |Q_bar|,")
    print("T = 0.07) over the 50k interval ending at the snapshot.")
    print("=" * 118)
    print("2a. Sign of Q_bar on visited states")
    print(f"{'critics':22s} {'snapshot':>9s} {'seeds':>5s} {'frac Q_bar<=0':>22s} {'med Q_bar':>9s} "
          f"{'med ratio(|Q|)':>14s} {'logged w_e':>22s}")
    for r in summary:
        f = lambda k: f"{r[k]:.3f} [{r[k + '_lo']:.3f},{r[k + '_hi']:.3f}]"
        print(f"{r['arm']:22s} {r['step']:>9d} {r['n_seeds']:>5d} {f('frac_qbar_le0'):>22s} "
              f"{r['median_qbar_over_seeds']:9.3f} {r['median_ratio_abs_over_seeds']:14.3f} "
              f"{f('logged_would_engage_T007'):>22s}")
    for T in P2_T:
        print(f"\n2b. T = {T}: engagement and mean budget delta_ada/delta0, |Q_bar| gate vs signed gate")
        print(f"{'critics':22s} {'snapshot':>9s} {'eng |Q|':>22s} {'eng signed':>22s} {'budget |Q|':>22s} "
              f"{'budget signed':>22s} {'signed never':>12s}")
        for r in summary:
            f = lambda k: f"{r[k]:.3f} [{r[k + '_lo']:.3f},{r[k + '_hi']:.3f}]"
            print(f"{r['arm']:22s} {r['step']:>9d} {f(f'eng_abs_T{T}'):>22s} {f(f'eng_sgn_T{T}'):>22s} "
                  f"{f(f'budget_abs_T{T}'):>22s} {f(f'budget_sgn_T{T}'):>22s} "
                  f"{r[f'n_seeds_sgn_never_T{T}']:>5d} of {r['n_seeds']:<3d}")
    print("signed never = seeds whose signed gate fires on no visited state (Q_bar <= 0 there, or ratio < T), so its")
    print("budget is exactly delta0; per-seed values are in part2_per_seed.csv.")
    print(f"Reference: {REF_BAND[0]}-{REF_BAND[1]} delta0 as we read Zhang et al. Fig. 5(d).")
    print()


# ----------------------------------------------------------------------------- figures
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5,
        "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 1.8, "pdf.fonttype": 42,
    })
    return plt


def fig_budget(data, plt):
    panels = [
        ("PointGoal1, two-timescale recipe (tt), 2M steps",
         [("E1", "T=0.07 (E1, s100-114)", C_BLUE, "-"), ("T035", "T=0.035 (E4)", C_ORANGE, "--"),
          ("T14", "T=0.14 (E4)", C_AQUA, "-."), ("ggrad", "T=0.07, gate_grad=1 (E2 ext.)", C_YELLOW, ":")]),
        ("PointGoal1, literal recipe (lit), 1M steps",
         [("lit", "T=0.07 (E1b)", C_BLUE, "-"), ("litT03", "T=0.03 (X1)", C_ORANGE, "--")]),
        ("Other tasks (E3), tt, 2M steps, T=0.07",
         [("car", "CarGoal1 (delta0 2.5)", C_VIOLET, "-"), ("circle", "PointCircle1 (delta0 5.0)", C_MAGENTA, "--")]),
    ]
    rows = [("delta_avg", "n_pi", "mean budget, all policy updates\n(delta_avg / delta0)"),
            ("delta_risk_avg", "n_risk", "mean budget, risk-sensitive updates\n(delta_risk_avg / delta0)")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharey=True)
    for ri, (col, wcol, ylab) in enumerate(rows):
        for ci, (title, series) in enumerate(panels):
            ax = axes[ri, ci]
            band = ax.axhspan(*REF_BAND, color="#9a9890", alpha=0.25, lw=0, zorder=0)
            ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)
            ax.axvline(0.3, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
            if ri == 0 and ci == 0:
                ax.text(0.33, 1.015, "Tc = 300k: exploration prob. reaches its 0.1 floor", fontsize=7,
                        color=INK2, va="bottom")
            for key, lab, color, ls in series:
                step, m, lo, hi = part1_curves(data[key], col, wcol)
                x = step / 1e6
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
                ax.plot(x, m, color=color, ls=ls, label=lab, marker="o", ms=2.5)
            ax.set_ylim(0, 1.08)
            ax.set_xlim(0, 2.05 if ci != 1 else 1.03)
            if ri == 0:
                ax.set_title(title)
            if ri == 1:
                ax.set_xlabel("training step (millions of environment steps)")
            if ci == 0:
                ax.set_ylabel(ylab)
    fig.suptitle("DESCRIPTIVE: the gate's mean budget delta_ada / delta0 during training (pooled over seeds; "
                 "shading = 95% bootstrap over seeds; each point = the 50k interval ending there)\n"
                 "grey band = 0.76-0.88 delta0, as we read Zhang et al. Fig. 5(d); dotted line = delta0 (no "
                 "tightening); dashed line = Tc = 300k", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ci in range(3):                        # legends under each column, added after the layout is fixed
        h, l = axes[0, ci].get_legend_handles_labels()
        if ci == 0:
            h, l = h + [band], l + ["0.76-0.88: as we read Zhang et al. Fig. 5(d)"]
        axes[1, ci].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2 if ci == 0 else 1,
                           handlelength=2.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_budget_vs_step.{ext}"), dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def fig_counterfactual(summary, plt):
    cats = [(r["key"], r["step"]) for r in summary]
    lab = {"tt": "tt", "lit": "lit"}
    xt = [f"{lab[k]}\n{s / 1e6:g}M" for k, s in cats]
    xs = np.arange(len(cats), dtype=float)
    xs[np.array([k == "lit" for k, _ in cats])] += 0.6           # gap between recipes
    conds = [("abs", 0.07, C_BLUE, "o", "full", "|Q_bar| (port), T=0.07"),
             ("sgn", 0.07, C_ORANGE, "o", "full", "signed Q_bar (Eq. 16), T=0.07"),
             ("abs", 0.03, C_BLUE, "s", "none", "|Q_bar| (port), T=0.03"),
             ("sgn", 0.03, C_ORANGE, "s", "none", "signed Q_bar (Eq. 16), T=0.03")]
    off = np.array([-0.24, -0.08, 0.08, 0.24])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ai, (metric, ylab) in enumerate([("eng", "engagement\n(fraction of states where the gate fires)"),
                                         ("budget", "mean budget delta_ada / delta0")]):
        ax = axes[ai]
        if metric == "budget":
            band = ax.axhspan(*REF_BAND, color="#9a9890", alpha=0.25, lw=0, zorder=0,
                              label="as we read Zhang et al. Fig. 5(d)")
        for (g, T, color, mk, fill, label), o in zip(conds, off):
            k = f"{metric}_{g}_T{T}"
            m = np.array([r[k] for r in summary])
            lo = np.array([r[k + "_lo"] for r in summary])
            hi = np.array([r[k + "_hi"] for r in summary])
            for xi, r in zip(xs + o, summary):                     # one small dot per seed
                ys = np.array(r["per_seed"][k])
                jit = np.linspace(-0.035, 0.035, len(ys))
                ax.plot(xi + jit, ys, ".", color=color, ms=3.5, alpha=0.45, zorder=2)
            ax.errorbar(xs + o, m, yerr=[m - lo, hi - m], fmt=mk, color=color, mfc=color if fill == "full" else SURFACE,
                        mec=color, ms=7, capsize=2.5, lw=1.4, label=label + " (mean, 95% CI)", zorder=3)
        if metric == "eng":
            m = np.array([r["logged_would_engage_T007"] for r in summary])
            ax.plot(xs - 0.36, m, "x", color=INK, ms=7, mew=1.5, zorder=3,
                    label="logged would_engage in training (replay batches, |Q_bar|, T=0.07)")
        ax.set_xticks(xs, xt)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel(ylab)
        ax.set_xlabel("critics (ungated arm) and snapshot step")
    ax = axes[2]
    m = np.array([r["frac_qbar_le0"] for r in summary])
    lo = np.array([r["frac_qbar_le0_lo"] for r in summary])
    hi = np.array([r["frac_qbar_le0_hi"] for r in summary])
    for xi, r in zip(xs, summary):
        ys = np.array(r["per_seed"]["frac_qbar_le0"])
        ax.plot(xi + np.linspace(-0.06, 0.06, len(ys)), ys, ".", color=C_VIOLET, ms=3.5, alpha=0.45, zorder=2)
    ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="D", color=C_VIOLET, ms=7, capsize=2.5, lw=1.4, zorder=3)
    ax.set_xticks(xs, xt)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("fraction of visited states with Q_bar <= 0\n(signed gate cannot fire there)")
    ax.set_xlabel("critics (ungated arm) and snapshot step")
    for a, t in zip(axes, ("(a) engagement", "(b) mean budget", "(c) sign of Q_bar")):
        a.set_title(t, loc="left")
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    extra = [(h, l) for h, l in zip(h2, l2) if l not in l1]
    fig.legend(h1 + [h for h, _ in extra], l1 + ["0.76-0.88: as we read Zhang et al. Fig. 5(d)" for _ in extra],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, handlelength=2.4)
    fig.suptitle("EXPLORATORY: the gate the ungated critics would have, |Q_bar| (port) vs signed Q_bar (Eq. 16 as "
                 "written)\n5 held-out layouts x 1000 on-policy states per snapshot; tt critics s100-114, lit critics "
                 "s100-109; large markers = mean over seeds with 95% bootstrap interval, dots = single seeds",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_signed_counterfactual.{ext}"), dpi=200, bbox_inches="tight",
                    pad_inches=0.1)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def write_csv(path, rows):
    keys = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fresh", action="store_true", help="recompute every rollout (ignore cache/)")
    p.add_argument("--workers", type=int, default=3, help="processes for the rollouts (at most 3)")
    a = p.parse_args()

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("EXPLORATORY / DESCRIPTIVE: the gate's actual dose (results/paper/gate_dose/make.py). "
              "Not confirmatory; outside the Holm family.\n")
        data = load_part1()
        p1 = part1_table(data)
        print_part1(p1)
        seed_rows, p2, n_new, n_jobs = run_part2(a.fresh, a.workers)
        print_part2(p2, n_new, n_jobs)
    text = buf.getvalue()
    print(text, end="")
    with open(os.path.join(HERE, "tables.txt"), "w") as f:
        f.write(text)
    write_csv(os.path.join(HERE, "part1_dose_windows.csv"), p1)
    write_csv(os.path.join(HERE, "part2_signed_counterfactual.csv"),
              [{k: v for k, v in r.items() if k != "per_seed"} for r in p2])
    write_csv(os.path.join(HERE, "part2_per_seed.csv"), seed_rows)
    plt = style()
    fig_budget(data, plt)
    fig_counterfactual(p2, plt)


if __name__ == "__main__":
    main()
