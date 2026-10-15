"""EXPLORATORY (not confirmatory; outside the Holm family): does the Lag-U gate's uncertainty
signal anticipate cost?

The metric below was fixed in the task statement before any rollout or outcome was read.

Data. The FINAL snapshot (step N) of each run plays deterministic episodes (agent.act(explore=False))
on the held-out layouts EVAL_SEED + 1000 + k, k = 0..9 (EVAL_SEED = 10000), full horizon 1000:
  lagu-tt, lagu_nogate-tt           results/gate            seeds 100-114  (E1, 2M steps)
  lagu-lit, lagu_nogate-lit         results/gate_literal    seeds 100-109  (E1b, 1M steps)
  lagu-lit-T03                      results/gate_literal_x  seeds 100-109  (X1, 1M steps)
No run is excluded (lagu_nogate-tt s103 stays in).

Per step t (state s_t, action pi(s_t)), from the snapshot's actor, ensemble and cost critic:
  Q_bar      ensemble mean of the members' first heads at (s_t, pi(s_t))           (Eq. 10)
  sqrt(Q_eu) sqrt(population variance over members + 1e-12), as in agent.py       (Eq. 11)
  ratio      sqrt(Q_eu) / (|Q_bar| + ratio_eps), the gate's quantity exactly as agent.py computes it (Eq. 16)
  Q_c        learned cost critic at (s_t, pi(s_t)) (positive reference)
  c_t        realised cost returned by env.step(pi(s_t))
Target y_k(t) = 1 if c_j > 0 for some j in t+1..t+k, k in {10, 50}; steps t > n - 1 - k (no full
k-step future) are excluded. Predictors where higher should mean danger: ratio, sqrt(Q_eu), Q_c;
-Q_bar is also reported.

Statistics (numpy only).
  * Pooled AUROC per arm, predictor and k: all usable steps of all runs of the arm, rank statistic
    with tied values given average ranks.
  * 95% seed-cluster bootstrap: resample runs with replacement, 10,000 resamples, percentile
    interval, numpy.random.default_rng(0). Computed exactly through the run-by-run pair-count matrix
    U[r, s] = #(positive of run r > negative of run s) + 0.5 #(ties): the pooled AUROC of a resample
    with run multiplicities w is w'Uw / ((w.P)(w.N)), identical to re-ranking the duplicated data.
  * Per-run AUROC (one run's 10 episodes), median over the runs where it is defined.
  * Difference AUROC(predictor) - AUROC(Q_c), paired over the same resamples (the pre-stated
    comparison of ratio / sqrt(Q_eu) with Q_c).
  * Fraction of usable steps with ratio >= T, T in {0.03, 0.07}; rate of y_k among those steps vs
    the others and the relative risk, with the same cluster bootstrap.
Pre-stated interpretation: AUROC near 0.5 means the signal does not anticipate cost.

Run from the repository root (rollouts are cached in traces/, keyed by the snapshot's SHA-256):
    .venv/bin/python results/paper/uncertainty_anticipation/make.py [--workers 3]
Outputs (next to this file): auroc_table.txt (this printout), auroc_table.csv, threshold_table.csv,
fig_auroc.pdf / fig_auroc.png.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ENV = "SafetyPointGoal1-v0"
ARMS = [  # (label, directory, seeds, short description for the figure)
    ("lagu-tt", "results/gate", range(100, 115), "E1 gate\nT=0.07"),
    ("lagu_nogate-tt", "results/gate", range(100, 115), "E1 no gate"),
    ("lagu-lit", "results/gate_literal", range(100, 110), "E1b gate\nT=0.07"),
    ("lagu_nogate-lit", "results/gate_literal", range(100, 110), "E1b no gate"),
    ("lagu-lit-T03", "results/gate_literal_x", range(100, 110), "X1 gate\nT=0.03"),
]
EPISODES = 10
KS = (10, 50)
THRESHOLDS = (0.03, 0.07)
PREDICTORS = [("ratio", "ratio = sqrt(Q_eu)/|Q_bar|"), ("std", "sqrt(Q_eu)"),
              ("qc", "Q_c"), ("negqbar", "-Q_bar")]
B = 10_000
TRACES = os.path.join(HERE, "traces")


def stem_of(label: str, d: str, seed: int) -> str:
    return os.path.join(ROOT, d, f"{label}_{ENV}_s{seed}")


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ------------------------------------------------------------------------------------ rollouts
def rollout(job: tuple) -> str:
    """Play EPISODES held-out episodes with the run's final snapshot; save per-step values."""
    import safety_gymnasium
    import torch
    from lagu.evaluate import HELDOUT_SEED, load_agent

    torch.set_num_threads(1)
    label, d, seed = job
    stem = stem_of(label, d, seed)
    with open(stem + ".json") as f:
        meta = json.load(f)
    n_final = int(meta["args"]["steps"])
    snap_path = os.path.join(stem + ".snap", f"{n_final}.pt")
    digest = sha256(snap_path)
    out = os.path.join(TRACES, os.path.basename(stem) + ".npz")
    if os.path.exists(out):
        with np.load(out) as z:
            if str(z["sha256"]) == digest and int(z["episodes"]) == EPISODES:
                return f"{os.path.basename(stem)}: cached"
    agent = load_agent(meta, torch.load(snap_path, weights_only=True))
    cfg = agent.cfg
    env = safety_gymnasium.make(meta["args"]["env"])
    cols = {k: [] for k in ("ep", "t", "qbar", "std", "ratio", "qc", "cost", "reward")}
    for k in range(EPISODES):
        o, _ = env.reset(seed=HELDOUT_SEED + k)
        obs, cs, rs = [], [], []
        done = False
        while not done:
            obs.append(o)
            o, r, c, term, trunc, _ = env.step(agent.act(o, explore=False))
            cs.append(float(c))
            rs.append(float(r))
            done = term or trunc
        with torch.no_grad():                       # one batched critic call per episode
            ob = torch.as_tensor(np.stack(obs), dtype=torch.float32)
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
        cols["cost"].append(np.asarray(cs, np.float32))
        cols["reward"].append(np.asarray(rs, np.float32))
    env.close()
    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    tmp = out[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, sha256=digest, episodes=EPISODES, step=n_final, T=cfg.T,
                        use_gate=cfg.use_gate, seed_base=HELDOUT_SEED, **arrays)
    os.replace(tmp, out)
    return (f"{os.path.basename(stem)}: return {arrays['reward'].sum() / EPISODES:.1f} "
            f"cost {arrays['cost'].sum() / EPISODES:.1f}")


# ---------------------------------------------------------------------------------- statistics
def avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with tied values given their average rank."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, first, counts = np.unique(xs, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(avg, counts)
    return ranks


def auroc(x: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC via the rank statistic; nan if a class is empty."""
    p = int(y.sum())
    n = len(y) - p
    if p == 0 or n == 0:
        return float("nan")
    r = avg_ranks(x)
    return float((r[y].sum() - p * (p + 1) / 2.0) / (p * n))


def pair_matrix(xs: list, ys: list) -> tuple:
    """U[r, s] = #(pos of run r > neg of run s) + 0.5 #(equal); P[r], N[r] class counts."""
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


def weights(R: int) -> np.ndarray:
    """Run multiplicities of the B cluster-bootstrap resamples (fresh default_rng(0))."""
    idx = np.random.default_rng(0).integers(0, R, (B, R))
    W = np.zeros((B, R))
    np.add.at(W, (np.arange(B)[:, None], idx), 1.0)
    return W


def boot_auroc(U, P, N, W) -> np.ndarray:
    num = np.einsum("br,rs,bs->b", W, U, W)
    den = (W @ P) * (W @ N)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def pct(v: np.ndarray) -> tuple:
    """95% percentile interval. nan (undefined, 0/0) resamples are dropped; +inf (a ratio with a zero
    denominator) is a valid value and stays in, so a bound inside the infinite tail is reported as inf."""
    v = v[~np.isnan(v)]
    if not len(v):
        return (np.nan, np.nan)
    q = np.percentile(np.where(np.isinf(v), np.sign(v) * 1e300, v), [2.5, 97.5])
    return tuple(float(np.sign(x) * np.inf) if abs(x) >= 1e299 else float(x) for x in q)


def targets(cost: np.ndarray, ep: np.ndarray, k: int) -> tuple:
    """(usable mask, y_k) over the concatenated steps of a run's episodes."""
    use = np.zeros(len(cost), bool)
    y = np.zeros(len(cost), bool)
    for e in np.unique(ep):
        ix = np.flatnonzero(ep == e)
        c = cost[ix] > 0
        n = len(ix)
        cs = np.concatenate([[0], np.cumsum(c)])
        t = np.arange(max(n - k, 0))                 # t + k <= n - 1
        use[ix[t]] = True
        y[ix[t]] = (cs[t + k + 1] - cs[t + 1]) > 0   # any cost in t+1..t+k
    return use, y


def load_runs(label: str, d: str, seeds) -> list:
    runs = []
    for s in seeds:
        z = np.load(os.path.join(TRACES, f"{label}_{ENV}_s{s}.npz"))
        r = {k: z[k] for k in ("ep", "t", "qbar", "std", "ratio", "qc", "cost", "reward")}
        r["negqbar"] = -r["qbar"]
        r["seed"] = s
        r["T"] = float(z["T"])
        for k in ("qbar", "std", "ratio", "qc"):
            if not np.all(np.isfinite(r[k])):
                raise SystemExit(f"{label} s{s}: non-finite {k}; the pre-stated metric has no rule for it")
        runs.append(r)
    return runs


# ------------------------------------------------------------------------------------- figure
def figure(res: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # validated categorical slots 1-4
    markers = ["o", "s", "^", "D"]                            # secondary encoding
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": ink,
                         "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.edgecolor": "#c3c2b7"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, facecolor=surface)
    off = np.linspace(-0.27, 0.27, len(PREDICTORS))
    for ax, k in zip(axes, KS):
        ax.set_facecolor(surface)
        ax.axhline(0.5, color=muted, lw=1, ls="--", zorder=1)
        for j, (p, plabel) in enumerate(PREDICTORS):
            xs, ys, lo, hi = [], [], [], []
            for i, (arm, *_r) in enumerate(ARMS):
                c = res[(arm, k, p)]
                xs.append(i + off[j]), ys.append(c["auc"]), lo.append(c["auc"] - c["lo"]), hi.append(c["hi"] - c["auc"])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=markers[j], ms=6.5, color=colors[j], ecolor=colors[j],
                        elinewidth=1.4, capsize=0, mec=surface, mew=1.0, label=plabel, zorder=3)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([f"{a[0]}\n{a[3]}" for a in ARMS], fontsize=7.5)
        ax.set_xlim(-0.5, len(ARMS) - 0.5)
        ax.set_title(f"cost within the next k = {k} steps", fontsize=10, color=ink)
        ax.grid(axis="y", color=grid, lw=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Pooled AUROC (dimensionless)")
    axes[0].set_ylim(0.3, 0.85)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="none", marker=markers[j], ms=6.5, color=colors[j], mec=surface)
         for j in range(len(PREDICTORS))] + [Line2D([], [], color=muted, lw=1, ls="--")]
    l = [pl for _, pl in PREDICTORS] + ["chance (0.5 = no anticipation)"]
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.905))
    fig.suptitle("EXPLORATORY: does the gate's uncertainty signal anticipate cost?\nSafetyPointGoal1, final "
                 "snapshot, 10 held-out episodes per run; points = pooled AUROC, bars = 95% seed-cluster "
                 "bootstrap", fontsize=9.5, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, facecolor=surface)
    plt.close(fig)


# --------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=3, help="at most 3")
    a = ap.parse_args()
    workers = max(1, min(3, a.workers))
    os.makedirs(TRACES, exist_ok=True)

    jobs = [(arm, d, s) for arm, d, seeds, _ in ARMS for s in seeds]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            for msg in pool.imap_unordered(rollout, jobs):
                print("  rollout", msg, flush=True)
    else:
        for j in jobs:
            print("  rollout", rollout(j), flush=True)

    lines = []
    say = lambda s="": (lines.append(s), print(s))
    res, csv_rows, thr_rows = {}, [], []
    say("EXPLORATORY - does the gate's uncertainty signal anticipate cost? (not confirmatory)")
    say(f"SafetyPointGoal1-v0, final snapshot of each run, {EPISODES} deterministic held-out episodes per run "
        f"(layouts 11000-{11000 + EPISODES - 1}).")
    say("y_k(t) = 1 if any cost in steps t+1..t+k. Pooled AUROC with 95% seed-cluster bootstrap (runs "
        f"resampled, {B:,} resamples, rng 0); per-run AUROC median (runs where defined / runs).")
    say()
    say("== EXPLORATORY: context per arm (mean over runs of the 10 episodes)")
    say(f"  {'arm':16s} {'runs':>4s} {'return':>7s} {'cost':>6s} {'steps with c>0':>14s} "
        f"{'y10 rate':>8s} {'y50 rate':>8s}  {'runs with no cost':>17s}")
    data = {}
    for arm, d, seeds, _ in ARMS:
        runs = load_runs(arm, d, seeds)
        data[arm] = runs
        ret = np.mean([r["reward"].sum() / EPISODES for r in runs])
        cst = np.mean([r["cost"].sum() / EPISODES for r in runs])
        cpos = np.mean(np.concatenate([r["cost"] > 0 for r in runs]))
        yr = {}
        for k in KS:
            yy = [targets(r["cost"], r["ep"], k) for r in runs]
            yr[k] = np.concatenate([y[u] for u, y in yy]).mean()
        nocost = sum(int(r["cost"].sum() == 0) for r in runs)
        say(f"  {arm:16s} {len(runs):4d} {ret:7.1f} {cst:6.1f} {cpos:14.3f} {yr[10]:8.3f} {yr[50]:8.3f}  "
            f"{nocost:17d}")
    say()

    for k in KS:
        say(f"== EXPLORATORY: AUROC for cost within the next k = {k} steps")
        say(f"  {'arm':16s} {'predictor':28s} {'pooled AUROC [95% CI]':>24s} {'per-run median (n)':>19s} "
            f"{'minus Q_c [95% CI]':>26s}")
        for arm, d, seeds, _ in ARMS:
            runs = data[arm]
            R = len(runs)
            W = weights(R)
            masks = [targets(r["cost"], r["ep"], k) for r in runs]
            ys = [y[u] for u, y in masks]
            boots = {}
            for p, plabel in PREDICTORS:
                xs = [r[p][u].astype(np.float64) for r, (u, _) in zip(runs, masks)]
                x_all, y_all = np.concatenate(xs), np.concatenate(ys)
                auc = auroc(x_all, y_all)
                U, P, N = pair_matrix(xs, ys)
                assert abs(U.sum() / (P.sum() * N.sum()) - auc) < 1e-9, "pair-count and rank AUROC disagree"
                bt = boot_auroc(U, P, N, W)
                boots[p] = bt
                lo, hi = pct(bt)
                per = np.array([auroc(x, y) for x, y in zip(xs, ys)])
                ok = np.isfinite(per)
                med = float(np.median(per[ok])) if ok.any() else float("nan")
                res[(arm, k, p)] = dict(auc=auc, lo=lo, hi=hi, med=med, n_def=int(ok.sum()),
                                        undefined_boot=int((~np.isfinite(bt)).sum()))
            for p, plabel in PREDICTORS:
                c = res[(arm, k, p)]
                if p == "qc":
                    diff, dlo, dhi = 0.0, np.nan, np.nan
                    dtxt = f"{'(reference)':>26s}"
                else:
                    diff = c["auc"] - res[(arm, k, "qc")]["auc"]
                    dlo, dhi = pct(boots[p] - boots["qc"])
                    dtxt = f"{diff:+8.3f} [{dlo:+.3f}, {dhi:+.3f}]"
                c.update(diff_qc=diff, diff_lo=dlo, diff_hi=dhi)
                say(f"  {arm:16s} {plabel:28s} {c['auc']:7.3f} [{c['lo']:.3f}, {c['hi']:.3f}] "
                    f"{c['med']:11.3f} ({c['n_def']:2d}/{R:2d}) {dtxt}")
                csv_rows.append(dict(analysis="EXPLORATORY", arm=arm, k=k, predictor=p, runs=R,
                                     pooled_auroc=c["auc"], ci_lo=c["lo"], ci_hi=c["hi"],
                                     per_run_median=c["med"], per_run_defined=c["n_def"],
                                     minus_qc=diff, minus_qc_lo=dlo, minus_qc_hi=dhi,
                                     undefined_resamples=c["undefined_boot"]))
            say()
        say()

    say("== EXPLORATORY: gate threshold. Fraction of usable steps with ratio >= T; rate of y_k among them vs "
        "the others; relative risk (RR) with 95% seed-cluster bootstrap")
    say(f"  {'arm':16s} {'T':>5s} {'k':>3s} {'frac >= T':>9s} {'rate >= T':>9s} {'rate < T':>9s} "
        f"{'RR [95% CI]':>24s} {'boot inf':>9s} {'boot 0/0':>9s}")
    for arm, d, seeds, _ in ARMS:
        runs = data[arm]
        W = weights(len(runs))
        for T in THRESHOLDS:
            for k in KS:
                cnt = []
                for r in runs:
                    u, y = targets(r["cost"], r["ep"], k)
                    f = r["ratio"][u] >= T
                    yy = y[u]
                    cnt.append((f.sum(), (f & yy).sum(), (~f).sum(), (~f & yy).sum()))
                cnt = np.array(cnt, float)
                nf, pf, nn, pn = cnt.sum(0)
                frac = nf / (nf + nn)
                bn = W @ cnt
                with np.errstate(invalid="ignore", divide="ignore"):
                    rf, rn = pf / nf, pn / nn
                    rr = rf / rn
                    brr = (bn[:, 1] / bn[:, 0]) / (bn[:, 3] / bn[:, 2])
                lo, hi = pct(brr)
                n_inf, n_nan = int(np.isinf(brr).sum()), int(np.isnan(brr).sum())
                say(f"  {arm:16s} {T:5.2f} {k:3d} {frac:9.3f} {rf:9.3f} {rn:9.3f} "
                    f"{rr:8.2f} [{lo:6.2f}, {hi:6.2f}] {n_inf:9d} {n_nan:9d}")
                thr_rows.append(dict(analysis="EXPLORATORY", arm=arm, T=T, k=k, frac_flagged=frac,
                                     rate_flagged=rf, rate_unflagged=rn, rel_risk=rr, rr_lo=lo, rr_hi=hi,
                                     n_flagged=int(nf), n_unflagged=int(nn), resamples_inf=n_inf,
                                     resamples_undefined=n_nan))
    say()
    say("Notes: ratio is computed for the ungated arms too (the gate's would-be quantity). An undefined "
        "bootstrap resample (AUROC: no positive or no negative step; RR: 0/0) is dropped from its interval "
        "(counts in auroc_table.csv and in the 'boot 0/0' column); an RR resample with no cost after any "
        "unflagged step is +inf and stays in ('boot inf').")

    with open(os.path.join(HERE, "auroc_table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for name, rows in (("auroc_table.csv", csv_rows), ("threshold_table.csv", thr_rows)):
        with open(os.path.join(HERE, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    figure(res, os.path.join(HERE, "fig_auroc"))
    print("wrote auroc_table.txt, auroc_table.csv, threshold_table.csv, fig_auroc.pdf, fig_auroc.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
