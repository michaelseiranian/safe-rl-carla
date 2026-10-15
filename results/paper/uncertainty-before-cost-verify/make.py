"""EXPLORATORY - independent verification of results/paper/uncertainty_anticipation (not confirmatory;
outside the Holm family). Does the Lag-U gate's uncertainty signal anticipate cost?

Written from the task statement and the lagu code only; it shares no code with the analysis it checks.
Every number below is EXPLORATORY, except the per-arm context, which is DESCRIPTIVE.

Pre-specified metric (from the task, unchanged):
  runs     lagu-tt, lagu_nogate-tt (results/gate, s100-114); lagu-lit, lagu_nogate-lit (results/gate_literal,
           s100-109); lagu-lit-T03 (results/gate_literal_x, s100-109). No run is excluded.
  policy   the FINAL snapshot (<stem>.snap/<args.steps>.pt), deterministic actor, full horizon (1000),
           layouts EVAL_SEED + 1000 + k, k = 0..9 (EVAL_SEED = 10000).
  per step Q_bar = mean of the 3 members' first heads at (s_t, pi(s_t)); sqrt(Q_eu) = sqrt(population variance
           + 1e-12); ratio = sqrt(Q_eu) / (|Q_bar| + ratio_eps); Q_c = cost critic at (s_t, pi(s_t));
           c_t = cost returned by env.step(pi(s_t)).
  target   y_k(t) = 1 if c_j > 0 for some j in t+1..t+k, k in {10, 50}; only t with t + k <= n - 1.
  stats    pooled AUROC (Mann-Whitney rank statistic, tied values get average ranks); 95% percentile interval
           from a run-resampling bootstrap (10,000 resamples, numpy default_rng(0), fresh per arm and k, the
           same resamples for every predictor so differences are paired); per-run AUROC median; fraction of
           steps with ratio >= T, T in {0.03, 0.07}, rate of y_k among them vs the rest, relative risk.

Verifier additions, each marked in its table title, none replacing a pre-specified number:
  (V1) cross-check of every trace against the checked analysis's traces and of the batched critic values
       against LagU.values_at;
  (V2) "cost rate" read as the per-step rate of c_t > 0 (the checked analysis used the rate of y_k);
  (V3) onset sensitivity: AUROC on steps where the agent is not already in a hazard (c_{t-1} = 0 and c_t = 0),
       to separate advance warning from "already in a hazard";
  (V4) lagu-tt relative risk stratified by run (Mantel-Haenszel), to test the "between-run effect" claim.

Run from the repository root (traces are cached in traces/, keyed by the snapshot's SHA-256):
    .venv/bin/python results/paper/uncertainty-before-cost-verify/make.py [--workers 3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
CHECKED = os.path.join(ROOT, "results", "paper", "uncertainty_anticipation")   # read only
TRACES = os.path.join(HERE, "traces")
ENV = "SafetyPointGoal1-v0"
LAYOUT0 = 10_000 + 1000
N_EP = 10
KS = (10, 50)
TS = (0.03, 0.07)
B = 10_000
GROUPS = [("lagu-tt", "results/gate", list(range(100, 115))),
          ("lagu_nogate-tt", "results/gate", list(range(100, 115))),
          ("lagu-lit", "results/gate_literal", list(range(100, 110))),
          ("lagu_nogate-lit", "results/gate_literal", list(range(100, 110))),
          ("lagu-lit-T03", "results/gate_literal_x", list(range(100, 110)))]
PRED = ["ratio", "sqrtQeu", "Qc", "negQbar"]
PRED_LABEL = {"ratio": "ratio sqrt(Q_eu)/|Q_bar|", "sqrtQeu": "sqrt(Q_eu)", "Qc": "Q_c", "negQbar": "-Q_bar"}


def stem(arm: str, d: str, seed: int) -> str:
    return os.path.join(ROOT, d, f"{arm}_{ENV}_s{seed}")


# ---------------------------------------------------------------------------------------------- rollouts
def play(job):
    import safety_gymnasium
    import torch
    from lagu.agent import Config, LagU
    torch.set_num_threads(1)
    arm, d, seed = job
    st = stem(arm, d, seed)
    meta = json.load(open(st + ".json"))
    N = int(meta["args"]["steps"])
    path = os.path.join(st + ".snap", f"{N}.pt")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    out = os.path.join(TRACES, f"{arm}_s{seed}.npz")
    if os.path.exists(out):
        z = np.load(out)
        if str(z["sha"]) == sha:
            return f"{arm} s{seed} cached"
    snap = torch.load(path, weights_only=True)
    cfg = Config(**snap["cfg"])
    assert not cfg.cost_twin and cfg.M == 3
    ag = LagU(meta["obs_dim"], meta["act_dim"], cfg)
    ag.actor.load_state_dict(snap["actor"])
    ag.members.load_state_dict(snap["members"])
    ag.cost.load_state_dict(snap["cost"])
    env = safety_gymnasium.make(meta["args"]["env"])
    rec = {k: [] for k in ("ep", "qbar", "sqrtQeu", "ratio", "Qc", "c", "r")}
    max_dev = 0.0
    for k in range(N_EP):
        o, _ = env.reset(seed=LAYOUT0 + k)
        S, C, R = [], [], []
        while True:
            S.append(o)
            o, r, c, term, trunc, _ = env.step(ag.act(o, explore=False))
            C.append(c)
            R.append(r)
            if term or trunc:
                break
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(S, np.float32))
            sa = torch.cat([s, ag.actor(s)], 1)
            q = torch.stack([m.q1(sa)[:, 0] for m in ag.members]).double()     # (3, n), first heads
            qbar = q.mean(0)
            sd = torch.sqrt(((q - qbar) ** 2).mean(0) + 1e-12)
            qc = ag.cost(sa)[:, 0].double()
        if k == 0:                                                              # V1: per-step API agrees
            for t in range(0, len(S), 100):
                vb, vs, vc = ag.values_at(S[t])
                max_dev = max(max_dev, abs(vb - qbar[t].item()), abs(vs - sd[t].item()), abs(vc - qc[t].item()))
        rec["ep"].append(np.full(len(C), k))
        rec["qbar"].append(qbar.numpy())
        rec["sqrtQeu"].append(sd.numpy())
        rec["ratio"].append((sd / (qbar.abs() + cfg.ratio_eps)).numpy())
        rec["Qc"].append(qc.numpy())
        rec["c"].append(np.asarray(C, float))
        rec["r"].append(np.asarray(R, float))
    env.close()
    np.savez_compressed(out, sha=sha, T=cfg.T, use_gate=cfg.use_gate, values_at_maxdev=max_dev,
                        **{k: np.concatenate(v) for k, v in rec.items()})
    return f"{arm} s{seed} played; values_at max |dev| {max_dev:.2e}"


# ------------------------------------------------------------------------------------------------ stats
def avg_rank(x):
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    top = np.cumsum(cnt)
    return (top - (cnt - 1) / 2.0)[inv]


def mw_auc(x, y):
    p = int(y.sum())
    n = y.size - p
    if p == 0 or n == 0:
        return np.nan
    return (avg_rank(x)[y].sum() - p * (p + 1) / 2.0) / (p * n)


def window_any(c, k):
    """y_k(t) for t = 0..n-1-k: any c_j > 0 for j in t+1..t+k."""
    hit = (c > 0).astype(np.int64)
    w = np.lib.stride_tricks.sliding_window_view(hit[1:], k)       # row t = hit[t+1 .. t+k]
    return w.max(1).astype(bool)


def usable(run, k, onset=False):
    """(per-predictor arrays, y) over usable steps of a run."""
    xs = {p: [] for p in PRED}
    ys = []
    for e in range(N_EP):
        m = run["ep"] == e
        c = run["c"][m]
        y = window_any(c, k)                                         # length n - k
        n_use = y.size
        keep = np.ones(n_use, bool)
        if onset:
            prev = np.concatenate([[0.0], c[:-1]])[:n_use]
            keep = (prev == 0) & (c[:n_use] == 0)
        for p in PRED:
            v = -run["qbar"][m] if p == "negQbar" else run[p][m]
            xs[p].append(v[:n_use][keep])
        ys.append(y[keep])
    return {p: np.concatenate(v) for p, v in xs.items()}, np.concatenate(ys)


def resamples(R):
    rng = np.random.default_rng(0)
    W = np.zeros((B, R))
    for b in range(B):
        np.add.at(W[b], rng.integers(0, R, R), 1)
    return W


def pair_counts(xs, ys):
    """U[r, s] = sum over (positive of run r, negative of run s) of 1[x_pos > x_neg] + 0.5 1[equal]."""
    R = len(xs)
    U = np.zeros((R, R))
    for r in range(R):
        pos = xs[r][ys[r]]
        for s in range(R):
            neg = xs[s][~ys[s]]
            if pos.size and neg.size:
                U[r, s] = mw_auc(np.concatenate([pos, neg]),
                                 np.r_[np.ones(pos.size, bool), np.zeros(neg.size, bool)]) * pos.size * neg.size
    return U, np.array([y.sum() for y in ys], float), np.array([(~y).sum() for y in ys], float)


def pctl(v):
    v = v[~np.isnan(v)]
    v = np.where(np.isposinf(v), 1e300, v)
    lo, hi = np.percentile(v, [2.5, 97.5])
    f = lambda q: np.inf if q >= 1e299 else q
    return f(lo), f(hi)


def fmt(x, w=6, d=3):
    return f"{x:{w}.{d}f}" if np.isfinite(x) else f"{'inf' if x > 0 else 'nan':>{w}s}"


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(TRACES, exist_ok=True)
    jobs = [(arm, d, s) for arm, d, seeds in GROUPS for s in seeds]
    with mp.get_context("spawn").Pool(max(1, min(3, a.workers))) as pool:
        for msg in pool.imap_unordered(play, jobs):
            print("  ", msg, flush=True)

    lines = []

    def out(s=""):
        lines.append(s)
        print(s, flush=True)

    data = {arm: [dict(np.load(os.path.join(TRACES, f"{arm}_s{s}.npz")), seed=s) for s in seeds]
            for arm, _, seeds in GROUPS}

    # V1 ------------------------------------------------------------------------------------------------
    out("== V1 (verifier check): traces vs the checked analysis's traces; batched critic vs LagU.values_at")
    worst = {}
    for arm, _, seeds in GROUPS:
        for run in data[arm]:
            z = np.load(os.path.join(CHECKED, "traces", f"{arm}_{ENV}_s{run['seed']}.npz"))
            pairs = [("c", "cost"), ("r", "reward"), ("qbar", "qbar"), ("sqrtQeu", "std"), ("ratio", "ratio"),
                     ("Qc", "qc")]
            assert z["cost"].size == run["c"].size, (arm, run["seed"])
            assert np.array_equal(z["ep"], run["ep"])
            for mine, theirs in pairs:
                rel = np.max(np.abs(run[mine] - z[theirs]) / (np.abs(z[theirs]) + 1e-3))
                worst[mine] = max(worst.get(mine, 0.0), float(rel))
            worst["values_at"] = max(worst.get("values_at", 0.0), float(run["values_at_maxdev"]))
    out("  max relative difference over all 60 runs: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    out()

    # context -------------------------------------------------------------------------------------------
    out("== DESCRIPTIVE: context, final snapshot, 10 held-out episodes per run (layouts 11000-11009)")
    out(f"  {'arm':16s} runs  return    cost  return<5  no-cost runs  y10 rate  y50 rate")
    for arm, _, seeds in GROUPS:
        runs = data[arm]
        ret = np.array([r["r"].sum() / N_EP for r in runs])
        cst = np.array([r["c"].sum() / N_EP for r in runs])
        yr = [np.mean(np.concatenate([usable(r, k)[1] for r in runs])) for k in KS]
        out(f"  {arm:16s} {len(runs):4d} {ret.mean():7.1f} {cst.mean():7.1f} {int((ret < 5).sum()):5d}/{len(runs):<3d}"
            f" {int((cst == 0).sum()):8d}      {yr[0]:8.3f}  {yr[1]:8.3f}")
    out()

    def auroc_block(onset):
        res = {}
        for k in KS:
            tag = "V3 onset sensitivity (c_{t-1}=0 and c_t=0)" if onset else "pre-specified"
            out(f"== EXPLORATORY ({tag}): AUROC for any cost in t+1..t+{k}")
            out(f"  {'arm':16s} {'predictor':26s} {'pooled [95% CI]':>22s} {'run median':>11s} {'n':>5s} "
                f"{'run min..max':>13s} {'minus Q_c [95% CI]':>27s}")
            for arm, _, seeds in GROUPS:
                runs = data[arm]
                W = resamples(len(runs))
                per = [usable(r, k, onset) for r in runs]
                ys = [y for _, y in per]
                boot = {}
                for p in PRED:
                    xs = [x[p] for x, _ in per]
                    auc = mw_auc(np.concatenate(xs), np.concatenate(ys))
                    U, P, Nn = pair_counts(xs, ys)
                    assert abs(U.sum() / (P.sum() * Nn.sum()) - auc) < 1e-10
                    with np.errstate(invalid="ignore", divide="ignore"):
                        boot[p] = np.einsum("br,rs,bs->b", W, U, W) / ((W @ P) * (W @ Nn))
                    if p == "ratio" and k == KS[0] and not onset:                # duplicated data == formula
                        for b in range(25):
                            idx = np.repeat(np.arange(len(runs)), W[b].astype(int))
                            dup = mw_auc(np.concatenate([xs[i] for i in idx]), np.concatenate([ys[i] for i in idx]))
                            assert abs(dup - boot[p][b]) < 1e-10 or (np.isnan(dup) and np.isnan(boot[p][b]))
                    pr = np.array([mw_auc(x, y) for x, y in zip(xs, ys)])
                    ok = np.isfinite(pr)
                    lo, hi = pctl(boot[p])
                    res[(arm, k, p)] = dict(auc=auc, lo=lo, hi=hi, med=np.median(pr[ok]), n=int(ok.sum()),
                                            mn=pr[ok].min(), mx=pr[ok].max(), nanboot=int(np.isnan(boot[p]).sum()))
                for p in PRED:
                    c = res[(arm, k, p)]
                    if p == "Qc":
                        dtxt = "(reference)"
                    else:
                        dlo, dhi = pctl(boot[p] - boot["Qc"])
                        dtxt = f"{c['auc'] - res[(arm, k, 'Qc')]['auc']:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
                        c.update(d=c['auc'] - res[(arm, k, 'Qc')]['auc'], dlo=dlo, dhi=dhi)
                    out(f"  {arm:16s} {PRED_LABEL[p]:26s} {c['auc']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] "
                        f"{c['med']:11.3f} {c['n']:2d}/{len(runs):<2d} {c['mn']:.2f}..{c['mx']:.2f}  {dtxt:>27s}")
                out()
        return res

    res = auroc_block(onset=False)
    res_on = auroc_block(onset=True)

    # thresholds ----------------------------------------------------------------------------------------
    out("== EXPLORATORY (pre-specified): steps with ratio >= T; rate of y_k among them vs the rest; relative risk")
    out("   V2 columns (verifier): the same split with 'cost rate' = per-step rate of c_t > 0 over all steps")
    out(f"  {'arm':16s} {'T':>5s} {'k':>3s} {'frac>=T':>8s} {'y>=T':>7s} {'y<T':>7s} {'RR [95% CI]':>24s}"
        f" {'#inf':>5s} {'#0/0':>5s} | {'c>=T':>7s} {'c<T':>7s} {'RR_c':>6s}")
    thr = {}
    for arm, _, seeds in GROUPS:
        runs = data[arm]
        W = resamples(len(runs))
        for T in TS:
            for k in KS:
                cnt = []
                for r in runs:
                    x, y = usable(r, k)
                    f = x["ratio"] >= T
                    cnt.append([f.sum(), (f & y).sum(), (~f).sum(), (~f & y).sum()])
                cnt = np.array(cnt, float)
                n1, a1, n0, a0 = cnt.sum(0)
                rr = (a1 / n1) / (a0 / n0) if n0 and a0 else np.inf
                bw = W @ cnt
                with np.errstate(invalid="ignore", divide="ignore"):
                    brr = (bw[:, 1] / bw[:, 0]) / (bw[:, 3] / bw[:, 2])
                lo, hi = pctl(brr)
                fa = np.concatenate([r["ratio"] >= T for r in runs])
                ca = np.concatenate([r["c"] > 0 for r in runs])
                c1, c0 = ca[fa].mean(), (ca[~fa].mean() if (~fa).any() else np.nan)
                thr[(arm, T, k)] = dict(frac=n1 / (n1 + n0), rr=rr, lo=lo, hi=hi, cnt=cnt)
                out(f"  {arm:16s} {T:5.2f} {k:3d} {n1 / (n1 + n0):8.3f} {a1 / n1:7.3f} {a0 / n0:7.3f} "
                    f"{fmt(rr, 6, 2)} [{fmt(lo, 6, 2)}, {fmt(hi, 6, 2)}] {int(np.isposinf(brr).sum()):5d} "
                    f"{int(np.isnan(brr).sum()):5d} | {c1:7.3f} {c0:7.3f} {fmt(c1 / c0 if c0 > 0 else np.inf, 6, 2)}")
    out()

    # V4 ------------------------------------------------------------------------------------------------
    out("== V4 (verifier, EXPLORATORY): relative risk stratified by run (Mantel-Haenszel), within-run comparison")
    out(f"  {'arm':16s} {'T':>5s} {'k':>3s} {'pooled RR':>9s} {'MH RR':>7s} {'runs with steps on both sides':>30s}")
    for arm, _, seeds in GROUPS:
        for T in TS:
            for k in KS:
                cnt = thr[(arm, T, k)]["cnt"]
                n1, a1, n0, a0 = cnt.T
                Nr = n1 + n0
                num, den = np.sum(a1 * n0 / Nr), np.sum(a0 * n1 / Nr)
                mh = num / den if den > 0 else np.inf
                both = int(((n1 > 0) & (n0 > 0)).sum())
                out(f"  {arm:16s} {T:5.2f} {k:3d} {fmt(thr[(arm, T, k)]['rr'], 9, 2)} {fmt(mh, 7, 2)} "
                    f"{both:>18d}/{len(cnt)}")
    out()
    tt = data["lagu-tt"]
    below = [float(np.mean(r["ratio"] < 0.07)) for r in tt]
    out("  lagu-tt per run: share of steps with ratio < 0.07 / 10-episode return / cost: " +
        "; ".join(f"s{r['seed']} {b:.2f}/{r['r'].sum() / N_EP:.1f}/{r['c'].sum() / N_EP:.0f}" for r, b in zip(tt, below)))
    out()

    with open(os.path.join(HERE, "verify_table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    figure(res, res_on)
    print("wrote verify_table.txt, fig_verify.pdf, fig_verify.png")


def figure(res, res_on):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    surf, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    col = {"ratio": "#2a78d6", "sqrtQeu": "#eb6834", "Qc": "#1baf7a", "negQbar": muted}
    mk = {"ratio": "o", "sqrtQeu": "s", "Qc": "^", "negQbar": "D"}
    plt.rcParams.update({"font.size": 9, "text.color": ink, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.edgecolor": "#c3c2b7"})
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.6), sharey=True, facecolor=surf)
    off = np.linspace(-0.27, 0.27, len(PRED))
    arms = [g[0] for g in GROUPS]
    for row, (rr, name) in enumerate([(res, "all usable steps (pre-specified)"),
                                      (res_on, "steps not already in a hazard (verifier sensitivity)")]):
        for colj, k in enumerate(KS):
            ax = axs[row, colj]
            ax.set_facecolor(surf)
            ax.axhline(0.5, color=muted, lw=1, ls="--", zorder=1)
            for j, p in enumerate(PRED):
                x = np.arange(len(arms)) + off[j]
                y = np.array([rr[(a, k, p)]["auc"] for a in arms])
                lo = np.array([rr[(a, k, p)]["lo"] for a in arms])
                hi = np.array([rr[(a, k, p)]["hi"] for a in arms])
                ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=mk[p], ms=6, color=col[p], elinewidth=1.4,
                            mec=surf, mew=1, zorder=3)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(["lagu-tt\n(E1 gate)", "lagu_nogate-tt\n(E1 no gate)", "lagu-lit\n(E1b gate)",
                                "lagu_nogate-lit\n(E1b no gate)", "lagu-lit-T03\n(X1 gate)"], fontsize=7.5)
            ax.set_title(f"{name}\ncost within the next k = {k} steps", fontsize=9.5)
            ax.grid(axis="y", color=grid, lw=0.6, zorder=0)
            for sd in ("top", "right"):
                ax.spines[sd].set_visible(False)
            if colj == 0:
                ax.set_ylabel("AUROC (dimensionless; 0.5 = chance)")
            if row == 1:
                ax.set_xlabel("arm (final snapshot, 10 held-out episodes per run)")
    axs[0, 0].set_ylim(0.2, 0.9)
    h = [Line2D([], [], ls="none", marker=mk[p], color=col[p], ms=6, mec=surf) for p in PRED]
    h.append(Line2D([], [], color=muted, ls="--", lw=1))
    fig.legend(h, [PRED_LABEL[p] for p in PRED] + ["chance"], loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("EXPLORATORY (verification): pooled AUROC of each predictor for an upcoming cost, "
                 "95% run-resampling bootstrap bars", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_verify.{ext}"), dpi=200, facecolor=surf)


if __name__ == "__main__":
    main()
