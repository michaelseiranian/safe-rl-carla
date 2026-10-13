"""EXPLORATORY / DESCRIPTIVE: full-curve time courses of the Lag-U gate study (prereg.md 16, item 5).

Nothing in this script is confirmatory. The primary endpoint stays the 2M (1M for the literal
recipe) held-out evaluation of prereg.md section 5; every number here comes from the in-training
evaluation rows (`eval_*`, 10 episodes on layouts EVAL_SEED + k) and the interval columns of the
run CSVs, every 50k steps, with no chosen windows.

    cd /Users/michaelseiranian/Desktop/safe-rl-carla
    .venv/bin/python results/paper/timecourse/make.py

Experiments (SafetyPointGoal1-v0, delta0 = 2.5 in every arm's reference):
  E1   lagu-tt vs lagu_nogate-tt, seeds 100-114 (the H3 pairs)                       results/gate
  E2   E1's arms plus lagu_nogate-tt-match (delta0 = 1.12158); lagu-tt and the matched arm
       on seeds 100-119, lagu_nogate-tt on 100-114 (no 115-119 runs exist). Extension seeds
       115-119 are marked: dashed thin lines = median over 100-114 only; dotted line in the
       collapse panel = collapsed runs among 115-119.                                 results/gate
  E1b  lagu-lit vs lagu_nogate-lit, seeds 100-109, 1M                                 results/gate_literal
  X1   lagu-lit-T03 vs lagu_nogate-lit-match (delta0 = 2.14758) vs lagu_nogate-lit,
       seeds 100-109, 1M                                    results/gate_literal_x, results/gate_literal

Panels, per arm and per evaluation row (step t = 50k, 100k, ...):
  (a) eval_return, (b) eval_cost (cost limit 25 drawn): median and IQR (25th-75th percentile) over seeds.
  (c) number of runs with eval_return < 5 at that row (non-finite counts as < 5, prereg.md 11).
  (d) mean budget: sum_seeds(delta_avg * n_pi) / sum_seeds(n_pi) / 2.5 (line); IQR over seeds of
      delta_avg / 2.5 (band). References: E2's constant 1.12158 / 2.5 and the band 0.76-0.88,
      the paper's dose as we read Zhang et al. Fig. 5(d).
  (e) lambda_avg, (f) qc_avg, (g) qstd_avg: median and IQR over seeds. Interval columns: the row
      at t summarises the policy updates in (t - 50k, t].

Horizon table (E1 only, EXPLORATORY): for N in {0.5M, 1M, 1.5M, 2M}, each run's mean of the
in-training rows with 0.8N <= step <= N (the prereg.md section 5 secondary window applied at
horizon N), then per arm the mean and median over seeds and the collapse count (window return < 5
or non-finite); paired differences lagu - lagu_nogate with 95% paired-bootstrap intervals
(10,000 resamples, numpy.random.default_rng(0), lagu.analyze.paired_bootstrap). The 2M held-out
primary endpoint is printed under it for reference only.

Outputs (next to this script): fig_<exp>.pdf / .png, horizon_table.txt / .csv,
timecourse_data.csv (every plotted value, long format; seed_set "all" or, in E2, "s100-114" /
"s115-119"; metric "collapsed_count" stores the count in the median column; metric "budget" has
the IQR of delta_avg / 2.5 in q25/median/q75 and the plotted n_pi-weighted mean in
npi_weighted_mean), and a DESCRIPTIVE digest printed to stdout at every 250k (a fixed grid).
No run is excluded. Numpy + matplotlib only.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from lagu.analyze import endpoints, load, paired_bootstrap  # noqa: E402

ENV = "SafetyPointGoal1-v0"
DELTA0 = 2.5
COST_LIMIT = 25.0
COLLAPSE = 5.0
D_E2 = 1.12158                       # prereg.md 15: E2's matched constant budget
PAPER_BAND = (0.76, 0.88)            # prereg.md 16 item 7: Fig. 5(d) as we read it
EXT = set(range(115, 120))           # extension seeds (prereg.md 4, 10.5)
S15, S20, S10 = list(range(100, 115)), list(range(100, 120)), list(range(100, 110))

# Okabe-Ito hues (validated with the dataviz palette checker: all pairs pass, light surface);
# marker shape is the redundant encoding.
STYLE = {"gate": ("#0072B2", "o"), "nogate": ("#D55E00", "s"), "match": ("#009E73", "^")}

EXPERIMENTS = [
    # key, title, [(dir, arm, seeds, role)], steps
    ("E1", "E1: gate on vs gate off, two-timescale recipe (seeds 100-114)",
     [("gate", "lagu-tt", S15, "gate"), ("gate", "lagu_nogate-tt", S15, "nogate")], 2_000_000),
    ("E2", "E2: gate vs ungated vs matched constant budget, two-timescale recipe",
     [("gate", "lagu-tt", S20, "gate"), ("gate", "lagu_nogate-tt", S15, "nogate"),
      ("gate", "lagu_nogate-tt-match", S20, "match")], 2_000_000),
    ("E1b", "E1b: gate on vs gate off, literal recipe (seeds 100-109)",
     [("gate_literal", "lagu-lit", S10, "gate"), ("gate_literal", "lagu_nogate-lit", S10, "nogate")], 1_000_000),
    ("X1", "X1: gate at the paper's dose (T = 0.03) vs matched vs ungated, literal recipe (seeds 100-109)",
     [("gate_literal_x", "lagu-lit-T03", S10, "gate"), ("gate_literal_x", "lagu_nogate-lit-match", S10, "match"),
      ("gate_literal", "lagu_nogate-lit", S10, "nogate")], 1_000_000),
]

PANELS = [  # column, title, y label
    ("eval_return", "(a) return", "eval return per episode"),
    ("eval_cost", "(b) cost", "eval cost per episode"),
    ("collapsed", "(c) runs with eval return < 5", "number of runs"),
    ("budget", "(d) mean budget", r"$\delta$ / 2.5 (fraction of $\delta_0$)"),
    ("lambda_avg", r"(e) $\lambda$ (interval mean)", r"$\lambda$ (dimensionless)"),
    ("qc_avg", r"(f) cost critic $Q_c$ (interval mean)", r"$Q_c$ (discounted cost)"),
    ("qstd_avg", r"(g) ensemble std $\sqrt{Q_{eu}}$ (interval mean)", r"$\sqrt{Q_{eu}}$ (discounted return)"),
]
QUANT_COLS = ("eval_return", "eval_cost", "lambda_avg", "qc_avg", "qstd_avg")

_cache: dict[str, list[dict]] = {}


def runs_of(d: str, arm: str, seeds: list[int], steps: int) -> list[dict]:
    """The runs of one arm, one per requested seed, all complete on the common 50k grid."""
    if d not in _cache:
        _cache[d] = [r for r in load(os.path.join(ROOT, "results", d)) if r["env"] == ENV]
    by_seed = {r["seed"]: r for r in _cache[d] if r["arm"] == arm}
    missing = [s for s in seeds if s not in by_seed]
    assert not missing, f"{d}/{arm}: missing seeds {missing}"
    rs = [by_seed[s] for s in seeds]
    grid = np.arange(50_000, steps + 1, 50_000, dtype=float)
    for r in rs:
        st = np.array([row["step"] for row in r["rows"]], float)
        assert np.array_equal(st, grid), f"{r['stem']}: rows are not the 50k grid up to {steps}"
    return rs


def matrix(rs: list[dict], key: str) -> np.ndarray:
    return np.array([[row.get(key, np.nan) for row in r["rows"]] for r in rs], float)


def arm_label(r: dict, n: int, seeds: list[int]) -> str:
    cfg = r["cfg"]
    knob = f"gate on, T = {cfg['T']:g}" if cfg["use_gate"] else f"gate off, $\\delta_0$ = {cfg['delta0']:.6g}"
    ext = sorted(EXT & set(seeds))
    span = f"s{seeds[0]}-{seeds[-1]}" + (f", ext. s{ext[0]}-{ext[-1]}" if ext else "")
    return f"{r['arm']} ({knob}; n = {n}: {span})"


def series(rs: list[dict]) -> dict:
    """Per-step summaries over the seeds of one arm (everything that is plotted)."""
    out = {}
    for k in QUANT_COLS:
        m = matrix(rs, k)
        out[k] = np.nanpercentile(m, [25, 50, 75], axis=0)
    ret = matrix(rs, "eval_return")
    out["collapsed"] = (~(ret >= COLLAPSE)).sum(axis=0)          # NaN counts as collapsed
    dl, npi = matrix(rs, "delta_avg"), matrix(rs, "n_pi")
    ok = np.isfinite(dl) & np.isfinite(npi) & (npi > 0)
    out["budget_wmean"] = np.where(ok, dl * npi, 0).sum(0) / np.where(ok, npi, 0).sum(0) / DELTA0
    out["budget"] = np.nanpercentile(np.where(ok, dl, np.nan) / DELTA0, [25, 50, 75], axis=0)
    return out


def figure(key: str, title: str, arms: list[tuple], steps: int, rows_out: list[list]) -> list[str]:
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4), facecolor="white")
    axes = axes.ravel()
    x_all = np.arange(50_000, steps + 1, 50_000) / 1e6
    legend_items, digest = [], []
    every = max(1, len(x_all) // 8)
    all_zero = True
    for j, (d, arm, seeds, role) in enumerate(arms):
        rs = runs_of(d, arm, seeds, steps)
        color, marker = STYLE[role]
        S = series(rs)
        core = [r for r in rs if r["seed"] not in EXT]
        S_core = series(core) if len(core) < len(rs) else None
        n_ext = len(rs) - len(core)
        lab = arm_label(rs[0], len(rs), seeds)
        legend_items.append(Line2D([], [], color=color, marker=marker, lw=2, ms=6, label=lab))
        # markers staggered per arm so that coinciding lines still show every arm's marker
        mk = dict(marker=marker, markevery=(j * max(1, every // len(arms)), every), ms=5)
        all_zero &= bool(np.all(S["collapsed"] == 0))
        for ax, (col, _, _) in zip(axes, PANELS):
            if col == "collapsed":
                ax.plot(x_all, S["collapsed"], color=color, lw=2, **mk)
                if n_ext:
                    ext_count = S["collapsed"] - S_core["collapsed"]
                    ax.plot(x_all, ext_count, color=color, lw=1.4, ls=":")
                continue
            q25, q50, q75 = S[col]
            ax.fill_between(x_all, q25, q75, color=color, alpha=0.16, lw=0)
            line = S["budget_wmean"] if col == "budget" else q50
            ax.plot(x_all, line, color=color, lw=2, **mk)
            if n_ext and col != "budget":
                ax.plot(x_all, S_core[col][1], color=color, lw=1.1, ls="--")
        # long-format data: every plotted value
        for i, t in enumerate(np.arange(50_000, steps + 1, 50_000)):
            for col in QUANT_COLS + ("budget",):
                q25, q50, q75 = S[col][:, i]
                extra = S["budget_wmean"][i] if col == "budget" else np.nan
                rows_out.append([key, arm, len(rs), "all", int(t), col, q50, q25, q75, extra])
                if S_core is not None and col != "budget":
                    c25, c50, c75 = S_core[col][:, i]
                    rows_out.append([key, arm, len(core), "s100-114", int(t), col, c50, c25, c75, np.nan])
            rows_out.append([key, arm, len(rs), "all", int(t), "collapsed_count", S["collapsed"][i],
                             np.nan, np.nan, np.nan])
            if S_core is not None:
                rows_out.append([key, arm, n_ext, "s115-119", int(t), "collapsed_count",
                                 S["collapsed"][i] - S_core["collapsed"][i], np.nan, np.nan, np.nan])
        # digest at a fixed grid (every 250k), declared before looking at outcomes
        for i, t in enumerate(np.arange(50_000, steps + 1, 50_000)):
            if t % 250_000 == 0:
                digest.append(f"  {arm:24s} n={len(rs):2d} {t / 1e6:4.2f}M  return {S['eval_return'][1, i]:6.2f} "
                              f"[{S['eval_return'][0, i]:6.2f},{S['eval_return'][2, i]:6.2f}]  cost "
                              f"{S['eval_cost'][1, i]:6.2f} [{S['eval_cost'][0, i]:6.2f},{S['eval_cost'][2, i]:6.2f}]  "
                              f"collapsed {int(S['collapsed'][i]):2d}/{len(rs)}  budget {S['budget_wmean'][i]:.3f}  "
                              f"lambda {S['lambda_avg'][1, i]:.4f}  qc {S['qc_avg'][1, i]:.3f}  "
                              f"qstd {S['qstd_avg'][1, i]:.4f}")

    n_max = max(len(s) for _, _, s, _ in arms)
    for ax, (col, ptitle, ylab) in zip(axes, PANELS):
        ax.set_title(ptitle, fontsize=10, loc="left")
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlabel("environment steps (millions)", fontsize=9)
        ax.set_xlim(0, steps / 1e6)
        ax.grid(True, color="#d9d9d9", lw=0.6)
        ax.tick_params(labelsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if col == "eval_cost":
            ax.axhline(COST_LIMIT, color="#333333", lw=1.2, ls="--", zorder=1)
            ax.set_ylim(bottom=0)
        if col == "collapsed":
            if all_zero:
                ax.text(0.5, 0.5, "no run below 5 at any row, in any arm", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8.5, color="#333333")
            ax.set_ylim(-0.3, n_max + 0.5)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if col == "budget":
            ax.axhspan(*PAPER_BAND, color="#9a9a9a", alpha=0.22, lw=0, zorder=0)
            ax.axhline(D_E2 / DELTA0, color="#333333", lw=1.2, ls="-.", zorder=1)
            ax.set_ylim(0, 1.05)
    # panel 8: legend and reading notes
    lax = axes[7]
    lax.axis("off")
    ref = [Patch(color="#9a9a9a", alpha=0.35, label="0.76-0.88: paper's dose, as we read Zhang et al. Fig. 5(d)"),
           Line2D([], [], color="#333333", ls="-.", lw=1.2, label=f"E2 constant {D_E2} / 2.5 = {D_E2 / DELTA0:.3f}"),
           Line2D([], [], color="#333333", ls="--", lw=1.2, label="cost limit 25 (panel b)")]
    if any(set(s) & EXT for _, _, s, _ in arms):
        ref += [Line2D([], [], color="#555555", ls="--", lw=1.1, label="median over s100-114 only (extension seeds removed)"),
                Line2D([], [], color="#555555", ls=":", lw=1.4, label="(c) collapsed runs among extension seeds s115-119")]
    lax.legend(handles=legend_items + ref, loc="upper left", fontsize=8, frameon=False, borderaxespad=0)
    lax.text(0.0, 0.02,
             "Lines: median over seeds; bands: IQR (25th-75th pct).\n"
             "(d) line: n_pi-weighted mean of delta_avg over seeds, / 2.5;\n"
             "     band: IQR over seeds of delta_avg / 2.5.\n"
             "(e)-(g): interval means over the policy updates in (t-50k, t].\n"
             "In-training eval rows: 10 episodes each, every 50k steps.\n"
             "EXPLORATORY. Primary endpoint: held-out evaluation at the end.",
             transform=lax.transAxes, fontsize=8, va="bottom", color="#333333")
    fig.suptitle(f"{title}; SafetyPointGoal1-v0. EXPLORATORY time course", fontsize=12, x=0.01, y=0.99,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_{key}.{ext}"), dpi=200, facecolor="white")
    plt.close(fig)
    return digest


def horizon_table() -> tuple[list[str], list[list]]:
    arms = [("lagu-tt", "gate"), ("lagu_nogate-tt", "nogate")]
    R = {a: runs_of("gate", a, S15, 2_000_000) for a, _ in arms}
    lines = ["EXPLORATORY horizon table, E1 (SafetyPointGoal1-v0, seeds 100-114, 15 pairs).",
             "In-training window = rows with 0.8N <= step <= N (eval_return / eval_cost, 10 episodes per row).",
             "Collapsed = window return < 5 or non-finite. Delta = lagu-tt - lagu_nogate-tt, paired by seed,",
             "95% paired bootstrap (10,000 resamples, default_rng(0)). Not confirmatory; the primary endpoint",
             "is the 2M held-out evaluation (last block).", ""]
    hdr = (f"{'N':>5} {'rows':>4} | {'arm':15s} {'ret mean':>8} {'ret med':>8} {'cost mean':>9} {'cost med':>8} "
           f"{'collapsed':>9}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    csv_rows = [["N", "rows_in_window", "arm", "n", "return_mean", "return_median", "cost_mean", "cost_median",
                 "collapsed", "d_return_mean", "d_return_lo95", "d_return_hi95", "d_cost_mean", "d_cost_lo95",
                 "d_cost_hi95"]]
    for N in (500_000, 1_000_000, 1_500_000, 2_000_000):
        W = {}
        for a, _ in arms:
            ret = np.array([np.mean([x["eval_return"] for x in r["rows"] if 0.8 * N - 1e-6 <= x["step"] <= N]) for r in R[a]])
            cost = np.array([np.mean([x["eval_cost"] for x in r["rows"] if 0.8 * N - 1e-6 <= x["step"] <= N]) for r in R[a]])
            nrow = sum(1 for x in R[a][0]["rows"] if 0.8 * N - 1e-6 <= x["step"] <= N)
            W[a] = (ret, cost, int((~(ret >= COLLAPSE)).sum()), nrow)
        d = np.column_stack([W["lagu-tt"][0] - W["lagu_nogate-tt"][0], W["lagu-tt"][1] - W["lagu_nogate-tt"][1]])
        ci = paired_bootstrap(d)[95]
        for a, _ in arms:
            ret, cost, col, nrow = W[a]
            lines.append(f"{N / 1e6:4.1f}M {nrow:4d} | {a:15s} {ret.mean():8.2f} {np.median(ret):8.2f} "
                         f"{cost.mean():9.2f} {np.median(cost):8.2f} {col:5d}/15")
            csv_rows.append([N, nrow, a, 15, ret.mean(), np.median(ret), cost.mean(), np.median(cost), col]
                            + ([d[:, 0].mean(), *ci[0], d[:, 1].mean(), *ci[1]] if a == "lagu-tt" else [""] * 6))
        lines.append(f"{'':>11}   delta return {d[:, 0].mean():+6.2f} [{ci[0][0]:+6.2f}, {ci[0][1]:+6.2f}]   "
                     f"delta cost {d[:, 1].mean():+6.2f} [{ci[1][0]:+6.2f}, {ci[1][1]:+6.2f}]")
        if N == 2_000_000:   # cross-check against lagu.analyze's secondary endpoint at 2M
            for a, _ in arms:
                e = [endpoints(r) for r in R[a]]
                assert np.allclose([x["window_return"] for x in e], W[a][0]), a
                assert np.allclose([x["window_cost"] for x in e], W[a][1]), a
    # reference: the primary endpoint (2M held-out, 5 snapshots x 20 layouts), from <stem>.heldout.json
    lines += ["", "Reference only, the primary endpoint at 2M (held-out, 100 episodes per run):"]
    for a, _ in arms:
        e = [endpoints(r, held=True) for r in R[a]]
        ret, cost = np.array([x["return"] for x in e]), np.array([x["cost"] for x in e])
        lines.append(f"  {'2.0M held-out':>14} | {a:15s} {ret.mean():8.2f} {np.median(ret):8.2f} "
                     f"{cost.mean():9.2f} {np.median(cost):8.2f} {sum(x['collapsed'] for x in e):5d}/15")
    return lines, csv_rows


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.edgecolor": "#444444"})
    data_rows = [["experiment", "arm", "n_seeds", "seed_set", "step", "metric", "median", "q25", "q75",
                  "npi_weighted_mean"]]
    print(__doc__.split("\n\n")[0])
    print()
    for key, title, arms, steps in EXPERIMENTS:
        digest = figure(key, title, arms, steps, data_rows)
        print(f"DESCRIPTIVE time-course digest, {key} (median [IQR] over seeds at every 250k; "
              f"budget = n_pi-weighted mean delta_avg / 2.5; lambda, qc, qstd = medians):")
        print("\n".join(digest))
        print()
    lines, csv_rows = horizon_table()
    print("\n".join(lines))
    with open(os.path.join(HERE, "horizon_table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "horizon_table.csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    with open(os.path.join(HERE, "timecourse_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(data_rows[0])
        for r in data_rows[1:]:
            w.writerow([x if isinstance(x, (str, int, np.integer)) else (f"{x:.6g}" if np.isfinite(x) else "")
                        for x in r])


if __name__ == "__main__":
    main()
