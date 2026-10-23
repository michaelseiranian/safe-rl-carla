"""EXPLORATORY / DESCRIPTIVE appendix (post hoc, not confirmatory): operating characteristics of the
pre-registered section 7 verdict rule (prereg.md), and the robustness of two labels it produced.

Run from the repository root (about 2 minutes, one process):

    .venv/bin/python results/paper/verdict_oc/make.py

Writes, next to this file: tables.txt (everything printed), results.json, and the figures
fig_label_probabilities.{pdf,png} and fig_label_robustness.{pdf,png}. Reads only existing results
(results/gate, results/gate_literal); nothing under lagu/ or results/ is modified. The verdict rule,
the endpoints, the collapse definition and the paired bootstrap are the pre-registered code in
lagu/analyze.py, imported unchanged.

Part A (EXPLORATORY). Simulated operating characteristics of the section 7 rule.
  Noise model: the 15 observed E1 paired differences (lagu-tt - lagu_nogate-tt, SafetyPointGoal1,
  held-out endpoint, seeds 100-114), resampled jointly (return and cost together, keeping their
  correlation) after centring each column on 0. A simulated study draws n pairs (n = 15 or 20)
  with replacement, adds the scenario's mean shift, draws each run's collapse independently with
  probability pA (arm A) or pB (arm B), and applies section 7 with 2,000 paired-bootstrap resamples
  (percentile intervals, as lagu/analyze.py). 2,000 studies per scenario and n (the task's numbers;
  no reduction). Each (scenario, n) uses numpy.random.default_rng(0), so all scenarios at one n share
  their random numbers (common random numbers): they differ only by the shift or the collapse
  probabilities. Monte Carlo SE of a probability is at most 1.1 percentage points.
  Scenarios (fixed before any simulation was run):
    null, pA = pB = 0.2 (0.2 = E1 control-arm collapse rate, 3/15)
    null, pA = pB = 0.4 (0.4 = pooled E1 collapse rate, 12/30)
    null, exchangeable pairs: the 15 raw pairs with their own collapse flags, each drawn pair's arms
      swapped with probability 1/2 (sign-flipped differences, swapped flags); keeps the coupling
      between collapse and return difference that the independent model drops
    return -5, return -10 (cost shift 0, pA = pB = 0.2)
    cost -5, cost -10 (return shift 0, pA = pB = 0.2)
    collapse +0.2 (pA 0.4, pB 0.2) and +0.4 (pA 0.6, pB 0.2; the observed E1 rates), no mean shift
  Also the exact probability that the collapse clause (A has >= 3 more collapsed runs than B) fires,
  from two independent binomials, as a check on the simulation and over a grid of collapse rates.

Part B (EXPLORATORY). Collapse-threshold robustness of the E1 H3 label: the pre-registered
  contrast (lagu.analyze.pair, 10,000 resamples, seed 0, held-out endpoint) recomputed with the
  collapse threshold set to 2, 3, ..., 12 instead of 5 (a run collapses if its held-out return is
  below the threshold or it diverged). Only the collapse counts change; the intervals do not.

Part C (DESCRIPTIVE). Sensitivity of the E1b label to the bootstrap seed: the pre-registered E1b
  contrast (lagu-lit - lagu_nogate-lit, seeds 100-109, held-out, 10,000 resamples) recomputed with
  bootstrap seeds 0-1999. This is NOT a re-analysis: the pre-registered label is the one at seed 0.
  It describes how close that label sits to the Helps/Equivalent boundary.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
import lagu.analyze as A  # noqa: E402  (pre-registered analysis code, used unchanged)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENV = A.PRIMARY_ENV
LABELS = ("Helps", "Hurts", "Shift", "Equivalent", "Inconclusive")
N_STUDIES, N_BOOT_SIM, CHUNK = 2000, 2000, 100
NS = (15, 20)
THRESHOLDS = range(2, 13)
N_SEEDS_E1B = 2000
P0 = 0.2                      # E1 control-arm collapse rate (3/15): baseline for every non-null row
LINES: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


# ---------------------------------------------------------------- data (pre-registered code paths)

def paired_endpoints(dirpath: str, a: str, b: str, seeds) -> list:
    """[(seed, endpoints A, endpoints B)] exactly as lagu.analyze.pair builds them."""
    G = A.cells(A.load(str(ROOT / dirpath)), ENV, seeds)
    ga, gb = G[(ENV, a)], G[(ENV, b)]
    common = sorted(set(ga) & set(gb))
    held = all(A.heldout_ok(r) for s in common for r in (ga[s], gb[s]))
    assert held, f"{dirpath}: not every run has a held-out file"
    return [(s, A.endpoints(ga[s], None, True), A.endpoints(gb[s], None, True)) for s in common]


# ---------------------------------------------------------------- Part A: simulation

def hurts_interval_clause(r95, c95) -> bool:
    (rl, rh), (cl, ch) = r95, c95
    return bool((cl > 0 and rh < A.MARGIN_RET) or (rh < 0 and cl > -A.MARGIN_COST))


def simulate(d, flags, n, shift=(0.0, 0.0), pA=P0, pB=P0, exchangeable=False, seed=0) -> dict:
    """Label frequencies of the section 7 rule over N_STUDIES simulated studies of n pairs.
    d: (15, 2) observed paired differences (return, cost); flags: (15, 2) collapse flags (A, B)."""
    rng = np.random.default_rng(seed)
    dc = d - d.mean(axis=0)
    counts = dict.fromkeys(LABELS, 0)
    hurts_collapse_only = clause_fires = 0
    for start in range(0, N_STUDIES, CHUNK):
        m = min(CHUNK, N_STUDIES - start)
        idx = rng.integers(0, len(d), (m, n))
        if exchangeable:
            swap = rng.integers(0, 2, (m, n)).astype(bool)
            study = d[idx] * np.where(swap, -1.0, 1.0)[..., None]
            fa, fb = flags[idx, 0], flags[idx, 1]
            ca, cb = np.where(swap, fb, fa).sum(1), np.where(swap, fa, fb).sum(1)
        else:
            study = dc[idx] + np.asarray(shift, float)
            ca = (rng.random((m, n)) < pA).sum(1)
            cb = (rng.random((m, n)) < pB).sum(1)
        bidx = rng.integers(0, n, (m, N_BOOT_SIM, n))
        means = study[np.arange(m)[:, None, None], bidx].mean(axis=2)          # (m, B, 2)
        q = np.percentile(means, [2.5, 97.5, 5, 95], axis=1)                      # (4, m, 2)
        for k in range(m):
            r95, c95 = (q[0, k, 0], q[1, k, 0]), (q[0, k, 1], q[1, k, 1])
            r90, c90 = (q[2, k, 0], q[3, k, 0]), (q[2, k, 1], q[3, k, 1])
            lab = A.verdict(r95, c95, r90, c90, int(ca[k]), int(cb[k]))
            counts[lab] += 1
            hurts_collapse_only += lab == "Hurts" and not hurts_interval_clause(r95, c95)
            clause_fires += int(ca[k]) - int(cb[k]) >= 3
    p = {k: v / N_STUDIES for k, v in counts.items()}
    return {"p": p, "hurts_collapse_only": hurts_collapse_only / N_STUDIES,
            "collapse_clause_fires": clause_fires / N_STUDIES}


def binom_pmf(n: int, p: float) -> np.ndarray:
    return np.array([math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])


def p_clause_exact(n: int, pA: float, pB: float) -> float:
    """P(X_A - X_B >= 3), X_A ~ Bin(n, pA), X_B ~ Bin(n, pB) independent."""
    a, b = binom_pmf(n, pA), binom_pmf(n, pB)
    return float(sum(a[i] * b[j] for i in range(n + 1) for j in range(n + 1) if i - j >= 3))


SCENARIOS = [  # name, intended reading, shift (return, cost), pA, pB, exchangeable
    ("null, p_collapse 0.2 both", "Equiv/Inconcl", (0, 0), 0.2, 0.2, False),
    ("null, p_collapse 0.4 both", "Equiv/Inconcl", (0, 0), 0.4, 0.4, False),
    ("null, exchangeable pairs", "Equiv/Inconcl", (0, 0), None, None, True),
    ("return -5", "Hurts", (-5, 0), P0, P0, False),
    ("return -10", "Hurts", (-10, 0), P0, P0, False),
    ("cost -5", "Helps", (0, -5), P0, P0, False),
    ("cost -10", "Helps", (0, -10), P0, P0, False),
    ("collapse +0.2 (0.4 vs 0.2)", "Hurts", (0, 0), 0.4, 0.2, False),
    ("collapse +0.4 (0.6 vs 0.2)", "Hurts", (0, 0), 0.6, 0.2, False),
]


def part_a(E1: list) -> dict:
    d = np.array([[a["return"] - b["return"], a["cost"] - b["cost"]] for _, a, b in E1], float)
    flags = np.array([[a["collapsed"], b["collapsed"]] for _, a, b in E1], float)
    assert np.isfinite(d).all() and len(d) == 15
    sd = d.std(axis=0, ddof=1)
    out("=" * 100)
    out("PART A (EXPLORATORY, post hoc). Operating characteristics of the section 7 verdict rule")
    out("=" * 100)
    out(f"Noise model: E1 paired differences lagu-tt - lagu_nogate-tt, {ENV}, held-out, seeds "
        f"{E1[0][0]}-{E1[-1][0]} (n = {len(d)}).")
    out(f"  observed mean dReturn {d[:, 0].mean():.2f} (SD {sd[0]:.2f}), mean dCost {d[:, 1].mean():.2f} "
        f"(SD {sd[1]:.2f}), corr {np.corrcoef(d.T)[0, 1]:.2f}; collapses A {int(flags[:, 0].sum())}/15, "
        f"B {int(flags[:, 1].sum())}/15.")
    out("  Each column is centred on 0 before resampling (except the exchangeable null). Collapse is drawn")
    out(f"  independently of the differences (except the exchangeable null). {N_STUDIES:,} simulated studies x "
        f"{N_BOOT_SIM:,} bootstrap resamples")
    out("  per row (no reduction); common random numbers across rows of one n; Monte Carlo SE <= 1.1 points.")
    res: dict = {"noise_model": {"d_return": d[:, 0].tolist(), "d_cost": d[:, 1].tolist(),
                                 "collapsed_a": flags[:, 0].tolist(), "collapsed_b": flags[:, 1].tolist()},
                 "n_studies": N_STUDIES, "n_boot": N_BOOT_SIM, "rows": []}
    for n in NS:
        out()
        out(f"EXPLORATORY. Table A{NS.index(n) + 1}. Probability (%) of each section 7 label, n = {n} pairs.")
        out(f"  {'scenario':<29}{'reading':<14}{'Helps':>6}{'Hurts':>7}{'(coll.':>7}{'Shift':>7}"
            f"{'Equiv':>7}{'Incon':>7}{'clause':>8}{'exact':>7}")
        out(f"  {'':<29}{'':<14}{'':>6}{'':>7}{'only)':>7}{'':>7}{'':>7}{'':>7}{'>=3':>8}{'>=3':>7}")
        for name, reading, shift, pA, pB, exch in SCENARIOS:
            r = simulate(d, flags, n, shift, pA, pB, exch)
            ex = p_clause_exact(n, pA, pB) if not exch else float("nan")
            p = r["p"]
            out(f"  {name:<29}{reading:<14}{100 * p['Helps']:>6.1f}{100 * p['Hurts']:>7.1f}"
                f"{100 * r['hurts_collapse_only']:>7.1f}{100 * p['Shift']:>7.1f}{100 * p['Equivalent']:>7.1f}"
                f"{100 * p['Inconclusive']:>7.1f}{100 * r['collapse_clause_fires']:>8.1f}"
                f"{'' if exch else f'{100 * ex:.1f}':>7}")
            res["rows"].append({"n": n, "scenario": name, "intended": reading, "shift_return": shift[0],
                                "shift_cost": shift[1], "pA": pA, "pB": pB, "exchangeable": exch, **r,
                                "collapse_clause_exact": None if exch else ex})
    out("  Columns: (coll. only) = Hurts reached only through the collapse clause; clause >=3 = simulated")
    out("  P(A has >= 3 more collapsed runs than B), whatever the label; exact >=3 = the same from two binomials.")

    grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    out()
    out("EXPLORATORY. Table A3. Exact P(collapse clause fires) = P(X_A - X_B >= 3), X ~ Binomial(n, p), "
        "independent (%).")
    out(f"  {'pA':>5}{'pB':>6}{'n = 15':>9}{'n = 20':>9}")
    res["clause_exact"] = []
    for pA, pB in [(p, p) for p in grid] + [(0.4, 0.2), (0.6, 0.2)]:
        v = [p_clause_exact(n, pA, pB) for n in NS]
        out(f"  {pA:>5.1f}{pB:>6.1f}{100 * v[0]:>9.1f}{100 * v[1]:>9.1f}" + ("   (null)" if pA == pB else ""))
        res["clause_exact"].append({"pA": pA, "pB": pB, **{f"n{n}": x for n, x in zip(NS, v)}})
    return res


# ---------------------------------------------------------------- Part B: collapse threshold

def part_b() -> dict:
    runs = A.load(str(ROOT / "results/gate"))
    out()
    out("=" * 100)
    out("PART B (EXPLORATORY). Collapse-threshold robustness of the E1 H3 label (pre-registered threshold 5)")
    out("=" * 100)
    E = paired_endpoints("results/gate", "lagu-tt", "lagu_nogate-tt", range(100, 115))
    ra = sorted(round(a["return"], 2) for _, a, _ in E)
    rb = sorted(round(b["return"], 2) for _, _, b in E)
    out(f"  held-out returns, sorted. lagu-tt: {ra}")
    out(f"                            lagu_nogate-tt: {rb}")
    out(f"  seeds with a diverged run (prereg.md 11): {[s for s, a, b in E if a['diverged'] or b['diverged']]}")
    out()
    out("EXPLORATORY. Table B1. E1 (lagu-tt - lagu_nogate-tt, 15 pairs, held-out, 10,000 resamples, seed 0):")
    out("  collapse = held-out return < threshold, or diverged.")
    out(f"  {'threshold':>9}{'coll. A':>9}{'coll. B':>9}{'A - B':>7}{'Fisher p':>10}  {'label':<13}clauses that fire")
    rows, saved = [], A.COLLAPSE
    try:
        for thr in THRESHOLDS:
            A.COLLAPSE = float(thr)          # endpoints() reads this module global at call time
            r = A.pair(runs, "lagu-tt", "lagu_nogate-tt", None, ENV, A.N_BOOT, 0).get(ENV)
            assert r["n_pairs"] == 15 and r["source"] == "heldout"
            r95, c95 = r["return"]["ci95"], r["cost"]["ci95"]
            fired = (["interval"] if hurts_interval_clause(r95, c95) else []) + \
                    (["collapse (>= 3 more)"] if r["collapses_a"] - r["collapses_b"] >= 3 else [])
            out(f"  {thr:>9}{r['collapses_a']:>9}{r['collapses_b']:>9}{r['collapses_a'] - r['collapses_b']:>7}"
                f"{r['fisher_p']:>10.3f}  {r['verdict']:<13}{', '.join(fired) or '-'}"
                + ("   <- pre-registered" if thr == 5 else ""))
            rows.append({"threshold": thr, "collapses_a": r["collapses_a"], "collapses_b": r["collapses_b"],
                         "fisher_p": r["fisher_p"], "label": r["verdict"], "clauses": fired,
                         "return_ci95": r95, "cost_ci95": c95})
    finally:
        A.COLLAPSE = saved
    pre = next(x for x in rows if x["threshold"] == 5)
    assert (pre["collapses_a"], pre["collapses_b"], pre["label"]) == (9, 3, "Hurts"), "does not reproduce section 15"
    out(f"  Intervals (identical at every threshold): dReturn 95% [{pre['return_ci95'][0]:.2f}, "
        f"{pre['return_ci95'][1]:.2f}], dCost 95% [{pre['cost_ci95'][0]:.2f}, {pre['cost_ci95'][1]:.2f}];")
    out("  without the collapse clause these give Inconclusive. Threshold 5 reproduces section 15 (9 vs 3, Hurts).")
    return {"rows": rows, "returns_a_sorted": ra, "returns_b_sorted": rb}


# ---------------------------------------------------------------- Part C: E1b bootstrap seed

def part_c() -> dict:
    out()
    out("=" * 100)
    out("PART C (DESCRIPTIVE). E1b label vs bootstrap seed. NOT a re-analysis: the label is the seed-0 one.")
    out("=" * 100)
    E = paired_endpoints("results/gate_literal", "lagu-lit", "lagu_nogate-lit", range(100, 110))
    labs, rl, ch = [], [], []
    for seed in range(N_SEEDS_E1B):
        r = A.contrast(E, A.N_BOOT, seed)
        labs.append(r["verdict"])
        rl.append(r["return"]["ci95"][0])
        ch.append(r["cost"]["ci95"][1])
    labs, rl, ch = np.array(labs), np.array(rl), np.array(ch)
    d = np.array([a["return"] - b["return"] for _, a, b in E])
    frac = {k: float(np.mean(labs == k)) for k in LABELS}
    out(f"DESCRIPTIVE. Table C1. E1b lagu-lit - lagu_nogate-lit, {len(E)} pairs, held-out, {A.N_BOOT:,} "
        f"resamples per seed, seeds 0-{N_SEEDS_E1B - 1}.")
    out(f"  mean dReturn {d.mean():.4f}; seed 0 (pre-registered): label {labs[0]}, dReturn 95% lower bound "
        f"{rl[0]:.4f}, dCost 95% upper bound {ch[0]:.3f}")
    for k in LABELS:
        out(f"  {k:<13}{int(np.sum(labs == k)):>6} of {N_SEEDS_E1B}  ({100 * frac[k]:.2f}%)")
    qs = np.percentile(rl, [0, 2.5, 50, 97.5, 100])
    out(f"  dReturn 95% lower bound over seeds: min {qs[0]:.4f}, 2.5% {qs[1]:.4f}, median {qs[2]:.4f}, "
        f"97.5% {qs[3]:.4f}, max {qs[4]:.4f}; SD {rl.std(ddof=1):.4f}; > 0 in {100 * np.mean(rl > 0):.1f}%")
    out(f"  dCost 95% upper bound over seeds: {ch.min():.3f} to {ch.max():.3f} (Helps' return clause needs < 5:"
        f" {'always met' if ch.max() < A.MARGIN_COST else 'not always met'}), so Helps <=> return lower bound > 0.")
    return {"fractions": frac, "seed0_label": str(labs[0]), "seed0_return_lb": float(rl[0]),
            "return_lb": rl.tolist(), "cost_ub_min": float(ch.min()), "cost_ub_max": float(ch.max()),
            "labels": labs.tolist()}


# ---------------------------------------------------------------- figures

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
LABEL_COLORS = {"Helps": "#2a78d6", "Hurts": "#eb6834", "Shift": "#1baf7a",
                "Equivalent": "#eda100", "Inconclusive": "#e87ba4"}   # reference palette slots 1-5, validated
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_probabilities(res: dict) -> None:
    names = [s[0] for s in SCENARIOS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, n in zip(axes, NS):
        rows = {r["scenario"]: r for r in res["rows"] if r["n"] == n}
        for y, name in enumerate(names):
            left = 0.0
            for lab in LABELS:
                w = 100 * rows[name]["p"][lab]
                if w > 0:
                    ax.barh(y, w, left=left, height=0.7, color=LABEL_COLORS[lab], edgecolor=SURF, linewidth=1.5)
                    if w >= 7:
                        ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=7.5, color=INK)
                left += w
        ax.set_xlim(0, 100)
        ax.set_xlabel("probability of label (% of 2,000 simulated studies)")
        ax.set_title(f"n = {n} pairs", fontsize=10, color=INK)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=LABEL_COLORS[k]) for k in LABELS]
    fig.legend(handles, LABELS, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.55, 1.04))
    fig.suptitle("EXPLORATORY: section 7 label probabilities under E1 noise (scenario = true mean shift / "
                 "collapse rates)", fontsize=10, y=1.10, color=INK)
    save(fig, "fig_label_probabilities")


def fig_robustness(rb: dict, rc: dict) -> None:
    plt.rcParams.update({"font.size": 10.5})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.9))
    t = [r["threshold"] for r in rb["rows"]]
    ca, cb = [r["collapses_a"] for r in rb["rows"]], [r["collapses_b"] for r in rb["rows"]]
    a1.plot(t, ca, "-o", color="#1baf7a", lw=2, ms=6, label="lagu-tt (gate)")      # slot 3 (aqua)
    a1.plot(t, cb, "-s", color="#4a3aa7", lw=2, ms=6, label="lagu_nogate-tt")      # slot 7 (violet)
    a1.axvline(5, color=INK2, lw=1, ls="--")
    a1.text(5.15, 14.3, "pre-specified (5)", fontsize=9.5, color=INK2, va="top")
    labels = {r["label"] for r in rb["rows"]}
    if len(labels) == 1:                         # one label at every threshold: state it once
        a1.text(12, 6.6, f"H3 label: {labels.pop()}\nat every threshold", ha="right", va="top", fontsize=9.5, color=INK2)
    else:
        for x, ya, yb, r in zip(t, ca, cb, rb["rows"]):
            a1.text(x, max(ya, yb) + 0.7, r["label"], ha="center", fontsize=8, color=INK2)
    a1.set_xlabel("collapse threshold (held-out return, return units)")
    a1.set_ylabel("collapsed runs (of 15)")
    a1.set_ylim(0, 15.5)
    a1.set_xticks(t)
    a1.set_title("(a) EXPLORATORY: E1 collapses and H3 label\nvs threshold", fontsize=10.5, color=INK)
    a1.legend(frameon=False, loc="center right", fontsize=9.5)
    lb = np.array(rc["return_lb"])
    w = 0.0005                                   # bin width; 0 is a bin edge, so no bin mixes labels
    bins = w * np.arange(np.floor(lb.min() / w), np.ceil(lb.max() / w) + 1)
    a2.hist(lb[lb <= 0], bins=bins, color=LABEL_COLORS["Equivalent"], edgecolor=SURF, linewidth=0.8,
            label=f"Equivalent ({100 * rc['fractions']['Equivalent']:.2f}%)")
    a2.hist(lb[lb > 0], bins=bins, color=LABEL_COLORS["Helps"], edgecolor=SURF, linewidth=0.8,
            label=f"Helps ({100 * rc['fractions']['Helps']:.2f}%)")
    a2.axvline(0, color=INK, lw=1)
    a2.axvline(rc["seed0_return_lb"], color=INK2, lw=1, ls="--")
    a2.set_ylim(0, a2.get_ylim()[1] * 1.45)      # headroom so the legend and note clear the bars
    a2.text(rc["seed0_return_lb"], a2.get_ylim()[1] * 0.72, "seed 0 \n(pre-specified) ", fontsize=9.5,
            color=INK2, va="top", ha="right")
    a2.set_xlabel("E1b $\\Delta$return 95% lower bound (return units)")
    a2.set_ylabel(f"bootstrap seeds (of {N_SEEDS_E1B:,})")
    a2.set_title("(b) EXPLORATORY (post hoc): E1b label\nvs bootstrap seed", fontsize=10.5, color=INK)
    a2.legend(loc="upper right", title="label (share of seeds)", title_fontsize=9.5, fontsize=9.5,
               frameon=True, facecolor=SURF, edgecolor=SURF, framealpha=1)
    for ax in (a1, a2):
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_label_robustness")


def main() -> None:
    E1 = paired_endpoints("results/gate", "lagu-tt", "lagu_nogate-tt", range(100, 115))
    ra = part_a(E1)
    rb = part_b()
    rc = part_c()
    fig_probabilities(ra)
    fig_robustness(rb, rc)
    (OUT / "tables.txt").write_text("\n".join(LINES) + "\n")
    with open(OUT / "results.json", "w") as f:
        json.dump({"part_a_simulation": ra, "part_b_threshold": rb, "part_c_e1b_seed": rc}, f, indent=1)
    out()
    out(f"wrote {OUT / 'tables.txt'}, results.json, fig_label_probabilities.{{pdf,png}}, "
        f"fig_label_robustness.{{pdf,png}}")


if __name__ == "__main__":
    main()
