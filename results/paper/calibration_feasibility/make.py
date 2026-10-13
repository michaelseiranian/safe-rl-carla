"""DESCRIPTIVE tables for the paper: cost-critic calibration, feasibility counts, and the E0 health
rule on the confirmatory ungated seeds. Nothing here is confirmatory; no hypothesis is tested.

Run from the repository root (so that `import lagu` works):

    .venv/bin/python results/paper/calibration_feasibility/make.py

It prints every table and writes, next to this file: tables.md (the same tables), per_run.csv
(one row per run) and calibration_dots.{pdf,png} (per-run held-out calibration).

(1) DESCRIPTIVE. Held-out calibration of the cost critic, sum(qc_traj) / sum(mc_cost_traj) pooled
    over seeds and the 5 snapshots of each run's <stem>.heldout.json (prereg.md 5 and 8/H5; the
    computation is lagu.analyze.endpoints + pooled_calib, which produced the H5 numbers), with a 95%
    percentile bootstrap over seeds (10,000 resamples, numpy.random.default_rng(0) per arm).
    Columns, in this order: the pre-registered pooled value; the same without lagu_nogate-tt s103
    (PointGoal1 E1 arm only); the median of per-run calibration (each run's own pooled ratio over
    its 5 snapshots). Then the in-training calibration of the E1 arms (seeds 100-114) from the
    evaluation rows at step 1M and at the final step (2M): sum(qc_traj) / sum(mc_cost_traj) over
    seeds of that one row, same bootstrap.
(2) DESCRIPTIVE. Feasibility per arm, on the held-out primary endpoint: runs with cost <= 25; runs
    with return >= 15 and cost <= 25; collapses (return < 5, or a diverged run, prereg.md 5 and
    11), split into cost <= 25 ("safe stalls") and cost > 25. The return thresholds are in
    PointGoal1 units and are applied unchanged to CarGoal1 and PointCircle1.
(3) DESCRIPTIVE. Parts (a) calibration and (b) no collapse of the E0 rule (prereg.md 10.2), read
    with lagu.analyze.e0_rule, applied to the 15 confirmatory lagu_nogate-tt PointGoal1 seeds
    (100-114) on their evaluation rows at 850k, 900k, 950k and 1M: each disjoint 3-seed block
    (100-102, ..., 112-114) as one E0 cell, and all 15 seeds as one cell. The E0 cell that chose
    this recipe (lagu_nogate-tt-l2e-6, selection seeds 0-2) is recomputed as the reference row.

Every seed listed in the task is used and no run is excluded; the 'without s103' column is an
additional, clearly labelled view of one arm only.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import lagu.analyze as A  # noqa: E402

N_BOOT, BOOT_SEED = 10_000, 0
PG = "SafetyPointGoal1-v0"
DIRS = ["results/gate", "results/gate_literal", "results/gate_literal_x", "results/gate_x"]
S = lambda a, b: list(range(a, b + 1))
# (dir, env, arm, experiment, expected seeds); the table order
ARMS = [
    ("results/gate", PG, "lagu-tt", "E1 + H4 ext.", S(100, 119)),
    ("results/gate", PG, "lagu_nogate-tt", "E1", S(100, 114)),
    ("results/gate", PG, "lagu_nogate-tt-match", "E2 (δ0 = 1.12158)", S(100, 119)),
    ("results/gate", PG, "lagu-tt-ggrad", "gate gradient", S(100, 109)),
    ("results/gate", PG, "td3lag-tt", "TD3-Lag", S(100, 109)),
    ("results/gate", PG, "lagu_nogate-tt-d175", "E4 (δ0 = 1.75)", S(100, 107)),
    ("results/gate", PG, "lagu_nogate-tt-d125", "E4 (δ0 = 1.25)", S(100, 107)),
    ("results/gate", PG, "lagu-tt-T035", "E4 (T = 0.035)", S(100, 107)),
    ("results/gate", PG, "lagu-tt-T14", "E4 (T = 0.14)", S(100, 107)),
    ("results/gate_literal", PG, "lagu-lit", "E1b", S(100, 109)),
    ("results/gate_literal", PG, "lagu_nogate-lit", "E1b", S(100, 109)),
    ("results/gate_literal", PG, "td3lag-lit", "E1b TD3-Lag", S(100, 109)),
    ("results/gate_literal_x", PG, "lagu-lit-T03", "X1 (T = 0.03)", S(100, 109)),
    ("results/gate_literal_x", PG, "lagu_nogate-lit-match", "X1 (δ0 = 2.14758)", S(100, 109)),
]
for env in ("SafetyCarGoal1-v0", "SafetyPointCircle1-v0"):
    ARMS += [("results/gate_x", env, "lagu-tt", "E3", S(100, 109)),
             ("results/gate_x", env, "lagu_nogate-tt", "E3", S(100, 109)),
             ("results/gate_x", env, "td3lag-tt", "E3 TD3-Lag", S(100, 104))]
E1_SEEDS = S(100, 114)
DROP = ("results/gate", PG, "lagu_nogate-tt", 103)       # the run whose critics blew up (prereg 15)
BLOCKS = [S(s, s + 2) for s in range(100, 115, 3)]
OUT_MD = []


# ---------------------------------------------------------------- helpers

def out(s: str = "") -> None:
    print(s)
    OUT_MD.append(s)


def table(title: str, header: list[str], rows: list[list[str]], notes: list[str] = ()) -> None:
    out(f"### {title}")
    out()
    out("| " + " | ".join(header) + " |")
    out("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        out("| " + " | ".join(r) + " |")
    out()
    for n in notes:
        out(n)
    if notes:
        out()


def f(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    a = abs(x)
    return f"{x:.1f}" if a >= 100 else f"{x:.2f}" if a >= 10 else f"{x:.3f}"


def fci(v: tuple) -> str:
    r, lo, hi = v
    return f"{f(r)} [{f(lo)}, {f(hi)}]"


def seeds_str(seeds) -> str:
    seeds = sorted(seeds)
    return f"{seeds[0]}-{seeds[-1]}" if seeds == S(seeds[0], seeds[-1]) else ",".join(map(str, seeds))


def short(env: str) -> str:
    return env.replace("Safety", "").replace("-v0", "")


def pooled(E: list[dict]) -> tuple:
    """sum(qc) / sum(mc) over runs and its 95% seed-bootstrap interval (lagu.analyze.pooled_calib)."""
    return A.pooled_calib(E, N_BOOT, BOOT_SEED)


def median_calib(E: list[dict]) -> tuple[float, int]:
    c = np.array([e["calib"] for e in E], float)
    return A.fmedian(c), int((~np.isfinite(c)).sum())


# ---------------------------------------------------------------- load and check coverage

loaded = {d: A.load(d) for d in DIRS}
G = {}
for d, runs in loaded.items():
    for (env, lab), g in A.cells(runs).items():
        G[(d, env, lab)] = g
missing = []
for d, env, lab, _, seeds in ARMS:
    have = sorted(G.get((d, env, lab), {}))
    if have != seeds:
        missing.append(f"{d} {env} {lab}: have {have}, expected {seeds}")
listed = {(d, env, lab) for d, env, lab, _, _ in ARMS}
extra = [k for k in G if k not in listed]
if missing or extra:
    raise SystemExit(f"run coverage differs from the task list: missing {missing}, unlisted {extra}")
for k, g in G.items():
    for s, r in g.items():
        if not A.heldout_ok(r):
            raise SystemExit(f"{r['stem']}: no held-out file for its end step")
        if not A.end_of(r) == r["rows"][-1]["step"]:
            raise SystemExit(f"{r['stem']}: not at its final step")

EP = {k: {s: A.endpoints(r, held=True) for s, r in g.items()} for k, g in G.items()}

out("# Cost-critic calibration, feasibility, and the E0 health rule on E1 seeds (DESCRIPTIVE)")
out()
out("Generated by `results/paper/calibration_feasibility/make.py`. All tables are DESCRIPTIVE; none "
    "is part of the confirmatory analysis. Intervals are 95% percentile bootstraps over seeds "
    f"({N_BOOT:,} resamples, numpy.random.default_rng({BOOT_SEED}) per row). Calibration is "
    "Σ qc_traj / Σ mc_cost_traj: the cost critic's mean prediction over on-policy states with at "
    "least half an episode left, divided by the discounted cost-to-go realised from those states "
    "(1 = calibrated, < 1 = underestimates cost). No run is diverged and every run has its held-out "
    "file at its final step.")
out()

# ---------------------------------------------------------------- (1) held-out calibration

rows, csv_rows = [], []
for d, env, lab, exp, seeds in ARMS:
    E = [EP[(d, env, lab)][s] for s in seeds]
    med, n_nan = median_calib(E)
    c = np.array([e["calib"] for e in E])
    wo = "-"
    if (d, env, lab) == DROP[:3]:
        wo = fci(pooled([EP[(d, env, lab)][s] for s in seeds if s != DROP[3]]))
    rows.append([f"{short(env)} {d.split('/')[-1]}", f"`{lab}`", exp, seeds_str(seeds), str(len(E)),
                 fci(pooled(E)), wo, f(med) + (f" ({n_nan} undefined)" if n_nan else ""),
                 f"{f(np.nanmin(c))} to {f(np.nanmax(c))}", str(sum(e["qc"] < 0 for e in E))])
    if lab == "lagu-tt" and d == "results/gate":   # the pre-registered E1 seed set, as an extra row
        E1 = [EP[(d, env, lab)][s] for s in E1_SEEDS]
        c1 = np.array([e["calib"] for e in E1])
        rows.append([f"{short(env)} gate", f"`{lab}`", "E1 seeds only", seeds_str(E1_SEEDS), str(len(E1)),
                     fci(pooled(E1)), "-", f(median_calib(E1)[0]),
                     f"{f(np.nanmin(c1))} to {f(np.nanmax(c1))}", str(sum(e["qc"] < 0 for e in E1))])
    for s in seeds:
        e, r = EP[(d, env, lab)][s], G[(d, env, lab)][s]
        snaps = r["heldout"]["per_snapshot"]
        csv_rows.append({"dir": d, "env": env, "arm": lab, "experiment": exp, "seed": s,
                         "resumed": int(e["resumed"]), "heldout_return": e["return"], "heldout_cost": e["cost"],
                         "sum_qc_traj": e["qc"], "sum_mc_cost_traj": e["mc"], "calibration": e["calib"],
                         "min_snapshot_qc_traj": min(x["qc_traj"] for x in snaps),
                         "diverged": int(e["diverged"]), "collapsed": int(e["collapsed"]),
                         "cost_le_25": int(e["cost"] <= A.COST_LIMIT),
                         "return_ge_15_and_cost_le_25": int(e["return"] >= 15 and e["cost"] <= A.COST_LIMIT)})

table("Table 1. Held-out cost-critic calibration per arm (DESCRIPTIVE)",
      ["Task / dir", "Arm", "Experiment", "Seeds", "n", "Pooled Σqc/Σmc [95% CI] (pre-registered statistic)",
       "Pooled without s103 [95% CI]", "Median per-run", "Per-run range", "Runs with Σqc < 0"],
      rows,
      ["Pooled: Σ qc_traj / Σ mc_cost_traj over the 5 held-out snapshots (20 layouts each) and all seeds "
       "of the arm; this is the H5 statistic of prereg.md 8. Per-run: the same ratio over one run's 5 "
       "snapshots. 'Without s103' drops only `lagu_nogate-tt` s103 (PointGoal1), whose cost critic "
       "diverged in value (per-run calibration about 1100) without a non-finite entry, so it is not a "
       "diverged run under prereg.md 11 and stays in the pre-registered column. A pooled ratio is "
       "unstable when a few runs have large or negative Σqc (the critic predicts negative cost on some "
       "states although costs are ≥ 0) or a small Σmc (collapsed runs that incur little cost); the "
       "median and the range are given for that reason. The 'E1 seeds only' row is the pre-registered "
       "E1 seed set of `lagu-tt` (100-114), for comparison with `lagu_nogate-tt`."])

# in-training calibration of the E1 arms at 1M and at the final step
rows = []
for lab in ("lagu-tt", "lagu_nogate-tt"):
    g = G[("results/gate", PG, lab)]
    for name, pick in (("1M", lambda r: [x for x in r["rows"] if x["step"] == 1_000_000]),
                       ("final (2M)", lambda r: [r["rows"][-1]])):
        E, per = {}, {}
        for s in E1_SEEDS:
            row = pick(g[s])
            assert len(row) == 1, (g[s]["stem"], name)
            row = row[0]
            assert name == "1M" or row["step"] == 2_000_000, g[s]["stem"]
            E[s] = {"qc": row["qc_traj"], "mc": row["mc_cost_traj"]}
            per[s] = A.ratio_sum([row["qc_traj"]], [row["mc_cost_traj"]])
        pr = np.array(list(per.values()))
        wo = fci(pooled([E[s] for s in E1_SEEDS if s != DROP[3]])) if lab == DROP[2] else "-"
        und = int((~np.isfinite(pr)).sum())
        rows.append([f"`{lab}`", name, seeds_str(E1_SEEDS), str(len(E)), fci(pooled(list(E.values()))), wo,
                     f(A.fmedian(pr)) + (f" ({und} undefined)" if und else ""),
                     f"{f(np.nanmin(pr))} to {f(np.nanmax(pr))}", str(sum(e["qc"] < 0 for e in E.values()))])
table("Table 1b. In-training cost-critic calibration of the E1 arms, PointGoal1 (DESCRIPTIVE)",
      ["Arm", "Row", "Seeds", "n", "Pooled Σqc/Σmc [95% CI]", "Pooled without s103 [95% CI]",
       "Median per-run", "Per-run range", "Runs with qc < 0"],
      rows,
      ["One in-training evaluation row per run (10 deterministic episodes on the in-training layouts "
       "EVAL_SEED + k, not the held-out layouts), pooled over seeds 100-114. A per-run ratio at one "
       "row is undefined when that row's mc_cost_traj is 0."])

# ---------------------------------------------------------------- (2) feasibility

rows = []
for d, env, lab, exp, seeds in ARMS:
    variants = [(exp, seeds)]
    if lab == "lagu-tt" and d == "results/gate":
        variants.append(("E1 seeds only", E1_SEEDS))
    for ex, ss in variants:
        E = [EP[(d, env, lab)][s] for s in ss]
        ret, cost = np.array([e["return"] for e in E]), np.array([e["cost"] for e in E])
        col = np.array([e["collapsed"] for e in E])
        assert np.all(np.isfinite(ret)) and np.all(np.isfinite(cost))
        rows.append([f"{short(env)} {d.split('/')[-1]}", f"`{lab}`", ex, seeds_str(ss), str(len(E)),
                     str(int((cost <= 25).sum())), str(int(((ret >= 15) & (cost <= 25)).sum())),
                     str(int(col.sum())), str(int((col & (cost <= 25)).sum())), str(int((col & (cost > 25)).sum()))])
table("Table 2. Feasibility on the held-out primary endpoint, counts of runs (DESCRIPTIVE)",
      ["Task / dir", "Arm", "Experiment", "Seeds", "n", "Cost ≤ 25", "Return ≥ 15 and cost ≤ 25",
       "Collapsed (return < 5)", "Safe stalls (collapsed, cost ≤ 25)", "Collapsed, cost > 25"],
      rows,
      ["Held-out return and cost: mean over 100 episodes (5 snapshots × 20 held-out layouts). Collapse "
       "follows prereg.md 5 and 11 (return < 5 or diverged; no run diverged). The return thresholds "
       "15 and 5 are PointGoal1 units and are applied unchanged to CarGoal1 and PointCircle1, whose "
       "return scales differ."])

# ---------------------------------------------------------------- (3) E0 health rule on E1 seeds

def e0_ab(runs, seeds) -> dict:
    res = A.e0_rule(runs, end=1_000_000, env=PG, seeds=set(seeds))
    (c,) = res["cells"].values()
    assert all(w == [850_000, 900_000, 950_000, 1_000_000] for w in c["window_steps"]), c["window_steps"]
    return c


def e0_row(name, runs, seeds) -> list[str]:
    c = e0_ab(runs, seeds)
    by_seed = {r["seed"]: r for r in runs}
    neg = [s for s in c["seeds"] if np.any(A.col(by_seed[s], "qc_traj", 850_000, 1_000_000) < 0)]
    return [name, seeds_str(c["seeds"]), str(4 * len(c["seeds"])), f(c["calib"]),
            ", ".join(f(x) for x in c["calib_seed"]), ", ".join(map(str, neg)) or "none",
            "pass" if c["a"] else "FAIL", f(c["return"]), ", ".join(f"{x:.1f}" for x in c["return_seed"]),
            ", ".join(map(str, c["diverged_seeds"])) or "none", "pass" if c["b"] else "FAIL",
            "healthy" if c["a"] and c["b"] else "not healthy"]


ng = [r for r in loaded["results/gate"] if r["arm"] == "lagu_nogate-tt" and r["env"] == PG]
e0 = [r for r in A.load("results/e0_dual") if r["arm"] == "lagu_nogate-tt-l2e-6"]
ref = e0_ab(e0, [0, 1, 2])
dec = json.load(open("results/e0_dual/e0_decision.json"))["cells"]["lagu_nogate-tt-l2e-6"]
assert ref["a"] == dec["a"] and ref["b"] == dec["b"] and np.isclose(ref["calib"], dec["calib"]), "E0 recompute differs"
rows = [e0_row("E0 selection cell `lagu_nogate-tt-l2e-6` (reference)", e0, [0, 1, 2])]
rows += [e0_row(f"block {i + 1}", ng, b) for i, b in enumerate(BLOCKS)]
rows += [e0_row("all 15 seeds as one cell", ng, E1_SEEDS)]
table("Table 3. E0 rule parts (a)-(b) on the confirmatory `lagu_nogate-tt` seeds, PointGoal1, rows 850k-1M (DESCRIPTIVE)",
      ["Cell", "Seeds", "Rows", "(a) Σqc/Σmc", "(a) per-seed ratio", "(a) seeds with a row qc < 0", "(a)",
       "(b) mean return", "(b) per-seed mean return", "(b) diverged seeds", "(b)", "Healthy = (a) and (b)"],
      rows,
      ["Rule (prereg.md 10.2), applied with `lagu.analyze.e0_rule` to each seed's in-training evaluation "
       "rows at 850k, 900k, 950k and 1M. (a): pooled Σ qc_traj / Σ mc_cost_traj over the cell's rows in "
       "[0.65, 1.35], every seed's ratio over its own 4 rows ≥ 0.5, and no row with qc_traj < 0. (b): "
       "mean eval_return over the cell's rows ≥ 20 and no seed's 4-row mean below 10 (a diverged seed "
       "counts as below 10). Blocks are the disjoint consecutive triples 100-102, ..., 112-114, fixed "
       "before reading any value. These confirmatory runs are 2M-step runs of the recipe that E0 "
       "chose on seeds 0-2; the window is the E0 window (850k-1M), not their primary window."])

# ---------------------------------------------------------------- files

with open(os.path.join(HERE, "tables.md"), "w") as fh:
    fh.write("\n".join(OUT_MD) + "\n")
with open(os.path.join(HERE, "per_run.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
    w.writeheader()
    w.writerows(csv_rows)

# ---------------------------------------------------------------- optional figure: per-run calibration

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
RUN, POOL = "#2a78d6", "#eb6834"          # categorical slots 1-2 of the dataviz reference palette
LO, HI = -1.0, 3.0                        # display window; values outside are drawn at the edge
labels = [f"{lab}  ({short(env)}, {exp}, n={len(seeds)})" for d, env, lab, exp, seeds in ARMS]
fig, ax = plt.subplots(figsize=(12, 0.36 * len(ARMS) + 1.9), facecolor=SURF)
fig.subplots_adjust(left=0.35, right=0.76, top=0.92, bottom=0.12)
ax.set_facecolor(SURF)
ax.axvspan(0.65, 1.35, color=GRID, alpha=0.6, lw=0, zorder=0)
ax.axvline(1.0, color=INK2, lw=1, zorder=1)
ax.axvline(0.0, color=GRID, lw=1, zorder=1)
for i, (d, env, lab, exp, seeds) in enumerate(ARMS):
    y = len(ARMS) - 1 - i
    E = [EP[(d, env, lab)][s] for s in seeds]
    c = np.array([e["calib"] for e in E])
    n_undef = int((~np.isfinite(c)).sum())
    c = c[np.isfinite(c)]
    jit = np.linspace(-0.18, 0.18, len(c))
    inside = (c >= LO) & (c <= HI)
    ax.scatter(c[inside], y + jit[inside], s=22, color=RUN, edgecolor=SURF, linewidth=0.8, zorder=3)
    note = []
    for side, m, xx in ((c < LO, "<", LO), (c > HI, ">", HI)):
        if side.any():
            ax.scatter(np.full(side.sum(), xx), y + jit[side], marker=m, s=34, color=RUN, zorder=3, clip_on=False)
            note.append(", ".join(f"{v:.3g}" for v in sorted(c[side])))
    p = pooled(E)[0]
    ax.scatter([min(max(p, LO), HI)], [y], marker="D", s=46, color=POOL, edgecolor=INK, linewidth=0.6,
               zorder=4, clip_on=False)
    if not LO <= p <= HI:
        note.append(f"pooled {p:.3g}")
    if n_undef:
        note.append(f"{n_undef} undefined (Σmc = 0), not drawn")
    if note:
        ax.annotate("; ".join(note), (1.02, y), xycoords=("axes fraction", "data"), ha="left", va="center",
                    fontsize=7, color=INK2, annotation_clip=False)
ax.annotate("values outside the axis", (1.02, len(ARMS) - 0.35), xycoords=("axes fraction", "data"),
            ha="left", va="bottom", fontsize=7.5, color=INK, annotation_clip=False)
ax.set_yticks(range(len(ARMS)))
ax.set_yticklabels(labels[::-1], fontsize=7.5, color=INK)
ax.set_xlim(LO - 0.12, HI + 0.12)
ax.set_ylim(-0.7, len(ARMS) - 0.3)
ax.set_xlabel("Held-out cost-critic calibration, Σ qc_traj / Σ mc_cost_traj (dimensionless; 1 = calibrated)",
              fontsize=8.5, color=INK)
ax.tick_params(colors=INK, labelsize=7.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(INK2)
h = [ax.scatter([], [], s=22, color=RUN), ax.scatter([], [], marker="D", s=46, color=POOL, edgecolor=INK, linewidth=0.6),
     ax.fill_between([], [], color=GRID, alpha=0.6)]
fig.legend(h, ["one run (5 snapshots × 20 held-out layouts)", "arm, pooled over seeds (pre-registered statistic)",
               "[0.65, 1.35] band (prereg H5 / E0 (a))"],
           loc="lower center", ncol=3, fontsize=7.5, frameon=False, labelcolor=INK)
fig.suptitle("DESCRIPTIVE: per-run cost-critic calibration on held-out layouts "
             f"(values outside [{LO:g}, {HI:g}] drawn at the edge as ◀/▶ and listed at right)",
             fontsize=9.5, color=INK, x=0.02, ha="left")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"calibration_dots.{ext}"), dpi=200, facecolor=SURF)
print(f"wrote {HERE}/tables.md, per_run.csv, calibration_dots.pdf/png")
