"""DESCRIPTIVE. The pre-registered secondary prediction '|Q_bar| feedback loop' (prereg.md section 9,
outside the Holm family), evaluated as committed in prereg.md section 16 item 5. Nothing here is
confirmatory.

Exact wording (prereg.md section 9):
    "If the median of |qbar_avg| over lagu-tt rows after 300k (all seeds) falls below 0.3, then
    engagement over those rows exceeds 80% and the ratio of the cell means of final lambda,
    lagu / lagu_nogate, exceeds 3."
Section 16 item 5: evaluated on lagu-tt seeds 100-114 (the E1 seeds), with seeds 100-119 as a
sensitivity check.

Definitions (fixed before reading the values; they are those of lagu.analyze.qbar_loop):
- Rows "after 300k": the interval rows with 300k < step <= N (N = 2M): 34 rows per run. The row at
  300k summarises (250k, 300k] and is excluded.
- Median |qbar_avg|: the median of |qbar_avg| pooled over those rows of every lagu-tt run in the set.
- Engagement over those rows (prereg.md section 5): sum(engagement * n_risk) / sum(n_risk) over the
  same rows, pooled over seeds (the fraction of risk-sensitive samples on which the gate fired).
- Final lambda: the 'lambda' column of a run's last row (step 2M). Cell mean: the arithmetic mean
  over the cell's seeds. The ratio is mean(lagu-tt) / mean(lagu_nogate-tt); lagu_nogate-tt exists
  only for seeds 100-114 (prereg.md 16 item 5), so both seed sets use those 15 runs in the
  denominator. No run is excluded (lagu_nogate-tt s103 stays in).
- 95% intervals (DESCRIPTIVE): percentile bootstrap over runs, 10,000 resamples,
  numpy.random.default_rng(0). Each resample draws lagu-tt seeds with replacement (rows of a drawn
  run move together) and, for the ratio, lagu_nogate-tt seeds independently.
The script cross-checks every point value against lagu.analyze.qbar_loop (read only).

Run from the repository root:  .venv/bin/python results/paper/qbar_loop/make.py
Writes table.md, qbar_loop.json and qbar_loop.{pdf,png} next to this file and prints the tables.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
OUT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results/gate")
ENV = "SafetyPointGoal1-v0"
A, B = "lagu-tt", "lagu_nogate-tt"
SETS = {"primary (seeds 100-114)": list(range(100, 115)),
        "sensitivity (seeds 100-119)": list(range(100, 120))}
B_SEEDS = list(range(100, 115))
LO, N = 300_000, 2_000_000                 # rows with LO < step <= N
THR_Q, THR_ENG, THR_RATIO = 0.3, 0.8, 3.0
N_BOOT = 10_000


def load(arm, seed):
    with open(f"{R}/{arm}_{ENV}_s{seed}.csv") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    step = col("step")
    assert len(np.unique(step)) == len(step), f"{arm} s{seed}: duplicate steps"
    assert step[-1] == N, f"{arm} s{seed}: last step {step[-1]:.0f}, expected {N}"
    w = (step > LO) & (step <= N)
    run = {"step": step, "qbar_avg": col("qbar_avg"), "engagement": col("engagement"),
           "n_risk": col("n_risk"), "lambda": col("lambda"), "resumed": bool(np.nanmax(col("resumed")) == 1)}
    run["absq_w"], run["eng_w"], run["nrisk_w"] = (np.abs(run["qbar_avg"][w]), run["engagement"][w],
                                                   run["n_risk"][w])
    assert all(np.all(np.isfinite(run[k])) for k in ("absq_w", "eng_w", "nrisk_w")), f"{arm} s{seed}: non-finite"
    run["final_lambda"] = float(run["lambda"][-1])
    return run


def stats(Q, E, NR, La, Lb):
    """Pooled median |qbar_avg|, pooled engagement, ratio of cell means. Q/E/NR: (runs, rows)."""
    return (float(np.median(Q)), float((E * NR).sum() / NR.sum()), float(La.mean() / Lb.mean()))


def analyse(lagu, nogate, seeds):
    Q = np.stack([lagu[s]["absq_w"] for s in seeds])
    E = np.stack([lagu[s]["eng_w"] for s in seeds])
    NR = np.stack([lagu[s]["nrisk_w"] for s in seeds])
    La = np.array([lagu[s]["final_lambda"] for s in seeds])
    Lb = np.array([nogate[s]["final_lambda"] for s in B_SEEDS])
    med, eng, ratio = stats(Q, E, NR, La, Lb)
    rng = np.random.default_rng(0)
    ia = rng.integers(0, len(seeds), (N_BOOT, len(seeds)))
    ib = rng.integers(0, len(B_SEEDS), (N_BOOT, len(B_SEEDS)))
    b_med = np.median(Q[ia].reshape(N_BOOT, -1), axis=1)
    b_eng = (E * NR)[ia].sum((1, 2)) / NR[ia].sum((1, 2))
    b_ratio = La[ia].mean(1) / Lb[ib].mean(1)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"n_lagu_runs": len(seeds), "n_nogate_runs": len(B_SEEDS), "n_rows": int(Q.size),
            "median_abs_qbar_after_300k": med, "median_abs_qbar_ci95": ci(b_med),
            "engagement_after_300k": eng, "engagement_ci95": ci(b_eng),
            "mean_final_lambda_lagu": float(La.mean()), "mean_final_lambda_nogate": float(Lb.mean()),
            "final_lambda_ratio": ratio, "final_lambda_ratio_ci95": ci(b_ratio),
            "antecedent_met": bool(med < THR_Q), "engagement_gt_0.8": bool(eng > THR_ENG),
            "ratio_gt_3": bool(ratio > THR_RATIO),
            "prediction_holds": bool(med < THR_Q and eng > THR_ENG and ratio > THR_RATIO)}


def crosscheck(res):
    """Every point value must equal lagu.analyze.qbar_loop on the same runs (read-only import)."""
    from lagu import analyze
    runs = [r for r in analyze.load(R) if r["env"] == ENV and r["arm"] in (A, B)]
    for name, seeds in SETS.items():
        sel = [r for r in runs if (r["arm"] == A and r["seed"] in seeds) or (r["arm"] == B and r["seed"] in B_SEEDS)]
        ref = analyze.qbar_loop(sel, A, B, ENV)
        for k in ("median_abs_qbar_after_300k", "engagement_after_300k", "final_lambda_ratio"):
            assert abs(ref[k] - res[name][k]) <= 1e-12 * max(1.0, abs(ref[k])), (name, k, ref[k], res[name][k])
        assert ref["triggered"] == res[name]["antecedent_met"] and ref["holds"] == res[name]["prediction_holds"]
    return "lagu.analyze.qbar_loop agrees on every point value and on both booleans (tolerance 1e-12)"


def fmt_ci(x, c, d=3):
    return f"{x:.{d}f} [{c[0]:.{d}f}, {c[1]:.{d}f}]"


def tables(res, lagu, nogate):
    yn = lambda b: "yes" if b else "no"
    t = ["### DESCRIPTIVE - prereg section 9 |Q̄| feedback-loop prediction (SafetyPointGoal1-v0, 2M steps)",
         "",
         "Rows: lagu-tt interval rows with 300k < step <= 2M (34 per run). Final λ: `lambda` at 2M; "
         "denominator lagu_nogate-tt seeds 100-114 in both sets. 95% percentile bootstrap over runs "
         "(10,000 resamples, seed 0).",
         "",
         "| seed set | lagu-tt runs | median \\|qbar_avg\\| (< 0.3?) | engagement (> 0.80?) "
         "| final-λ ratio lagu / lagu_nogate (> 3?) | antecedent met | prediction holds |",
         "|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        t.append(f"| {name} | {r['n_lagu_runs']} | {fmt_ci(r['median_abs_qbar_after_300k'], r['median_abs_qbar_ci95'])} "
                 f"({yn(r['antecedent_met'])}) | {fmt_ci(r['engagement_after_300k'], r['engagement_ci95'])} "
                 f"({yn(r['engagement_gt_0.8'])}) | {fmt_ci(r['final_lambda_ratio'], r['final_lambda_ratio_ci95'], 4)} "
                 f"({yn(r['ratio_gt_3'])}) | {yn(r['antecedent_met'])} | {yn(r['prediction_holds'])} |")
    t += ["", "Cell means of final λ: " + "; ".join(
        f"{name}: lagu-tt {r['mean_final_lambda_lagu']:.4f}, lagu_nogate-tt {r['mean_final_lambda_nogate']:.2f}"
        for name, r in res.items()) + ".", "",
        "### DESCRIPTIVE - per run (no run excluded)", "",
        "| seed | lagu-tt median \\|qbar_avg\\| after 300k | lagu-tt engagement after 300k | lagu-tt final λ "
        "| lagu-tt resumed | lagu_nogate-tt final λ | lagu_nogate-tt resumed |",
        "|---|---|---|---|---|---|---|"]
    for s in sorted(lagu):
        a = lagu[s]
        eng = float((a["eng_w"] * a["nrisk_w"]).sum() / a["nrisk_w"].sum())
        b = nogate.get(s)
        nb = (f"{b['final_lambda']:.4g} | {yn(b['resumed'])}" if b else "not run | -")
        t.append(f"| {s} | {np.median(a['absq_w']):.3f} | {eng:.3f} | {a['final_lambda']:.4g} | {yn(a['resumed'])} | {nb} |")
    return "\n".join(t)


def figure(res, lagu, nogate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    SURF, INK, INK2, GRID, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e1dc", "#efeeea"
    C1, C2 = "#2a78d6", "#eb6834"                      # reference categorical slots 1-2 (CVD-validated)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5})
    prim, sens = SETS["primary (seeds 100-114)"], SETS["sensitivity (seeds 100-119)"]
    rp, rs = res["primary (seeds 100-114)"], res["sensitivity (seeds 100-119)"]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.1), constrained_layout=True, facecolor=SURF)
    for a in ax:
        a.set_facecolor(SURF)
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    colour = lambda s: C1 if s in prim else C2
    BOX = dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.9)
    for i, (key, thr, val_p, val_s, ttl, ylab) in enumerate((
            ("qbar_avg", THR_Q, rp["median_abs_qbar_after_300k"], rs["median_abs_qbar_after_300k"],
             "(a) |qbar_avg| of lagu-tt, every run", "|qbar_avg|  (reward-critic units, log scale)"),
            ("engagement", THR_ENG, rp["engagement_after_300k"], rs["engagement_after_300k"],
             "(b) gate engagement of lagu-tt, every run", "engagement  (fraction of risk-sensitive samples)"))):
        a = ax[i]
        a.axvspan(0, LO / 1e6, color=SHADE, lw=0, zorder=0)
        for s in sens:
            y = lagu[s][key]
            a.plot(lagu[s]["step"] / 1e6, np.abs(y) if key == "qbar_avg" else y, color=colour(s), lw=0.9,
                   alpha=0.55 if s in prim else 0.8, zorder=2)
        a.axhline(thr, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=3)
        a.plot([LO / 1e6, N / 1e6], [val_p, val_p], color=INK, lw=2.2, zorder=4)
        a.plot([LO / 1e6, N / 1e6], [val_s, val_s], color=INK, lw=2.2, ls=(0, (1.2, 1.6)), zorder=4)
        a.set_xlim(0, 2.0)
        a.set_xlabel("environment steps (millions)")
        a.set_ylabel(ylab)
        a.set_title(ttl, loc="left")
        a.text(0.15, 0.98 if key == "qbar_avg" else 0.02, "rows not\nin window", transform=a.get_xaxis_transform(),
               ha="center", va="top" if key == "qbar_avg" else "bottom", color=INK2, fontsize=7.5, bbox=BOX, zorder=5)
        if key == "qbar_avg":
            a.set_yscale("log")
            a.text(0.32, thr * 0.85, f"threshold {thr}", color=INK2, fontsize=7.5, ha="left", va="top", bbox=BOX, zorder=5)
            a.text(0.32, val_p * 1.3, f"pooled median {val_p:.2f} (s100-114) / {val_s:.2f} (s100-119)", color=INK,
                   fontsize=7.5, ha="left", va="bottom", bbox=BOX, zorder=5)
        else:
            a.set_ylim(0, 1.02)
            a.text(0.32, 0.36, f"threshold {thr} (dashed)\npooled {val_p:.3f} (s100-114) / {val_s:.3f} (s100-119)",
                   color=INK, fontsize=7.5, ha="left", va="top", bbox=BOX, zorder=5)
    # (c) final lambda per run
    a = ax[2]
    rng = np.random.default_rng(1)                     # jitter only
    groups = [("lagu-tt\ns100-114", [lagu[s]["final_lambda"] for s in prim], C1),
              ("lagu-tt\ns115-119", [lagu[s]["final_lambda"] for s in sens if s not in prim], C2),
              ("lagu_nogate-tt\ns100-114", [nogate[s]["final_lambda"] for s in B_SEEDS], INK2)]
    for x, (_, v, c) in enumerate(groups):
        a.scatter(x + rng.uniform(-0.12, 0.12, len(v)), v, s=30, color=c, edgecolor=SURF, linewidth=1.0, zorder=3)
    a.plot([-0.3, 0.3], [rp["mean_final_lambda_lagu"]] * 2, color=INK, lw=2.2, zorder=4)
    a.plot([-0.3, 1.3], [rs["mean_final_lambda_lagu"]] * 2, color=INK, lw=2.2, ls=(0, (1.2, 1.6)), zorder=4)
    a.plot([1.7, 2.3], [rp["mean_final_lambda_nogate"]] * 2, color=INK, lw=2.2, zorder=4)
    a.annotate("s103", (2.12, nogate[103]["final_lambda"]), xytext=(1.45, nogate[103]["final_lambda"] / 3),
               fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a.set_yscale("log")
    a.set_xticks(range(3), [g[0] for g in groups])
    a.set_xlim(-0.5, 2.5)
    a.set_ylabel("final λ at 2M  (log scale)")
    a.set_title("(c) final λ per run; bars = cell means", loc="left")
    a.text(0.02, 0.98, f"ratio of cell means (threshold {THR_RATIO:g}):\n"
                       f"{rp['final_lambda_ratio']:.4f} (s100-114) / {rs['final_lambda_ratio']:.4f} (s100-119)",
           transform=a.transAxes, fontsize=7.5, color=INK, va="top", bbox=BOX, zorder=5)
    handles = [Line2D([], [], color=C1, lw=1.2, label="lagu-tt seeds 100-114 (primary set)"),
               Line2D([], [], color=C2, lw=1.2, label="lagu-tt seeds 115-119 (added in the sensitivity set)"),
               Line2D([], [], color=INK, lw=2.2, label="pooled over rows after 300k / cell mean, seeds 100-114"),
               Line2D([], [], color=INK, lw=2.2, ls=(0, (1.2, 1.6)), label="same, seeds 100-119"),
               Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 3)), label="pre-registered threshold")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    yn = lambda b: "met" if b else "not met"
    fig.suptitle("DESCRIPTIVE - prereg section 9 |Q̄| feedback-loop prediction, SafetyPointGoal1-v0, 2M steps. "
                 f"Antecedent (median |qbar_avg| < {THR_Q}): {yn(rp['antecedent_met'])} for s100-114, "
                 f"{yn(rs['antecedent_met'])} for s100-119", fontsize=10, color=INK, x=0.01, ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"qbar_loop.{ext}"), dpi=200, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def main():
    lagu = {s: load(A, s) for s in SETS["sensitivity (seeds 100-119)"]}
    nogate = {s: load(B, s) for s in B_SEEDS}
    res = {name: analyse(lagu, nogate, seeds) for name, seeds in SETS.items()}
    check = crosscheck(res)
    table = tables(res, lagu, nogate)
    print(table)
    print("\ncross-check:", check)
    with open(os.path.join(OUT, "table.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(OUT, "qbar_loop.json"), "w") as f:
        json.dump({"label": "DESCRIPTIVE", "prereg": "section 9 |Q_bar| feedback loop; section 16 item 5",
                   "thresholds": {"median_abs_qbar": THR_Q, "engagement": THR_ENG, "final_lambda_ratio": THR_RATIO},
                   "window": f"{LO} < step <= {N}", "n_boot": N_BOOT, "boot_seed": 0, "results": res,
                   "per_run_final_lambda": {"lagu-tt": {s: lagu[s]["final_lambda"] for s in lagu},
                                            "lagu_nogate-tt": {s: nogate[s]["final_lambda"] for s in nogate}},
                   "crosscheck": check}, f, indent=1)
    figure(res, lagu, nogate)
    print("wrote table.md, qbar_loop.json, qbar_loop.pdf, qbar_loop.png in", OUT)


if __name__ == "__main__":
    main()
