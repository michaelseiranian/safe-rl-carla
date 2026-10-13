"""DESCRIPTIVE. Independent verification of results/paper/qbar_loop (prereg.md section 9, the
'|Q_bar| feedback loop' secondary prediction, evaluated as committed in section 16 item 5).
Nothing here is confirmatory, and nothing here changes the pre-registered definitions.

What this script does
1. Re-derives every key number from the raw CSVs in results/gate with its own code (stdlib csv and
   statistics; numpy only for the bootstrap). It does NOT import lagu.analyze or the author's make.py.
   Pre-registered definitions (prereg.md sections 5, 9, 16 item 5):
   - rows "after 300k": interval rows with 300k < step <= 2M (the row at 300k covers (250k, 300k]);
   - median |qbar_avg|: median over the pooled rows of every lagu-tt run in the seed set;
   - engagement over those rows: sum(engagement * n_risk) / sum(n_risk), pooled over seeds;
   - final lambda: the 'lambda' column in the run's last row (step 2M); ratio of cell means
     mean(lagu-tt) / mean(lagu_nogate-tt), denominator lagu_nogate-tt seeds 100-114 (the only ones run);
   - seed sets: lagu-tt 100-114 (primary) and 100-119 (sensitivity). No run is excluded.
   - 95% intervals: percentile bootstrap over runs, 10,000 resamples, numpy.random.default_rng(0)
     (lagu-tt run indices drawn first, then lagu_nogate-tt indices, independently).
2. Compares those numbers with the author's qbar_loop.json, table.md and with prereg_results.json.
3. Checks the factual claims in the author's summary and caveats.
4. Alternative definitions (DESCRIPTIVE robustness only, NOT pre-registered, never replacing the
   pre-registered result): window step >= 300k; median of per-run medians; unweighted engagement;
   lambda_avg instead of lambda at the last row. They only ask whether the antecedent's outcome
   depends on a definitional choice.
5. Re-runs a byte-identical copy of the author's make.py twice in ./rerun_author/ (outputs land
   there, never in results/paper/qbar_loop) and compares its outputs with the published ones.

Run from the repository root:  .venv/bin/python results/paper/qbar-loop-verify/make.py
Writes verify.json and table.md next to this file and prints the tables.
"""
import csv
import filecmp
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys

import numpy as np

ROOT = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(ROOT, "results", "gate")
AUTH = os.path.join(ROOT, "results", "paper", "qbar_loop")
ENV = "SafetyPointGoal1-v0"
A, B = "lagu-tt", "lagu_nogate-tt"
SEEDSETS = [("primary s100-114", list(range(100, 115))), ("sensitivity s100-119", list(range(100, 120)))]
B_SEEDS = list(range(100, 115))
N_END, LO = 2_000_000, 300_000
THR = {"median": 0.3, "engagement": 0.8, "ratio": 3.0}
NB = 10_000


# ------------------------------------------------------------------------------------ raw data
def read_run(arm, seed):
    path = os.path.join(GATE, f"{arm}_{ENV}_s{seed}.csv")
    with open(path, newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        body = [r for r in rd if r]
    ix = {k: i for i, k in enumerate(head)}

    def num(r, k):
        s = r[ix[k]].strip()
        return float(s) if s else math.nan

    rows = [{k: num(r, k) for k in ("step", "qbar_avg", "engagement", "n_risk", "n_pi", "lambda",
                                     "lambda_avg", "resumed")} for r in body]
    steps = [int(r["step"]) for r in rows]
    checks = {"n_rows": len(rows), "unique_steps": len(set(steps)) == len(steps),
              "sorted": steps == sorted(steps), "last_step": steps[-1],
              "resumed_any": any(r["resumed"] == 1 for r in rows)}
    return {"rows": rows, "checks": checks}


def window(run, lo_inclusive=False):
    return [r for r in run["rows"] if (r["step"] >= LO if lo_inclusive else r["step"] > LO) and r["step"] <= N_END]


def final_row(run):
    return max(run["rows"], key=lambda r: r["step"])


# ------------------------------------------------------------------------------- statistics
def pooled_median_abs(runs, lo_inclusive=False):
    vals = [abs(r["qbar_avg"]) for run in runs for r in window(run, lo_inclusive)]
    assert all(math.isfinite(v) for v in vals)
    return statistics.median(vals), len(vals)


def pooled_engagement(runs, weighted=True, lo_inclusive=False):
    rows = [r for run in runs for r in window(run, lo_inclusive)]
    assert all(math.isfinite(r["engagement"]) and math.isfinite(r["n_risk"]) for r in rows)
    if not weighted:
        return math.fsum(r["engagement"] for r in rows) / len(rows)
    return math.fsum(r["engagement"] * r["n_risk"] for r in rows) / math.fsum(r["n_risk"] for r in rows)


def cell_mean(runs, key="lambda"):
    v = [final_row(run)[key] for run in runs]
    assert all(math.isfinite(x) for x in v)
    return math.fsum(v) / len(v)


def bootstrap(lagu_runs, nog_runs):
    """Cluster (run) bootstrap: rows of a drawn run move together; the two arms resample independently."""
    rng = np.random.default_rng(0)
    na, nb = len(lagu_runs), len(nog_runs)
    ia = rng.integers(0, na, size=(NB, na))
    ib = rng.integers(0, nb, size=(NB, nb))
    q = [np.array([abs(r["qbar_avg"]) for r in window(run)]) for run in lagu_runs]
    en = [np.array([r["engagement"] * r["n_risk"] for r in window(run)]).sum() for run in lagu_runs]
    nr = [np.array([r["n_risk"] for r in window(run)]).sum() for run in lagu_runs]
    la = np.array([final_row(run)["lambda"] for run in lagu_runs])
    lb = np.array([final_row(run)["lambda"] for run in nog_runs])
    med = np.empty(NB)
    eng = np.empty(NB)
    for b in range(NB):
        med[b] = np.median(np.concatenate([q[i] for i in ia[b]]))
        eng[b] = sum(en[i] for i in ia[b]) / sum(nr[i] for i in ia[b])
    ratio = la[ia].mean(axis=1) / lb[ib].mean(axis=1)
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"median_ci": pct(med), "engagement_ci": pct(eng), "ratio_ci": pct(ratio),
            "ratio_draws": ratio, "ib": ib}


# ------------------------------------------------------------------------- author re-run
def sha(p):
    """Content hash; for a PDF the /CreationDate that matplotlib embeds is removed first."""
    import re
    with open(p, "rb") as f:
        b = f.read()
    if p.endswith(".pdf"):
        b = re.sub(rb"/CreationDate \(D:[^)]*\)", b"", b)
    return hashlib.sha256(b).hexdigest()[:16]


def rerun_author():
    """Run a copy of the author's make.py twice; its outputs land in ./rerun_author/<k>/."""
    out = {}
    for k in ("run1", "run2"):
        d = os.path.join(HERE, "rerun_author", k)
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(os.path.join(AUTH, "make.py"), os.path.join(d, "make.py"))
        p = subprocess.run([sys.executable, os.path.join(d, "make.py")], cwd=ROOT, capture_output=True, text=True)
        out[k] = {"returncode": p.returncode, "stderr_tail": p.stderr[-400:], "dir": d}
    same_src = filecmp.cmp(os.path.join(AUTH, "make.py"), os.path.join(HERE, "rerun_author", "run1", "make.py"),
                           shallow=False)
    files = {}
    for fn in ("qbar_loop.json", "table.md", "qbar_loop.png", "qbar_loop.pdf"):
        pub = os.path.join(AUTH, fn)
        r1 = os.path.join(HERE, "rerun_author", "run1", fn)
        r2 = os.path.join(HERE, "rerun_author", "run2", fn)
        files[fn] = {"published_vs_rerun_identical": filecmp.cmp(pub, r1, shallow=False),
                     "rerun1_vs_rerun2_identical": filecmp.cmp(r1, r2, shallow=False),
                     "sha_published_no_creationdate": sha(pub), "sha_rerun1_no_creationdate": sha(r1)}
    return {"script_copy_identical": same_src, "runs": out, "files": files}


def pdf_difference_is_metadata_only(a, b):
    """True if two PDFs differ only in /CreationDate (matplotlib embeds the time)."""
    import re
    ra, rb = open(a, "rb").read(), open(b, "rb").read()
    strip = lambda x: re.sub(rb"/CreationDate \(D:[^)]*\)", b"", x)
    return strip(ra) == strip(rb)


# ------------------------------------------------------------------------------------- main
def main():
    lagu = {s: read_run(A, s) for s in range(100, 120)}
    nog = {s: read_run(B, s) for s in B_SEEDS}
    missing_nog = [s for s in range(115, 120) if os.path.exists(os.path.join(GATE, f"{B}_{ENV}_s{s}.csv"))]

    # integrity of every input run
    integ = []
    for arm, d in ((A, lagu), (B, nog)):
        for s, run in d.items():
            c = run["checks"]
            integ.append((arm, s, c["n_rows"], c["unique_steps"], c["sorted"], c["last_step"], c["resumed_any"],
                          len(window(run))))
    assert all(x[3] and x[4] and x[5] == N_END for x in integ)

    res = {}
    for name, seeds in SEEDSETS:
        ra = [lagu[s] for s in seeds]
        rb = [nog[s] for s in B_SEEDS]
        med, nrows = pooled_median_abs(ra)
        eng = pooled_engagement(ra)
        ma, mb = cell_mean(ra), cell_mean(rb)
        bs = bootstrap(ra, rb)
        idx103 = B_SEEDS.index(103)
        no103 = ~(bs["ib"] == idx103).any(axis=1)
        r_no = bs["ratio_draws"][no103]
        res[name] = {
            "n_lagu": len(seeds), "n_nogate": len(B_SEEDS), "n_rows": nrows,
            "median_abs_qbar": med, "median_ci95": bs["median_ci"],
            "engagement": eng, "engagement_ci95": bs["engagement_ci"],
            "mean_final_lambda_lagu": ma, "mean_final_lambda_nogate": mb,
            "ratio": ma / mb, "ratio_ci95": bs["ratio_ci"],
            "antecedent_met": med < THR["median"], "engagement_gt_0.8": eng > THR["engagement"],
            "ratio_gt_3": ma / mb > THR["ratio"],
            "prediction_holds": med < THR["median"] and eng > THR["engagement"] and ma / mb > THR["ratio"],
            # claim check on the author's caveat ("about 35% of resamples leave s103 out ... ratios near 1-2.4")
            "claimcheck_frac_resamples_without_s103": float(no103.mean()),
            "claimcheck_ratio_in_those_resamples_min_p5_p50_p95_max": [float(x) for x in np.percentile(r_no, [0, 5, 50, 95, 100])],
            "claimcheck_frac_all_resamples_ratio_gt_3": float((bs["ratio_draws"] > 3).mean()),
            # alternative definitions (NOT pre-registered; robustness of the antecedent only)
            "alt_median_window_step_ge_300k": pooled_median_abs(ra, lo_inclusive=True)[0],
            "alt_median_of_per_run_medians": statistics.median(
                [statistics.median([abs(r["qbar_avg"]) for r in window(run)]) for run in ra]),
            "alt_frac_rows_abs_qbar_lt_0.3": sum(abs(r["qbar_avg"]) < 0.3 for run in ra for r in window(run)) / nrows,
            "alt_engagement_unweighted": pooled_engagement(ra, weighted=False),
            "alt_ratio_lambda_avg_last_row": cell_mean(ra, "lambda_avg") / cell_mean(rb, "lambda_avg"),
        }

    per_run = {}
    for s in range(100, 120):
        w = window(lagu[s])
        per_run[s] = {"median_abs_qbar": statistics.median([abs(r["qbar_avg"]) for r in w]),
                      "engagement": math.fsum(r["engagement"] * r["n_risk"] for r in w) / math.fsum(r["n_risk"] for r in w),
                      "final_lambda_lagu": final_row(lagu[s])["lambda"],
                      "resumed_lagu": lagu[s]["checks"]["resumed_any"],
                      "final_lambda_nogate": final_row(nog[s])["lambda"] if s in nog else None,
                      "resumed_nogate": nog[s]["checks"]["resumed_any"] if s in nog else None}

    # ---------------------------------------------------- comparison with the author's outputs
    with open(os.path.join(AUTH, "qbar_loop.json")) as f:
        aj = json.load(f)
    amap = {"primary s100-114": "primary (seeds 100-114)", "sensitivity s100-119": "sensitivity (seeds 100-119)"}
    pairs = [("median_abs_qbar", "median_abs_qbar_after_300k"), ("engagement", "engagement_after_300k"),
             ("ratio", "final_lambda_ratio"), ("mean_final_lambda_lagu", "mean_final_lambda_lagu"),
             ("mean_final_lambda_nogate", "mean_final_lambda_nogate"), ("n_rows", "n_rows"),
             ("median_ci95", "median_abs_qbar_ci95"), ("engagement_ci95", "engagement_ci95"),
             ("ratio_ci95", "final_lambda_ratio_ci95"), ("antecedent_met", "antecedent_met"),
             ("engagement_gt_0.8", "engagement_gt_0.8"), ("ratio_gt_3", "ratio_gt_3"),
             ("prediction_holds", "prediction_holds")]
    cmp_rows = []
    for name in res:
        for mine, theirs in pairs:
            x, y = res[name][mine], aj["results"][amap[name]][theirs]
            if isinstance(x, list):
                d = max(abs(u - v) / max(1.0, abs(v)) for u, v in zip(x, y))
                ok = d <= 1e-12
            elif isinstance(x, bool):
                d, ok = 0.0 if x == y else 1.0, x == y
            else:
                d = abs(x - y) / max(1.0, abs(y))
                ok = d <= 1e-12
            cmp_rows.append((name, mine, ok, d))
    per_run_json_ok = all(abs(per_run[s]["final_lambda_lagu"] - aj["per_run_final_lambda"]["lagu-tt"][str(s)]) <= 1e-15
                          for s in range(100, 120)) and \
        all(abs(per_run[s]["final_lambda_nogate"] - aj["per_run_final_lambda"]["lagu_nogate-tt"][str(s)]) <= 1e-15
            for s in B_SEEDS)
    # the author's per-run table, at its printed precision
    tab_ok = True
    with open(os.path.join(AUTH, "table.md")) as f:
        for line in f:
            p = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(p) == 7 and p[0].isdigit():
                s = int(p[0])
                tab_ok &= p[1] == f"{per_run[s]['median_abs_qbar']:.3f}" and p[2] == f"{per_run[s]['engagement']:.3f}" \
                    and p[3] == f"{per_run[s]['final_lambda_lagu']:.4g}" \
                    and p[4] == ("yes" if per_run[s]["resumed_lagu"] else "no")
                if s in nog:
                    tab_ok &= p[5] == f"{per_run[s]['final_lambda_nogate']:.4g}" and \
                        p[6] == ("yes" if per_run[s]["resumed_nogate"] else "no")
    with open(os.path.join(ROOT, "prereg_results.json")) as f:
        pr = json.load(f)["secondary"]["qbar_loop"]
    sens = res["sensitivity s100-119"]
    prereg_ok = all(abs(a - b) <= 1e-12 * max(1, abs(b)) for a, b in (
        (sens["median_abs_qbar"], pr["median_abs_qbar_after_300k"]), (sens["engagement"], pr["engagement_after_300k"]),
        (sens["ratio"], pr["final_lambda_ratio"]))) and pr["triggered"] == sens["antecedent_met"] and \
        pr["holds"] == sens["prediction_holds"]

    # ---------------------------------------------------- claim checks from the author's summary
    meds = {s: per_run[s]["median_abs_qbar"] for s in range(100, 120)}
    engs = {s: per_run[s]["engagement"] for s in range(100, 120)}
    nog_other = [per_run[s]["final_lambda_nogate"] for s in B_SEEDS if s != 103]
    claims = [
        ("no run's median |qbar_avg| < 0.3; lowest s105 = 0.363", min(meds.values()) >= 0.3 and
         min(meds, key=meds.get) == 105 and round(meds[105], 3) == 0.363),
        ("highest per-run median s116 = 13.7", max(meds, key=meds.get) == 116 and round(meds[116], 1) == 13.7),
        ("per-run engagement (s100-119) ranges 0.691-0.994", round(min(engs.values()), 3) == 0.691 and
         round(max(engs.values()), 3) == 0.994),
        ("lagu_nogate-tt s103 final lambda 8987", round(per_run[103]["final_lambda_nogate"]) == 8987),
        ("other 14 lagu_nogate-tt final lambda within 0.0036-0.886", round(min(nog_other), 4) == 0.0036 and
         round(max(nog_other), 3) == 0.886),
        ("lagu-tt cell means 0.4009 / 0.4282", round(res["primary s100-114"]["mean_final_lambda_lagu"], 4) == 0.4009 and
         round(sens["mean_final_lambda_lagu"], 4) == 0.4282),
        ("lagu_nogate-tt cell mean 599.40", round(res["primary s100-114"]["mean_final_lambda_nogate"], 2) == 599.40),
        ("resumed: lagu-tt s109-114, lagu_nogate-tt s100-105 (and no others)",
         [s for s in range(100, 120) if per_run[s]["resumed_lagu"]] == list(range(109, 115)) and
         [s for s in B_SEEDS if per_run[s]["resumed_nogate"]] == list(range(100, 106))),
        ("every CSV has 40 rows, unique sorted steps, last step 2M, 34 window rows",
         all(x[2] == 40 and x[7] == 34 for x in integ)),
        ("lagu_nogate-tt seeds 115-119 were never run", missing_nog == []),
        ("about 35% of ratio resamples exclude s103", all(abs(r["claimcheck_frac_resamples_without_s103"] - 0.35) < 0.02
                                                         for r in res.values())),
        ("both ratio upper bounds below 3", all(r["ratio_ci95"][1] < 3 for r in res.values())),
        ("sensitivity row equals prereg_results.json secondary.qbar_loop", prereg_ok),
    ]

    rer = rerun_author()
    pdf_meta_only = pdf_difference_is_metadata_only(os.path.join(AUTH, "qbar_loop.pdf"),
                                                     os.path.join(HERE, "rerun_author", "run1", "qbar_loop.pdf"))

    # ------------------------------------------------------------------------------- print
    L = []
    L += ["### DESCRIPTIVE - independent re-derivation, prereg section 9 |Q̄| feedback loop (SafetyPointGoal1-v0, 2M)",
          "", "Pre-registered definitions; own code (csv + statistics; numpy bootstrap, 10,000 resamples, "
              "default_rng(0), percentile 95%).", "",
          "| seed set | runs (lagu / nogate) | rows | median \\|qbar_avg\\| (< 0.3?) | engagement (> 0.8?) "
          "| final-λ ratio (> 3?) | antecedent | prediction holds |", "|---|---|---|---|---|---|---|---|"]
    yn = lambda b: "yes" if b else "no"
    for name, r in res.items():
        L.append(f"| {name} | {r['n_lagu']} / {r['n_nogate']} | {r['n_rows']} | {r['median_abs_qbar']:.4f} "
                 f"[{r['median_ci95'][0]:.3f}, {r['median_ci95'][1]:.3f}] ({yn(r['antecedent_met'])}) | "
                 f"{r['engagement']:.4f} [{r['engagement_ci95'][0]:.3f}, {r['engagement_ci95'][1]:.3f}] "
                 f"({yn(r['engagement_gt_0.8'])}) | {r['ratio']:.6f} [{r['ratio_ci95'][0]:.5f}, {r['ratio_ci95'][1]:.4f}] "
                 f"({yn(r['ratio_gt_3'])}) | {yn(r['antecedent_met'])} | {yn(r['prediction_holds'])} |")
    L += ["", "Cell means of final λ: " + "; ".join(
        f"{n}: lagu-tt {r['mean_final_lambda_lagu']:.6f}, lagu_nogate-tt {r['mean_final_lambda_nogate']:.4f}"
        for n, r in res.items()), ""]
    L += ["### DESCRIPTIVE - agreement with the author's qbar_loop.json (relative tolerance 1e-12)", "",
          "| seed set | quantity | agrees | max rel. diff |", "|---|---|---|---|"]
    L += [f"| {n} | {q} | {yn(ok)} | {d:.1e} |" for n, q, ok, d in cmp_rows]
    L += ["", f"Per-run final λ in author JSON agrees: {yn(per_run_json_ok)}. "
              f"Author per-run table agrees at printed precision: {yn(tab_ok)}. "
              f"prereg_results.json secondary.qbar_loop agrees with the sensitivity row: {yn(prereg_ok)}.", ""]
    L += ["### DESCRIPTIVE - claims in the author's summary/caveats", "", "| claim | holds |", "|---|---|"]
    L += [f"| {c} | {yn(ok)} |" for c, ok in claims]
    L += ["", "Bootstrap resamples whose lagu_nogate-tt draw excludes s103 (checks the caveat 'ratios near 1-2.4'):", ""]
    for n, r in res.items():
        q = r["claimcheck_ratio_in_those_resamples_min_p5_p50_p95_max"]
        L.append(f"- {n}: fraction {r['claimcheck_frac_resamples_without_s103']:.4f}; ratio min {q[0]:.3f}, "
                 f"p5 {q[1]:.3f}, median {q[2]:.3f}, p95 {q[3]:.3f}, max {q[4]:.3f}; "
                 f"fraction of ALL resamples with ratio > 3: {r['claimcheck_frac_all_resamples_ratio_gt_3']:.4f}")
    # context for the author's sentence "the gate fires on >90% ... even though |Q_bar| is well above 0.3":
    # across runs, does a lower |qbar_avg| go with higher engagement (the direction the loop assumes)?
    def spearman(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    L += ["", "Across runs (DESCRIPTIVE context, not a test): Spearman rank correlation between a run's median "
              "|qbar_avg| and its engagement after 300k:"]
    for name, seeds in SEEDSETS:
        rho = spearman([meds[s] for s in seeds], [engs[s] for s in seeds])
        low = sorted(seeds, key=lambda s: meds[s])[:5]
        L.append(f"- {name}: rho = {rho:+.3f}; the 5 runs with the lowest median |qbar_avg| "
                 + ", ".join(f"s{s} ({meds[s]:.3f} -> eng {engs[s]:.3f})" for s in low))
    L += ["", "### DESCRIPTIVE - alternative definitions (NOT pre-registered; robustness of the antecedent only)", "",
          "| seed set | median, window step >= 300k | median of per-run medians | fraction of rows with \\|qbar_avg\\| < 0.3 "
          "| engagement, unweighted | ratio with lambda_avg at 2M |", "|---|---|---|---|---|---|"]
    for n, r in res.items():
        L.append(f"| {n} | {r['alt_median_window_step_ge_300k']:.3f} | {r['alt_median_of_per_run_medians']:.3f} | "
                 f"{r['alt_frac_rows_abs_qbar_lt_0.3']:.3f} | {r['alt_engagement_unweighted']:.3f} | "
                 f"{r['alt_ratio_lambda_avg_last_row']:.6f} |")
    L += ["", "### DESCRIPTIVE - re-run of the author's make.py (copy, outputs in ./rerun_author/)", "",
          f"Copy identical to published script: {yn(rer['script_copy_identical'])}. Return codes: "
          f"{rer['runs']['run1']['returncode']}, {rer['runs']['run2']['returncode']}.", "",
          "| file | published == re-run | re-run 1 == re-run 2 |", "|---|---|---|"]
    L += [f"| {fn} | {yn(v['published_vs_rerun_identical'])} | {yn(v['rerun1_vs_rerun2_identical'])} |"
          for fn, v in rer["files"].items()]
    L += ["", f"PDF differs from the published one only in /CreationDate: {yn(pdf_meta_only)}."]
    text = "\n".join(L)
    print(text)
    with open(os.path.join(HERE, "table.md"), "w") as f:
        f.write(text + "\n")
    clean = {n: {k: v for k, v in r.items()} for n, r in res.items()}
    with open(os.path.join(HERE, "verify.json"), "w") as f:
        json.dump({"label": "DESCRIPTIVE", "results": clean, "per_run": {str(k): v for k, v in per_run.items()},
                   "claims": [{"claim": c, "holds": bool(ok)} for c, ok in claims],
                   "agreement": [{"set": n, "quantity": q, "agrees": bool(ok), "max_rel_diff": d}
                                 for n, q, ok, d in cmp_rows],
                   "per_run_json_ok": per_run_json_ok, "author_table_ok": bool(tab_ok), "prereg_results_ok": prereg_ok,
                   "rerun": rer, "pdf_differs_only_in_creation_date": pdf_meta_only}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
