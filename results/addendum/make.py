"""EXPLORATORY analyses of the 2026-11-10 addendum (prereg.md section 16, item 15).

Nothing here is confirmatory and nothing enters the Holm family. The paired contrasts A1, A2 and B1
and the engagement test A3 come from the `lagu.analyze` commands listed in the addendum (a1.json,
a2.json, a3.json, b1.json). This script adds what the addendum declares without a command and
tabulates the runs one by one:

  A    held-out endpoint of every run of lagu-tt-signed, lagu-tt (the |Q_bar| gate) and
       lagu_nogate-tt on seeds 100-114, and the dose of both gates with the definitions of
       results/paper/gate_dose/make.py part 1: engagement = sum(engagement*n_risk)/sum(n_risk),
       budget(all) = sum(delta_avg*n_pi)/sum(n_pi)/delta0, budget(risk) =
       sum(delta_risk_avg*n_risk)/sum(n_risk)/delta0, pooled over seeds, rows with lo < step <= hi,
       95% percentile bootstrap over seeds (10,000 resamples, numpy.random.default_rng(0)).
  B1   collapsed runs (held-out return < 5 or diverged, prereg.md 11) of lagu_nogate-tt on seeds
       115-119, with the two-sided Fisher exact test against its count on seeds 100-114.
  C1   td3lag-omni at 3M steps: the held-out endpoint and the in-training window (rows with
       step >= 0.8N), mean +- sd (ddof 1) over seeds 0-2, beside OmniSafe's published 25.27 +- 2.74
       return and 28.00 +- 15.75 cost (another protocol; prereg.md 14).
  C2   the same runs' rows up to 1M steps against the E9.6 runs of results/pareto (same seeds).
  P1-P4  the addendum's four predictions, read off the numbers above.

Rows marked [extra] were not declared in the addendum; they are descriptive.

The script refuses to run unless every run of the addendum is complete and has a held-out file at
its own end step (prereg.md section 16, item 5).

    .venv/bin/python results/addendum/make.py     # writes extra.txt and extra.json in this directory
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import lagu.analyze as A  # noqa: E402

PG = "SafetyPointGoal1-v0"
N_BOOT = 10_000
WINDOWS = [("25k-300k", 25_000, 300_000), ("300k-1M", 300_000, 1_000_000),
           ("1M-2M", 1_000_000, 2_000_000), ("whole run", 0, 10**12)]
OMNISAFE = {"return": (25.27, 2.74), "cost": (28.00, 15.75)}      # prereg.md 14, TD3Lag, 3M steps
E1_SEEDS, EXT_SEEDS, OMNI_SEEDS = list(range(100, 115)), list(range(115, 120)), [0, 1, 2]
SKIP = ("sps", "resumed")          # columns that may differ between two identical computations


def get(runs, label, seeds):
    d = {r["seed"]: r for r in runs if r["arm"] == label and r["env"] == PG}
    missing = [s for s in seeds if s not in d]
    if missing:
        raise SystemExit(f"{label}: no run for seeds {missing}")
    return [d[s] for s in seeds]


def held(run):
    return A.endpoints(run, None, True)


def msd(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def boot_ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(0).integers(0, len(num), (N_BOOT, len(num)))
    stat = num[idx].sum(-1) / den[idx].sum(-1)
    return [float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))]


def wsums(x, w, mask):
    ok = mask & np.isfinite(x) & (w > 0)
    return float((x[ok] * w[ok]).sum()), float(w[ok].sum())


def dose(rs):
    out = {}
    for name, lo, hi in WINDOWS:
        sums = {"engagement": [], "budget_all": [], "budget_risk": []}
        for r in rs:
            d0, step = float(r["cfg"]["delta0"]), A.col(r, "step")
            m = (step > lo) & (step <= hi)
            sums["engagement"].append(wsums(A.col(r, "engagement"), A.col(r, "n_risk"), m))
            sums["budget_all"].append(wsums(A.col(r, "delta_avg") / d0, A.col(r, "n_pi"), m))
            sums["budget_risk"].append(wsums(A.col(r, "delta_risk_avg") / d0, A.col(r, "n_risk"), m))
        row = {}
        for k, v in sums.items():
            v = np.array(v)
            row[k] = {"value": float(v[:, 0].sum() / v[:, 1].sum()), "ci95": boot_ratio(v[:, 0], v[:, 1])}
        out[name] = row
    return out


def qbar_sign(rs, lo=300_000):
    """[extra] Share of the 50k-step logging intervals after `lo` whose mean Q_bar is <= 0, and the
    median of that mean, pooled over the rows of the runs."""
    q = np.concatenate([A.col(r, "qbar_avg")[A.col(r, "step") > lo] for r in rs])
    q = q[np.isfinite(q)]
    return {"rows": int(len(q)), "share_nonpositive": float((q <= 0).mean()), "median": float(np.median(q))}


def fmt(c, p=3):
    return f"{c['value']:.{p}f} [{c['ci95'][0]:.{p}f}, {c['ci95'][1]:.{p}f}]"


def main():
    ADD, GATE, PAR = (A.load(f"results/{d}") for d in ("addendum", "gate", "pareto"))
    bad = [r["stem"] for r in ADD if not A.heldout_ok(r) or r["rows"][-1]["step"] < r["steps"]]
    if bad or len(ADD) != 23:
        raise SystemExit(f"addendum runs incomplete or without a held-out file ({len(ADD)} loaded): {bad}")
    res: dict = {"n_runs": len(ADD), "resumed_runs": [r["stem"] for r in ADD if np.any(A.col(r, "resumed") == 1)]}
    print("EXPLORATORY: addendum of 2026-11-10 (prereg.md section 16, item 15). Outside the Holm family.")
    print(f"All {len(ADD)} addendum runs are complete and have a held-out file at their own end step; "
          f"resumed runs: {res['resumed_runs'] or 'none'}.")

    # ------------------------------------------------------------------ A
    arms = {"lagu-tt-signed": get(ADD, "lagu-tt-signed", E1_SEEDS), "lagu-tt": get(GATE, "lagu-tt", E1_SEEDS),
            "lagu_nogate-tt": get(GATE, "lagu_nogate-tt", E1_SEEDS)}
    if not all(A.heldout_ok(r) for rs in arms.values() for r in rs):
        raise SystemExit("a run of the A contrasts has no held-out file")
    E = {k: [held(r) for r in rs] for k, rs in arms.items()}
    print("\nA. Held-out endpoint per seed (return / cost; * = collapsed: return < 5 or diverged), seeds 100-114")
    print(f"  {'seed':>4s}  {'signed gate':>16s}  {'|Q_bar| gate':>16s}  {'no gate':>16s}")
    for i, s in enumerate(E1_SEEDS):
        cell = lambda e: f"{e['return']:6.2f} /{e['cost']:6.2f}{'*' if e['collapsed'] else ' '}"
        print(f"  {s:4d}  " + "  ".join(f"{cell(E[k][i]):>16s}" for k in arms))
    res["A_arms"] = {}
    for k in arms:
        ret, cost = [e["return"] for e in E[k]], [e["cost"] for e in E[k]]
        lam = [A.lambda_time(r) for r in arms[k]]
        res["A_arms"][k] = {"return_mean": float(np.mean(ret)), "cost_mean": float(np.mean(cost)),
                            "collapsed": int(sum(e["collapsed"] for e in E[k])),
                            "cost_le_25": int(sum(c <= A.COST_LIMIT for c in cost)),
                            "cost_le_25_not_collapsed": int(sum(c <= A.COST_LIMIT and not e["collapsed"]
                                                                for c, e in zip(cost, E[k]))),
                            "lambda_time_mean": float(np.mean(lam)), "lambda_time_median": float(np.median(lam)),
                            "per_seed": {s: {"return": e["return"], "cost": e["cost"], "collapsed": e["collapsed"]}
                                         for s, e in zip(E1_SEEDS, E[k])}}
        a = res["A_arms"][k]
        print(f"  {k:16s} mean return {a['return_mean']:6.2f}  mean cost {a['cost_mean']:6.2f}  collapsed "
              f"{a['collapsed']:2d}/15  cost <= 25: {a['cost_le_25']}/15 ({a['cost_le_25_not_collapsed']} not collapsed)"
              f"  time-avg lambda mean {a['lambda_time_mean']:.4f} median {a['lambda_time_median']:.4f}")

    print("\nA3. Dose of the two gates, seeds 100-114 (pooled over seeds; [95% bootstrap over seeds])")
    print(f"  {'arm':16s} {'window':10s} {'engagement':>22s} {'budget(all)/d0':>22s} {'budget(risk)/d0':>22s}")
    res["dose"] = {}
    for k in ("lagu-tt-signed", "lagu-tt"):
        res["dose"][k] = dose(arms[k])
        for w, row in res["dose"][k].items():
            print(f"  {k:16s} {w:10s} {fmt(row['engagement']):>22s} {fmt(row['budget_all']):>22s} "
                  f"{fmt(row['budget_risk']):>22s}")
    res["qbar_after_300k"] = {k: qbar_sign(rs) for k, rs in arms.items()}
    print("  [extra] 50k-step logging intervals after 300k with mean Q_bar <= 0, and the median of that mean:")
    for k, q in res["qbar_after_300k"].items():
        print(f"    {k:16s} {100 * q['share_nonpositive']:5.1f}% of {q['rows']} rows; median {q['median']:+.3f}")

    # ------------------------------------------------------------------ B1
    new, old = get(ADD, "lagu_nogate-tt", EXT_SEEDS), arms["lagu_nogate-tt"]
    ext = {"lagu-tt": get(GATE, "lagu-tt", EXT_SEEDS), "lagu_nogate-tt-match": get(GATE, "lagu_nogate-tt-match", EXT_SEEDS),
           "lagu_nogate-tt": new}
    if not all(A.heldout_ok(r) for rs in ext.values() for r in rs):
        raise SystemExit("a run on seeds 115-119 has no held-out file")
    X = {k: [held(r) for r in rs] for k, rs in ext.items()}
    print("\nB1. Seeds 115-119, held-out endpoint (return / cost; * = collapsed)")
    print(f"  {'seed':>4s}  {'|Q_bar| gate':>16s}  {'constant budget':>16s}  {'no gate (new)':>16s}")
    for i, s in enumerate(EXT_SEEDS):
        cell = lambda e: f"{e['return']:6.2f} /{e['cost']:6.2f}{'*' if e['collapsed'] else ' '}"
        print(f"  {s:4d}  " + "  ".join(f"{cell(X[k][i]):>16s}" for k in ext))
    k_new, k_old = int(sum(e["collapsed"] for e in X["lagu_nogate-tt"])), res["A_arms"]["lagu_nogate-tt"]["collapsed"]
    p = A.fisher_exact(k_new, len(new) - k_new, k_old, len(old) - k_old)
    res["B1"] = {"collapsed_115_119": k_new, "n_115_119": len(new), "collapsed_100_114": k_old, "n_100_114": len(old),
                 "fisher_p": p, "collapsed_by_arm_115_119": {k: int(sum(e["collapsed"] for e in X[k])) for k in ext},
                 "per_seed": {k: {s: {"return": e["return"], "cost": e["cost"], "collapsed": e["collapsed"]}
                                  for s, e in zip(EXT_SEEDS, X[k])} for k in ext},
                 "nogate_return_mean_115_119": float(np.mean([e["return"] for e in X["lagu_nogate-tt"]])),
                 "nogate_cost_mean_115_119": float(np.mean([e["cost"] for e in X["lagu_nogate-tt"]]))}
    print(f"  collapsed no-gate runs: {k_new} of {len(new)} on seeds 115-119 against {k_old} of {len(old)} on seeds "
          f"100-114; two-sided Fisher exact p = {p:.4f}")
    print(f"  collapsed on seeds 115-119 by arm: {res['B1']['collapsed_by_arm_115_119']}")

    # ------------------------------------------------------------------ C1
    omni = get(ADD, "td3lag-omni", OMNI_SEEDS)
    H, W = [held(r) for r in omni], [A.endpoints(r, None, False) for r in omni]
    lam = [A.at(r, "lambda", r["steps"]) for r in omni]
    res["C1"] = {"N": [int(e["N"]) for e in H], "final_lambda": lam}
    print("\nC1. td3lag-omni (OmniSafe's configuration) at 3M steps, seeds 0-2")
    print(f"  {'seed':>4s}  {'held-out ret / cost':>22s}  {'in-training ret / cost':>24s}  {'final lambda':>12s}")
    for s, h, w, l in zip(OMNI_SEEDS, H, W, lam):
        print(f"  {s:4d}  {h['return']:9.2f} /{h['cost']:7.2f}{'*' if h['collapsed'] else ' '}   "
              f"{w['return']:10.2f} /{w['cost']:7.2f}{'*' if w['collapsed'] else ' '}   {l:12.4f}")
    for name, es in (("heldout", H), ("in_training", W)):
        r, c = msd([e["return"] for e in es]), msd([e["cost"] for e in es])
        res["C1"][name] = {"return_mean": r[0], "return_sd": r[1], "cost_mean": c[0], "cost_sd": c[1],
                           "per_seed_return": [e["return"] for e in es], "per_seed_cost": [e["cost"] for e in es],
                           "collapsed": int(sum(e["collapsed"] for e in es)),
                           "cost_le_25": int(sum(e["cost"] <= A.COST_LIMIT for e in es))}
        print(f"  {name:12s} return {r[0]:.2f} +- {r[1]:.2f}   cost {c[0]:.2f} +- {c[1]:.2f}   (mean +- sd, ddof 1; "
              f"collapsed {res['C1'][name]['collapsed']}/3; cost <= 25: {res['C1'][name]['cost_le_25']}/3)")
    print(f"  OmniSafe      return {OMNISAFE['return'][0]:.2f} +- {OMNISAFE['return'][1]:.2f}   cost "
          f"{OMNISAFE['cost'][0]:.2f} +- {OMNISAFE['cost'][1]:.2f}   (as printed in its table; another protocol)")
    print("  [extra] in-training return / cost / lambda by 500k block (mean of the rows in each block):")
    res["C1"]["blocks"] = {}
    for s, r in zip(OMNI_SEEDS, omni):
        blocks = []
        for lo in range(0, 3_000_000, 500_000):
            m = lambda k: float(np.nanmean(A.col(r, k, lo + 1, lo + 500_000)))
            blocks.append({"to": lo + 500_000, "return": m("eval_return"), "cost": m("eval_cost"), "lambda": m("lambda")})
        res["C1"]["blocks"][s] = blocks
        print(f"    seed {s}: " + "  ".join(f"{b['to'] / 1e6:.1f}M {b['return']:5.1f}/{b['cost']:5.1f}/{b['lambda']:.2f}" for b in blocks))

    # ------------------------------------------------------------------ C2
    old_omni = get(PAR, "td3lag-omni", OMNI_SEEDS)
    res["C2"] = {"per_seed": {}}
    print("\nC2. The same runs up to 1M steps against the E9.6 runs (results/pareto, same seeds)")
    for s, n, o in zip(OMNI_SEEDS, omni, old_omni):
        rn = {int(r["step"]): r for r in n["rows"] if r["step"] <= 1_000_000}
        ro = {int(r["step"]): r for r in o["rows"]}
        steps = sorted(set(rn) & set(ro))
        keys = [k for k in ro[steps[0]] if k in rn[steps[0]] and k not in SKIP]
        same = lambda a, b: a == b or (not np.isfinite(a) and not np.isfinite(b))
        ident = [t for t in steps if all(same(rn[t][k], ro[t][k]) for k in keys)]
        first_diff = next((t for t in steps if t not in ident), None)
        maxd = {k: float(np.nanmax(np.abs([rn[t][k] - ro[t][k] for t in steps]))) for k in ("eval_return", "eval_cost", "lambda")}
        wn, wo = A.endpoints(n, 1_000_000, False), A.endpoints(o, 1_000_000, False)
        res["C2"]["per_seed"][s] = {"rows_compared": len(steps), "columns_compared": len(keys), "identical_rows": len(ident),
                                    "first_differing_step": first_diff, "max_abs_diff": maxd,
                                    "window_new": {"return": wn["return"], "cost": wn["cost"]},
                                    "window_e96": {"return": wo["return"], "cost": wo["cost"]}}
        print(f"  seed {s}: {len(ident)} of {len(steps)} rows identical in {len(keys)} columns (first differing step "
              f"{first_diff}); max |diff| return {maxd['eval_return']:.3g}, cost {maxd['eval_cost']:.3g}, lambda "
              f"{maxd['lambda']:.3g}; 0.8M-1M window: new {wn['return']:.2f} / {wn['cost']:.2f}, E9.6 {wo['return']:.2f} / {wo['cost']:.2f}")
    for name, key in (("new", "window_new"), ("e96", "window_e96")):
        r = msd([res["C2"]["per_seed"][s][key]["return"] for s in OMNI_SEEDS])
        c = msd([res["C2"]["per_seed"][s][key]["cost"] for s in OMNI_SEEDS])
        res["C2"][name] = {"return_mean": r[0], "return_sd": r[1], "cost_mean": c[0], "cost_sd": c[1]}
        print(f"  0.8M-1M in-training window, {name:4s}: return {r[0]:.2f} +- {r[1]:.2f}   cost {c[0]:.2f} +- {c[1]:.2f}")
    res["C2"]["settings_that_differ"] = sorted(
        k for k in set(omni[0]["cfg"]) | set(old_omni[0]["cfg"]) if omni[0]["cfg"].get(k) != old_omni[0]["cfg"].get(k))
    print(f"  cfg keys that differ between the two sets of runs: {res['C2']['settings_that_differ'] or 'none'}")

    # ------------------------------------------------------------------ predictions
    a1 = A.pair(ADD + GATE, "lagu-tt-signed", "lagu_nogate-tt", seeds=set(E1_SEEDS))[PG]
    if a1["source"] != "heldout":
        raise SystemExit("A1 did not use the held-out endpoint")
    D_signed, D_abs = (res["dose"][k]["whole run"]["budget_all"]["value"] for k in ("lagu-tt-signed", "lagu-tt"))
    win = res["C1"]["in_training"]["return_mean"]
    res["predictions"] = {
        "P1": {"signed_budget_all": D_signed, "abs_budget_all": D_abs, "holds": bool(D_signed > D_abs)},
        "P2": {"a1_return_diff": a1["return"]["diff"], "a1_return_ci95": a1["return"]["ci95"], "e1_return_diff": -10.17,
               "holds": bool(abs(a1["return"]["diff"]) < 10.17)},
        "P3": {"collapsed_nogate_115_119": k_new, "at_least_3_of_5": bool(k_new >= 3)},
        "P4": {"in_training_return": win, "omnisafe": OMNISAFE["return"][0], "tolerance": OMNISAFE["return"][1],
               "holds": bool(abs(win - OMNISAFE["return"][0]) <= OMNISAFE["return"][1])}}
    P = res["predictions"]
    print("\nPredictions of the addendum (stated before the runs; not tests)")
    print(f"  P1 signed whole-run budget(all) {D_signed:.3f} d0 above the |Q_bar| gate's {D_abs:.3f} d0: "
          f"{'holds' if P['P1']['holds'] else 'fails'}")
    print(f"  P2 A1 return difference {a1['return']['diff']:+.2f} (95% [{a1['return']['ci95'][0]:.2f}, "
          f"{a1['return']['ci95'][1]:.2f}]) closer to 0 than -10.17: {'holds' if P['P2']['holds'] else 'fails'}")
    print(f"  P3 at least 3 of 5 no-gate runs on seeds 115-119 collapse (the reading that the collapses belong to "
          f"the seeds): {k_new} of 5 -> {'supported' if P['P3']['at_least_3_of_5'] else 'not supported'}")
    print(f"  P4 in-training return at 3M {win:.2f} within {OMNISAFE['return'][1]} of {OMNISAFE['return'][0]}: "
          f"{'holds' if P['P4']['holds'] else 'fails'}")
    return res


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = main()
    text = buf.getvalue()
    sys.stdout.write(text)
    with open(os.path.join(HERE, "extra.txt"), "w") as f:
        f.write(text)
    A.write_json(out, os.path.join(HERE, "extra.json"))
