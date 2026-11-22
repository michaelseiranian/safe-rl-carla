"""EXPLORATORY analyses of the 2026-11-16 addendum (prereg.md section 16, item 17): Lag-U in the
configuration closest to the description of Zhang et al. (2024), with their learning rates.

Nothing here is confirmatory and nothing enters the Holm family. This file was written and committed
before any run of the addendum started. Labels:

  DESC_NG   lagu_nogate-lit-desc   no gate; cost target of Eq. 14 (no smoothing noise); Glorot init
  DESC_G    lagu-lit-desc          the same with the gate of Eq. 16 as printed (signed Q_bar), detached
  OCT_NG    lagu_nogate-lit        the October no-gate runs (results/gate_literal): the port's
                                   smoothed cost target and PyTorch's default initialisation

All on SafetyPointGoal1-v0, seeds 100-109, 1M steps, held-out endpoint (lagu.evaluate).

  K1  Cost critic, DESC_NG against OCT_NG: pooled prediction ratio sum(qc_traj) / sum(mc_cost_traj)
      on held-out episodes with its 95% bootstrap interval over runs (lagu.analyze.pooled_calib);
      the per-run ratios; the mean paired difference of the per-run ratios with a 95% percentile
      bootstrap interval over seed pairs (lagu.analyze.paired_bootstrap, 10,000 resamples, seed 0);
      the mean gap E[Q_c - delta] over the last fifth of training (gap_avg); the multiplier: its
      largest median over runs at a logged row, its largest value in any run, and the mean of its
      time average (lagu.analyze.lambda_time).
  K2  DESC_NG against OCT_NG: paired difference in held-out return and cost (lagu.analyze.pair).
  K3  Gate ablation in this configuration, DESC_G against DESC_NG: lagu.analyze.pair and
      lagu.analyze.h2; the gate's firing share and budget(all) on the windows 25k-300k, 300k-1M and
      the whole run (definitions of results/addendum/make.py: engagement =
      sum(engagement*n_risk)/sum(n_risk), budget(all) = sum(delta_avg*n_pi)/sum(n_pi)/delta0, pooled
      over seeds, 95% percentile bootstrap over seeds, 10,000 resamples, numpy default_rng(0)).
  K4  Runs with held-out cost <= 25, per arm.
  Q1-Q4  The addendum's predictions, read off the numbers above.
  TRIGGER  Whether the two single-change arms of the contingent step are to be run: yes if the 95%
      interval of the paired difference in the per-run prediction ratio (K1) or in held-out cost
      (K2) excludes zero.

The script refuses to run unless every run is complete and has a held-out file at its own end step
(prereg.md section 16, item 5).

    .venv/bin/python results/addendum2/make.py              # writes k.txt and k.json in this directory
    .venv/bin/python results/addendum2/make.py --selftest   # code path only, on stand-in October runs
"""
from __future__ import annotations

import io
import json
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
SEEDS = list(range(100, 110))
N_BOOT = 10_000
DESC_NG, DESC_G, OCT_NG = "lagu_nogate-lit-desc", "lagu-lit-desc", "lagu_nogate-lit"
WINDOWS = [("25k-300k", 25_000, 300_000), ("300k-1M", 300_000, 1_000_000), ("whole run", 0, 10**12)]


def get(runs, label):
    d = {r["seed"]: r for r in runs if r["arm"] == label and r["env"] == PG}
    missing = [s for s in SEEDS if s not in d]
    if missing:
        raise SystemExit(f"{label}: no run for seeds {missing}")
    return [d[s] for s in SEEDS]


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
        sums = {"engagement": [], "budget_all": []}
        for r in rs:
            d0, step = float(r["cfg"]["delta0"]), A.col(r, "step")
            m = (step > lo) & (step <= hi)
            sums["engagement"].append(wsums(A.col(r, "engagement"), A.col(r, "n_risk"), m))
            sums["budget_all"].append(wsums(A.col(r, "delta_avg") / d0, A.col(r, "n_pi"), m))
        out[name] = {}
        for k, v in sums.items():
            v = np.array(v)
            out[name][k] = {"value": float(v[:, 0].sum() / v[:, 1].sum()), "ci95": boot_ratio(v[:, 0], v[:, 1])}
    return out


def critic(rs):
    """K1 quantities of one arm."""
    E = [A.endpoints(r, None, True) for r in rs]
    cal, lo, hi = A.pooled_calib(E, N_BOOT, 0)
    by_step: dict = {}
    largest = 0.0
    for r in rs:
        for step, lam in zip(A.col(r, "step"), A.col(r, "lambda")):
            if step > 0 and np.isfinite(lam):
                by_step.setdefault(int(step), []).append(float(lam))
                largest = max(largest, float(lam))
    med = {s: float(np.median(v)) for s, v in by_step.items() if len(v) == len(rs)}
    top = max(med, key=med.get)
    return {"calib": cal, "calib_ci95": [lo, hi], "calib_per_run": [e["calib"] for e in E],
            "gap_mean": float(np.mean([e["gap_avg"] for e in E])),
            "lambda_largest_median": med[top], "lambda_largest_median_step": top, "lambda_largest": largest,
            "lambda_time_mean": float(np.mean([A.lambda_time(r) for r in rs])),
            "return_mean": float(np.mean([e["return"] for e in E])), "cost_mean": float(np.mean([e["cost"] for e in E])),
            "cost_per_run": [e["cost"] for e in E], "return_per_run": [e["return"] for e in E],
            "cost_le_25": int(sum(e["cost"] <= A.COST_LIMIT for e in E)),
            "collapsed": int(sum(e["collapsed"] for e in E))}


def ci(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


def main(selftest: bool = False):
    global DESC_NG, DESC_G
    if selftest:
        new = A.load("results/gate_literal_x")
        DESC_NG, DESC_G = "lagu_nogate-lit-match", "lagu-lit-T03"       # stand-ins: code path only
        print("SELF-TEST on stand-in October runs (results/gate_literal_x). NOT a result of the addendum.")
    else:
        new = A.load("results/addendum2")
    old = A.load("results/gate_literal")
    labels = sorted({r["arm"] for r in new if r["env"] == PG})
    has_gate = DESC_G in labels
    want = [DESC_NG] + ([DESC_G] if has_gate else [])
    arms = {k: get(new, k) for k in want}
    arms[OCT_NG] = get(old, OCT_NG)
    bad = [r["stem"] for rs in arms.values() for r in rs
           if not A.heldout_ok(r) or r["rows"][-1]["step"] < r["steps"]]
    if bad:
        raise SystemExit(f"runs incomplete or without a held-out file: {bad}")
    res: dict = {"selftest": selftest, "arms": want + [OCT_NG], "seeds": SEEDS,
                 "resumed_runs": [r["stem"] for k in want for r in arms[k] if np.any(A.col(r, "resumed") == 1)]}
    print("EXPLORATORY: addendum of 2026-11-16 (prereg.md section 16, item 17). Outside the Holm family.")
    print(f"Runs: {', '.join(f'{k} ({len(v)})' for k, v in arms.items())}; all complete with a held-out file; "
          f"resumed runs: {res['resumed_runs'] or 'none'}.")

    # ------------------------------------------------------------------ K1
    C = {k: critic(rs) for k, rs in arms.items()}
    res["K1"] = C
    print("\nK1. Cost critic on held-out episodes (prediction ratio = sum of predictions / sum of discounted cost incurred)")
    for k in arms:
        c = C[k]
        print(f"  {k:22s} ratio {c['calib']:.3f} [{c['calib_ci95'][0]:.3f}, {c['calib_ci95'][1]:.3f}]  per run "
              f"{min(c['calib_per_run']):.3f} to {max(c['calib_per_run']):.3f}  gap {c['gap_mean']:+.3f}  lambda: largest "
              f"median {c['lambda_largest_median']:.3f} at {c['lambda_largest_median_step'] / 1e6:.2f}M, largest "
              f"{c['lambda_largest']:.3f}, time-average mean {c['lambda_time_mean']:.4f}")
    d = np.array(C[DESC_NG]["calib_per_run"]) - np.array(C[OCT_NG]["calib_per_run"])
    k1_ci = [float(v) for v in A.paired_bootstrap(d, N_BOOT, 0)[95][0]]
    res["K1_diff"] = {"mean": float(d.mean()), "ci95": k1_ci, "per_seed": dict(zip(SEEDS, map(float, d)))}
    print(f"  per-run ratio, {DESC_NG} minus {OCT_NG}, paired by seed: {d.mean():+.3f} {ci(k1_ci)}")

    # ------------------------------------------------------------------ K2
    runs = [r for rs in arms.values() for r in rs]
    k2 = A.pair(runs, DESC_NG, OCT_NG, env=PG, seeds=set(SEEDS))[PG]
    res["K2"] = k2
    print("\nK2. Held-out return and cost")
    buf = io.StringIO()
    with redirect_stdout(buf):
        A.print_pair({PG: k2})
    print(buf.getvalue().rstrip())

    # ------------------------------------------------------------------ K3
    if has_gate:
        k3 = A.pair(runs, DESC_G, DESC_NG, env=PG, seeds=set(SEEDS))[PG]
        h2 = A.h2(runs, DESC_G, DESC_NG, env=PG, seeds=set(SEEDS))[PG]
        dz = dose(arms[DESC_G])
        res["K3"] = {"pair": k3, "h2": h2, "dose": dz}
        print("\nK3. Gate ablation in this configuration")
        buf = io.StringIO()
        with redirect_stdout(buf):
            A.print_pair({PG: k3})
        print(buf.getvalue().rstrip())
        print(f"  time-averaged lambda: {h2['lambda_a']:.4f} vs {h2['lambda_b']:.4f}; median paired difference "
              f"{h2['median_diff']:+.4f}; one-sided Wilcoxon p {h2['p_one_sided']:.3f}")
        for name, row in dz.items():
            e, b = row["engagement"], row["budget_all"]
            print(f"  {name:10s} firing share {e['value']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]   budget(all) "
                  f"{b['value']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] of delta0")
    else:
        res["K3"] = None
        print("\nK3. Not run: this addendum has no gated arm.")

    # ------------------------------------------------------------------ K4
    print("\nK4. Runs with held-out cost <= 25")
    for k in arms:
        print(f"  {k:22s} {C[k]['cost_le_25']} of {len(arms[k])} (collapsed runs: {C[k]['collapsed']})")

    # ------------------------------------------------------------------ predictions and trigger
    q = {"Q1": bool(C[DESC_NG]["calib"] < 0.65),
         "Q2": bool(C[DESC_NG]["cost_mean"] > A.COST_LIMIT and C[DESC_NG]["cost_le_25"] == 0),
         "Q3": bool(k2["cost"]["ci95"][0] <= 0 <= k2["cost"]["ci95"][1])}
    if has_gate:
        k3p, dz = res["K3"]["pair"], res["K3"]["dose"]
        q["Q4"] = bool(k3p["return"]["ci95"][0] <= 0 <= k3p["return"]["ci95"][1]
                       and k3p["cost"]["ci95"][0] <= 0 <= k3p["cost"]["ci95"][1]
                       and dz["whole run"]["budget_all"]["value"] > 0.9)
    res["predictions"] = q
    trigger = not (k1_ci[0] <= 0 <= k1_ci[1]) or not (k2["cost"]["ci95"][0] <= 0 <= k2["cost"]["ci95"][1])
    res["trigger_single_change_arms"] = bool(trigger)
    print("\nPredictions (stated before the runs; not tests): "
          + "; ".join(f"{k} {'held' if v else 'did not hold'}" for k, v in q.items()))
    print(f"Contingent step: the single-change arms are {'TO BE RUN' if trigger else 'not run'} "
          f"(ratio difference {ci(k1_ci)}, cost difference {ci(k2['cost']['ci95'])}).")
    return res


if __name__ == "__main__":
    st = "--selftest" in sys.argv[1:]
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = main(st)
    text = buf.getvalue()
    print(text, end="")
    if not st:
        with open(os.path.join(HERE, "k.txt"), "w") as f:
            f.write(text)
        A.write_json(out, os.path.join(HERE, "k.json"))
