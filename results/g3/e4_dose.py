"""E4 dose-response (prereg section 9, descriptive; section 16 item 6). Seeds 100-107, held-out endpoint.

delta0 family: lagu_nogate at delta0 2.5 (E1 arm), 1.75, 1.25; the E2 matched arm (1.12158) is shown
as an extra point on the same axis, outside the pre-registered family. T family: lagu at T 0.035,
0.07 (E1 arm), 0.14. Arm means with 95% percentile bootstrap intervals over seeds (10,000 resamples,
seed 0), collapses (held-out return < 5), time-averaged lambda and the mean budget over all policy
updates as a fraction of the nominal 2.5.
"""
import csv
import json

import numpy as np

R = "results/gate"
SEEDS = range(100, 108)
FAMILIES = {
    "delta0 (lagu_nogate)": [("2.5", "lagu_nogate-tt"), ("1.75", "lagu_nogate-tt-d175"),
                             ("1.25", "lagu_nogate-tt-d125"), ("1.12 (E2, extra)", "lagu_nogate-tt-match")],
    "T (lagu)": [("0.035", "lagu-tt-T035"), ("0.07", "lagu-tt"), ("0.14", "lagu-tt-T14")],
}


def wmean(x, w):
    ok = np.isfinite(x) & (w > 0)
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")


def run(label, s):
    stem = f"{R}/{label}_SafetyPointGoal1-v0_s{s}"
    h = json.load(open(stem + ".heldout.json"))["pooled"]
    rows = list(csv.DictReader(open(stem + ".csv")))
    col = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    return h["eval_return"], h["eval_cost"], wmean(col("lambda_avg"), col("n_pi")), wmean(col("delta_avg"), col("n_pi"))


def ci(x):
    idx = np.random.default_rng(0).integers(0, len(x), (10000, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


for fam, arms in FAMILIES.items():
    print(f"== {fam}")
    print(f"  {'knob':18s} {'return':>22s} {'cost':>24s} {'collapsed':>9s} {'lambda med':>10s} {'budget/2.5':>10s}")
    means = []
    for knob, label in arms:
        v = np.array([run(label, s) for s in SEEDS])
        r, c = v[:, 0], v[:, 1]
        (rl, rh), (cl, ch) = ci(r), ci(c)
        means.append((knob, r.mean(), c.mean()))
        print(f"  {knob:18s} {r.mean():6.2f} [{rl:6.2f},{rh:6.2f}]  {c.mean():6.2f} [{cl:6.2f},{ch:6.2f}] "
              f"{int((r < 5).sum()):5d}/8   {np.median(v[:, 2]):10.3f} {np.nanmean(v[:, 3]) / 2.5:9.2f}")
    pre = [m for m in means if "extra" not in m[0]]
    mono = lambda xs: all(a <= b for a, b in zip(xs, xs[1:])) or all(a >= b for a, b in zip(xs, xs[1:]))
    print(f"  monotone arm means (pre-registered points only): return {mono([m[1] for m in pre])}, "
          f"cost {mono([m[2] for m in pre])}")
