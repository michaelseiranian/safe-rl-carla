"""Summarise a results directory and run the pre-registered analyses (prereg.md sections 5-10).

    python -m lagu.analyze results/pilot                                # table + curves (default)
    python -m lagu.analyze results/e0_dual --e0-rule                    # 10.2 -> e0_decision.json
    python -m lagu.analyze results/gate --budget-D lagu-tt              # 10.4: prints D alone
    python -m lagu.analyze results/gate --pair lagu-tt lagu_nogate-tt [--end-step N]
    python -m lagu.analyze results/gate --h2 lagu-tt lagu_nogate-tt
    python -m lagu.analyze results/gate_literal --pair lagu-lit lagu_nogate-lit --h5
    python -m lagu.analyze results/gate --unpaired lagu-tt td3lag-tt     # descriptive IQM
    python -m lagu.analyze --prereg prereg.md results/gate results/gate_literal results/gate_x

Default table: per (env, arm[-variant]) across seeds, using each run's last evaluation row: return
(mean, sd, IQM with a bootstrap 95% CI), cost, violation rate, mean gate engagement over the run,
final lambda, and critic calibration (Q_c and Q_bar at the first state divided by the realised
Monte-Carlo cost-to-go and return). Writes one curves PNG per env when matplotlib is available.

Runs are labelled <arm>[-<tag>] as in their stems. A run's primary endpoint is its
<stem>.heldout.json (lagu.evaluate) when every run in the comparison has one for the same end step
N; otherwise the mean of its eval rows with step >= 0.8N (N = --end-step or the run's last step).
Columns missing from older CSVs read as NaN. numpy only (no scipy).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import warnings

import numpy as np

NAN = float("nan")
N_BOOT = 10_000
COST_LIMIT = 25.0
COLLAPSE = 5.0                    # primary return below this, or a diverged run, is a collapse
MARGIN_RET, MARGIN_COST = 2.5, 5.0
EARLY = (25_001, 300_000)         # engagement window: 25k < step <= 300k
PRIMARY_ENV = "SafetyPointGoal1-v0"
# prereg.md 11: a non-finite value in these columns (after the warm-up) means the run diverged
DIVERGE = ("eval_return", "eval_cost", "qc_traj", "qbar_traj", "q_loss", "c_loss", "lambda")
SECONDARY = ("window_return", "window_cost", "cum_cost", "cum_cost_500k", "lambda_avg",
             "lambda_shadow", "engagement", "would_engage", "qc_avg", "calib")


def _num(v) -> float:
    try:
        return float(v) if v not in ("", None) else NAN
    except (TypeError, ValueError):
        return NAN


def load(dirpath: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(dirpath, "*.csv"))):
        stem = path[:-4]
        if not os.path.exists(stem + ".json"):
            continue
        with open(path) as f:
            rows = [{k: _num(v) for k, v in r.items() if k is not None} for r in csv.DictReader(f)]
        rows = [r for r in rows if np.isfinite(r.get("step", NAN))]
        if not rows:
            continue
        with open(stem + ".json") as f:
            meta = json.load(f)
        a = meta["args"]
        held = None
        if os.path.exists(stem + ".heldout.json"):
            with open(stem + ".heldout.json") as f:
                held = json.load(f)
        label = a["arm"] + ("-" + a["tag"] if a.get("tag") else "")
        runs.append({"arm": label, "env": a["env"], "seed": a["seed"], "rows": rows,
                     "stem": os.path.basename(stem), "tag": a.get("tag") or "",
                     "steps": int(a.get("steps") or 0), "start_steps": int(a.get("start_steps") or 0),
                     "cfg": meta.get("cfg", {}), "heldout": held})
    return runs


def iqm(x) -> float:
    x = np.sort(np.asarray(x, float))
    k = len(x) // 4
    return float(x[k:len(x) - k].mean()) if len(x) >= 4 else float(x.mean())


def boot_ci(x, stat=iqm, n: int = 2000, seed: int = 0):
    """Percentile bootstrap CI; undefined below 5 runs, where it only restates the min and max."""
    x = np.asarray(x, float)
    if len(x) < 5:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    s = [stat(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def ratio(num, den):
    """Ratio of means over runs with both values finite (robust to near-zero denominators)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    return float(num[ok].mean() / den[ok].mean()) if ok.any() and abs(den[ok].mean()) > 1e-9 else NAN


def sd(x) -> float:
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if len(x) > 1 else NAN


def at_common_step(rs: list[dict]) -> list[dict]:
    """Each run's row at the largest step every run has reached."""
    common = min(r["rows"][-1]["step"] for r in rs)
    return [max((row for row in r["rows"] if row["step"] <= common), key=lambda row: row["step"]) for r in rs]


# ---------------------------------------------------------------- statistics (numpy only)

def fmean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else NAN


def fmedian(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else NAN


def wmean(x, w) -> float:
    """Weighted mean over entries where both are finite and the weight is positive."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else NAN


def ratio_sum(num, den) -> float:
    """sum(num) / sum(den) over entries where both are finite."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    return float(num[ok].sum() / den[ok].sum()) if ok.any() and abs(den[ok].sum()) > 1e-9 else NAN


def paired_bootstrap(d, n_boot: int = N_BOOT, seed: int = 0, levels=(95, 90)) -> dict:
    """Percentile CIs of the mean paired difference, resampling pairs (rows of d) with replacement.
    Columns of d (e.g. return and cost) share each resample. Returns {level: array (k, 2)}."""
    d = np.asarray(d, float)
    d = d[:, None] if d.ndim == 1 else d
    if len(d) < 2:
        return {lv: np.full((d.shape[1], 2), NAN) for lv in levels}
    means = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return {lv: np.percentile(means, [(100 - lv) / 2, (100 + lv) / 2], axis=0).T for lv in levels}


def midranks(x) -> np.ndarray:
    x = np.asarray(x, float)
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def wilcoxon(d, alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank test of the paired differences d against 0. Zero differences are
    dropped and tied |d| get average ranks. Exact for n <= 20 by enumerating all 2^n sign flips of
    the ranks; normal approximation (tie-corrected, no continuity correction) above that.
    alternative: 'two-sided', 'greater' (d > 0) or 'less' (d < 0). No finite difference: p = NaN
    (no data); finite differences that are all zero: p = 1 (prereg.md 6)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"n": 0, "w_plus": NAN, "p": NAN, "exact": True}
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "p": 1.0, "exact": True}
    r = midranks(np.abs(d))
    w, mu = float(r[d > 0].sum()), float(r.sum()) / 2
    if n <= 20:
        dist = np.zeros(1)
        for x in r:                          # W+ under every sign assignment (2^n values)
            dist = np.concatenate([dist, dist + x])
        tol = 1e-9
        if alternative == "greater":
            p = np.mean(dist >= w - tol)
        elif alternative == "less":
            p = np.mean(dist <= w + tol)
        else:
            p = np.mean(np.abs(dist - mu) >= abs(w - mu) - tol)
        return {"n": n, "w_plus": w, "p": float(p), "exact": True}
    _, t = np.unique(np.abs(d), return_counts=True)
    z = (w - mu) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - float((t ** 3 - t).sum()) / 48)
    p = {"greater": 0.5 * math.erfc(z / math.sqrt(2)), "less": 0.5 * math.erfc(-z / math.sqrt(2))}.get(
        alternative, math.erfc(abs(z) / math.sqrt(2)))
    return {"n": n, "w_plus": w, "p": float(p), "exact": False}


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]]: the total hypergeometric
    probability of the tables (same margins) no more probable than the observed one."""
    r1, r2, c1 = a + b, c + d, a + c
    counts = {x: math.comb(r1, x) * math.comb(r2, c1 - x) for x in range(max(0, c1 - r2), min(r1, c1) + 1)}
    return sum(v for v in counts.values() if v <= counts[a]) / math.comb(r1 + r2, c1)


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as p)."""
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p, kind="mergesort")):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def prob_improvement(x, y, sign: int = 1):
    """P(sign*X > sign*Y) + P(X == Y)/2 over all (x, y) pairs; batched over leading axes."""
    x, y = sign * np.asarray(x, float)[..., :, None], sign * np.asarray(y, float)[..., None, :]
    return ((x > y) + 0.5 * (x == y)).mean(axis=(-2, -1))


def _iqm_rows(m) -> np.ndarray:
    m = np.sort(m, axis=1)
    k = m.shape[1] // 4
    return m[:, k:m.shape[1] - k].mean(axis=1)


def verdict(ret95, cost95, ret90, cost90, collapses_a: int = 0, collapses_b: int = 0) -> str:
    """prereg.md section 7 on A - B; the first label that matches applies (NaN bounds never match)."""
    (rl, rh), (cl, ch) = ret95, cost95
    if (ch < 0 and rl > -MARGIN_RET) or (rl > 0 and ch < MARGIN_COST):
        return "Helps"
    if (cl > 0 and rh < MARGIN_RET) or (rh < 0 and cl > -MARGIN_COST) or collapses_a - collapses_b >= 3:
        return "Hurts"
    if (rl > 0 and cl > 0) or (rh < 0 and ch < 0):
        return "Shift"
    (rl, rh), (cl, ch) = ret90, cost90
    if -MARGIN_RET < rl and rh < MARGIN_RET and -MARGIN_COST < cl and ch < MARGIN_COST:
        return "Equivalent"
    return "Inconclusive"


# ---------------------------------------------------------------- per-run quantities

def cells(runs: list[dict], env: str | None = None, seeds=None) -> dict:
    """{(env, label): {seed: run}}; the same (env, label, seed) in two directories is an error."""
    out: dict = {}
    for r in runs:
        if (env and r["env"] != env) or (seeds is not None and r["seed"] not in seeds):
            continue
        g = out.setdefault((r["env"], r["arm"]), {})
        if r["seed"] in g:
            raise SystemExit(f"{r['stem']} appears twice ({g[r['seed']]['stem']}); pass one directory per run")
        g[r["seed"]] = r
    return out


def envs_with(G: dict, *labels) -> list[str]:
    return sorted({e for e, _ in G if all((e, lab) in G for lab in labels)})


def col(run: dict, key: str, lo: float = -np.inf, hi: float = np.inf) -> np.ndarray:
    return np.array([r.get(key, NAN) for r in run["rows"] if lo <= r["step"] <= hi], float)


def at(run: dict, key: str, step: float) -> float:
    """key in the run's last row with step <= step (NaN if none)."""
    rows = [r for r in run["rows"] if r["step"] <= step]
    return max(rows, key=lambda r: r["step"]).get(key, NAN) if rows else NAN


def diverged(run: dict, upto: float = np.inf) -> bool:
    """prereg.md 11: a non-finite value in a DIVERGE column the CSV has, in a row after the random
    warm-up and at or before step `upto` (the analysis horizon N)."""
    lo = run.get("start_steps", 0)
    return any(k in r and not np.isfinite(r[k]) for r in run["rows"] if lo < r["step"] <= upto for k in DIVERGE)


def _mean(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean()) if len(x) else NAN


def _finite(*xs) -> bool:
    """Every array non-empty and every value finite (prereg.md 10.2: a non-finite value that a
    condition reads makes the cell fail that condition)."""
    return all(np.size(x) > 0 and bool(np.all(np.isfinite(x))) for x in xs)


def end_of(run: dict, end=None) -> float:
    """N: --end-step, else the run's step count (or its last row if a resume went beyond it)."""
    return float(end) if end else float(max(run["steps"], run["rows"][-1]["step"]))


def lambda_time(run: dict) -> float:
    """Time-averaged lambda: sum(lambda_avg * n_pi) / sum(n_pi) over all rows of the run."""
    return wmean(col(run, "lambda_avg"), col(run, "n_pi"))


def engagement(rs: list[dict], lo: float = EARLY[0], hi: float = EARLY[1]) -> float:
    """sum(engagement * n_risk) / sum(n_risk) over the rows in [lo, hi], pooled over runs."""
    return wmean(np.concatenate([col(r, "engagement", lo, hi) for r in rs] or [[]]),
                 np.concatenate([col(r, "n_risk", lo, hi) for r in rs] or [[]]))


def heldout_ok(run: dict, end=None) -> bool:
    h = run.get("heldout")
    return bool(h) and "pooled" in h and _num(h.get("end_step")) == end_of(run, end)


def endpoints(run: dict, end=None, held: bool = False) -> dict:
    """Primary (return, cost, calibration) and secondary endpoints of one run (prereg.md 5).
    Evaluation columns use the rows with step >= 0.8N, interval columns (*_avg, would_engage) the
    rows with step > 0.8N: the row at 0.8N summarises the updates before it. A run collapses if
    its primary return is below COLLAPSE or it diverged (prereg.md 11); a run with no endpoint
    data (e.g. incomplete, no rows in the window) is missing, not collapsed."""
    N = end_of(run, end)
    w = lambda k: col(run, k, 0.8 * N - 1e-6, N)
    wi = lambda k: col(run, k, 0.8 * N + 1e-6, N)
    last = run["rows"][-1]["step"]
    e = {"N": N, "complete": bool(last >= N), "n_window": int(len(w("step"))),
         "resumed": bool(np.any(col(run, "resumed") == 1)),
         # NaN-propagating: a non-finite eval row makes the window mean non-finite (prereg.md 11)
         "window_return": _mean(w("eval_return")), "window_cost": _mean(w("eval_cost")),
         "cum_cost": at(run, "cum_cost", N),
         "cum_cost_500k": at(run, "cum_cost", 500_000) if last >= 500_000 else NAN,
         "lambda_avg": lambda_time(run), "lambda_shadow": at(run, "lambda_shadow", N),
         "engagement": engagement([run]),
         "would_engage": wmean(wi("would_engage"), wi("n_risk")),
         "qc_avg": wmean(wi("qc_avg"), wi("n_pi")), "gap_avg": wmean(wi("gap_avg"), wi("n_pi")),
         "diverged": diverged(run, N)}
    if held:
        h = run["heldout"]
        snaps = h.get("per_snapshot") or [h["pooled"]]
        e["return"], e["cost"] = _num(h["pooled"].get("eval_return")), _num(h["pooled"].get("eval_cost"))
        qc, mc = (np.array([_num(s.get(k)) for s in snaps]) for k in ("qc_traj", "mc_cost_traj"))
        e["diverged"] = bool(e["diverged"] or not (np.isfinite(e["return"]) and np.isfinite(e["cost"])))
    else:
        e["return"], e["cost"] = e["window_return"], e["window_cost"]
        qc, mc = w("qc_traj"), w("mc_cost_traj")
    e["collapsed"] = bool(e["diverged"] or e["return"] < COLLAPSE)
    ok = np.isfinite(qc) & np.isfinite(mc)
    e["qc"], e["mc"] = (float(qc[ok].sum()), float(mc[ok].sum())) if ok.any() else (NAN, NAN)
    e["calib"] = ratio_sum([e["qc"]], [e["mc"]])
    return e


def pooled_calib(E: list[dict], n_boot: int = N_BOOT, seed: int = 0):
    """sum(qc_traj) / sum(mc_cost_traj) over runs, and its 95% bootstrap CI over runs."""
    qm = np.array([(e["qc"], e["mc"]) for e in E if np.isfinite(e["qc"]) and np.isfinite(e["mc"])], float)
    r = ratio_sum(qm[:, 0], qm[:, 1]) if len(qm) else NAN
    if len(qm) < 2 or not np.isfinite(r):
        return r, NAN, NAN
    idx = np.random.default_rng(seed).integers(0, len(qm), (n_boot, len(qm)))
    with np.errstate(divide="ignore", invalid="ignore"):
        b = qm[idx, 0].sum(1) / qm[idx, 1].sum(1)
    lo, hi = np.percentile(b[np.isfinite(b)], [2.5, 97.5])
    return r, float(lo), float(hi)


# ---------------------------------------------------------------- analyses

def contrast(E: list, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Paired primary analysis of A - B (prereg.md 6-7). E = [(seed, endpoints A, endpoints B)].
    Pairs with a non-finite return or cost leave the intervals and tests; a diverged run still
    counts as a collapse (prereg.md 11)."""
    ra, rb = (np.array([x[i]["return"] for x in E], float) for i in (1, 2))
    ca, cb = (np.array([x[i]["cost"] for x in E], float) for i in (1, 2))
    dr, dc = ra - rb, ca - cb
    ok = np.isfinite(dr) & np.isfinite(dc)
    ci = paired_bootstrap(np.column_stack([dr[ok], dc[ok]]), n_boot, seed)
    n = len(E)
    out = {"n_pairs": n, "n_left_out": int(n - ok.sum()), "seeds": [x[0] for x in E]}
    for i, (name, a, b, d, m) in enumerate((("return", ra, rb, dr, MARGIN_RET), ("cost", ca, cb, dc, MARGIN_COST))):
        lo90, hi90 = ci[90][i]
        out[name] = {"mean_a": fmean(a[ok]), "mean_b": fmean(b[ok]), "diff": fmean(d[ok]),
                     "ci95": [float(v) for v in ci[95][i]], "ci90": [float(lo90), float(hi90)],
                     "wilcoxon_p": wilcoxon(d[ok])["p"], "tost_equivalent": bool(-m < lo90 and hi90 < m)}
    out["collapses_a"], out["collapses_b"] = (sum(bool(x[i]["collapsed"]) for x in E) for i in (1, 2))
    out["fisher_p"] = fisher_exact(out["collapses_a"], n - out["collapses_a"], out["collapses_b"], n - out["collapses_b"])
    out["verdict"] = verdict(out["return"]["ci95"], out["cost"]["ci95"], out["return"]["ci90"],
                             out["cost"]["ci90"], out["collapses_a"], out["collapses_b"])
    return out


def pair(runs, A: str, B: str, end=None, env=None, n_boot: int = N_BOOT, seed: int = 0, seeds=None) -> dict:
    """Paired contrast A - B per env: primary endpoints, verdict, secondary endpoints and the
    sensitivity analysis without the pairs that contain a resumed run."""
    G, out = cells(runs, env, seeds), {}
    for e in envs_with(G, A, B):
        ga, gb = G[(e, A)], G[(e, B)]
        common = sorted(set(ga) & set(gb))
        if not common:
            continue
        held = all(heldout_ok(r, end) for s in common for r in (ga[s], gb[s]))
        E = [(s, endpoints(ga[s], end, held), endpoints(gb[s], end, held)) for s in common]
        res = contrast(E, n_boot, seed)
        resumed = [s for s, a, b in E if a["resumed"] or b["resumed"]]
        res.update(env=e, a=A, b=B, source="heldout" if held else "eval rows with step >= 0.8N",
                   N=sorted({x[i]["N"] for x in E for i in (1, 2)}),
                   unpaired_seeds=sorted(set(ga) ^ set(gb)),
                   incomplete=[g[s]["stem"] for s, a, b in E for g, x in ((ga, a), (gb, b)) if not x["complete"]],
                   diverged=[g[s]["stem"] for s, a, b in E for g, x in ((ga, a), (gb, b)) if x["diverged"]],
                   missing_heldout=[] if held else [g[s]["stem"] for s in common for g in (ga, gb)
                                                    if not heldout_ok(g[s], end)],
                   resumed_seeds=resumed,
                   sensitivity=(contrast([x for x in E if x[0] not in resumed], n_boot, seed)
                                if resumed else None))
        sec = {}
        for k in SECONDARY:
            a, b = (np.array([x[i][k] for x in E], float) for i in (1, 2))
            d = a - b
            ok = np.isfinite(d)
            sec[k] = {"mean_a": fmean(a), "mean_b": fmean(b), "diff": fmean(d[ok]), "n": int(ok.sum()),
                      "ci95": [float(v) for v in paired_bootstrap(d[ok], n_boot, seed)[95][0]],
                      "wilcoxon_p": wilcoxon(d[ok])["p"]}
        res["secondary"] = sec
        out[e] = res
    return out


def h2_status(eng: float, med: float, p: float, alpha: float = 0.05) -> str:
    if not (np.isfinite(eng) and np.isfinite(med) and np.isfinite(p)):
        return "undetermined"
    if eng < 0.10:
        return "refuted_vacuous"
    if med <= 0:
        return "refuted_lambda"
    return "confirmed" if eng > 0.10 and p < alpha else "not_confirmed"


def h2(runs, A: str, B: str, env=None, seeds=None) -> dict:
    """H2: engagement of A over 25k < step <= 300k, and one-sided exact Wilcoxon that A's
    time-averaged lambda exceeds B's, paired by seed. Status uses the unadjusted p."""
    G, out = cells(runs, env, seeds), {}
    for e in envs_with(G, A, B):
        ga, gb = G[(e, A)], G[(e, B)]
        common = sorted(set(ga) & set(gb))
        la = np.array([lambda_time(ga[s]) for s in common], float)
        lb = np.array([lambda_time(gb[s]) for s in common], float)
        d = la - lb
        fin = d[np.isfinite(d)]
        med = fmedian(fin)
        eng = engagement(list(ga.values()))
        p = wilcoxon(fin, "greater")["p"]
        out[e] = {"a": A, "b": B, "seeds": common, "engagement_a": eng,
                  "engagement_b": engagement(list(gb.values())), "lambda_a": fmean(la), "lambda_b": fmean(lb),
                  "median_diff": med, "n": int(len(fin)), "p_one_sided": p, "status": h2_status(eng, med, p)}
    return out


def h5_arm(rs: list[dict], end=None, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """H5 for one literal-recipe arm: held-out cost IQM CI, |gap| vs delta0/4, pooled calibration."""
    held = all(heldout_ok(r, end) for r in rs)
    E = [endpoints(r, end, held) for r in rs]
    cost = np.array([e["cost"] for e in E], float)
    cost = cost[np.isfinite(cost)]
    lo, hi = boot_ci(cost, iqm, n_boot, seed)
    gap = fmean([e["gap_avg"] for e in E])
    delta0 = _num(rs[0]["cfg"].get("delta0"))
    cal, cal_lo, cal_hi = pooled_calib(E, n_boot, seed)
    icost = iqm(cost) if len(cost) else NAN
    parts = {"cost_iqm_lb_above_25": bool(lo > COST_LIMIT), "gap_within_quarter_delta0": bool(abs(gap) <= 0.25 * delta0),
             "calib_below_0.65": bool(cal < 0.65), "calib_ub_below_1": bool(cal_hi < 1)}
    return {"source": "heldout" if held else "eval rows with step >= 0.8N", "seeds": sorted(r["seed"] for r in rs),
            "cost_iqm": icost, "cost_iqm_ci95": [lo, hi], "gap": gap, "delta0": delta0,
            "calib": cal, "calib_ci95": [cal_lo, cal_hi],
            "p_cost_above_25": wilcoxon(cost - COST_LIMIT, "greater")["p"], "parts": parts,
            "confirmed": all(parts.values()), "refuted": bool(cal >= 0.65 or icost <= COST_LIMIT)}


def h5(runs, labels, end=None, env=None, n_boot: int = N_BOOT, seed: int = 0, seeds=None) -> dict:
    G = cells(runs, env, seeds)
    return {f"{e} {lab}": {"env": e, "label": lab, **h5_arm([g[s] for s in sorted(g)], end, n_boot, seed)}
            for (e, lab), g in sorted(G.items()) if lab in labels}


def unpaired(runs, A: str, B: str, end=None, env=None, n_boot: int = N_BOOT, seed: int = 0, seeds=None) -> dict:
    """Descriptive A vs B, not paired: IQM over seeds with stratified (per-arm) bootstrap 95% CIs
    and the probability of improvement (higher return, lower cost)."""
    G, out = cells(runs, env, seeds), {}
    for e in envs_with(G, A, B):
        ra, rb = list(G[(e, A)].values()), list(G[(e, B)].values())
        held = all(heldout_ok(r, end) for r in ra + rb)
        Ea, Eb = [endpoints(r, end, held) for r in ra], [endpoints(r, end, held) for r in rb]
        res = {"a": A, "b": B, "n_a": len(ra), "n_b": len(rb), "source": "heldout" if held else "eval rows with step >= 0.8N"}
        for k, sign in (("return", 1), ("cost", -1)):
            xa, xb = (np.array([x[k] for x in E], float) for E in (Ea, Eb))
            xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
            if not (len(xa) and len(xb)):
                res[k] = None
                continue
            rng = np.random.default_rng(seed)
            ba = xa[rng.integers(0, len(xa), (n_boot, len(xa)))]
            bb = xb[rng.integers(0, len(xb), (n_boot, len(xb)))]
            ia, ib = _iqm_rows(ba), _iqm_rows(bb)
            q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            res[k] = {"iqm_a": iqm(xa), "iqm_b": iqm(xb), "ci_a": q(ia), "ci_b": q(ib),
                      "diff": iqm(xa) - iqm(xb), "ci_diff": q(ia - ib),
                      "prob_improvement": float(prob_improvement(xa, xb, sign)),
                      "ci_prob_improvement": q(prob_improvement(ba, bb, sign))}
        out[e] = res
    return out


def budget_D(runs, label: str, env=None, seeds=None) -> dict:
    """D = sum(delta_avg * n_pi) / sum(n_pi) over every row with policy updates (n_pi > 0),
    pooled over the seeds of one label (prereg.md 10.4), and the G2 branch it implies."""
    G = cells(runs, env, seeds)
    keys = [k for k in G if k[1] == label]
    if len(keys) != 1:
        raise SystemExit(f"--budget-D {label}: found in envs {sorted(e for e, _ in keys)}; need exactly one (use --env)")
    rs = [G[keys[0]][s] for s in sorted(G[keys[0]])]
    D = wmean(np.concatenate([col(r, "delta_avg") for r in rs]), np.concatenate([col(r, "n_pi") for r in rs]))
    eng = engagement(rs)
    branch = ("undetermined" if not (np.isfinite(D) and np.isfinite(eng)) else
              "vacuous: skip the matched arm; enqueue ggrad and E4 at T in {0.035, 0.0175}" if eng < 0.10 else
              "D > 2.4: skip the matched arm; enqueue ggrad and E4's T-family" if D > 2.4 else "enqueue E2")
    return {"env": keys[0][0], "label": label, "seeds": sorted(G[keys[0]]), "D": D, "engagement": eng,
            "incomplete": [r["stem"] for r in rs if r["rows"][-1]["step"] < r["steps"]], "g2_branch": branch}


def e0_rule(runs, end: int = 1_000_000, env=None, seeds=None) -> dict:
    """prereg.md 10.2: conditions (a)-(d) per cell on each seed's rows with 0.85 end <= step <= end
    (850k, 900k, 950k, 1M), the outcome class (i)-(v), the chosen cell and the step budget. A
    non-finite value that a condition reads fails it; a diverged seed (up to `end`) fails (b)."""
    G = cells(runs, env, seeds)
    if len({e for e, _ in G}) != 1:
        raise SystemExit(f"--e0-rule needs runs of exactly one env, found {sorted({e for e, _ in G})} (use --env)")
    v = lambda rows, k: np.array([r.get(k, NAN) for r in rows], float)
    res = {}
    for (e, lab), g in sorted(G.items()):
        rs = [g[s] for s in sorted(g)]
        W = [[r for r in run["rows"] if 0.85 * end - 1e-6 <= r["step"] <= end] for run in rs]
        rows = [r for w in W for r in w]
        qc = v(rows, "qc_traj")
        lam600, lam_end = ([at(r, "lambda_avg", t) for r in rs] for t in (0.6 * end, end))
        tc_early = np.concatenate([col(r, "train_cost", 0.25 * end, 0.4 * end) for r in rs])
        div = [s for s, r in zip(sorted(g), rs) if diverged(r, end)]
        c = {"env": e, "seeds": sorted(g), "window_steps": [[int(r["step"]) for r in w] for w in W],
             "complete": all(run["rows"][-1]["step"] >= end for run in rs),
             "lambda_lr": _num(rs[0]["cfg"].get("lambda_lr")),
             "kind": "tt" if rs[0]["tag"].startswith("tt") else "omsw" if rs[0]["tag"].startswith("omsw") else "other",
             "calib": ratio_sum(qc, v(rows, "mc_cost_traj")),
             "calib_seed": [ratio_sum(v(w, "qc_traj"), v(w, "mc_cost_traj")) for w in W],
             "return": fmean(v(rows, "eval_return")), "return_seed": [fmean(v(w, "eval_return")) for w in W],
             "lambda": fmean(v(rows, "lambda_avg")), "gap": fmean(v(rows, "gap_avg")),
             "lambda_600k": fmean(lam600), "lambda_end": fmean(lam_end),
             "cost": fmean(v(rows, "eval_cost")), "train_cost": fmean(v(rows, "train_cost")),
             "train_cost_250_400k": fmean(tc_early), "diverged_seeds": div,
             "would_engage": wmean(v(rows, "would_engage"), v(rows, "n_risk")),
             "would_engage_25_300k": wmean(np.concatenate([col(r, "would_engage", *EARLY) for r in rs]),
                                           np.concatenate([col(r, "n_risk", *EARLY) for r in rs])),
             "qbar_avg": fmean(v(rows, "qbar_avg")),
             "abs_qbar_avg_median_after_300k": fmedian(np.abs(np.concatenate(
                 [col(r, "qbar_avg", 300_001, end) for r in rs])))}
        c["a"] = bool(_finite(qc, v(rows, "mc_cost_traj")) and 0.65 <= c["calib"] <= 1.35
                      and all(x >= 0.5 for x in c["calib_seed"]) and not np.any(qc < 0))
        c["b"] = bool(_finite(v(rows, "eval_return")) and not div and c["return"] >= 20
                      and all(x >= 10 for x in c["return_seed"]))
        c["c"] = bool(_finite(v(rows, "lambda_avg"), v(rows, "gap_avg"), lam600, lam_end) and c["lambda"] >= 0.02
                      and (abs(c["gap"]) <= 0.3 or c["lambda_end"] > 1.2 * c["lambda_600k"]))
        c["d"] = bool(_finite(v(rows, "eval_cost"), v(rows, "train_cost"), tc_early)
                      and (c["cost"] <= 35 or c["train_cost"] <= c["train_cost_250_400k"] - 10))
        res[lab] = c
    speed = lambda lab: (res[lab]["lambda_lr"] if np.isfinite(res[lab]["lambda_lr"]) else -1.0, res[lab]["kind"] == "tt")
    healthy = [lab for lab, c in res.items() if c["a"] and c["b"]]
    binds = [lab for lab, c in res.items() if c["c"] and c["d"]]
    full = lambda kind: [lab for lab in healthy if lab in binds and res[lab]["kind"] == kind]
    if not any(c["a"] for c in res.values()):
        outcome, chosen, reason = "v", None, "no cell passes (a): STOP and debug; only E1b may run"
    elif full("tt"):
        outcome, chosen, reason = "i", max(full("tt"), key=speed), "fastest tt cell passing (a)-(d)"
    elif full("omsw"):
        outcome, chosen, reason = "ii", max(full("omsw"), key=speed), "no tt cell passes; omsw passes (a)-(d)"
    elif not healthy:
        outcome, chosen, reason = "v", None, "(iii)/(iv) would apply but no cell is healthy: STOP and debug"
    elif not binds:
        outcome, chosen, reason = "iii", max(healthy, key=speed), "no cell binds: fastest healthy cell at 2x steps"
    else:
        outcome, chosen, reason = "iv", max(healthy, key=speed), ("some cells bind but none is healthy: fastest "
                                                                    "healthy cell at 2x steps; lead with the negative result")
    steps = None
    if outcome in ("i", "ii"):
        c = res[chosen]
        stable = abs(c["lambda_end"] - c["lambda_600k"]) < 0.2 * c["lambda_600k"] and abs(c["gap"]) <= 0.3
        steps = int(end if stable else 2 * end)
    elif outcome in ("iii", "iv"):
        steps = int(2 * end)
    ext = None
    if steps == 2 * end:   # 10.3 E0-ext: >= 2 of the chosen cell's seeds below return 5 over 1.6N-2N
        rs = [G[(res[chosen]["env"], chosen)][s] for s in res[chosen]["seeds"]]
        W = [col(r, "eval_return", 1.6 * end - 1e-6, 2 * end) for r in rs]
        means = [float(w.mean()) if len(w) else NAN for w in W]      # a non-finite row makes the mean NaN
        low = sum(len(w) > 0 and not m >= 5 for w, m in zip(W, means))   # NaN (diverged) counts as below 5
        ext = {"seed_mean_return": means, "complete": all(r["rows"][-1]["step"] >= 2 * end for r in rs),
               "move_window_to_end_step": bool(low >= 2)}
    return {"end_step": end, "cells": res, "healthy": healthy, "binds": binds, "outcome": outcome,
            "chosen": chosen, "lambda_lr": res[chosen]["lambda_lr"] if chosen else None, "steps": steps,
            "reason": reason, "complete": all(c["complete"] for c in res.values()), "e0_ext": ext}


def _sha256(path: str):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def prereg(runs, prereg_md: str, tag: str = "tt", env: str = PRIMARY_ENV, end=None,
           n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """prereg.md sections 6-9 over the E1, E1b, E2 and E3 runs: H2-H5 with Holm over the eight
    p-values of section 6 (a test without data leaves the family), plus the E3 and |Q_bar|
    secondary predictions. The --end-step window applies to E1/E2 on the primary env only."""
    A1, B1, B2, Al, Bl = f"lagu-{tag}", f"lagu_nogate-{tag}", f"lagu_nogate-{tag}-match", "lagu-lit", "lagu_nogate-lit"
    E1_SEEDS, E1B_SEEDS = range(100, 115), range(100, 110)
    fam: dict[str, float] = {}
    out: dict = {"prereg_md": os.path.abspath(prereg_md), "prereg_sha256": _sha256(prereg_md),
                 "analyze_sha256": _sha256(__file__), "env": env, "end_step": end, "tag": tag,
                 "labels": {"E1": [A1, B1], "E1b": [Al, Bl], "E2": [A1, B2]}, "n_boot": n_boot, "boot_seed": seed}
    try:
        out["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                         capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        out["git_head"] = None

    h2r = {"E1": h2(runs, A1, B1, env, E1_SEEDS).get(env), "E1b": h2(runs, Al, Bl, env, E1B_SEEDS).get(env)}
    for k, r in h2r.items():
        if r and np.isfinite(r["p_one_sided"]):
            fam[f"H2_{k}"] = r["p_one_sided"]
    h3 = pair(runs, A1, B1, end, env, n_boot, seed).get(env)
    h4 = pair(runs, A1, B2, end, env, n_boot, seed).get(env)
    for name, r in (("H3", h3), ("H4", h4)):
        for k in ("return", "cost"):
            if r and np.isfinite(r[k]["wilcoxon_p"]):
                fam[f"{name}_{k}"] = r[k]["wilcoxon_p"]
    h5r = h5(runs, (Al, Bl), None, env, n_boot, seed, E1B_SEEDS)
    for lab in (Al, Bl):
        r = h5r.get(f"{env} {lab}")
        if r and np.isfinite(r["p_cost_above_25"]):
            fam[f"H5_{lab}"] = r["p_cost_above_25"]

    names = list(fam)
    adj = dict(zip(names, holm([fam[k] for k in names]))) if names else {}
    out["holm"] = {k: {"p": fam[k], "p_holm": float(adj[k]), "reject_0.05": bool(adj[k] < 0.05)} for k in names}

    for k, r in h2r.items():
        if r:
            r["p_holm"] = adj.get(f"H2_{k}", NAN)
            r["status"] = h2_status(r["engagement_a"], r["median_diff"], r["p_holm"])
    st = [r["status"] if r else "missing" for r in h2r.values()]
    out["H2"] = {**h2r, "status": ("confirmed" if all(s == "confirmed" for s in st) else
                                   "refuted" if any(s.startswith("refuted") for s in st) else
                                   "missing" if "missing" in st else "not_confirmed")}
    vacuous = bool(h2r["E1"] and h2r["E1"]["engagement_a"] < 0.10)
    if h3:
        v = h3["verdict"]
        h3["status"] = ("refuted" if v == "Helps" else "inconclusive" if v == "Inconclusive" else "confirmed")
        h3["vacuous"] = vacuous
    out["H3"] = h3 or {"status": "missing"}
    if h4:
        v = h4["verdict"]
        zero_in = all(h4[k]["ci95"][0] <= 0 <= h4[k]["ci95"][1] for k in ("return", "cost"))
        h4["status"] = ("confirmed" if v == "Equivalent" or (v == "Inconclusive" and zero_in) else
                        "refuted" if v in ("Helps", "Hurts", "Shift") else "not_confirmed")
    out["H4"] = h4 or {"status": "missing (E2 not run)"}

    lit = [h5r.get(f"{env} {lab}") for lab in (Al, Bl)]
    nog = cells(runs, env, E1_SEEDS).get((env, B1))
    band = None
    if nog:
        rs = [nog[s] for s in sorted(nog)]
        held = all(heldout_ok(r, end) for r in rs)
        cal, lo, hi = pooled_calib([endpoints(r, end, held) for r in rs], n_boot, seed)
        band = {"label": B1, "calib": cal, "calib_ci95": [lo, hi], "in_band": bool(0.65 <= cal <= 1.35)}
    failing = [f"{lab}: {p}" for lab, r in zip((Al, Bl), lit) if r for p, ok in r["parts"].items() if not ok]
    if band and not band["in_band"]:
        failing.append(f"{B1}: calibration outside [0.65, 1.35]")
    status = ("missing" if not all(lit) else "refuted" if any(r["refuted"] for r in lit) else
              "confirmed" if not failing and band else "partially_supported")
    out["H5"] = {"literal": dict(zip((Al, Bl), lit)), "E1_nogate_calibration": band, "failing": failing, "status": status}

    e3 = {}
    if h3:
        for e, r in pair(runs, A1, B1, None, None, n_boot, seed).items():
            if e != env:
                same = all(np.sign(r[k]["diff"]) == np.sign(h3[k]["diff"]) for k in ("return", "cost"))
                e3[e] = {k: r[k] for k in ("return", "cost")} | {"n_pairs": r["n_pairs"], "same_signs_as_E1": bool(same)}
    out["secondary"] = {"E3": e3, "qbar_loop": qbar_loop(runs, A1, B1, env, end),
                        "not_computed_here": ["gradient parity (lagu/probe.py)", "E4 dose-response"]}
    return out


def qbar_loop(runs, A: str, B: str, env: str, end=None):
    """prereg.md 9: median |qbar_avg| over A's rows after 300k; if < 0.3, engagement there > 80%
    and the ratio of the cell means of final lambda (A / B) > 3."""
    G = cells(runs, env)
    if (env, A) not in G or (env, B) not in G:
        return None
    ra, rb = list(G[(env, A)].values()), list(G[(env, B)].values())
    med = fmedian(np.abs(np.concatenate([col(r, "qbar_avg", 300_001, end_of(r, end)) for r in ra])))
    eng = wmean(np.concatenate([col(r, "engagement", 300_001, end_of(r, end)) for r in ra]),
                np.concatenate([col(r, "n_risk", 300_001, end_of(r, end)) for r in ra]))
    lb = fmean([at(r, "lambda", end_of(r, end)) for r in rb])
    lam_ratio = fmean([at(r, "lambda", end_of(r, end)) for r in ra]) / lb if lb else NAN
    return {"median_abs_qbar_after_300k": med, "engagement_after_300k": eng, "final_lambda_ratio": lam_ratio,
            "triggered": bool(med < 0.3), "holds": bool(med < 0.3 and eng > 0.8 and lam_ratio > 3)}


# ---------------------------------------------------------------- output

def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, range, np.ndarray)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def write_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=1)
    print("wrote", path)


def _f(x, p: int = 2) -> str:
    return f"{x:.{p}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "nan"


def _ci(c, p: int = 2) -> str:
    return f"[{_f(c[0], p)}, {_f(c[1], p)}]"


def print_contrast(r: dict, indent: str = "  ") -> None:
    print(f"{indent}pairs {r['n_pairs']} (left out, non-finite: {r['n_left_out']})  seeds {r['seeds']}")
    for k, m in (("return", MARGIN_RET), ("cost", MARGIN_COST)):
        x = r[k]
        print(f"{indent}{k:6s} A {_f(x['mean_a'])}  B {_f(x['mean_b'])}  A-B {_f(x['diff'])}  95% {_ci(x['ci95'])}"
              f"  90% {_ci(x['ci90'])}  Wilcoxon p={_f(x['wilcoxon_p'], 4)}  TOST(+-{m}) {'yes' if x['tost_equivalent'] else 'no'}")
    print(f"{indent}collapses (return < {COLLAPSE:g} or diverged): A {r['collapses_a']}  B {r['collapses_b']}"
          f"  Fisher p={_f(r['fisher_p'], 4)}")
    print(f"{indent}verdict: {r['verdict']}")


def print_pair(res: dict) -> None:
    for e, r in res.items():
        print(f"== {r['a']} - {r['b']}  {e}  paired by seed; endpoint: {r['source']}; N {[int(n) for n in r['N']]}")
        for k in ("unpaired_seeds", "incomplete", "missing_heldout", "diverged"):
            if r[k]:
                print(f"  WARNING {k}: {r[k]}")
        if e != PRIMARY_ENV:
            print(f"  NOTE the margins (and so the verdict) are in {PRIMARY_ENV} units; prereg 9 does not "
                  "apply them to this task: read the intervals and signs")
        print_contrast(r)
        if r["sensitivity"] is not None:
            print(f"  sensitivity without the pairs containing a resumed run (seeds {r['resumed_seeds']}):")
            print_contrast(r["sensitivity"], "    ")
        else:
            print("  sensitivity: no pair contains a resumed run")
        print(f"  secondary (A - B, paired)      {'A':>9s} {'B':>9s} {'A-B':>9s}  95% CI              p      n")
        for k, x in r["secondary"].items():
            print(f"    {k:26s} {_f(x['mean_a'], 3):>9s} {_f(x['mean_b'], 3):>9s} {_f(x['diff'], 3):>9s}  "
                  f"{_ci(x['ci95'], 3):18s} {_f(x['wilcoxon_p'], 3):>6s} {x['n']:3d}")


def cli(argv=None) -> None:
    p = argparse.ArgumentParser(description="Summarise results and run the pre-registered analyses.")
    p.add_argument("dirs", nargs="*", default=["results/pilot"])
    p.add_argument("--env", help="restrict to one task")
    p.add_argument("--seeds", type=int, nargs="+", help="restrict to these seeds")
    p.add_argument("--end-step", type=int, help="N of the primary window [0.8N, N] (default: each run's steps)")
    p.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap resamples")
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--e0-rule", action="store_true", help="E0 rule (prereg 10.2); writes <dir>/e0_decision.json")
    p.add_argument("--budget-D", metavar="LABEL", help="print D (prereg 10.4) alone on stdout")
    p.add_argument("--pair", nargs=2, metavar=("A", "B"))
    p.add_argument("--h2", nargs=2, metavar=("A", "B"))
    p.add_argument("--h5", nargs="*", metavar="LABEL", help="default: the --pair labels, else every *-lit label")
    p.add_argument("--unpaired", nargs=2, metavar=("A", "B"))
    p.add_argument("--prereg", metavar="PREREG_MD", help="H2-H5 with Holm; writes prereg_results.json next to it")
    p.add_argument("--tag", default="tt", help="recipe tag of E1/E2 for --prereg (tt, or omsw after E0 outcome ii)")
    p.add_argument("--json", metavar="PATH", help="also write the results of the requested modes here")
    a = p.parse_args(argv)
    modes = (a.e0_rule, a.budget_D, a.pair, a.h2, a.h5 is not None, a.unpaired, a.prereg)
    if not any(modes):
        for d in a.dirs:
            main(d)
        return
    runs = [r for d in a.dirs for r in load(d)]
    seeds = set(a.seeds) if a.seeds else None
    kw = dict(n_boot=a.boot, seed=a.boot_seed)
    results: dict = {}
    if a.budget_D:
        r = results["budget_D"] = budget_D(runs, a.budget_D, a.env, seeds)
        print(f"{r['label']} {r['env']} seeds {r['seeds']}: D = {_f(r['D'], 6)}; engagement over 25k-300k = "
              f"{_f(r['engagement'], 4)}; G2 branch: {r['g2_branch']}", file=sys.stderr)
        if r["incomplete"]:
            print(f"WARNING incomplete runs: {r['incomplete']}", file=sys.stderr)
        if not np.isfinite(r["D"]):     # never hand delta0=nan to D=$(...) and the E2 launch
            raise SystemExit("D is not finite (no delta_avg/n_pi rows with n_pi > 0?); nothing printed on stdout")
        print(f"{r['D']:.6g}")
    if a.e0_rule:
        r = results["e0_rule"] = e0_rule(runs, a.end_step or 1_000_000, a.env, seeds)
        print(f"E0 rule at end step {r['end_step']} (each seed's rows at 0.85N-N):")
        print(f"  {'cell':28s} {'lam_lr':>8s} calib  ret    lam     gap    lam600  lamEnd  cost   a b c d")
        for lab, c in r["cells"].items():
            print(f"  {lab:28s} {c['lambda_lr']:8.1g} {_f(c['calib'])} {_f(c['return'], 1):>5s} {_f(c['lambda'], 4)} "
                  f"{_f(c['gap'], 3):>6s} {_f(c['lambda_600k'], 4)}  {_f(c['lambda_end'], 4)}  {_f(c['cost'], 1):>5s}  "
                  + " ".join("x" if c[k] else "." for k in "abcd") + ("" if c["complete"] else "  INCOMPLETE"))
        print(f"  outcome ({r['outcome']}): {r['reason']}; chosen {r['chosen']}, lambda_lr {r['lambda_lr']}, steps {r['steps']}")
        if r["e0_ext"]:
            x = r["e0_ext"]
            move = f"move the PointGoal1 window to --end-step {r['end_step']}" if x["move_window_to_end_step"] else "keep the window"
            print(f"  E0-ext (10.3): seed mean return over 1.6N-2N {[_f(m) for m in x['seed_mean_return']]} -> {move}"
                  + ("" if x["complete"] else "  (E0-ext runs incomplete: not decided yet)"))
        write_json(r, os.path.join(a.dirs[0], "e0_decision.json"))
    if a.pair:
        results["pair"] = pair(runs, *a.pair, a.end_step, a.env, seeds=seeds, **kw)
        print_pair(results["pair"])
    if a.h2:
        results["h2"] = h2(runs, *a.h2, a.env, seeds)
        for e, r in results["h2"].items():
            print(f"== H2 {r['a']} vs {r['b']} {e}: engagement(A, 25k-300k) {_f(r['engagement_a'], 3)}; time-avg lambda "
                  f"A {_f(r['lambda_a'], 4)} B {_f(r['lambda_b'], 4)}, median diff {_f(r['median_diff'], 4)}, "
                  f"one-sided Wilcoxon p={_f(r['p_one_sided'], 4)} (n={r['n']}, unadjusted) -> {r['status']}")
    if a.h5 is not None:
        labels = a.h5 or (a.pair or sorted({r["arm"] for r in runs if r["tag"] == "lit"}))
        results["h5"] = h5(runs, labels, a.end_step, a.env, seeds=seeds, **kw)
        for k, r in results["h5"].items():
            print(f"== H5 {k} ({r['source']}): cost IQM {_f(r['cost_iqm'])} 95% {_ci(r['cost_iqm_ci95'])}; |gap| "
                  f"{_f(abs(r['gap']), 3)} vs {_f(0.25 * r['delta0'], 3)}; calib {_f(r['calib'])} 95% {_ci(r['calib_ci95'])}; "
                  f"p(cost > 25)={_f(r['p_cost_above_25'], 4)} -> "
                  f"{'confirmed' if r['confirmed'] else 'refuted' if r['refuted'] else 'partial'} {r['parts']}")
    if a.unpaired:
        results["unpaired"] = unpaired(runs, *a.unpaired, a.end_step, a.env, seeds=seeds, **kw)
        for e, r in results["unpaired"].items():
            print(f"== {r['a']} vs {r['b']} {e}, unpaired ({r['n_a']} vs {r['n_b']} seeds; {r['source']})")
            for k in ("return", "cost"):
                x = r[k]
                if x:
                    print(f"  {k:6s} IQM A {_f(x['iqm_a'])} {_ci(x['ci_a'])}  B {_f(x['iqm_b'])} {_ci(x['ci_b'])}  "
                          f"A-B {_f(x['diff'])} {_ci(x['ci_diff'])}  P(A better) {_f(x['prob_improvement'])} "
                          f"{_ci(x['ci_prob_improvement'])}")
    if a.prereg:
        r = results["prereg"] = prereg(runs, a.prereg, a.tag, a.env or PRIMARY_ENV, a.end_step, **kw)
        print("Holm family (prereg.md 6):")
        for k, x in r["holm"].items():
            print(f"  {k:28s} p={_f(x['p'], 4)}  Holm={_f(x['p_holm'], 4)}  {'reject' if x['reject_0.05'] else ''}")
        for h in ("H2", "H3", "H4", "H5"):
            extra = f" (verdict {r[h]['verdict']})" if "verdict" in r[h] else ""
            print(f"{h}: {r[h]['status']}{extra}")
        write_json(r, os.path.join(os.path.dirname(os.path.abspath(a.prereg)), "prereg_results.json"))
    if a.json:
        write_json(results, a.json)


def main(dirpath: str) -> None:
    by: dict[tuple[str, str], list[dict]] = {}
    for r in load(dirpath):
        by.setdefault((r["env"], r["arm"]), []).append(r)
    print("sd uses ddof=1; the IQM CI is printed only with >= 5 seeds; all runs of a cell are read at "
          "their common step; Qc/MC and Q/MC are ratios of means (trajectory-level when logged).")
    print(f"{'env':20s} {'arm':22s} n  {'step':>7s}  return mean+-sd   IQM [95% CI]       "
          f"cost mean+-sd   viol  engage  lambda   Qc/MC   Q/MC")
    for (env, arm), rs in sorted(by.items()):
        last = at_common_step(rs)
        g = lambda k: [l.get(k, NAN) for l in last]
        ret, cost = g("eval_return"), g("eval_cost")
        lo, hi = boot_ci(ret)
        ci = f"[{lo:6.2f},{hi:6.2f}]" if np.isfinite(lo) else "[ n<5 seeds  ]"
        eng = [np.nanmean([row.get("engagement", NAN) for row in r["rows"]]) for r in rs]
        traj = np.isfinite(g("qc_traj")).any() if "qc_traj" in last[0] else False
        qc_mc = ratio(g("qc_traj"), g("mc_cost_traj")) if traj else ratio(g("qc0"), g("mc_cost0"))
        q_mc = ratio(g("qbar_traj"), g("mc_ret_traj")) if traj else ratio(g("qbar0"), g("mc_ret0"))
        print(f"{env:20s} {arm:22s} {len(rs)}  {int(min(g('step'))):>7d}  "
              f"{np.mean(ret):6.2f}+-{sd(ret):5.2f}  {iqm(ret):6.2f} {ci}  "
              f"{np.mean(cost):6.2f}+-{sd(cost):5.2f}  {np.mean(g('eval_violation')):.2f}  "
              f"{np.mean(eng):.3f}   {np.mean(g('lambda')):7.3g}  {qc_mc:6.2f}  {q_mc:6.2f}")
    try:
        plot(by, dirpath)
    except ImportError:
        print("(matplotlib not installed; no plots)")


# one panel per tuple; keys in a panel are drawn solid, dashed, dotted
PANELS = [("eval_return",), ("eval_cost",), ("lambda",), ("engagement",)]
NEW_PANELS = [("lambda_avg", "lambda_shadow"), ("engagement", "would_engage"),
              ("qc_avg", "qc_traj", "mc_cost_traj"), ("delta_avg", "delta_risk_avg")]
STYLES = ("-", "--", ":")


def plot(by, dirpath: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new = any(np.isfinite(row.get("lambda_avg", NAN)) for rs in by.values() for r in rs for row in r["rows"])
    panels = PANELS + (NEW_PANELS if new else [])
    for env in sorted({e for e, _ in by}):
        fig, axes = plt.subplots(len(panels) // 4, 4, figsize=(16.8, 3.2 * (len(panels) // 4)), squeeze=False)
        axes = axes.ravel()
        arms = [(arm, rs) for (e, arm), rs in sorted(by.items()) if e == env]
        for i, (arm, rs) in enumerate(arms):
            common = sorted(set.intersection(*[{row["step"] for row in r["rows"]} for r in rs]))
            steps = np.array(common)
            for ax, keys in zip(axes, panels):
                for k, ls in zip(keys, STYLES):
                    m = np.array([[next(row.get(k, NAN) for row in r["rows"] if row["step"] == st)
                                   for st in common] for r in rs])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        mu, s = np.nanmean(m, 0), np.nanstd(m, 0)
                    ax.plot(steps, mu, ls, color=f"C{i}", label=arm if ls == "-" else None)
                    ax.fill_between(steps, mu - s, mu + s, color=f"C{i}", alpha=0.2 if ls == "-" else 0.08)
                ax.set_title(" / ".join(f"{k} ({ls})" if len(keys) > 1 else k for k, ls in zip(keys, STYLES)),
                             fontsize=8 if len(keys) > 1 else 10)
                ax.set_xlabel("env steps")
        axes[0].legend(fontsize=7)
        fig.suptitle(env)
        fig.tight_layout()
        out = os.path.join(dirpath, f"curves_{env}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    cli()
