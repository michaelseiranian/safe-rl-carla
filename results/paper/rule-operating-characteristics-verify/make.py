"""EXPLORATORY / DESCRIPTIVE verification (post hoc, not confirmatory) of results/paper/verdict_oc.

Independent re-derivation of the operating-characteristics appendix of the prereg.md section 7
verdict rule. This script does NOT import lagu/ for any number it reports: it reads the raw run files
(<stem>.json, <stem>.csv, <stem>.heldout.json) itself and re-implements the section 7 rule, the
section 5/11 collapse definition, the section 6 paired bootstrap and the Fisher exact test from the
prereg text. Only at the very end (check V5) is lagu.analyze.verdict imported, read-only and with
bytecode writing disabled, to confirm that the pre-registered code and this re-implementation give
the same label on a grid of random and boundary inputs.

Run from the repository root (one process, about 1 minute):

    .venv/bin/python results/paper/rule-operating-characteristics-verify/make.py

Writes next to this file: tables.txt (everything printed), results.json, fig_verify_agreement.{pdf,png}.
Reads results/gate, results/gate_literal and results/paper/verdict_oc/results.json (to compare).

V0 (DESCRIPTIVE). Data: E1 = lagu-tt vs lagu_nogate-tt, SafetyPointGoal1-v0, seeds 100-114;
   E1b = lagu-lit vs lagu_nogate-lit, seeds 100-109; held-out pooled return/cost; collapse =
   held-out return < 5 or diverged (prereg.md 11). No run excluded; uniqueness of (arm, seed) checked.
V1 (EXPLORATORY). The simulation of verdict_oc Part A, re-implemented with a different but
   distributionally equivalent sampler: multinomial bootstrap weights instead of index resampling,
   binomial collapse counts instead of summed uniforms, and an independent random stream per row
   (derived from numpy.random.default_rng(0)) instead of common random numbers. Same scenarios,
   2,000 studies x 2,000 resamples. Agreement is judged by z = (mine - theirs) / SE of a difference
   of two independent proportions.
V2 (EXPLORATORY). Part B, the E1 H3 label at collapse thresholds 2..12, from the raw held-out returns,
   with the pre-registered bootstrap call (10,000 resamples, default_rng(0), pairs in seed order).
V3 (DESCRIPTIVE). Part C, the E1b label at bootstrap seeds 0..1999 (10,000 resamples each); plus how
   often the section 7 Equivalent condition (both 90% intervals inside the margins) holds at all.
V4 (EXPLORATORY). Checks of specific statements in the verdict_oc summary/caveats.
V5. lagu.analyze.verdict vs this re-implementation on random and boundary inputs.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ENV = "SafetyPointGoal1-v0"
LABELS = ("Helps", "Hurts", "Shift", "Equivalent", "Inconclusive")
MR, MC = 2.5, 5.0                     # prereg.md 7 margins (return, cost)
COLLAPSE = 5.0                        # prereg.md 5
DIVERGE = ("eval_return", "eval_cost", "qc_traj", "qbar_traj", "q_loss", "c_loss", "lambda")  # prereg.md 11
N_STUDIES, N_BOOT_SIM, CHUNK = 2000, 2000, 100
N_BOOT = 10_000
LINES: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


# ------------------------------------------------------------------------------------------ data

def fnum(v) -> float:
    try:
        return float(v) if v not in ("", None) else float("nan")
    except ValueError:
        return float("nan")


def load_run(d: Path, label: str, seed: int) -> dict:
    stem = d / f"{label}_{ENV}_s{seed}"
    meta = json.loads(Path(str(stem) + ".json").read_text())
    a = meta["args"]
    lab = a["arm"] + ("-" + a["tag"] if a.get("tag") else "")
    assert (lab, a["seed"], a["env"]) == (label, seed, ENV), stem
    with open(str(stem) + ".csv") as f:
        rows = [{k: fnum(v) for k, v in r.items() if k is not None} for r in csv.DictReader(f)]
    rows = [r for r in rows if np.isfinite(r.get("step", np.nan))]
    N = float(max(int(a.get("steps") or 0), rows[-1]["step"]))
    held = json.loads(Path(str(stem) + ".heldout.json").read_text())
    assert float(held["end_step"]) == N, f"{stem}: held-out end_step {held['end_step']} != N {N}"
    ret, cost = fnum(held["pooled"]["eval_return"]), fnum(held["pooled"]["eval_cost"])
    lo = int(a.get("start_steps") or 0)
    div = any(k in r and not np.isfinite(r[k]) for r in rows if lo < r["step"] <= N for k in DIVERGE)
    div = div or not (np.isfinite(ret) and np.isfinite(cost))
    return {"seed": seed, "return": ret, "cost": cost, "diverged": div, "N": N,
            "resumed": any(r.get("resumed") == 1 for r in rows), "last_step": rows[-1]["step"]}


def check_unique(d: Path, labels, seeds) -> dict:
    """Count run metadata files per (label, seed) in the directory (prereg.md 11: no exclusions)."""
    cnt: dict = {}
    for p in d.glob("*.json"):
        if p.name.endswith(".heldout.json"):
            continue
        try:
            a = json.loads(p.read_text())["args"]
        except (KeyError, TypeError, json.JSONDecodeError):   # launch.json is a list, not run metadata
            continue
        lab = a["arm"] + ("-" + a["tag"] if a.get("tag") else "")
        if lab in labels and a["env"] == ENV and a["seed"] in seeds:
            cnt[(lab, a["seed"])] = cnt.get((lab, a["seed"]), 0) + 1
    return cnt


def pairs(dirname: str, la: str, lb: str, seeds) -> list:
    d = ROOT / dirname
    cnt = check_unique(d, (la, lb), set(seeds))
    assert all(cnt.get((l, s), 0) == 1 for l in (la, lb) for s in seeds), f"{dirname}: missing/duplicate runs {cnt}"
    return [(s, load_run(d, la, s), load_run(d, lb, s)) for s in seeds]


# ------------------------------------------------------------------------------ statistics (own)

def verdict(r95, c95, r90, c90, ca: int, cb: int) -> str:
    """prereg.md section 7, transcribed from the table (first matching label)."""
    rl, rh = r95
    cl, ch = c95
    if (ch < 0 and rl > -MR) or (rl > 0 and ch < MC):
        return "Helps"
    if (cl > 0 and rh < MR) or (rh < 0 and cl > -MC) or (ca - cb >= 3):
        return "Hurts"
    if (rl > 0 and cl > 0) or (rh < 0 and ch < 0):
        return "Shift"
    if -MR < r90[0] and r90[1] < MR and -MC < c90[0] and c90[1] < MC:
        return "Equivalent"
    return "Inconclusive"


def hurts_by_interval(r95, c95) -> bool:
    return bool((c95[0] > 0 and r95[1] < MR) or (r95[1] < 0 and c95[0] > -MC))


def prereg_bootstrap(d: np.ndarray, seed: int, n_boot: int = N_BOOT):
    """prereg.md 6: resample seed pairs with replacement, mean paired difference, percentile CIs."""
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(axis=1)
    q = np.percentile(m, [2.5, 97.5, 5, 95], axis=0)          # (4, 2)
    return (q[0, 0], q[1, 0]), (q[0, 1], q[1, 1]), (q[2, 0], q[3, 0]), (q[2, 1], q[3, 1])


def fisher_two_sided(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided Fisher exact p (sum of table probabilities <= the observed one), integer arithmetic."""
    k, N = x1 + x2, n1 + n2
    w = {x: math.comb(n1, x) * math.comb(n2, k - x) for x in range(max(0, k - n2), min(n1, k) + 1)}
    return sum(v for v in w.values() if v <= w[x1]) / math.comb(N, k)


def p_diff_at_least(n: int, pA: float, pB: float, k: int) -> float:
    """P(X_A - X_B >= k), X ~ Bin(n, p) independent, by convolution of the two pmfs."""
    pmf = lambda p: np.array([math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)])
    a, b = pmf(pA), pmf(pB)
    diff = np.convolve(a, b[::-1])                     # index j <-> X_A - X_B = j - n
    return float(diff[n + k:].sum())


# ------------------------------------------------------------------------------- V1 simulation

SCENARIOS = [  # name (as in verdict_oc), shift (return, cost), pA, pB, exchangeable
    ("null, p_collapse 0.2 both", (0, 0), 0.2, 0.2, False),
    ("null, p_collapse 0.4 both", (0, 0), 0.4, 0.4, False),
    ("null, exchangeable pairs", (0, 0), None, None, True),
    ("return -5", (-5, 0), 0.2, 0.2, False),
    ("return -10", (-10, 0), 0.2, 0.2, False),
    ("cost -5", (0, -5), 0.2, 0.2, False),
    ("cost -10", (0, -10), 0.2, 0.2, False),
    ("collapse +0.2 (0.4 vs 0.2)", (0, 0), 0.4, 0.2, False),
    ("collapse +0.4 (0.6 vs 0.2)", (0, 0), 0.6, 0.2, False),
]


def simulate(rng, d, flags, n, shift, pA, pB, exch) -> dict:
    dc = d - d.mean(axis=0)
    lab_count = dict.fromkeys(LABELS, 0)
    coll_only = clause = r90_in = c90_in = 0
    min_hw = np.array([np.inf, np.inf])
    w = np.full(n, 1.0 / n)
    for start in range(0, N_STUDIES, CHUNK):
        m = min(CHUNK, N_STUDIES - start)
        idx = rng.integers(0, len(d), (m, n))
        if exch:
            sgn = np.where(rng.random((m, n)) < 0.5, -1.0, 1.0)
            study = d[idx] * sgn[..., None]
            fa, fb = flags[idx, 0], flags[idx, 1]
            ca = np.where(sgn < 0, fb, fa).sum(1)
            cb = np.where(sgn < 0, fa, fb).sum(1)
        else:
            study = dc[idx] + np.asarray(shift, float)
            ca, cb = rng.binomial(n, pA, m), rng.binomial(n, pB, m)
        counts = rng.multinomial(n, w, size=(m, N_BOOT_SIM)).astype(float)      # (m, B, n)
        means = np.einsum("mbn,mnk->mbk", counts, study) / n                    # (m, B, 2)
        q = np.percentile(means, [2.5, 97.5, 5, 95], axis=1)                    # (4, m, 2)
        for j in range(m):
            r95, c95 = (q[0, j, 0], q[1, j, 0]), (q[0, j, 1], q[1, j, 1])
            r90, c90 = (q[2, j, 0], q[3, j, 0]), (q[2, j, 1], q[3, j, 1])
            lab = verdict(r95, c95, r90, c90, int(ca[j]), int(cb[j]))
            lab_count[lab] += 1
            coll_only += lab == "Hurts" and not hurts_by_interval(r95, c95)
            clause += int(ca[j] - cb[j] >= 3)
            r90_in += -MR < r90[0] and r90[1] < MR
            c90_in += -MC < c90[0] and c90[1] < MC
            min_hw = np.minimum(min_hw, [(r90[1] - r90[0]) / 2, (c90[1] - c90[0]) / 2])
    return {"p": {k: v / N_STUDIES for k, v in lab_count.items()}, "hurts_collapse_only": coll_only / N_STUDIES,
            "collapse_clause_fires": clause / N_STUDIES, "frac_return90_inside": r90_in / N_STUDIES,
            "frac_cost90_inside": c90_in / N_STUDIES, "min_halfwidth90": min_hw.tolist()}


def z_diff(a: float, b: float, n: int = N_STUDIES) -> float:
    p = (a + b) / 2
    se = math.sqrt(2 * p * (1 - p) / n)
    return 0.0 if se == 0 else (a - b) / se


# ---------------------------------------------------------------------------------------- main

def main() -> None:
    res: dict = {}
    theirs = json.loads((ROOT / "results/paper/verdict_oc/results.json").read_text())

    # ---- V0 data
    E1 = pairs("results/gate", "lagu-tt", "lagu_nogate-tt", range(100, 115))
    E1b = pairs("results/gate_literal", "lagu-lit", "lagu_nogate-lit", range(100, 110))
    out("=" * 100)
    out("V0 (DESCRIPTIVE). Raw held-out endpoints read directly from <stem>.heldout.json (pooled)")
    out("=" * 100)
    for name, E in (("E1 lagu-tt - lagu_nogate-tt (N = 2M)", E1), ("E1b lagu-lit - lagu_nogate-lit (N = 1M)", E1b)):
        out(f"{name}: {len(E)} pairs, every run present once with a held-out file at end_step = N")
        out(f"  {'seed':>5}{'ret A':>9}{'ret B':>9}{'cost A':>9}{'cost B':>9}{'dRet':>9}{'dCost':>9}"
            f"{'collA':>7}{'collB':>7}{'divA':>6}{'divB':>6}{'resumed':>9}")
        for s, a, b in E:
            out(f"  {s:>5}{a['return']:>9.2f}{b['return']:>9.2f}{a['cost']:>9.2f}{b['cost']:>9.2f}"
                f"{a['return'] - b['return']:>9.2f}{a['cost'] - b['cost']:>9.2f}"
                f"{int(a['diverged'] or a['return'] < COLLAPSE):>7}{int(b['diverged'] or b['return'] < COLLAPSE):>7}"
                f"{int(a['diverged']):>6}{int(b['diverged']):>6}{int(a['resumed'] or b['resumed']):>9}")
    d = np.array([[a["return"] - b["return"], a["cost"] - b["cost"]] for _, a, b in E1])
    flags = np.array([[a["diverged"] or a["return"] < COLLAPSE, b["diverged"] or b["return"] < COLLAPSE]
                      for _, a, b in E1], float)
    sd = d.std(axis=0, ddof=1)
    corr = float(np.corrcoef(d.T)[0, 1])
    out()
    out(f"E1 noise model: mean dReturn {d[:, 0].mean():.2f} (SD {sd[0]:.2f}), mean dCost {d[:, 1].mean():.2f} "
        f"(SD {sd[1]:.2f}), corr {corr:.2f}; collapsed A {int(flags[:, 0].sum())}/15, B {int(flags[:, 1].sum())}/15")
    nm = theirs["part_a_simulation"]["noise_model"]
    same = (np.allclose(nm["d_return"], d[:, 0], atol=1e-9) and np.allclose(nm["d_cost"], d[:, 1], atol=1e-9)
            and nm["collapsed_a"] == flags[:, 0].tolist() and nm["collapsed_b"] == flags[:, 1].tolist())
    out(f"  verdict_oc noise model (results.json) identical to these raw values: {same}")
    res["V0"] = {"d": d.tolist(), "flags": flags.tolist(), "sd": sd.tolist(), "corr": corr, "noise_model_match": same}

    # ---- V1 simulation
    out()
    out("=" * 100)
    out("V1 (EXPLORATORY). Independent re-simulation of verdict_oc Part A (2,000 studies x 2,000 resamples)")
    out("=" * 100)
    out("  Sampler differs from verdict_oc on purpose (multinomial bootstrap weights, binomial collapse counts,")
    out("  independent stream per row from default_rng(0)); agreement => z within MC error.")
    master = np.random.default_rng(0)
    row_seeds = master.integers(0, 2 ** 32, size=len(SCENARIOS) * 2)
    trow = {(r["n"], r["scenario"]): r for r in theirs["part_a_simulation"]["rows"]}
    sim_rows, zs = [], []
    k = 0
    for n in (15, 20):
        out()
        out(f"EXPLORATORY. Table V1-{n}. Label probability (%), n = {n}: mine / verdict_oc (z of the difference).")
        out(f"  {'scenario':<29}{'Helps':>17}{'Hurts':>17}{'coll.-only':>17}{'Shift':>15}{'Equiv':>13}{'Incon':>17}"
            f"{'clause>=3':>17}{'exact>=3':>9}")
        for name, shift, pA, pB, exch in SCENARIOS:
            r = simulate(np.random.default_rng(int(row_seeds[k])), d, flags, n, shift, pA, pB, exch)
            k += 1
            t = trow[(n, name)]
            cells_ = []
            for key, mine, th in ([(lab, r["p"][lab], t["p"][lab]) for lab in LABELS[:2]]
                                  + [("coll_only", r["hurts_collapse_only"], t["hurts_collapse_only"])]
                                  + [(lab, r["p"][lab], t["p"][lab]) for lab in LABELS[2:]]
                                  + [("clause", r["collapse_clause_fires"], t["collapse_clause_fires"])]):
                z = z_diff(mine, th)
                zs.append({"n": n, "scenario": name, "quantity": key, "mine": mine, "theirs": th, "z": z})
                cells_.append(f"{100 * mine:5.1f}/{100 * th:5.1f}({z:+4.1f})")
            ex = p_diff_at_least(n, pA, pB, 3) if not exch else float("nan")
            exs = "" if exch else f"{100 * ex:.1f}"
            out(f"  {name:<29}" + "".join(f"{c:>17}" for c in cells_[:3]) + f"{cells_[3]:>17}"
                + f"{cells_[4]:>17}{cells_[5]:>17}{cells_[6]:>17}{exs:>9}")
            sim_rows.append({"n": n, "scenario": name, **r, "exact_clause": None if exch else ex})
    zarr = np.array([x["z"] for x in zs])
    out()
    out(f"  {len(zarr)} compared proportions: max |z| = {np.abs(zarr).max():.2f}; |z| > 1.96 in {int(np.sum(np.abs(zarr) > 1.96))}"
        f" (expected about {0.05 * len(zarr):.1f} by chance); |z| > 3 in {int(np.sum(np.abs(zarr) > 3))}.")
    fr = np.array([x["frac_return90_inside"] for x in sim_rows])
    fc = np.array([x["frac_cost90_inside"] for x in sim_rows])
    mh = np.array([x["min_halfwidth90"] for x in sim_rows])
    out(f"  Equivalent preconditions over all {len(sim_rows) * N_STUDIES:,} simulated studies: return 90% CI inside "
        f"+-2.5 in {100 * fr.mean():.2f}% (max per row {100 * fr.max():.1f}%), cost 90% CI inside +-5 in {100 * fc.mean():.3f}%;")
    out(f"  smallest 90% half-widths seen: return {mh[:, 0].min():.2f}, cost {mh[:, 1].min():.2f}.")
    res["V1"] = {"rows": sim_rows, "comparisons": zs}

    # exact clause table vs theirs
    out()
    out("EXPLORATORY. Table V1-exact. P(X_A - X_B >= 3) by convolution (%), mine vs verdict_oc Table A3.")
    exact_rows = []
    for t in theirs["part_a_simulation"]["clause_exact"]:
        m15, m20 = p_diff_at_least(15, t["pA"], t["pB"], 3), p_diff_at_least(20, t["pA"], t["pB"], 3)
        ok = abs(m15 - t["n15"]) < 1e-12 and abs(m20 - t["n20"]) < 1e-12
        out(f"  pA {t['pA']:.1f} pB {t['pB']:.1f}: n=15 {100 * m15:6.2f} vs {100 * t['n15']:6.2f};  n=20 {100 * m20:6.2f} vs "
            f"{100 * t['n20']:6.2f}   identical: {ok}")
        exact_rows.append({**t, "mine_n15": m15, "mine_n20": m20, "identical": ok})
    res["V1_exact"] = exact_rows

    # ---- V2 threshold
    out()
    out("=" * 100)
    out("V2 (EXPLORATORY). E1 H3 label vs collapse threshold, from raw held-out returns (prereg bootstrap, seed 0)")
    out("=" * 100)
    r95, c95, r90, c90 = prereg_bootstrap(d, 0)
    out(f"  intervals: dReturn 95% [{r95[0]:.2f}, {r95[1]:.2f}] 90% [{r90[0]:.2f}, {r90[1]:.2f}]; "
        f"dCost 95% [{c95[0]:.2f}, {c95[1]:.2f}] 90% [{c90[0]:.2f}, {c90[1]:.2f}]")
    out(f"  label with the collapse clause switched off: {verdict(r95, c95, r90, c90, 0, 0)}")
    tb = {x["threshold"]: x for x in theirs["part_b_threshold"]["rows"]}
    out(f"  {'thr':>4}{'collA':>7}{'collB':>7}{'Fisher p':>10}  {'label':<10}{'verdict_oc row':>28}  match")
    v2 = []
    ra = np.array([a["return"] for _, a, _ in E1])
    rb = np.array([b["return"] for _, _, b in E1])
    da = np.array([a["diverged"] for _, a, _ in E1])
    db = np.array([b["diverged"] for _, _, b in E1])
    for thr in range(2, 13):
        ca, cb = int(np.sum(da | (ra < thr))), int(np.sum(db | (rb < thr)))
        fp = fisher_two_sided(ca, 15, cb, 15)
        lab = verdict(r95, c95, r90, c90, ca, cb)
        t = tb[thr]
        ok = (ca, cb, lab) == (t["collapses_a"], t["collapses_b"], t["label"]) and abs(fp - t["fisher_p"]) < 1e-12 \
            and np.allclose(t["return_ci95"], r95, atol=1e-9) and np.allclose(t["cost_ci95"], c95, atol=1e-9)
        out(f"  {thr:>4}{ca:>7}{cb:>7}{fp:>10.3f}  {lab:<10}{t['collapses_a']:>12}/{t['collapses_b']:<3}{t['fisher_p']:.3f} "
            f"{t['label']:<6}  {ok}")
        v2.append({"threshold": thr, "ca": ca, "cb": cb, "fisher_p": fp, "label": lab, "match": ok})
    # nearest returns to the threshold grid (how fragile is each count)
    allr = np.sort(np.concatenate([ra, rb]))
    out(f"  held-out returns within 0.25 of an integer threshold 2..12: "
        f"{[round(float(x), 3) for x in allr if np.min(np.abs(x - np.arange(2, 13))) < 0.25]}")
    res["V2"] = {"r95": r95, "c95": c95, "r90": r90, "c90": c90, "rows": v2}

    # ---- V3 E1b seeds
    out()
    out("=" * 100)
    out("V3 (DESCRIPTIVE). E1b label vs bootstrap seed 0..1999 (10,000 resamples each). Not a re-analysis.")
    out("=" * 100)
    db_ = np.array([[a["return"] - b["return"], a["cost"] - b["cost"]] for _, a, b in E1b])
    cAb = sum(a["diverged"] or a["return"] < COLLAPSE for _, a, _ in E1b)
    cBb = sum(b["diverged"] or b["return"] < COLLAPSE for _, _, b in E1b)
    labs, rl, ch, equiv_cond = [], [], [], []
    for seed in range(2000):
        R95, C95, R90, C90 = prereg_bootstrap(db_, seed)
        labs.append(verdict(R95, C95, R90, C90, cAb, cBb))
        rl.append(R95[0])
        ch.append(C95[1])
        equiv_cond.append(-MR < R90[0] and R90[1] < MR and -MC < C90[0] and C90[1] < MC)
    labs, rl, ch, equiv_cond = np.array(labs), np.array(rl), np.array(ch), np.array(equiv_cond)
    tc = theirs["part_c_e1b_seed"]
    same_labels = labs.tolist() == tc["labels"]
    same_lb = np.allclose(rl, tc["return_lb"], atol=1e-12, rtol=0)
    out(f"  mean dReturn {db_[:, 0].mean():.4f}, mean dCost {db_[:, 1].mean():.4f}; collapses {cAb} vs {cBb}")
    out(f"  seed 0: {labs[0]}, dReturn 95% LB {rl[0]:.4f}, dCost 95% UB {ch[0]:.3f}")
    for lab in LABELS:
        out(f"  {lab:<13}{int(np.sum(labs == lab)):>6} of 2000 ({100 * np.mean(labs == lab):.2f}%)")
    q = np.percentile(rl, [0, 2.5, 50, 97.5, 100])
    out(f"  dReturn 95% LB: min {q[0]:.4f}, 2.5% {q[1]:.4f}, median {q[2]:.4f}, 97.5% {q[3]:.4f}, max {q[4]:.4f}, "
        f"SD {rl.std(ddof=1):.4f}, > 0 in {100 * np.mean(rl > 0):.2f}%")
    out(f"  dCost 95% UB range {ch.min():.3f} to {ch.max():.3f}")
    out(f"  section 7 Equivalent condition (both 90% CIs inside the margins) holds in {100 * equiv_cond.mean():.2f}% of seeds,")
    out("  i.e. also at every seed labelled Helps: the seed only decides which of the two labels takes precedence.")
    out(f"  per-seed labels identical to verdict_oc results.json: {same_labels}; per-seed return LB identical: {same_lb}")
    res["V3"] = {"fractions": {lab: float(np.mean(labs == lab)) for lab in LABELS}, "seed0": labs[0],
                 "seed0_lb": float(rl[0]), "lb_sd": float(rl.std(ddof=1)), "equiv_cond_frac": float(equiv_cond.mean()),
                 "labels_identical": same_labels, "lb_identical": bool(same_lb)}

    # ---- V4 statement checks
    out()
    out("=" * 100)
    out("V4 (EXPLORATORY). Checks of specific statements in the verdict_oc report")
    out("=" * 100)
    i106 = [s for s, _, _ in E1].index(106)
    keep = np.arange(15) != i106
    out(f"  caveat 1: seed 106 dCost = {d[i106, 1]:+.1f}; dCost SD with all 15 pairs {sd[1]:.2f}, "
        f"SD of the other 14 (check of the claim only, not an analysis row) {d[keep, 1].std(ddof=1):.2f}")
    for p0 in (0.2, 0.4):
        out(f"  caveat 5: exact P(X_A - X_B >= 6 | n=15, pA=pB={p0}) = {100 * p_diff_at_least(15, p0, p0, 6):.2f}%  "
            f"(>= 3: {100 * p_diff_at_least(15, p0, p0, 3):.2f}%)")
    gaps = [(x["n"], x["scenario"], abs(x["collapse_clause_fires"] - x["exact_clause"]))
            for x in [{"n": r["n"], "scenario": r["scenario"], "collapse_clause_fires": r["collapse_clause_fires"],
                       "exact_clause": r["collapse_clause_exact"]} for r in theirs["part_a_simulation"]["rows"]]
            if x["exact_clause"] is not None]
    g = max(gaps, key=lambda x: x[2])
    out(f"  summary: largest |simulated - exact| clause gap in verdict_oc = {100 * g[2]:.2f} points at n={g[0]}, {g[1]}")
    out(f"  summary: max Monte Carlo SE of a proportion with 2,000 studies = {100 * math.sqrt(0.25 / N_STUDIES):.2f} points")
    out("  docstring of verdict_oc/make.py says 'about 2 minutes'; summary says 'about 35 s' (timed separately).")
    # figure colours of verdict_oc (copied from its make.py): WCAG contrast of each fill on its background
    def rel_lum(hx: str) -> float:
        c = np.array([int(hx[i:i + 2], 16) for i in (1, 3, 5)]) / 255
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
    fills = {"Helps": "#2a78d6", "Hurts": "#eb6834", "Shift": "#1baf7a", "Equivalent": "#eda100",
             "Inconclusive": "#e87ba4"}
    cr = {k: (rel_lum("#fcfcfb") + 0.05) / (rel_lum(v) + 0.05) for k, v in fills.items()}
    out("  verdict_oc label fills, WCAG contrast on its background #fcfcfb: "
        + ", ".join(f"{k} {v:.2f}" for k, v in cr.items()) + f"; below 3:1: {[k for k, v in cr.items() if v < 3]}")
    small = sum(1 for r in theirs["part_a_simulation"]["rows"] for lab in LABELS if 0 < 100 * r["p"][lab] < 7)
    out(f"  fig_label_probabilities: {small} non-empty bar segments are < 7% and so carry no direct number label")
    res["V4"] = {"sd_cost_without_106": float(d[keep, 1].std(ddof=1)), "contrast": cr, "unlabelled_segments": small}

    # ---- V5 cross-check of the pre-registered verdict() against this transcription
    sys.path.insert(0, str(ROOT))
    import lagu.analyze as A  # noqa: E402  (read-only; bytecode writing disabled above)
    rng = np.random.default_rng(0)
    grid = np.array([-5.0001, -5.0, -4.9999, -2.5001, -2.5, -2.4999, -1e-9, 0.0, 1e-9, 2.4999, 2.5, 2.5001,
                     4.9999, 5.0, 5.0001])
    mism = 0
    tot = 0
    for _ in range(200_000):
        v = rng.choice(grid, 8) if rng.random() < 0.5 else rng.normal(0, 6, 8)
        r95, c95 = tuple(sorted(v[:2])), tuple(sorted(v[2:4]))
        r90, c90 = tuple(sorted(v[4:6])), tuple(sorted(v[6:8]))
        ca, cb = int(rng.integers(0, 16)), int(rng.integers(0, 16))
        tot += 1
        mism += verdict(r95, c95, r90, c90, ca, cb) != A.verdict(r95, c95, r90, c90, ca, cb)
    out()
    out(f"V5. lagu.analyze.verdict vs the prereg-table transcription here: {mism} disagreements in {tot:,} inputs "
        "(half on the margin/zero boundaries).")
    res["V5"] = {"n": tot, "disagreements": int(mism)}

    fig(zs, theirs, sim_rows)
    (OUT / "tables.txt").write_text("\n".join(LINES) + "\n")
    with open(OUT / "results.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    out(f"wrote {OUT / 'tables.txt'}, results.json, fig_verify_agreement.{{pdf,png}}")


# --------------------------------------------------------------------------------------- figure

def fig(zs, theirs, sim_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#111111", "#4d4d4d", "#dddddd", "#ffffff"
    # Okabe-Ito (colour-blind safe), plus a distinct marker per quantity
    style = {"Helps": ("#0072B2", "o"), "Hurts": ("#D55E00", "s"), "coll_only": ("#CC79A7", "D"),
             "Shift": ("#009E73", "^"), "Equivalent": ("#E69F00", "v"), "Inconclusive": ("#56B4E9", "P"),
             "clause": ("#000000", "x")}
    names = {"coll_only": "Hurts via collapse clause only", "clause": "collapse clause fires (>= 3 more)"}
    plt.rcParams.update({"font.size": 9, "text.color": ink, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.edgecolor": "#999999", "figure.facecolor": surf,
                         "axes.facecolor": surf, "savefig.facecolor": surf})
    f, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    xx = np.linspace(0, 100, 201)
    band = 100 * 2 * np.sqrt(2 * (xx / 100) * (1 - xx / 100) / N_STUDIES)
    a1.fill_between(xx, xx - band, xx + band, color="#bbbbbb", alpha=0.5, lw=0, label="\u00b12 SE of a difference")
    a1.plot([0, 100], [0, 100], color=ink2, lw=0.8)
    for key, (c, mk) in style.items():
        pts = [(100 * z["theirs"], 100 * z["mine"]) for z in zs if z["quantity"] == key]
        a1.scatter(*zip(*pts), s=22, color=c, marker=mk, label=names.get(key, key), zorder=3,
                   linewidths=1.2 if mk == "x" else 0.4, edgecolors=None if mk == "x" else surf)
    a1.set_xlabel("verdict_oc estimate (% of 2,000 simulated studies)")
    a1.set_ylabel("independent re-simulation (% of 2,000 studies)")
    a1.set_title("(a) EXPLORATORY: Part A probabilities, both n, all scenarios", fontsize=9.5)
    a1.set_xlim(-2, 100)
    a1.set_ylim(-2, 100)
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")
    z = np.array([x["z"] for x in zs])
    keys = list(style)
    for i, key in enumerate(keys):
        zz = np.array([x["z"] for x in zs if x["quantity"] == key])
        a2.scatter(zz, np.full(len(zz), i) + np.linspace(-0.25, 0.25, len(zz)), s=18, color=style[key][0],
                   marker=style[key][1], linewidths=1.2 if style[key][1] == "x" else 0.4,
                   edgecolors=None if style[key][1] == "x" else surf)
    for v in (-1.96, 1.96):
        a2.axvline(v, color=ink2, lw=0.8, ls="--")
    a2.axvline(0, color=ink2, lw=0.8)
    a2.set_yticks(range(len(keys)), [names.get(k, k) for k in keys])
    a2.invert_yaxis()
    a2.set_xlabel("z = (re-simulation - verdict_oc) / SE of difference\n(dashed lines: z = \u00b11.96)")
    a2.set_title(f"(b) EXPLORATORY: agreement, max |z| = {np.abs(z).max():.2f} over {len(z)} proportions", fontsize=9.5)
    lim = max(3.0, float(np.abs(z).max()) + 0.3)
    a2.set_xlim(-lim, lim)
    for ax in (a1, a2):
        ax.grid(True, color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    f.tight_layout()
    for ext in ("pdf", "png"):
        f.savefig(OUT / f"fig_verify_agreement.{ext}", dpi=200, bbox_inches="tight")
    plt.close(f)


if __name__ == "__main__":
    main()
