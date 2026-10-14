"""Statistics and decision rules of the method study (prereg_method.md sections 6, 8, 11-16).

Numpy only (no scipy). Everything here is a pure function of per-run numbers, so safe2/analyze.py
only has to read the run files and call these. A "task list" is a list with one 1-D array per task
(SafetyPointGoal1, SafetyCarGoal1, SafetyPointCircle1), holding one value per run (seed).

Primary endpoint (section 11). For each arm and task the penalised score is applied to the ARM'S
held-out means: S_t = [mean R - 1.0 * max(0, mean C - 25)] / R_ref(t). The primary contrast is
Delta_s = mean over tasks of (S_t of the candidate - S_t of B*). The hinge is applied to arm means,
not per run, so a run-to-run spread of cost around 25 is not charged as a violation (a per-run hinge
charges an arm sitting at a mean cost of 25 about 0.4 sd_C, which rewarded under-spending; review
item 1). The test is a one-sided stratified permutation test: within each task the (R, C) pairs of
the two arms are re-split at random and Delta_s is recomputed. Its nominal level ALPHA_TEST is set
below the family level ALPHA so that the size stays <= ALPHA under the unequal-shape nulls of
Appendix G.1 (review item 12). Intervals use the delta method: a Welch-type stratified t on the
linearised per-run scores [R - w (C - 25)] / R_ref, w = 1[arm mean C > 25], whose mean is exactly
the arm's S_t. Bootstrap percentile intervals are not used for inference.

Every decision function refuses non-finite input (review item 3): safe2/analyze.py first applies
the pre-registered imputation (impute_task, impute_selection), then calls the decision rules.
"""
from __future__ import annotations

import hashlib
import inspect
import math

import numpy as np

from lagu.analyze import fisher_exact

COST_LIMIT = 25.0
PENALTY = 1.0              # return units per cost unit above the limit
COLLAPSE_R = 0.2           # a run whose normalised held-out return r is below this has collapsed
ALPHA = 0.025              # family-wise one-sided level; the primary family holds H_BVS only
ALPHA_TEST = 0.016         # nominal level of the permutation test (Appendix G.1: size <= ALPHA under
                           # the unequal-shape nulls; calibrated by `python -m safe2.power --calibrate`)
N_PERM = 100_000           # permutations of the confirmatory test (RNG seed 0)
GUARD_DR = -0.10           # G-r: no task may have a normalised return contrast below this
GUARD_COST_TOL = 5.0       # G-c: candidate task mean cost <= max(25, B*'s) + 5 in every task
GUARD_COLLAPSE_EXTRA = 2   # G-col: candidate collapses <= B* collapses + 2
TIE = 0.5                  # selection: scores within this of the best are tied
KILL_Z, KILL_SD_FLOOR = 2.0, 3.0
EQ_R, EQ_C = 0.10, 0.20    # equivalence margins (normalised r, and c = C / 25, i.e. 5 cost units);
                           # the same 5 units define "at matched cost" in section 13

# Functions whose source is hash-checked by `safe2.analyze --unblind` against tag method-v0
# (section 10; review item 4). Any difference is a deviation and the method-v0 version decides.
DECISION_FUNCTIONS = ("penalised", "arm_score", "hinge_diff", "linearised", "collapsed", "impute_task",
                      "impute_selection", "hinge_perm_test", "strat_perm_test", "welch_strat", "guards",
                      "label", "primary", "selection_score", "eligible", "rank", "kill_b")


def _finite(name: str, *arrays) -> None:
    for a in arrays:
        a = np.asarray(a, float)
        if a.size == 0 or not np.all(np.isfinite(a)):
            raise ValueError(f"{name}: inputs must be non-empty and finite (impute first)")


# --- endpoints ----------------------------------------------------------------------------------
def penalised(R, C, limit: float = COST_LIMIT, weight: float = PENALTY) -> np.ndarray:
    """R - weight * max(0, C - limit), elementwise. Used on arm means and per episode, never as the
    per-run primary score."""
    R, C = np.asarray(R, float), np.asarray(C, float)
    return R - weight * np.maximum(0.0, C - limit)


def arm_score(R, C, r_ref: float) -> float:
    """S of one arm on one task: [mean R - max(0, mean C - 25)] / R_ref, on the runs' held-out means."""
    return float(penalised(np.mean(np.asarray(R, float)), np.mean(np.asarray(C, float)))) / r_ref


def hinge_diff(Rx, Cx, Ry, Cy, r_ref) -> float:
    """Delta_s: mean over tasks of arm_score(x) - arm_score(y)."""
    return float(np.mean([arm_score(a, b, rr) - arm_score(c, d, rr)
                          for a, b, c, d, rr in zip(Rx, Cx, Ry, Cy, r_ref)]))


def linearised(R, C, r_ref: float) -> np.ndarray:
    """Per-run linearised score [R - w (C - 25)] / R_ref with w = 1[mean C > 25] (delta method).
    Its mean equals arm_score(R, C, r_ref)."""
    R, C = np.asarray(R, float), np.asarray(C, float)
    w = PENALTY if C.mean() > COST_LIMIT else 0.0
    return (R - w * (C - COST_LIMIT)) / r_ref


def collapsed(R, r_ref: float) -> np.ndarray:
    """Collapse flag per run from the endpoint: normalised return below COLLAPSE_R, or non-finite.
    (A run whose whitelisted logged values went non-finite is also collapsed; analyze passes that
    flag to primary() as `diverged`.)"""
    R = np.asarray(R, float)
    with np.errstate(invalid="ignore"):
        return ~np.isfinite(R) | (R / r_ref < COLLAPSE_R)


def violation_share(episode_costs, limit: float = COST_LIMIT) -> float:
    """Fraction of a run's held-out episodes whose cost exceeds the limit."""
    c = np.asarray(episode_costs, float)
    return float((c > limit).mean())


def cvar_upper(episode_costs, alpha: float = 0.10) -> float:
    """CVaR_alpha of episode cost: the mean of the worst ceil(alpha * n) episode costs of a run."""
    c = np.sort(np.asarray(episode_costs, float))[::-1]
    return float(c[:max(1, math.ceil(alpha * len(c)))].mean())


# --- imputation (sections 11 and 16) --------------------------------------------------------------
def impute_task(Rx, Cx, Ry, Cy, done_x=None, done_y=None) -> dict:
    """Pre-registered imputation for one task, applied before any decision rule.

    x = candidate, y = B*. Arrays hold one entry per PLANNED run (NaN for a run without a held-out
    file); done_* marks runs that finished (default: every run finished).
      * a finished run with a non-finite R or C (either arm): R := 0, C := the worst finite per-run
        cost of that task in either arm; it is then a collapse (R / R_ref = 0 < 0.2);
      * an unfinished candidate run (still unfinished after 3 resumes): the same worst case;
      * an unfinished B* run: R := median of B*'s finished finite R, C := median of their C; it
        counts as collapsed only if that median R does.
    Returns the imputed arrays and the masks of imputed runs."""
    Rx, Cx, Ry, Cy = (np.array(a, float, copy=True) for a in (Rx, Cx, Ry, Cy))
    dx = np.ones(len(Rx), bool) if done_x is None else np.asarray(done_x, bool)
    dy = np.ones(len(Ry), bool) if done_y is None else np.asarray(done_y, bool)
    okx = dx & np.isfinite(Rx) & np.isfinite(Cx)
    oky = dy & np.isfinite(Ry) & np.isfinite(Cy)
    finite_c = np.r_[Cx[okx], Cy[oky]]
    if finite_c.size == 0 or not oky.any():
        raise ValueError("impute_task: no finished finite B* run (or no finite cost) in this task")
    worst = float(finite_c.max())
    bad_x = ~okx                                    # unfinished or non-finite candidate run
    Rx[bad_x], Cx[bad_x] = 0.0, worst
    nonfin_y = dy & ~oky
    Ry[nonfin_y], Cy[nonfin_y] = 0.0, worst
    miss_y = ~dy
    Ry[miss_y], Cy[miss_y] = float(np.median(Ry[oky])), float(np.median(Cy[oky]))
    return {"Rx": Rx, "Cx": Cx, "Ry": Ry, "Cy": Cy, "worst_cost": worst,
            "imputed_x": bad_x, "imputed_y": nonfin_y | miss_y, "missing_y": miss_y}


def impute_selection(configs: list[dict]) -> list[dict]:
    """Selection-data imputation: a seed whose selection-window R or C is non-finite gets R := 0 and
    C := the largest finite per-seed C among all configs ranked together, and is flagged collapsed."""
    allc = np.concatenate([np.asarray(c["C"], float) for c in configs])
    allc = allc[np.isfinite(allc)]
    worst = float(allc.max()) if allc.size else 2 * COST_LIMIT
    out = []
    for c in configs:
        R, C = np.array(c["R"], float, copy=True), np.array(c["C"], float, copy=True)
        bad = ~(np.isfinite(R) & np.isfinite(C))
        R[bad], C[bad] = 0.0, worst
        out.append(dict(c, R=R, C=C, collapsed=np.asarray(c["collapsed"], bool) | bad))
    return out


# --- the stratified contrast and its tests ------------------------------------------------------
def strat_diff(xs, ys) -> float:
    """Mean over tasks of (mean of x_t - mean of y_t), for per-run quantities."""
    return float(np.mean([np.mean(x) - np.mean(y) for x, y in zip(xs, ys)]))


def strat_perm_test(xs, ys, n_perm: int = N_PERM, seed: int = 0, chunk: int = 5000) -> dict:
    """One-sided (x > y) stratified permutation test of strat_diff for a per-run quantity.

    Within each task the pooled runs are split at random into groups of the original sizes; the
    p-value is (1 + #{permuted statistic >= observed}) / (1 + n_perm)."""
    xs = [np.asarray(x, float) for x in xs]
    ys = [np.asarray(y, float) for y in ys]
    _finite("strat_perm_test", *xs, *ys)
    rng = np.random.default_rng(seed)
    obs = strat_diff(xs, ys)
    tol = 1e-12 * max(1.0, abs(obs))
    hits, done = 0, 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        stat = np.zeros(b)
        for x, y in zip(xs, ys):
            z = np.concatenate([x, y])
            nx, ny = len(x), len(y)
            idx = np.argsort(rng.random((b, nx + ny)), axis=1)[:, :nx]
            sx = z[idx].sum(axis=1)
            stat += sx / nx - (z.sum() - sx) / ny
        stat /= len(xs)
        hits += int((stat >= obs - tol).sum())
        done += b
    return {"stat": obs, "p": (1 + hits) / (1 + n_perm), "n_perm": n_perm, "seed": seed}


def hinge_perm_test(Rx, Cx, Ry, Cy, r_ref, n_perm: int = N_PERM, seed: int = 0, chunk: int = 5000) -> dict:
    """One-sided (x > y) stratified permutation test of hinge_diff (the primary test).

    Within each task the runs' (R, C) pairs are re-split at random into groups of the original
    sizes and the arm-mean hinge is recomputed for each split. Same RNG stream as strat_perm_test,
    so when every arm mean cost stays at or below 25 it equals strat_perm_test on R / R_ref."""
    Rx, Cx, Ry, Cy = ([np.asarray(a, float) for a in arm] for arm in (Rx, Cx, Ry, Cy))
    _finite("hinge_perm_test", *Rx, *Cx, *Ry, *Cy, r_ref)
    rng = np.random.default_rng(seed)
    obs = hinge_diff(Rx, Cx, Ry, Cy, r_ref)
    tol = 1e-12 * max(1.0, abs(obs))
    hits, done = 0, 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        stat = np.zeros(b)
        for rx, cx, ry, cy, rr in zip(Rx, Cx, Ry, Cy, r_ref):
            zr, zc = np.concatenate([rx, ry]), np.concatenate([cx, cy])
            nx, ny = len(rx), len(ry)
            idx = np.argsort(rng.random((b, nx + ny)), axis=1)[:, :nx]
            srx, scx = zr[idx].sum(axis=1), zc[idx].sum(axis=1)
            mrx, mcx = srx / nx, scx / nx
            mry, mcy = (zr.sum() - srx) / ny, (zc.sum() - scx) / ny
            stat += (penalised(mrx, mcx) - penalised(mry, mcy)) / rr
        stat /= len(Rx)
        hits += int((stat >= obs - tol).sum())
        done += b
    return {"stat": obs, "p": (1 + hits) / (1 + n_perm), "n_perm": n_perm, "seed": seed}


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Upper tail P(T > t) of Student's t with df degrees of freedom."""
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def t_ppf(q: float, df: float) -> float:
    """Quantile of Student's t (bisection on t_sf)."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - t_sf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def welch_strat(xs, ys, levels=(0.95, 0.90)) -> dict:
    """Welch-type t for strat_diff: SE^2 = sum over tasks and arms of s^2 / n / T^2, Satterthwaite df.
    Returns the statistic, SE, df, one-sided p (x > y) and two-sided intervals at `levels`."""
    xs = [np.asarray(x, float) for x in xs]
    ys = [np.asarray(y, float) for y in ys]
    _finite("welch_strat", *xs, *ys)
    T = len(xs)
    v = [np.var(g, ddof=1) / len(g) / T ** 2 for g in xs + ys]
    n = [len(g) for g in xs + ys]
    se = math.sqrt(sum(v))
    stat = strat_diff(xs, ys)
    if se == 0.0:
        p = 0.0 if stat > 0 else 1.0
        return {"stat": stat, "se": 0.0, "df": float("inf"), "p": p,
                **{f"ci{round(100 * L)}": (stat, stat) for L in levels}}
    df = se ** 4 / sum(vi ** 2 / (ni - 1) for vi, ni in zip(v, n))
    out = {"stat": stat, "se": se, "df": df, "p": t_sf(stat / se, df)}
    for L in levels:
        h = t_ppf(1 - (1 - L) / 2, df) * se
        out[f"ci{round(100 * L)}"] = (stat - h, stat + h)
    return out


# --- the primary decision -----------------------------------------------------------------------
def guards(Rx, Cx, Ry, Cy, r_ref, colx, coly) -> dict:
    """Guards of the success rule (point estimates). Per-task lists of per-run held-out return and
    cost (x = candidate, y = B*), R_ref per task, and per-task collapse flags.
      G-r:   every task's (mean R_x - mean R_y) / R_ref >= -0.10;
      G-c:   every task's mean C_x <= max(25, mean C_y) + 5;
      G-col: pooled collapses of x <= those of y + 2."""
    dr = [(np.mean(rx) - np.mean(ry)) / rr for rx, ry, rr in zip(Rx, Ry, r_ref)]
    cx = [float(np.mean(c)) for c in Cx]
    cy = [float(np.mean(c)) for c in Cy]
    ncx = int(sum(np.sum(c) for c in colx))
    ncy = int(sum(np.sum(c) for c in coly))
    return {"task_dr": dr, "task_cost_x": cx, "task_cost_y": cy, "collapses_x": ncx, "collapses_y": ncy,
            "return": all(d >= GUARD_DR for d in dr),
            "cost": all(a <= max(COST_LIMIT, b) + GUARD_COST_TOL for a, b in zip(cx, cy)),
            "collapse": ncx <= ncy + GUARD_COLLAPSE_EXTRA}


def label(dr_ci95, dc_ci95, dr_ci90, dc_ci90, ds_ci95, extra_collapses: int) -> str:
    """Descriptive label of a contrast that is not confirmed (first match): Hurts, Shift, Equivalent,
    Inconclusive. d* are Welch intervals of the stratified normalised contrasts (r = R / R_ref,
    c = C / 25, s = the linearised arm-mean penalised score)."""
    if ds_ci95[1] < 0 or extra_collapses >= 3:
        return "Hurts"
    if (dr_ci95[0] > 0 and dc_ci95[0] > 0) or (dr_ci95[1] < 0 and dc_ci95[1] < 0):
        return "Shift"
    if -EQ_R < dr_ci90[0] and dr_ci90[1] < EQ_R and -EQ_C < dc_ci90[0] and dc_ci90[1] < EQ_C:
        return "Equivalent"
    return "Inconclusive"


def primary(Rx, Cx, Ry, Cy, r_ref, diverged_x=None, diverged_y=None, n_perm: int = N_PERM,
            seed: int = 0, alpha: float = ALPHA_TEST) -> dict:
    """The confirmatory test of one candidate against B*.

    Rx, Cx, Ry, Cy: per-task lists of per-run held-out mean return and cost (x = candidate, y = B*),
    already imputed (impute_task); r_ref: R_ref per task; diverged_*: per-task flags of runs whose
    whitelisted logged values went non-finite (collapsed whatever their endpoint). 'beats' is True
    iff the permutation p <= alpha and every guard holds. Raises ValueError on non-finite input."""
    Rx, Cx, Ry, Cy = ([np.asarray(a, float) for a in arm] for arm in (Rx, Cx, Ry, Cy))
    _finite("primary", *Rx, *Cx, *Ry, *Cy, r_ref)
    if not all(rr > 0 for rr in r_ref):
        raise ValueError("primary: R_ref must be positive")
    divx = diverged_x or [np.zeros(len(R), bool) for R in Rx]
    divy = diverged_y or [np.zeros(len(R), bool) for R in Ry]
    colx = [collapsed(R, rr) | np.asarray(d, bool) for R, rr, d in zip(Rx, r_ref, divx)]
    coly = [collapsed(R, rr) | np.asarray(d, bool) for R, rr, d in zip(Ry, r_ref, divy)]
    perm = hinge_perm_test(Rx, Cx, Ry, Cy, r_ref, n_perm=n_perm, seed=seed)
    wel = welch_strat([linearised(R, C, rr) for R, C, rr in zip(Rx, Cx, r_ref)],
                      [linearised(R, C, rr) for R, C, rr in zip(Ry, Cy, r_ref)])
    g = guards(Rx, Cx, Ry, Cy, r_ref, colx, coly)
    wr = welch_strat([R / rr for R, rr in zip(Rx, r_ref)], [R / rr for R, rr in zip(Ry, r_ref)])
    wc = welch_strat([C / COST_LIMIT for C in Cx], [C / COST_LIMIT for C in Cy])
    beats = perm["p"] <= alpha and g["return"] and g["cost"] and g["collapse"]
    if beats:
        verdict = "Beats B*"
    elif perm["p"] <= alpha:
        verdict = "Not confirmed: guard failed (" + ", ".join(k for k in ("return", "cost", "collapse")
                                                             if not g[k]) + ")"
    else:
        verdict = label(wr["ci95"], wc["ci95"], wr["ci90"], wc["ci90"], wel["ci95"],
                        g["collapses_x"] - g["collapses_y"])
    return {"ds": perm["stat"], "p_perm": perm["p"], "alpha_test": alpha, "welch": wel, "welch_r": wr,
            "welch_c": wc, "guards": g, "beats": bool(beats), "verdict": verdict,
            "fisher_collapse": fisher_exact(g["collapses_x"], sum(len(c) for c in colx) - g["collapses_x"],
                                            g["collapses_y"], sum(len(c) for c in coly) - g["collapses_y"])}


# --- pilot selection and kill rule (b) ----------------------------------------------------------
def selection_score(R_seeds, C_seeds) -> float:
    """Arm-mean penalised score of a config: mean R - max(0, mean C - 25) over its seeds'
    selection-window means (the hinge on the config mean, as in the primary endpoint)."""
    _finite("selection_score", R_seeds, C_seeds)
    return float(penalised(np.mean(np.asarray(R_seeds, float)), np.mean(np.asarray(C_seeds, float))))


def eligible(collapsed_seeds) -> bool:
    """A config is ineligible when at least two thirds of its seeds collapsed (2 of 3, 4 of 6)."""
    c = np.asarray(collapsed_seeds, bool)
    return c.sum() * 3 < 2 * len(c)


def rank(configs: list[dict], tie: float = TIE, eligible_only: bool = True) -> list[str]:
    """Order configs best first. Each config: name, R and C (per-seed selection-window means, already
    imputed), collapsed (per seed) and hours (compute per run). Repeatedly take the best score;
    configs within `tie` of it are ordered by fewer collapses, then lower mean cost, then lower
    compute, then name. eligible_only=False appends the ineligible configs, ranked by the same rule,
    after the eligible ones (used when fewer than the needed number of configs are eligible)."""
    for c in configs:
        _finite(f"rank ({c['name']})", c["R"], c["C"], [c["hours"]])

    def order(pool):
        pool = [dict(c, score=selection_score(c["R"], c["C"]), ncol=int(np.sum(c["collapsed"])),
                     cost=float(np.mean(c["C"]))) for c in pool]
        out = []
        while pool:
            best = max(c["score"] for c in pool)
            tied = [c for c in pool if c["score"] >= best - tie]
            pick = min(tied, key=lambda c: (c["ncol"], c["cost"], c["hours"], c["name"]))
            out.append(pick["name"])
            pool.remove(pick)
        return out

    ok = [c for c in configs if eligible(c["collapsed"])]
    rest = [c for c in configs if not eligible(c["collapsed"])]
    return order(ok) + ([] if eligible_only else order(rest))


def kill_b(cand_R, cand_C, base_R, base_C, z: float = KILL_Z, sd_floor: float = KILL_SD_FLOOR) -> dict:
    """Kill rule (b): kill iff the candidate's selection score is more than z standard errors below
    B*'s. SE = sqrt(s_c^2 / n_c + s_b^2 / n_b), s = per-seed SD (ddof 1) of the linearised scores
    R - w (C - 25) (w = 1[config mean C > 25]), each floored at sd_floor."""
    _finite("kill_b", cand_R, cand_C, base_R, base_C)
    lc = linearised(cand_R, cand_C, 1.0)
    lb = linearised(base_R, base_C, 1.0)
    sc = max(float(np.std(lc, ddof=1)), sd_floor)
    sb = max(float(np.std(lb, ddof=1)), sd_floor)
    se = math.sqrt(sc ** 2 / len(lc) + sb ** 2 / len(lb))
    diff = selection_score(cand_R, cand_C) - selection_score(base_R, base_C)
    return {"diff": diff, "se": se, "kill": bool(diff < -z * se)}


def source_hashes(names=DECISION_FUNCTIONS) -> dict:
    """SHA-256 of the source text of each decision function (recorded at method-v0, checked by
    `safe2.analyze --unblind`)."""
    mod = inspect.getmodule(source_hashes)
    return {n: hashlib.sha256(inspect.getsource(getattr(mod, n)).encode()).hexdigest() for n in names}
