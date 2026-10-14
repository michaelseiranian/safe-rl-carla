"""Operating characteristics of the pre-registered rules on synthetic data (prereg_method.md section 15,
Appendix G).

    python -m safe2.power [--reps 2000] [--perms 1000] [--seed 0] [--sections ABCDEF]
    python -m safe2.power --calibrate [--reps 4000] [--perms 1000]

No study data are read. Each synthetic run has held-out means R ~ N(mu_R, sd_R) and
C = max(0, mu_C + sd_C (rho z1 + sqrt(1 - rho^2) z2)), with z1 the return's own normal draw; with
probability p_col the run has collapsed instead (R ~ N(0.05 R_ref, 0.02 R_ref), C ~ |N(5, 5)|).
Every task uses PointGoal1 units (R_ref = 26.3). The primary statistic is the stratified
arm-mean penalised score (safe2.stats.hinge_diff) tested at stats.ALPHA_TEST. Printed:
  A. type-I error under identical arm distributions: the permutation test, the Welch-type t on the
     linearised scores, how often each guard holds, and the full success rule;
  A2. (--calibrate) size under unequal-shape nulls with equal means (a tight arm against an arm with
     collapses), and the largest nominal level that keeps every such size <= 0.025;
  B. power of the test and of the full rule, including the review's under-spending scenarios, with
     the per-run-hinge statistic of the earlier draft for comparison;
  C. the minimum detectable stratified score difference (normal approximation, simulated SD);
  D. why the matched-cost hypothesis cannot be tested, and how often the Equivalent label is reachable;
  E. kill rule (b): probability of killing a candidate as good as, or better than, B*;
  F. how often each form of the cost guard holds at a tie, as the number of runs grows.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from safe2 import stats

R_REF = 26.3
TASKS = 3
Z80 = 0.841621


def zq(p: float) -> float:
    """Upper-p quantile of the standard normal (bisection on erfc)."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(mid / math.sqrt(2)) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def draw(rng, n, mu_R, mu_C, sd_R, sd_C, rho, p_col):
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    R = mu_R + sd_R * z1
    C = np.maximum(0.0, mu_C + sd_C * (rho * z1 + math.sqrt(1 - rho ** 2) * z2))
    col = rng.random(n) < p_col
    R = np.where(col, rng.normal(0.05 * R_REF, 0.02 * R_REF, n), R)
    C = np.where(col, np.abs(rng.normal(5.0, 5.0, n)), C)
    return R, C


def study(rng, n, x, y):
    """x, y: dicts of draw() arguments for candidate and B*. Returns per-task lists Rx, Cx, Ry, Cy."""
    Rx, Cx, Ry, Cy = [], [], [], []
    for _t in range(TASKS):
        a, b = draw(rng, n, **x), draw(rng, n, **y)
        Rx.append(a[0]), Cx.append(a[1]), Ry.append(b[0]), Cy.append(b[1])
    return Rx, Cx, Ry, Cy


def arm(mu_R, mu_C, sd_R, sd_C, rho=0.0, p_col=0.1) -> dict:
    return dict(mu_R=mu_R, mu_C=mu_C, sd_R=sd_R, sd_C=sd_C, rho=rho, p_col=p_col)


def simulate(rng, reps, perms, n, x, y, old=False) -> np.ndarray:
    """Rates of: [0] hinge permutation test at ALPHA_TEST, [1] Welch on linearised scores at
    ALPHA_TEST, [2-4] guards G-r, G-c (adopted, +5), G-col, [5] all guards, [6] full rule,
    [7] full rule with the strict G-c of the earlier draft, [8] the earlier draft's per-run-hinge
    test at 0.025 and [9] its full rule (per-run test and strict G-c); 8-9 only with old=True."""
    out = np.zeros(10)
    rr = [R_REF] * TASKS
    for _ in range(reps):
        Rx, Cx, Ry, Cy = study(rng, n, x, y)
        p = stats.hinge_perm_test(Rx, Cx, Ry, Cy, rr, n_perm=perms, seed=int(rng.integers(2 ** 31)))["p"]
        rej = p <= stats.ALPHA_TEST
        lx = [stats.linearised(R, C, R_REF) for R, C in zip(Rx, Cx)]
        ly = [stats.linearised(R, C, R_REF) for R, C in zip(Ry, Cy)]
        wel = stats.welch_strat(lx, ly, levels=())["p"] <= stats.ALPHA_TEST
        g = stats.guards(Rx, Cx, Ry, Cy, rr, [stats.collapsed(R, R_REF) for R in Rx],
                         [stats.collapsed(R, R_REF) for R in Ry])
        ok = g["return"] and g["cost"] and g["collapse"]
        strict = all(a <= max(stats.COST_LIMIT, b) for a, b in zip(g["task_cost_x"], g["task_cost_y"]))
        ok_old = g["return"] and strict and g["collapse"]
        row = [rej, wel, g["return"], g["cost"], g["collapse"], ok, rej and ok, rej and ok_old, 0, 0]
        if old:
            sx = [stats.penalised(R, C) / R_REF for R, C in zip(Rx, Cx)]
            sy = [stats.penalised(R, C) / R_REF for R, C in zip(Ry, Cy)]
            po = stats.strat_perm_test(sx, sy, n_perm=perms, seed=int(rng.integers(2 ** 31)))["p"] <= stats.ALPHA
            row[8], row[9] = po, po and ok_old
        out += row
    return out / reps


def fmt(r, old=False) -> str:
    s = (f"{r[0]:.3f} {r[1]:.3f} | {r[2]:.2f}  {r[3]:.2f}   {r[4]:.2f}  {r[5]:.2f} | "
         f"{r[6]:.3f} {r[7]:.3f}")
    return s + (f" | {r[8]:.3f} {r[9]:.3f}" if old else "")


SDS = [(6, 15, 0.0), (9, 25, 0.0), (9, 25, 0.5)]


def section_a(rng, reps, perms):
    print(f"A. Type-I error, identical arms (mu_R 15, mu_C 25), nominal level {stats.ALPHA_TEST}")
    print("   sd_R sd_C rho p_col  n | perm  welch | g_ret g_cost g_col  all | full  full(strict G-c)")
    for n in (12, 16):
        for sd_R, sd_C, rho in SDS:
            for p_col in (0.0, 0.2, 0.4):
                a = arm(15, 25, sd_R, sd_C, rho, p_col)
                r = simulate(rng, reps, perms, n, a, a)
                print(f"   {sd_R:>4} {sd_C:>4} {rho:>3} {p_col:>5} {n:>2} | " + fmt(r))


def shape_null(rng, n, p, sd_wide, tight_is_candidate):
    """Unequal-shape null with equal mean return and identical costs well below the limit (the hinge
    is inactive, so this isolates the shape effect the review measured in score units): a tight arm
    R ~ N(0.45, 0.15) R_ref against an arm that collapses (R ~ N(0.05, 0.02) R_ref) with probability
    p and is otherwise wide (N(hi, sd_wide) R_ref), hi chosen so that both means are 0.45 R_ref."""
    hi = (0.45 - 0.05 * p) / (1 - p)
    Rt, Rw, Ct, Cw = [], [], [], []
    for _t in range(TASKS):
        Rt.append(rng.normal(0.45 * R_REF, 0.15 * R_REF, n))
        col = rng.random(n) < p
        Rw.append(np.where(col, rng.normal(0.05 * R_REF, 0.02 * R_REF, n), rng.normal(hi * R_REF, sd_wide * R_REF, n)))
        Ct.append(np.maximum(0.0, rng.normal(10.0, 5.0, n)))
        Cw.append(np.maximum(0.0, rng.normal(10.0, 5.0, n)))
    return (Rt, Ct, Rw, Cw) if tight_is_candidate else (Rw, Cw, Rt, Ct)


def section_a2(rng, reps, perms, calibrate):
    grid = np.round(np.arange(0.010, 0.02501, 0.0005), 4)
    print(f"A2. Size under unequal-shape nulls (n = 16, equal mean score, hinge inactive), {reps} studies each")
    print("   wide-arm collapse share, wide SD | candidate | size at 0.025 | size at ALPHA_TEST "
          f"({stats.ALPHA_TEST})")
    worst = np.zeros(len(grid))
    rr = [R_REF] * TASKS
    for p in (0.0, 0.2, 0.4):
        for sd_w in (0.4, 0.6):
            for tight in (True, False):
                ps = np.array([stats.hinge_perm_test(*shape_null(rng, 16, p, sd_w, tight), rr, n_perm=perms,
                                                     seed=int(rng.integers(2 ** 31)))["p"] for _ in range(reps)])
                sizes = np.array([(ps <= a).mean() for a in grid])
                worst = np.maximum(worst, sizes)
                print(f"   {p:>4} {sd_w:>4} | {'tight' if tight else 'wide ':>9} | {np.mean(ps <= 0.025):.4f}"
                      f"        | {np.mean(ps <= stats.ALPHA_TEST):.4f}")
    if calibrate:
        okay = grid[worst <= 0.025]
        print("   largest size over these nulls, by nominal level:")
        print("   " + "  ".join(f"{a:.4f}:{w:.4f}" for a, w in zip(grid, worst)))
        print(f"   => largest nominal level with every size <= 0.025: {okay.max() if okay.size else None}")


SCENARIOS = [
    ("return +3, same cost 25", (18, 25), (15, 25), None),
    ("return +5, same cost 25", (20, 25), (15, 25), None),
    ("return +8, same cost 25", (23, 25), (15, 25), None),
    ("return +5, both at cost 20", (20, 20), (15, 20), None),
    ("return +5, cost 20 vs B* 25", (20, 20), (15, 25), None),
    ("B* over: 15/35 vs 14/25", (14, 25), (15, 35), None),
    ("B* over: 15/45 vs 14/25", (14, 25), (15, 45), None),
    ("under-spend: 15/15 vs 15/25", (15, 15), (15, 25), None),
    ("review example: 12/5 vs 14.5/25", (12, 5), (14.5, 25), None),
    ("same means, cand cost SD / 3", (15, 25), (15, 25), "third"),
    ("same means, cand cost SD 5 vs 15", (15, 25), (15, 25), (6, 5, 6, 15)),
    ("same means, cand cost SD 10 vs 25", (15, 25), (15, 25), (9, 10, 9, 25)),
]


def section_b(rng, reps, perms):
    print("\nB. Power, n = 16 per arm per task, p_col 0.1 in both arms; 'old' = per-run hinge test at 0.025")
    print("   scenario (candidate vs B*: mu_R/mu_C)    sd_R sd_C rho | perm  welch | g_ret g_cost g_col  all |"
          " full  full(strict G-c) | old-test old-full")
    for name, (xr, xc), (yr, yc), sds in SCENARIOS:
        settings = SDS if sds in (None, "third") else [(sds[0], sds[3], 0.0)]
        for sd_R, sd_C, rho in settings:
            if sds is None:
                x, y = arm(xr, xc, sd_R, sd_C, rho), arm(yr, yc, sd_R, sd_C, rho)
            elif sds == "third":
                x, y = arm(xr, xc, sd_R, sd_C / 3, rho), arm(yr, yc, sd_R, sd_C, rho)
            else:
                x, y = arm(xr, xc, sds[0], sds[1]), arm(yr, yc, sds[2], sds[3])
            r = simulate(rng, reps, perms, 16, x, y, old=True)
            print(f"   {name:<37} {sd_R:>4} {sd_C:>4} {rho:>3} | " + fmt(r, old=True))


def section_c(rng):
    print(f"\nC. Minimum detectable Delta_s (80% power, one-sided {stats.ALPHA_TEST}, normal approximation;"
          " SD of Delta_s simulated under the null, 20000 studies)")
    print("   sd_R sd_C rho p_col | SD(Delta_s) n=12  n=16 | MDE n=12        MDE n=16   (x 26.3 = PointGoal1 units)")
    k = zq(stats.ALPHA_TEST) + Z80
    rr = [R_REF] * TASKS
    for sd_R, sd_C, rho in SDS + [(6, 25, 0.5)]:
        for p_col in (0.0, 0.2):
            a = arm(15, 25, sd_R, sd_C, rho, p_col)
            sdd = {}
            for n in (12, 16):
                sdd[n] = float(np.std([stats.hinge_diff(*study(rng, n, a, a), rr) for _ in range(20000)]))
            print(f"   {sd_R:>4} {sd_C:>4} {rho:>3} {p_col:>5} | {sdd[12]:.3f}      {sdd[16]:.3f} | "
                  f"{k * sdd[12]:.3f} ({k * sdd[12] * R_REF:.1f})   {k * sdd[16]:.3f} ({k * sdd[16] * R_REF:.1f})")


def section_d(rng):
    print("\nD. Matched cost and the Equivalent label: equal true means in both arms, n = 16 per arm per task")
    for sd_C in (15, 25):
        se = sd_C * math.sqrt(2 / 16)
        p1 = math.erf(2.5 / se / math.sqrt(2))
        se_pool = se / math.sqrt(TASKS)
        print(f"   cost SD {sd_C}: SE of a per-task mean-cost difference {se:.1f}; "
              f"P(|dC| <= 2.5) per task {p1:.2f}, in all 3 tasks {p1 ** 3:.3f}; "
              f"pooled 90% half-width {1.645 * se_pool:.1f} cost units")
    print("   P(Equivalent: dr 90% CI inside +-0.10 and dc 90% CI inside +-0.20), 4000 null studies:")
    for sd_R, sd_C, rho in SDS:
        for p_col in (0.0, 0.1):
            a = arm(15, 25, sd_R, sd_C, rho, p_col)
            hits, hr, hc = 0, [], []
            for _ in range(4000):
                Rx, Cx, Ry, Cy = study(rng, 16, a, a)
                wr = stats.welch_strat([R / R_REF for R in Rx], [R / R_REF for R in Ry])
                wc = stats.welch_strat([C / 25 for C in Cx], [C / 25 for C in Cy])
                hr.append((wr["ci90"][1] - wr["ci90"][0]) / 2), hc.append((wc["ci90"][1] - wc["ci90"][0]) / 2)
                hits += stats.label(wr["ci95"], wc["ci95"], wr["ci90"], wc["ci90"], (-1, 1), 0) == "Equivalent"
            print(f"   sd_R {sd_R} sd_C {sd_C} rho {rho} p_col {p_col}: P(Equivalent) {hits / 4000:.3f}; "
                  f"median 90% half-width dr {np.median(hr):.3f}, dc {np.median(hc):.3f}")


def section_e(rng):
    print("\nE. Kill rule (b): candidate best-of-4 (3 seeds) vs B* = best 6-seed mean among the top 3 of 10 "
          "(costs below the limit, so the score is the return)")
    print("   per-seed score SD | P(kill) true advantage 0   +2    +4")
    N = 20_000
    for sd in (3, 5, 8, 12):
        out = []
        for adv in (0.0, 2.0, 4.0):
            kills = 0
            for _ in range(N // 100):
                base = rng.normal(0, sd, (100, 10, 6))
                top3 = np.argsort(-base[:, :, :3].mean(2), axis=1)[:, :3]
                b6 = np.take_along_axis(base, top3[:, :, None], 1)
                bstar = b6[np.arange(100), b6.mean(2).argmax(1)]
                cand = rng.normal(adv, sd, (100, 4, 3))
                cstar = cand[np.arange(100), cand.mean(2).argmax(1)]
                kills += sum(stats.kill_b(c + 15, np.full(3, 10.0), b + 15, np.full(6, 10.0))["kill"]
                             for c, b in zip(cstar, bstar))
            out.append(kills / N)
        print(f"   {sd:>17} | {out[0]:.3f}  {out[1]:.3f}  {out[2]:.3f}")


def section_f(rng):
    print("\nF. How often the cost guard holds when both arms have the same true cost distribution "
          "(mu_C 25, p_col 0.1), 20000 studies")
    print("   sd_C    n | strict per task (draft) | per task + 5 (adopted) | pooled mean_t[Cx - max(25, Cy)] <= 0")
    for sd_C in (15, 25):
        for n in (16, 64, 256):
            a = arm(15, 25, 6, sd_C)
            s1 = s2 = s3 = 0
            for _ in range(20000):
                _, Cx, _, Cy = study(rng, n, a, a)
                cx, cy = [c.mean() for c in Cx], [c.mean() for c in Cy]
                s1 += all(u <= max(25, v) for u, v in zip(cx, cy))
                s2 += all(u <= max(25, v) + stats.GUARD_COST_TOL for u, v in zip(cx, cy))
                s3 += np.mean([u - max(25, v) for u, v in zip(cx, cy)]) <= 0
            print(f"   {sd_C:>4} {n:>4} | {s1 / 20000:.3f}                   | {s2 / 20000:.3f}"
                  f"                  | {s3 / 20000:.3f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--perms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sections", default="ABCDEF")
    p.add_argument("--calibrate", action="store_true", help="run section A2 and print the calibrated level")
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    if a.calibrate:
        section_a2(rng, a.reps, a.perms, calibrate=True)
        return
    for s in a.sections:
        {"A": lambda: section_a(rng, a.reps, a.perms), "B": lambda: section_b(rng, a.reps, a.perms),
         "C": lambda: section_c(rng), "D": lambda: section_d(rng), "E": lambda: section_e(rng),
         "F": lambda: section_f(rng)}[s]()


if __name__ == "__main__":
    main()
