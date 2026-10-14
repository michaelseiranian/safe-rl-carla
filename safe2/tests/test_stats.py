"""safe2.stats: distribution functions and tests against hand-computed cases, the decision rules on
constructed inputs, and the size of the permutation test under a null."""
import itertools
import math

import numpy as np
import pytest

from safe2 import stats as st


def test_betainc_and_t_distribution_match_tabulated_values():
    assert st.betainc(2, 3, 0.5) == pytest.approx(11 / 16, abs=1e-12)          # binomial identity
    assert st.betainc(0.5, 0.5, 0.5) == pytest.approx(0.5, abs=1e-12)
    assert st.t_sf(2.0, 10) == pytest.approx(0.036694, abs=1e-6)
    assert st.t_sf(2.228139, 10) == pytest.approx(0.025, abs=1e-6)
    assert st.t_sf(-2.228139, 10) == pytest.approx(0.975, abs=1e-6)
    assert st.t_sf(1.959964, 1e7) == pytest.approx(0.025, abs=1e-5)             # normal limit
    assert st.t_sf(0.0, 3) == pytest.approx(0.5, abs=1e-12)
    assert st.t_ppf(0.975, 10) == pytest.approx(2.228139, abs=1e-5)
    assert st.t_ppf(0.95, 5) == pytest.approx(2.015048, abs=1e-5)


def test_penalised_score_collapse_violation_and_cvar():
    assert st.penalised([20, 20, 20], [10, 25, 31]).tolist() == [20, 20, 14]
    assert st.collapsed([5.0, 5.3, float("nan")], 26.3).tolist() == [True, False, True]   # 0.2 * 26.3 = 5.26
    c = np.arange(1, 21, dtype=float)                                               # 20 episodes
    assert st.violation_share(c) == 0.0 and st.violation_share(c + 10) == pytest.approx(0.25)
    assert st.cvar_upper(c) == pytest.approx((20 + 19) / 2)                         # worst ceil(0.1 * 20) = 2
    assert st.cvar_upper(np.arange(1, 201, dtype=float)) == pytest.approx(np.mean(np.arange(181, 201)))


def test_permutation_test_matches_full_enumeration():
    x, y = [np.array([1.0, 2.0, 4.0])], [np.array([0.0, 0.5, 3.0])]
    z = np.r_[x[0], y[0]]
    obs = x[0].mean() - y[0].mean()
    stats_all = [z[list(c)].mean() - np.delete(z, list(c)).mean() for c in itertools.combinations(range(6), 3)]
    exact = np.mean([s >= obs - 1e-12 for s in stats_all])
    r = st.strat_perm_test(x, y, n_perm=60_000, seed=1)
    assert r["stat"] == pytest.approx(obs)
    assert r["p"] == pytest.approx(exact, abs=0.006)


def test_permutation_test_is_stratified_and_deterministic():
    # a task with a huge level shared by both arms must not dilute the within-task contrast
    xs = [np.array([1.0, 1.1, 1.2, 1.3]), np.array([101.0, 101.1, 101.2, 101.3])]
    ys = [np.array([0.0, 0.1, 0.2, 0.3]), np.array([100.0, 100.1, 100.2, 100.3])]
    r1 = st.strat_perm_test(xs, ys, n_perm=2000, seed=0)
    r2 = st.strat_perm_test(xs, ys, n_perm=2000, seed=0)
    assert r1 == r2
    assert r1["stat"] == pytest.approx(1.0)
    assert r1["p"] == pytest.approx(1 / 2001, abs=0.004)   # only the identity-like split reaches it
    assert st.strat_perm_test(ys, xs, n_perm=2000, seed=0)["p"] > 0.95


def test_permutation_test_size_under_the_null():
    rng = np.random.default_rng(3)
    p = [st.strat_perm_test([rng.normal(size=12) for _ in range(3)], [rng.normal(size=12) for _ in range(3)],
                            n_perm=400, seed=k)["p"] for k in range(300)]
    assert 0.01 <= np.mean(np.asarray(p) <= 0.05) <= 0.10


def test_welch_matches_hand_computation():
    x, y = np.array([1.0, 2, 3, 4]), np.array([0.0, 1, 1, 2])
    r = st.welch_strat([x], [y])
    se = math.sqrt(np.var(x, ddof=1) / 4 + np.var(y, ddof=1) / 4)
    df = se ** 4 / ((np.var(x, ddof=1) / 4) ** 2 / 3 + (np.var(y, ddof=1) / 4) ** 2 / 3)
    assert r["stat"] == pytest.approx(1.5) and r["se"] == pytest.approx(se) and r["df"] == pytest.approx(df)
    assert r["p"] == pytest.approx(st.t_sf(1.5 / se, df))
    lo, hi = r["ci95"]
    assert (hi - lo) / 2 == pytest.approx(st.t_ppf(0.975, df) * se, rel=1e-6)
    # two tasks: the statistic is the mean of the task contrasts and each task's variance enters / T^2
    r2 = st.welch_strat([x, x], [y, y])
    assert r2["stat"] == pytest.approx(1.5) and r2["se"] == pytest.approx(se / math.sqrt(2))


def test_guards_and_verdicts_on_constructed_cases():
    rr = [26.3, 26.3, 26.3]
    base_R = [np.full(16, 15.0) + np.linspace(-1, 1, 16) for _ in range(3)]
    base_C = [np.full(16, 25.0) + np.linspace(-2, 2, 16) for _ in range(3)]
    # a clear return gain at the same cost: beats B*
    r = st.primary([R + 8 for R in base_R], base_C, base_R, base_C, rr, n_perm=2000)
    assert r["beats"] and r["verdict"] == "Beats B*" and r["p_perm"] < 0.001
    # the same gain bought with cost above max(25, C_B*) + 5 in one task: the cost guard blocks the claim
    Cx = [base_C[0] + 6.0, base_C[1], base_C[2]]
    r = st.primary([R + 8 for R in base_R], Cx, base_R, base_C, rr, n_perm=2000)
    assert not r["beats"] and r["verdict"].startswith("Not confirmed") and "cost" in r["verdict"]
    assert r["guards"]["return"] and not r["guards"]["cost"]
    # a large pooled gain that hides a loss of more than 0.10 r in one task: the return guard blocks it
    Rx = [base_R[0] + 20, base_R[1] + 20, base_R[2] - 3.0]
    r = st.primary(Rx, base_C, base_R, base_C, rr, n_perm=2000)
    assert r["p_perm"] < 0.025 and not r["guards"]["return"] and not r["beats"]
    # under-spending does not win: 12/5 against 14.5/25 has the lower arm-mean penalised score
    Rx = [np.full(16, 12.0) + np.linspace(-1, 1, 16) for _ in range(3)]
    Cx = [np.full(16, 5.0) + np.linspace(-1, 1, 16) for _ in range(3)]
    Ry = [R - 0.5 for R in base_R]
    r = st.primary(Rx, Cx, Ry, base_C, rr, n_perm=2000)
    assert not r["beats"] and r["ds"] < 0 and r["verdict"] == "Hurts"
    # three extra collapses also block the claim
    Rx = [R + 8 for R in base_R]
    Rx[0] = Rx[0].copy()
    Rx[0][:3] = 1.0
    g = st.guards(Rx, base_C, base_R, base_C, rr, [st.collapsed(R, 26.3) for R in Rx],
                  [st.collapsed(R, 26.3) for R in base_R])
    assert g["collapses_x"] == 3 and g["collapses_y"] == 0 and not g["collapse"]
    # a run flagged as diverged counts as collapsed whatever its endpoint
    div = [np.zeros(16, bool) for _ in range(3)]
    div[1][:3] = True
    r = st.primary([R + 8 for R in base_R], base_C, base_R, base_C, rr, diverged_x=div, n_perm=500)
    assert r["guards"]["collapses_x"] == 3 and not r["guards"]["collapse"] and not r["beats"]


def test_arm_mean_hinge_does_not_charge_cost_spread():
    """Review item 1: an arm whose runs straddle 25 with mean 25 pays no penalty; per run it would."""
    R = np.full(16, 15.0)
    C = np.r_[np.full(8, 10.0), np.full(8, 40.0)]                 # mean 25
    assert st.arm_score(R, C, 1.0) == pytest.approx(15.0)
    assert st.penalised(R, C).mean() == pytest.approx(15.0 - 7.5)  # the old per-run hinge
    # same means, lower spread: no difference in the primary statistic
    Cn = np.full(16, 25.0)
    assert st.hinge_diff([R], [Cn], [R], [C], [1.0]) == pytest.approx(0.0)
    # over the limit the arm pays exactly the excess of its mean
    assert st.arm_score(R, C + 4, 2.0) == pytest.approx((15.0 - 4.0) / 2.0)


def test_linearised_scores_reproduce_the_arm_score():
    rng = np.random.default_rng(0)
    for mu_c in (10.0, 24.0, 26.0, 40.0):
        R, C = rng.normal(15, 6, 16), rng.normal(mu_c, 10, 16)
        assert st.linearised(R, C, 26.3).mean() == pytest.approx(st.arm_score(R, C, 26.3), abs=1e-12)


def test_hinge_test_equals_the_per_run_test_below_the_limit():
    """With every arm mean cost below 25 for every split, the hinge test is the test of R / R_ref."""
    rng = np.random.default_rng(1)
    Rx, Ry = [rng.normal(16, 5, 12) for _ in range(3)], [rng.normal(15, 5, 12) for _ in range(3)]
    Cx, Cy = [rng.uniform(0, 20, 12) for _ in range(3)], [rng.uniform(0, 20, 12) for _ in range(3)]
    rr = [26.3, 30.0, 20.0]
    a = st.hinge_perm_test(Rx, Cx, Ry, Cy, rr, n_perm=3000, seed=4)
    b = st.strat_perm_test([R / r for R, r in zip(Rx, rr)], [R / r for R, r in zip(Ry, rr)], n_perm=3000, seed=4)
    assert a["stat"] == pytest.approx(b["stat"]) and a["p"] == pytest.approx(b["p"])


def test_hinge_permutation_recomputes_the_hinge_per_split():
    """Above the limit the split matters: full enumeration on a tiny case."""
    Rx, Cx = np.array([10.0, 12.0, 11.0]), np.array([20.0, 22.0, 24.0])
    Ry, Cy = np.array([13.0, 12.5, 14.0]), np.array([40.0, 35.0, 30.0])
    R, C = np.r_[Rx, Ry], np.r_[Cx, Cy]
    obs = st.hinge_diff([Rx], [Cx], [Ry], [Cy], [1.0])
    allst = []
    for comb in itertools.combinations(range(6), 3):
        m = np.zeros(6, bool)
        m[list(comb)] = True
        allst.append(st.hinge_diff([R[m]], [C[m]], [R[~m]], [C[~m]], [1.0]))
    exact = np.mean([v >= obs - 1e-12 for v in allst])
    r = st.hinge_perm_test([Rx], [Cx], [Ry], [Cy], [1.0], n_perm=60_000, seed=2)
    assert r["stat"] == pytest.approx(obs) and r["p"] == pytest.approx(exact, abs=0.006)


def test_non_finite_inputs_are_refused_not_scored():
    """Review item 3: one NaN cost in B* used to yield 'Beats B*' with p = 0.0005."""
    rr = [26.3] * 3
    R = [np.full(16, 15.0) + np.linspace(-1, 1, 16) for _ in range(3)]
    C = [np.full(16, 25.0) + np.linspace(-2, 2, 16) for _ in range(3)]
    Cy = [c.copy() for c in C]
    Cy[0][0] = np.nan
    with pytest.raises(ValueError):
        st.primary(R, C, R, Cy, rr, n_perm=200)
    with pytest.raises(ValueError):
        st.hinge_perm_test(R, C, R, Cy, rr, n_perm=200)
    with pytest.raises(ValueError):
        st.kill_b([12.0, np.nan, 12.0], [10.0, 10, 10], [14.0] * 6, [10.0] * 6)
    with pytest.raises(ValueError):
        st.rank([cfg("B1", [15, np.nan, 15], [20, 20, 20])])
    with pytest.raises(ValueError):
        st.selection_score([15, 15], [np.inf, 20])


def test_imputation_is_worst_case_for_the_candidate_and_median_for_missing_b_star():
    Rx, Cx = np.array([10.0, 12.0, np.nan, 11.0]), np.array([20.0, 22.0, 24.0, 21.0])
    Ry, Cy = np.array([13.0, 14.0, 15.0, np.nan]), np.array([30.0, np.nan, 26.0, 0.0])
    out = st.impute_task(Rx, Cx, Ry, Cy, done_y=[True, True, True, False])
    # candidate run 2 is non-finite: R 0, C = worst finite cost of the task in either arm (30)
    assert out["Rx"][2] == 0.0 and out["Cx"][2] == 30.0 and out["imputed_x"].tolist() == [False, False, True, False]
    # B* run 1 finished with a NaN cost: R 0, C 30 (a collapse); run 3 never finished: B*'s medians
    assert out["Ry"][1] == 0.0 and out["Cy"][1] == 30.0
    assert out["Ry"][3] == pytest.approx(np.median([13.0, 15.0])) and out["Cy"][3] == pytest.approx(28.0)
    assert out["missing_y"].tolist() == [False, False, False, True]
    assert st.collapsed(out["Ry"], 26.3).tolist() == [False, True, False, False]
    # an unfinished candidate run gets the worst case even if a partial value exists
    out = st.impute_task([10.0, 9.0], [5.0, 6.0], [12.0, 12.0], [20.0, 21.0], done_x=[True, False])
    assert out["Rx"][1] == 0.0 and out["Cx"][1] == 21.0
    # and the imputed study is scored, never silently passed
    rr = [26.3] * 3
    base = [np.full(8, 15.0) + np.linspace(-1, 1, 8) for _ in range(3)]
    costs = [np.full(8, 25.0) + np.linspace(-2, 2, 8) for _ in range(3)]
    Cy = [c.copy() for c in costs]
    Cy[0][0] = np.nan
    imp = [st.impute_task(base[t], costs[t], base[t], Cy[t]) for t in range(3)]
    r = st.primary([i["Rx"] for i in imp], [i["Cx"] for i in imp], [i["Ry"] for i in imp],
                   [i["Cy"] for i in imp], rr, n_perm=2000)
    assert not r["beats"] and r["p_perm"] > 0.025


def test_selection_imputation_flags_and_worst_cost():
    cs = st.impute_selection([cfg("A", [15, np.nan, 14], [20, 22, 21]), cfg("B", [10, 10, 10], [30, 31, np.nan])])
    assert cs[0]["R"].tolist() == [15, 0, 14] and cs[0]["C"].tolist() == [20, 31, 21]
    assert cs[0]["collapsed"].tolist() == [False, True, False] and cs[1]["collapsed"].tolist() == [False, False, True]
    assert st.rank(cs) == ["A", "B"]          # finite after imputation: A 9.67 - 0 vs B 10 - 5.67


def test_decision_function_hashes_cover_every_listed_function():
    h = st.source_hashes()
    assert set(h) == set(st.DECISION_FUNCTIONS) and all(len(v) == 64 for v in h.values())


def test_label_order():
    w = (-0.05, 0.05)
    assert st.label(w, w, w, w, (-0.3, -0.01), 0) == "Hurts"
    assert st.label((0.01, 0.2), (0.01, 0.3), w, w, (-0.1, 0.1), 0) == "Shift"
    assert st.label(w, w, (-0.09, 0.09), (-0.19, 0.19), (-0.1, 0.1), 0) == "Equivalent"
    assert st.label(w, w, (-0.2, 0.09), w, (-0.1, 0.1), 0) == "Inconclusive"
    assert st.label(w, w, w, w, (-0.1, 0.1), 3) == "Hurts"


def cfg(name, R, C, col=(False, False, False), hours=3.0):
    return {"name": name, "R": np.asarray(R, float), "C": np.asarray(C, float),
            "collapsed": np.asarray(col, bool), "hours": hours}


def test_selection_rank_eligibility_and_ties():
    configs = [cfg("B1", [15, 15, 15], [30, 30, 30]),             # score 10
               cfg("B2", [14, 14, 14], [20, 20, 20]),             # score 14
               cfg("B3", [14.3, 14.3, 14.3], [24, 24, 24]),       # 14.3, tied with B2 within 0.5
               cfg("B4", [30, 30, 1], [10, 10, 0], col=(False, True, True)),   # ineligible (2 of 3)
               cfg("S1", [14.2, 14.2, 14.2], [20, 20, 20], hours=2.4)]        # tied; same cost as B2, cheaper
    assert st.selection_score([15, 15], [30, 20]) == pytest.approx(15.0)     # hinge on the mean cost (25)
    assert st.selection_score([15, 15], [30, 30]) == pytest.approx(10.0)
    assert not st.eligible([False, True, True]) and st.eligible([False, False, True])
    assert not st.eligible([True] * 4 + [False] * 2) and st.eligible([True] * 3 + [False] * 3)
    order = st.rank(configs)
    assert order[:3] == ["S1", "B2", "B3"] and order[-1] == "B1" and "B4" not in order
    assert st.rank(configs, eligible_only=False) == order + ["B4"]           # ineligible ones go last


def test_kill_rule_b_uses_two_standard_errors_with_a_floor():
    base = np.array([14.0, 14.0, 14.0, 14.0, 14.0, 14.0])
    low = np.full(6, 10.0)                                     # costs below the limit: score = return
    r = st.kill_b([12.0, 12.0, 12.0], low[:3], base, low)      # SDs 0 -> floored at 3
    assert r["se"] == pytest.approx(math.sqrt(9 / 3 + 9 / 6)) and not r["kill"]
    assert st.kill_b([9.0, 9.5, 10.0], low[:3], base, low)["kill"]   # 4.5 below, SE 2.12
    noisy = st.kill_b([2.0, 12.0, 22.0], low[:3], np.r_[base, base] + np.linspace(-6, 6, 12), np.full(12, 10.0))
    assert not noisy["kill"]
    # over the limit the linearised score carries the cost: 14/35 scores 4, so 12/20 is not killed
    r = st.kill_b([12.0, 12.0, 12.0], [20.0, 20, 20], base, np.full(6, 35.0))
    assert r["diff"] == pytest.approx(12.0 - 4.0) and not r["kill"]
