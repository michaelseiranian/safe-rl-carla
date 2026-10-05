"""lagu.analyze: statistics against hand-computed cases, and every mode on small synthetic runs."""
import csv
import itertools
import json
import math
import os

import numpy as np
import pytest

from lagu import analyze as an

ENV = "SafetyPointGoal1-v0"
OLD_FIELDS = ["step", "episodes", "train_return", "train_cost", "eval_return", "eval_cost", "eval_violation",
              "lambda", "engagement", "n_explore", "n_risk", "explore_prob", "qbar", "qstd", "qc", "delta_mean",
              "ratio_p50", "ratio_p90", "q_loss", "c_loss", "sps"]   # the pilot's schema, before CC1


def write_run(d, arm, seed, rows, tag="", steps=None, env=ENV, cfg=None, heldout=None):
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(str(d), f"{arm}{'-' + tag if tag else ''}_{env}_s{seed}")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    args = {"env": env, "arm": arm, "seed": seed, "steps": steps or int(rows[-1]["step"]), "tag": tag,
            "start_steps": 10_000}
    with open(stem + ".json", "w") as f:
        json.dump({"args": args, "cfg": {"delta0": 2.5, "lambda_lr": 1e-5, **(cfg or {})},
                   "horizon": 1000, "obs_dim": 60, "act_dim": 2}, f)
    if heldout is not None:
        with open(stem + ".heldout.json", "w") as f:
            json.dump(heldout, f)
    return stem


def held(ret, cost, end=1_000_000, qc=5.0, mc=10.0, n=5):
    """The schema lagu.evaluate writes."""
    per = [{"index": i, "target": end * (0.8 + 0.05 * i), "step": int(end * (0.8 + 0.05 * i)),
            "eval_return": ret, "eval_cost": cost, "eval_violation": 0.0, "qc_traj": qc, "mc_cost_traj": mc}
           for i in range(n)]
    return {"stem": "x", "end_step": end, "snapshots": n, "episodes": 20, "per_snapshot": per,
            "pooled": {"eval_return": ret, "eval_cost": cost, "eval_violation": 0.0, "qc_traj": qc, "mc_cost_traj": mc}}


def rows_at(steps, **cols):
    """One row per step; a column value may be a constant or a function of the step."""
    return [{"step": s, **{k: (v(s) if callable(v) else v) for k, v in cols.items()}} for s in steps]


def patch_csv(stem, fn):
    """Rewrite <stem>.csv with fn(row) for every row (values are strings); None drops the row.
    fn must give every row the same keys."""
    with open(str(stem) + ".csv") as f:
        rows = [r for r in map(fn, csv.DictReader(f)) if r is not None]
    with open(str(stem) + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def at_step(step, **vals):
    """patch_csv function: set vals in the row at `step`."""
    return lambda r: {**r, **vals} if int(float(r["step"])) == step else r


# ---------------------------------------------------------------- statistics

def brute_wilcoxon(d, alternative):
    d = [x for x in d if x != 0]
    r = an.midranks(np.abs(d))
    w = sum(ri for ri, x in zip(r, d) if x > 0)
    mu = r.sum() / 2
    sums = [sum(ri for ri, s in zip(r, signs) if s) for signs in itertools.product((0, 1), repeat=len(d))]
    if alternative == "greater":
        return np.mean([s >= w - 1e-9 for s in sums])
    if alternative == "less":
        return np.mean([s <= w + 1e-9 for s in sums])
    return np.mean([abs(s - mu) >= abs(w - mu) - 1e-9 for s in sums])


def test_wilcoxon_exact_matches_hand_computed_cases():
    # n = 3, all positive: W+ = 6 is the single most extreme of 8 sign patterns on each side
    assert an.wilcoxon([1, 2, 3])["p"] == pytest.approx(2 / 8)
    assert an.wilcoxon([1, 2, 3], "greater")["p"] == pytest.approx(1 / 8)
    assert an.wilcoxon([1, 2, 3], "less")["p"] == pytest.approx(1.0)
    # n = 4, W+ = 8 (mean 5): W+ in {0, 1, 2, 8, 9, 10} is 6 of 16 patterns; W+ >= 8 is 3 of 16
    assert an.wilcoxon([1, -2, 3, 4])["p"] == pytest.approx(6 / 16)
    assert an.wilcoxon([1, -2, 3, 4], "greater")["p"] == pytest.approx(3 / 16)
    # ties get midranks (1.5, 1.5, 3); zeros are dropped
    assert an.wilcoxon([1, 1, 2])["p"] == pytest.approx(2 / 8)
    assert an.wilcoxon([1, 1, -2])["p"] == pytest.approx(1.0)
    assert an.wilcoxon([0, 1, 2, 3])["p"] == pytest.approx(2 / 8)
    assert an.wilcoxon([0, 0])["n"] == 0 and math.isnan(an.wilcoxon([])["p"])
    rng = np.random.default_rng(3)
    for n in (5, 8):
        d = np.round(rng.normal(0.3, 1, n), 1)           # rounding creates ties
        for alt in ("two-sided", "greater", "less"):
            assert an.wilcoxon(d, alt)["p"] == pytest.approx(brute_wilcoxon(d, alt))
    big = an.wilcoxon(np.arange(1, 22))
    assert not big["exact"] and 0 < big["p"] < 1e-3


def test_fisher_exact_matches_hand_computed_cases():
    assert an.fisher_exact(3, 0, 0, 3) == pytest.approx(2 / 20)          # 2 * 1 / C(6, 3)
    assert an.fisher_exact(0, 3, 3, 0) == pytest.approx(2 / 20)
    assert an.fisher_exact(1, 9, 1, 9) == pytest.approx(1.0)
    # margins 10/6 and 9/7: P(a) proportional to C(10, a) C(6, 9 - a); tables a in {3, 8, 9} are
    # no more likely than a = 8, giving (120 + 270 + 10) / C(16, 9)
    assert an.fisher_exact(8, 2, 1, 5) == pytest.approx(400 / 11440)
    assert an.fisher_exact(0, 0, 0, 0) == 1.0


def test_holm_step_down():
    np.testing.assert_allclose(an.holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
    np.testing.assert_allclose(an.holm([0.5, 0.6]), [1.0, 1.0])
    np.testing.assert_allclose(an.holm([0.03]), [0.03])


def test_paired_bootstrap_is_seeded_and_resamples_columns_jointly():
    d = np.column_stack([np.arange(8.0), 2 * np.arange(8.0)])
    a, b = an.paired_bootstrap(d, 10_000, seed=7), an.paired_bootstrap(d, 10_000, seed=7)
    np.testing.assert_array_equal(a[95], b[95])
    assert not np.array_equal(a[95], an.paired_bootstrap(d, 10_000, seed=8)[95])
    np.testing.assert_allclose(a[95][1], 2 * a[95][0])          # same resample for both columns
    assert a[95][0][0] <= a[90][0][0] <= a[90][0][1] <= a[95][0][1]
    np.testing.assert_array_equal(an.paired_bootstrap([3.0, 3.0, 3.0])[95], [[3.0, 3.0]])
    assert np.isnan(an.paired_bootstrap([1.0])[95]).all()


@pytest.mark.parametrize("ret95, cost95, ret90, cost90, ca, cb, label", [
    ((-2, 1), (-8, -1), (-1, 1), (-7, -2), 0, 0, "Helps"),        # cost down, return not worse by 2.5
    ((1, 3), (1, 4), (1, 3), (1, 4), 0, 0, "Helps"),               # also a Shift, but Helps comes first
    ((1, 3), (-3, 4), (1, 3), (-3, 4), 3, 0, "Helps"),             # Helps before the collapse clause
    ((-3, -1), (-4, -1), (-3, -1), (-4, -1), 0, 0, "Hurts"),       # mirror of Helps, before Shift
    ((-1, 2), (1, 8), (-1, 2), (1, 8), 0, 0, "Hurts"),             # cost up, return not better by 2.5
    ((-1, 1), (-1, 1), (-1, 1), (-1, 1), 4, 1, "Hurts"),           # 3 more collapses, before Equivalent
    ((-1, 1), (-1, 1), (-1, 1), (-1, 1), 3, 1, "Equivalent"),      # only 2 more collapses
    ((0.5, 4), (1, 6), (0.5, 4), (1, 6), 0, 0, "Shift"),           # return up by >= 2.5, else Hurts
    ((-4, -1), (-8, -1), (-4, -1), (-8, -1), 0, 0, "Shift"),
    ((-3, 3), (-6, 6), (-2, 2), (-4, 4), 0, 0, "Equivalent"),      # 90% inside the margins
    ((-3, 3), (-6, 6), (-3, 2), (-4, 4), 0, 0, "Inconclusive"),
    ((math.nan,) * 2, (math.nan,) * 2, (math.nan,) * 2, (math.nan,) * 2, 0, 0, "Inconclusive"),
])
def test_verdict_precedence(ret95, cost95, ret90, cost90, ca, cb, label):
    assert an.verdict(ret95, cost95, ret90, cost90, ca, cb) == label


def test_probability_of_improvement_and_batched_iqm():
    # pairs (x, y): (1,0) (1,2) (2,0) (2,2) (3,0) (3,2) -> 1 + 0 + 1 + 0.5 + 1 + 1 = 4.5 of 6
    assert an.prob_improvement([1, 2, 3], [0, 2]) == pytest.approx(0.75)
    assert an.prob_improvement([1, 2, 3], [0, 2], sign=-1) == pytest.approx(0.25)
    m = np.random.default_rng(0).normal(size=(6, 9))
    np.testing.assert_allclose(an.prob_improvement(m, m[::-1]), [an.prob_improvement(a, b) for a, b in zip(m, m[::-1])])
    for n in range(1, 10):
        np.testing.assert_allclose(an._iqm_rows(m[:, :n]), [an.iqm(r) for r in m[:, :n]])


# ---------------------------------------------------------------- loading and the default table

def test_old_csv_without_new_columns_loads_and_the_default_table_runs(tmp_path, capsys):
    d = tmp_path / "pilot"
    for arm in ("lagu", "lagu_nogate"):
        for s in (0, 1):
            rows = rows_at([50_000, 100_000], **{k: 1.0 for k in OLD_FIELDS if k != "step"})
            write_run(d, arm, s, rows, steps=1_000_000)
    runs = an.load(str(d))
    assert len(runs) == 4 and set(runs[0]["rows"][0]) == set(OLD_FIELDS)
    e = an.endpoints(runs[0], end=100_000)
    assert e["window_return"] == 1.0 and e["engagement"] == 1.0
    assert all(math.isnan(e[k]) for k in ("lambda_avg", "cum_cost", "would_engage", "qc_avg", "calib"))
    assert e["resumed"] is False
    an.cli([str(d)])
    out = capsys.readouterr().out
    assert "lagu_nogate" in out and os.path.exists(d / f"curves_{ENV}.png")
    res = an.pair(runs, "lagu", "lagu_nogate", end=100_000, n_boot=200)
    assert res[ENV]["verdict"] == "Equivalent" and res[ENV]["source"] == "eval rows with step >= 0.8N"


def test_default_plots_include_the_new_panels(tmp_path):
    d = tmp_path / "new"
    for s in (0, 1):
        write_run(d, "lagu", s, rows_at([50_000, 100_000], eval_return=1.0, eval_cost=2.0, **{"lambda": 0.1},
                                        engagement=0.2, lambda_avg=0.1, lambda_shadow=0.05, would_engage=0.3,
                                        qc_avg=1.0, qc_traj=1.0, mc_cost_traj=1.5, delta_avg=2.4, delta_risk_avg=2.0))
    an.main(str(d))
    assert os.path.exists(d / f"curves_{ENV}.png")


# ---------------------------------------------------------------- budget D

def test_budget_D_weights_by_n_pi_and_prints_only_the_number(tmp_path, capsys):
    d = tmp_path / "gate"
    write_run(d, "lagu", 100, [{"step": 10_000, "delta_avg": "", "n_pi": 0, "engagement": 0.0, "n_risk": 0},
                               {"step": 50_000, "delta_avg": 2.0, "n_pi": 1, "engagement": 0.2, "n_risk": 10},
                               {"step": 100_000, "delta_avg": 2.2, "n_pi": 3, "engagement": 0.4, "n_risk": 30}], tag="tt")
    write_run(d, "lagu", 101, [{"step": 50_000, "delta_avg": 2.5, "n_pi": 4, "engagement": 0.0, "n_risk": 60},
                               {"step": 100_000, "delta_avg": 9.9, "n_pi": "", "engagement": 1.0, "n_risk": ""}], tag="tt")
    D = (2.0 * 1 + 2.2 * 3 + 2.5 * 4) / (1 + 3 + 4)
    r = an.budget_D(an.load(str(d)), "lagu-tt")
    assert r["D"] == pytest.approx(D) and r["seeds"] == [100, 101]
    assert r["engagement"] == pytest.approx((0.2 * 10 + 0.4 * 30) / 100) and r["g2_branch"] == "enqueue E2"
    an.cli([str(d), "--budget-D", "lagu-tt"])
    cap = capsys.readouterr()
    assert cap.out.strip() == f"{D:.6g}" and "G2 branch" in cap.err
    write_run(d, "lagu", 100, [{"step": 50_000, "delta_avg": 2.0, "n_pi": 1}], tag="tt", env="SafetyCarGoal1-v0")
    with pytest.raises(SystemExit):
        an.budget_D(an.load(str(d)), "lagu-tt")
    assert an.budget_D(an.load(str(d)), "lagu-tt", env=ENV)["D"] == pytest.approx(D)


# ---------------------------------------------------------------- E0 rule

E0_STEPS = range(50_000, 1_000_001, 50_000)


def e0_cell(d, tag, lr, ret=25.0, calib=1.0, lam=0.05, lam600=None, gap=0.1, cost=30.0,
            tc_early=40.0, tc_late=40.0, qc_override=None, ext_ret=None):
    """Three seeds of one E0 cell. Values outside the conditions' windows are chosen so that a
    wrong window would change the outcome. ext_ret: per-seed returns of E0-ext rows to 2M."""
    for s in (0, 1, 2):
        r = ret[s] if isinstance(ret, (list, tuple)) else ret
        rows = []
        for st in (E0_STEPS if ext_ret is None else range(50_000, 2_000_001, 50_000)):
            win = 850_000 <= st <= 1_000_000
            rows.append({"step": st, "eval_return": r if win else ext_ret[s] if st >= 1_600_000 else 0.0,
                         "eval_cost": cost if win else 99.0,
                         "train_cost": tc_late if win else tc_early if 250_000 <= st <= 400_000 else 99.0,
                         "qc_traj": (qc_override or {}).get((s, st), calib * 10 if win else -5.0),
                         "mc_cost_traj": 10.0, "gap_avg": gap if win else 9.0,
                         "lambda_avg": lam600 if (st == 600_000 and lam600 is not None) else lam,
                         "n_pi": 25_000, "n_risk": 500, "would_engage": 0.2, "qbar_avg": 0.8})
        write_run(d, "lagu_nogate", s, rows, tag=tag, steps=1_000_000, cfg={"lambda_lr": lr})


def e0(d):
    return an.e0_rule(an.load(str(d)))


CELLS = (("tt-l1e-5", 1e-5), ("tt-l2e-6", 2e-6), ("tt-l4e-7", 4e-7), ("omsw-l4e-7", 4e-7))


def test_e0_outcome_i_picks_the_fastest_passing_tt_cell_and_1M(tmp_path, capsys):
    for tag, lr in CELLS:
        e0_cell(tmp_path, tag, lr, ret=2.0 if tag == "tt-l1e-5" else 25.0)
    r = e0(tmp_path)
    assert r["cells"]["lagu_nogate-tt-l1e-5"]["b"] is False
    assert (r["outcome"], r["chosen"], r["steps"]) == ("i", "lagu_nogate-tt-l2e-6", 1_000_000)
    an.cli([str(tmp_path), "--e0-rule"])
    with open(tmp_path / "e0_decision.json") as f:
        saved = json.load(f)
    assert saved["outcome"] == "i" and saved["lambda_lr"] == 2e-6 and saved["complete"] is True
    assert saved["cells"]["lagu_nogate-tt-l2e-6"]["window_steps"][0] == [850_000, 900_000, 950_000, 1_000_000]


def test_e0_outcome_ii_iii_iv_v(tmp_path):
    d = tmp_path / "ii"
    for tag, lr in CELLS:
        e0_cell(d, tag, lr, calib=0.3 if tag.startswith("tt") else 1.0)
    assert (e0(d)["outcome"], e0(d)["chosen"]) == ("ii", "lagu_nogate-omsw-l4e-7")
    d = tmp_path / "iii"                                  # all healthy, none binds
    for tag, lr in CELLS:
        e0_cell(d, tag, lr, lam=0.001, cost=50.0)
    r = e0(d)
    assert (r["outcome"], r["chosen"], r["steps"]) == ("iii", "lagu_nogate-tt-l1e-5", 2_000_000)
    assert r["e0_ext"]["complete"] is False and r["e0_ext"]["move_window_to_end_step"] is False
    e0_cell(d, "tt-l1e-5", 1e-5, lam=0.001, cost=50.0, ext_ret=[2.0, 4.0, 30.0])    # E0-ext resumed to 2M
    r = e0(d)
    assert (r["outcome"], r["chosen"]) == ("iii", "lagu_nogate-tt-l1e-5")          # decided on rows <= 1M
    assert r["e0_ext"] == {"seed_mean_return": [2.0, 4.0, 30.0], "complete": True, "move_window_to_end_step": True}
    d = tmp_path / "iv"                                   # fast cells bind but collapse; slow ones never bind
    for tag, lr in CELLS:
        fast = lr > 1e-6
        e0_cell(d, tag, lr, ret=2.0 if fast else 25.0, lam=0.05 if fast else 0.001, cost=30.0 if fast else 50.0)
    r = e0(d)
    assert (r["outcome"], r["chosen"], r["steps"]) == ("iv", "lagu_nogate-tt-l4e-7", 2_000_000)   # tie -> tt
    d = tmp_path / "v"
    for tag, lr in CELLS:
        e0_cell(d, tag, lr, calib=0.3)
    r = e0(d)
    assert (r["outcome"], r["chosen"], r["steps"]) == ("v", None, None)
    d = tmp_path / "v2"                                   # calibrated, but every cell collapses and none binds
    for tag, lr in CELLS:
        e0_cell(d, tag, lr, ret=2.0, lam=0.001, cost=50.0)
    assert e0(d)["outcome"] == "v"


def test_e0_conditions_edge_cases(tmp_path):
    # (b): cell mean 22.7 >= 20, but seed 2 is below 10
    e0_cell(tmp_path / "b", "tt-l1e-5", 1e-5, ret=[30.0, 30.0, 8.0])
    assert e0(tmp_path / "b")["cells"]["lagu_nogate-tt-l1e-5"]["b"] is False
    # (a): pooled 109/120 and seed ratio 29/40 are fine, but one window row has qc_traj < 0
    e0_cell(tmp_path / "a", "tt-l1e-5", 1e-5, qc_override={(1, 1_000_000): -1.0})
    c = e0(tmp_path / "a")["cells"]["lagu_nogate-tt-l1e-5"]
    assert c["calib"] == pytest.approx(109 / 120) and c["a"] is False
    # (c) through the lambda rise (|gap| too large) and (d) through the train-cost drop (eval cost
    # too high); lambda is not stable, so the budget is 2M
    e0_cell(tmp_path / "cd", "tt-l1e-5", 1e-5, gap=1.0, lam=0.05, lam600=0.03, cost=50.0, tc_early=45.0, tc_late=30.0)
    r = e0(tmp_path / "cd")
    c = r["cells"]["lagu_nogate-tt-l1e-5"]
    assert c["c"] and c["d"] and (r["outcome"], r["steps"]) == ("i", 2_000_000)
    e0_cell(tmp_path / "c2", "tt-l1e-5", 1e-5, gap=1.0, lam=0.05, lam600=0.045)       # 0.05 < 1.2 * 0.045
    assert e0(tmp_path / "c2")["cells"]["lagu_nogate-tt-l1e-5"]["c"] is False


# ---------------------------------------------------------------- paired contrast

STEPS = (200_000, 400_000, 600_000, 800_000, 1_000_000)


def gate_runs(d, seeds=range(100, 105), heldouts=True, tag="tt", a="lagu", b="lagu_nogate", end=1_000_000):
    """A beats B on the held-out endpoint (Helps); the in-training eval rows are identical."""
    for s in seeds:
        k = s - 100
        for arm, ret, cost in ((a, 23 + k + 0.1 * k, 34 + 2 * k - 0.2 * k), (b, 20 + k, 40 + 2 * k)):
            rows = rows_at(STEPS, eval_return=20.0 + k, eval_cost=40.0, qc_traj=5.0, mc_cost_traj=10.0,
                           cum_cost=lambda st: st / 1000, lambda_avg=0.05 if arm == a else 0.03, n_pi=1000,
                           engagement=0.4, n_risk=100, resumed=0, gap_avg=0.1)
            write_run(d, arm, s, rows, tag=tag, heldout=held(ret, cost, end) if heldouts else None)


def test_pair_heldout_endpoint_verdict_and_secondaries(tmp_path, capsys):
    gate_runs(tmp_path)
    res = an.pair(an.load(str(tmp_path)), "lagu-tt", "lagu_nogate-tt")[ENV]
    assert res["source"] == "heldout" and res["n_pairs"] == 5 and res["n_left_out"] == 0
    assert res["return"]["diff"] == pytest.approx(3.2) and res["cost"]["diff"] == pytest.approx(-6.4)
    assert 3.0 <= res["return"]["ci95"][0] <= res["return"]["ci95"][1] <= 3.4
    assert res["return"]["wilcoxon_p"] == pytest.approx(2 / 32) and res["verdict"] == "Helps"
    assert res["collapses_a"] == res["collapses_b"] == 0 and res["fisher_p"] == 1.0
    assert res["sensitivity"] is None
    sec = res["secondary"]
    assert sec["window_return"]["diff"] == 0.0 and sec["cum_cost"]["mean_a"] == 1000.0
    assert sec["cum_cost_500k"]["mean_a"] == 400.0 and sec["lambda_avg"]["diff"] == pytest.approx(0.02)
    assert sec["calib"]["mean_a"] == pytest.approx(0.5)
    again = an.pair(an.load(str(tmp_path)), "lagu-tt", "lagu_nogate-tt")[ENV]
    assert again["return"]["ci95"] == res["return"]["ci95"]          # seeded bootstrap
    an.cli([str(tmp_path), "--pair", "lagu-tt", "lagu_nogate-tt"])
    assert "verdict: Helps" in capsys.readouterr().out


def test_pair_falls_back_to_eval_rows_without_heldout_or_for_another_end_step(tmp_path):
    gate_runs(tmp_path)
    os.remove(tmp_path / f"lagu-tt_{ENV}_s102.heldout.json")
    res = an.pair(an.load(str(tmp_path)), "lagu-tt", "lagu_nogate-tt")[ENV]
    assert res["source"].startswith("eval rows") and res["missing_heldout"] == [f"lagu-tt_{ENV}_s102"]
    assert res["return"]["diff"] == 0.0 and res["verdict"] == "Equivalent"
    gate_runs(tmp_path)
    res = an.pair(an.load(str(tmp_path)), "lagu-tt", "lagu_nogate-tt", end=600_000)[ENV]
    assert res["source"].startswith("eval rows") and res["N"] == [600_000.0]
    assert res["secondary"]["cum_cost"]["mean_a"] == 600.0


def test_pair_sensitivity_drops_resumed_pairs_and_nan_counts_as_collapse(tmp_path):
    gate_runs(tmp_path)
    stem = str(tmp_path / f"lagu-tt_{ENV}_s101")
    with open(stem + ".csv") as f:
        rows = list(csv.DictReader(f))
    rows[-1]["resumed"] = "1"
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(tmp_path / f"lagu-tt_{ENV}_s104.heldout.json", "w") as f:
        json.dump(held(float("nan"), 30.0), f)
    res = an.pair(an.load(str(tmp_path)), "lagu-tt", "lagu_nogate-tt")[ENV]
    assert res["n_pairs"] == 5 and res["n_left_out"] == 1 and res["collapses_a"] == 1
    assert res["resumed_seeds"] == [101]
    s = res["sensitivity"]
    assert s["n_pairs"] == 4 and s["seeds"] == [100, 102, 103, 104] and s["n_left_out"] == 1


# ---------------------------------------------------------------- H2, H5, unpaired, prereg

def test_h2_engagement_window_and_one_sided_wilcoxon(tmp_path):
    for s in range(100, 105):
        k = s - 100
        # the rows at 25k and 350k lie outside 25k < step <= 300k and would dominate if counted
        eng = rows_at([25_000, 50_000, 300_000, 350_000], engagement=lambda st: 0.5 if 25_000 < st <= 300_000 else 1.0,
                      n_risk=lambda st: 100 if 25_000 < st <= 300_000 else 10 ** 6)
        write_run(tmp_path, "lagu", s, [{**r, "lambda_avg": 0.1 if i < 2 else 0.2 + k / 100, "n_pi": 1 if i < 2 else 3}
                                        for i, r in enumerate(eng)], tag="tt")
        write_run(tmp_path, "lagu_nogate", s, rows_at([50_000, 300_000], engagement=0.0, n_risk=100,
                                                      lambda_avg=0.1, n_pi=1), tag="tt")
    runs = an.load(str(tmp_path))
    assert an.lambda_time(runs[0]) == pytest.approx((0.1 * 2 + 0.2 * 6) / 8)
    r = an.h2(runs, "lagu-tt", "lagu_nogate-tt")[ENV]
    assert r["engagement_a"] == pytest.approx(0.5) and r["p_one_sided"] == pytest.approx(1 / 32)
    assert r["status"] == "confirmed" and r["median_diff"] > 0
    assert an.h2_status(0.05, 1.0, 0.01) == "refuted_vacuous"
    assert an.h2_status(0.5, 0.0, 0.01) == "refuted_lambda"
    assert an.h2_status(0.5, 1.0, 0.2) == "not_confirmed"
    assert an.h2_status(float("nan"), 1.0, 0.01) == "undetermined"


def lit_runs(d, seeds=range(100, 105)):
    for s in seeds:
        k = s - 100
        for arm in ("lagu", "lagu_nogate"):
            write_run(d, arm, s, rows_at(STEPS, gap_avg=0.1, n_pi=1000, lambda_avg=0.001 * (2 if arm == "lagu" else 1) + k * 1e-4,
                                         engagement=0.3 if arm == "lagu" else 0.0, n_risk=100, eval_return=5.0),
                      tag="lit", heldout=held(10.0, 40.0 + 2 * k + (1 if arm == "lagu" else 0), qc=3.0, mc=10.0))


def test_h5_literal_recipe_claims(tmp_path, capsys):
    lit_runs(tmp_path)
    res = an.h5(an.load(str(tmp_path)), ["lagu-lit", "lagu_nogate-lit"])
    r = res[f"{ENV} lagu_nogate-lit"]
    assert r["source"] == "heldout" and r["cost_iqm"] == pytest.approx(44.0)       # mean of 42, 44, 46
    assert r["cost_iqm_ci95"][0] > 25 and r["p_cost_above_25"] == pytest.approx(1 / 32)
    assert r["calib"] == pytest.approx(0.3) and r["calib_ci95"] == [pytest.approx(0.3)] * 2
    assert r["gap"] == pytest.approx(0.1) and r["delta0"] == 2.5
    assert r["confirmed"] and not r["refuted"]
    an.cli([str(tmp_path), "--pair", "lagu-lit", "lagu_nogate-lit", "--h5"])
    assert capsys.readouterr().out.count("-> confirmed") == 2


def test_unpaired_iqm_and_probability_of_improvement(tmp_path):
    for i, s in enumerate(range(100, 105)):
        write_run(tmp_path, "lagu", s, rows_at(STEPS, eval_return=1.0), tag="tt", heldout=held(30.0 + i, 10.0 + i))
        write_run(tmp_path, "td3lag", s + 10, rows_at(STEPS, eval_return=1.0), tag="tt", heldout=held(20.0 + i, 20.0 + i))
    r = an.unpaired(an.load(str(tmp_path)), "lagu-tt", "td3lag-tt", n_boot=500)[ENV]
    assert r["source"] == "heldout" and r["return"]["iqm_a"] == pytest.approx(32.0)
    assert r["return"]["prob_improvement"] == 1.0 and r["return"]["ci_prob_improvement"] == [1.0, 1.0]
    assert r["cost"]["prob_improvement"] == 1.0 and r["cost"]["diff"] == pytest.approx(-10.0)


def test_prereg_holm_family_seed_sets_and_json(tmp_path, capsys):
    gate, lit = tmp_path / "gate", tmp_path / "gate_literal"
    gate_runs(gate, seeds=range(100, 106))
    gate_runs(gate, seeds=[115])                          # an extension pair: H3 uses it, H2 must not
    for s in list(range(100, 106)) + [115]:
        for arm, lam in (("lagu", 0.05 if s != 115 else 0.0), ("lagu_nogate", 0.03)):
            stem = str(gate / f"{arm}-tt_{ENV}_s{s}")
            with open(stem + ".csv") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                r["lambda_avg"] = lam + (s - 100) * 1e-3
            with open(stem + ".csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    lit_runs(lit)
    pre = tmp_path / "prereg.md"
    pre.write_text("# prereg\n")
    an.cli(["--prereg", str(pre), str(gate), str(lit)])
    with open(tmp_path / "prereg_results.json") as f:
        r = json.load(f)
    fam = r["holm"]
    assert set(fam) == {"H2_E1", "H2_E1b", "H3_return", "H3_cost", "H5_lagu-lit", "H5_lagu_nogate-lit"}   # no E2
    names = list(fam)
    np.testing.assert_allclose([fam[k]["p_holm"] for k in names], an.holm([fam[k]["p"] for k in names]))
    assert r["H2"]["E1"]["seeds"] == list(range(100, 106)) and r["H2"]["E1"]["p_one_sided"] == pytest.approx(1 / 64)
    assert r["H3"]["n_pairs"] == 7 and r["H3"]["verdict"] == "Helps" and r["H3"]["status"] == "refuted"
    assert r["H4"]["status"].startswith("missing")
    assert r["H5"]["E1_nogate_calibration"]["in_band"] is False       # E1 nogate held-out calib is 0.5
    assert r["H5"]["status"] == "partially_supported"
    assert r["prereg_sha256"] and r["analyze_sha256"]
    assert "Holm family" in capsys.readouterr().out


# ---------------------------------------------------------------- regressions (review of the prereg rules)

def test_all_zero_differences_give_wilcoxon_p_one_and_stay_in_the_holm_family(tmp_path):
    """prereg 6: 'If no difference is non-zero, p = 1.' Bit-identical arms (a gate that never fires)
    must not drop H3 from the Holm family or leave H2 undetermined."""
    assert an.wilcoxon([0.0, 0.0, 0.0])["p"] == 1.0 and an.wilcoxon([0.0], "greater")["p"] == 1.0
    assert math.isnan(an.wilcoxon([float("nan")])["p"])                 # no finite difference: no data
    for s in range(100, 105):
        for arm in ("lagu", "lagu_nogate"):
            write_run(tmp_path, arm, s, rows_at(STEPS, eval_return=20.0, lambda_avg=0.05, n_pi=100,
                                                engagement=0.5, n_risk=10), tag="tt", heldout=held(20.0 + s, 30.0))
    runs = an.load(str(tmp_path))
    res = an.pair(runs, "lagu-tt", "lagu_nogate-tt", n_boot=200)[ENV]
    assert res["return"]["wilcoxon_p"] == res["cost"]["wilcoxon_p"] == 1.0 and res["verdict"] == "Equivalent"
    h = an.h2(runs, "lagu-tt", "lagu_nogate-tt")[ENV]
    assert h["p_one_sided"] == 1.0 and h["status"] == "refuted_lambda"
    pre = tmp_path / "prereg.md"
    pre.write_text("x")
    r = an.prereg(runs, str(pre), n_boot=200)
    assert {"H2_E1", "H3_return", "H3_cost"} <= set(r["holm"]) and r["H3"]["status"] == "confirmed"


def test_budget_D_without_data_fails_instead_of_printing_nan(tmp_path, capsys):
    """D=$(... --budget-D lagu-tt) must not hand delta0=nan to the E2 launch."""
    write_run(tmp_path, "lagu", 100, rows_at([50_000, 100_000], engagement=0.2, n_risk=10), tag="tt")   # old schema
    with pytest.raises(SystemExit) as ex:
        an.cli([str(tmp_path), "--budget-D", "lagu-tt"])
    assert ex.value.code not in (None, 0) and capsys.readouterr().out == ""


def test_interval_columns_use_rows_after_0_8N(tmp_path):
    """prereg 5 and H5: interval columns over the last 20% use step > 0.8N; the row at 0.8N covers
    the updates before 0.8N."""
    for s in range(100, 105):
        write_run(tmp_path, "lagu", s, rows_at(STEPS, gap_avg=lambda st: 9.0 if st == 800_000 else 0.1,
                                               qc_avg=lambda st: 9.0 if st == 800_000 else 2.0,
                                               would_engage=lambda st: 1.0 if st == 800_000 else 0.25,
                                               n_pi=1000, n_risk=100, eval_return=5.0),
                  tag="lit", heldout=held(10.0, 40.0, qc=3.0))
    runs = an.load(str(tmp_path))
    e = an.endpoints(runs[0])
    assert (e["gap_avg"], e["qc_avg"], e["would_engage"]) == (pytest.approx(0.1), 2.0, 0.25)
    assert an.h5(runs, ["lagu-lit"], n_boot=200)[f"{ENV} lagu-lit"]["parts"]["gap_within_quarter_delta0"]


def test_e0_nonfinite_values_fail_the_condition_that_reads_them(tmp_path):
    """prereg 10.2: a non-finite value that a condition reads fails it; a diverged seed fails (b);
    rows are chosen by step (850k-1M), not as the last 4 rows."""
    cell = "lagu_nogate-tt-l1e-5"
    stem = lambda d, s: d / f"{cell}_{ENV}_s{s}"
    abcd = lambda d: [e0(d)["cells"][cell][k] for k in "abcd"]
    e0_cell(tmp_path / "ok", "tt-l1e-5", 1e-5)
    assert abcd(tmp_path / "ok") == [True] * 4
    e0_cell(tmp_path / "a", "tt-l1e-5", 1e-5)
    patch_csv(stem(tmp_path / "a", 1), at_step(900_000, qc_traj="nan"))
    assert abcd(tmp_path / "a")[0] is False                     # used to drop the row and pass
    e0_cell(tmp_path / "c", "tt-l1e-5", 1e-5, gap=1.0, lam600=0.03)   # (c) through the lambda rise
    assert abcd(tmp_path / "c")[2] is True
    patch_csv(stem(tmp_path / "c", 2), at_step(1_000_000, gap_avg="nan"))
    assert abcd(tmp_path / "c") == [True, True, False, True]
    e0_cell(tmp_path / "d", "tt-l1e-5", 1e-5)
    patch_csv(stem(tmp_path / "d", 0), at_step(950_000, eval_cost="nan"))
    assert abcd(tmp_path / "d")[3] is False
    e0_cell(tmp_path / "b", "tt-l1e-5", 1e-5)                   # q_loss NaN at 500k: seed 0 diverged
    patch_csv(stem(tmp_path / "b", 0), lambda r: {**r, "q_loss": "nan" if r["step"] == "500000" else "1.0"})
    c = e0(tmp_path / "b")["cells"][cell]
    assert [c[k] for k in "abcd"] == [True, False, True, True] and c["diverged_seeds"] == [0]
    e0_cell(tmp_path / "inc", "tt-l1e-5", 1e-5)                 # seed 0 stopped at 900k
    patch_csv(stem(tmp_path / "inc", 0), lambda r: r if float(r["step"]) <= 900_000 else None)
    c = e0(tmp_path / "inc")["cells"][cell]
    assert c["window_steps"][0] == [850_000, 900_000] and c["complete"] is False
    d = tmp_path / "ext"                                        # outcome iii, E0-ext rows to 2M
    e0_cell(d, "tt-l1e-5", 1e-5, lam=0.001, cost=50.0, ext_ret=[30.0, 30.0, 30.0])
    for s in (0, 1):
        patch_csv(stem(d, s), at_step(1_800_000, eval_return="nan"))
    r = e0(d)
    assert r["outcome"] == "iii" and r["cells"][cell]["b"] is True     # divergence after 1M: not in E0
    assert r["e0_ext"]["move_window_to_end_step"] is True               # NaN counts as below 5


def test_diverged_runs_collapse_and_incomplete_runs_are_missing(tmp_path):
    nan = float("nan")
    run = {"start_steps": 10_000, "rows": [{"step": 5_000, "q_loss": nan, "jc": nan},
                                           {"step": 20_000, "q_loss": 1.0, "jc": nan, "would_engage": nan}]}
    assert not an.diverged(run)                                 # warm-up row, jc and would_engage ignored
    run["rows"].append({"step": 30_000, "eval_return": 1.0, "c_loss": nan})
    assert an.diverged(run) and not an.diverged(run, upto=25_000)
    gate_runs(tmp_path / "div")                                 # A helps on the held-out endpoint
    patch_csv(tmp_path / "div" / f"lagu-tt_{ENV}_s100", at_step(400_000, eval_return="nan"))
    res = an.pair(an.load(str(tmp_path / "div")), "lagu-tt", "lagu_nogate-tt", n_boot=200)[ENV]
    assert res["collapses_a"] == 1 and res["n_left_out"] == 0   # held-out return 23 is finite
    assert res["diverged"] == [f"lagu-tt_{ENV}_s100"]
    gate_runs(tmp_path / "fb", heldouts=False)                  # fallback: the NaN row is in the window
    patch_csv(tmp_path / "fb" / f"lagu-tt_{ENV}_s100", at_step(800_000, eval_return="nan"))
    res = an.pair(an.load(str(tmp_path / "fb")), "lagu-tt", "lagu_nogate-tt", n_boot=200)[ENV]
    assert res["collapses_a"] == 1 and res["n_left_out"] == 1 and res["n_pairs"] == 5
    d = tmp_path / "inc"                                      # A stopped at 600k of 1M, no heldout
    for s in range(100, 105):
        write_run(d, "lagu", s, rows_at(STEPS[:3], eval_return=20.0, eval_cost=30.0), tag="tt", steps=1_000_000)
        write_run(d, "lagu_nogate", s, rows_at(STEPS, eval_return=20.0, eval_cost=30.0), tag="tt")
    res = an.pair(an.load(str(d)), "lagu-tt", "lagu_nogate-tt", n_boot=200)[ENV]
    assert res["collapses_a"] == 0 and res["n_left_out"] == 5 and len(res["incomplete"]) == 5
    assert res["verdict"] == "Inconclusive"                     # was Hurts: 5 'collapses'
