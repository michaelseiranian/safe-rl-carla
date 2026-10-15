"""DESCRIPTIVE tables and figure for the paper: E3 per task, Lag-U vs TD3-Lag, and the E9 pareto table.

Every analysis in this file is DESCRIPTIVE. None is confirmatory, none enters the Holm family, and
no window, seed set or threshold here was chosen after looking at outcomes: windows, seed sets and
the collapse threshold are the ones fixed in prereg.md sections 5, 6, 9 and 11.

    cd /Users/michaelseiranian/Desktop/safe-rl-carla
    .venv/bin/python results/paper/e3_td3lag_e9/make.py

(1) E3 (prereg.md section 9, secondary prediction, outside the Holm family). Per task
    (SafetyCarGoal1-v0, SafetyPointCircle1-v0; results/gate_x): paired lagu-tt minus
    lagu_nogate-tt on the held-out primary endpoint (seeds 100-109), 95% paired bootstrap
    intervals, collapsed runs (held-out return < 5 or diverged, prereg.md 11) with the two-sided
    Fisher exact p, the gated arm's engagement and mean budget D, and the section 9 sign check
    against E1 (E1, 15 pairs, seeds 100-114: return -, cost -). The section 7 margins and labels
    are not applied to these tasks (prereg.md 9).
    NOT BLIND: the E3 contrasts were first computed by an audit after gate G3, before this script
    was written, so the E3 numbers had been seen when this file was written.
(2) Lag-U vs TD3-Lag (prereg.md 6, "Descriptive only"). Unpaired, IQM over seeds with stratified
    (per-arm) bootstrap 95% intervals, probability of improvement P(Lag-U return > TD3-Lag
    return) and P(Lag-U cost < TD3-Lag cost), on the held-out endpoint. CONFOUNDED: the td3lag arms
    use action noise 0.1 and M = 1 critic member, lagu uses action noise 0 and M = 3 with the
    exploration bonus, so a difference is not attributable to the gate or to the dual.
(3) E9 (prereg.md 3 and 13). Inventory of results/pareto seed 2 (the top-up) and the 3-seed
    summary (seeds 0-2) of the in-training final window: mean of eval_return and eval_cost over
    the rows with step >= 0.8N, N = 1M (prereg.md 5). Fixed-lambda rows are evidence of
    stationary-lambda exploitation only (prereg.md 13), not a bound on adaptive duals.

Statistics are numpy only and reuse lagu/analyze.py (paired_bootstrap, unpaired, fisher_exact,
endpoints): 10,000 resamples, percentile intervals, numpy.random.default_rng(0).

Outputs (in this directory): tables.md (the printed tables), results.json, e3_paired.pdf/.png.
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import lagu.analyze as A  # noqa: E402

PG1, CAR, CIRC = "SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyPointCircle1-v0"
SHORT = {PG1: "PointGoal1", CAR: "CarGoal1", CIRC: "PointCircle1"}
E1_SEEDS, E3_SEEDS, TD3_SEEDS = range(100, 115), range(100, 110), range(100, 110)
PARETO_SEEDS = (0, 1, 2)
PARETO_VARIANTS = ("fix005", "fix01", "fix02", "omni-fix03", "omni-fix1", "omni")
N_BOOT, SEED = A.N_BOOT, 0

RUNS = {d: A.load(f"results/{d}") for d in ("gate", "gate_literal", "gate_x", "pareto")}


def f(x, p=2):
    return A._f(x, p)


def ci(c, p=2):
    return A._ci(c, p)


def sgn(x):
    return "+" if x > 0 else "-" if x < 0 else "0"


# ---------------------------------------------------------------- (1) E3 per task

def gate_mechanics(rs):
    """Engagement (pooled sum(engagement*n_risk)/sum(n_risk)) over 25k < step <= 300k (the H2
    window) and over the whole run; mean budget D = sum(delta_avg*n_pi)/sum(n_pi) over every row with
    n_pi > 0, pooled over seeds (prereg.md 10.4), and D / delta0."""
    D = A.wmean(np.concatenate([A.col(r, "delta_avg") for r in rs]), np.concatenate([A.col(r, "n_pi") for r in rs]))
    delta0 = float(rs[0]["cfg"]["delta0"])
    return {"engagement_25k_300k": A.engagement(rs), "engagement_run": A.engagement(rs, 0, np.inf),
            "D": D, "delta0": delta0, "D_over_delta0": D / delta0,
            "per_seed_D": {r["seed"]: A.wmean(A.col(r, "delta_avg"), A.col(r, "n_pi")) for r in rs}}


def e3_task(runs, env, seeds):
    res = A.pair(runs, "lagu-tt", "lagu_nogate-tt", env=env, seeds=seeds, n_boot=N_BOOT, seed=SEED)[env]
    assert res["source"] == "heldout", (env, res["missing_heldout"])
    assert not res["unpaired_seeds"] and not res["incomplete"], env
    G = A.cells(runs, env, seeds)
    ga, gb = G[(env, "lagu-tt")], G[(env, "lagu_nogate-tt")]
    per = []
    for s in res["seeds"]:
        ea, eb = A.endpoints(ga[s], None, True), A.endpoints(gb[s], None, True)
        per.append({"seed": s, "return_a": ea["return"], "return_b": eb["return"], "d_return": ea["return"] - eb["return"],
                    "cost_a": ea["cost"], "cost_b": eb["cost"], "d_cost": ea["cost"] - eb["cost"],
                    "collapsed_a": ea["collapsed"], "collapsed_b": eb["collapsed"],
                    "resumed": ea["resumed"] or eb["resumed"]})
    res.pop("verdict", None)      # prereg.md 9: the section 7 labels are not applied to E3
    if res.get("sensitivity"):
        res["sensitivity"].pop("verdict", None)
    return {"contrast": res, "per_seed": per, "gate": gate_mechanics([ga[s] for s in sorted(ga)])}


def part1():
    out = {"E1_reference": e3_task(RUNS["gate"], PG1, E1_SEEDS)}
    for env in (CAR, CIRC):
        out[env] = e3_task(RUNS["gate_x"], env, E3_SEEDS)
    e1 = out["E1_reference"]["contrast"]
    s_ret, s_cost = sgn(e1["return"]["diff"]), sgn(e1["cost"]["diff"])
    for env in (CAR, CIRC):
        c = out[env]["contrast"]
        out[env]["sign_check"] = {"E1_return_sign": s_ret, "E1_cost_sign": s_cost,
                                  "return_sign": sgn(c["return"]["diff"]), "cost_sign": sgn(c["cost"]["diff"]),
                                  "return_same": sgn(c["return"]["diff"]) == s_ret,
                                  "cost_same": sgn(c["cost"]["diff"]) == s_cost}
        out[env]["sign_check"]["prediction_holds"] = bool(out[env]["sign_check"]["return_same"]
                                                          and out[env]["sign_check"]["cost_same"])

    print("## Table 1. E3 per task: lagu-tt minus lagu_nogate-tt, held-out primary endpoint (DESCRIPTIVE, "
          "secondary prediction, prereg.md 9; NOT BLIND: E3 contrasts were first computed by an audit after G3)\n")
    print("Paired by seed; mean over seeds; A - B with 95% paired bootstrap interval (10,000 resamples). "
          "Collapsed = held-out return < 5 or diverged (raw task units, not rescaled), Fisher exact two-sided. "
          "Engagement and D are for the gated arm (lagu-tt), pooled over its seeds. The section 7 margins and "
          "labels are not applied to E3. The E1 row is the confirmatory H3 contrast, repeated for reference.\n")
    print("| Task | Pairs | Return gate | Return no gate | Δreturn [95%] | Cost gate | Cost no gate | Δcost [95%] "
          "| Collapsed gate / no gate (Fisher p) | Engagement 25k-300k | Engagement whole run | Mean budget D (D/δ0) "
          "| Signs (Δreturn, Δcost) | Same signs as E1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, label in (("E1_reference", "PointGoal1 (E1, reference)"), (CAR, "CarGoal1 (E3)"), (CIRC, "PointCircle1 (E3)")):
        c, g = out[key]["contrast"], out[key]["gate"]
        r, k = c["return"], c["cost"]
        sc = out[key].get("sign_check")
        same = "reference" if sc is None else (
            f"return {'yes' if sc['return_same'] else 'no'}, cost {'yes' if sc['cost_same'] else 'no'}")
        print(f"| {label} | {c['n_pairs']} (s{c['seeds'][0]}-{c['seeds'][-1]}) | {f(r['mean_a'])} | {f(r['mean_b'])} "
              f"| {f(r['diff'])} {ci(r['ci95'])} | {f(k['mean_a'])} | {f(k['mean_b'])} | {f(k['diff'])} {ci(k['ci95'])} "
              f"| {c['collapses_a']} / {c['collapses_b']} ({f(c['fisher_p'], 3)}) | {f(g['engagement_25k_300k'], 3)} "
              f"| {f(g['engagement_run'], 3)} | {f(g['D'], 3)} ({f(g['D_over_delta0'], 3)}, δ0 = {g['delta0']:g}) "
              f"| ({sgn(r['diff'])}, {sgn(k['diff'])}) | {same} |")
    print()
    for env in (CAR, CIRC):
        sc = out[env]["sign_check"]
        print(f"- {SHORT[env]}: section 9 prediction (both mean paired differences have E1's signs, "
              f"{sc['E1_return_sign']}/{sc['E1_cost_sign']}): **{'holds' if sc['prediction_holds'] else 'does not hold'}**"
              f" (Δreturn {sc['return_sign']}, Δcost {sc['cost_sign']}).")
    for key in ("E1_reference", CAR, CIRC):
        c = out[key]["contrast"]
        extra = []
        if c["n_left_out"]:
            extra.append(f"{c['n_left_out']} pair(s) left out of the intervals (non-finite)")
        if c["diverged"]:
            extra.append(f"diverged: {c['diverged']}")
        if c["resumed_seeds"]:
            s = c["sensitivity"]
            extra.append(f"pairs with a resumed run: {c['resumed_seeds']}; without them ({s['n_pairs']} pairs) "
                         f"Δreturn {f(s['return']['diff'])} {ci(s['return']['ci95'])}, "
                         f"Δcost {f(s['cost']['diff'])} {ci(s['cost']['ci95'])}")
        else:
            extra.append("no resumed runs")
        print(f"- {SHORT[key if key != 'E1_reference' else PG1]}{' (E1)' if key == 'E1_reference' else ''}: "
              + "; ".join(extra) + ".")
    print()

    print("## Table 1b. E3 per seed: held-out return and cost (DESCRIPTIVE; NOT BLIND, see Table 1)\n")
    print("| Task | Seed | Return gate | Return no gate | Δreturn | Cost gate | Cost no gate | Δcost | Collapsed (gate, no gate) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for env in (CAR, CIRC):
        for p in out[env]["per_seed"]:
            print(f"| {SHORT[env]} | {p['seed']} | {f(p['return_a'])} | {f(p['return_b'])} | {f(p['d_return'])} "
                  f"| {f(p['cost_a'])} | {f(p['cost_b'])} | {f(p['d_cost'])} "
                  f"| ({'yes' if p['collapsed_a'] else 'no'}, {'yes' if p['collapsed_b'] else 'no'}) |")
    print()
    return out


# ---------------------------------------------------------------- (2) Lag-U vs TD3-Lag

TD3_COMPARISONS = (
    ("gate (E1 recipe, 2M)", "gate", PG1, "lagu-tt", "td3lag-tt"),
    ("gate_literal (literal recipe, 1M)", "gate_literal", PG1, "lagu-lit", "td3lag-lit"),
    ("gate_x (E1 recipe, 2M)", "gate_x", CAR, "lagu-tt", "td3lag-tt"),
    ("gate_x (E1 recipe, 2M)", "gate_x", CIRC, "lagu-tt", "td3lag-tt"),
)


def part2():
    out = []
    for name, d, env, a, b in TD3_COMPARISONS:
        G = A.cells(RUNS[d], env, TD3_SEEDS)
        ra, rb = G[(env, a)], G[(env, b)]
        missing = [g[s]["stem"] for g in (ra, rb) for s in sorted(g) if not A.heldout_ok(g[s])]
        absent = [f"{lab} s{s}" for lab, g in ((a, ra), (b, rb)) for s in TD3_SEEDS
                  if s not in g and not (lab.startswith("td3lag") and d == "gate_x" and s > 104)]
        cfg = {lab: {k: next(iter(g.values()))["cfg"][k] for k in ("action_noise", "M", "use_bonus", "use_gate")}
               for lab, g in ((a, ra), (b, rb))}
        res = A.unpaired(RUNS[d], a, b, env=env, seeds=TD3_SEEDS, n_boot=N_BOOT, seed=SEED)[env]
        out.append({"comparison": name, "dir": d, "env": env, "a": a, "b": b, "seeds_a": sorted(ra), "seeds_b": sorted(rb),
                    "missing_heldout": missing, "absent_runs": absent, "cfg": cfg, **res})

    print("## Table 2. Lag-U vs TD3-Lag, held-out primary endpoint, unpaired (DESCRIPTIVE only, prereg.md 6; "
          "CONFOUNDED: td3lag arms use action noise 0.1 and M = 1, lagu uses action noise 0 and M = 3 with the "
          "exploration bonus)\n")
    print("IQM over seeds; stratified (per-arm) bootstrap 95% intervals, 10,000 resamples. P(return) = P(Lag-U "
          "return > TD3-Lag return), P(cost) = P(Lag-U cost < TD3-Lag cost), over all seed pairs, ties counted 1/2.\n")
    print("| Directory | Task | n Lag-U / TD3-Lag | Return IQM Lag-U | Return IQM TD3-Lag | Return IQM diff | P(return) "
          "| Cost IQM Lag-U | Cost IQM TD3-Lag | Cost IQM diff | P(cost) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out:
        x, y = r["return"], r["cost"]
        print(f"| {r['comparison']}: {r['a']} vs {r['b']} | {SHORT[r['env']]} | {r['n_a']} / {r['n_b']} "
              f"| {f(x['iqm_a'])} {ci(x['ci_a'])} | {f(x['iqm_b'])} {ci(x['ci_b'])} | {f(x['diff'])} {ci(x['ci_diff'])} "
              f"| {f(x['prob_improvement'])} {ci(x['ci_prob_improvement'])} "
              f"| {f(y['iqm_a'])} {ci(y['ci_a'])} | {f(y['iqm_b'])} {ci(y['ci_b'])} | {f(y['diff'])} {ci(y['ci_diff'])} "
              f"| {f(y['prob_improvement'])} {ci(y['ci_prob_improvement'])} |")
    print()
    for r in out:
        assert r["source"] == "heldout", r["missing_heldout"]
        seeds_b = f"s{r['seeds_b'][0]}-{r['seeds_b'][-1]}"
        cf = "; ".join(f"{k}: action_noise {v['action_noise']:g}, M {v['M']}, bonus {v['use_bonus']}, gate {v['use_gate']}"
                       for k, v in r["cfg"].items())
        print(f"- {r['dir']} {SHORT[r['env']]}: Lag-U seeds s{r['seeds_a'][0]}-{r['seeds_a'][-1]}, TD3-Lag seeds {seeds_b}; "
              f"held-out file for every run: {'yes' if not r['missing_heldout'] else 'NO, missing ' + str(r['missing_heldout'])}"
              f"{'; absent runs ' + str(r['absent_runs']) if r['absent_runs'] else ''}. Config: {cf}.")
    lit = next(r for r in out if r["dir"] == "gate_literal")
    print(f"- td3lag-lit held-out check (prereg.md 16 item 5): {len(lit['seeds_b'])} of 10 runs present (seeds "
          f"{lit['seeds_b']}), missing held-out files: {[s for s in lit['missing_heldout'] if s.startswith('td3lag')] or 'none'}.")
    print("- With 5 TD3-Lag seeds (gate_x) the IQM keeps the middle 3 runs, so its interval is coarse.")
    print()
    return out


# ---------------------------------------------------------------- (3) E9 pareto

def part3():
    G = A.cells(RUNS["pareto"], PG1)
    inv, summ = [], []
    for v in PARETO_VARIANTS:
        g = G[(PG1, "td3lag-" + v)]
        assert sorted(g) == list(PARETO_SEEDS), (v, sorted(g))
        r2 = g[2]
        stem = os.path.join("results/pareto", r2["stem"])
        snaps = sorted(int(x[:-3]) for x in os.listdir(stem + ".snap") if x.endswith(".pt")) if os.path.isdir(stem + ".snap") else []
        cfg = r2["cfg"]
        with open(stem + ".json") as fh:
            args = json.load(fh)["args"]
        cols = list(r2["rows"][0].keys())
        inv.append({"variant": v, "stem": r2["stem"], "recipe": args.get("recipe"), "dual": cfg["dual"],
                    "lambda_init": cfg["lambda_init"], "lambda_lr": cfg["lambda_lr"],
                    "actor_lr": cfg["actor_lr"] or cfg["lr"], "critic_lr": cfg["critic_lr"] or cfg["lr"],
                    "action_noise": cfg["action_noise"], "M": cfg["M"],
                    "rows": len(r2["rows"]), "last_step": int(r2["rows"][-1]["step"]), "n_columns": len(cols),
                    "files": {ext: os.path.exists(stem + ext) for ext in (".csv", ".json", ".log", ".pt", ".ckpt.pt", ".heldout.json")},
                    "snapshots": len(snaps), "snap_first": snaps[0] if snaps else None, "snap_last": snaps[-1] if snaps else None,
                    "resumed": bool(np.any(A.col(r2, "resumed") == 1)), "diverged": A.diverged(r2, 1_000_000),
                    "cum_cost": A.at(r2, "cum_cost", 1_000_000),
                    "seed01_columns": len(g[0]["rows"][0].keys())})
        per = {s: A.endpoints(g[s]) for s in PARETO_SEEDS}
        assert all(e["N"] == 1_000_000 and e["complete"] and e["n_window"] == 5 for e in per.values()), v
        ret = np.array([per[s]["window_return"] for s in PARETO_SEEDS])
        cost = np.array([per[s]["window_cost"] for s in PARETO_SEEDS])
        lam = np.array([A.at(g[s], "lambda", 1_000_000) for s in PARETO_SEEDS])
        summ.append({"variant": v, "recipe": args.get("recipe"), "dual": cfg["dual"], "lambda_init": cfg["lambda_init"],
                     "return": dict(zip(PARETO_SEEDS, ret)), "cost": dict(zip(PARETO_SEEDS, cost)),
                     "return_mean": ret.mean(), "return_sd": A.sd(ret), "cost_mean": cost.mean(), "cost_sd": A.sd(cost),
                     "return_mean_s01": ret[:2].mean(), "cost_mean_s01": cost[:2].mean(),
                     "final_lambda": dict(zip(PARETO_SEEDS, lam)),
                     "collapsed": int(sum(per[s]["collapsed"] for s in PARETO_SEEDS)),
                     "at_or_below_limit": int(np.sum(cost <= A.COST_LIMIT))})

    print("## Table 3. E9: what results/pareto holds for seed 2 (the pre-registered top-up; DESCRIPTIVE)\n")
    print("All runs SafetyPointGoal1-v0, td3lag (M = 1, action noise 0.1), 1M steps. `paper` recipe = literal "
          "learning rates (actor and critics 3e-4); `omnisafe` = actor 5e-6, critics 1e-3. No seed-2 run was resumed "
          "or diverged unless stated; no run in results/pareto has a held-out file.\n")
    print("| Variant | Recipe | Dual | λ (fixed or initial) | Rows / last step | Files | Snapshots | Columns (s2 / s0-s1) | cum_cost at 1M |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in inv:
        files = ", ".join(k for k, ok in r["files"].items() if ok)
        print(f"| td3lag-{r['variant']} | {r['recipe']} | {r['dual']} | {r['lambda_init']:g} | {r['rows']} / {r['last_step']} "
              f"| {files} | {r['snapshots']} ({r['snap_first']}-{r['snap_last']}, every 50k) | {r['n_columns']} / {r['seed01_columns']} "
              f"| {f(r['cum_cost'], 0)} |")
    bad = [r["stem"] for r in inv if r["resumed"] or r["diverged"] or r["last_step"] != 1_000_000]
    print(f"\n- Seed-2 runs resumed, diverged or short of 1M: {bad or 'none'}. Seeds 0-1 were written with the older "
          "31-column CSV schema (no cum_cost, lambda_avg, resumed); their args and cfg equal seed 2's except seed, "
          "ckpt_every and drop_ckpt.\n")

    print("## Table 4. Pareto table: TD3-Lag fixed-λ and OmniSafe-recipe variants, in-training final window, 3 seeds "
          "(DESCRIPTIVE; fixed-λ rows are evidence of stationary-λ exploitation only, prereg.md 13)\n")
    print("Final window = mean of eval_return / eval_cost over rows with step >= 800k (5 rows, 10 episodes each, "
          "in-training layouts EVAL_SEED + k). Mean ± sd (ddof 1) over seeds 0-2; collapsed = window return < 5; "
          "final λ at 1M per seed.\n")
    print("| Variant | Recipe | Dual | λ | Return s0 / s1 / s2 | Return mean ± sd (3 seeds) | Return mean s0-s1 "
          "| Cost s0 / s1 / s2 | Cost mean ± sd (3 seeds) | Cost mean s0-s1 | Collapsed | Cost ≤ 25 | Final λ s0 / s1 / s2 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summ:
        rs = " / ".join(f(r["return"][s], 1) for s in PARETO_SEEDS)
        cs = " / ".join(f(r["cost"][s], 1) for s in PARETO_SEEDS)
        ls = " / ".join(f(r["final_lambda"][s], 3) for s in PARETO_SEEDS)
        print(f"| td3lag-{r['variant']} | {r['recipe']} | {r['dual']} | {r['lambda_init']:g} | {rs} "
              f"| {f(r['return_mean'], 1)} ± {f(r['return_sd'], 1)} | {f(r['return_mean_s01'], 1)} | {cs} "
              f"| {f(r['cost_mean'], 1)} ± {f(r['cost_sd'], 1)} | {f(r['cost_mean_s01'], 1)} | {r['collapsed']}/3 "
              f"| {r['at_or_below_limit']}/3 | {ls} |")
    print()
    return {"seed2_inventory": inv, "summary": summ}


# ---------------------------------------------------------------- figure

def figure(e3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid, shade, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3", "#ecebe6", "#2a78d6"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "text.color": ink, "axes.titlesize": 9.5})
    panels = (("E1_reference", "PointGoal1, E1 (reference)"), (CAR, "CarGoal1, E3"), (CIRC, "PointCircle1, E3"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.9), facecolor=surface, constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        c, per = e3[key]["contrast"], e3[key]["per_seed"]
        dr = np.array([p["d_return"] for p in per])
        dc = np.array([p["d_cost"] for p in per])
        ax.set_facecolor(surface)
        xr = max(np.abs(np.r_[dr, c["return"]["ci95"]]).max() * 1.12, 1.0)
        yr = max(np.abs(np.r_[dc, c["cost"]["ci95"]]).max() * 1.12, 1.0)
        ax.set_xlim(-xr, xr)
        ax.set_ylim(-yr, yr)
        ax.add_patch(plt.Rectangle((-xr, -yr), xr, yr, color=shade, zorder=0, lw=0,
                                   label="E1's signs (Δreturn < 0 and Δcost < 0)"))
        ax.axhline(0, color=grid, lw=1, zorder=1)
        ax.axvline(0, color=grid, lw=1, zorder=1)
        ax.scatter(dr, dc, s=36, color=blue, edgecolor=surface, linewidth=1.5, zorder=3, label="one seed pair")
        r, k = c["return"], c["cost"]
        ax.errorbar(r["diff"], k["diff"],
                    xerr=[[r["diff"] - r["ci95"][0]], [r["ci95"][1] - r["diff"]]],
                    yerr=[[k["diff"] - k["ci95"][0]], [k["ci95"][1] - k["diff"]]],
                    fmt="D", ms=7, color=ink, mfc=ink, mec=surface, mew=1.5, elinewidth=1.6, capsize=3, zorder=4,
                    label="mean, 95% paired bootstrap CI")
        ax.set_title(f"{title}: {c['n_pairs']} pairs (s{c['seeds'][0]}-{c['seeds'][-1]})\n"
                     f"Δreturn {f(r['diff'])} {ci(r['ci95'])}\nΔcost {f(k['diff'])} {ci(k['ci95'])}\n"
                     f"collapsed runs, gate / no gate: {c['collapses_a']} / {c['collapses_b']}", color=ink)
        ax.set_xlabel("Δ held-out return per episode\n(lagu − lagu_nogate)")
        ax.grid(False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Δ held-out cost per episode\n(lagu − lagu_nogate)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle("E3 vs E1: paired gate effect per task (DESCRIPTIVE; E3 not blind, first computed by an audit "
                 "after G3). Axes differ per panel.", fontsize=9.5, color=ink)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"e3_paired.{ext}"), dpi=200, facecolor=surface)
    plt.close(fig)


def main():
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("# E3, Lag-U vs TD3-Lag, E9 pareto (DESCRIPTIVE; produced by results/paper/e3_td3lag_e9/make.py)\n")
        e3 = part1()
        td3 = part2()
        e9 = part3()
    text = buf.getvalue()
    print(text, end="")
    with open(os.path.join(HERE, "tables.md"), "w") as fh:
        fh.write(text)
    A.write_json({"label": "DESCRIPTIVE", "e3_not_blind": True, "n_boot": N_BOOT, "rng_seed": SEED,
                  "E3": e3, "lagu_vs_td3lag": td3, "E9": e9}, os.path.join(HERE, "results.json"))
    figure(e3)
    print("wrote", os.path.join(HERE, "e3_paired.pdf"), "and .png")


if __name__ == "__main__":
    main()
