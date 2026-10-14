"""safe2.analyze before unblinding (SPEC section 9): selection with imputation and tie-breaks, R_ref, B*,
Saute* and L*, the Stage-0 gate decisions, the candidate kill rules (a), (b), (c), 8.2 and 8.4, the plan
and the cut list, write-once records, the decision path's type-I error under a simulated null at the
family level 0.025, the guards and imputation through that path, and that no subcommand before
--unblind opens a path under sealed/."""
from __future__ import annotations

import builtins
import json
import math
import os
import shutil

import numpy as np
import pytest

from safe2 import analyze as an
from safe2 import configs, power, seeds, stats, train
from safe2.tests.analyze_helpers import write_run, write_snapshot

PG1, CG1, PC1 = an.TASKS
SEL = seeds.SELECTION_SEEDS


# --- a synthetic pilot ------------------------------------------------------------------------------
P1_VALUES = {   # config: (R per seed or constant, C) on seeds 0-2; see test_p1_baselines for the ranking
    "R2": (26.0, 60.0), "B5": (16.0, 24.0), "B7": (16.3, 24.0), "B4": (16.2, 24.0), "B9": (16.1, 22.0),
    "SauteA": (16.0, 22.0), "B8": (17.0, 40.0), "B6": ([10.0, 3.0, 10.0], 20.0), "B2": ([1.0, 1.0, 10.0], 20.0),
}
BVS_VALUES = {"BVS-SQ-k0.02": (18.0, 24.0), "BVS-SQ-k0.1": (20.0, 24.0), "BVS-SN-k0.02": (12.0, 20.0),
              "BVS-SN-k0.1": ([25.0, 2.0, 2.0], 20.0)}
SHL_VALUES = {"SHL-g0.90-B4": (17.0, 24.0), "SHL-g0.90-B5": (15.0, 24.0), "SHL-g0.95-B4": (14.0, 24.0),
              "SHL-g0.95-B5": (13.0, 24.0)}
INT_RATE = {0: 0.5, 1: 0.1, 2: 0.1}        # per-seed training intervention rate of BVS-SQ-k0.1


def bvs_cols(seed):
    def cols(step):
        sh = max(0, step - 25_000)
        ep = sh / 1000
        rd = max(0, step - 35_000)
        return {"int_cum": INT_RATE[seed] * sh, "shield_steps_cum": sh, "F_cum": 2.0 * ep, "ep_cum": ep,
                "eval_violation": 0.2, "rho_num": INT_RATE[seed] * rd, "rho_den": rd}
    return cols


def _rc(values, s):
    R, C = values
    return (R[s] if isinstance(R, list) else R), C


def build_pilot(root):
    p1 = os.path.join(root, "p1")
    for name in list(configs.BASELINE_POOL) + ["R2"]:
        for s in SEL:
            R, C = _rc(P1_VALUES.get(name, (10.0, 20.0)), s)
            if name == "B4" and s == 1:
                write_run(p1, name, PG1, s, R, C, diverge_at=900_000)          # diverged, ran to the end
            elif name == "B3" and s == 2:
                write_run(p1, name, PG1, s, R, C, diverge_at=850_000, upto=850_000, finished=False)   # stopped
            else:
                write_run(p1, name, PG1, s, R, C)
    for name, (R, C) in (("SauteA", (10.0, 22.0)), ("B9", (16.1, 22.0)), ("B7", (16.3, 24.0))):
        for s in seeds.EXTRA_SELECTION_SEEDS:
            write_run(p1, name, PG1, s, R, C)
    for vals in (BVS_VALUES, SHL_VALUES):
        for name, v in vals.items():
            for s in SEL:
                write_run(p1, name, PG1, s, *_rc(v, s), cols=bvs_cols(s) if name.startswith("BVS") else None)
    for s, m in zip(SEL, (2.0, 3.0, 4.0)):
        write_snapshot(p1, train.stem_of("BVS-SQ-k0.1", PG1, s), 1_000_000, m_dep=m)
    p2 = os.path.join(root, "p2")
    for task, vals, m in ((CG1, {"B9": (12.0, 22.0), "R2": (20.0, 50.0), "BVS-SQ-k0.1": (14.0, 24.0)}, 1.0),
                          (PC1, {"B9": (30.0, 20.0), "R2": (40.0, 60.0), "BVS-SQ-k0.1": (35.0, 24.0)}, 6.0)):
        for name, (R, C) in vals.items():
            for s in SEL:
                write_run(p2, name, task, s, R, C)
                if name.startswith("BVS"):
                    write_snapshot(p2, train.stem_of(name, task, s), 1_000_000, m_dep=m + s - 1)
    st0 = os.path.join(root, "stage0")
    os.makedirs(st0)
    with open(os.path.join(st0, "g0a.json"), "w") as f:
        json.dump({"mean_return": 23.0, "mean_cost": 20.0, "n_episodes": 90}, f)
    with open(os.path.join(st0, "g0b.json"), "w") as f:            # probe g0b's format
        json.dump({"recall": {"pooled": {"recall": 0.85, "fa_per_episode": 12.0}},
                   "ranking": {"states": 1100, "spearman": 0.75}, "decision": {"proceed": True}}, f)
    with open(os.path.join(st0, "s04.json"), "w") as f:
        shares = (0.7, 0.65, 0.5)
        json.dump({"runs": {train.stem_of(n, PG1, s): {"share": shares[s]} for n in SHL_VALUES for s in SEL}}, f)
    with open(os.path.join(root, "PREREG_SHA256.txt"), "w") as f:     # the selection code, as at method-v0
        f.write("# method-v0 test\n" + "".join(f"{an._sha(os.path.join(an.REPO, p))}  {p}\n"
                                               for p in an.SELECTION_CODE))
    with open(os.path.join(root, "FUNC_SHA256.json"), "w") as f:
        json.dump(stats.source_hashes(), f)
    os.makedirs(os.path.join(root, "sealed", "conf"))
    with open(os.path.join(root, "sealed", "conf", "x.rows"), "w") as f:
        f.write("never read before unblinding\n")


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """The pilot chain run through the CLI: p1-baselines, bstar, candidates --stage p1, candidates, plan."""
    root = str(tmp_path_factory.mktemp("method"))
    build_pilot(root)
    for cmd in (["p1-baselines"], ["bstar"], ["candidates", "--stage", "p1"], ["candidates"], ["plan"]):
        assert an.main(cmd + ["--root", root]) == 0, cmd
    rec = {n: json.load(open(os.path.join(root, *n.split("/")))) for n in
           ("p1/top3.json", "v05.json", "p1/candidates.json", "v1.json", "plan.json")}
    return root, rec


# --- selection ----------------------------------------------------------------------------------------
def test_p1_baselines_imputation_collapse_and_tie_breaks(study):
    _, rec = study
    top = rec["p1/top3.json"]
    assert top["r_ref"][PG1] == pytest.approx(26.0)                     # R2's mean R
    # round 1: B7 16.3 is best; within 0.5: B4 (1 collapse), B9 and SauteA (cost 22), B5 (cost 24);
    # SauteA beats B9 on compute (2.4 h literal vs 3.0 h OmniSafe). Round 2: B9 on cost. Round 3: B7 over B5
    # on compute. The 10-score configs order by compute then name; B2 (2 of 3 collapsed) goes last.
    assert top["top3"] == ["SauteA", "B9", "B7"]
    assert top["order"] == ["SauteA", "B9", "B7", "B5", "B4", "B7b", "B7c", "SauteB", "B1", "B6", "B3", "B8", "B2"]
    t = top["table"]
    assert t["B4"]["collapses"] == 1 and t["B4"]["R"] == [16.2] * 3           # a finite flag of 0: collapsed
    assert t["B3"]["R"][2] == 0.0 and t["B3"]["C"][2] == 40.0                 # stopped after diverging: R 0, worst C
    assert t["B3"]["collapses"] == 1 and t["B6"]["collapses"] == 1            # 3 / 26 < 0.2
    assert not t["B2"]["eligible"] and t["B2"]["collapses"] == 2
    assert t["B8"]["score"] == pytest.approx(17.0 - 15.0)                      # hinge on the config's mean cost


def test_bstar_saute_star_l_star_and_gates(study):
    _, rec = study
    v = rec["v05.json"]
    # six seeds: B7 16.3 / 24 and B9 16.1 / 22 tie within 0.5; B9 has the lower cost; SauteA drops to 13
    assert v["b_star"] == "B9" and v["b_star_base"] == "omni"
    assert v["r_bstar_p1"] == pytest.approx(16.1)
    assert v["saute_star"] == "SauteA" and v["l_star"] == "SauteA"            # L* ties B7, wins on cost
    assert v["run_saute_star"] and not v["run_l_star"]                        # L* is Saute*: no separate arm
    assert v["g0a"]["return_threshold"] == pytest.approx(22.1) and v["g0a"]["proceed"]
    assert v["g0b"]["pass_i"] and v["g0b"]["pass_ii"] and v["bvs_stage0_proceed"]


def test_bstar_record_was_hashed_and_records_are_write_once(study, tmp_path):
    root, rec = study
    lines = open(os.path.join(root, "PREREG_SHA256.txt")).read().splitlines()
    k = 1 + len(an.SELECTION_CODE)
    assert lines[k].startswith("# method-v0.5") and lines[k + 1].endswith("v05.json")
    assert lines[k + 2].startswith("# method-v1") and lines[k + 3].endswith("v1.json") and lines[k + 4].endswith("plan.json")
    assert an.hash_entries("\n".join(lines))[lines[k + 1].split("  ")[1]] == an._sha(os.path.join(root, "v05.json"))
    # re-running a step that would write the same record changes nothing; different content refuses
    assert an.main(["bstar", "--root", root]) == 0
    assert open(os.path.join(root, "PREREG_SHA256.txt")).read().splitlines() == lines
    path = str(tmp_path / "r.json")
    assert an.write_record(path, {"a": 1}) == "written" and an.write_record(path, {"a": 1}) == "unchanged"
    with pytest.raises(an.Refusal):
        an.write_record(path, {"a": 2})
    assert an.write_record(path, {"a": 2}, overwrite=True) == "written"


def test_bstar_refuses_a_tampered_top3(study, tmp_path):
    root = str(tmp_path / "m")
    shutil.copytree(study[0], root)
    top = json.load(open(os.path.join(root, "p1", "top3.json")))
    top["top3"] = ["B9", "SauteA", "B7"]
    json.dump(top, open(os.path.join(root, "p1", "top3.json"), "w"))
    with pytest.raises(an.Refusal, match="recomputation"):
        an.bstar(root)


def test_incomplete_inputs_refuse_and_the_cli_prints_why(study, tmp_path, capsys):
    root = str(tmp_path / "m")
    shutil.copytree(study[0], root)
    os.remove(os.path.join(root, "p1", train.stem_of("B5", PG1, 1) + ".pt"))        # running, not finished
    assert an.main(["p1-baselines", "--root", root]) == 1
    out = capsys.readouterr().out
    assert "refusing: inputs incomplete" in out and "B5_SafetyPointGoal1-v0_s1: incomplete" in out
    # a run made with other settings than the registry's is refused too
    write_run(os.path.join(root, "p1"), "B5", PG1, 1, 16.0, 24.0, rc=configs.make("B5", kp=0.01))
    with pytest.raises(an.Refusal, match="differs from the registry"):
        an.p1_baselines(root)


def test_a_live_or_queued_run_is_not_complete(study, tmp_path, monkeypatch):
    """A run on its 3rd resume or a diverged run still training (the trainer runs on to 1M) is 'running'
    while live or queued, not 'unfinished' / 'diverged', so no selection is made from it (H2)."""
    root = str(tmp_path / "m")
    shutil.copytree(study[0], root)
    p1 = os.path.join(root, "p1")
    s3 = write_run(p1, "B5", PG1, 1, 16.0, 24.0, upto=600_000, finished=False, n_resumes=3)
    sd = write_run(p1, "B6", PG1, 0, 10.0, 20.0, upto=600_000, finished=False, diverge_at=300_000)
    assert an.load_run(p1, "B5", PG1, 1)["state"] == "unfinished"          # stopped: no process, not queued
    assert an.load_run(p1, "B6", PG1, 0)["state"] == "diverged"
    live = {(os.path.realpath(p1), s3), (os.path.realpath(p1), sd)}
    assert an.load_run(p1, "B5", PG1, 1, pending=live)["state"] == "running"
    monkeypatch.setattr(an, "pending_runs", lambda: live)
    with pytest.raises(an.Refusal) as e:
        an.p1_baselines(root)
    assert f"p1/{s3}: running" in e.value.problems and f"p1/{sd}: running" in e.value.problems
    monkeypatch.setattr(an, "pending_runs", lambda: set())
    an.p1_baselines(root)                                                   # both stopped: decidable
    # the queue and the live trainers feed pending_runs, keyed by the real output directory
    from safe2 import health, launch
    monkeypatch.undo()
    monkeypatch.setattr(health, "live_states", lambda: {(p1, s3): "R"})
    monkeypatch.setattr(launch, "read_queue", lambda: [{"out": p1, "stem": sd, "cmd": []}])
    assert an.pending_runs() == live


def test_record_subcommands_refuse_code_changed_since_v0(study, tmp_path, monkeypatch, capsys):
    """M6: p1-baselines, bstar, candidates and plan refuse when the selection code or a decision function
    differs from method-v0, unless deviations.json lists it."""
    root = str(tmp_path / "m")
    shutil.copytree(study[0], root)
    hfile = os.path.join(root, "PREREG_SHA256.txt")
    text = open(hfile).read()
    open(hfile, "w").write(text.replace(an._sha(os.path.join(an.REPO, "safe2/stats.py")), "0" * 64, 1))
    for cmd in (["p1-baselines"], ["bstar"], ["candidates"], ["plan"]):
        assert an.main(cmd + ["--root", root]) == 1
        assert "safe2/stats.py differs from its method-v0 hash" in capsys.readouterr().out
    json.dump([{"file": "safe2/stats.py", "entry": "section 22, D1"}], open(os.path.join(root, "deviations.json"), "w"))
    assert an.main(["p1-baselines", "--root", root]) == 0
    open(hfile, "w").write(text)
    f = json.load(open(os.path.join(root, "FUNC_SHA256.json")))
    f["label"] = "0" * 64
    json.dump(f, open(os.path.join(root, "FUNC_SHA256.json"), "w"))
    assert an.main(["bstar", "--root", root]) == 1
    assert "decision function label changed" in capsys.readouterr().out


# --- Stage-0 gates and kill rules ---------------------------------------------------------------------
def test_bstar_applies_the_gate_thresholds(study, tmp_path):
    """G0a needs return >= max(18, R_B*(P1) + 6) = 22.1 here and cost <= 25 (probe.g0a_decision); a failed
    gate kills BVS at Stage 0."""
    root = str(tmp_path / "m")
    shutil.copytree(study[0], root)
    path = os.path.join(root, "stage0", "g0a.json")
    for ret, cost, ok in ((22.2, 25.0, True), (22.0, 20.0, False), (30.0, 25.5, False)):
        json.dump({"mean_return": ret, "mean_cost": cost}, open(path, "w"))
        v = an.bstar(root)
        assert v["g0a"]["return_threshold"] == pytest.approx(22.1)
        assert v["g0a"]["proceed"] is ok and v["bvs_stage0_proceed"] is ok
    json.dump({"mean_cost": 20.0}, open(path, "w"))
    with pytest.raises(an.Refusal, match="mean_return"):
        an.bstar(root)
    g0b = os.path.join(root, "stage0", "g0b.json")
    b = json.load(open(g0b))
    b["ranking"]["states"] = 999                       # fewer than 1000 pooled states fails (ii)
    json.dump(b, open(g0b, "w"))
    json.dump({"mean_return": 23.0, "mean_cost": 20.0}, open(path, "w"))
    with pytest.raises(an.Refusal, match="probe's decision differs"):
        an.bstar(root)
    b["decision"]["proceed"] = False
    json.dump(b, open(g0b, "w"))
    v = an.bstar(root)
    assert not v["g0b"]["pass_ii"] and v["g0b"]["pass_i"] and not v["bvs_stage0_proceed"]


def _bvs_run(seed, rate, F=2.0, viol=0.2, drop=None):
    rows = []
    for step in range(50_000, 1_000_001, 50_000):
        if step == drop:
            continue
        sh = step - 25_000
        rows.append({"step": step, "int_cum": rate * sh, "shield_steps_cum": sh, "F_cum": F * sh / 1000,
                     "ep_cum": sh / 1000, "eval_violation": viol, "rho_num": rate * sh, "rho_den": sh})
    return {"id": f"p1/x_s{seed}", "rows": rows, "eval_episodes": 20}


def test_kill_82_pools_over_seeds_and_the_500k_1m_window():
    r = an.kill_82([_bvs_run(0, 0.5), _bvs_run(1, 0.1), _bvs_run(2, 0.1)])
    assert r["intervention_rate"] == pytest.approx(0.7 / 3) and not r["kill"]      # one seed at 50% is not enough
    assert an.kill_82([_bvs_run(0, 0.5), _bvs_run(1, 0.5), _bvs_run(2, 0.0)])["kill"]
    assert an.kill_82([_bvs_run(s, 0.1, F=3.5) for s in SEL])["reasons"] == ["mean F > 3"]
    r = an.kill_82([_bvs_run(0, 0.1, viol=0.4), _bvs_run(1, 0.1, viol=0.2), _bvs_run(2, 0.1, viol=0.2)])
    assert r["eval_violation_share"] == pytest.approx(0.8 / 3) and r["reasons"] == ["violation share > 0.25"]
    assert r["episodes_evaluated"] == 3 * 11 * 20



def _nan_at(run, step, key):
    for row in run["rows"]:
        if row["step"] == step:
            row[key] = float("nan")
    return run


def test_a_stopped_or_non_finite_bvs_seed_is_left_out_and_two_are_needed():
    """Protocol section 6 (M7): 8.2 and rho* pool over the seeds with complete finite inputs, list the
    others, and fewer than 2 qualifying seeds kills BVS* under 8.2 (rho* is then never recorded)."""
    r = an.kill_82([_bvs_run(0, 0.1, drop=500_000), _bvs_run(1, 0.1), _bvs_run(2, 0.3)])
    assert r["seeds_used"] == 2 and r["excluded"] == ["p1/x_s0: rows 500k-1M missing"]
    assert r["intervention_rate"] == pytest.approx(0.2) and not r["kill"]
    stopped = {"id": "p1/x_s2", "rows": _bvs_run(2, 0.1)["rows"][:5], "eval_episodes": 20}   # stopped at 250k
    runs = [_bvs_run(0, 0.1), _nan_at(_bvs_run(1, 0.5), 1_000_000, "F_cum"), stopped]
    r = an.kill_82(runs)
    assert r["kill"] and r["seeds_used"] == 1 and len(r["excluded"]) == 2
    assert an.rho_star(runs) is None
    runs = [_bvs_run(0, 0.1), _nan_at(_bvs_run(1, 0.5), 700_000, "eval_violation"), _bvs_run(2, 0.3)]
    assert an.rho_star(runs) == pytest.approx(0.2) and an.kill_82(runs)["seeds_used"] == 2


def test_kill_84_and_rho_star():
    s04 = {"runs": {train.stem_of("S", PG1, s): {"share": v} for s, v in zip(SEL, (0.61, 0.2, 0.9))}}
    assert an.kill_84(s04, "S")["kill"] and an.kill_84(s04, "S")["over_0.6"] == 2
    s04["runs"][train.stem_of("S", PG1, 0)]["share"] = 0.6                   # not above 0.6
    assert not an.kill_84(s04, "S")["kill"]
    del s04["runs"][train.stem_of("S", PG1, 2)]
    with pytest.raises(an.Refusal):
        an.kill_84(s04, "S")
    runs = [_bvs_run(0, 0.5), _bvs_run(1, 0.1), _bvs_run(2, 0.0)]
    assert an.rho_star(runs) == pytest.approx(0.2)


def test_candidates_select_eligible_only_and_apply_the_kill_rules(study):
    _, rec = study
    c = rec["p1/candidates.json"]
    bvs, shl = c["BVS"], c["SHL"]
    assert bvs["star"] == "BVS-SQ-k0.1" and "BVS-SN-k0.1" not in bvs["order"]      # ineligible, best score
    assert bvs["kills"]["8.2"]["intervention_rate"] == pytest.approx(0.7 / 3) and bvs["killed"] is None
    assert bvs["kills"]["b"]["diff"] == pytest.approx(20.0 - 16.1) and not bvs["kills"]["b"]["kill"]
    assert bvs["rho_star"] == pytest.approx(0.7 / 3)
    assert shl["star"] == "SHL-g0.90-B4" and shl["killed"] == "8.4"                 # no fallback (8.5)
    v1 = rec["v1.json"]
    assert v1["bvs_star"] == "BVS-SQ-k0.1" and v1["shl_star"] is None and v1["shl_killed"] == "8.4"
    assert v1["r_ref"] == {PG1: pytest.approx(26.0), CG1: pytest.approx(20.0), PC1: pytest.approx(40.0)}
    assert v1["m_phs"] == {PG1: pytest.approx(3.0), CG1: pytest.approx(1.0), PC1: pytest.approx(6.0)}
    assert v1["family"] == ["H_BVS"] and v1["rho_star"] == pytest.approx(0.7 / 3)
    assert not any(k["kill"] for k in v1["kill_c"]["BVS"].values())


def test_kill_rule_c_and_stage0_kill(study, tmp_path):
    root = str(tmp_path / "m")
    shutil.copytree(study[0], root)
    for s in SEL:                     # BVS* over 30 on PointCircle1 while B* is at 20: killed under (c)
        write_run(os.path.join(root, "p2"), "BVS-SQ-k0.1", PC1, s, 35.0, 32.0)
    v1 = an.candidates(root)
    assert v1["bvs_star"] is None and v1["bvs_selected"] == "BVS-SQ-k0.1"
    assert v1["bvs_killed"] == f"8.1(c): {PC1}" and v1["family"] == [] and v1["m_phs"] is None
    # Stage 0 failing kills BVS before any BVS run is read
    v05 = json.load(open(os.path.join(root, "v05.json")))
    v05["g0b"]["proceed"], v05["bvs_stage0_proceed"] = False, False
    json.dump(v05, open(os.path.join(root, "v05.json"), "w"))
    shutil.rmtree(os.path.join(root, "p1", train.stem_of("BVS-SQ-k0.1", PG1, 0) + ".snap"))
    p1 = an.candidate_p1(root, "BVS", configs.BVS_CONFIGS, v05, {"R": np.full(6, 16.1), "C": np.full(6, 22.0)})
    assert p1["star"] is None and p1["killed"] == "Stage 0 (g0b)"


def test_m_phs_pools_finite_seeds_and_two_are_needed(study, tmp_path):
    """Protocol section 6 (M7): m_PHS pools the seeds with a finite 1M m_dep; with fewer than 2 on a task,
    BVS* is killed under 8.1(c) and no m_PHS (nor rho*) is recorded."""
    root = str(tmp_path / "m")
    shutil.copytree(study[0], root)
    write_snapshot(os.path.join(root, "p2"), train.stem_of("BVS-SQ-k0.1", PC1, 0), 1_000_000, m_dep=float("nan"))
    v1 = an.candidates(root)
    assert v1["bvs_star"] == "BVS-SQ-k0.1" and v1["m_phs"][PC1] == pytest.approx(6.5)     # seeds 1, 2: 6 and 7
    assert len(v1["m_phs_excluded"][PC1]) == 1 and v1["m_phs_excluded"][PG1] == []
    write_snapshot(os.path.join(root, "p2"), train.stem_of("BVS-SQ-k0.1", PC1, 1), 1_000_000, m_dep=float("nan"))
    v1 = an.candidates(root)
    assert v1["bvs_star"] is None and v1["bvs_killed"].startswith(f"8.1(c): {PC1} (m_PHS")
    assert v1["m_phs"] is None and v1["rho_star"] is None and v1["family"] == []


def test_a_non_finite_m_dep_refuses(tmp_path):
    stem = train.stem_of("BVS-SQ-k0.1", PG1, 0)
    write_snapshot(tmp_path, stem, 1_000_000, m_dep=float("nan"))
    with pytest.raises(an.Refusal, match="not finite"):
        an.m_dep_at_1m({"dir": str(tmp_path), "stem": stem, "id": stem})


# --- plan and cuts --------------------------------------------------------------------------------------
def full_v1(**kw):
    v = {"b_star": "B5", "bvs_star": "BVS-SQ-k0.1", "shl_star": "SHL-g0.90-B4", "saute_star": "SauteB",
         "l_star": "B7", "run_saute_star": True, "run_l_star": True, "rho_star": 0.123456789,
         "m_phs": {t: 1.0 for t in an.TASKS}, "r_ref": {t: 20.0 for t in an.TASKS}, "family": ["H_BVS"],
         "bvs_killed": None, "shl_killed": None}
    v.update(kw)
    return v


def test_plan_lists_every_run_and_output_with_cut_groups(tmp_path):
    root = str(tmp_path)
    json.dump(full_v1(), open(os.path.join(root, "v1.json"), "w"))
    plan = an.make_plan(root)
    runs = plan["runs"]
    count = lambda arm, stage="conf": sum(r["arm"] == arm and r["stage"] == stage for r in runs)
    assert [count(a) for a in ("B*", "BVS*", "BVSU", "BVSD", "SHL*", "R1", "Saute*", "L*")] == \
        [48, 48, 12, 12, 36, 24, 24, 24]                                       # 228 (protocol section 19)
    assert [count(a, "cont3m") for a in ("B*", "BVS*", "SHL*")] == [3, 3, 3]
    assert len({r["id"] for r in runs}) == len(runs)
    by = {r["id"]: r for r in runs}
    assert all(r["drop_ckpt"] == 0 for r in runs if r["arm"] == "B*" and r["stage"] == "conf")
    assert by[f"conf/BVS-SQ-k0.1_{PG1}_s302"]["drop_ckpt"] == 0                  # kept for its continuation
    assert by[f"conf/BVS-SQ-k0.1_{PG1}_s303"]["drop_ckpt"] == 1
    assert by[f"conf/BVS-SQ-k0.1_{CG1}_s300"]["drop_ckpt"] == 1
    u = by[f"conf/BVSU_{PG1}_s300"]
    assert u["argv"][-4:] == ["--from-bvs", "BVS-SQ-k0.1", "--rho-star", repr(0.123456789)]
    assert train.parse(u["argv"][2:]).rho_star == 0.123456789                  # the trainer accepts the argv
    c = by[f"cont3m/B5_{PG1}_s300"]
    assert c["argv"][-3:] == ["--steps", "3000000", "--resume"] and c["cut_group"] == "1"
    assert c["resume_from"].endswith(os.path.join("conf", f"B5_{PG1}_s300.ckpt.pt"))
    groups = {g: sorted({(r["arm"], r["seed"]) for r in runs if r["cut_group"] == g and r["stage"] == "conf"})
              for g in ("2", "2b", "3", "5")}
    assert {s for a, s in groups["2"]} == set(range(308, 312)) and {a for a, s in groups["2"]} == {"SHL*"}
    assert {s for a, s in groups["2b"]} == set(range(300, 308))
    assert {a for a, s in groups["3"]} == {"R1", "Saute*", "L*"} and {s for a, s in groups["3"]} == {306, 307}
    assert {a for a, s in groups["5"]} == {"BVSU", "BVSD"} and {s for a, s in groups["5"]} == set(range(308, 312))
    assert not any(r["cut_group"] for r in runs if r["arm"] in ("B*", "BVS*") and r["stage"] == "conf")
    ev = plan["evaluations"]
    kinds = lambda k, m=None: sum(e["kind"] == k and (m is None or e.get("mode") == m) for e in ev)
    assert kinds("heldout", "phs") == kinds("phs_fit") == kinds("phs_quality") == 48
    assert kinds("heldout", "s_near") == kinds("heldout", "strict") == kinds("heldout", "shield_off") == 48
    assert kinds("heldout", "r3") == 10 and kinds("heldout", "deployed") == 228 + 9
    outs = [e["output"] for e in ev]
    assert len(set(outs)) == len(outs)                                          # no two evaluations share a file
    assert os.path.join("sealed", "heldout", f"B5_{PG1}_s300.deployed.n3000000.json.enc") in outs
    # an S-near BVS* has no s_near mode; no kill means no extra continuations
    json.dump(full_v1(bvs_star="BVS-SN-k0.1"), open(os.path.join(root, "v1.json"), "w"))
    assert not any(e.get("mode") == "s_near" for e in an.make_plan(root)["evaluations"])
    with pytest.raises(an.Refusal, match="freed"):
        an.make_plan(root, extra_continuations=True)


def test_plan_of_the_pilot_chain(study):
    _, rec = study
    plan = rec["plan.json"]
    arms = {r["arm"] for r in plan["runs"]}
    assert arms == {"B*", "BVS*", "BVSU", "BVSD", "R1", "Saute*"}               # SHL* killed; L* is Saute*
    assert {r["config"] for r in plan["runs"] if r["arm"] == "Saute*"} == {"SauteA"}
    assert json.load(open(os.path.join(study[0], "cuts.json"))) == []


def test_cuts_follow_the_list_order_and_stop_at_unblinding(tmp_path, capsys):
    root = str(tmp_path)
    json.dump(full_v1(), open(os.path.join(root, "v1.json"), "w"))
    json.dump(an.make_plan(root), open(os.path.join(root, "plan.json"), "w"))
    json.dump([], open(os.path.join(root, "cuts.json"), "w"))
    for bad, why in (("3", "item 1"), ("4", "section 22"), ("7", "unknown")):
        with pytest.raises(an.Refusal, match=why):
            an.cut(root, bad)
    assert an.main(["cut", "1", "--root", root]) == 0
    with pytest.raises(an.Refusal, match="already"):
        an.cut(root, "1")
    with pytest.raises(an.Refusal, match="item 2 "):
        an.cut(root, "2b")
    for item in ("2", "2b", "3", "5"):
        an.cut(root, item)
    cuts = json.load(open(os.path.join(root, "cuts.json")))
    assert [c["item"] for c in cuts] == ["1", "2", "2b", "3", "5"] and all(c["reason"] == "throughput" for c in cuts)
    assert [c["stage"] for c in cuts] == ["cont3m", "conf", "conf", "conf", "conf"]
    assert len(cuts[0]["stems"]) == 9 and len(cuts[3]["stems"]) == 3 * 2 * 3 and len(cuts[4]["stems"]) == 8
    assert an.cut_ids(cuts[:1]) == {r["id"] for r in json.load(open(os.path.join(root, "plan.json")))["runs"]
                                    if r["stage"] == "cont3m"}
    open(os.path.join(root, "UNBLINDED.txt"), "w").close()
    json.dump([], open(os.path.join(root, "cuts.json"), "w"))
    with pytest.raises(an.Refusal, match="UNBLINDED"):
        an.cut(root, "1")


def test_no_subcommand_before_unblinding_opens_sealed(study, tmp_path, monkeypatch):
    root = str(tmp_path / "m")
    shutil.copytree(study[0], root)
    for name in ("p1/top3.json", "v05.json", "p1/candidates.json", "v1.json", "plan.json", "cuts.json"):
        os.remove(os.path.join(root, name))
    real_open = builtins.open

    def guarded(file, *a, **k):
        if isinstance(file, (str, os.PathLike)) and "sealed" in os.path.realpath(file).split(os.sep):
            raise AssertionError(f"opened {file}")
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    for cmd in (["p1-baselines"], ["bstar"], ["candidates", "--stage", "p1"], ["candidates"], ["plan"], ["cut", "1"]):
        assert an.main(cmd + ["--root", root]) == 0, cmd


# --- the decision path: type-I error, guards, imputation, Holm ------------------------------------------
def fake_runs(R, C, prefix="x", state=None, diverged=None):
    state = state or ["finished"] * len(R)
    diverged = diverged or [False] * len(R)
    return [{"id": f"{prefix}{i}", "state": st, "diverged": dv, "n_resumes": 0,
             "out": None if st != "finished" else {"pooled": {"R": float(r), "C": float(c)}}}
            for i, (r, c, st, dv) in enumerate(zip(R, C, state, diverged))]


def study_runs(rng, n, x, y, p_miss=0.0):
    X, Y = [], []
    for _ in an.TASKS:
        for arm_args, out, pre in ((x, X, "x"), (y, Y, "y")):
            R, C = power.draw(rng, n, **arm_args)
            st = ["unfinished" if u < p_miss else "finished" for u in rng.random(n)]
            if all(s == "unfinished" for s in st):
                st[0] = "finished"
            out.append(fake_runs(R, C, pre, st))
    return X, Y


def _null_rates(seed, reps, p_miss):
    rng = np.random.default_rng(seed)
    a = power.arm(15, 25, 9, 25, 0.0, 0.2)
    beats = rej = 0
    for k in range(reps):
        X, Y = study_runs(rng, 16, a, a, p_miss=p_miss)
        res = an.decide(stats, X, Y, [power.R_REF] * 3, n_perm=400, seed=k)
        beats += an.confirmed(stats, res, ["H_BVS"])["beats"]
        rej += res["p_perm"] <= stats.ALPHA_TEST
    return beats / reps, rej / reps


def test_type_one_error_of_the_full_rule_under_a_simulated_null():
    """Identical arms (15 / 25, sd 9 / 25, 20% collapses, n = 16 per arm and task): the full decision path
    (imputation, permutation test at ALPHA_TEST, guards, Holm) declares 'beats' in at most the family level
    0.025 of the studies, and the test alone rejects at a rate in the range of Appendix G.1 (not a
    degenerate 0). With 5% unfinished runs in both arms the worst-case imputation keeps it below 0.025."""
    beats, rej = _null_rates(20261011, 1500, 0.0)
    assert beats <= rej
    assert beats <= stats.ALPHA
    assert 0.004 <= rej <= stats.ALPHA
    beats, rej = _null_rates(5, 300, 0.05)
    assert beats <= stats.ALPHA and rej <= stats.ALPHA


def test_power_sanity_under_a_clear_gain():
    rng = np.random.default_rng(1)
    x, y = power.arm(23, 25, 6, 15, 0.0, 0.1), power.arm(15, 25, 6, 15, 0.0, 0.1)
    hits = sum(an.confirmed(stats, an.decide(stats, *study_runs(rng, 16, x, y), [power.R_REF] * 3, n_perm=400,
                                             seed=k), ["H_BVS"])["beats"] for k in range(40))
    assert hits >= 30


def test_guards_imputation_and_diverged_runs_through_the_decision_path():
    rr = [26.3] * 3
    base = [np.full(16, 15.0) + np.linspace(-1, 1, 16) for _ in range(3)]
    cost = [np.full(16, 22.0) + np.linspace(-2, 2, 16) for _ in range(3)]
    Y = [fake_runs(R, C, "y") for R, C in zip(base, cost)]
    X = [fake_runs(R + 8, C, "x") for R, C in zip(base, cost)]
    r = an.decide(stats, X, Y, rr, n_perm=2000)
    assert r["beats"] and an.confirmed(stats, r, ["H_BVS"])["p_holm"] == r["p_perm"]
    # cost above max(25, C_B*) + 5 in one task: the guard blocks the claim
    Xc = [fake_runs(base[0] + 8, cost[0] + 9, "x")] + X[1:]
    r = an.decide(stats, Xc, Y, rr, n_perm=2000)
    assert r["p_perm"] <= stats.ALPHA_TEST and not r["beats"] and r["verdict"] == "Not confirmed: guard failed (cost)"
    # three diverged candidate runs that finished: their finite endpoints are kept (not imputed, protocol
    # section 11) but they count as collapsed, so G-col fails
    div = [True] * 3 + [False] * 13
    Xd = [fake_runs(base[0] + 8, cost[0], "x", diverged=div)] + X[1:]
    r = an.decide(stats, Xd, Y, rr, n_perm=2000)
    assert r["guards"]["collapses_x"] == 3 and not r["guards"]["collapse"] and not r["beats"]
    assert r["imputed_x"][0] == []
    a = an.arrays(Xd[0])
    assert a["R"][:3] == pytest.approx(base[0][:3] + 8) and a["diverged"][:3].all()
    assert all(an._clean_run(x) for x in Xd[0])                # drop_imputed keeps them
    # diverged runs that stopped (no held-out endpoint) are imputed R 0 / worst cost
    Xs = [fake_runs(base[0] + 8, cost[0], "x", state=["diverged"] * 3 + ["finished"] * 13, diverged=div)] + X[1:]
    r = an.decide(stats, Xs, Y, rr, n_perm=2000)
    assert r["imputed_x"][0] == ["x0", "x1", "x2"] and r["guards"]["collapses_x"] == 3
    assert not any(an._clean_run(x) for x in Xs[0][:3])
    # an unfinished B* run takes B*'s median; an unfinished candidate run the worst case
    imp = an.impute(stats, [fake_runs([10.0, 11.0], [5.0, 6.0], state=["finished", "unfinished"])],
                    [fake_runs([12.0, 14.0, 13.0], [20.0, 21.0, 9.0], "y", state=["finished", "finished", "unfinished"])])[0]
    assert imp["Rx"][1] == 0.0 and imp["Cx"][1] == 21.0
    assert imp["Ry"][2] == pytest.approx(13.0) and imp["Cy"][2] == pytest.approx(20.5)
    with pytest.raises(an.Refusal):
        an.confirmed(stats, r, [])


def test_contrast_reports_secondary_endpoints():
    rng = np.random.default_rng(0)
    from safe2.tests.analyze_helpers import heldout_output

    def runs(R, C, pre):
        return [{"id": f"{pre}{i}", "state": "finished", "diverged": False, "n_resumes": 0,
                 "out": heldout_output("s", "deployed", r, c, rng)} for i, (r, c) in enumerate(zip(R, C))]

    X = [runs([20, 21, 22, 23], [20, 22, 24, 26], "x") for _ in range(3)]
    Y = [runs([15, 16, 17, 18], [20, 22, 24, 26], "y") for _ in range(3)]
    c = an.contrast(X, Y, [20.0] * 3, list(an.TASKS))
    assert c["pooled"]["ds"]["stat"] == pytest.approx(5.0 / 20.0)
    assert c["pooled"]["dc"]["stat"] == pytest.approx(0.0, abs=1e-12)
    assert c["per_task"][PG1]["S_x"] == pytest.approx(21.5 / 20.0)
    assert c["pooled"]["dviolation"] is not None and c["pooled"]["dcvar10"] is not None
    assert c["pooled"]["collapses_x"] == 0 and c["pooled"]["fisher_collapse"] == 1.0


def test_the_launcher_reads_the_plan_and_the_cuts(tmp_path, monkeypatch):
    """Contract with safe2.launch --from-plan: every planned run's argv is a complete trainer command under
    the queue directory, and the cut entries remove exactly the cut runs of their stage."""
    launch = pytest.importorskip("safe2.launch")
    root = str(tmp_path)
    json.dump(full_v1(), open(os.path.join(root, "v1.json"), "w"))
    plan = an.make_plan(root)
    json.dump(plan, open(os.path.join(root, "plan.json"), "w"))
    json.dump([], open(os.path.join(root, "cuts.json"), "w"))
    monkeypatch.setattr(launch, "QDIR", os.path.realpath(root))
    for stage, n in (("conf", 228), ("cont3m", 9)):
        keep, n_cut = launch.load_plan(os.path.join(root, "plan.json"), stage)
        assert len(keep) == n and n_cut == 0
        assert {s["stem"] for _, s in keep} == {r["id"].split("/")[1] for r in plan["runs"] if r["stage"] == stage}
    an.cut(root, "1")
    an.cut(root, "2")
    keep, n_cut = launch.load_plan(os.path.join(root, "plan.json"), "cont3m")
    assert keep == [] and n_cut == 9
    keep, n_cut = launch.load_plan(os.path.join(root, "plan.json"), "conf")
    assert n_cut == 12 and len(keep) == 228 - 12
    assert all(r["arm"] != "SHL*" or r["seed"] < 308 for r, _ in keep)


def test_a_continuation_of_a_diverged_or_unfinished_run_is_not_run(tmp_path, monkeypatch):
    """review-ops M3: a cont3m run whose conf source diverged or stopped unfinished is never started
    (SPEC 3.6), so it is 'not-run (source <state>)', which completeness accepts; one of a finished source
    that was never started stays 'missing'."""
    monkeypatch.setattr(an, "pending_runs", lambda: set())
    root = str(tmp_path / "m")
    conf = os.path.join(root, "conf")
    write_run(conf, "B5", PG1, 300, diverge_at=300_000, upto=400_000, finished=False)
    write_run(conf, "B5", PG1, 301, upto=600_000, finished=False, n_resumes=3)
    write_run(conf, "B5", PG1, 302)
    plan = {"runs": [{"id": f"{st}/{train.stem_of('B5', PG1, s)}", "stage": st, "arm": "B*", "config": "B5",
                      "env": PG1, "seed": s, "steps": n}
                     for st, n in (("conf", 1_000_000), ("cont3m", 3_000_000)) for s in (300, 301, 302)]}
    runs = an.plan_runs(root, plan, set())
    states = {s: runs[f"cont3m/{train.stem_of('B5', PG1, s)}"]["state"] for s in (300, 301, 302)}
    assert states == {300: "not-run (source diverged)", 301: "not-run (source unfinished)", 302: "missing"}
    assert not any(st in an.INCOMPLETE for s, st in states.items() if s != 302)
