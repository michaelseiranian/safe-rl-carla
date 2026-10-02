"""Held-out evaluation (lagu/evaluate.py) on synthetic runs: snapshot choice, layout blocks, weight
loading, pooling, skip/force rules, and identical results in-process and across worker processes."""
import json
import os
from dataclasses import asdict

import pytest
import torch

safety_gymnasium = pytest.importorskip("safety_gymnasium")

from lagu import evaluate as ho
from lagu.agent import Config, LagU
from lagu.train import ARMS, EVAL_SEED, save_snapshot

ENV = "SafetyPointCircle1-v0"   # 500-step episodes keep the real rollouts short
OBS, ACT = 28, 2


def make_run(d, saved, total, arm="lagu", seed=0, **cfg_kw):
    stem = os.path.join(str(d), f"{arm}_{ENV}_s{seed}")
    cfg = Config(hidden=16, **{**ARMS[arm], **cfg_kw})
    with open(stem + ".json", "w") as f:
        json.dump({"args": {"env": ENV, "arm": arm, "seed": seed, "steps": total, "tag": ""},
                   "cfg": asdict(cfg), "horizon": 500, "obs_dim": OBS, "act_dim": ACT}, f)
    for step in saved:
        torch.manual_seed(1000 * seed + step)       # every snapshot has different weights
        save_snapshot(stem + ".snap", LagU(OBS, ACT, cfg), step)
    return stem


def load(stem):
    with open(stem + ".heldout.json") as f:
        return json.load(f)


def test_pick_takes_the_nearest_saved_step_to_each_target():
    every50k = list(range(50_000, 2_000_001, 50_000))
    assert [s for _, s in ho.pick(every50k, 1_000_000, 5)] == [800_000, 850_000, 900_000, 950_000, 1_000_000]
    assert [s for _, s in ho.pick(every50k, 2_000_000, 5)] == [1_600_000, 1_700_000, 1_800_000, 1_900_000,
                                                              2_000_000]
    assert [s for _, s in ho.pick([10_000, 20_000], 20_000, 5)] == [20_000] * 5    # 20k smoke run
    assert ho.pick([10, 20], 15 / 0.8, 2)[0] == (15.0, 10)                           # a tie goes earlier
    assert ho.pick(every50k, 1_000_000, 1) == [(1_000_000.0, 1_000_000)]


def test_layout_blocks_weights_and_pooling(tmp_path, monkeypatch):
    stem = make_run(tmp_path, [700, 800, 850, 900, 950, 1000], 1000, cost_twin=True)
    snaps = {s: torch.load(f"{stem}.snap/{s}.pt", weights_only=True) for s in ho.snapshot_steps(stem + ".snap")}
    calls, loaded = [], []

    def same(agent, snap):
        return all(torch.equal(getattr(agent, n).state_dict()[k], v)
                   for n in ("actor", "members", "cost", "cost2") for k, v in snap[n].items())

    def fake(env, agent, episodes, cost_limit, gamma, horizon, seed_base):
        loaded.append([s for s, snap in snaps.items() if same(agent, snap)])   # which weights it holds
        calls.append((seed_base, episodes, cost_limit, gamma, horizon))
        return {"eval_return": float(seed_base), "eval_cost": 1.0, "eval_violation": float(len(calls) % 2),
                "qc_traj": 2.0, "mc_cost_traj": 4.0}

    monkeypatch.setattr(ho, "evaluate", fake)
    assert ho.main([str(tmp_path), "--workers", "1", "--episodes", "7"]) == 0
    assert calls == [(EVAL_SEED + 1000 + 20 * i, 7, 25.0, 0.99, 500) for i in range(5)]
    assert loaded == [[800], [850], [900], [950], [1000]]
    out = load(stem)
    assert [s["step"] for s in out["per_snapshot"]] == [800, 850, 900, 950, 1000]
    assert [s["seed_base"] for s in out["per_snapshot"]] == [11_000, 11_020, 11_040, 11_060, 11_080]
    assert (out["end_step"], out["snapshots"], out["episodes"]) == (1000, 5, 7)
    assert out["pooled"]["eval_return"] == pytest.approx(11_040.0)
    assert out["pooled"]["eval_violation"] == pytest.approx(0.6)
    assert out["pooled"]["qc_traj"] == 2.0 and out["pooled"]["mc_cost_traj"] == 4.0

    # an explicit end step moves the window; a result made with other settings is never reused silently
    before = open(stem + ".heldout.json").read()
    assert ho.main([str(tmp_path), "--workers", "1", "--episodes", "7"]) == 0          # same settings: skip
    assert ho.main([str(tmp_path), "--workers", "1", "--end-step", "900"]) == 1        # stale: left alone
    assert open(stem + ".heldout.json").read() == before and len(calls) == 5
    assert ho.main([str(tmp_path), "--workers", "1", "--end-step", "900", "--snapshots", "3", "--force"]) == 0
    assert [s["step"] for s in load(stem)["per_snapshot"]] == [700, 800, 900]            # 720, 810, 900
    assert loaded[5:] == [[700], [800], [900]]


def test_incomplete_runs_and_old_directories_are_skipped(tmp_path):
    stem = make_run(tmp_path, [500, 1000], 2000)                  # still training toward 2000
    old = tmp_path / "old"
    old.mkdir()
    (old / "launch.json").write_text("[]")
    (old / f"td3lag_{ENV}_s0.json").write_text(json.dumps({"args": {"steps": 10}}))
    assert ho.main([str(tmp_path), str(old), "--workers", "4"]) == 0
    assert not os.path.exists(stem + ".heldout.json")
    assert sorted(os.listdir(old)) == sorted(["launch.json", f"td3lag_{ENV}_s0.json"])
    with pytest.raises(SystemExit):
        ho.main([str(tmp_path), "--episodes", "21"])             # would overlap the next layout block


def test_real_rollouts_match_in_process_and_across_workers(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        make_run(d, [150, 200], 200, arm="lagu", seed=0)
        make_run(d, [150, 200], 200, arm="td3lag", seed=1)
    common = ["--snapshots", "2", "--episodes", "1"]
    assert ho.main([str(a), "--workers", "1"] + common) == 0
    assert ho.main([str(b), "--workers", "2"] + common) == 0
    for name in (f"lagu_{ENV}_s0", f"td3lag_{ENV}_s1"):
        ra, rb = load(os.path.join(a, name)), load(os.path.join(b, name))
        assert ra == rb
        s0, s1 = ra["per_snapshot"]
        assert (s0["step"], s1["step"]) == (150, 200) and (s0["seed_base"], s1["seed_base"]) == (11_000, 11_020)
        for k in ("eval_return", "eval_cost", "eval_violation", "qc_traj", "mc_cost_traj", "qbar_traj",
                  "mc_ret_traj"):
            assert ra["pooled"][k] == pytest.approx((s0[k] + s1[k]) / 2), k


def test_a_reloaded_snapshot_reproduces_the_in_training_eval_row(tmp_path, request):
    """load_agent restores every network evaluate() reads (twin cost critic included)."""
    import csv
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "lagu.train", "--env", ENV, "--arm", "lagu", "--seed", "0",
                          "--steps", "1000", "--eval-every", "1000", "--eval-episodes", "1", "--hidden", "16",
                          "--start-steps", "300", "--cost-twin", "1", "--out", str(tmp_path)],
                         cwd=request.config.rootpath, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    stem = os.path.join(tmp_path, f"lagu_{ENV}_s0")
    with open(stem + ".json") as f:
        meta = json.load(f)
    with open(stem + ".csv") as f:
        row = list(csv.DictReader(f))[-1]
    agent = ho.load_agent(meta, torch.load(stem + ".snap/1000.pt", weights_only=True))
    ev = ho.evaluate(safety_gymnasium.make(ENV), agent, 1, agent.cfg.cost_limit, agent.cfg.gamma, meta["horizon"])
    assert {k: float(row[k]) for k in ev} == ev


def fake_evaluate(env, agent, episodes, cost_limit, gamma, horizon, seed_base):
    return {"eval_return": float(seed_base), "eval_cost": 0.0, "eval_violation": 0.0, "qc_traj": 1.0,
            "mc_cost_traj": 1.0}


def test_orphan_snapshots_missing_directories_and_extended_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ho, "evaluate", fake_evaluate)
    good = make_run(tmp_path, [150, 200], 200)
    orphan = tmp_path / "orphan_x_s0.snap"                     # snapshots whose json is gone: not a run
    orphan.mkdir()
    torch.save({}, orphan / "200.pt")
    assert ho.main([str(tmp_path), "--workers", "1", "--snapshots", "2"]) == 0
    assert os.path.exists(good + ".heldout.json") and not os.path.exists(tmp_path / "orphan_x_s0.heldout.json")
    assert ho.main([str(tmp_path / "typo"), "--workers", "1"]) == 1      # a mistyped directory is an error

    # extended with --resume past the --steps its json records: the default window would be wrong
    (tmp_path / "ext").mkdir()
    ext = make_run(tmp_path / "ext", [800, 900, 1000, 1500, 2000], 1000)
    assert ho.main([str(tmp_path / "ext"), "--workers", "1"]) == 1
    assert not os.path.exists(ext + ".heldout.json") and "--end-step" in capsys.readouterr().out
    assert ho.main([str(tmp_path / "ext"), "--workers", "1", "--end-step", "2000", "--snapshots", "2"]) == 0
    assert [s["step"] for s in load(ext)["per_snapshot"]] == [1500, 2000]


def test_a_heldout_made_from_other_weights_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(ho, "evaluate", fake_evaluate)
    stem = make_run(tmp_path, [100, 150, 200], 200, cost_twin=True)
    assert ho.main([str(tmp_path), "--workers", "1", "--snapshots", "2"]) == 0      # picks 150 and 200
    before = open(stem + ".heldout.json").read()
    cfg = Config(hidden=16, **{**ARMS["lagu"], "cost_twin": True})
    torch.manual_seed(7)
    save_snapshot(stem + ".snap", LagU(OBS, ACT, cfg), 100)                          # not in the window
    assert ho.main([str(tmp_path), "--workers", "1", "--snapshots", "2"]) == 0
    save_snapshot(stem + ".snap", LagU(OBS, ACT, cfg), 200)                          # retrained in place
    assert ho.main([str(tmp_path), "--workers", "1", "--snapshots", "2"]) == 1
    assert open(stem + ".heldout.json").read() == before
    assert ho.main([str(tmp_path), "--workers", "1", "--snapshots", "2", "--force"]) == 0
    assert ho.main([str(tmp_path), "--workers", "1", "--snapshots", "2"]) == 0


def test_a_snapshot_missing_a_network_is_an_error_not_random_weights(tmp_path):
    stem = make_run(tmp_path, [200], 200, cost_twin=True)
    with open(stem + ".json") as f:
        meta = json.load(f)
    snap = torch.load(stem + ".snap/200.pt", weights_only=True)
    ho.load_agent(meta, snap)
    for name in ("actor", "members", "cost", "cost2"):
        with pytest.raises((KeyError, TypeError, AttributeError)):
            ho.load_agent(meta, {**snap, name: None})
    torch.save({**snap, "cost2": None}, stem + ".snap/200.pt")
    assert ho.main([str(tmp_path), "--workers", "1", "--snapshots", "1"]) == 1
    assert not os.path.exists(stem + ".heldout.json")
