"""Opt-in smoke runs on the real task (minutes each; skipped unless SAFE2_SMOKE=1).

    SAFE2_SMOKE=1 SAFE2_SMOKE_OUT=<scratch dir> python -m pytest safe2/tests/test_smoke.py [-k NAME]

* test_smoke_base_arm: a 20k-step run of every base config (hidden 64; warm-up 5k and dual from 10k
  so learning and every dual act inside 20k steps): two finite rows, every whitelisted column written,
  snapshots, checkpoint and final file.
* test_snapshot_round_trip_30k: SPEC test 25 in its stated form, a 30k-step dev run with --hidden 64
  --eval-every 10000 --snap-every 10000 --snap-from 0 through the CLI; the held-out evaluator's policy
  replays every in-training evaluation (20 episodes) bit for bit.
Output goes to SAFE2_SMOKE_OUT (never under results/)."""
from __future__ import annotations

import csv
import os

import pytest

from safe2 import configs, evaluate, train

pytestmark = pytest.mark.skipif(os.environ.get("SAFE2_SMOKE") != "1", reason="set SAFE2_SMOKE=1 to run")
ENV = "SafetyPointGoal1-v0"
BASE_ARMS = [n for n, c in configs.CONFIGS.items() if c.kind == "base"]


def out_dir(tmp_path, name):
    root = os.environ.get("SAFE2_SMOKE_OUT")
    assert root is None or "/results/" not in os.path.realpath(root) + "/"
    return os.path.join(root, name) if root else str(tmp_path / name)


@pytest.mark.parametrize("name", BASE_ARMS)
def test_smoke_base_arm(tmp_path, name):
    rc = configs.make(name, hidden=64, start_steps=5000, dual_start=10_000, eval_every=10_000, eval_episodes=2,
                      snap_every=10_000, snap_from=0.0, ckpt_every=10_000)
    out = out_dir(tmp_path, name)
    stem = train.run(rc, ENV, 950, out, steps=20_000)
    with open(os.path.join(out, stem + ".csv")) as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [10_000, 20_000]
    last = rows[-1]
    assert int(last["updates"]) == 15_000 and last["nonfinite_actions"] == "0"
    assert all(r[g] == "1" for r in rows for g in train.GROUPS), rows
    for col in train.whitelist(rc):
        assert last[col] not in ("", "nan"), col
    if rc.dual != "none":
        assert last["jc"] != ""
    assert sorted(os.listdir(os.path.join(out, stem + ".snap"))) == ["10000.pt", "20000.pt"]
    assert os.path.exists(os.path.join(out, stem + ".pt")) and os.path.exists(os.path.join(out, stem + ".ckpt.pt"))


@pytest.mark.parametrize("name", ["B5", "SauteA"])
def test_snapshot_round_trip_30k(tmp_path, name):
    out = out_dir(tmp_path, "rt30k_" + name)
    seen = {}
    rc = train.build_config(train.parse(["--config", name, "--env", ENV, "--seed", "951", "--out", out,
                                         "--steps", "30000", "--hidden", "64", "--eval-every", "10000",
                                         "--snap-every", "10000", "--snap-from", "0"]))
    stem = train.run(rc, ENV, 951, out, steps=30_000, on_eval=lambda step, eps: seen.__setitem__(step, eps))
    env = train._make_env(ENV)
    assert sorted(seen) == [10_000, 20_000, 30_000]
    for step, eps in seen.items():
        policy = evaluate.policy_from_snapshot(os.path.join(out, stem + ".snap", f"{step}.pt"))
        again = evaluate.rollout(env, policy, "train_eval", range(rc.eval_episodes))
        assert len(again) == 20
        for e1, e2 in zip(eps, again):
            assert {k: v for k, v in e1.items() if k != "traj"} == e2
