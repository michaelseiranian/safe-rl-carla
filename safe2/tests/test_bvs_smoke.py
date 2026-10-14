"""Opt-in smoke runs of the BVS family on the real task (minutes each; skipped unless SAFE2_SMOKE=1).

    SAFE2_SMOKE=1 SAFE2_SMOKE_OUT=<scratch dir> python -m pytest safe2/tests/test_bvs_smoke.py [-k NAME]

* test_smoke_bvs_variant: a 20k-step run of each BVS config, BVSU and BVSD (hidden 64; warm-up 5k,
  shield from 8k, margin dual from 10k, BVS-U's quantile T from 12k, so every mechanism acts inside
  20k steps): two finite rows, every whitelisted column written, the shield counters moving,
  snapshots, checkpoint and final file. BVSU uses rho* = 0.05, a development value (the study's rho*
  comes from v1.json).
* test_bvs_snapshot_round_trip_30k: SPEC test 25 in its stated form for the BVS family, a 30k-step dev
  run with --hidden 64 --eval-every 10000 --snap-every 10000 --snap-from 0 through the CLI parser; the
  held-out evaluator's policy replays every in-training evaluation (20 episodes, shield on, m_dep,
  layout-seeded sampler) bit for bit.
Output goes to SAFE2_SMOKE_OUT (never under results/)."""
from __future__ import annotations

import csv
import os

import pytest

from safe2 import configs, evaluate, train
from safe2.tests.helpers import same

pytestmark = pytest.mark.skipif(os.environ.get("SAFE2_SMOKE") != "1", reason="set SAFE2_SMOKE=1 to run")
ENV = "SafetyPointGoal1-v0"
VARIANTS = configs.BVS_CONFIGS + ("BVSU", "BVSD")
DEV_RHO = 0.05


def out_dir(tmp_path, name):
    root = os.environ.get("SAFE2_SMOKE_OUT")
    assert root is None or "/results/" not in os.path.realpath(root) + "/"
    return os.path.join(root, name) if root else str(tmp_path / name)


def config(name: str, **kw) -> configs.RunConfig:
    if name in configs.DERIVED:
        return configs.from_bvs("BVS-SQ-k0.1", configs.DERIVED[name], DEV_RHO, **kw)
    return configs.make(name, **kw)


@pytest.mark.parametrize("name", VARIANTS)
def test_smoke_bvs_variant(tmp_path, name):
    rc = config(name, hidden=64, start_steps=5000, shield_start=8000, margin_start=10_000, u_start=12_000,
                eval_every=10_000, eval_episodes=2, snap_every=10_000, snap_from=0.0, ckpt_every=10_000)
    out = out_dir(tmp_path, name)
    stem = train.run(rc, ENV, 952, out, steps=20_000)
    with open(os.path.join(out, stem + ".csv")) as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [10_000, 20_000]
    last = rows[-1]
    assert int(last["updates"]) == 15_000 and last["nonfinite_actions"] == "0"
    assert all(r[g] == "1" for r in rows for g in train.GROUPS), rows
    for col in train.whitelist(rc):
        assert last[col] not in ("", "nan"), col
    assert int(last["shield_steps_cum"]) == 12_000 and int(last["int_cum"]) > 0
    assert float(last["m_tilde"]) != rc.m0 and int(last["ep_cum"]) == 12
    if rc.kind == "bvs_u":
        assert float(last["T"]) != rc.T0 and int(last["trig_cum"]) > 0
        assert last["trig007_num"] == "0"                 # counts only steps > 35,000 (prereg 14.3)
    assert sorted(os.listdir(os.path.join(out, stem + ".snap"))) == ["10000.pt", "20000.pt"]
    assert os.path.exists(os.path.join(out, stem + ".pt")) and os.path.exists(os.path.join(out, stem + ".ckpt.pt"))


@pytest.mark.parametrize("name", ["BVS-SQ-k0.1", "BVSU"])
def test_bvs_snapshot_round_trip_30k(tmp_path, name):
    out = out_dir(tmp_path, "rt30k_" + name)
    seen = {}
    argv = ["--config", name, "--env", ENV, "--seed", "953", "--out", out, "--steps", "30000", "--hidden", "64",
            "--eval-every", "10000", "--snap-every", "10000", "--snap-from", "0"]
    if name == "BVSU":
        argv += ["--from-bvs", "BVS-SQ-k0.1", "--rho-star", str(DEV_RHO)]
    rc = train.build_config(train.parse(argv))
    stem = train.run(rc, ENV, 953, out, steps=30_000, on_eval=lambda step, eps: seen.__setitem__(step, eps))
    env = train._make_env(ENV)
    assert sorted(seen) == [10_000, 20_000, 30_000]
    for step, eps in seen.items():
        policy = evaluate.policy_from_snapshot(os.path.join(out, stem + ".snap", f"{step}.pt"))
        again = evaluate.rollout(env, policy, "train_eval", range(rc.eval_episodes))
        assert len(again) == 20
        assert same(eps, again), step
    assert sum(e["interventions"] for e in seen[30_000]) > 0
