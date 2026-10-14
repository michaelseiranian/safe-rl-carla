"""Opt-in smoke run of the Stage-0 plumbing on real training files (minutes; skipped unless SAFE2_SMOKE=1).

    SAFE2_SMOKE=1 SAFE2_SMOKE_OUT=<scratch dir> python -m pytest safe2/tests/test_probe_smoke.py

A 26k-step dev run (seed 952, hidden 64) of `lagu.train --arm td3lag --recipe omnisafe` and of safe2's B1:
G0b's replay-identity diff must report "identical" over all 26,000 rows (B1 equals td3lag-omni until its
dual starts). Then G0a and G0b run end to end on the B1 files, on the development layouts, with the fake
posthoc backend. Output goes to SAFE2_SMOKE_OUT (never under results/)."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from safe2 import configs, probe, train

pytestmark = pytest.mark.skipif(os.environ.get("SAFE2_SMOKE") != "1", reason="set SAFE2_SMOKE=1 to run")
ENV = "SafetyPointGoal1-v0"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_g0_plumbing_on_real_runs(tmp_path):
    base = os.environ.get("SAFE2_SMOKE_OUT")
    assert base is None or "/results/" not in os.path.realpath(base) + "/"
    out = os.path.join(base, "probe_smoke") if base else str(tmp_path)
    lagu_dir, b1_dir = os.path.join(out, "lagu"), os.path.join(out, "safe2")
    subprocess.run([sys.executable, "-m", "lagu.train", "--env", ENV, "--arm", "td3lag", "--recipe", "omnisafe",
                    "--tag", "omni", "--seed", "952", "--steps", "26000", "--hidden", "64", "--eval-episodes", "2",
                    "--out", lagu_dir], check=True, cwd=ROOT, capture_output=True,
                   env={**os.environ, "OMP_NUM_THREADS": "1"})
    stem = train.run(configs.make("B1", hidden=64, eval_episodes=2), ENV, 952, b1_dir, steps=26_000)
    lagu_ckpt = os.path.join(lagu_dir, f"td3lag-omni_{ENV}_s952.ckpt.pt")
    assert probe.replay_identity(os.path.join(b1_dir, stem + ".ckpt.pt"), lagu_ckpt, 26_000) == \
        {"result": "identical", "rows": 26_000}

    g0a = probe.g0a(b1_dir, os.path.join(out, "stage0"), probe.G0aConfig(block="dev", episodes=1, B=0.0),
                    run_seeds=(952,), workers=1, force=True)       # force: SAFE2_SMOKE_OUT may be reused
    assert g0a["n_episodes"] == 1
    cfg = probe.G0bConfig(block="dev", identity_rows=26_000, fit_episodes=(2, 20), holdout_episodes=(21, 26),
                          updates=10, states_per_seed=30, max_episodes=3, min_states=10)
    g0b = probe.g0b(b1_dir, lagu_dir, os.path.join(out, "stage0"), cfg, run_seeds=(952,), workers=1,
                    backend="safe2.tests.fake_posthoc", force=True)
    assert g0b["identity"][stem]["result"] == "identical" and set(g0b["decision"]) == {"pass_i", "pass_ii", "proceed"}
