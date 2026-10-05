"""End-to-end trainer checks on the real Safety-Gymnasium task (a few seconds each)."""
import csv
import subprocess
import sys

import pytest

safety_gymnasium = pytest.importorskip("safety_gymnasium")

BASE = [sys.executable, "-m", "lagu.train", "--env", "SafetyPointGoal1-v0", "--eval-episodes", "1",
        "--hidden", "16", "--start-steps", "300", "--eval-every", "1000"]


def run(args, cwd):
    out = subprocess.run(BASE + args, cwd=cwd, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return out.stdout


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_resume_never_duplicates_rows_and_logs_trajectory_calibration(tmp_path, request):
    root = request.config.rootpath
    out = str(tmp_path)
    run(["--arm", "lagu", "--seed", "0", "--steps", "2000", "--out", out], root)
    run(["--arm", "lagu", "--seed", "0", "--steps", "3000", "--out", out, "--resume"], root)
    r = rows(f"{out}/lagu_SafetyPointGoal1-v0_s0.csv")
    assert [int(float(x["step"])) for x in r] == [1000, 2000, 3000]
    assert all(x["qc_traj"] not in ("", "nan") for x in r)


def test_resume_refuses_a_changed_config(tmp_path, request):
    root = request.config.rootpath
    out = str(tmp_path)
    run(["--arm", "td3lag", "--seed", "1", "--steps", "1000", "--out", out], root)
    bad = subprocess.run(BASE + ["--arm", "td3lag", "--seed", "1", "--steps", "2000", "--out", out,
                                 "--resume", "--lambda-lr", "0.5"],
                         cwd=root, capture_output=True, text=True, timeout=600)
    assert bad.returncode != 0 and "refusing to resume" in (bad.stdout + bad.stderr)


def test_omnisafe_recipe_runs_and_the_episodic_dual_moves(tmp_path, request):
    root = request.config.rootpath
    out = str(tmp_path)
    run(["--arm", "td3lag", "--recipe", "omnisafe", "--seed", "0", "--steps", "3000", "--out", out,
         "--dual-start", "1000", "--lambda-lr", "0.01", "--replay", "5000"], root)
    r = rows(f"{out}/td3lag_SafetyPointGoal1-v0_s0.csv")
    assert float(r[-1]["jc"]) >= 0.0 and float(r[-1]["lambda"]) >= 0.0


def test_new_columns_snapshots_schema_guard_and_drop_ckpt(tmp_path, request):
    import os
    import torch
    root = request.config.rootpath
    out = str(tmp_path)
    run(["--arm", "lagu", "--seed", "2", "--steps", "2000", "--out", out, "--drop-ckpt", "0"], root)
    stem = f"{out}/lagu_SafetyPointGoal1-v0_s2"
    r = rows(stem + ".csv")
    for col in ("gap_avg", "qc_avg", "lambda_avg", "lambda_shadow", "would_engage", "cum_cost", "n_pi"):
        assert r[-1][col] not in ("", "nan"), col
    # episodes are 1000 steps, so at step 1000 the cumulative cost equals that episode's cost
    assert abs(float(r[0]["cum_cost"]) - float(r[0]["train_cost"])) < 1e-6
    assert sorted(os.listdir(stem + ".snap")) == ["1000.pt", "2000.pt"]
    ck = torch.load(stem + ".ckpt.pt", weights_only=False)
    assert isinstance(ck["replay"]["o"], torch.Tensor)
    # a CSV with a foreign header cannot be resumed into
    with open(stem + ".csv") as f:
        body = f.read().split("\n", 1)[1]
    with open(stem + ".csv", "w") as f:
        f.write("step,other\n" + body)
    bad = subprocess.run(BASE + ["--arm", "lagu", "--seed", "2", "--steps", "3000", "--out", out, "--resume"],
                         cwd=root, capture_output=True, text=True, timeout=600)
    assert bad.returncode != 0 and "refusing to resume" in (bad.stdout + bad.stderr)
    run(["--arm", "td3lag", "--seed", "3", "--steps", "1000", "--out", out, "--drop-ckpt", "1"], root)
    assert os.path.exists(f"{out}/td3lag_SafetyPointGoal1-v0_s3.pt")
    assert not os.path.exists(f"{out}/td3lag_SafetyPointGoal1-v0_s3.ckpt.pt")


def test_launcher_markers_distinguish_prefix_names():
    from lagu.launch import build_specs, is_live

    class A:
        env, arms, seeds, variant, steps, eval_every, resume, nice = (
            "SafetyPointGoal1-v0", ["lagu"], [1], ["tt:actor_lr=5e-6"], 1000, 500, False, 10)
        out = "/x/results/gate"
    spec = build_specs(A)[0]
    other_out = ["/p/python -m lagu.train --env SafetyPointGoal1-v0 --arm lagu --seed 1 --steps 1000 "
                 "--eval-every 500 --out /x/results/gate_literal --actor-lr 5e-6 --tag tt "]
    other_tag = [other_out[0].replace("gate_literal", "gate").replace("--tag tt ", "--tag tt-match ")]
    other_arm = [other_out[0].replace("gate_literal", "gate").replace("--arm lagu ", "--arm lagu_nogate ")]
    same = [other_out[0].replace("gate_literal", "gate")]
    assert not is_live(spec, other_out) and not is_live(spec, other_tag) and not is_live(spec, other_arm)
    assert is_live(spec, same)


def test_resume_after_dropped_checkpoint_never_retrains(tmp_path, request):
    import hashlib
    root = request.config.rootpath
    out = str(tmp_path)
    run(["--arm", "td3lag", "--seed", "4", "--steps", "1000", "--out", out, "--drop-ckpt", "1"], root)
    csv_path = f"{out}/td3lag_SafetyPointGoal1-v0_s4.csv"
    before = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
    msg = run(["--arm", "td3lag", "--seed", "4", "--steps", "1000", "--out", out, "--drop-ckpt", "1", "--resume"], root)
    assert "already complete" in msg
    assert hashlib.sha256(open(csv_path, "rb").read()).hexdigest() == before
    ext = subprocess.run(BASE + ["--arm", "td3lag", "--seed", "4", "--steps", "2000", "--out", out, "--resume"],
                         cwd=root, capture_output=True, text=True, timeout=600)
    assert ext.returncode != 0 and "checkpoint dropped" in (ext.stdout + ext.stderr)


def test_old_numpy_replay_checkpoints_load_across_sizes():
    import numpy as np
    from lagu.train import Replay
    src = Replay(3, 2, 10)
    for i in range(14):                                   # wraps: newest 10 of 14 kept
        src.add(np.full(3, i), np.zeros(2), i, 0, np.full(3, i + 1), 0)
    state = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in src.state().items()}
    for size in (6, 10, 20):
        dst = Replay(3, 2, size)
        dst.load(state)
        kept = sorted(int(x) for x in dst.r[:dst.n])
        assert kept == list(range(14 - min(10, size), 14))


def test_scheduler_never_exceeds_cap_and_skips_duplicates(tmp_path, monkeypatch):
    import lagu.launch as L
    q = tmp_path / "q"
    q.mkdir()
    monkeypatch.setattr(L, "QDIR", str(q))
    for name in ("QUEUE", "CAP", "PIDFILE", "QLOG", "PAUSED", "POWER"):
        monkeypatch.setattr(L, name, str(q / {"QUEUE": ".queue.jsonl", "CAP": ".queue.cap",
                                               "PIDFILE": ".queue.pid", "QLOG": ".queue.log",
                                               "PAUSED": ".queue.paused", "POWER": ".queue.power"}[name]))
    live = ["/v/bin/python3 -m lagu.train --env E --arm x --seed 9 --out /o "]      # python3, not python
    spawned = []
    monkeypatch.setattr(L, "live_trainers", lambda: list(live))
    monkeypatch.setattr(L, "trainer_pids", lambda: [1])
    monkeypatch.setattr(L, "on_mains", lambda: True)
    monkeypatch.setattr(L, "signal_all", lambda pids, sig: None)
    monkeypatch.setattr(L.shutil, "which", lambda name: None)
    monkeypatch.setattr(L, "spawn", lambda spec: spawned.append(spec["stem"]) or 1)
    monkeypatch.setattr(L.time, "sleep", lambda s: (_ for _ in ()).throw(SystemExit))

    class A:
        env, arms, seeds, variant, steps, eval_every, resume, nice = "E", ["lagu"], [1, 1, 2, 3], [], 10, 5, False, 10
        out = str(tmp_path / "out")
    specs = L.build_specs(A)
    seen, uniq = set(), []
    for s in specs:
        if s["stem"] not in seen:
            seen.add(s["stem"]); uniq.append(s)
    L.write_queue(uniq)
    (q / ".queue.cap").write_text("3")
    try:
        L.run_scheduler(period=0)
    except SystemExit:
        pass
    assert len(spawned) == 2 and len(set(spawned)) == 2      # 1 live + 2 new = cap 3
    assert len(L.read_queue()) == 1
    assert L.TRAINER.match(live[0].strip() + " ")


def test_shadow_lambda_is_exact_for_the_ungated_arm():
    import torch
    from lagu.agent import Config, LagU
    torch.manual_seed(0)
    ag = LagU(8, 2, Config(hidden=16, M=3, use_bonus=True, use_gate=False, lambda_lr=1e-2, lambda_init=0.1,
                           delta0=0.05), seed=0)
    g = torch.Generator().manual_seed(1)
    for i in range(200):
        ag.env_step = 150_000 + i
        ag.update((torch.randn(32, 8, generator=g), torch.rand(32, 2, generator=g) * 2 - 1,
                   torch.randn(32, generator=g), torch.ones(32), torch.randn(32, 8, generator=g), torch.zeros(32)))
        assert ag.lmbda_shadow == ag.lmbda
