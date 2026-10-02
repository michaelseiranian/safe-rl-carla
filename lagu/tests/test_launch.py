"""Queue scheduler: power-aware freezing, hand pause, and recovery after a reboot."""
import json
import os
import signal
import subprocess
import types

import pytest

import lagu.launch as L


@pytest.fixture
def q(tmp_path, monkeypatch):
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(L, "QDIR", str(d))
    for name, f in {"QUEUE": ".queue.jsonl", "CAP": ".queue.cap", "PIDFILE": ".queue.pid", "QLOG": ".queue.log",
                    "PAUSED": ".queue.paused", "POWER": ".queue.power"}.items():
        monkeypatch.setattr(L, name, str(d / f))
    state = {"pids": [101, 102], "signals": [], "spawned": [], "mains": True}
    monkeypatch.setattr(L, "trainer_pids", lambda: list(state["pids"]))
    monkeypatch.setattr(L, "live_trainers", lambda: [])
    monkeypatch.setattr(L, "signal_all", lambda pids, sig: state["signals"].append((tuple(pids), sig)))
    monkeypatch.setattr(L, "spawn", lambda spec: state["spawned"].append(spec["stem"]) or 7)
    monkeypatch.setattr(L, "on_mains", lambda: state["mains"])
    monkeypatch.setattr(L.shutil, "which", lambda name: None)       # no real caffeinate in tests
    return d, state


def spec(out, stem, resume=False):
    return {"stem": stem, "cmd": ["py", "-m", "lagu.train", "--out", out], "out": out, "tag": "",
            "markers": ["--never-matches "], "resume": resume, "nice": 10}


def test_battery_freezes_and_mains_unfreezes(q):
    d, st = q
    L.write_queue([spec(str(d / "o"), "a_s1")])
    st["mains"] = False
    assert L.Scheduler().tick() is False
    assert st["signals"][-1] == ((101, 102), signal.SIGSTOP) and st["spawned"] == []
    assert len(L.read_queue()) == 1                                 # nothing started on battery
    st["mains"] = True
    L.Scheduler().tick()
    assert ((101, 102), signal.SIGCONT) in st["signals"] and st["spawned"] == ["a_s1"]


def test_hand_pause_wins_over_mains_and_always_policy(q):
    d, st = q
    open(L.POWER, "w").write("always")
    open(L.PAUSED, "w").write("")
    L.Scheduler().tick()
    assert st["signals"][-1][1] == signal.SIGSTOP
    import os
    os.remove(L.PAUSED)
    st["mains"] = False                                             # 'always' trains on battery
    L.Scheduler().tick()
    assert st["signals"][-1][1] == signal.SIGCONT


def test_scheduler_stays_while_trainers_live_and_exits_when_all_done(q):
    d, st = q
    L.write_queue([])
    assert L.Scheduler().tick() is False                            # trainers still alive
    st["pids"] = []
    assert L.Scheduler().tick() is True


def test_recover_requeues_interrupted_runs_with_resume_and_new_ckpt_period(q, monkeypatch):
    d, st = q
    out = str(d / "gate")
    import os
    os.makedirs(out)
    base = ["nice", "-n", "10", "caffeinate", "-i", "/v/bin/python", "-m", "lagu.train", "--env", "E",
            "--arm", "lagu", "--steps", "100", "--out", out, "--ckpt-every", "250000", "--tag", "tt"]
    runs = [base + ["--seed", s] for s in ("1", "2", "3")]
    json.dump([{"time": "t", "stem": f"lagu-tt_E_s{s}", "pid": 1, "cmd": c} for s, c in zip("123", runs)],
              open(os.path.join(out, "launch.json"), "w"))
    open(os.path.join(out, "lagu-tt_E_s1.pt"), "w").write("done")     # finished: skipped
    open(os.path.join(out, "lagu-tt_E_s2.ckpt.pt"), "w").write("ck")  # interrupted with a checkpoint
    L.write_queue([spec(out, "td3lag_E_s9")])                        # s3: started, no checkpoint
    monkeypatch.setattr(L, "ensure_scheduler", lambda: None)
    L.recover(100_000)
    qd = L.read_queue()
    stems = [x["stem"] for x in qd]
    assert stems == ["lagu-tt_E_s2", "lagu-tt_E_s3", "td3lag_E_s9"]   # interrupted first, then the queue
    s2, s3, old = qd
    assert s2["resume"] and "--resume" in s2["cmd"] and not s3["resume"]
    for x in qd:
        assert x["cmd"][x["cmd"].index("--ckpt-every") + 1] == "100000"
    assert s2["cmd"][1:3] == ["-m", "lagu.train"] and "caffeinate" not in s2["cmd"]
    L.recover(100_000)                                               # idempotent
    assert [x["stem"] for x in L.read_queue()] == stems


def fake_ps(root):
    r = f"{root}/results"
    return [(11, f"/v/bin/python -m lagu.train --seed 1 --out {r}/gate --tag tt"),
            (12, f"/v/bin/python3.10 -m lagu.train --out {r}/e0_dual --seed 2"),
            (13, "/v/bin/python -m lagu.train --seed 0 --out /tmp/pytest-1/certify0 --arm lagu"),  # test run
            (14, f"/v/bin/python -m lagu.train --out {r}x/gate --seed 3"),                       # sibling dir
            (15, f"/v/bin/python -m lagu.launch --scheduler --out {r}/gate"),
            (16, "/v/bin/python -m lagu.train --out results/pareto --seed 4")]                  # relative


def test_queue_only_touches_trainers_writing_under_results(monkeypatch):
    # regression: the scheduler froze the test suite's own trainers (it matched every lagu.train)
    ps = fake_ps(L.ROOT)
    def run(args, **kw):
        rows = [f"{p} {c}" for p, c in ps] if "pid=,command=" in args else [c for _, c in ps]
        return types.SimpleNamespace(stdout="\n".join(rows) + "\n")
    monkeypatch.setattr(L.subprocess, "run", run)
    assert L.trainer_pids() == [11, 12, 16]
    assert len(L.live_trainers()) == 3


def test_pause_script_only_touches_trainers_writing_under_results(tmp_path):
    root, bin_ = tmp_path / "repo", tmp_path / "bin"
    root.mkdir(), bin_.mkdir()
    rows = "\n".join(f"{p:>6} {c}" for p, c in fake_ps(root))
    (bin_ / "ps").write_text(f"#!/bin/sh\ncat <<'EOF'\n{rows}\nEOF\n")
    (bin_ / "ps").chmod(0o755)
    script = os.path.join(L.ROOT, "lagu", "pause.sh")
    src = open(script).read()
    pids = src[src.index("pids() {"):src.index("\ncase ")]
    out = subprocess.run(["bash", "-c", pids + "\npids"], cwd=root, capture_output=True, text=True,
                         env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"})
    assert out.stdout.split() == ["11", "12", "16"], out.stderr


def test_launch_refuses_out_outside_results(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["lagu.launch", "--out", str(tmp_path / "x"), "--max-procs", "2"])
    with pytest.raises(SystemExit) as e:
        L.main()
    assert e.value.code == 2 and not (tmp_path / "x").exists()
