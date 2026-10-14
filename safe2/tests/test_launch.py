"""The method queue (SPEC section 6.2-6.3, test 34): power-aware freezing, the shared cap, recovery,
--from-plan minus cuts, and pause.sh. Adapted from lagu/tests/test_launch.py. No real trainer, scheduler
or ps is ever started: every process-facing function is replaced by a fake."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types

import pytest

import safe2.launch as L
from safe2 import train

ENV = "SafetyPointGoal1-v0"          # 3M continuations run on PointGoal1 only (the trainer refuses others)


@pytest.fixture
def q(tmp_path, monkeypatch):
    d = tmp_path / "results" / "method"
    d.mkdir(parents=True)
    monkeypatch.setattr(L, "QDIR", str(d))
    for name, f in {"QUEUE": ".queue.jsonl", "CAP": ".queue.cap", "PIDFILE": ".queue.pid", "QLOG": ".queue.log",
                    "PAUSED": ".queue.paused", "POWER": ".queue.power"}.items():
        monkeypatch.setattr(L, name, str(d / f))
    state = {"pids": [101, 102], "signals": [], "spawned": [], "mains": True, "alive": 2, "live": []}
    monkeypatch.setattr(L, "trainer_pids", lambda: list(state["pids"]))
    monkeypatch.setattr(L, "live_trainers", lambda: list(state["live"]))
    monkeypatch.setattr(L, "n_alive", lambda: state["alive"])
    monkeypatch.setattr(L, "signal_all", lambda pids, sig: state["signals"].append((tuple(pids), sig)))
    monkeypatch.setattr(L, "spawn", lambda spec: state["spawned"].append(spec["stem"]) or 7)
    monkeypatch.setattr(L, "on_mains", lambda: state["mains"])
    monkeypatch.setattr(L, "ensure_scheduler", lambda: None)        # never a real scheduler in tests
    monkeypatch.setattr(L.shutil, "which", lambda name: None)       # no real caffeinate in tests
    return d, state


def cmd(out, config="B5", seed=300, *extra) -> list[str]:
    return ["/v/bin/python", "-m", "safe2.train", "--config", config, "--env", ENV, "--seed", str(seed),
            "--steps", "1000000", "--out", str(out), "--drop-ckpt", "0", *extra]


def spec(out, config="B5", seed=300, *extra) -> dict:
    return L.spec_from_cmd(cmd(out, config, seed, *extra))


def stem(config="B5", seed=300) -> str:
    return f"{config}_{ENV}_s{seed}"


def write_csv(out, st, flag_rows) -> None:
    """A public CSV (blinded schema) with one row per (out, net, loss, dual) flag tuple."""
    os.makedirs(out, exist_ok=True)
    rows = [",".join(train.HEALTH_FIELDS)]
    for i, flags in enumerate(flag_rows, 1):
        rows.append(",".join([str(50_000 * i), "1", "1", "100.0", "1.0", *map(str, flags), "0", "0", "0"]))
    open(os.path.join(out, st + ".csv"), "w").write("\n".join(rows) + "\n")


def write_json(out, st, n_resumes) -> None:
    os.makedirs(out, exist_ok=True)
    json.dump({"args": {}, "resumes": [{}] * n_resumes, "n_resumes": n_resumes}, open(os.path.join(out, st + ".json"), "w"))


def touch(path, text="x") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)


def test_constants_match_the_trainer():
    assert L.FLAGS == tuple(train.GROUPS) and L.CONTINUATION_STEPS == train.CONTINUATION_STEPS
    assert L.MAX_RESUMES == 3


# --- power and pause (as lagu) ---------------------------------------------------------------------
def test_battery_freezes_and_mains_unfreezes(q):
    d, st = q
    L.write_queue([spec(d / "p1", "B1", 0)])
    st["mains"] = False
    assert L.Scheduler().tick() is False
    assert st["signals"][-1] == ((101, 102), signal.SIGSTOP) and st["spawned"] == []
    assert len(L.read_queue()) == 1                                 # nothing started on battery
    st["mains"] = True
    L.Scheduler().tick()
    assert ((101, 102), signal.SIGCONT) in st["signals"] and st["spawned"] == [stem("B1", 0)]


def test_hand_pause_wins_over_mains_and_always_policy(q):
    d, st = q
    open(L.POWER, "w").write("always")
    open(L.PAUSED, "w").write("")
    L.Scheduler().tick()
    assert st["signals"][-1][1] == signal.SIGSTOP
    os.remove(L.PAUSED)
    st["mains"] = False                                             # 'always' trains on battery
    L.Scheduler().tick()
    assert st["signals"][-1][1] == signal.SIGCONT


def test_scheduler_stays_while_trainers_live_and_exits_when_all_done(q):
    d, st = q
    L.write_queue([])
    assert L.Scheduler().tick() is False                            # managed runs still alive
    st["pids"] = []
    assert L.Scheduler().tick() is True


# --- the cap and the refusals ------------------------------------------------------------------------
def test_the_cap_counts_every_study_trainer_lagu_included(q):
    d, st = q
    open(L.CAP, "w").write("3")
    st["alive"] = 2                                                 # e.g. two Lag-U runs in the main checkout
    L.write_queue([spec(d / "p1", "B1", s) for s in (0, 1, 2)])
    L.Scheduler().tick()
    assert st["spawned"] == [stem("B1", 0)]
    assert [x["stem"] for x in L.read_queue()] == [stem("B1", 1), stem("B1", 2)]


def test_the_scheduler_never_starts_a_diverged_or_three_resume_run_or_a_bare_continuation(q, capsys):
    d, st = q
    conf, cont = d / "conf", d / "cont3m"
    write_csv(conf, stem(seed=300), [(1, 1, 1, 1), (1, 0, 1, 1)])     # diverged at its second row
    touch(conf / f"{stem(seed=300)}.ckpt.pt")
    write_json(conf, stem(seed=301), 3)                              # resumed 3 times already
    touch(conf / f"{stem(seed=301)}.ckpt.pt")
    bare = L.spec_from_cmd([x if x != "1000000" else "3000000" for x in cmd(cont, "B5", 302, "--resume")])
    L.write_queue([spec(conf, "B5", 300, "--resume"), spec(conf, "B5", 301, "--resume"), bare,
                   spec(conf, "B5", 303)])
    L.Scheduler().tick()
    assert st["spawned"] == [stem(seed=303)] and L.read_queue() == []
    printed = capsys.readouterr().out
    for s, why in ((300, "diverged"), (301, "unfinished"), (302, "no checkpoint")):
        assert f"skip {stem(seed=s)}: {why}" in printed


def test_blocked_reasons(q):
    d, _ = q
    conf = d / "conf"
    write_csv(conf, stem(seed=300), [(1, 1, 1, 1), (1, 1, 0, 1)])
    write_json(conf, stem(seed=301), 3)
    write_json(conf, stem(seed=302), 2)
    write_csv(conf, stem(seed=302), [(1, 1, 1, 1)])
    assert L.blocked(spec(conf, "B5", 300)) == "diverged"
    assert L.blocked(spec(conf, "B5", 301)) == "unfinished"
    assert L.blocked(spec(conf, "B5", 302)) is None
    c = L.spec_from_cmd([x if x != "1000000" else "3000000" for x in cmd(d / "cont3m", "B5", 300, "--resume")])
    assert L.blocked(c) == "no checkpoint"
    touch(d / "cont3m" / f"{stem(seed=300)}.ckpt.pt")
    assert L.blocked(c) is None


def test_a_torn_last_csv_line_is_ignored(q):
    d, _ = q
    write_csv(d / "conf", stem(), [(1, 1, 1, 1)])
    with open(d / "conf" / f"{stem()}.csv", "a") as f:
        f.write("100000,2,2,99.0,2.0,1,0")                           # a crash mid-write: no zero is read
    assert len(L.health_rows(str(d / "conf"), stem())) == 1 and L.blocked(spec(d / "conf")) is None


# --- recovery ----------------------------------------------------------------------------------------
def test_recover_resumes_from_checkpoints_and_skips_diverged_and_unfinished_runs(q, capsys):
    d, st = q
    out = d / "conf"
    runs = [["nice", "-n", "10", "/old/bin/python3.10"] + cmd(out, "B5", s)[1:] for s in range(300, 306)]
    touch(out / "launch.json", json.dumps([{"time": "t", "stem": stem(seed=s), "pid": 1, "cmd": c}
                                           for s, c in zip(range(300, 306), runs)]))
    touch(out / f"{stem(seed=300)}.pt")                              # finished: skipped
    touch(out / f"{stem(seed=301)}.ckpt.pt")                         # interrupted with a checkpoint: resume
    write_json(out, stem(seed=302), 0)                               # crashed before its first checkpoint: restart
    touch(out / f"{stem(seed=303)}.ckpt.pt")
    write_csv(out, stem(seed=303), [(0, 1, 1, 1)])                   # diverged: never resumed
    touch(out / f"{stem(seed=304)}.ckpt.pt")
    write_json(out, stem(seed=304), 3)                               # 3 resumes: unfinished
    touch(out / f"{stem(seed=305)}.ckpt.pt")
    st["live"] = [" ".join(runs[5]) + " "]                           # still alive: left alone
    L.write_queue([spec(d / "p1", "B1", 0)])
    L.recover()
    qd = L.read_queue()
    assert [x["stem"] for x in qd] == [stem(seed=301), stem(seed=302), stem("B1", 0)]
    s301, s302, _ = qd
    assert s301["resume"] and s301["cmd"][-1] == "--resume" and not s302["resume"] and "--resume" not in s302["cmd"]
    for x in qd[:2]:
        assert x["cmd"][:3] == [sys.executable, "-m", "safe2.train"] and "nice" not in x["cmd"]
        assert "--ckpt-every" not in x["cmd"]                       # the trainer refuses dev flags here
        assert train.refusal(train.parse(x["cmd"][3:])) is None
    printed = capsys.readouterr().out
    assert f"diverged   conf/{stem(seed=303)}" in printed and f"unfinished conf/{stem(seed=304)}" in printed
    L.recover()                                                      # idempotent
    assert [x["stem"] for x in L.read_queue()] == [x["stem"] for x in qd]


# --- which processes the queue touches ---------------------------------------------------------------
def fake_ps(root, main="/main/checkout"):
    m, r = f"{root}/results/method", f"{root}/results"
    return [(11, "R", f"/v/bin/python -m safe2.train --config B5 --seed 300 --out {m}/conf"),
            (12, "T", f"/v/bin/python3.10 -m safe2.train --out results/method/p1 --seed 0"),    # relative
            (13, "R", "/v/bin/python -m safe2.train --seed 900 --out /tmp/pytest-1/results/method/x"),  # test run
            (14, "R", f"/v/bin/python -m safe2.train --out {r}/methodx/p1 --seed 1"),          # sibling dir
            (15, "R", f"/v/bin/python -m safe2.train --out {r}/gate --seed 2"),                # not method/
            (16, "R", f"/v/bin/python -m safe2.train --out {m}/../gate --seed 3"),             # escapes method/
            (17, "R", f"/v/bin/python -m lagu.train --out {m}/conf --seed 4"),                 # Lag-U's trainer
            (18, "R", f"/v/bin/python -m lagu.train --seed 1 --out {main}/results/gate --tag tt"),  # main checkout
            (19, "R", f"/v/bin/python -m safe2.launch --scheduler --out {m}/p1"),
            (20, "R", "/v/bin/python -m lagu.train --seed 0 --out /tmp/pytest-2/certify0"),
            (21, "R", f"/v/bin/python -m lagu.evaluate --out {main}/results/gate")]


def test_managed_accepts_only_safe2_train_under_results_method_and_the_cap_counts_both_studies(monkeypatch):
    ps = fake_ps(L.ROOT)
    def run(args, **kw):
        assert args == ["ps", "-axo", "pid=,stat=,command="]
        return types.SimpleNamespace(stdout="\n".join(f"{p:>6} {s:<4} {c}" for p, s, c in ps) + "\n")
    monkeypatch.setattr(L.subprocess, "run", run)
    monkeypatch.setattr(L, "CAP_ROOTS", [os.path.join(L.ROOT, "results"), "/main/checkout/results"])
    assert L.trainer_pids() == [11, 12]
    assert len(L.live_trainers()) == 2
    assert L.n_alive() == 7                    # 11, 12, 14, 15, 16 (safe2 under results/), 17, 18 (Lag-U)
    assert [p for p, _, c in ps if L.counted(c)] == [11, 12, 14, 15, 16, 17, 18]


def test_checkouts_are_the_main_checkout_and_its_worktrees(tmp_path, monkeypatch):
    main, wt = tmp_path / "main", tmp_path / "wt"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    wt.mkdir()
    (main / ".git" / "worktrees" / "wt" / "gitdir").write_text(f"{wt}/.git\n")
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt\n")
    for root in (wt, main):
        monkeypatch.setattr(L, "ROOT", str(root))
        assert L.checkouts() == sorted([str(main), str(wt)])
    assert os.path.join(L.__file__.rsplit("/safe2/", 1)[0], "results") in L.CAP_ROOTS   # the real repository


def test_pause_script_only_freezes_safe2_study_runs(tmp_path):
    root, bin_ = tmp_path / "repo", tmp_path / "bin"
    root.mkdir(), bin_.mkdir()
    rows = "\n".join(f"{p:>6} {c}" for p, _, c in fake_ps(root))
    (bin_ / "ps").write_text(f"#!/bin/sh\ncat <<'EOF'\n{rows}\nEOF\n")
    (bin_ / "ps").chmod(0o755)
    script = os.path.join(L.ROOT, "safe2", "pause.sh")
    src = open(script).read()
    pids = src[src.index("pids() {"):src.index("\ncase ")]
    out = subprocess.run(["bash", "-c", pids + "\npids"], cwd=root, capture_output=True, text=True,
                         env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"})
    assert out.stdout.split() == ["11", "12"], out.stderr
    assert "-m safe2.launch --recover" in src and "lagu.launch" not in src
    assert "Q=results/method" in src and "results/.queue" not in src


# --- the CLI -----------------------------------------------------------------------------------------
def test_launch_refuses_out_outside_results_method(tmp_path, monkeypatch, q):
    d, _ = q
    for out in (tmp_path / "x", d.parent / "gate"):
        with pytest.raises(SystemExit) as e:
            L.main(["--out", str(out), "--configs", "B1", "--max-procs", "2"])
        assert e.value.code == 2 and not out.exists()
    with pytest.raises(SystemExit) as e:
        L.main(["--out", str(d / "p1"), "--configs", "B10", "--max-procs", "2"])  # not a registered config
    assert e.value.code == 2


def test_cli_commands_pass_the_trainers_study_guard(q, capsys):
    d, _ = q
    L.main(["--out", str(d / "p1"), "--configs", "B1", "SHL-g0.90-B4", "--seeds", "0", "1", "--drop-ckpt", "1",
            "--dry-run"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    for ln in lines:
        argv = ln.split()
        assert argv[:3] == [sys.executable, "-m", "safe2.train"]
        a = train.parse(argv[3:])
        assert train.refusal(a) is None and a.drop_ckpt == 1 and a.steps == 1_000_000
    assert not (d / "p1").exists() and L.read_queue() == []           # a dry run touches nothing
    L.main(["--out", str(d / "p1"), "--configs", "B5", "BVSU", "BVSD", "--from-bvs", "BVS-SQ-k0.1",
            "--rho-star", "0.0312", "--seeds", "0", "--dry-run"])
    b5, bvsu, bvsd = (ln.split()[3:] for ln in capsys.readouterr().out.splitlines())
    assert "--from-bvs" not in b5 and "--rho-star" not in b5 and "--rho-star" not in bvsd
    assert bvsu[bvsu.index("--rho-star") + 1] == "0.0312" and bvsd[bvsd.index("--from-bvs") + 1] == "BVS-SQ-k0.1"
    for argv in (b5, bvsu, bvsd):
        a = train.parse(argv)
        assert train.refusal(a) is None and train.build_config(a).name == argv[1]
    for bad in (["--configs", "BVSD"], ["--configs", "BVSU", "--from-bvs", "BVS-SQ-k0.1"]):
        with pytest.raises(SystemExit) as e:
            L.main(["--out", str(d / "p1"), "--seeds", "0", "--dry-run"] + bad)
        assert e.value.code == 2


def test_cli_queue_keeps_a_hand_set_cap_and_skips_started_runs(q, capsys):
    d, _ = q
    write_json(d / "p1", stem("B2", 1), 0)                            # started earlier: --recover's job
    L.main(["--out", str(d / "p1"), "--configs", "B1", "B2", "--env", ENV, "--seeds", "0", "1", "--max-procs", "4"])
    assert [x["stem"] for x in L.read_queue()] == [stem("B1", 0), stem("B1", 1), stem("B2", 0)]
    assert open(L.CAP).read() == "4" and "already started" in capsys.readouterr().out
    L.main(["--out", str(d / "p1"), "--configs", "B3", "--env", ENV, "--seeds", "0", "--max-procs", "9", "--front"])
    assert open(L.CAP).read() == "4"                                  # never silently overridden
    assert L.read_queue()[0]["stem"] == stem("B3", 0)
    L.main(["--out", str(d / "p1"), "--configs", "B3", "--env", ENV, "--seeds", "0", "--max-procs", "9", "--set-cap"])
    assert open(L.CAP).read() == "9" and len(L.read_queue()) == 4     # de-duplicated


# --- --from-plan -------------------------------------------------------------------------------------
def plan_run(stage, out, config, seed, steps=1_000_000, **kw):
    """A plan.json run as safe2.analyze.make_plan writes it (argv after the interpreter)."""
    argv = ["-m", "safe2.train", "--config", config, "--env", ENV, "--seed", str(seed), "--out", str(out),
            "--drop-ckpt", "1"] + (["--steps", "3000000", "--resume"] if steps == 3_000_000 else [])
    return {"id": f"{stage}/{stem(config, seed)}", "stage": stage, "arm": config, "config": config, "env": ENV,
            "seed": seed, "steps": steps, "out": str(out), "drop_ckpt": 1, "cut_group": None, "argv": argv, **kw}


def write_plan(d, runs, cuts):
    json.dump({"runs": runs, "evaluations": []}, open(d / "plan.json", "w"))
    if cuts is not None:
        json.dump(cuts, open(d / "cuts.json", "w"))
    return str(d / "plan.json")


def test_from_plan_queues_the_stage_with_its_exact_argv_minus_the_cuts(q, capsys):
    d, _ = q
    runs = [plan_run("conf", d / "conf", "B5", 300), plan_run("conf", d / "conf", "B5", 301),
            plan_run("conf", d / "conf", "R1", 300), plan_run("conf", d / "conf", "R1", 306),
            plan_run("p1", d / "p1", "B1", 0),
            plan_run("cont3m", d / "cont3m", "B5", 300, steps=3_000_000)]
    cuts = [{"date": "2026-10-20", "item": "1", "what": "3M continuations", "stage": "cont3m",
             "stems": [stem("B5", 300)], "reason": "throughput"},
            {"date": "2026-10-21", "item": "3", "what": "R1, Saute*, L* to 6 seeds", "stage": "conf",
             "stems": [stem("R1", 306)], "reason": "throughput"}]
    plan = write_plan(d, runs, cuts)
    L.write_queue([spec(d / "p2", "B5", 0)])
    L.main(["--from-plan", plan, "--stage", "conf", "--front"])
    qd = L.read_queue()
    assert [x["stem"] for x in qd] == [stem("B5", 300), stem("B5", 301), stem("R1", 300), stem("B5", 0)]
    for x, r in zip(qd[:3], runs[:3]):
        assert x["cmd"] == [sys.executable] + r["argv"] and not x["resume"]
        assert train.refusal(train.parse(x["cmd"][3:])) is None
    assert "(1 cut" in capsys.readouterr().out                      # the cont3m cut leaves conf/B5_s300 alone
    L.main(["--from-plan", plan, "--stage", "cont3m"])                # its only run is cut
    assert len(L.read_queue()) == 4


def test_from_plan_refuses_a_missing_plan_or_cuts_file_a_stageless_cut_and_an_unknown_stage(q):
    d, _ = q
    plan = write_plan(d, [plan_run("conf", d / "conf", "B5", 300)], None)
    with pytest.raises(SystemExit, match="cuts.json is missing"):
        L.main(["--from-plan", plan, "--stage", "conf"])
    write_plan(d, [plan_run("conf", d / "conf", "B5", 300)], [{"item": "1", "stems": [stem("B5", 300)]}])
    with pytest.raises(SystemExit, match="must name its stage"):
        L.main(["--from-plan", plan, "--stage", "conf"])
    write_plan(d, [plan_run("conf", d / "conf", "B5", 300)], [])
    with pytest.raises(SystemExit, match="no run in stage"):
        L.main(["--from-plan", plan, "--stage", "p9"])
    with pytest.raises(SystemExit, match="does not exist"):
        L.main(["--from-plan", str(d / "nope.json"), "--stage", "conf"])
    assert L.read_queue() == []


def test_from_plan_skips_started_runs_and_reads_a_full_trainer_command(q, capsys):
    d, _ = q
    r = plan_run("conf", d / "conf", "B5", 300)
    r2 = plan_run("conf", d / "conf", "B5", 301)
    r2["argv"] = r2["argv"][2:]                                     # the trainer arguments alone
    write_json(d / "conf", stem(seed=300), 0)
    plan = write_plan(d, [r, r2], [])
    L.main(["--from-plan", plan, "--stage", "conf"])
    assert [x["stem"] for x in L.read_queue()] == [stem(seed=301)]
    assert L.read_queue()[0]["cmd"] == [sys.executable, "-m", "safe2.train"] + r2["argv"]
    assert "already started" in capsys.readouterr().out


def test_a_relative_plan_out_is_taken_from_the_repository_root():
    c = L.trainer_cmd(["-m", "safe2.train", "--config", "B5", "--out", "results/method/conf", "--seed", "300"])
    assert c[c.index("--out") + 1] == os.path.join(L.ROOT, "results", "method", "conf")


def test_a_continuation_gets_its_1m_checkpoint_only_after_the_heldout_evaluation(q, capsys):
    d, _ = q
    src = d / "conf" / f"{stem(seed=300)}.ckpt.pt"
    touch(src, "the 1M checkpoint")
    touch(d / "conf" / f"{stem(seed=300)}.pt")                       # the 1M run finished
    r = plan_run("cont3m", d / "cont3m", "B5", 300, steps=3_000_000, resume_from=str(src))
    plan = write_plan(d, [r], [])
    L.main(["--from-plan", plan, "--stage", "cont3m"])
    dst = d / "cont3m" / f"{stem(seed=300)}.ckpt.pt"
    assert L.read_queue() == [] and not dst.exists()
    assert "held-out evaluation is not written yet" in capsys.readouterr().out
    touch(d / "sealed" / "heldout" / f"{stem(seed=300)}.deployed.json.enc", "sealed")
    L.main(["--from-plan", plan, "--stage", "cont3m"])
    assert dst.read_text() == "the 1M checkpoint"
    (x,) = L.read_queue()
    assert x["resume"] and x["cmd"][-1] == "--resume" and x["cmd"][x["cmd"].index("--steps") + 1] == "3000000"
    assert L.blocked(x) is None and train.refusal(train.parse(x["cmd"][3:])) is None


def test_from_plan_reads_the_plan_and_cuts_that_safe2_analyze_writes(q):
    """The contract with safe2.analyze: every planned run becomes a command the trainer's study guard
    accepts, continuations carry their checkpoint source, and `analyze cut` ids remove runs."""
    from safe2 import analyze
    d, _ = q
    v1 = {"b_star": "B5", "bvs_star": "BVS-SQ-k0.1", "shl_star": "SHL-g0.90-B5", "saute_star": "SauteA",
          "l_star": "B7", "run_saute_star": True, "run_l_star": True, "rho_star": 0.0312, "m_phs": {}, "r_ref": {},
          "family": 1, "bvs_killed": False, "shl_killed": False}
    json.dump(v1, open(d / "v1.json", "w"))
    plan = analyze.make_plan(str(d))
    json.dump(plan, open(d / "plan.json", "w"))
    json.dump([], open(d / "cuts.json", "w"))
    for stage in ("conf", "cont3m"):
        planned, n_cut = L.load_plan(str(d / "plan.json"), stage)
        assert n_cut == 0 and len(planned) == sum(r["stage"] == stage for r in plan["runs"]) > 0
        for r, s in planned:
            assert s["out"] == str(d / stage) and s["stem"] == r["id"].split("/", 1)[1]
            a = train.parse(s["cmd"][3:])
            assert train.refusal(a) is None and train.build_config(a).name == r["config"]
            assert (stage == "cont3m") == s["resume"] == bool(r.get("resume_from"))
            if r.get("resume_from"):                                  # the 1M checkpoint of the same run
                assert L._resolve(r["resume_from"]) == str(d / "conf" / (s["stem"] + ".ckpt.pt"))
    analyze.cut(str(d), "1")                                          # cut item 1: every 3M continuation
    assert L.load_plan(str(d / "plan.json"), "cont3m")[0] == []
    assert L.load_plan(str(d / "plan.json"), "conf")[1] == 0


# --- review fixes (review-ops.md H1, M1-M4, L2-L4, test gaps) -----------------------------------------------
def test_a_finished_run_is_final_and_3m_runs_only_go_to_cont3m(q, capsys):
    """H1 and M1: a --resume (or a 3M extension) of a finished run is never queued or started, and the
    trainer's refusals (3M outside cont3m/, 1M in cont3m/) are applied before queueing and starting."""
    d, st = q
    conf = d / "conf"
    for seed in (301, 302):
        touch(conf / f"{stem(seed=seed)}.ckpt.pt")              # B* keeps its checkpoint
    touch(conf / f"{stem(seed=301)}.pt")                         # 301 finished; 302 interrupted
    L.main(["--out", str(conf), "--configs", "B5", "--seeds", "301", "302", "--steps", "3000000", "--resume",
            "--max-procs", "12"])
    out = capsys.readouterr().out
    assert f"skip {stem(seed=301)}: already finished" in out
    assert f"skip {stem(seed=302)}: refused: --steps must be 1000000" in out and L.read_queue() == []
    L.main(["--out", str(conf), "--configs", "B5", "--seeds", "301", "--resume", "--max-procs", "12"])
    assert "already finished" in capsys.readouterr().out and L.read_queue() == []
    L.write_queue([spec(conf, "B5", 301, "--resume"), spec(d / "cont3m", "B5", 300, "--resume")])
    L.Scheduler().tick()                                         # queued earlier: dropped at start time too
    assert st["spawned"] == [] and L.read_queue() == []
    assert L.blocked(spec(d / "cont3m", "B5", 300)).startswith("refused: cont3m/ holds only 3M continuations")


def test_cut_runs_are_never_started_unqueued_by_from_plan_and_not_recovered(q, capsys):
    """M2: a cut takes effect on queued runs (scheduler, --from-plan) and on --recover."""
    d, st = q
    conf = d / "conf"
    runs = [plan_run("conf", conf, "SHL-g0.90-B5", s) for s in (307, 308)]
    plan = write_plan(d, runs, [])
    L.main(["--from-plan", plan, "--stage", "conf"])
    assert len(L.read_queue()) == 2
    cuts = [{"date": "d", "item": "2", "what": "SHL* to 8 seeds", "stage": "conf",
             "stems": [stem("SHL-g0.90-B5", 308)], "reason": "throughput"}]
    json.dump(cuts, open(d / "cuts.json", "w"))
    st["live"] = []
    L.main(["--from-plan", plan, "--stage", "conf"])
    assert f"unqueued conf/{stem('SHL-g0.90-B5', 308)}: cut" in capsys.readouterr().out
    assert [x["stem"] for x in L.read_queue()] == [stem("SHL-g0.90-B5", 307)]
    L.write_queue([spec(conf, "SHL-g0.90-B5", 308)])               # e.g. queued by hand before the cut
    L.Scheduler().tick()
    assert st["spawned"] == [] and L.read_queue() == []
    touch(conf / "launch.json", json.dumps([{"cmd": cmd(conf, "SHL-g0.90-B5", 308)}]))
    write_json(conf, stem("SHL-g0.90-B5", 308), 0)                # started, then killed
    L.recover()
    assert L.read_queue() == [] and f"cut        conf/{stem('SHL-g0.90-B5', 308)}" in capsys.readouterr().out


def test_continuations_skip_a_diverged_unfinished_or_missing_source(q, capsys):
    """M3: a continuation resumes its 1M run, so it is never started from a diverged or unfinished run;
    a missing checkpoint is a note, not an abort of the whole --from-plan."""
    d, _ = q
    conf, cont = d / "conf", d / "cont3m"
    runs = []
    for seed in (300, 301, 302):
        touch(conf / f"{stem(seed=seed)}.ckpt.pt")
        touch(d / "sealed" / "heldout" / f"{stem(seed=seed)}.deployed.json.enc")
        runs.append(plan_run("cont3m", cont, "B5", seed, steps=3_000_000,
                             resume_from=str(conf / f"{stem(seed=seed)}.ckpt.pt")))
    write_csv(conf, stem(seed=300), [(1, 1, 1, 1), (1, 1, 0, 1)])    # diverged (and finished)
    touch(conf / f"{stem(seed=300)}.pt")
    write_json(conf, stem(seed=301), 3)                              # unfinished after 3 resumes
    touch(conf / f"{stem(seed=302)}.pt")
    os.remove(conf / f"{stem(seed=302)}.ckpt.pt")                     # finished, checkpoint gone
    L.main(["--from-plan", write_plan(d, runs, []), "--stage", "cont3m"])
    out = capsys.readouterr().out
    for seed, why in ((300, "diverged"), (301, "unfinished"), (302, "checkpoint missing")):
        assert f"skip cont3m/{stem(seed=seed)}: source {why}" in out
    assert L.read_queue() == [] and not cont.exists()


def test_the_cap_reserves_the_slots_the_lagu_scheduler_will_refill(tmp_path, monkeypatch):
    """M4: Lag-U's live runs, or min(its cap, alive + queued) while its queue holds work; frozen Lag-U
    runs keep their slots unless its cap is 0. The method study's own runs count frozen or not."""
    main, wt = tmp_path / "main" / "results", tmp_path / "wt" / "results"
    main.mkdir(parents=True), wt.mkdir(parents=True)
    monkeypatch.setattr(L, "CAP_ROOTS", [str(main), str(wt)])
    ps = [(1, "R", f"/v/bin/python -m safe2.train --seed 300 --out {wt}/method/conf"),
          (2, "T", f"/v/bin/python -m safe2.train --seed 301 --out {wt}/method/conf"),     # frozen: counts
          (3, "R", f"/v/bin/python -m lagu.train --seed 0 --out {main}/gate"),
          (4, "T", f"/v/bin/python -m lagu.train --seed 1 --out {main}/gate")]             # frozen Lag-U run
    monkeypatch.setattr(L, "ps_rows", lambda: list(ps))
    assert L.n_alive() == 2 + 2                                        # no Lag-U queue: its 2 runs
    (main / ".queue.jsonl").write_text('{"stem": "a"}\n' * 117)
    (main / ".queue.cap").write_text("6")
    assert L.n_alive() == 2 + 6                                        # it will refill to its cap of 6
    (main / ".queue.cap").write_text("0")                              # lagu/pause.sh cap 0
    assert L.n_alive() == 2 + 1                                        # only its running run
    (main / ".queue.jsonl").write_text("")
    (main / ".queue.cap").write_text("12")
    assert L.n_alive() == 2 + 2


def test_refused_commands_are_never_queued_and_recover_holds_them(q, capsys):
    """L2: a command the trainer refuses (a seed outside the study) is reported, not queued, and
    --recover does not re-queue it."""
    d, _ = q
    L.main(["--out", str(d / "conf"), "--configs", "B5", "--seeds", "316", "--max-procs", "12"])
    assert "refused: seed 316 is not a study training seed" in capsys.readouterr().out and L.read_queue() == []
    touch(d / "conf" / "launch.json", json.dumps([{"cmd": cmd(d / "conf", "B5", 316)}]))
    L.recover()
    assert L.read_queue() == [] and "refused: seed 316" in capsys.readouterr().out


def test_the_scheduler_never_starts_a_queued_fresh_run_that_has_started_since(q):
    """L3: a fresh spec queued before its run started is dropped at start time; --recover's restart
    (no checkpoint) carries --restart, which the trainer needs to start a started run again."""
    d, st = q
    conf = d / "conf"
    L.write_queue([spec(conf, "B5", 302)])
    write_json(conf, stem(seed=302), 2)
    touch(conf / f"{stem(seed=302)}.ckpt.pt")
    L.Scheduler().tick()
    assert st["spawned"] == [] and L.read_queue() == []
    os.remove(conf / f"{stem(seed=302)}.ckpt.pt")                     # crashed before a checkpoint instead
    touch(conf / "launch.json", json.dumps([{"cmd": cmd(conf, "B5", 302)}]))
    L.recover()
    (x,) = L.read_queue()
    assert x["restart"] and x["cmd"][-1] == "--restart" and train.refusal(train.parse(x["cmd"][3:])) is None
    L.Scheduler().tick()
    assert st["spawned"] == [stem(seed=302)]


def test_the_cli_always_queues(q, capsys):
    """L4: without --max-procs the runs still go through the queue and its cap; no direct spawn."""
    d, st = q
    L.main(["--out", str(d / "p1"), "--configs", "B1", "--seeds", "0", "1"])
    assert st["spawned"] == [] and [x["stem"] for x in L.read_queue()] == [stem("B1", 0), stem("B1", 1)]
    assert not os.path.exists(L.CAP) and "effective cap 12 (default)" in capsys.readouterr().out


def test_spawn_starts_a_niced_single_thread_trainer_in_its_own_session(tmp_path, monkeypatch):
    """Test gap: spawn() itself (always faked elsewhere): nice, OMP/MKL threads, cwd, log, launch.json."""
    seen = {}

    class Popen:
        def __init__(self, args, **kw):
            seen.update(args=args, **kw)
            self.pid = 4242
    monkeypatch.setattr(L.subprocess, "Popen", Popen)
    out = tmp_path / "results" / "method" / "p1"
    s = L.spec_from_cmd(cmd(out, "B1", 0))
    assert L.spawn(s) == 4242 and L.spawn(s) == 4242
    assert seen["args"][:3] == ["nice", "-n", "10"] and seen["args"][3:] == s["cmd"]
    assert seen["env"]["OMP_NUM_THREADS"] == "1" == seen["env"]["MKL_NUM_THREADS"]
    assert seen["cwd"] == L.ROOT and seen["start_new_session"] and seen["stdin"] == subprocess.DEVNULL
    assert seen["stdout"].name == str(out / f"{stem('B1', 0)}.log")
    entries = json.load(open(out / "launch.json"))
    assert [e["pid"] for e in entries] == [4242, 4242] and entries[0]["cmd"] == s["cmd"]


def test_pause_script_pause_resume_cap_power_and_status(tmp_path):
    """Test gap: pause.sh itself in a sandbox copy with ps, kill, pmset and the interpreter shimmed."""
    repo, bin_ = tmp_path / "repo", tmp_path / "bin"
    (repo / "safe2").mkdir(parents=True), (repo / ".venv" / "bin").mkdir(parents=True), bin_.mkdir()
    script = repo / "safe2" / "pause.sh"
    script.write_text(open(os.path.join(L.ROOT, "safe2", "pause.sh")).read())
    script.chmod(0o755)
    calls = tmp_path / "calls"
    rows = "\n".join(f"{p:>6} {c}" for p, _, c in fake_ps(repo))
    shims = {"ps": f"#!/bin/sh\ncat <<'EOF'\n{rows}\nEOF\n", "kill": f'#!/bin/sh\necho "kill $*" >> {calls}\n',
             "pmset": "#!/bin/sh\necho \"Now drawing from 'AC Power'\"\n"}
    for name, text in shims.items():
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    py = repo / ".venv" / "bin" / "python"
    py.write_text(f'#!/bin/sh\necho "python $*" >> {calls}\n')
    py.chmod(0o755)
    (tmp_path / "bash_env").write_text("enable -n kill\n")          # `kill` is a builtin: use the shim instead
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "BASH_ENV": str(tmp_path / "bash_env")}
    run = lambda *a: subprocess.run(["bash", str(script), *a], capture_output=True, text=True, env=env, cwd=tmp_path)
    q_dir = repo / "results" / "method"
    assert "2 runs frozen" in run("pause").stdout and (q_dir / ".queue.paused").exists()
    assert run("cap", "0").stdout == "cap: 0\n" and (q_dir / ".queue.cap").read_text() == "0\n"
    assert run("power", "always").returncode == 0 and (q_dir / ".queue.power").read_text() == "always\n"
    assert run("cap", "x").returncode == 2 and run("power", "x").returncode == 2
    assert run("resume").returncode == 0 and not (q_dir / ".queue.paused").exists()
    assert (q_dir / ".queue.cap").read_text() == "12\n"                 # resume lifts a cap of 0
    assert calls.read_text().splitlines() == ["kill -STOP 11 12", "kill -CONT 11 12", "python -m safe2.launch --recover"]
    status = run("status").stdout
    assert "policy: always" in status and "cap: 12" in status and "paused by hand: no" in status


def test_retire_moves_a_run_and_its_sealed_files_out_of_the_study(q, capsys):
    """L6: --retire STAGE/STEM (bug reruns) moves the run files, its sealed records and its held-out
    outputs (1M ones for conf/, 3M ones for cont3m/) under results/method/retired/, outside sealed/, and
    removes it from launch.json and the queue; a live run is refused."""
    d, st = q
    conf, s300 = d / "conf", stem(seed=300)
    for f in (f"conf/{s300}.csv", f"conf/{s300}.json", f"conf/{s300}.pt", f"conf/{s300}.log",
              f"conf/{s300}.snap/1000000.pt", f"conf/{stem(seed=3001)}.csv", f"sealed/conf/{s300}.rows",
              f"sealed/heldout/{s300}.deployed.json.enc", f"sealed/heldout/{s300}.deployed.sha",
              f"sealed/heldout/{s300}.deployed.n3000000.json.enc", f"sealed/heldout/{s300}.phs.json.enc",
              f"phs/{s300}.posthoc.pt", f"sealed/phs/{s300}.quality.json.enc", f"cont3m/{s300}.csv"):
        touch(d / f)
    touch(conf / "launch.json", json.dumps([{"cmd": cmd(conf, "B5", 300)}, {"cmd": cmd(conf, "B5", 301)}]))
    L.write_queue([spec(conf, "B5", 300, "--resume"), spec(conf, "B5", 301)])
    st["live"] = [" ".join(cmd(conf, "B5", 302)) + " "]
    L.main(["--retire", f"conf/{s300}", f"conf/{stem(seed=302)}", "conf/x/y"])
    out = capsys.readouterr().out
    assert f"retired conf/{s300}: 11 files" in out and f"refused conf/{stem(seed=302)}: it is running" in out
    assert "refused conf/x/y" in out
    (ret,) = os.listdir(d / "retired")
    moved = {os.path.relpath(os.path.join(r, f), d / "retired" / ret) for r, _, fs in os.walk(d / "retired" / ret)
             for f in fs}
    assert f"sealed/heldout/{s300}.deployed.json.enc" in moved and f"conf/{s300}.snap/1000000.pt" in moved
    assert os.path.exists(d / "sealed" / "heldout" / f"{s300}.deployed.n3000000.json.enc")   # cont3m's: kept
    assert os.path.exists(d / "cont3m" / f"{s300}.csv") and os.path.exists(conf / f"{stem(seed=3001)}.csv")
    assert [e["cmd"][e["cmd"].index("--seed") + 1] for e in json.load(open(conf / "launch.json"))] == ["301"]
    assert [x["stem"] for x in L.read_queue()] == [stem(seed=301)]
    from safe2 import seal
    assert seal.verify(str(d)) == ["sealed/heldout/" + f"{s300}.deployed.n3000000.json.enc: no hash-log entry"]
