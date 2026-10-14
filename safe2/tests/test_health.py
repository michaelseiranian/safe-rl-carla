"""The health tool (SPEC section 6.1, test 32): one line per run with only the allowed fields, every
state, and never a path under sealed/ opened, even when the public CSV of a selection seed holds
outcome columns."""
from __future__ import annotations

import builtins
import io
import json
import os
import re
import subprocess

import pytest

import safe2.health as H
import safe2.launch as L
from safe2 import train

ENV = "E-v0"
SECRETS = ("17.125", "31.5", "0.0425", "123456.75")          # outcome values that must never be printed
LINE = re.compile(r"^(p1|conf)/\S+_E-v0_s\d+ +(queued|running|frozen|finished|crashed|diverged|unfinished\(3 resumes\)) +"
                  r"step=(\d+|-) sps=(\d+|-) resumes=\d+ finite=([01]{4}@\d+|1111|-)$")


def stem(config, seed):
    return f"{config}_{ENV}_s{seed}"


def cmd(out, config, seed):
    return ["/v/bin/python", "-m", "safe2.train", "--config", config, "--env", ENV, "--seed", str(seed),
            "--steps", "1000000", "--out", str(out), "--drop-ckpt", "0"]


def write_csv(out, st, flag_rows, outcomes: bool):
    """A public CSV; with outcomes (seeds < 300) every OUTCOME_FIELDS column holds a secret value."""
    fields = train.HEALTH_FIELDS + (train.OUTCOME_FIELDS if outcomes else [])
    lines = [",".join(fields)]
    for i, flags in enumerate(flag_rows, 1):
        row = [str(50_000 * i), str(10 * i), str(9 * i), "123.4", "400.0", *map(str, flags), "0", "0", "0"]
        if outcomes:
            row += [SECRETS[j % len(SECRETS)] for j in range(len(train.OUTCOME_FIELDS))]
        lines.append(",".join(row))
    open(os.path.join(out, st + ".csv"), "w").write("\n".join(lines) + "\n")


def write_json(out, st, n_resumes=0):
    json.dump({"args": {}, "config": {}, "resumes": [{}] * n_resumes, "n_resumes": n_resumes},
              open(os.path.join(out, st + ".json"), "w"))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    m = tmp_path / "results" / "method"
    p1, conf, sealed = m / "p1", m / "conf", m / "sealed" / "conf"
    for x in (p1, conf, sealed):
        x.mkdir(parents=True)
    monkeypatch.setattr(L, "QDIR", str(m))
    monkeypatch.setattr(L, "QUEUE", str(m / ".queue.jsonl"))
    # p1 (seeds < 300: public CSVs hold outcomes)
    write_json(p1, stem("B1", 0)), write_csv(p1, stem("B1", 0), [(1, 1, 1, 1)] * 2, True)
    open(p1 / f"{stem('B1', 0)}.pt", "w").write("final")                           # finished
    write_json(p1, stem("B2", 1)), write_csv(p1, stem("B2", 1), [(1, 1, 1, 1), (1, 0, 1, 1), (1, 0, 0, 1)], True)
    open(p1 / f"{stem('B2', 1)}.pt", "w").write("final")                           # diverged (finished)
    write_json(p1, stem("B3", 2)), write_csv(p1, stem("B3", 2), [(1, 1, 1, 1)], True)    # running
    json.dump({"rank": ["B5", "B4", "B1"]}, open(p1 / "top3.json", "w"))           # not a run
    # conf (seeds >= 300: health columns only; outcomes sealed)
    write_json(conf, stem("B5", 300)), write_csv(conf, stem("B5", 300), [(1, 1, 1, 1)] * 3, False)   # frozen
    write_json(conf, stem("B5", 301), 3), write_csv(conf, stem("B5", 301), [(1, 1, 1, 1)], False)    # unfinished
    write_json(conf, stem("B5", 303), 1)                                                            # crashed
    json.dump([{"time": "t", "stem": stem("B5", s), "pid": 1, "cmd": cmd(conf, "B5", s)} for s in (300, 303, 304)],
              open(conf / "launch.json", "w"))                                                      # 304: launched only
    open(sealed / f"{stem('B5', 300)}.rows", "w").write("300000\tsecret\n")
    L.write_queue([L.spec_from_cmd(cmd(conf, "B5", 302)), L.spec_from_cmd(cmd(conf, "B5", 304))])  # queued
    ps = [(41, "R", " ".join(cmd(p1, "B3", 2))), (42, "T", " ".join(cmd(conf, "B5", 300))),
          (43, "R", " ".join(cmd(tmp_path / "smoke", "B5", 900)))]                 # not managed: ignored
    monkeypatch.setattr(L, "ps_rows", lambda: list(ps))
    def guard(real):                                  # every way to open a file: builtins, io and os
        def guarded(path, *a, **k):
            if not isinstance(path, int):
                assert os.sep + "sealed" + os.sep not in os.path.abspath(os.fsdecode(path)), f"health opened {path}"
            return real(path, *a, **k)
        return guarded
    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(io, "open", guard(io.open))
    monkeypatch.setattr(os, "open", guard(os.open))
    return m


def test_health_prints_only_the_allowed_fields_and_never_opens_sealed(tree, capsys):
    H.main([])                                                       # default: the stage dirs that exist
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if not ln.startswith("#")]
    assert lines and all(LINE.match(ln) for ln in lines), out
    for secret in SECRETS:
        assert secret not in out, secret
    for word in ("return", "cost", "lambda", "eval", "train_"):
        assert all(word not in ln for ln in lines), word
    for col in train.OUTCOME_FIELDS:                                 # LINE already fixes the keys; belt and braces
        assert all(f" {col}=" not in ln for ln in lines), col
    states = {ln.split()[0]: ln.split()[1] for ln in lines}
    assert states == {
        f"p1/{stem('B1', 0)}": "finished", f"p1/{stem('B2', 1)}": "diverged", f"p1/{stem('B3', 2)}": "running",
        f"conf/{stem('B5', 300)}": "frozen", f"conf/{stem('B5', 301)}": "unfinished(3", f"conf/{stem('B5', 302)}": "queued",
        f"conf/{stem('B5', 303)}": "crashed", f"conf/{stem('B5', 304)}": "queued"}
    by = {ln.split()[0]: ln for ln in lines}
    assert by[f"p1/{stem('B1', 0)}"].endswith("step=100000 sps=123 resumes=0 finite=1111")
    assert by[f"p1/{stem('B2', 1)}"].endswith("step=150000 sps=123 resumes=0 finite=1011@100000")
    assert by[f"conf/{stem('B5', 301)}"].endswith("resumes=3 finite=1111")
    assert by[f"conf/{stem('B5', 302)}"].endswith("step=- sps=- resumes=0 finite=-")
    heads = [ln for ln in out.splitlines() if ln.startswith("#")]
    assert heads == [f"# {tree / 'p1'}: 3 runs (1 running, 1 finished, 1 diverged)",
                     f"# {tree / 'conf'}: 5 runs (2 queued, 1 frozen, 1 crashed, 1 unfinished(3 resumes))"]


def test_health_refuses_a_directory_under_sealed(tree):
    with pytest.raises(SystemExit, match="sealed"):
        H.report([str(tree / "sealed" / "conf")])
    rows = tree / "sealed" / "conf" / f"{stem('B5', 300)}.rows"
    for opener in (open, io.open, lambda p: os.open(p, os.O_RDONLY)):    # the guard itself sees every route
        with pytest.raises(AssertionError, match="health opened"):
            opener(rows)


def test_health_sh_runs_the_module(tmp_path):
    """The wrapper reaches safe2.health from any working directory (here on an empty directory)."""
    script = os.path.join(L.ROOT, "safe2", "health.sh")
    (tmp_path / "empty").mkdir()
    p = subprocess.run([script, "empty"], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    assert p.stdout == f"# {tmp_path.resolve() / 'empty'}: 0 runs\n" or p.stdout == f"# {tmp_path / 'empty'}: 0 runs\n"
