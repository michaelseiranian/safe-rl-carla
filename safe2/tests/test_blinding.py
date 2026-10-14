"""Blinding of a confirmatory-seed run (SPEC section 10, test 29), through the real CLI in a subprocess
with a temporary key: stdout and stderr hold only the section 3.8 console lines, the public CSV holds
only HEALTH_FIELDS, and the sealed record file decrypts to the outcome rows. An uncaught error prints
only its type and the step."""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys

import pytest

from safe2 import seal, train
from safe2.tests.helpers import make_key

ROOT = __file__.rsplit("/safe2/", 1)[0]
ENV = "SafetyPointGoal1-v0"
STEM = f"R2_{ENV}_s300"
ARGS = ["--config", "R2", "--env", ENV, "--seed", "300", "--steps", "1200", "--hidden", "16", "--eval-every", "600",
        "--eval-episodes", "1", "--snap-every", "600", "--ckpt-every", "600"]
LINE = re.compile(rf"^\[{re.escape(STEM)}\] step=\d+ sps=\d+ finite=[01]$")


@pytest.fixture(scope="module")
def blinded_run(tmp_path_factory):
    pytest.importorskip("safety_gymnasium")
    root = tmp_path_factory.mktemp("blind")
    key = make_key(root)
    out = root / "conf"
    p = subprocess.run([sys.executable, "-m", "safe2.train"] + ARGS + ["--out", str(out)], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    return root, key, out, p


def test_console_holds_only_step_sps_and_finite(blinded_run):
    _, _, _, p = blinded_run
    assert p.returncode == 0, p.stdout
    assert p.stderr == ""                                     # library warnings went to /dev/null
    lines = p.stdout.splitlines()
    assert len(lines) == 2 and all(LINE.match(x) for x in lines), lines


def test_public_files_hold_no_outcome_and_sealed_rows_decrypt(blinded_run):
    root, key, out, _ = blinded_run
    with open(out / f"{STEM}.csv") as f:
        r = list(csv.reader(f))
    assert r[0] == train.HEALTH_FIELDS and len(r) == 3
    meta = json.load(open(out / f"{STEM}.json"))
    assert meta["blind"] is True and not set(meta) & set(train.OUTCOME_FIELDS)
    recs = seal.read_records(str(root / "sealed" / "conf" / f"{STEM}.rows"), key)
    assert [x["step"] for x in recs] == [600, 1200]
    assert set(recs[0]) == {"step"} | set(train.OUTCOME_FIELDS)
    assert isinstance(recs[1]["eval_return"], float) and recs[1]["lambda"] == ""     # R2 has no dual
    assert seal.verify(str(root)) == []


def test_an_uncaught_error_prints_only_its_type_and_step(tmp_path):
    pytest.importorskip("safety_gymnasium")
    make_key(tmp_path)
    code = ("import sys, safe2.train as t\n"
            "def boom(*a, **k):\n"
            "    raise RuntimeError('secret eval_return=12.5')\n"
            "t.evaluate.rollout = boom\n"
            f"sys.exit(t.main({ARGS + ['--out', str(tmp_path / 'conf')]!r}, quiet_fds=True))\n")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert p.returncode == 1 and p.stderr == ""
    assert p.stdout == f"[{STEM}] ERROR RuntimeError step=600\n"
