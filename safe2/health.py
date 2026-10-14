"""Outcome-free run health of the method study (prereg_method.md section 10; safe2/SPEC.md section 6.1).

    python -m safe2.health [DIR ...]       # default: results/method/{p1,p2,conf,cont3m}
    safe2/health.sh [DIR ...]              # the same

One line per run, and one summary line (#) per directory:

    <dir>/<stem>  <state>  step=<last step>  sps=<steps/s>  resumes=<n_resumes>  finite=<flags>

- state: queued | running | frozen | finished | crashed | diverged | unfinished(3 resumes).
  running and frozen come from ps (a stopped process is frozen); diverged means some row has a finite
  flag of 0 (whatever else holds); crashed means started but not alive, not finished and not queued.
- step and sps are those of the last public row (one per in-training evaluation, every 50k).
- finite: 1111 when every row has its four flags (outcomes, network, losses, dual) at 1, otherwise
  the four flags of the first row with a zero and its step (e.g. 1011@350000); - before the first row.

Liveness is the step advancing between calls, the state, and safe2/pause.sh status; crashes are in
results/method/.queue.log.

It reads only the public CSVs (of which it keeps step, sps and the finite flags), the run jsons
(n_resumes), launch.json, the queue file and ps. It never opens a path under sealed/, refuses a
directory there, and never prints a return, a cost or any other outcome, whatever the seed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

from safe2 import launch as L

STAGES = ("p1", "p2", "conf", "cont3m")
STATES = ("queued", "running", "frozen", "finished", "crashed", "diverged", "unfinished(3 resumes)")
RUN_STEM = re.compile(r"^.+_[^_]+-v\d+_s\d+$")      # <config>_<env>_s<seed>; not top3.json or launch.json


def stems_in(d: str, queue: list[dict]) -> set[str]:
    """Every run of directory d: queued, launched (launch.json) or started (its run json exists)."""
    stems = {q["stem"] for q in queue if q["out"] == d}
    try:
        with open(os.path.join(d, "launch.json")) as f:
            entries = json.load(f)
    except (OSError, ValueError):
        entries = []
    for e in entries:
        spec = L.spec_from_cmd(e.get("cmd", []))
        if spec and spec["out"] == d:
            stems.add(spec["stem"])
    for path in glob.glob(os.path.join(d, "*.json")):
        stem = os.path.basename(path)[:-len(".json")]
        if RUN_STEM.match(stem):
            stems.add(stem)
    return stems


def live_states() -> dict[tuple[str, str], str]:
    """(out, stem) -> ps state of every live managed trainer."""
    found = {}
    for _, stat, cmd in L.ps_rows():
        if L.managed(cmd):
            spec = L.spec_from_cmd(cmd.split())
            if spec:
                found[(spec["out"], spec["stem"])] = stat
    return found


def state_of(d: str, stem: str, rows: list[dict], stat: str | None, queued: bool) -> str:
    if stat is not None:
        return "frozen" if "T" in stat else "running"
    if L.first_zero(rows) is not None:
        return "diverged"
    if os.path.exists(os.path.join(d, stem + ".pt")):
        return "finished"
    if L.n_resumes(d, stem) >= L.MAX_RESUMES:
        return "unfinished(3 resumes)"
    return "queued" if queued else "crashed"


def line(d: str, stem: str, state: str, rows: list[dict]) -> str:
    last, zero = (rows[-1] if rows else None), L.first_zero(rows)
    try:
        sps = f"{float(last['sps']):.0f}" if last else "-"
    except ValueError:
        sps = "-"
    finite = "-" if not rows else ("1111" if zero is None else "".join(zero[f] for f in L.FLAGS) + "@" + zero["step"])
    name = f"{os.path.basename(d)}/{stem}"
    return (f"{name:<48} {state:<21} step={last['step'] if last else '-'} sps={sps} "
            f"resumes={L.n_resumes(d, stem)} finite={finite}")


def report(dirs: list[str]) -> list[str]:
    """The health lines of every run of these directories."""
    queue = L.read_queue()
    live = live_states()
    out = []
    for d in dirs:
        d = os.path.normpath(os.path.abspath(d))
        if "sealed" in d.split(os.sep):
            raise SystemExit(f"{d}: health never reads under sealed/")
        queued = {q["stem"] for q in queue if q["out"] == d}
        counts = dict.fromkeys(STATES, 0)
        lines = []
        for stem in sorted(stems_in(d, queue)):
            rows = L.health_rows(d, stem)
            state = state_of(d, stem, rows, live.get((d, stem)), stem in queued)
            counts[state] += 1
            lines.append(line(d, stem, state, rows))
        summary = ", ".join(f"{n} {s}" for s, n in counts.items() if n)
        out += [f"# {d}: {len(lines)} runs" + (f" ({summary})" if summary else "")] + lines
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="*", help="run directories (default: the stage directories of results/method)")
    a = p.parse_args(argv)
    dirs = a.dirs or [os.path.join(L.QDIR, s) for s in STAGES if os.path.isdir(os.path.join(L.QDIR, s))]
    for ln in report(dirs):
        print(ln)


if __name__ == "__main__":
    main()
