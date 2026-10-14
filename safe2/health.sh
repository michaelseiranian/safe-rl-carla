#!/usr/bin/env bash
# safe2/health.sh [DIR ...]   = python -m safe2.health [DIR ...]
# Per run only the stem, state (queued, running, frozen, finished, crashed, diverged, unfinished), last step,
# steps/s, resume count and finite flags; never a return or a cost. Default: results/method/{p1,p2,conf,cont3m}.
R="$(cd "$(dirname "$0")/.." && pwd)" || exit 1
PY=$R/.venv/bin/python
[ -x "$PY" ] || PY=/Users/michaelseiranian/Desktop/safe-rl-carla/.venv/bin/python   # the worktree has no .venv
PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m safe2.health "$@"
