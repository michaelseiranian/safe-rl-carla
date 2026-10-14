#!/usr/bin/env bash
# safe2/pause.sh pause | resume | status | power auto|always | cap N
#   pause   freeze every method-study run in place now and keep it frozen (nothing is lost)
#   resume  allow training again; after a reboot this also re-queues every interrupted run from
#           its last checkpoint and restarts the scheduler (python -m safe2.launch --recover)
#   power   auto   = train only on mains power, freeze on battery (default)
#           always = train on battery too
#   cap N   at most N study trainers at once, the Lag-U study's included (12 by default; takes effect as runs finish)
#   status  what is running, frozen and queued
# Closing the lid just sleeps the Mac; training waits and carries on when it wakes.
# A copy of lagu/pause.sh (safe2/SPEC.md section 6.2): the queue files are results/method/.queue.*, it acts
# only on `-m safe2.train` runs writing under results/method/, and resume runs `-m safe2.launch --recover`.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
[ -x "$PY" ] || PY=/Users/michaelseiranian/Desktop/safe-rl-carla/.venv/bin/python   # the worktree has no .venv
Q=results/method
# study runs only: safe2.train writing under results/method/ (never the test suite's, smoke runs or Lag-U's)
pids() { ps -axo pid=,command= | awk -v r="$PWD/results/method/" '$2 ~ /python[0-9.]*$/ && $3 == "-m" && $4 == "safe2.train" {
  for (i = 5; i < NF; i++) if ($i == "--out" && $(i+1) !~ /\.\./ && (index($(i+1), r) == 1 || index($(i+1), "results/method/") == 1)) { print $1; break } }'; }
case "$1" in
  pause)
    mkdir -p $Q && touch $Q/.queue.paused
    p=$(pids); [ -n "$p" ] && kill -STOP $p
    echo "paused: $(echo $p | wc -w | tr -d ' ') runs frozen; they stay frozen until 'safe2/pause.sh resume'" ;;
  resume)
    rm -f $Q/.queue.paused
    p=$(pids); [ -n "$p" ] && kill -CONT $p
    [ "$(cat $Q/.queue.cap 2>/dev/null)" = "0" ] && echo 12 > $Q/.queue.cap
    $PY -m safe2.launch --recover
    echo "resumed. Training runs only on mains power unless you set 'safe2/pause.sh power always'." ;;
  power)
    case "$2" in auto|always) mkdir -p $Q && echo "$2" > $Q/.queue.power; echo "power policy: $2";;
      *) echo "usage: safe2/pause.sh power auto|always"; exit 2;; esac ;;
  cap)
    [[ "$2" =~ ^[0-9]+$ ]] || { echo "usage: safe2/pause.sh cap N"; exit 2; }
    mkdir -p $Q && echo "$2" > $Q/.queue.cap; echo "cap: $2" ;;
  status)
    src=$(pmset -g batt | head -1 | sed "s/Now drawing from //; s/'//g")
    sch=$(cat $Q/.queue.pid 2>/dev/null); alive=no
    [ -n "$sch" ] && ps -o command= -p "$sch" 2>/dev/null | grep -q -- '--scheduler' && alive=yes
    echo "power: $src   policy: $(cat $Q/.queue.power 2>/dev/null || echo auto)   paused by hand: $([ -f $Q/.queue.paused ] && echo yes || echo no)"
    echo "scheduler alive: $alive   cap: $(cat $Q/.queue.cap 2>/dev/null)   queued: $(cat $Q/.queue.jsonl 2>/dev/null | wc -l | tr -d ' ')"
    p=$(pids)
    if [ -n "$p" ]; then
      ps -o stat=,%cpu= -p $(echo $p | tr ' ' ',') | awk '{ if ($1 ~ /T/) f++; else r++ } END { printf "runs: %d training, %d frozen\n", r, f }'
    else echo "runs: none alive"; fi ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
