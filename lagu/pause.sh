#!/usr/bin/env bash
# lagu/pause.sh pause | resume | status | power auto|always | cap N
#   pause   freeze every training run in place now and keep it frozen (nothing is lost)
#   resume  allow training again; after a reboot this also re-queues every interrupted run from
#           its last checkpoint and restarts the scheduler
#   power   auto   = train only on mains power, freeze on battery (default)
#           always = train on battery too
#   cap N   at most N runs at once (12 by default; takes effect as runs finish)
#   status  what is running, frozen and queued
# Closing the lid just sleeps the Mac; training waits and carries on when it wakes.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
# study runs only: trainers writing under results/ (never the test suite's or scratch runs)
pids() { ps -axo pid=,command= | awk -v r="$PWD/results/" '$2 ~ /python[0-9.]*$/ && $3 == "-m" && $4 == "lagu.train" {
  for (i = 5; i < NF; i++) if ($i == "--out" && (index($(i+1), r) == 1 || index($(i+1), "results/") == 1)) { print $1; break } }'; }
case "$1" in
  pause)
    touch results/.queue.paused
    p=$(pids); [ -n "$p" ] && kill -STOP $p
    echo "paused: $(echo $p | wc -w | tr -d ' ') runs frozen; they stay frozen until 'lagu/pause.sh resume'" ;;
  resume)
    rm -f results/.queue.paused
    p=$(pids); [ -n "$p" ] && kill -CONT $p
    [ "$(cat results/.queue.cap 2>/dev/null)" = "0" ] && echo 12 > results/.queue.cap
    $PY -m lagu.launch --recover
    echo "resumed. Training runs only on mains power unless you set 'lagu/pause.sh power always'." ;;
  power)
    case "$2" in auto|always) echo "$2" > results/.queue.power; echo "power policy: $2";;
      *) echo "usage: lagu/pause.sh power auto|always"; exit 2;; esac ;;
  cap)
    [[ "$2" =~ ^[0-9]+$ ]] || { echo "usage: lagu/pause.sh cap N"; exit 2; }
    echo "$2" > results/.queue.cap; echo "cap: $2" ;;
  status)
    src=$(pmset -g batt | head -1 | sed "s/Now drawing from //; s/'//g")
    sch=$(cat results/.queue.pid 2>/dev/null); alive=no
    [ -n "$sch" ] && ps -o command= -p "$sch" 2>/dev/null | grep -q -- '--scheduler' && alive=yes
    echo "power: $src   policy: $(cat results/.queue.power 2>/dev/null || echo auto)   paused by hand: $([ -f results/.queue.paused ] && echo yes || echo no)"
    echo "scheduler alive: $alive   cap: $(cat results/.queue.cap 2>/dev/null)   queued: $(wc -l < results/.queue.jsonl 2>/dev/null | tr -d ' ')"
    p=$(pids)
    if [ -n "$p" ]; then
      ps -o stat=,%cpu= -p $(echo $p | tr ' ' ',') | awk '{ if ($1 ~ /T/) f++; else r++ } END { printf "runs: %d training, %d frozen\n", r, f }'
    else echo "runs: none alive"; fi ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
