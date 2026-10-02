"""Launch a grid of detached training runs that outlive this shell and this session.

    python -m lagu.launch --out results/lr_sweep --arms td3lag --seeds 0 1 --steps 500000 \
        --eval-every 25000 --variant lr1e-3:lambda_lr=1e-3 --variant epi:dual=episodic,lambda_lr=0.035

Each (arm, variant, seed) becomes one `python -m lagu.train` process in its own session, wrapped
in `caffeinate -i` (macOS) so the machine does not idle-sleep while it runs; a closed lid still
sleeps. Re-run the same command with --resume to continue every run from its last checkpoint.

With --max-procs N the runs go into a shared queue (results/.queue.jsonl) instead, and one
detached scheduler starts them whenever fewer than N trainers are alive, re-reading the cap from
results/.queue.cap every minute (edit that file to change it). --front puts the new runs at the
head of the queue. Runs that are already alive, or already finished without --resume, are
skipped. Check progress with lagu/status.sh <out>; the scheduler logs to results/.queue.log.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDIR = os.path.join(ROOT, "results")
QUEUE, CAP, PIDFILE, QLOG = (os.path.join(QDIR, f) for f in (".queue.jsonl", ".queue.cap", ".queue.pid", ".queue.log"))
TRAINER = re.compile(r"^\S*python[0-9.]* -m lagu[.]train ")   # python, python3, python3.10 ...


def parse_variant(spec: str):
    name, _, kv = spec.partition(":")
    flags = []
    for item in filter(None, kv.split(",")):
        k, v = item.split("=", 1)
        flags += ["--" + k.strip().replace("_", "-"), v.strip()]
    return name, flags


def live_trainers() -> list[str]:
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    return [ln + " " for ln in out.splitlines() if TRAINER.match(ln)]


def is_live(spec: dict, live: list[str]) -> bool:
    return any(all(m in ln for m in spec["markers"]) and (spec["tag"] or "--tag " not in ln) for ln in live)


def build_specs(args) -> list[dict]:
    variants = [parse_variant(v) for v in args.variant] or [("", [])]
    specs = []
    for arm in args.arms:
        for name, flags in variants:
            for seed in args.seeds:
                cmd = [sys.executable, "-m", "lagu.train", "--env", args.env, "--arm", arm,
                       "--seed", str(seed), "--steps", str(args.steps),
                       "--eval-every", str(args.eval_every), "--out", args.out] + flags
                if name:
                    cmd += ["--tag", name]
                if args.resume:
                    cmd.append("--resume")
                stem = f"{arm}{'-' + name if name else ''}_{args.env}_s{seed}"
                markers = [f"--env {args.env} ", f"--arm {arm} ", f"--seed {seed} ", f"--out {args.out} "]
                if name:
                    markers.append(f"--tag {name} ")
                specs.append({"stem": stem, "cmd": cmd, "out": args.out, "tag": name, "markers": markers,
                              "resume": bool(args.resume), "nice": args.nice})
    return specs


def spawn(spec: dict) -> int:
    prefix = ["nice", "-n", str(spec["nice"])] + (["caffeinate", "-i"] if shutil.which("caffeinate") else [])
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    os.makedirs(spec["out"], exist_ok=True)
    with open(os.path.join(spec["out"], spec["stem"] + ".log"), "a") as log:
        proc = subprocess.Popen(prefix + spec["cmd"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    manifest = os.path.join(spec["out"], "launch.json")
    try:
        prev = json.load(open(manifest)) if os.path.exists(manifest) else []
    except ValueError:                                            # a corrupt manifest never blocks a launch
        prev = []
    with open(manifest + ".tmp", "w") as f:
        json.dump(prev + [{"time": time.strftime("%Y-%m-%d %H:%M:%S"), "stem": spec["stem"],
                           "pid": proc.pid, "cmd": spec["cmd"]}], f, indent=1)
    os.replace(manifest + ".tmp", manifest)
    return proc.pid


def finished(spec: dict) -> bool:
    return os.path.exists(os.path.join(spec["out"], spec["stem"] + ".pt"))


def resumable(spec: dict) -> bool:
    return os.path.exists(os.path.join(spec["out"], spec["stem"] + ".ckpt.pt"))


class Locked:
    def __enter__(self):
        os.makedirs(QDIR, exist_ok=True)
        self.f = open(QUEUE + ".lock", "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def read_queue() -> list[dict]:
    if not os.path.exists(QUEUE):
        return []
    with open(QUEUE) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_queue(specs: list[dict]) -> None:
    tmp = QUEUE + ".tmp"
    with open(tmp, "w") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")
    os.replace(tmp, QUEUE)


def scheduler_alive() -> bool:
    try:
        pid = int(open(PIDFILE).read().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
    return "lagu.launch" in cmd and "--scheduler" in cmd


def ensure_scheduler() -> None:
    """Call with the queue lock held: the pidfile is written here, before the lock is released."""
    if scheduler_alive():
        return
    prefix = ["caffeinate", "-i"] if shutil.which("caffeinate") else []
    with open(QLOG, "a") as log:
        proc = subprocess.Popen(prefix + [sys.executable, "-m", "lagu.launch", "--scheduler"], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    with open(PIDFILE, "w") as f:
        f.write(str(proc.pid))
    print(f"started queue scheduler (pid {proc.pid}); log: {QLOG}")


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def run_scheduler(period: float = 60.0) -> None:
    log(f"scheduler started (pid {os.getpid()})")
    while True:
        done = False
        try:
            try:
                cap = int(open(CAP).read().strip())
            except (OSError, ValueError):
                cap = 12
            with Locked():
                queue = read_queue()
                live = live_trainers()
                n_live, keep = len(live), []
                for spec in queue:
                    if is_live(spec, live):
                        log(f"skip {spec['stem']}: already running")
                    elif finished(spec) and not (spec["resume"] and resumable(spec)):
                        log(f"skip {spec['stem']}: already finished")
                    elif n_live < cap:
                        log(f"start {spec['stem']} (pid {spawn(spec)}; {n_live + 1}/{cap} slots)")
                        live.append(" ".join(spec["cmd"]) + " ")   # a duplicate later in this tick is seen as live
                        n_live += 1
                    else:
                        keep.append(spec)
                write_queue(keep)
                if not keep:                                      # decide and clear the pidfile under the lock:
                    log("queue empty; scheduler exiting")          # an enqueue now starts a fresh scheduler
                    if os.path.exists(PIDFILE):
                        os.remove(PIDFILE)
                    done = True
        except Exception:                                         # a transient failure never kills the queue
            log("tick failed:\n" + traceback.format_exc())
        if done:
            return
        time.sleep(period)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--env", default="SafetyPointGoal1-v0")
    p.add_argument("--arms", nargs="+", default=["lagu"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--variant", action="append", default=[],
                   help="name:key=val,key=val ; keys are lagu.train flags (no variant = defaults)")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--nice", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-procs", type=int, default=0, help="queue the runs; at most N trainers alive")
    p.add_argument("--front", action="store_true", help="with --max-procs: put these runs first")
    p.add_argument("--set-cap", action="store_true", help="with --max-procs: overwrite results/.queue.cap")
    p.add_argument("--scheduler", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.scheduler:
        run_scheduler()
        return
    if not args.out:
        p.error("--out is required")
    args.out = os.path.abspath(args.out)          # children run with cwd=ROOT; keep every path absolute
    specs = build_specs(args)
    if args.dry_run:
        for s in specs:
            print(" ".join(shlex.quote(c) for c in s["cmd"]))
        return
    os.makedirs(args.out, exist_ok=True)
    if args.max_procs:
        with Locked():
            queue = read_queue()
            seen, new = {s["stem"] + s["out"] for s in queue}, []
            for s in specs:                                       # dedup against the queue and within this call
                k = s["stem"] + s["out"]
                if k not in seen:
                    seen.add(k)
                    new.append(s)
            write_queue(new + queue if args.front else queue + new)
            if args.set_cap or not os.path.exists(CAP):           # never silently override a hand-lowered cap
                with open(CAP, "w") as f:
                    f.write(str(args.max_procs))
            cap = open(CAP).read().strip()
            ensure_scheduler()
        print(f"queued {len(new)} runs ({len(specs) - len(new)} duplicate or already queued); effective cap {cap}")
        return
    live, n = live_trainers(), 0
    for s in specs:
        if is_live(s, live):
            print(f"skip {s['stem']}: a live process is already writing it")
            continue
        spawn(s)
        n += 1
    print(f"launched {n} detached runs -> {args.out}   (progress: lagu/status.sh {args.out})")


if __name__ == "__main__":
    main()
