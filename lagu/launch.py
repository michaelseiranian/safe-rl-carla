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

The scheduler is power-aware (results/.queue.power: "auto" by default, or "always"). Under "auto"
it trains only on mains power: on battery it freezes every study run (any trainer writing under
results/) in place (SIGSTOP, nothing lost) and unfreezes them when power returns. results/.queue.paused freezes everything regardless
(lagu/pause.sh pause|resume). While training it holds one `caffeinate -i` so the Mac does not
idle-sleep; closing the lid still sleeps it, and training simply waits.

After a reboot, `python -m lagu.launch --recover` (or lagu/pause.sh resume) re-queues every run
that was started but never finished, from its last checkpoint, and restarts the scheduler. To
retire a run for good, delete its entries from <out>/launch.json; killing it is not enough.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDIR = os.path.join(ROOT, "results")
QUEUE, CAP, PIDFILE, QLOG = (os.path.join(QDIR, f) for f in (".queue.jsonl", ".queue.cap", ".queue.pid", ".queue.log"))
PAUSED, POWER = os.path.join(QDIR, ".queue.paused"), os.path.join(QDIR, ".queue.power")

TRAINER = re.compile(r"^\S*python[0-9.]* -m lagu[.]train ")   # python, python3, python3.10 ...
OUT = re.compile(r" --out (\S+)")


def parse_variant(spec: str):
    name, _, kv = spec.partition(":")
    flags = []
    for item in filter(None, kv.split(",")):
        k, v = item.split("=", 1)
        flags += ["--" + k.strip().replace("_", "-"), v.strip()]
    return name, flags


def managed(cmd: str) -> bool:
    """A study run: `python -m lagu.train` writing under results/. Trainers writing anywhere else
    (the test suite, scratch runs) are never counted, frozen or unfrozen by the queue."""
    m = TRAINER.match(cmd) and OUT.search(cmd)
    return bool(m) and os.path.join(ROOT, m.group(1)).startswith(QDIR + os.sep)


def live_trainers() -> list[str]:
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    return [ln + " " for ln in out.splitlines() if managed(ln)]


def trainer_pids() -> list[int]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    pids = []
    for ln in out.splitlines():
        pid, _, cmd = ln.strip().partition(" ")
        if managed(cmd.strip()):
            pids.append(int(pid))
    return pids


def on_mains() -> bool:
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    except OSError:
        return True                                     # no pmset (not a Mac): treat as mains
    return "AC Power" in out or "Battery Power" not in out


def training_allowed() -> tuple[bool, str]:
    if os.path.exists(PAUSED):
        return False, "paused by hand"
    try:
        policy = open(POWER).read().strip() or "auto"
    except OSError:
        policy = "auto"
    if policy == "always" or on_mains():
        return True, "on mains power" if policy == "auto" else "policy always"
    return False, "on battery"


def signal_all(pids: list[int], sig) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


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
    prefix = ["nice", "-n", str(spec["nice"])]          # the scheduler keeps the Mac awake, not each run
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
    with open(QLOG, "a") as log:
        proc = subprocess.Popen([sys.executable, "-m", "lagu.launch", "--scheduler"], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    with open(PIDFILE, "w") as f:
        f.write(str(proc.pid))
    print(f"started queue scheduler (pid {proc.pid}); log: {QLOG}")


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


class Scheduler:
    def __init__(self):
        self.caff = None          # one caffeinate -i held while training is allowed and work exists
        self.state = None         # last allow/deny reason, to log transitions only once

    def awake(self, want: bool) -> None:
        if want and (self.caff is None or self.caff.poll() is not None) and shutil.which("caffeinate"):
            self.caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
        elif not want and self.caff is not None and self.caff.poll() is None:
            self.caff.terminate()
            self.caff = None

    def tick(self) -> bool:
        """One pass. Returns True when there is nothing left to do (queue empty, no trainer alive)."""
        try:
            cap = int(open(CAP).read().strip())
        except (OSError, ValueError):
            cap = 12
        allow, why = training_allowed()
        if why != self.state:
            log(("training allowed: " if allow else "training frozen: ") + why)
            self.state = why
        with Locked():
            pids = trainer_pids()
            if not allow:
                signal_all(pids, signal.SIGSTOP)           # freeze in place; nothing is lost
                self.awake(False)
                return False
            signal_all(pids, signal.SIGCONT)               # harmless on running processes
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
            done = not keep and not trainer_pids()
            self.awake(not done)
            if done:                                      # decide and clear the pidfile under the lock:
                log("queue empty and no trainer alive; scheduler exiting")
                if os.path.exists(PIDFILE):
                    os.remove(PIDFILE)
            return done


def run_scheduler(period: float = 60.0) -> None:
    log(f"scheduler started (pid {os.getpid()})")
    sch = Scheduler()
    while True:
        try:
            if sch.tick():
                return
        except Exception:                                 # a transient failure never kills the queue
            log("tick failed:\n" + traceback.format_exc())
        time.sleep(period)


def spec_from_cmd(cmd: list[str], nice: int = 10) -> dict | None:
    """Rebuild a queue spec from a recorded trainer command line."""
    try:
        i = cmd.index("lagu.train")
    except ValueError:
        return None
    cmd = [sys.executable] + cmd[i - 1:]                   # keep "-m lagu.train ...", drop any old prefix
    get = lambda flag, default="": cmd[cmd.index(flag) + 1] if flag in cmd else default
    env, arm, seed, out, tag = get("--env", "SafetyPointGoal1-v0"), get("--arm"), get("--seed"), get("--out"), get("--tag")
    stem = f"{arm}{'-' + tag if tag else ''}_{env}_s{seed}"
    markers = [f"--env {env} ", f"--arm {arm} ", f"--seed {seed} ", f"--out {out} "] + ([f"--tag {tag} "] if tag else [])
    return {"stem": stem, "cmd": cmd, "out": out, "tag": tag, "markers": markers,
            "resume": "--resume" in cmd, "nice": nice}


def with_ckpt_every(spec: dict, every: int) -> dict:
    cmd = list(spec["cmd"])
    if "--ckpt-every" in cmd:
        cmd[cmd.index("--ckpt-every") + 1] = str(every)
    else:
        cmd += ["--ckpt-every", str(every)]
    return {**spec, "cmd": cmd}


def recover(ckpt_every: int) -> None:
    """Re-queue, at the front, every run that was started but never finished; resume it from its
    checkpoint when one exists. Also set the checkpoint period of everything still queued."""
    started = {}
    for manifest in sorted(glob.glob(os.path.join(QDIR, "*", "launch.json"))):
        try:
            entries = json.load(open(manifest))
        except ValueError:
            continue
        for e in entries:
            for r in (e.get("runs") or [e]):              # old manifests group runs per launch
                spec = spec_from_cmd(r.get("cmd", []))
                if spec:
                    started[(spec["out"], spec["stem"])] = spec
    with Locked():
        queue = [with_ckpt_every(q, ckpt_every) for q in read_queue()]
        queued = {(q["out"], q["stem"]) for q in queue}
        live = live_trainers()
        new = []
        for key, spec in started.items():
            if key in queued or finished(spec) or is_live(spec, live):
                continue
            cmd = [c for c in spec["cmd"] if c != "--resume"]
            spec = {**spec, "cmd": cmd, "resume": False}
            if resumable(spec):
                spec = {**spec, "cmd": cmd + ["--resume"], "resume": True}
            new.append(with_ckpt_every(spec, ckpt_every))
        write_queue(new + queue)
        ensure_scheduler()
    for spec in new:
        print(("resume  " if spec["resume"] else "restart ") + spec["stem"])
    print(f"re-queued {len(new)} interrupted runs at the front; {len(queue)} still queued; "
          f"checkpoint every {ckpt_every} steps")


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
    p.add_argument("--recover", action="store_true", help="re-queue interrupted runs after a reboot")
    p.add_argument("--ckpt-every", type=int, default=100_000, help="with --recover: checkpoint period")
    args = p.parse_args()
    if args.scheduler:
        run_scheduler()
        return
    if args.recover:
        recover(args.ckpt_every)
        return
    if not args.out:
        p.error("--out is required")
    args.out = os.path.abspath(args.out)          # children run with cwd=ROOT; keep every path absolute
    if not args.out.startswith(QDIR + os.sep):    # managed() only sees runs there: cap, freeze, de-duplication
        p.error("--out must be under results/")
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
