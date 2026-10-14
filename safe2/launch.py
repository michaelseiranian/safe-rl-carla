"""The method-study queue: detached `python -m safe2.train` runs, power-aware freezing and recovery
(safe2/SPEC.md section 6.3; an adapted copy of lagu/launch.py, which is not changed).

    python -m safe2.launch --out results/method/p1 --configs B1 B2 --env SafetyPointGoal1-v0 \
        --seeds 0 1 2 [--steps 1000000] [--drop-ckpt 0|1] [--from-bvs NAME --rho-star X] [--resume] \
        [--max-procs 12 [--set-cap]] [--front] [--dry-run]
    python -m safe2.launch --from-plan results/method/plan.json --stage conf [--front] [--max-procs N] [--dry-run]
    python -m safe2.launch --recover
    python -m safe2.launch --retire STAGE/STEM [STAGE/STEM ...]

Each (config, env, seed) becomes one `python -m safe2.train` process in its own session, niced, with
OMP_NUM_THREADS = MKL_NUM_THREADS = 1, its console appended to <out>/<stem>.log and its command
recorded in <out>/launch.json. --out must lie under results/method/. The runs always go into the
method queue, results/method/.queue.jsonl, and one detached scheduler starts them whenever the cap
leaves a slot. It re-reads the cap from results/method/.queue.cap every minute (12 when the file is
missing; --max-procs N writes it when it does not exist yet, --set-cap always; or `safe2/pause.sh cap
N`). --front puts the new runs at the head of the queue. The scheduler logs to
results/method/.queue.log.

Cap. The cap counts every live `-m safe2.train` process (frozen included) whose --out lies under the
results/ directory of any checkout of this repository (the main checkout and its worktrees), plus the
Lag-U study's slots: per checkout, its live `-m lagu.train` runs or, while its queue
(<checkout>/results/.queue.jsonl, read only) holds work, as many as its own cap (.queue.cap there) lets
its scheduler refill. So the method queue never takes a slot the Lag-U scheduler is about to refill;
lower Lag-U's cap (lagu/pause.sh cap N) to give the method study 12 - N slots. Frozen Lag-U runs keep
their slots unless Lag-U's cap is 0. Freezing, unfreezing, recovery and de-duplication act only on the
runs managed() accepts: safe2.train writing under results/method/ of this checkout.

Power. As in lagu.launch: results/method/.queue.power is "auto" (the default: train only on mains
power; on battery every managed run is frozen in place with SIGSTOP and unfrozen with SIGCONT when
power returns) or "always". results/method/.queue.paused freezes everything regardless
(safe2/pause.sh pause | resume). While training, the scheduler holds one `caffeinate -i`.

Refusals (SPEC 3.6), checked when a run is queued and again before it starts (blocked()). A run is
never started or resumed when it is cut (results/method/cuts.json), when the trainer would refuse its
command (safe2.train.refusal: e.g. 3M outside cont3m/, a seed that is not a study seed), when its
public CSV shows a finite flag of 0 ("diverged"), when its json records 3 resumes ("unfinished"), or
when it is a 3M continuation without its checkpoint. A finished run (its .pt exists) is final, with or
without --resume. A run that has already started (its json or checkpoint exists) is not started
again from scratch: `--recover` resumes it from its checkpoint, or restarts it with the trainer's
--restart when it crashed before its first checkpoint. The checkpoint period is the trainer's
(100k); unlike lagu.launch, --recover never changes it (the trainer refuses dev-only flags under
results/method/).

--from-plan queues every entry of plan.json's "runs" (safe2.analyze plan) whose "stage" equals --stage,
with the exact "argv" recorded there: the trainer command after the interpreter (`-m safe2.train ...`),
whose relative --out is taken from the repository root. The runs cut in cuts.json (next to plan.json;
it must exist) are left out: an entry {"date", "item", "what", "stage", "stems", "reason"} (safe2.analyze
cut) cuts <stage>/<stem> for each of its stems; queued runs that are now cut are unqueued, and live
ones are listed for the lead to stop. A planned run with "resume_from" (a 3M continuation) first gets
that 1M checkpoint copied to <out>/<stem>.ckpt.pt, and only once its 1M run is finished and not
diverged and its 1M held-out evaluation exists (sealed/heldout/<stem>.deployed.json.enc;
prereg_method.md section 9); otherwise it prints "skip <stage>/<stem>: source ..." and goes on.

After a reboot, `python -m safe2.launch --recover` (or safe2/pause.sh resume) re-queues at the front
every started, unfinished run of results/method/*/launch.json, from its checkpoint (or from scratch
when it crashed before its first checkpoint), prints "diverged" or "unfinished" for the runs it must
not resume, and restarts the scheduler.

--retire STAGE/STEM (a bug rerun, protocol section 16) moves a run that is not alive out of the study:
its run files (<stem>.csv, .json, .pt, .ckpt.pt, .log, .snap/) and its sealed files (the record file
sealed/<stage>/<stem>.rows and every held-out output sealed/heldout/<stem>.*) go to
results/method/retired/<UTC time>/ (outside sealed/, so `seal verify` no longer expects them; their
hashes stay in HASHLOG.tsv), its entries leave <out>/launch.json and the queue. Log every retirement
in protocol section 22. Killing a run is not enough to retire it.
"""
from __future__ import annotations

import argparse
import csv
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

from safe2 import configs, train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDIR = os.path.join(ROOT, "results", "method")
QUEUE, CAP, PIDFILE, QLOG = (os.path.join(QDIR, f) for f in (".queue.jsonl", ".queue.cap", ".queue.pid", ".queue.log"))
PAUSED, POWER = os.path.join(QDIR, ".queue.paused"), os.path.join(QDIR, ".queue.power")

TRAINER = re.compile(r"^\S*python[0-9.]* -m safe2[.]train ")               # managed: this study's trainer
ANY_TRAINER = re.compile(r"^\S*python[0-9.]* -m (?:lagu|safe2)[.]train ")  # counted against the cap
OUT = re.compile(r" --out (\S+)")
CONTINUATION_STEPS = 3_000_000            # safe2.train.CONTINUATION_STEPS (tests check the copy)
MAX_RESUMES = 3                           # SPEC 3.6: never resumed after 3 resumes ("unfinished")
FLAGS = ("finite_out", "finite_net", "finite_loss", "finite_dual")   # safe2.train.GROUPS (tests check)
DEFAULT_CAP = 12


def _resolve(path: str) -> str:
    """An absolute, normalised path; a relative one is taken from the repository root (where the
    launcher starts every trainer)."""
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(ROOT, path))


def checkouts() -> list[str]:
    """The root of every checkout of this repository: the main checkout and each linked worktree, read
    from the .git files (no git call)."""
    dotgit = os.path.join(ROOT, ".git")
    common = dotgit
    if os.path.isfile(dotgit):                  # a worktree: "gitdir: <main>/.git/worktrees/<name>"
        with open(dotgit) as f:
            common = os.path.dirname(os.path.dirname(f.read().split("gitdir:", 1)[1].strip()))
    roots = {ROOT, os.path.dirname(common)}
    for path in glob.glob(os.path.join(common, "worktrees", "*", "gitdir")):
        with open(path) as f:
            roots.add(os.path.dirname(f.read().strip()))
    return sorted(roots)


try:
    CAP_ROOTS = [os.path.join(c, "results") for c in checkouts()]
except (OSError, IndexError):                   # no readable .git: this checkout only
    CAP_ROOTS = [os.path.join(ROOT, "results")]


def managed(cmd: str) -> bool:
    """A method-study run: `python -m safe2.train` writing under results/method/. Trainers writing
    anywhere else (the test suite, smoke runs, the Lag-U study) are never frozen, unfrozen or
    recovered by this queue."""
    m = TRAINER.match(cmd) and OUT.search(cmd)
    return bool(m) and _resolve(m.group(1)).startswith(QDIR + os.sep)


def counted(cmd: str) -> bool:
    """A study trainer of either study (lagu.train or safe2.train writing under any checkout's
    results/): what the cap counts."""
    m = ANY_TRAINER.match(cmd) and OUT.search(cmd)
    return bool(m) and any(_resolve(m.group(1)).startswith(r + os.sep) for r in CAP_ROOTS)


def ps_rows() -> list[tuple[int, str, str]]:
    """(pid, state, command) of every process."""
    out = subprocess.run(["ps", "-axo", "pid=,stat=,command="], capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        f = ln.split(None, 2)
        if len(f) == 3 and f[0].isdigit():
            rows.append((int(f[0]), f[1], f[2].strip()))
    return rows


def trainer_pids() -> list[int]:
    return [pid for pid, _, cmd in ps_rows() if managed(cmd)]


def live_trainers() -> list[str]:
    return [cmd + " " for _, _, cmd in ps_rows() if managed(cmd)]


def lagu_queue(root: str) -> tuple[int, int]:
    """(cap, queued runs) of the Lag-U queue of one checkout's results/ directory, read only (the files
    of lagu/launch.py; a missing cap file means its default 12)."""
    try:
        cap = int(open(os.path.join(root, ".queue.cap")).read().strip())
    except (OSError, ValueError):
        cap = DEFAULT_CAP
    try:
        with open(os.path.join(root, ".queue.jsonl")) as f:
            queued = sum(1 for ln in f if ln.strip())
    except OSError:
        queued = 0
    return cap, queued


def n_alive() -> int:
    """Slots taken against the cap: every live safe2.train study trainer (frozen included), plus, per
    checkout, the Lag-U study's slots max(running, min(Lag-U cap, alive + queued)): its live runs, or
    as many as its scheduler will refill while its queue holds work (frozen Lag-U runs count unless
    its cap is 0)."""
    used, lagu = 0, {r: [0, 0] for r in CAP_ROOTS}             # root -> [alive, running]
    for _, stat, cmd in ps_rows():
        if not counted(cmd):
            continue
        if TRAINER.match(cmd):
            used += 1
            continue
        out = _resolve(OUT.search(cmd).group(1))
        root = next(r for r in CAP_ROOTS if out.startswith(r + os.sep))
        lagu[root][0] += 1
        lagu[root][1] += "T" not in stat
    for root, (alive, running) in lagu.items():
        cap, queued = lagu_queue(root)
        used += max(running, min(cap, alive + queued))
    return used


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
    return any(all(m in ln for m in spec["markers"]) for ln in live)


# --- run specs ----------------------------------------------------------------------------------
def _flag(cmd: list[str], flag: str) -> str | None:
    i = cmd.index(flag) + 1 if flag in cmd else len(cmd)
    return cmd[i] if i < len(cmd) else None


def spec_from_cmd(cmd: list[str], nice: int = 10) -> dict | None:
    """A queue spec from a trainer command line (anything before `-m safe2.train`, such as nice or an
    old interpreter path, is replaced by this interpreter). None if it is not a complete safe2.train
    command. The stem is <config>_<env>_s<seed> (SPEC 0); the markers find its live process."""
    try:
        i = cmd.index("safe2.train")
    except ValueError:
        return None
    if i < 1 or cmd[i - 1] != "-m":
        return None
    cmd = [sys.executable] + cmd[i - 1:]
    config, env, seed, out = (_flag(cmd, f) for f in ("--config", "--env", "--seed", "--out"))
    if None in (config, env, seed, out):
        return None
    return {"stem": f"{config}_{env}_s{seed}", "cmd": cmd, "out": _resolve(out),
            "markers": [f"--config {config} ", f"--env {env} ", f"--seed {seed} ", f"--out {out} "],
            "resume": "--resume" in cmd, "restart": "--restart" in cmd, "nice": nice}


def build_specs(args) -> list[dict]:
    specs = []
    for config in args.configs:
        for env in args.env:
            for seed in args.seeds:
                cmd = [sys.executable, "-m", "safe2.train", "--config", config, "--env", env, "--seed", str(seed),
                       "--steps", str(args.steps), "--out", args.out, "--drop-ckpt", str(args.drop_ckpt)]
                if config in configs.DERIVED:                     # BVSU / BVSD only (the trainer refuses others)
                    cmd += ["--from-bvs", args.from_bvs]
                    if configs.DERIVED[config] == "bvs_u":
                        cmd += ["--rho-star", args.rho_star]
                if args.resume:
                    cmd.append("--resume")
                specs.append(spec_from_cmd(cmd, args.nice))
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


# --- what a run's public files say ----------------------------------------------------------------
def _path(spec: dict, ext: str) -> str:
    return os.path.join(spec["out"], spec["stem"] + ext)


def finished(spec: dict) -> bool:
    return os.path.exists(_path(spec, ".pt"))


def resumable(spec: dict) -> bool:
    return os.path.exists(_path(spec, ".ckpt.pt"))


def started(spec: dict) -> bool:
    return os.path.exists(_path(spec, ".json")) or resumable(spec)


def health_rows(out: str, stem: str) -> list[dict]:
    """step, sps and the four finite flags of every complete row of a run's public CSV; no other
    column is kept. A torn last line (a crash mid-write) is dropped."""
    path = os.path.join(out, stem + ".csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if None in r or None in r.values() or not (r.get("step") or "").isdigit():
                continue
            rows.append({k: r.get(k, "") for k in ("step", "sps") + FLAGS})
    return rows


def first_zero(rows: list[dict]) -> dict | None:
    """The first row with a finite flag of 0 (the run diverged; SPEC 3.6), or None."""
    return next((r for r in rows if any(r[f] == "0" for f in FLAGS)), None)


def n_resumes(out: str, stem: str) -> int:
    """The resume count recorded in the run's json (0 before the run starts): safe2.train.prior_resumes
    of the json (a continuation's json counts only the continuation's own resumes)."""
    try:
        with open(os.path.join(out, stem + ".json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return 0
    return train.prior_resumes(meta)


def cut_ids() -> set:
    """<stage>/<stem> of every run cut in results/method/cuts.json (safe2.analyze.cut_ids; empty before
    plan.json exists)."""
    path = os.path.join(QDIR, "cuts.json")
    try:
        with open(path) as f:
            cuts = json.load(f)
    except (OSError, ValueError):
        return set()
    return {f"{c['stage']}/{st}" for c in cuts if "stage" in c for st in c.get("stems", [])}


def trainer_refusal(spec: dict) -> str | None:
    """What safe2.train would refuse in this command (its study guard), checked before queueing."""
    try:
        return train.refusal(train.parse(spec["cmd"][3:]))
    except SystemExit:                                  # argparse error (already printed to stderr)
        return "not a valid trainer command"


def blocked(spec: dict) -> str | None:
    """Why a run must not be started or resumed: "cut", "refused: <the trainer's reason>", "diverged",
    "unfinished" or "no checkpoint" (None = it may)."""
    if _name(spec) in cut_ids():
        return "cut"
    why = trainer_refusal(spec)
    if why:
        return f"refused: {why}"
    if first_zero(health_rows(spec["out"], spec["stem"])) is not None:
        return "diverged"
    if n_resumes(spec["out"], spec["stem"]) >= MAX_RESUMES:
        return "unfinished"
    if _flag(spec["cmd"], "--steps") == str(CONTINUATION_STEPS) and not resumable(spec):
        return "no checkpoint"                          # a 3M continuation never starts from scratch
    return None


def why_skip(spec: dict, live: list[str]) -> str | None:
    """Why a new launch request for this run is not queued or started (None = it is)."""
    if is_live(spec, live):
        return "already running"
    if finished(spec):                                  # final: safe2 never extends a run in place
        return "already finished"
    if started(spec) and not spec["resume"]:
        return "already started (python -m safe2.launch --recover resumes it)"
    return blocked(spec)


# --- the queue ----------------------------------------------------------------------------------
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
    return "safe2.launch" in cmd and "--scheduler" in cmd


def ensure_scheduler() -> None:
    """Call with the queue lock held: the pidfile is written here, before the lock is released."""
    if scheduler_alive():
        return
    with open(QLOG, "a") as log:
        proc = subprocess.Popen([sys.executable, "-m", "safe2.launch", "--scheduler"], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    with open(PIDFILE, "w") as f:
        f.write(str(proc.pid))
    print(f"started queue scheduler (pid {proc.pid}); log: {QLOG}")


def enqueue(specs: list[dict], front: bool, max_procs: int = 0, set_cap: bool = False) -> tuple[int, str]:
    """Add specs to the queue (de-duplicated against it and within the call) and make sure the
    scheduler runs. Returns (number added, effective cap)."""
    with Locked():
        queue = read_queue()
        seen, new = {(s["out"], s["stem"]) for s in queue}, []
        for s in specs:
            if (s["out"], s["stem"]) not in seen:
                seen.add((s["out"], s["stem"]))
                new.append(s)
        write_queue(new + queue if front else queue + new)
        if max_procs and (set_cap or not os.path.exists(CAP)):   # never silently override a hand-lowered cap
            with open(CAP, "w") as f:
                f.write(str(max_procs))
        cap = open(CAP).read().strip() if os.path.exists(CAP) else f"{DEFAULT_CAP} (default)"
        ensure_scheduler()
    return len(new), cap


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
        """One pass. Returns True when there is nothing left to do (queue empty, no managed run alive)."""
        try:
            cap = int(open(CAP).read().strip())
        except (OSError, ValueError):
            cap = DEFAULT_CAP
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
            n_live, keep = n_alive(), []                   # the cap counts the Lag-U study's runs too
            for spec in queue:
                if is_live(spec, live):
                    log(f"skip {spec['stem']}: already running")
                elif finished(spec):
                    log(f"skip {spec['stem']}: already finished")
                elif started(spec) and not (spec["resume"] or spec.get("restart")):
                    log(f"skip {spec['stem']}: already started (--recover resumes it)")
                elif n_live < cap:
                    why_not = blocked(spec)
                    if why_not:
                        log(f"skip {spec['stem']}: {why_not}")
                        continue
                    log(f"start {spec['stem']} (pid {spawn(spec)}; {n_live + 1}/{cap} slots)")
                    live.append(" ".join(spec["cmd"]) + " ")   # a duplicate later in this tick is seen as live
                    n_live += 1
                else:
                    keep.append(spec)
            write_queue(keep)
            done = not keep and not trainer_pids()
            self.awake(not done)
            if done:                                      # decide and clear the pidfile under the lock:
                log("queue empty and no managed run alive; scheduler exiting")
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


# --- recovery -------------------------------------------------------------------------------------
def _name(spec: dict) -> str:
    return f"{os.path.basename(spec['out'])}/{spec['stem']}"


def recover() -> None:
    """Re-queue, at the front, every run that was started but never finished: resumed from its
    checkpoint when one exists, else restarted. Diverged and 3-resume runs are reported, not queued."""
    started_runs = {}
    for manifest in sorted(glob.glob(os.path.join(QDIR, "*", "launch.json"))):
        try:
            entries = json.load(open(manifest))
        except ValueError:
            continue
        for e in entries:
            spec = spec_from_cmd(e.get("cmd", []))
            if spec:
                started_runs[(spec["out"], spec["stem"])] = spec
    held = []
    with Locked():
        queue = read_queue()
        queued = {(q["out"], q["stem"]) for q in queue}
        live = live_trainers()
        new = []
        for key, spec in started_runs.items():
            if key in queued or finished(spec) or is_live(spec, live):
                continue
            cmd = [c for c in spec["cmd"] if c not in ("--resume", "--restart")]
            spec = {**spec, "cmd": cmd, "resume": False, "restart": False}
            if resumable(spec):
                spec = {**spec, "cmd": cmd + ["--resume"], "resume": True}
            elif started(spec):                         # crashed before its first checkpoint
                spec = {**spec, "cmd": cmd + ["--restart"], "restart": True}
            why = blocked(spec)
            if why:
                held.append((why, spec))
            else:
                new.append(spec)
        write_queue(new + queue)
        ensure_scheduler()
    for spec in new:
        print(("resume  " if spec["resume"] else "restart ") + _name(spec))
    for why, spec in held:
        print(f"{why:<10} {_name(spec)}")
    print(f"re-queued {len(new)} interrupted runs at the front; {len(held)} not resumable; {len(queue)} still queued")


# --- the plan ---------------------------------------------------------------------------------------
def trainer_cmd(argv: list[str]) -> list[str]:
    """[python, -m, safe2.train, *args] from a plan argv (`-m safe2.train` and the trainer arguments, or
    the arguments alone), with a relative --out made absolute from the repository root."""
    args = list(argv)
    if "safe2.train" in args:
        args = args[args.index("safe2.train") + 1:]
    if "--out" in args and args.index("--out") + 1 < len(args):
        i = args.index("--out") + 1
        args[i] = _resolve(args[i])
    return [sys.executable, "-m", "safe2.train"] + args


def load_plan(plan_path: str, stage: str, nice: int = 10) -> tuple[list[tuple[dict, dict]], int]:
    """[(planned run, spec)] of every run of `stage` in plan.json minus the cuts, and the number cut."""
    if not os.path.exists(plan_path):
        raise SystemExit(f"{plan_path} does not exist (analyze plan writes it at method-v1)")
    with open(plan_path) as f:
        plan = json.load(f)
    cuts_path = os.path.join(os.path.dirname(os.path.abspath(plan_path)), "cuts.json")
    if not os.path.exists(cuts_path):
        raise SystemExit(f"{cuts_path} is missing (analyze plan writes it as []); refusing to queue")
    with open(cuts_path) as f:
        cuts = json.load(f)
    if any("stage" not in c for c in cuts):               # a stem alone also names its 3M continuation
        raise SystemExit(f"{cuts_path}: every cut entry must name its stage; refusing to queue")
    cut = {f"{c['stage']}/{st}" for c in cuts for st in c["stems"]}      # = safe2.analyze.cut_ids
    mine, n_cut = [], 0
    for r in plan["runs"]:
        if r["stage"] != stage:
            continue
        spec = spec_from_cmd(trainer_cmd(r["argv"]), nice)
        if spec is None:
            raise SystemExit(f"plan.json: a run without --config, --env, --seed or --out: {r['argv']}")
        if not spec["out"].startswith(QDIR + os.sep):
            raise SystemExit(f"plan.json: {spec['stem']} writes outside {QDIR}")
        if r.get("id", f"{stage}/{spec['stem']}") in cut:
            n_cut += 1
        else:
            mine.append((r, spec))
    return mine, n_cut


def heldout_done(stem: str) -> bool:
    """The run's 1M held-out evaluation exists (checked by name only; nothing under sealed/ is read)."""
    return os.path.exists(os.path.join(QDIR, "sealed", "heldout", f"{stem}.deployed.json.enc"))


def copy_ckpt(src: str, dst: str) -> None:
    """Copy a checkpoint atomically (tmp, fsync, rename): a crash never leaves a truncated file."""
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def source_state(src: str, stem: str) -> str | None:
    """Why the 1M run behind a continuation's checkpoint `src` must not be continued (None = it may):
    a continuation resumes that run, and a diverged or unfinished run is never resumed (SPEC 3.6)."""
    out = os.path.dirname(src)
    if first_zero(health_rows(out, stem)) is not None:
        return "diverged"
    if not os.path.exists(os.path.join(out, stem + ".pt")):
        return "unfinished"
    if not os.path.exists(src):
        return "checkpoint missing"
    return None


def unqueue_cut() -> None:
    """Drop queued runs that are now cut and list live ones (the lead stops those by hand)."""
    cut = cut_ids()
    if not cut:
        return
    with Locked():
        queue = read_queue()
        write_queue([q for q in queue if _name(q) not in cut])
    for q in queue:
        if _name(q) in cut:
            print(f"unqueued {_name(q)}: cut")
    for ln in live_trainers():
        spec = spec_from_cmd(ln.split())
        if spec and _name(spec) in cut:
            print(f"running  {_name(spec)}: cut (stop it by hand)")


def queue_plan(args) -> None:
    planned, n_cut = load_plan(args.from_plan, args.stage, args.nice)
    if not planned and not n_cut:
        raise SystemExit(f"plan.json has no run in stage {args.stage!r}")
    if args.dry_run:
        for _, s in planned:
            print(" ".join(shlex.quote(c) for c in s["cmd"]))
        print(f"# {len(planned)} runs of stage {args.stage}; {n_cut} cut")
        return
    unqueue_cut()
    live, specs, notes = live_trainers(), [], []
    for r, spec in planned:
        src = r.get("resume_from")
        if src and not resumable(spec) and not finished(spec) and not is_live(spec, live):
            why = source_state(_resolve(src), spec["stem"])
            if why:
                notes.append(f"skip {_name(spec)}: source {why}")
                continue
            if not heldout_done(spec["stem"]):
                notes.append(f"skip {_name(spec)}: its 1M held-out evaluation is not written yet")
                continue
            os.makedirs(spec["out"], exist_ok=True)
            copy_ckpt(_resolve(src), _path(spec, ".ckpt.pt"))
            notes.append(f"copied {_resolve(src)} -> {_path(spec, '.ckpt.pt')}")
        why = why_skip(spec, live)
        if why:
            notes.append(f"skip {_name(spec)}: {why}")
        else:
            specs.append(spec)
    for n in notes:
        print(n)
    added, cap = enqueue(specs, args.front, args.max_procs, args.set_cap)
    print(f"queued {added} runs of stage {args.stage} ({n_cut} cut, {len(planned) - len(specs)} skipped, "
          f"{len(specs) - added} already queued); effective cap {cap}")


# --- retirement (bug reruns) -----------------------------------------------------------------------
HELDOUT_NAME = re.compile(r"^\.[a-z0-9_]+(?:\.n(\d+))?\.(?:json\.enc|sha)(?:\.tmp)?$")   # after "<stem>"


def retired_files(stage: str, stem: str) -> list[str]:
    """Every file of run <stage>/<stem> under QDIR: its run files, its sealed record file, its held-out
    outputs (the 3M ones for cont3m/, the others otherwise) and, outside cont3m/, its PHS fit and quality."""
    def own(d: str) -> list[str]:
        return sorted(glob.glob(os.path.join(QDIR, d, glob.escape(stem) + ".*")))
    files = own(stage) + own(os.path.join("sealed", stage))
    for path in own(os.path.join("sealed", "heldout")):
        m = HELDOUT_NAME.match(os.path.basename(path)[len(stem):])
        n = m and m.group(1)
        if m and (n == str(CONTINUATION_STEPS) if stage == "cont3m" else n is None):
            files.append(path)
    if stage != "cont3m":
        files += own("phs") + own(os.path.join("sealed", "phs"))
    return files


def retire(names: list[str]) -> None:
    """Move each run <stage>/<stem> that is not alive to QDIR/retired/<UTC time>/ (same relative paths)
    and remove it from its launch.json and the queue (module docstring)."""
    dest = os.path.join(QDIR, "retired", time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    alive = {(s["out"], s["stem"]) for s in (spec_from_cmd(ln.split()) for ln in live_trainers()) if s}
    with Locked():
        for name in names:
            stage, _, stem = name.partition("/")
            out = os.path.join(QDIR, stage)
            if not stage or not stem or "/" in stem or stage in ("sealed", "retired") or ".." in name:
                print(f"refused {name}: give STAGE/STEM, e.g. conf/B5_SafetyPointGoal1-v0_s300")
                continue
            if (out, stem) in alive:
                print(f"refused {name}: it is running (stop it first)")
                continue
            files = retired_files(stage, stem)
            if not files:
                print(f"refused {name}: nothing to retire")
                continue
            for path in files:
                target = os.path.join(dest, os.path.relpath(path, QDIR))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.replace(path, target)
            manifest = os.path.join(out, "launch.json")
            if os.path.exists(manifest):
                try:
                    entries = json.load(open(manifest))
                except ValueError:
                    entries = None
                if entries is not None:
                    keep = [e for e in entries if (spec_from_cmd(e.get("cmd", [])) or {}).get("stem") != stem]
                    with open(manifest + ".tmp", "w") as f:
                        json.dump(keep, f, indent=1)
                    os.replace(manifest + ".tmp", manifest)
            write_queue([q for q in read_queue() if (q["out"], q["stem"]) != (out, stem)])
            print(f"retired {name}: {len(files)} files -> {dest} (log it in protocol section 22)")


# --- CLI --------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", help="a stage directory under results/method/")
    p.add_argument("--configs", nargs="+", default=[], help="registry names (safe2.configs), or BVSU / BVSD")
    p.add_argument("--env", nargs="+", default=["SafetyPointGoal1-v0"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--drop-ckpt", type=int, choices=[0, 1], default=0)
    p.add_argument("--from-bvs", default=None, help="BVSU / BVSD: BVS* (passed to those configs only)")
    p.add_argument("--rho-star", default=None, help="BVSU: rho*, passed as written (to BVSU only)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--nice", type=int, default=10)
    p.add_argument("--dry-run", action="store_true", help="print the trainer commands only")
    p.add_argument("--max-procs", type=int, default=0,
                   help="write N to results/method/.queue.cap when it does not exist yet (12 by default)")
    p.add_argument("--front", action="store_true", help="queue: put these runs first")
    p.add_argument("--set-cap", action="store_true", help="with --max-procs: overwrite results/method/.queue.cap")
    p.add_argument("--from-plan", default=None, metavar="PLAN", help="queue the planned runs of --stage")
    p.add_argument("--stage", default=None, help="with --from-plan: p1, p2, conf or cont3m")
    p.add_argument("--scheduler", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--recover", action="store_true", help="re-queue interrupted runs after a reboot")
    p.add_argument("--retire", nargs="+", metavar="STAGE/STEM", help="move runs out of the study (bug reruns)")
    args = p.parse_args(argv)
    if args.scheduler:
        run_scheduler()
        return
    if args.recover:
        recover()
        return
    if args.retire:
        retire(args.retire)
        return
    if args.from_plan:
        if not args.stage:
            p.error("--from-plan needs --stage")
        queue_plan(args)
        return
    if not args.out or not args.configs:
        p.error("--out and --configs are required")
    args.out = os.path.abspath(args.out)          # children run with cwd=ROOT; keep every path absolute
    if not args.out.startswith(QDIR + os.sep):    # managed() only sees runs there: freeze, recovery, de-duplication
        p.error("--out must be under results/method/")
    unknown = [c for c in args.configs if c not in configs.CONFIGS and c not in configs.DERIVED]
    if unknown:
        p.error(f"unknown configs {unknown}")
    if any(c in configs.DERIVED for c in args.configs) and not args.from_bvs:
        p.error("BVSU and BVSD need --from-bvs (and BVSU --rho-star)")
    if "BVSU" in args.configs and args.rho_star is None:
        p.error("BVSU needs --rho-star")
    specs = build_specs(args)
    if args.dry_run:
        for s in specs:
            print(" ".join(shlex.quote(c) for c in s["cmd"]))
        return
    live, ok = live_trainers(), []
    for s in specs:
        why = why_skip(s, live)
        if why:
            print(f"skip {s['stem']}: {why}")
        else:
            ok.append(s)
    os.makedirs(args.out, exist_ok=True)
    added, cap = enqueue(ok, args.front, args.max_procs, args.set_cap)     # always queued: the cap holds
    print(f"queued {added} runs ({len(specs) - added} skipped, duplicate or already queued); effective cap {cap}")


if __name__ == "__main__":
    main()
