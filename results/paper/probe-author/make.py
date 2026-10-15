"""DESCRIPTIVE plumbing tests and run-cost measurement of probe.py (the gradient-parity probe of
prereg.md section 9).

    cd /Users/michaelseiranian/Desktop/safe-rl-carla && .venv/bin/python results/paper/probe-author/make.py

Reads NO study file. It (1) runs probe.py --dry-run (random untrained agents, no environment);
(2) trains two scratch models from scratch (hidden 256, the study's width; 20k steps; seed 7) into
the scratchpad if absent; (3) runs --stems --quick on them, then with a missing stem (exit status 2
expected), then --summarise-only from the cache (exit 0); (4) runs the --prereg code path
(declared_jobs, trajectory jobs, the co-primary and descriptive tables, the K_sens section, both
figures) with SETS pointed at the scratch models, checks cache reuse and that a job written by
another probe.py version is not reused; (5) measures the run cost on the scratch models through
the real probe_checkpoint at the full PROTOCOL constants (3 concurrent processes, then one process
clamped to the lower-tier cores) and extrapolates it to the declared --prereg run. Every output
goes to the scratchpad, never under results/. No number printed here says anything about the study.
At most 3 processes run at once.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time

REPO = "/Users/michaelseiranian/Desktop/safe-rl-carla"
HERE = os.path.join(REPO, "results/paper/probe-author")
SCR = ("/private/tmp/claude-501/-Users-michaelseiranian-Desktop-safe-rl-carla/"
       "47ca4e66-e2f4-4455-bfea-c56f9a36fb6d/scratchpad/probe-fix")
PY = os.path.join(REPO, ".venv/bin/python")
ENVV = {**os.environ, "PYTHONPATH": REPO}
RUN = os.path.join(SCR, "run")
STEP = 20000
STEMS = {arm: os.path.join(RUN, f"{arm}-scr_SafetyPointGoal1-v0_s7") for arm in ("lagu_nogate", "td3lag")}


def sh(args: list[str], want: int = 0) -> str:
    r = subprocess.run(args, cwd=REPO, env=ENVV, capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "AdroitHand" not in l)
    if r.returncode != want:
        raise SystemExit(f"FAILED (exit {r.returncode}, wanted {want}): {' '.join(args)}\n{out}")
    return out


def load_probe():
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)                  # spawned workers import probe.py by module name
    spec = importlib.util.spec_from_file_location("probe", os.path.join(HERE, "probe.py"))
    probe = importlib.util.module_from_spec(spec)
    sys.modules["probe"] = probe              # spawn workers re-import the module by this name
    spec.loader.exec_module(probe)
    return probe


def time_worker(arm: str, grid: str) -> None:
    """One checkpoint through probe.probe_checkpoint at the full PROTOCOL constants, with the grid cut
    to `grid`. Eligibility is forced so that every listed lambda is played at both K, or, when `grid` is
    lambda = 0 alone, ineligibility is forced (lambda = 0 is then the whole job, as for an ineligible
    checkpoint). Prints JSON."""
    import torch
    import safety_gymnasium
    probe = load_probe()
    torch.set_num_threads(1)
    acc = {"roll": 0.0, "parity": 0.0, "eval": 0.0, "n_eval": 0, "upd": 0.0, "n_steps": 0}

    def timed(name, f, count=None):
        def w(*a, **k):
            t0 = time.time()
            out = f(*a, **k)
            acc[name] += time.time() - t0
            if count:
                acc[count] += count_of(name, a, k)
            return out
        return w

    count_of = lambda name, a, k: 1 if name == "eval" else max(a[5])
    probe.collect_states = timed("roll", probe.collect_states)
    probe.parity = timed("parity", probe.parity)
    probe.play = timed("eval", probe.play, "n_eval")
    probe.frozen_lambda_actors = timed("upd", probe.frozen_lambda_actors, "n_steps")
    lams = [float(x) for x in grid.split(",")]
    force = -1e9 if len(lams) > 1 else 1e9
    P = {**probe.PROTOCOL, "grid": lams, "min_return": force, "min_frac_of_ckpt": force}
    agent, meta, _ = probe.load_agent(STEMS[arm], STEP)
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    t0 = time.time()
    res = probe.probe_checkpoint(agent, meta, env, P)
    acc["total"] = time.time() - t0
    acc.update(arm=arm, M=agent.cfg.M, hidden=agent.cfg.hidden, n_states=res["n_states"],
               played={K: len(b["grid"]) for K, b in res["byK"].items()})
    print("TIMING " + json.dumps(acc))


def main() -> None:
    os.chdir(REPO)
    probe = load_probe()
    print("== (1) probe.py --dry-run [DESCRIPTIVE plumbing check]")
    print(sh([PY, os.path.join(HERE, "probe.py"), "--dry-run"]))

    print("\n== (2) scratch models from scratch (hidden 256, 20k steps, seed 7; output in the scratchpad)")
    os.makedirs(RUN, exist_ok=True)
    procs = []
    for arm, stem in STEMS.items():
        if not os.path.exists(stem + ".pt"):
            procs.append(subprocess.Popen(
                [PY, "-m", "lagu.train", "--env", "SafetyPointGoal1-v0", "--arm", arm, "--tag", "scr", "--seed", "7",
                 "--steps", str(STEP), "--eval-every", "10000", "--eval-episodes", "2", "--hidden", "256", "--out", RUN],
                cwd=REPO, env=ENVV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        assert p.wait() == 0
    print("  " + ", ".join(f"{os.path.basename(s)}: snapshots {sorted(os.listdir(s + '.snap'))}" for s in STEMS.values()))

    print("\n== (3) probe.py --stems ... --quick [QUICK PLUMBING TEST]")
    out = os.path.join(SCR, "quick_stems")
    stems = [STEMS["lagu_nogate"] + f"@{STEP}", STEMS["td3lag"] + f"@{STEP}", STEMS["td3lag"] + "@final"]
    text = sh([PY, os.path.join(HERE, "probe.py"), "--quick", "--workers", "2", "--out", out, "--stems", *stems])
    print(text)
    assert "Table X [EXPLORATORY (hand-picked" in text and "label NOT EVALUABLE [EXPLORATORY]" in text, "--stems labels"
    with open(os.path.join(out, "probe.json")) as f:
        js = json.load(f)
    snap, fin = (next(r for r in js["results"] if r["stem"] == STEMS["td3lag"] and str(r["step"]) == s)
                 for s in (str(STEP), "final"))
    assert snap["byK"] == fin["byK"] and snap["parity"] == fin["parity"], "snapshot and final .pt differ"
    print("  snapshot and final .pt of the same run give identical numbers at both K: OK")
    missing = os.path.join(RUN, "td3lag-scr_SafetyPointGoal1-v0_s8")
    text = sh([PY, os.path.join(HERE, "probe.py"), "--quick", "--workers", "2", "--out", out, "--stems", *stems,
               missing + f"@{STEP}"], want=2)
    assert "1 declared files missing" in text and "exit status 2" in text
    print("  with one missing stem: exit status 2, " + next(l.strip() for l in text.splitlines() if "_s8@" in l))
    text = sh([PY, os.path.join(HERE, "probe.py"), "--quick", "--out", out, "--summarise-only", "--stems", *stems])
    assert "exit status 0" in text and "probing" not in text
    print("  --summarise-only from the cache: exit status 0, nothing re-probed")

    print("\n== (4) the --prereg code path on scratch stand-ins [QUICK PLUMBING TEST]")
    declared = list(probe.SETS)
    probe.SETS = [("S1", "co-primary", RUN, "lagu_nogate-scr", [7], STEP),
                  ("S2", "co-primary", RUN, "td3lag-scr", [7], STEP),
                  ("S3a", "descriptive", RUN, "td3lag-scr", [7, 8], STEP)]   # seed 8 does not exist: 'missing'
    P = {**probe.QUICK, "traj_every": 10000}
    out = os.path.join(SCR, "quick_prereg_path")
    os.makedirs(out, exist_ok=True)
    jobs = probe.declared_jobs("/", P)
    res = probe.run_all(jobs, P, out, 2, False)
    res2 = probe.run_all(jobs, P, out, 2, True)          # second pass must come entirely from the cache
    n_ok = sum("error" not in r for r in res)
    assert n_ok == sum("error" not in r for r in res2) == 9, "cache not reused"
    summary, text = probe.summarise(res2, P, probe.SETS, "QUICK PLUMBING TEST - NOT THE PRE-REGISTERED PROTOCOL")
    print(text)
    assert summary["overall"] == "NOT EVALUABLE" and "Table S3a [DESCRIPTIVE]" in text and "label NOT EVALUABLE [DESCRIPTIVE]" in text
    probe.figures(res2, probe.SETS, out, "QUICK PLUMBING TEST - NOT THE PRE-REGISTERED PROTOCOL", P)
    print(f"  figures: {sorted(f for f in os.listdir(out) if f.endswith(('.pdf', '.png')))} in {out}")
    jf = probe.job_file(out, jobs[0])
    with open(jf) as f:
        js = json.load(f)
    js["probe_sha256"] = "0" * 64                      # as if written by another version of probe.py
    with open(jf, "w") as f:
        json.dump(js, f)
    res3 = probe.run_all(jobs, P, out, 2, True)
    assert sum("error" not in r for r in res3) == n_ok - 1 and \
        next(r for r in res3 if r["stem"] == jobs[0]["stem"] and r["kind"] == "main")["error"].startswith("not run")
    probe.run_all(jobs, P, out, 1, False)               # re-probes that one job, restoring the cache
    print("  a job file written by another probe.py version is not reused: OK")

    print("\n== (5) run cost of the real --prereg run [DESCRIPTIVE measurement and projection]")
    print("  (a) 3 concurrent processes (the --workers 3 condition), grid cut to lambda = 0, 0.1, eligibility forced")
    arms = ["lagu_nogate", "td3lag", "lagu_nogate"]
    ps = [subprocess.Popen([PY, __file__, "--time-worker", arm, "0,0.1"], cwd=REPO, env=ENVV, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True) for arm in arms]
    outs = [p.communicate() for p in ps]
    assert all(p.returncode == 0 and "TIMING " in o for p, (o, _) in zip(ps, outs)), [e[-2000:] for _, e in outs]
    conc = [json.loads(o.split("TIMING ", 1)[1]) for o, _ in outs]
    print("    " + "\n    ".join(json.dumps(t) for t in conc), flush=True)
    print("  (b) 1 process clamped to the lower-tier cores (taskpolicy -c background), grid cut to lambda = 0, "
          "ineligibility forced", flush=True)
    p = subprocess.run(["taskpolicy", "-c", "background", PY, __file__, "--time-worker", "lagu_nogate", "0"],
                       cwd=REPO, env=ENVV, capture_output=True, text=True)
    assert p.returncode == 0 and "TIMING " in p.stdout, p.stderr[-2000:]
    slow = json.loads(p.stdout.split("TIMING ", 1)[1])
    print("    " + json.dumps(slow), flush=True)
    PR = probe.PROTOCOL
    r, G, Kmax = len(PR["rep_seeds"]), len(PR["grid"]), max(PR["K"], PR["K_sens"])
    n_ev_pt = r * len({PR["K"], PR["K_sens"]})          # evaluations per grid point (both K still playing)
    for t in conc + [slow]:
        t["t_eval"], t["t_step"] = t["eval"] / t["n_eval"], t["upd"] / t["n_steps"]
        t["t_point"] = r * Kmax * t["t_step"] + n_ev_pt * t["t_eval"]
        t["model"] = t["roll"] + t["parity"] + t["t_eval"] + (1 + max(t["played"].values())) * t["t_point"]
        print(f"    {t['arm']:<12} M={t['M']} hidden={t['hidden']}: rollout {t['roll']:.1f} s ({t['n_states']} states), "
              f"parity {t['parity']:.1f} s, 20-episode evaluation {t['t_eval']:.1f} s (n={t['n_eval']}), actor update "
              f"{t['t_step'] * 1e3:.2f} ms -> one grid point (r={r} x {Kmax:,} updates + {n_ev_pt} evaluations) "
              f"{t['t_point']:.0f} s; job {t['total']:.0f} s measured vs {t['model']:.0f} s from the components")
    by_M = {M: [t for t in conc if t["M"] == M] for M in (3, 1)}
    avg = lambda M, k: sum(t[k] for t in by_M[M]) / len(by_M[M])
    fixed = {M: avg(M, "roll") + avg(M, "parity") + avg(M, "t_eval") for M in by_M}
    upper = {M: fixed[M] + G * avg(M, "t_point") for M in by_M}       # eligible at both K, full grid, no stop
    lower = {M: fixed[M] + avg(M, "t_point") for M in by_M}           # ineligible at both K: lambda = 0 only
    traj = {M: avg(M, "roll") + avg(M, "parity") for M in by_M}
    M_of = {"lagu_nogate": 3, "td3lag": 1}
    main_jobs = [M_of[arm.split("-")[0]] for _, _, _, arm, seeds, _ in declared for _ in seeds]
    traj_jobs = [M_of[arm.split("-")[0]] for _, _, _, arm, seeds, step in declared for _ in seeds
                 for _ in range(PR["traj_every"], step + 1, PR["traj_every"])]

    def makespan(durs, speeds):        # imap_unordered, chunksize 1: each free worker takes the next job in order
        free = [0.0] * len(speeds)
        for d in durs:
            i = min(range(len(free)), key=free.__getitem__)
            free[i] += d * speeds[i]
        return max(free)

    s_slow = slow["t_point"] / avg(3, "t_point")
    print(f"  lower-tier core / concurrent-run speed ratio (time per grid point, M=3): {s_slow:.2f}")
    for name, per in (("upper bound (every checkpoint eligible at both K, full 22-point grid, no early stop)", upper),
                      ("lower bound (every checkpoint ineligible at both K: lambda = 0 only)", lower)):
        durs = [per[M] for M in main_jobs] + [traj[M] for M in traj_jobs]
        cpu = sum(durs) / 3600
        w3 = makespan(durs, [1.0] * 3) / 3600
        w6 = makespan(durs, [1.0] * 5 + [s_slow]) / 3600
        w6s = makespan(durs, [s_slow] * 6) / 3600
        print(f"  {name}: {len(main_jobs)} checkpoints + {len(traj_jobs)} trajectory snapshots = {cpu:.1f} CPU-hours "
              f"at the 3-concurrent speed; wall {w3:.1f} h at --workers 3; at --workers 6 {w6:.1f} h (5 workers at "
              f"that speed, 1 on a lower-tier core; not measured at 6), {w6s:.1f} h if all 6 ran at lower-tier speed")
    print(f"  per checkpoint: 1 rollout (S) + 1 evaluation (R_ckpt) + per grid point r x (K updates, K_sens nested) + "
          f"r x 2 evaluations (K and K_sens), i.e. at most 1 + {G} x {n_ev_pt} = {1 + G * n_ev_pt} evaluations and "
          f"{G * r * Kmax:,} actor updates")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--time-worker":
        time_worker(sys.argv[2], sys.argv[3])
    else:
        main()
