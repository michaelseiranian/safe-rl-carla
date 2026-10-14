# The method study (`safe2/`): how to run every stage

The study is pre-registered in `prereg_method.md` and specified in `safe2/SPEC.md`. This file only
says which command runs which stage. Where it disagrees with either of them, they win. It is hashed at
`method-v0` with every other file of `safe2/`, so an edit after v0 is a protocol section 22 deviation.

Every command runs from the worktree root with the study interpreter:

```bash
cd /Users/michaelseiranian/Desktop/safe-rl-carla-method
PY=/Users/michaelseiranian/Desktop/safe-rl-carla/.venv/bin/python      # Python 3.10, safety_gymnasium 1.0.0
```

Placeholders in angle brackets (`<B*>`, `<BVS*>`, `<rho*>`, ...) are the values recorded at
`method-v0.5` (`results/method/v05.json`) and `method-v1` (`results/method/v1.json`).

## Blinding: what you may look at before unblinding

- For seeds 300-315 and for anything on the held-out block 70000-70199, read only run health:
  `safe2/health.sh`, `safe2/pause.sh status` and `results/method/.queue.log`.
- Do not open `results/method/sealed/`, do not `tail` the logs or CSVs of seeds 300 and up, and do not
  use `lagu/status.sh` on `results/method/`.
- Seeds 0-5 (P1, P2) are selection data. Their public CSVs hold every outcome column on purpose.
- Steps/s and wall time of a BVS-family run weakly track how often its shield screens candidates. This
  side channel is disclosed (protocol section 10); never compare them across seeds 300 and up.

## The queue

| Command | What it does |
|---|---|
| `$PY -m safe2.launch --out results/method/<stage> --configs C... [--env E...] --seeds S... --drop-ckpt 0\|1 [--max-procs 12] [--front]` | queue runs (always through the queue); `--max-procs` writes the cap file if there is none; `--front` puts them first |
| `$PY -m safe2.launch --from-plan results/method/plan.json --stage conf\|cont3m [--front]` | queue every planned run of a stage, minus `cuts.json` |
| `$PY -m safe2.launch ... --dry-run` | print the trainer commands only |
| `safe2/pause.sh status` | power, policy, scheduler, cap, queue length, runs training and frozen |
| `safe2/pause.sh pause` / `resume` | freeze every method run now / unfreeze them and recover after a reboot |
| `safe2/pause.sh power auto\|always` | train only on mains power (the default) / on battery too |
| `safe2/pause.sh cap N` | at most N study trainers at once |
| `safe2/health.sh [DIR ...]` | per run: state, last step, steps/s, resumes, finite flags (never an outcome) |
| `$PY -m safe2.launch --recover` | after a reboot or a crash: resume interrupted runs from their checkpoints |
| `$PY -m safe2.launch --retire <stage>/<stem> ...` | bug reruns: move a run (files, sealed records, held-out outputs) to `results/method/retired/` |

How the queue behaves:

- **Shared slots.** The cap (12 by default, in `results/method/.queue.cap`) counts the method study's
  `safe2.train` runs (frozen ones too) and reserves the Lag-U study's slots: its live `lagu.train`
  runs, or, while the Lag-U queue (`<checkout>/results/.queue.jsonl`) holds work, as many as Lag-U's own
  cap lets its scheduler refill. The Lag-U scheduler does not count `safe2` runs, so to share the
  machine set Lag-U's cap: `lagu/pause.sh cap 4` leaves 8 slots to the method study. Frozen Lag-U runs
  (`lagu/pause.sh pause`) keep their slots until `lagu/pause.sh cap 0`; note that `lagu/pause.sh
  resume` puts a cap of 0 back to 12.
- **Power.** On battery, every method run is frozen in place (SIGSTOP; nothing is lost) and unfrozen
  when mains power returns. A closed lid sleeps the Mac and training simply waits.
- **What the launcher refuses.** It never starts or resumes, when queueing or at start time:
  - a run cut in `cuts.json` (queued cut runs are dropped; live ones are listed for you to stop);
  - a command the trainer would refuse (below);
  - a run whose public CSV shows a finite flag of 0 (diverged);
  - a run whose json records 3 resumes (unfinished);
  - a 3M continuation without its checkpoint, or whose 1M run is diverged or unfinished.
  A finished run is final, also with `--resume`. A run that has already started is not queued again
  as a fresh start; `--recover` resumes it (or restarts it with `--restart` if it crashed before its
  first checkpoint).
- **What the trainer refuses** under `results/method/`:
  - `--steps` other than 1,000,000 outside `cont3m/`; in `cont3m/` anything but `--steps 3000000
    --resume` on PointGoal1 seeds 300-304;
  - `--hidden` other than 256, and any dev-only flag;
  - a seed that is not a study training seed;
  - a fresh start of a run that started before (without `--restart`), and a resume from another
    run's checkpoint.
  Anywhere, a dev seed (900-999) above 50,000 steps: dev seeds are not blinded.
- **Per-run files** are in the stage directory: `<stem>.log` (console), `<stem>.csv` (public rows)
  and `<stem>.json`, plus `launch.json` for the whole directory. The stem is `<config>_<env>_s<seed>`.
  Crashes are in `results/method/.queue.log`.

## Before day 0: tests, key and tag `method-v0`

```bash
$PY -m pytest safe2/tests -q                                   # every test must pass at method-v0
SAFE2_SMOKE=1 SAFE2_SMOKE_OUT=<scratch dir> $PY -m pytest safe2/tests/test_smoke.py \
    safe2/tests/test_bvs_smoke.py safe2/tests/test_probe_smoke.py                     # opt-in long runs (20k-30k steps)
$PY -m safe2.power --calibrate --reps 8000                     # Appendix G; alpha_test

# the sealing key pair (SPEC section 4). The private key stays off the repository; you type its passphrase.
mkdir -p results/method/keys
/opt/homebrew/bin/openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:4096 -aes-256-cbc -out <private path>
/opt/homebrew/bin/openssl pkey -in <private path> -pubout -out results/method/keys/seal_pub.pem

# hash records (SPEC section 11): PREREG_SHA256.txt and FUNC_SHA256.json; then commit, tag and push
$PY -m safe2.analyze v0-hashes
git add -f results/method/PREREG_SHA256.txt results/method/FUNC_SHA256.json results/method/keys/seal_pub.pem
# commit; git tag method-v0; git push origin method method-v0; record the push time in protocol section 21
```

## Day 0-1: P1 baselines, SHL P1, S0-1 and S0-4

P1 runs PointGoal1 (the launcher's default `--env`) for seeds 0-2 at 1M steps. Queue priority is the
BVS path first, so B1 goes to the front: G0a and G0b need its final actors and replays.

```bash
# 13 baseline configs and R2 (42 runs). B1 keeps its checkpoint for G0a and G0b.
$PY -m safe2.launch --out results/method/p1 --configs B2 B3 B4 B5 B6 B7 B7b B7c B8 B9 SauteA SauteB R2 \
    --seeds 0 1 2 --drop-ckpt 1 --max-procs 12
$PY -m safe2.launch --out results/method/p1 --configs B1 --seeds 0 1 2 --drop-ckpt 0 --max-procs 12 --front
# SHL, 4 configs (12 runs). Checkpoints are kept for S0-4.
$PY -m safe2.launch --out results/method/p1 --configs SHL-g0.90-B4 SHL-g0.90-B5 SHL-g0.95-B4 SHL-g0.95-B5 \
    --seeds 0 1 2 --drop-ckpt 0 --max-procs 12

# Stage 0, which can run at once (outputs in results/method/stage0/)
M=/Users/michaelseiranian/Desktop/safe-rl-carla/results            # the main checkout, read-only
$PY -m safe2.probe s01
$PY -m safe2.probe s04 --nep 100 --runs \
    $M/pareto/td3lag-omni_SafetyPointGoal1-v0_s{0,1,2}.ckpt.pt \
    $M/pareto/td3lag-omni-fix03_SafetyPointGoal1-v0_s1.ckpt.pt \
    $M/pareto/td3lag-fix005_SafetyPointGoal1-v0_s0.ckpt.pt \
    $M/e0_dual/lagu_nogate-{omsw-l4e-7,tt-l2e-6,tt-l4e-7}_SafetyPointGoal1-v0_s0.ckpt.pt
```

After SHL P1 finishes, measure SHL's own share for kill rule 8.4:

```bash
$PY -m safe2.probe s04 --nep 100 --runs results/method/p1/SHL*_SafetyPointGoal1-v0_s{0,1,2}.ckpt.pt
# s04 merges runs by stem into results/method/stage0/s04.json (--out DIR to change); --force replaces runs already there
```

## Day 1-2: seeds 3-5, G0a and G0b, B*, then tag `method-v0.5`

```bash
$PY -m safe2.analyze p1-baselines         # R_ref(PointGoal1) and the top 3 (results/method/p1/top3.json)
$PY -m safe2.launch --out results/method/p1 --configs <top 3> --seeds 3 4 5 --drop-ckpt 1 --max-procs 12 --front

# once B1 seeds 0-2 are finished
$PY -m safe2.probe g0a                    # results/method/stage0/g0a.json
$PY -m safe2.probe g0b                    # results/method/stage0/g0b.json (fits and recall via safe2.posthoc)
# both refuse to overwrite their output; --force does, and appends a stub to deviations.json that you
# complete (the file and its section 22 entry text) before unblinding. g0b --reuse-fits reuses checked fits.

# once seeds 3-5 of the top 3 are finished
$PY -m safe2.analyze bstar                # B*, Saute*, L*, R_B*(P1), the G0a and G0b decisions -> v05.json
# commit v05.json, append its SHA-256 to PREREG_SHA256.txt, tag method-v0.5 and push it
```

After `method-v0.5`, start the baseline confirmatory runs and the baseline P2. Use the argv that
`analyze plan` will record: B* keeps its checkpoints (PHS, 3M continuations), and the other
baselines drop them. `--from-plan` later skips any planned run that has already started.

```bash
ENVS="SafetyPointGoal1-v0 SafetyCarGoal1-v0 SafetyPointCircle1-v0"
$PY -m safe2.launch --out results/method/conf --configs <B*> --env $ENVS --seeds $(seq 300 315) \
    --drop-ckpt 0 --max-procs 12
$PY -m safe2.launch --out results/method/conf --configs R1 <Saute*> <L*> --env $ENVS --seeds $(seq 300 307) \
    --drop-ckpt 1 --max-procs 12      # leave out Saute* if it is B*, and L* if it is B* or Saute*
$PY -m safe2.launch --out results/method/p2 --configs <B*> R2 --env SafetyCarGoal1-v0 SafetyPointCircle1-v0 \
    --seeds 0 1 2 --drop-ckpt 1 --max-procs 12
```

## Day 2-4: BVS P1, P2, the control health runs, then tag `method-v1`

Run these only if G0a and G0b both passed (`v05.json`). If BVS is killed at Stage 0, skip every BVS,
BVS-U, BVS-disc and PHS command below.

```bash
$PY -m safe2.launch --out results/method/p1 --configs BVS-SQ-k0.02 BVS-SQ-k0.1 BVS-SN-k0.02 BVS-SN-k0.1 \
    --seeds 0 1 2 --drop-ckpt 1 --max-procs 12 --front

# once the BVS and SHL P1 runs are finished: BVS*, SHL*, rho* and the P1 kill rules -> p1/candidates.json
$PY -m safe2.analyze candidates --stage p1
$PY -m safe2.launch --out results/method/p2 --configs <BVS*> <SHL*> --env SafetyCarGoal1-v0 SafetyPointCircle1-v0 \
    --seeds 0 1 2 --drop-ckpt 1 --max-procs 12 --front
$PY -m safe2.launch --out results/method/p1 --configs BVSU --from-bvs <BVS*> --rho-star <rho*> \
    --seeds 0 1 2 --drop-ckpt 1 --max-procs 12
$PY -m safe2.launch --out results/method/p1 --configs BVSD --from-bvs <BVS*> --seeds 0 1 2 --drop-ckpt 1 --max-procs 12

# once P2 is finished
$PY -m safe2.analyze candidates           # adds P2: R_ref (all tasks), kill rule (c), m_PHS -> v1.json
$PY -m safe2.analyze plan                 # plan.json and cuts.json = [] (--extra-continuations only after a kill)
# commit v1.json, plan.json and cuts.json, append their hashes to PREREG_SHA256.txt, tag method-v1 and push it
```

## Day 4-8: the confirmatory stage

```bash
$PY -m safe2.launch --from-plan results/method/plan.json --stage conf --max-procs 12
safe2/health.sh results/method/conf                       # as often as you like; never an outcome
```

`--from-plan` queues every planned run of the stage with the exact argv recorded in `plan.json`, minus
the runs in `cuts.json`; only a relative `--out` is rewritten, to the same directory as an absolute
path from the repository root (so the argv in a run's json differs from `plan.json` there alone). It skips runs that are already queued, running, started or finished. Run it
again after a cut or to pick up runs that were planned later.

Cuts are for throughput only, in the order of protocol section 19:

```bash
$PY -m safe2.analyze cut <ITEM>           # appends to cuts.json; commit and push it before any planned run is skipped
```

## Day 8-9: held-out evaluation, PHS, R3 and the 3M continuations

Every held-out output is sealed to `results/method/sealed/heldout/<stem>.<mode>.json.enc`. The console
prints only `<stem>: done`, `<stem>: skipped (sealed output exists)` or `<stem>: ERROR <type>`.

```bash
# the endpoint: 10 snapshots x 20 episodes per run (the modes and runs are listed in plan.json)
$PY -m safe2.evaluate results/method/conf --mode deployed --workers 12
$PY -m safe2.evaluate results/method/conf --mode shield_off --workers 12     # secondary BVS modes
$PY -m safe2.evaluate results/method/conf --mode strict --workers 12
$PY -m safe2.evaluate results/method/conf --mode s_near --workers 12        # only if BVS* uses S-Q

# PHS on each B* confirmatory run (replay episodes 1-900; quality on 901-1000 and the g0b layouts, sealed).
# plan.json lists the exact argv of every fit and quality check ("evaluations", kinds phs_fit and phs_quality).
$PY -m safe2.posthoc fit --ckpt results/method/conf/<stem>.ckpt.pt --episodes 1-900 \
    --out results/method/phs/<stem>.posthoc.pt --seed <seed>          # --seed defaults to the run seed
$PY -m safe2.probe phs-quality results/method/conf/<stem> [--root results/method] [--force]
# phs-quality checks the fit against the run (checkpoint SHA-256, episodes 1-900, 100k updates, seed),
# seals recall and ranking to sealed/phs/<stem>.quality.json.enc and prints only "<stem>: done".
# On PointCircle1 (no hazards) the ranking has no state: its Spearman is reported as not applicable.
$PY -m safe2.evaluate --phs results/method/conf --workers 12    # B* runs only; needs phs/<stem>.posthoc.pt and v1.json

# R3: the Lag-U study's lagu-lit seeds 100-109 (only the lagu-lit_* stems of the directory), on the held-out block
$PY -m safe2.evaluate --r3 /Users/michaelseiranian/Desktop/safe-rl-carla/results/gate_literal --seeds 100-109

# 3M continuations (PointGoal1, seeds 300-302 of B*, BVS*, SHL*). Each planned run with "resume_from" gets its
# 1M checkpoint copied into results/method/cont3m/ first, and only once its 1M run is finished, not diverged,
# and its 1M held-out evaluation exists. Their outputs are <stem>.deployed.n3000000.json.enc.
$PY -m safe2.launch --from-plan results/method/plan.json --stage cont3m
$PY -m safe2.evaluate results/method/cont3m --mode deployed --workers 12     # N = 3M (the runs' --steps)
```

The secondary modes skip the runs they do not apply to (`skipped (no strict mode)` for baselines, and
for S-N configs under `s_near`). Under `results/method/` the evaluator accepts only N = 1M, and 3M in
`cont3m/` (`ERROR EndStepRefused`). A sealed output whose `.sha` side file is missing reports
`ERROR ShaMissing`; re-run that run with `--force`.

## Daily, and at the end

```bash
$PY -m safe2.seal verify                  # every sealed record and file against HASHLOG.tsv; exit 1 on a mismatch
git add -f results/method/HASHLOG.tsv     # commit and push at least daily; record each push in protocol section 21

# day 10 (no later than day 12): only after every planned run, minus the cuts, is finished, diverged or
# unfinished(3 resumes), and every finished run has its held-out files
$PY -m safe2.analyze --unblind --key <private path>
# a re-run writes analysis/unblind.<n> and never replaces the first output (labelled a re-run if the code
# or deviations.json changed). deviations.json must be committed and pushed, each entry quoting section 22.

# at the analysis deadline (day 12) if planned runs are still queued, running or missing: the primary is
# computed anyway, with those runs (and finished runs without their deployed output) imputed as unfinished
$PY -m safe2.analyze --unblind --deadline --key <private path>
```

## Crashes, reboots and reruns

- **After a reboot**, run `safe2/pause.sh resume`. It unfreezes the runs, re-queues every interrupted
  run at the front and restarts the scheduler. It prints `resume`, `restart` (crashed before its
  first checkpoint), `diverged` or `unfinished` for each run.
- **A crashed run** resumes from its last checkpoint (every 100k) with `resumed = 1`, or restarts from
  scratch (`--restart`, also counted) when it crashed before its first checkpoint. It is resumed at
  most 3 times, then it is `unfinished(3 resumes)` and imputed. Every resume counts, also when a run
  crashes again before its next checkpoint. A 3M continuation is not a resume: its count starts at 0.
- **A diverged run** (a finite flag of 0) runs to its end and is never resumed.
- **Bug reruns** (protocol section 16) rerun every run of the affected arm at the affected stage.
  Stop the runs, then `$PY -m safe2.launch --retire <stage>/<stem> ...` for each: it moves the run
  files, the sealed record file and the held-out outputs to `results/method/retired/<UTC time>/`
  (outside `sealed/`, so `seal verify` stays clean; their hashes stay in HASHLOG.tsv) and removes the
  run from `launch.json` and the queue. Log each retirement in protocol section 22, then queue the
  reruns as usual. Killing a run is not enough to retire it: `--recover` would bring it back.

## After `method-v0`

Every command above exists and matches its `--help` (2026-10-14). Any later change to a file of
`safe2/`, this one included, or to sections 0-20 and the appendices of the protocol is a dated
protocol section 22 deviation, listed in `results/method/deviations.json` before `--unblind`.
