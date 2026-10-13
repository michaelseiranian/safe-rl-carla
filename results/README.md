# Results data

Every number in `prereg.md` (sections 15-16), `prereg_results.json` and the paper draft can be
recomputed from the files in this directory. Model snapshots and checkpoints (`*.pt`, `*.snap/`,
about 35 GB) and replay-derived caches are not included; they are available on request.

## Layout

| Directory | Contents |
|---|---|
| `pilot/`, `lr_sweep/`, `lr_sweep2/`, `pareto/` | exploratory runs that shaped the design (before the pre-registration) |
| `smoke/` | G0 smoke runs (not pre-registered) |
| `certify/` | W0 bit-identity certificate (H1): `certificate.md`, `certificate.json` |
| `e0_dual/` | E0 dual-step selection, seeds 0-2; `e0_decision.json` is the G1/G2 decision file |
| `gate/` | SafetyPointGoal1, 2M steps: E1 (`lagu-tt`, `lagu_nogate-tt`), E2 (`lagu_nogate-tt-match`), gate-gradient arm (`lagu-tt-ggrad`), `td3lag-tt`, E4 (`*-tt-d175`, `*-tt-d125`, `*-tt-T035`, `*-tt-T14`) |
| `gate_literal/` | E1b, the paper's literal recipe, 1M steps (`lagu-lit`, `lagu_nogate-lit`, `td3lag-lit`) |
| `gate_literal_x/` | X1, the gate at the paper's dose (`lagu-lit-T03`, `lagu_nogate-lit-match`) |
| `gate_x/` | E3, SafetyCarGoal1 and SafetyPointCircle1 |
| `g3/` | outputs of the G3, final confirmatory, X1 and E4 analyses |
| `paper/` | exploratory and descriptive analyses for the paper, each with a `make.py` that regenerates its tables and figures (PNG copies omitted; PDFs included), plus independent verification scripts (`*-verify/`) and `CORRECTIONS.md` |
| `probe/`, `probe_summary.txt` | gradient-parity probe output (section 16 item 14) |
| `PREREG_SHA256.txt` | the local hash record made before Wave 1 |

## Per-run files

Each run is named `<arm>_<Env>_s<seed>`:

- `.csv`: one row every 50k steps, with in-training evaluation (`eval_*`), interval means over the
  updates since the previous row (`*_avg`, `engagement`, `would_engage`, `n_pi`, `n_risk`),
  critic calibration (`qc_traj`, `mc_cost_traj`), cumulative cost and the `resumed` flag;
- `.json`: the run's arguments and full configuration, plus any resumes;
- `.heldout.json`: the primary endpoint, 5 snapshots x 20 held-out layouts, with each snapshot's
  SHA-256;
- `.log`: the training log.

`launch.json` in each directory records every launch command with its time.
