"""Selection, Stage-0 and kill decisions, the confirmatory plan, cuts and the unblinded analysis of the
method study (prereg_method.md sections 6-16 and 19; safe2/SPEC.md section 9).

    python -m safe2.analyze v0-hashes                       # PREREG_SHA256.txt and FUNC_SHA256.json (method-v0)
    python -m safe2.analyze p1-baselines                    # R_ref(PointGoal1) and the top 3 -> p1/top3.json
    python -m safe2.analyze bstar                           # B*, Saute*, L*, R_B*(P1), G0a, G0b -> v05.json
    python -m safe2.analyze candidates --stage p1           # BVS*, SHL*, rho*, P1 kill rules -> p1/candidates.json
    python -m safe2.analyze candidates                      # the same plus P2: R_ref, kill (c), m_PHS -> v1.json
    python -m safe2.analyze plan [--extra-continuations]    # plan.json and cuts.json = [] (method-v1)
    python -m safe2.analyze cut ITEM                        # a throughput cut (protocol section 19)
    python -m safe2.analyze --unblind --key PRIVATE.pem     # verification, then every table in one pass
    python -m safe2.analyze --unblind --deadline --key ...  # at the analysis deadline: incomplete runs imputed
    common options: --root DIR (default results/method), --overwrite (replace a decision record that differs)

Every subcommand writes analysis/<name>.txt and analysis/<name>.json under the root and refuses (exit 1,
printing why) when its inputs are incomplete. Every output section carries a label:
  CONFIRMATORY  the primary H_BVS (section 12), only from --unblind;
  DESCRIPTIVE   selection and gate decisions, and the pre-registered secondary, sensitivity, control,
                mechanism and other tables of section 14 (95% Welch intervals, no multiplicity correction);
  EXPLORATORY   BVS* - L* (14.1), SHL (14.5), S0-1, and anything else computed from seeds >= 300 or the
                held-out block outside --unblind.

Blinding. Before --unblind, the subcommands read only public files of selection seeds 0-5 (CSV rows at
800k-1M, run jsons, 1M snapshots for m_dep), stage0/ outputs and the decision records; they never open a
path under sealed/. --unblind is the only reader of sealed/ and of the private key. The key is unlocked
once by OpenSSL's own passphrase prompt (`openssl pkey`) and stays in memory: each decryption reaches
`openssl pkeyutl` through a pipe, so the unlocked key never touches disk.

Decision records (p1/top3.json, v05.json, p1/candidates.json, v1.json, plan.json) are write-once: a re-run
that would write different content refuses unless --overwrite. bstar and plan append the SHA-256 of the
records they write to PREREG_SHA256.txt (format `<sha256>  <path from the repository root>`, comment lines
start with '#'); --unblind reads that file as committed at tag method-v1.

File contracts read here and written by other tools (safe2/SPEC.md sections 5, 7 and 8):
  stage0/g0a.json        {"mean_return", "mean_cost", ...}                       (probe g0a)
  stage0/g0b.json        {"recall": {"pooled": {"recall", "fa_per_episode"}}, "ranking": {"spearman",
                         "states"}, "decision": {"proceed"}, ...}               (probe g0b)
  stage0/s04.json        {"runs": {<stem>: {"share": float, ...}, ...}, ...}   (probe s04)
  sealed/heldout/<name>.<mode>.json.enc   evaluate output; continuations (N != 1M) add ".n<N>" to <mode>
                         (heldout_rel); PHS uses mode "phs", R3 mode "r3"
  sealed/phs/<stem>.quality.json.enc     (probe phs-quality; reported as decrypted, no key is read)
  plan.json lists every one of these paths; --unblind checks exactly the paths the plan declares.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import glob
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict

import numpy as np

from lagu.analyze import _clean, fisher_exact, holm, iqm
from safe2 import configs, seal, seeds, stats, train

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(REPO, "results", "method")
NAN = float("nan")

PG1, CG1, PC1 = "SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyPointCircle1-v0"
TASKS = (PG1, CG1, PC1)
P2_TASKS = (CG1, PC1)
STEPS, CONT_STEPS = train.STUDY_STEPS, train.CONTINUATION_STEPS
SEL_STEPS = (800_000, 850_000, 900_000, 950_000, 1_000_000)     # selection window rows (section 6)
K82_FROM, K82_TO = 500_000, 1_000_000                           # kill rule 8.2 window
K82_ROWS = tuple(range(K82_FROM, K82_TO + 1, 50_000))           # 8.2 evaluation rows (11 per seed)
K82_INT, K82_F, K82_VIOL = 0.30, 3.0, 0.25                      # 8.2: intervention rate, mean F, violation share
KILL_C_COST = 30.0                                              # kill (c): mean cost above this while B*'s <= 25
K84_SHARE = 0.6                                                 # 8.4: quiet-state gradient-mass share
INV_MEAN, INV_P95 = 0.05, 0.25                                  # "approximately saturated" (section 14.4)
MATCHED_C = 0.20                                                # "at matched cost": 90% CI of dc in +-0.20 (section 13)
M_SAFE = 25.0                                                   # margin bound (Appendix A.6)
HOURS = {"omni": 3.0, "literal": 2.4, "bvs": 4.0, "bvs_u": 5.0, "bvs_disc": 4.0}   # section 19, per 1M run
INCOMPLETE = ("missing", "incomplete", "running")                # run states that block a decision
CONFIRMATORY, EXPLORATORY, DESCRIPTIVE = "CONFIRMATORY", "EXPLORATORY", "DESCRIPTIVE"
RERUN = "re-run after unblinding (not confirmatory)"
SELECTION = "selection (pre-registered decision rule)"
TAGS = ("method-v0", "method-v0.5", "method-v1")
V0_FILES = tuple(f"safe2/{m}.py" for m in ("stats", "analyze", "evaluate", "train", "base", "bvs", "seeds",
                                            "configs", "seal", "posthoc", "probe", "launch", "health")) + (
    "safe2/pause.sh", "safe2/health.sh", "prereg_method.md", "safe2/SPEC.md")
PROTOCOL = "prereg_method.md"
FROZEN_UNTIL, FROZEN_FROM = "## 21. Decision log", "## Appendix A."   # sections 21-22 change after v0
DEVIATIONS_HEAD = "## 22. Deviations"
RECORDS = ("v05.json", "v1.json", "plan.json")
RECORD_TAGS = {"p1/top3.json": "method-v0.5", "v05.json": "method-v0.5", "p1/candidates.json": "method-v1",
               "v1.json": "method-v1", "plan.json": "method-v1"}       # the tag that freezes each record
R3_DIR = "/Users/michaelseiranian/Desktop/safe-rl-carla/results/gate_literal"
R3_SEEDS = tuple(range(100, 110))
# Cut list of protocol section 19, in order. Item 4 (BVS-U / BVS-disc health runs on seeds 0-2) precedes
# method-v1, so it is a section 22 deviation, not a cuts.json entry.
CUT_ORDER = ("1", "2", "2b", "3", "5")
CUT_ITEMS = {"1": "3M continuations", "2": "SHL* confirmatory from 12 to 8 seeds per task",
             "2b": "SHL* confirmatory from 8 seeds to none", "3": "R1, Saute* and L* from 8 to 6 seeds per task",
             "5": "BVS-U and BVS-disc from 12 to 8 seeds"}
NOTE_COST_OBS = ("BVS, BVS-U, BVS-disc, Saute and PHS observe the realised in-episode cost C_t at deployment; "
                 "the other B-family configs, SHL, R1 and R2 do not.")
NOTE_LIMIT = "Constraint satisfaction is claimed only for task means <= 25."
NOTE_BVSU = ("BVS-U: under S-Q a triggered step can replace a V-allowed a_pi by an allowed candidate with a "
             "higher min_j Q_r, which BVS never does; under S-near it executes a_pi whenever V allows it.")
NOTE_PHS = ("BVS* - PHS is 'BVS vs a deployment-only shield on B*', not 'training with the shield': it also "
            "changes an online V for an offline one, the proposer, and the proposer's budget input.")


class Refusal(Exception):
    """A subcommand refuses to run; `problems` says why."""

    def __init__(self, problems):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


# --- small helpers --------------------------------------------------------------------------------
def _num(v) -> float:
    if v in ("", None):
        return NAN
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _sha(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _read_json(path: str):
    if not os.path.exists(path):
        raise Refusal(f"missing input {path}")
    with open(path) as f:
        return json.load(f)


def _dump(obj) -> str:
    return json.dumps(_clean(obj), indent=1) + "\n"


def _write_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_record(path: str, obj, overwrite: bool = False, root: str | None = None) -> str:
    """Write-once decision record: 'written', 'unchanged', or a refusal when different content exists.
    With `root`, a record whose tag (RECORD_TAGS) exists locally, or any record after UNBLINDED.txt, is
    never rewritten, even with --overwrite."""
    text = _dump(obj)
    if os.path.exists(path):
        with open(path) as f:
            old = f.read()
        if old == text:
            return "unchanged"
        if not overwrite:
            raise Refusal(f"{path} exists with different content (frozen decision record); pass --overwrite "
                          "only for a logged section 22 change")
        if root is not None:
            tag = RECORD_TAGS.get(os.path.relpath(path, root).replace(os.sep, "/"))
            if os.path.exists(os.path.join(root, "UNBLINDED.txt")):
                raise Refusal(f"{path}: UNBLINDED.txt exists, no decision record is rewritten")
            if tag and _git(_repo_of(root), "rev-parse", "--verify", "-q", f"refs/tags/{tag}", check=False).strip():
                raise Refusal(f"{path} is frozen by tag {tag}, which exists: not rewritten")
    _write_atomic(path, text)
    return "written"


def _repo_of(root: str) -> str:
    r = subprocess.run(["git", "-C", root, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return os.path.realpath(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else REPO


def _rel(path: str, repo: str) -> str:
    """Path from the repository root (as recorded in PREREG_SHA256.txt and plan.json argv)."""
    return os.path.relpath(os.path.realpath(path), repo)


def _git(repo: str, *args, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", repo] + list(args), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise Refusal(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def _git_show(repo: str, rev: str, rel: str) -> bytes | None:
    r = subprocess.run(["git", "-C", repo, "show", f"{rev}:{rel}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def hash_entries(text: str) -> dict:
    """{path: sha256} of a PREREG_SHA256.txt text; the last entry of a path wins."""
    out = {}
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            sha, _, path = line.partition("  ")
            out[path.strip()] = sha.strip()
    return out


def append_hashes(root: str, header: str, paths: list[str]) -> list[str]:
    """Append `# header` and the SHA-256 of each path to PREREG_SHA256.txt (only paths whose last recorded
    hash differs). Returns the paths appended."""
    hfile = os.path.join(root, "PREREG_SHA256.txt")
    if not os.path.exists(hfile):
        raise Refusal(f"{hfile} is missing: run `python -m safe2.analyze v0-hashes` at method-v0 first")
    repo = _repo_of(root)
    with open(hfile) as f:
        last = hash_entries(f.read())
    new = [(p, _sha(p)) for p in paths]
    new = [(p, h) for p, h in new if last.get(_rel(p, repo)) != h]
    if new:
        with open(hfile, "a") as f:
            f.write(f"# {header} {_utc()}\n" + "".join(f"{h}  {_rel(p, repo)}\n" for p, h in new))
    return [p for p, _ in new]


# --- public run files -----------------------------------------------------------------------------
def _read_csv(path: str, fields: list[str]) -> list[dict]:
    """Rows of a public CSV with the trainer's header (a torn last line is dropped); numbers parsed."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise Refusal(f"{path}: columns differ from the trainer's schema")
        rows = {}
        for r in reader:
            if None in r.values() or None in r or not (r.get("step") or "").isdigit():
                continue
            rows[int(r["step"])] = {k: (int(v) if k == "step" else _num(v)) for k, v in r.items()}
    return [rows[s] for s in sorted(rows)]


def pending_runs() -> set:
    """{(real out dir, stem)} of every trainer that is live (safe2.health.live_states) or queued
    (safe2.launch.read_queue): such a run is still in progress, whatever its files show."""
    from safe2 import health, launch
    keys = set(health.live_states()) | {(q["out"], q["stem"]) for q in launch.read_queue()
                                        if q.get("out") and q.get("stem")}
    return {(os.path.realpath(o), stem) for o, stem in keys}


def load_run(dirpath: str, config: str, env: str, seed: int, steps: int = STEPS, expect: dict | None = None,
             stage: str = "", pending: set | None = None) -> dict:
    """Public facts of one run (never sealed/): rows, finished, diverged, n_resumes and its state:
    running (live or queued: in `pending`, see pending_runs()) | finished | diverged (a finite flag of 0,
    stopped) | unfinished (3 resumes, not diverged, stopped) | incomplete | missing. `expect` (asdict of a
    RunConfig) is compared with the run's recorded config."""
    stem = train.stem_of(config, env, seed)
    live = pending is not None and (os.path.realpath(dirpath), stem) in pending
    base = os.path.join(dirpath, stem)
    run = {"id": f"{stage}/{stem}" if stage else stem, "stem": stem, "stage": stage, "config": config,
           "env": env, "seed": seed, "steps": steps, "dir": dirpath, "rows": [], "state": "missing",
           "finished": False, "diverged": False, "n_resumes": 0, "horizon": None, "eval_episodes": None}
    if not (os.path.exists(base + ".json") and os.path.exists(base + ".csv")):
        if live:
            run["state"] = "running"
        return run
    with open(base + ".json") as f:
        meta = json.load(f)
    a = meta.get("args", {})
    if a.get("config") != config or a.get("env") != env or int(a.get("seed", -1)) != seed:
        raise Refusal(f"{base}.json: args {a.get('config')}/{a.get('env')}/{a.get('seed')} do not match the stem")
    if expect is not None:
        diff = configs.config_diff(meta.get("config", {}), expect)
        if diff:
            raise Refusal(f"{stem}: recorded config differs from the registry: {sorted(diff)}")
    fields = train.HEALTH_FIELDS if train.blinded(seed) else train.HEALTH_FIELDS + train.OUTCOME_FIELDS
    rows = _read_csv(base + ".csv", fields)
    diverged = any(r[g] == 0 for r in rows for g in train.GROUPS)
    finished = os.path.exists(base + ".pt") and bool(rows) and rows[-1]["step"] >= steps
    n_res = int(meta.get("n_resumes", 0))
    state = ("running" if live else "finished" if finished else "diverged" if diverged else
             "unfinished" if n_res >= 3 else "incomplete")
    run.update(rows=rows, finished=finished, diverged=diverged, n_resumes=n_res, state=state,
               horizon=meta.get("horizon"), eval_episodes=meta.get("config", {}).get("eval_episodes"))
    return run


def require_complete(runs) -> None:
    """Refuse unless every run is finished, diverged or unfinished after 3 resumes (and not running)."""
    bad = [f"{r['id']}: {r['state']}" for r in runs if r["state"] in INCOMPLETE]
    if bad:
        raise Refusal(["inputs incomplete:"] + bad)


def _row(run: dict, step: int) -> dict | None:
    for r in run["rows"]:
        if r["step"] == step:
            return r
    return None


def selection_values(run: dict) -> tuple[float, float]:
    """A seed's selection R and C: means of eval_return and eval_cost over its rows at 800k-1M
    (non-finite when a row is missing, e.g. a run that stopped after diverging; imputed later)."""
    rows = [_row(run, s) for s in SEL_STEPS]
    if any(r is None for r in rows):
        return NAN, NAN
    return float(np.mean([r["eval_return"] for r in rows])), float(np.mean([r["eval_cost"] for r in rows]))


def hours(name: str) -> float:
    """Planned compute per 1M run (protocol section 19), the rank() tie-break after cost."""
    rc = configs.CONFIGS.get(name)
    kind = configs.DERIVED.get(name) or rc.kind
    return HOURS[kind] if kind != "base" else HOURS[rc.base]


def expected_config(name: str, bvs_star: str | None = None, rho_star: float = NAN) -> dict:
    if name in configs.DERIVED:
        return asdict(configs.from_bvs(bvs_star, configs.DERIVED[name], rho_star))
    return asdict(configs.make(name))


# --- selection ------------------------------------------------------------------------------------
def selection_inputs(entries: dict) -> list[dict]:
    """stats.rank inputs (before imputation) from {config: [runs by seed]}; diverged seeds are collapsed."""
    out = []
    for name, runs in entries.items():
        vals = [selection_values(r) for r in runs]
        out.append({"name": name, "seeds": [r["seed"] for r in runs], "R": np.array([v[0] for v in vals]),
                    "C": np.array([v[1] for v in vals]), "collapsed": np.array([r["diverged"] for r in runs]),
                    "hours": hours(name)})
    return out


def rank_group(entries: dict, r_ref: float, eligible_only: bool) -> dict:
    """Impute (stats.impute_selection over the configs ranked together), flag collapses against R_ref
    and rank. Returns the order, the imputed configs and a table."""
    imp = stats.impute_selection(selection_inputs(entries))
    for c in imp:
        c["collapsed"] = np.asarray(c["collapsed"], bool) | stats.collapsed(c["R"], r_ref)
    order = stats.rank(imp, eligible_only=eligible_only)
    table = {c["name"]: {"seeds": c["seeds"], "R": c["R"], "C": c["C"], "mean_R": float(np.mean(c["R"])),
                         "mean_C": float(np.mean(c["C"])), "score": stats.selection_score(c["R"], c["C"]),
                         "collapses": int(np.sum(c["collapsed"])), "eligible": bool(stats.eligible(c["collapsed"])),
                         "hours": c["hours"]} for c in imp}
    return {"order": order, "configs": {c["name"]: c for c in imp}, "table": table}


def mean_R_imputed(entries: dict) -> dict:
    """{config: mean over its seeds of the (imputed) selection R}, for R_ref."""
    return {c["name"]: float(np.mean(c["R"])) for c in stats.impute_selection(selection_inputs(entries))}


def _runs(dirpath: str, names, env: str, seed_list, stage: str) -> dict:
    pending = pending_runs()
    return {n: [load_run(dirpath, n, env, s, expect=expected_config(n), stage=stage, pending=pending)
                for s in seed_list] for n in names}


def p1_baselines(root: str) -> dict:
    """R_ref(PointGoal1) from R2 and the 13 baseline configs on seeds 0-2, and the top 3 of
    rank(13 configs, eligible_only=False)."""
    entries = _runs(os.path.join(root, "p1"), list(configs.BASELINE_POOL) + ["R2"], PG1, seeds.SELECTION_SEEDS,
                    "p1")
    require_complete([r for rs in entries.values() for r in rs])
    pool = {n: entries[n] for n in configs.BASELINE_POOL}
    by_cfg = {**mean_R_imputed(pool), **mean_R_imputed({"R2": entries["R2"]})}
    r_ref = max(by_cfg.values())
    if not r_ref > 0:
        raise Refusal(f"R_ref(PointGoal1) = {r_ref} is not positive")
    g = rank_group(pool, r_ref, eligible_only=False)
    return {"label": DESCRIPTIVE, "role": SELECTION, "task": PG1, "r_ref": {PG1: r_ref}, "r_ref_by_config": by_cfg,
            "order": g["order"], "top3": g["order"][:3], "table": g["table"]}


def _need(d: dict, keys, path: str) -> list:
    """Values at dotted keys of a json (e.g. "recall.pooled.recall"); refuses naming the missing ones."""
    out, miss = [], []
    for k in keys:
        v = d
        for part in k.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        (miss.append(k) if v is None else out.append(v))
    if miss:
        raise Refusal(f"{path}: missing keys {miss}")
    return out


def bstar(root: str) -> dict:
    """B*, Saute*, L*, R_B*(P1) and the G0a / G0b decisions (record for method-v0.5). The gate rules are
    safe2.probe's g0a_decision and g0b_decision, applied here to the numbers in stage0/."""
    from safe2 import probe
    top = _read_json(os.path.join(root, "p1", "top3.json"))
    again = p1_baselines(root)
    if again["top3"] != top["top3"] or again["r_ref"][PG1] != top["r_ref"][PG1]:
        raise Refusal("p1/top3.json differs from a recomputation of p1-baselines")
    r_ref = top["r_ref"][PG1]
    p1 = os.path.join(root, "p1")
    six = _runs(p1, top["top3"], PG1, seeds.SELECTION_SEEDS + seeds.EXTRA_SELECTION_SEEDS, "p1")
    sau = _runs(p1, ("SauteA", "SauteB"), PG1, seeds.SELECTION_SEEDS, "p1")
    lit = _runs(p1, ("B7", "B7b", "B7c", "SauteA", "SauteB"), PG1, seeds.SELECTION_SEEDS, "p1")
    require_complete([r for g in (six, sau, lit) for rs in g.values() for r in rs])
    gb = rank_group(six, r_ref, eligible_only=False)
    gs = rank_group(sau, r_ref, eligible_only=False)
    gl = rank_group(lit, r_ref, eligible_only=False)
    b_star, saute_star, l_star = gb["order"][0], gs["order"][0], gl["order"][0]
    r_bstar = float(np.mean(gb["configs"][b_star]["R"]))
    a = _read_json(os.path.join(root, "stage0", "g0a.json"))
    b = _read_json(os.path.join(root, "stage0", "g0b.json"))
    g0a = probe.g0a_decision(*[float(v) for v in _need(a, ("mean_return", "mean_cost"), "stage0/g0a.json")],
                             r_bstar)
    rec, fa, rho, n = _need(b, ("recall.pooled.recall", "recall.pooled.fa_per_episode", "ranking.spearman",
                                "ranking.states"), "stage0/g0b.json")
    g0b = dict(probe.g0b_decision({"recall": float(rec), "fa_per_episode": float(fa)}, int(n), _num(rho)),
               recall=float(rec), fa_per_episode=float(fa), spearman=_num(rho), n_states=int(n))
    if "decision" in b and bool(b["decision"].get("proceed")) != g0b["proceed"]:
        raise Refusal("stage0/g0b.json: probe's decision differs from the recomputed G0b decision")
    run_saute = configs.CONFIGS[b_star].saute_n == 0.0
    run_l = l_star != b_star and not (run_saute and l_star == saute_star)
    return {"label": DESCRIPTIVE, "role": SELECTION, "b_star": b_star, "b_star_base": configs.CONFIGS[b_star].base,
            "b_star_eligible": gb["table"][b_star]["eligible"], "saute_star": saute_star, "l_star": l_star,
            "run_saute_star": run_saute, "run_l_star": run_l, "r_ref": {PG1: r_ref}, "r_bstar_p1": r_bstar,
            "g0a": g0a, "g0b": g0b, "bvs_stage0_proceed": g0a["proceed"] and g0b["proceed"],
            "top3": top["top3"], "tables": {"p1_baselines": top["table"], "top3_six_seeds": gb["table"],
                                            "saute": gs["table"], "literal": gl["table"]},
            "orders": {"top3_six_seeds": gb["order"], "saute": gs["order"], "literal": gl["order"]}}


# --- candidates and kill rules --------------------------------------------------------------------
MIN_BVS_SEEDS = 2      # protocol section 6 (M7): fewer qualifying BVS* seeds kills BVS*


def bvs_seed_problem(r: dict) -> str | None:
    """Why a BVS* selection seed cannot enter 8.2 and rho* (protocol section 6, M7): its 500k and 1M rows
    and every evaluation row from 500k to 1M must exist, with finite 8.2 and rho* counters and finite
    evaluation violation shares. None when it qualifies."""
    a, b = _row(r, K82_FROM), _row(r, K82_TO)
    rows = [_row(r, s) for s in K82_ROWS]
    if a is None or b is None or any(x is None for x in rows):
        return f"{r['id']}: rows 500k-1M missing"
    vals = [x[c] for x in (a, b) for c in ("int_cum", "shield_steps_cum", "F_cum", "ep_cum")]
    vals += [b["rho_num"], b["rho_den"]] + [x["eval_violation"] for x in rows]
    if not all(math.isfinite(_num(v)) for v in vals):
        return f"{r['id']}: non-finite 8.2 or rho* inputs"
    return None


def kill_82(runs: list[dict]) -> dict:
    """Kill rule 8.2 for BVS*, pooled over its seeds and the window 500k-1M: intervention rate
    (sum of the int_cum increase / sum of the shield_steps_cum increase between the 500k and 1M rows),
    mean F (F_cum increase / ep_cum increase) and the in-training evaluation violation share over the
    rows at 500k, 550k, ..., 1M (each row weighted by its evaluation episodes). Pooled over the seeds
    that qualify (bvs_seed_problem); the others are listed, and fewer than 2 qualifying seeds kills."""
    excluded = [p for p in (bvs_seed_problem(r) for r in runs) if p]
    ok = [r for r in runs if bvs_seed_problem(r) is None]
    if len(ok) < MIN_BVS_SEEDS:
        return {"kill": True, "seeds_used": len(ok), "excluded": excluded,
                "reasons": [f"fewer than {MIN_BVS_SEEDS} seeds with complete finite 8.2 inputs"]}
    sums = dict(i=0.0, s=0.0, f=0.0, e=0.0, v=0.0, k=0.0)
    for r in ok:
        a, b = _row(r, K82_FROM), _row(r, K82_TO)
        vals = [b[c] - a[c] for c in ("int_cum", "shield_steps_cum", "F_cum", "ep_cum")]
        viol = [_row(r, s)["eval_violation"] for s in K82_ROWS]
        k = float(r["eval_episodes"] or seeds.TRAIN_EVAL_EPISODES)
        sums["i"] += vals[0]; sums["s"] += vals[1]; sums["f"] += vals[2]; sums["e"] += vals[3]
        sums["v"] += sum(v * k for v in viol); sums["k"] += k * len(viol)
    if sums["s"] <= 0 or sums["e"] <= 0:
        raise Refusal(["kill rule 8.2: no shielded steps or episodes in the window"])
    out = {"intervention_rate": sums["i"] / sums["s"], "mean_F": sums["f"] / sums["e"],
           "eval_violation_share": sums["v"] / sums["k"], "episodes_evaluated": int(sums["k"]),
           "seeds_used": len(ok), "excluded": excluded}
    out["reasons"] = [x for x, bad in (("intervention rate > 0.30", out["intervention_rate"] > K82_INT),
                                       ("mean F > 3", out["mean_F"] > K82_F),
                                       ("violation share > 0.25", out["eval_violation_share"] > K82_VIOL)) if bad]
    out["kill"] = bool(out["reasons"])
    return out


def kill_84(s04: dict, star: str) -> dict:
    """Kill rule 8.4 for SHL*: the quiet-state share of penalty action-gradient mass at 1M exceeds 0.6 in
    at least 2 of 3 seeds (stage0/s04.json)."""
    runs = s04.get("runs", {})
    shares = {}
    for s in seeds.SELECTION_SEEDS:
        stem = train.stem_of(star, PG1, s)
        if stem not in runs or "share" not in runs[stem]:
            raise Refusal(f"stage0/s04.json: no share for {stem}")
        shares[stem] = float(runs[stem]["share"])
    n = sum(v > K84_SHARE for v in shares.values())
    return {"shares": shares, "over_0.6": n, "kill": n >= 2}


def rho_star(runs: list[dict]) -> float | None:
    """rho* = sum of rho_num / sum of rho_den at the 1M rows of BVS*'s P1 seeds (steps 35,001-1M), pooled
    over the seeds that qualify (bvs_seed_problem; protocol section 6, M7). None (never a non-finite
    value) when fewer than 2 qualify: kill rule 8.2 then kills BVS*."""
    rows = [_row(r, STEPS) for r in runs if bvs_seed_problem(r) is None]
    if len(rows) < MIN_BVS_SEEDS:
        return None
    den = sum(x["rho_den"] for x in rows)
    if den <= 0:
        raise Refusal("rho*: no counted steps")
    return float(sum(x["rho_num"] for x in rows) / den)


def m_dep_at_1m(run: dict) -> float:
    """m_dep stored in the run's 1M snapshot (the newest resume's file when it was rewritten)."""
    import torch
    from safe2 import evaluate
    try:
        path = evaluate.snapshot_file(os.path.join(run["dir"], run["stem"] + ".snap"), STEPS)
        m = float(torch.load(path, weights_only=True)["m_dep"])
    except (OSError, KeyError) as e:
        raise Refusal(f"m_PHS: no 1M snapshot with m_dep for {run['id']} ({type(e).__name__})")
    if not math.isfinite(m):
        raise Refusal(f"m_PHS: m_dep of {run['id']} at 1M is not finite ({m})")
    return m


def _bstar_six(root: str, v05: dict) -> dict:
    """B*'s imputed selection R and C on seeds 0-5 (recomputed from public files; must reproduce v05)."""
    six = _runs(os.path.join(root, "p1"), v05["top3"], PG1, seeds.SELECTION_SEEDS + seeds.EXTRA_SELECTION_SEEDS, "p1")
    require_complete([r for rs in six.values() for r in rs])
    g = rank_group(six, v05["r_ref"][PG1], eligible_only=False)
    if g["order"][0] != v05["b_star"]:
        raise Refusal("v05.json's B* differs from a recomputation")
    return g["configs"][v05["b_star"]]


def candidate_p1(root: str, family: str, names, v05: dict, base: dict) -> dict:
    """Selection and P1 kill rules of one candidate family (BVS or SHL): the best eligible config on seeds
    0-2; kill (a), (b), and 8.2 (BVS) or 8.4 (SHL). No fallback (8.5)."""
    out = {"family": family, "star": None, "killed": None, "kills": {}}
    if family == "BVS" and not v05["bvs_stage0_proceed"]:
        out["killed"] = "Stage 0 (" + ", ".join(g for g in ("g0a", "g0b") if not v05[g]["proceed"]) + ")"
        return out
    entries = _runs(os.path.join(root, "p1"), names, PG1, seeds.SELECTION_SEEDS, "p1")
    require_complete([r for rs in entries.values() for r in rs])
    g = rank_group(entries, v05["r_ref"][PG1], eligible_only=True)
    out.update(order=g["order"], table=g["table"])
    if not g["order"]:
        out["killed"] = "8.1(a): no eligible config"
        return out
    star = out["star"] = g["order"][0]
    c = g["configs"][star]
    out["kills"]["a"] = {"collapses": int(np.sum(c["collapsed"])), "kill": bool(np.sum(c["collapsed"]) >= 2)}
    out["kills"]["b"] = stats.kill_b(c["R"], c["C"], base["R"], base["C"])
    if family == "BVS":
        out["kills"]["8.2"] = kill_82(entries[star])
        out["rho_star"] = rho_star(entries[star])
    else:
        out["kills"]["8.4"] = kill_84(_read_json(os.path.join(root, "stage0", "s04.json")), star)
    names_kill = {"a": "8.1(a)", "b": "8.1(b)", "8.2": "8.2", "8.4": "8.4"}
    hit = [names_kill[k] for k, v in out["kills"].items() if v["kill"]]
    out["killed"] = ", ".join(hit) or None
    return out


def candidates_p1(root: str) -> dict:
    v05 = _read_json(os.path.join(root, "v05.json"))
    base = _bstar_six(root, v05)
    return {"label": DESCRIPTIVE, "role": SELECTION, "b_star": v05["b_star"],
            "BVS": candidate_p1(root, "BVS", configs.BVS_CONFIGS, v05, base),
            "SHL": candidate_p1(root, "SHL", configs.SHL_CONFIGS, v05, base)}


def candidates(root: str) -> dict:
    """The method-v1 record: the P1 part (as candidates --stage p1) plus P2 on CarGoal1 and PointCircle1:
    R_ref from R2 and B*, kill (c), and m_PHS for a surviving BVS*."""
    v05 = _read_json(os.path.join(root, "v05.json"))
    p1 = candidates_p1(root)
    old = os.path.join(root, "p1", "candidates.json")
    if os.path.exists(old):
        with open(old) as f:
            if json.load(f) != json.loads(_dump(p1)):
                raise Refusal("p1/candidates.json differs from a recomputation of the P1 stage")
    b_star = v05["b_star"]
    alive = {fam: p1[fam]["star"] for fam in ("BVS", "SHL") if p1[fam]["star"] and not p1[fam]["killed"]}
    p2dir = os.path.join(root, "p2")
    r_ref, p2, kill_c = dict(v05["r_ref"]), {}, {fam: {} for fam in alive}
    for task in P2_TASKS:
        names = [b_star, "R2"] + list(alive.values())
        entries = _runs(p2dir, names, task, seeds.SELECTION_SEEDS, "p2")
        require_complete([r for rs in entries.values() for r in rs])
        means = mean_R_imputed(entries)
        r_ref[task] = max(means[b_star], means["R2"])
        if not r_ref[task] > 0:
            raise Refusal(f"R_ref({task}) = {r_ref[task]} is not positive")
        g = rank_group(entries, r_ref[task], eligible_only=False)
        p2[task] = g["table"]
        cb = g["table"][b_star]["mean_C"]
        for fam, star in alive.items():
            t = g["table"][star]
            kill_c[fam][task] = {"collapses": t["collapses"], "mean_C": t["mean_C"], "b_star_mean_C": cb,
                                 "kill": bool(t["collapses"] >= 2 or (t["mean_C"] > KILL_C_COST
                                                                       and cb <= stats.COST_LIMIT))}
    result = {"label": DESCRIPTIVE, "role": SELECTION, "b_star": b_star, "b_star_base": v05["b_star_base"],
              "saute_star": v05["saute_star"], "l_star": v05["l_star"], "run_saute_star": v05["run_saute_star"],
              "run_l_star": v05["run_l_star"], "r_ref": r_ref, "p1": p1, "p2": p2, "kill_c": kill_c}
    for fam in ("BVS", "SHL"):
        star, killed = p1[fam]["star"], p1[fam]["killed"]
        if fam in alive and any(v["kill"] for v in kill_c[fam].values()):
            killed = "8.1(c): " + ", ".join(t for t, v in kill_c[fam].items() if v["kill"])
        result[f"{fam.lower()}_star"] = star if killed is None else None
        result[f"{fam.lower()}_selected"] = star
        result[f"{fam.lower()}_killed"] = killed
    bvs = result["bvs_star"]
    result["rho_star"] = p1["BVS"].get("rho_star") if bvs else None
    if bvs:
        m, m_excl, short = {}, {}, []
        for task, d, stage in ((PG1, os.path.join(root, "p1"), "p1"), (CG1, p2dir, "p2"), (PC1, p2dir, "p2")):
            rs = [load_run(d, bvs, task, s, expect=expected_config(bvs), stage=stage) for s in seeds.SELECTION_SEEDS]
            vals, m_excl[task] = [], []
            for r in rs:            # a missing snapshot or a non-finite m_dep leaves the seed out (M7)
                try:
                    vals.append(m_dep_at_1m(r))
                except Refusal as e:
                    m_excl[task].append(str(e))
            if len(vals) < MIN_BVS_SEEDS:
                short.append(task)
            else:
                m[task] = float(np.mean(vals))
        result["m_phs_excluded"] = m_excl
        if short:               # protocol section 6: killed under 8.1(c), as for collapses in 2 of 3 seeds
            result["bvs_star"] = None
            result["bvs_killed"] = "8.1(c): " + ", ".join(f"{t} (m_PHS from fewer than {MIN_BVS_SEEDS} seeds)"
                                                           for t in short)
            result["rho_star"] = None
            bvs = None
        result["m_phs"] = m if bvs else None
    else:
        result["m_phs"] = None
    result["family"] = ["H_BVS"] if bvs else []
    result["family_size"] = len(result["family"])
    return result


# --- the confirmatory plan and cuts ----------------------------------------------------------------
def heldout_rel(name: str, mode: str, end_step: int = STEPS) -> str:
    """Path (under the root) of a sealed held-out output: sealed/heldout/<name>.<mode>[.n<N>].json.enc."""
    suffix = "" if end_step == STEPS else f".n{end_step}"
    return os.path.join("sealed", "heldout", f"{name}.{mode}{suffix}.json.enc")


def phs_quality_rel(stem: str) -> str:
    return os.path.join("sealed", "phs", f"{stem}.quality.json.enc")


def make_plan(root: str, extra_continuations: bool = False) -> dict:
    """plan.json (section 9 table): every confirmatory run with its stage, config, env, seed, steps, out,
    drop_ckpt, cut group and exact argv (after the interpreter), and every held-out evaluation, PHS fit,
    PHS quality check and R3 evaluation with the output path it must produce."""
    v1 = _read_json(os.path.join(root, "v1.json"))
    repo = _repo_of(root)
    bvs, shl, b = v1["bvs_star"], v1["shl_star"], v1["b_star"]
    if extra_continuations and not (v1["bvs_killed"] or v1["shl_killed"]):
        raise Refusal("--extra-continuations needs compute freed by a kill (section 19); no candidate was killed")
    cont = seeds.CONTINUATION_SEEDS + (seeds.EXTRA_CONTINUATION_SEEDS if extra_continuations else ())
    c16, c12, c8 = seeds.CONFIRMATORY_SEEDS, seeds.CONFIRMATORY_SEEDS[:12], seeds.CONFIRMATORY_SEEDS[:8]
    arms = [("B*", b, TASKS, c16)]
    if bvs:
        arms += [("BVS*", bvs, TASKS, c16), ("BVSU", "BVSU", (PG1,), c12), ("BVSD", "BVSD", (PG1,), c12)]
    if shl:
        arms.append(("SHL*", shl, TASKS, c12))
    arms.append(("R1", "R1", TASKS, c8))
    if v1["run_saute_star"]:
        arms.append(("Saute*", v1["saute_star"], TASKS, c8))
    if v1["run_l_star"]:
        arms.append(("L*", v1["l_star"], TASKS, c8))

    def cut_group(arm, seed):
        if arm == "SHL*":
            return "2" if seed >= 308 else "2b"
        if arm in ("R1", "Saute*", "L*") and seed >= 306:
            return "3"
        if arm in ("BVSU", "BVSD") and seed >= 308:
            return "5"
        return None

    runs, evals = [], []
    for stage, steps in (("conf", STEPS), ("cont3m", CONT_STEPS)):
        out = os.path.join(root, stage)
        for arm, cfg, tasks, seed_list in arms:
            if stage == "cont3m":
                if arm not in ("B*", "BVS*", "SHL*"):
                    continue
                tasks, seed_list = (PG1,), cont
            for task in tasks:
                for s in seed_list:
                    stem = train.stem_of(cfg, task, s)
                    keep = arm == "B*" or (arm in ("BVS*", "SHL*") and task == PG1 and s in cont)
                    drop = 0 if (stage == "conf" and keep) else 1
                    argv = ["-m", "safe2.train", "--config", cfg, "--env", task, "--seed", str(s), "--out",
                            _rel(out, repo), "--drop-ckpt", str(drop)]
                    extra = {}
                    if cfg in configs.DERIVED:
                        argv += ["--from-bvs", bvs]
                        extra["from_bvs"] = bvs
                        if cfg == "BVSU":
                            argv += ["--rho-star", repr(float(v1["rho_star"]))]
                            extra["rho_star"] = float(v1["rho_star"])
                    if stage == "cont3m":
                        argv += ["--steps", str(CONT_STEPS), "--resume"]
                        extra["resume_from"] = _rel(os.path.join(root, "conf", stem + ".ckpt.pt"), repo)
                    rid = f"{stage}/{stem}"
                    group = "1" if stage == "cont3m" else cut_group(arm, s)
                    runs.append({"id": rid, "stage": stage, "arm": arm, "config": cfg, "env": task, "seed": s,
                                 "steps": steps, "out": _rel(out, repo), "drop_ckpt": drop, "cut_group": group,
                                 "argv": argv, **extra})
                    modes = ["deployed"]
                    if arm == "BVS*" and stage == "conf":
                        modes += ["shield_off", "strict"] + (["s_near"] if configs.CONFIGS[bvs].rule == "SQ" else [])
                    for mode in modes:
                        evals.append({"id": f"{rid}.{mode}", "kind": "heldout", "run": rid, "mode": mode,
                                      "end_step": steps, "output": heldout_rel(stem, mode, steps), "cut_group": group})
                    if arm == "B*" and stage == "conf" and bvs:
                        fit = os.path.join(root, "phs", stem + ".posthoc.pt")
                        ckpt = _rel(os.path.join(root, "conf", stem + ".ckpt.pt"), repo)
                        evals += [
                            {"id": f"{rid}.phs_fit", "kind": "phs_fit", "run": rid, "output": os.path.join(
                                "phs", stem + ".posthoc.pt"), "cut_group": None,
                             "argv": ["-m", "safe2.posthoc", "fit", "--ckpt", ckpt, "--episodes", "1-900", "--out",
                                      _rel(fit, repo), "--seed", str(s)]},
                            {"id": f"{rid}.phs_quality", "kind": "phs_quality", "run": rid,
                             "output": phs_quality_rel(stem), "cut_group": None,
                             "argv": ["-m", "safe2.probe", "phs-quality", _rel(os.path.join(root, "conf", stem), repo)]},
                            {"id": f"{rid}.phs", "kind": "heldout", "run": rid, "mode": "phs", "end_step": STEPS,
                             "output": heldout_rel(stem, "phs"), "cut_group": None}]
    for s in R3_SEEDS:
        name = f"lagu-lit_{PG1}_s{s}"
        evals.append({"id": f"r3/{name}", "kind": "heldout", "run": None, "mode": "r3", "end_step": STEPS,
                      "source": os.path.join(R3_DIR, name), "output": heldout_rel(name, "r3"), "cut_group": None,
                      "argv": ["-m", "safe2.evaluate", "--r3", R3_DIR, "--seeds", f"{R3_SEEDS[0]}-{R3_SEEDS[-1]}"]})
    return {"b_star": b, "bvs_star": bvs, "shl_star": shl, "saute_star": v1["saute_star"], "l_star": v1["l_star"],
            "run_saute_star": v1["run_saute_star"], "run_l_star": v1["run_l_star"], "rho_star": v1["rho_star"],
            "m_phs": v1["m_phs"], "r_ref": v1["r_ref"], "family": v1["family"],
            "extra_continuations": extra_continuations, "runs": runs, "evaluations": evals}


def cut(root: str, item: str) -> dict:
    """Append a throughput cut (protocol section 19, in order) to cuts.json; refused after unblinding."""
    if os.path.exists(os.path.join(root, "UNBLINDED.txt")):
        raise Refusal("UNBLINDED.txt exists: no cut is allowed after unblinding")
    if item == "4":
        raise Refusal("cut item 4 (BVS-U / BVS-disc health runs on seeds 0-2) precedes method-v1: record it as "
                      "a section 22 deviation")
    if item not in CUT_ITEMS:
        raise Refusal(f"unknown cut item {item!r}; items in order: {', '.join(CUT_ORDER)}")
    plan = _read_json(os.path.join(root, "plan.json"))
    path = os.path.join(root, "cuts.json")
    cuts = _read_json(path)
    done = {c["item"] for c in cuts}
    if item in done:
        raise Refusal(f"cut item {item} is already in cuts.json")
    planned = lambda it: [r for r in plan["runs"] if r["cut_group"] == it]
    for earlier in CUT_ORDER[:CUT_ORDER.index(item)]:
        if earlier not in done and planned(earlier):
            raise Refusal(f"cuts are made in order: item {earlier} ({CUT_ITEMS[earlier]}) comes first")
    rs = planned(item)
    if not rs:
        raise Refusal(f"nothing in plan.json under cut item {item}")
    stage = {r["stage"] for r in rs}.pop()          # every cut item lies in one stage (conf or cont3m)
    # the public health state of each cut run is recorded: a cut follows throughput, not run health
    states = {r["id"]: load_run(os.path.join(root, r["stage"]), r["config"], r["env"], r["seed"], r["steps"],
                                stage=r["stage"])["state"] for r in rs}
    if all(v == "finished" for v in states.values()):
        raise Refusal(f"every run under cut item {item} has finished: a cut saves no compute")
    entry = {"date": _utc(), "item": item, "what": CUT_ITEMS[item], "stage": stage,
             "stems": [r["id"].split("/", 1)[1] for r in rs], "reason": "throughput",
             "states": {rid.split("/", 1)[1]: v for rid, v in states.items()}}
    _write_atomic(path, json.dumps(cuts + [entry], indent=1) + "\n")
    return {"label": DESCRIPTIVE, "cut": entry, "note": "commit and push cuts.json before any planned run is skipped"}


def cut_ids(cuts: list) -> set:
    """Run ids ("<stage>/<stem>") of the cut entries {"date", "item", "what", "stage", "stems", "reason"}
    (a stem alone is ambiguous between a confirmatory run and its 3M continuation; safe2.launch
    --from-plan reads the entries the same way)."""
    return {f"{c['stage']}/{s}" for c in cuts for s in c["stems"]}


# --- v0 hash record --------------------------------------------------------------------------------
def v0_hashes(root: str) -> dict:
    """Write PREREG_SHA256.txt (every file in safe2/ outside tests/, e.g. *.py, pause.sh, health.sh, SPEC.md,
    HOOKS.md and README.md, plus prereg_method.md and the public key, plus openssl version and git HEAD)
    and FUNC_SHA256.json (stats.source_hashes())."""
    repo = _repo_of(root)
    hfile, ffile = os.path.join(root, "PREREG_SHA256.txt"), os.path.join(root, "FUNC_SHA256.json")
    if os.path.exists(hfile) or os.path.exists(ffile):
        raise Refusal("PREREG_SHA256.txt or FUNC_SHA256.json exists already")
    files = sorted(os.path.relpath(p, repo) for p in glob.glob(os.path.join(repo, "safe2", "*"))
                   if os.path.isfile(p) and not p.endswith((".pyc", ".tmp")))      # every non-test file
    files = sorted(set(files) | {"safe2/pause.sh", "safe2/SPEC.md"})
    files += [PROTOCOL, _rel(os.path.join(root, "keys", "seal_pub.pem"), repo)]
    missing = [f for f in list(V0_FILES) + files if not os.path.exists(os.path.join(repo, f))]
    if missing:
        raise Refusal(["files to hash at method-v0 are missing:"] + sorted(set(missing)))
    ossl = subprocess.run([seal.openssl(), "version"], capture_output=True, text=True).stdout.strip()
    head = _git(repo, "rev-parse", "HEAD", check=False).strip() or "none"
    text = f"# method-v0 {_utc()} git {head} openssl {ossl}\n" + "".join(
        f"{_sha(os.path.join(repo, f))}  {f}\n" for f in files)
    _write_atomic(hfile, text)
    _write_atomic(ffile, json.dumps(stats.source_hashes(), indent=1) + "\n")
    return {"label": DESCRIPTIVE, "files": files, "functions": list(stats.DECISION_FUNCTIONS)}


SELECTION_CODE = tuple(f"safe2/{m}.py" for m in ("analyze", "stats", "probe", "configs", "seeds", "train"))


def check_selection_code(root: str) -> list[str]:
    """Before a decision record is written: the selection code (SELECTION_CODE) matches the method-v0
    section of PREREG_SHA256.txt and stats.source_hashes() matches FUNC_SHA256.json, unless the file or
    function is listed in deviations.json (protocol section 10: the v0 version decides)."""
    repo = _repo_of(root)
    hfile, ffile = os.path.join(root, "PREREG_SHA256.txt"), os.path.join(root, "FUNC_SHA256.json")
    if not (os.path.exists(hfile) and os.path.exists(ffile)):
        return ["PREREG_SHA256.txt or FUNC_SHA256.json is missing: run v0-hashes at method-v0 first"]
    with open(hfile) as f:
        text = f.read()
    v0_text = text.split("\n# ", 1)[0]                 # the first (method-v0) section only
    v0 = hash_entries(v0_text)
    dev_path = os.path.join(root, "deviations.json")
    deviations = _read_json(dev_path) if os.path.exists(dev_path) else []
    files = {d.get("file") for d in deviations if d.get("entry")}
    funcs = {d.get("function") for d in deviations if d.get("entry")}
    probs = [f"{f} differs from its method-v0 hash without a deviation entry" for f in SELECTION_CODE
             if _sha(os.path.join(repo, f)) != v0.get(f) and f not in files]
    with open(ffile) as f:
        rec = json.load(f)
    cur = stats.source_hashes()
    probs += [f"decision function {n} changed since method-v0 without a deviation entry"
              for n in sorted(set(rec) | set(cur)) if rec.get(n) != cur.get(n) and n not in funcs]
    return probs


# --- unblinding: verification ----------------------------------------------------------------------
def _frozen_part(text: str) -> str | None:
    """The protocol without sections 21 (decision log) and 22 (deviations), which change after v0. None
    unless the excluded span holds exactly those two headings (a section inserted there would escape)."""
    i, j = text.find(FROZEN_UNTIL), text.find(FROZEN_FROM)
    if i < 0 or j < i:
        return None
    heads = [ln.rstrip() for ln in text[i:j].splitlines() if ln.startswith("#")]
    return text[:i] + text[j:] if heads == [FROZEN_UNTIL, DEVIATIONS_HEAD] else None


def load_v0_stats(repo: str, tmpdir: str):
    """safe2/stats.py as committed at method-v0, loaded as a separate module."""
    src = _git_show(repo, "method-v0", "safe2/stats.py")
    if src is None:
        raise Refusal("cannot read safe2/stats.py at method-v0")
    path = os.path.join(tmpdir, "stats_method_v0.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("safe2_stats_method_v0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_tags(repo: str) -> list[str]:
    remote = {}
    for line in _git(repo, "ls-remote", "--tags", "origin").splitlines():
        sha, _, ref = line.partition("\t")
        remote[ref.strip()] = sha.strip()
    probs = []
    for tag in TAGS:
        local = _git(repo, "rev-parse", "--verify", "-q", f"refs/tags/{tag}", check=False).strip()
        if not local:
            probs.append(f"tag {tag} is missing locally")
        elif f"refs/tags/{tag}" not in remote:
            probs.append(f"tag {tag} is not on origin")
        elif remote[f"refs/tags/{tag}"] != local:
            probs.append(f"tag {tag} points to a different object on origin")
    if not probs:                                  # method-v0 <= method-v0.5 <= method-v1 <= HEAD
        chain = list(TAGS) + ["HEAD"]
        for a, b in zip(chain, chain[1:]):
            if subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", f"{a}^{{commit}}", f"{b}^{{commit}}"],
                              capture_output=True).returncode:
                probs.append(f"{a} is not an ancestor of {b}")
    return probs


def check_code(repo: str, root: str, deviations: list) -> tuple[list[str], dict, list[str]]:
    """Step 2: every file hashed at method-v0 matches (or is a listed deviation). The hash file used is the
    one committed at method-v1; the v0 and v0.5 versions must be prefixes of it. The protocol is compared
    without sections 21-22 against its method-v0 text. Returns (problems, hash entries, deviated files)."""
    rel = _rel(os.path.join(root, "PREREG_SHA256.txt"), repo)
    v1 = _git_show(repo, "method-v1", rel)
    if v1 is None:
        return [f"{rel} is not committed at method-v1"], {}, []
    probs = []
    with open(os.path.join(repo, rel), "rb") as f:
        if f.read() != v1:
            probs.append(f"{rel} differs from its method-v1 version")
    for tag in ("method-v0", "method-v0.5"):
        old = _git_show(repo, tag, rel)
        if old is None or not v1.startswith(old):
            probs.append(f"{rel} at {tag} is not a prefix of its method-v1 version")
    v0 = _git_show(repo, "method-v0", rel)
    entries = hash_entries(v1.decode())
    v0_entries = hash_entries(v0.decode()) if v0 is not None else {}
    listed = {d["file"] for d in deviations if d.get("file") and d.get("entry")}
    key_rel = _rel(os.path.join(root, "keys", "seal_pub.pem"), repo)
    for f in sorted(set(V0_FILES) | {key_rel}):
        if f not in v0_entries:
            probs.append(f"{f} has no method-v0 hash")
    deviated = []
    for f, h in sorted(v0_entries.items()):
        if f == PROTOCOL:
            tagged = _git_show(repo, "method-v0", f)
            with open(os.path.join(repo, f)) as fh:
                cur = _frozen_part(fh.read())
            if tagged is None or hashlib.sha256(tagged).hexdigest() != h:
                probs.append(f"{f}: its method-v0 version does not match the recorded hash")
            elif cur is None or cur != _frozen_part(tagged.decode()):
                (deviated if f in listed else probs).append(f"{f}: sections 0-20 or the appendices changed")
            continue
        if _sha(os.path.join(repo, f)) != h:
            (deviated if f in listed else probs).append(f"{f}: differs from its method-v0 hash")
    return probs, entries, deviated


def check_functions(repo: str, root: str, deviations: list, mod) -> tuple[list[str], list[str]]:
    """Step 3: per-function hashes of the decision functions against FUNC_SHA256.json at method-v0.
    Returns (problems, deviated functions listed in deviations.json)."""
    rel = _rel(os.path.join(root, "FUNC_SHA256.json"), repo)
    tagged = _git_show(repo, "method-v0", rel)
    if tagged is None:
        return [f"{rel} is not committed at method-v0"], []
    rec = json.loads(tagged)
    with open(os.path.join(repo, rel)) as f:
        probs = [] if json.load(f) == rec else [f"{rel} differs from its method-v0 version"]
    listed = {d["function"] for d in deviations if d.get("function") and d.get("entry")}
    names = sorted(set(rec) | set(mod.DECISION_FUNCTIONS))
    cur = {}
    for n in names:
        try:
            cur[n] = mod.source_hashes([n])[n]
        except (AttributeError, OSError, TypeError):
            cur[n] = None
    changed = [n for n in names if cur.get(n) != rec.get(n)]
    probs += [f"decision function {n} changed since method-v0 without a deviation entry" for n in changed
              if n not in listed]
    return probs, [n for n in changed if n in listed]


def pushed(repo: str, rel: str) -> list[str]:
    """Problems unless `rel` is committed, has no uncommitted changes and its last commit is on origin."""
    if subprocess.run(["git", "-C", repo, "ls-files", "--error-unmatch", rel], capture_output=True).returncode:
        return [f"{rel} is not committed"]
    probs = []
    if _git(repo, "status", "--porcelain", "--", rel).strip():
        probs.append(f"{rel} has uncommitted changes")
    last = _git(repo, "log", "-1", "--format=%H", "--", rel).strip()
    heads = [ln.split("\t")[0] for ln in _git(repo, "ls-remote", "--heads", "origin").splitlines() if ln.strip()]
    if not any(subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", last, h],
                              capture_output=True).returncode == 0 for h in heads):
        probs.append(f"{rel}: its last commit is not on origin")
    return probs


def check_deviations(repo: str, deviations: list) -> list[str]:
    """Every deviations.json entry names a file or function and its `entry` text occurs in protocol
    section 22 (the current text, which may change after v0)."""
    with open(os.path.join(repo, PROTOCOL)) as f:
        text = f.read()
    i, j = text.find(DEVIATIONS_HEAD), text.find(FROZEN_FROM)
    sec22 = text[i:j] if 0 <= i < j else ""
    probs = []
    for d in deviations:
        if not isinstance(d, dict) or not (d.get("file") or d.get("function")) or not d.get("entry"):
            probs.append(f"deviations.json: malformed entry {d!r}")
        elif d["entry"] not in sec22:
            probs.append(f"deviations.json: entry {d['entry']!r} is not in protocol section 22")
    return probs


def check_records(repo: str, root: str, entries: dict, deviations: list = ()) -> list[str]:
    """Step 4: v05.json, v1.json and plan.json match their recorded hashes and, byte for byte, their
    versions at the tag that froze them (method-v0.5, method-v1); PREREG_SHA256.txt (at method-v1) holds
    one entry per record unless the record is a listed deviation; HASHLOG.tsv, deviations.json (when
    present) and cuts.json are committed, clean and on origin (SPEC section 11); cuts.json is well formed."""
    probs = []
    hrel = _rel(os.path.join(root, "PREREG_SHA256.txt"), repo)
    htext = (_git_show(repo, "method-v1", hrel) or b"").decode()
    listed = {d["file"] for d in deviations if d.get("file") and d.get("entry")}
    for name in RECORDS:
        rel = _rel(os.path.join(root, name), repo)
        if rel not in entries:
            probs.append(f"{rel} has no recorded hash")
        elif _sha(os.path.join(repo, rel)) != entries[rel]:
            probs.append(f"{rel} differs from its recorded hash")
        tagged = _git_show(repo, RECORD_TAGS[name], rel)
        if tagged is None:
            probs.append(f"{rel} is not committed at {RECORD_TAGS[name]}")
        else:
            with open(os.path.join(repo, rel), "rb") as f:
                if f.read() != tagged:
                    probs.append(f"{rel} differs from its {RECORD_TAGS[name]} version")
        n = sum(1 for ln in htext.splitlines() if not ln.startswith("#") and ln.partition("  ")[2].strip() == rel)
        if n > 1 and rel not in listed:
            probs.append(f"{rel} was hashed {n} times in PREREG_SHA256.txt (rewritten) without a deviation entry")
    probs += pushed(repo, _rel(os.path.join(root, seal.HASHLOG), repo))
    if os.path.exists(os.path.join(root, "deviations.json")):
        probs += pushed(repo, _rel(os.path.join(root, "deviations.json"), repo))
    rel = _rel(os.path.join(root, "cuts.json"), repo)
    cut_probs = pushed(repo, rel)
    probs += cut_probs
    if f"{rel} is not committed" in cut_probs:
        return probs
    try:
        with open(os.path.join(repo, rel)) as f:
            cuts = json.load(f)
        plan = _read_json(os.path.join(root, "plan.json"))
        ids = {r["id"]: r["cut_group"] for r in plan["runs"]}
        group = lambda it: {i for i, g in ids.items() if g == it}
        done = []
        for c in cuts:
            if c.get("reason") != "throughput" or c.get("item") not in CUT_ITEMS or \
                    any(ids.get(i) != c["item"] for i in cut_ids([c])):
                probs.append(f"{rel}: malformed cut entry {c.get('item')!r}")
                continue
            if cut_ids([c]) != group(c["item"]):
                probs.append(f"{rel}: cut item {c['item']} does not cover its whole group")
            skipped = [it for it in CUT_ORDER[:CUT_ORDER.index(c["item"])] if it not in done and group(it)]
            if skipped:
                probs.append(f"{rel}: cut item {c['item']} precedes item {skipped[0]} (cut-list order)")
            done.append(c["item"])
    except (ValueError, KeyError, TypeError, Refusal) as e:
        probs.append(f"{rel}: unreadable ({type(e).__name__})")
    return probs


def plan_runs(root: str, plan: dict, cut: set) -> dict:
    """{run id: public facts} of every planned run that was not cut."""
    out, pending = {}, pending_runs()
    for p in plan["runs"]:
        if p["id"] in cut:
            continue
        exp = expected_config(p["config"], plan.get("bvs_star"), p.get("rho_star", NAN))
        r = load_run(os.path.join(root, p["stage"]), p["config"], p["env"], p["seed"], p["steps"], exp, p["stage"],
                     pending)
        src = out.get(f"conf/{r['stem']}") if p["stage"] == "cont3m" else None
        if r["state"] == "missing" and src is not None and src["state"] in ("diverged", "unfinished"):
            r["state"] = f"not-run (source {src['state']})"   # never resumed (SPEC 3.6; review-ops M3)
        r["arm"] = p["arm"]
        out[p["id"]] = r
    return out


def required_outputs(plan: dict, cut: set, runs: dict) -> list[dict]:
    """Planned evaluations that must exist: not cut, and their run finished (R3: always)."""
    need = []
    for e in plan["evaluations"]:
        if e.get("run") not in cut and (e.get("run") is None or runs[e["run"]]["state"] == "finished"):
            need.append(e)
    return need


def check_completeness(root: str, plan: dict, cut: set) -> tuple[list[str], dict, list[dict]]:
    """Step 5: every planned run (minus cuts) is finished, diverged or unfinished after 3 resumes; every
    finished run has its planned outputs; the hash log verifies."""
    runs = plan_runs(root, plan, cut)
    probs = [f"{rid}: {r['state']}" for rid, r in runs.items() if r["state"] in INCOMPLETE]
    need = required_outputs(plan, cut, runs)
    probs += [f"missing planned output {e['output']}" for e in need if not os.path.exists(os.path.join(root, e["output"]))]
    probs += [f"hash log: {p}" for p in seal.verify(root)]
    return probs, runs, need


DEADLINE = "computed at the analysis deadline"


def deadline_completeness(root: str, plan: dict, cut: set) -> tuple[list[str], dict, list[dict], dict]:
    """Step 5 at the analysis deadline (protocol sections 10 and 16, M8): every planned, non-cut run that
    is queued, running or missing, and every finished run without its deployed held-out output, is
    treated as unfinished (imputed by section 11: worst case for the candidate, B*'s median for B*).
    Other missing planned outputs are dropped. Both are listed; the hash log must still verify."""
    runs = plan_runs(root, plan, cut)
    late = {}
    for rid, r in runs.items():
        if r["state"] in INCOMPLETE:
            late[rid] = r["state"]
            r["state"] = "unfinished"
    need = required_outputs(plan, cut, runs)
    for e in need:
        if (e.get("run") is not None and e["kind"] == "heldout" and e.get("mode") == "deployed"
                and not os.path.exists(os.path.join(root, e["output"]))):
            late[e["run"]] = "finished without its deployed held-out output"
            runs[e["run"]]["state"] = "unfinished"
    need = required_outputs(plan, cut, runs)
    dropped = sorted(e["output"] for e in need if not os.path.exists(os.path.join(root, e["output"])))
    need = [e for e in need if os.path.exists(os.path.join(root, e["output"]))]
    probs = [f"hash log: {p}" for p in seal.verify(root)]
    by_arm: dict = {}
    for rid in sorted(late):
        by_arm.setdefault(runs[rid].get("arm"), []).append({"run": rid, "was": late[rid]})
    return probs, runs, need, {"label": DESCRIPTIVE, "note": DEADLINE, "imputed_as_unfinished": by_arm,
                               "dropped_outputs": dropped}


# --- unblinding: decryption -----------------------------------------------------------------------
class Unsealer:
    """Decryption with the lead's private key. OpenSSL unlocks the key once (its own passphrase prompt);
    the unlocked key stays in memory and reaches each `openssl pkeyutl -decrypt` through a pipe."""

    def __init__(self, key_path: str):
        self.pem = subprocess.run([seal.openssl(), "pkey", "-in", key_path], stdout=subprocess.PIPE,
                                  check=True).stdout

    def _unwrap(self, wrapped: bytes) -> bytes:
        r, w = os.pipe()
        try:
            buf = memoryview(self.pem)
            while buf:
                buf = buf[os.write(w, buf):]
        finally:
            os.close(w)
        try:
            return subprocess.run([seal.openssl(), "pkeyutl", "-decrypt", "-inkey", f"/dev/fd/{r}"] + seal._OAEP,
                                  input=wrapped, capture_output=True, check=True, pass_fds=(r,)).stdout
        finally:
            os.close(r)

    def decrypt(self, blob: bytes) -> bytes:
        w64, c64 = blob.strip().split(b".", 1)
        kiv = self._unwrap(base64.b64decode(w64))
        k, iv = kiv[:32], kiv[32:48]
        return subprocess.run([seal.openssl(), "enc", "-d", "-aes-256-ctr", "-K", k.hex(), "-iv", iv.hex()],
                              input=base64.b64decode(c64), capture_output=True, check=True).stdout

    def file(self, path: str):
        with open(path, "rb") as f:
            return json.loads(self.decrypt(f.read()))

    def records(self, path: str) -> list[dict]:
        if not os.path.exists(path):
            return []
        rows = [json.loads(self.decrypt(seal._RECORD.match(ln).group(2))) for ln in seal.read_lines(path)]
        return sorted(rows, key=lambda r: r["step"])


R3_STEPS = (800_000, 850_000, 900_000, 950_000, 1_000_000)     # safe2.evaluate.R3_STEPS


def expected_snapshots(mode: str, end_step: int) -> list[tuple[int, list[int]]]:
    """(step, sorted layouts) of each snapshot a held-out output must hold: snapshot i of N * {0.82, ...,
    1.00} plays layouts 70000 + 20 i + k; R3 snapshot j (800k, ..., 1M) plays blocks 2j and 2j + 1."""
    lay = lambda i: [seeds.heldout_layout(i, k) for k in range(seeds.HELDOUT_EPISODES)]
    if mode == "r3":
        return [(s, sorted(lay(2 * j) + lay(2 * j + 1))) for j, s in enumerate(R3_STEPS)]
    return [(s, lay(i)) for i, s in enumerate(seeds.heldout_targets(end_step))]


def _check_heldout(out: dict, e: dict, stem: str) -> None:
    """A decrypted held-out output belongs to the planned evaluation `e` of run `stem` (for R3 the lagu
    name): its stem, mode and end step match, its snapshot steps and per-snapshot layouts are the planned
    ones, and its pooled R and C are the means of its episodes."""
    name = e["output"]
    got = (out.get("stem"), out.get("mode"), out.get("end_step"))
    if got != (stem, e["mode"], e["end_step"]):
        raise Refusal(f"{name}: output is for {got}, not the planned ({stem}, {e['mode']}, {e['end_step']})")
    snaps = out.get("snapshots") or []
    have = [(s.get("step"), sorted(x["layout"] for x in s.get("episodes", []))) for s in snaps]
    if have != expected_snapshots(e["mode"], e["end_step"]):
        raise Refusal(f"{name}: snapshot steps or layouts differ from the planned held-out episodes")
    eps = [x for s in snaps for x in s["episodes"]]
    for k, col in (("R", "return"), ("C", "cost")):
        mean, stored = float(np.mean([_num(x.get(col)) for x in eps])), _num(out.get("pooled", {}).get(k))
        if not (math.isclose(stored, mean, rel_tol=1e-9, abs_tol=1e-9)        # a non-finite endpoint is
                or not (math.isfinite(stored) or math.isfinite(mean))):       # imputed later, not refused
            raise Refusal(f"{name}: pooled {k} is not the mean of its episodes")


# --- unblinding: per-run quantities -----------------------------------------------------------------
def view(run: dict, mode: str = "deployed", arm: str | None = None) -> dict:
    """A run seen through one held-out mode (BVS* secondary modes; PHS on a B* run)."""
    return dict(run, out=run["heldout"].get(mode), arm=arm or run["arm"], mode=mode)


def arrays(rs: list[dict]) -> dict:
    """Endpoint arrays of one arm and task: R, C (the held-out endpoint whenever the run has one, diverged
    or not; non-finite, and so imputed, for a run without one, e.g. one that stopped after diverging;
    protocol section 11), done (False only for 'unfinished after 3 resumes') and the diverged flags, which
    make a run collapsed in stats.primary."""
    R, C = [], []
    for r in rs:
        ok = r.get("out") is not None
        R.append(_num(r["out"]["pooled"].get("R")) if ok else NAN)
        C.append(_num(r["out"]["pooled"].get("C")) if ok else NAN)
    return {"R": np.array(R, float), "C": np.array(C, float),
            "done": np.array([r["state"] != "unfinished" for r in rs], bool),
            "diverged": np.array([r["diverged"] for r in rs], bool)}


def impute(mod, X: list, Y: list) -> list[dict]:
    """stats.impute_task per task (x = candidate rule, y = B* rule)."""
    out = []
    for xs, ys in zip(X, Y):
        a, b = arrays(xs), arrays(ys)
        i = mod.impute_task(a["R"], a["C"], b["R"], b["C"], done_x=a["done"], done_y=b["done"])
        out.append(dict(i, div_x=a["diverged"], div_y=b["diverged"]))
    return out


def decide(mod, X: list, Y: list, r_ref: list, n_perm: int = stats.N_PERM, seed: int = 0) -> dict:
    """The primary computation of section 12 for per-task run lists X (candidate) and Y (comparator):
    imputation, then stats.primary with the diverged flags, n_perm permutations, RNG seed and the module's
    ALPHA_TEST."""
    imp = impute(mod, X, Y)
    res = mod.primary([i["Rx"] for i in imp], [i["Cx"] for i in imp], [i["Ry"] for i in imp],
                      [i["Cy"] for i in imp], r_ref, diverged_x=[i["div_x"] for i in imp],
                      diverged_y=[i["div_y"] for i in imp], n_perm=n_perm, seed=seed, alpha=mod.ALPHA_TEST)
    res["n_x"] = [len(x) for x in X]
    res["n_y"] = [len(y) for y in Y]
    res["imputed_x"] = [[r["id"] for r, f in zip(xs, i["imputed_x"]) if f] for xs, i in zip(X, imp)]
    res["imputed_y"] = [[r["id"] for r, f in zip(ys, i["imputed_y"]) if f] for ys, i in zip(Y, imp)]
    return res


def confirmed(mod, res: dict, family: list) -> dict:
    """Holm over the primary family (size 1: H_BVS) at the module's ALPHA_TEST, with the guards."""
    if len(family) != 1:
        raise Refusal(f"the primary family must hold exactly H_BVS here, not {family}")
    adj = float(holm([res["p_perm"]])[0])
    reject = adj <= mod.ALPHA_TEST
    if reject != (res["p_perm"] <= mod.ALPHA_TEST):
        raise AssertionError("Holm with one hypothesis must equal the raw test")
    return {"p_holm": adj, "holm_reject": bool(reject), "beats": bool(reject and res["beats"])}


def _welch(xs, ys) -> dict | None:
    """welch_strat over the tasks where both arms have at least 2 finite values (None if none)."""
    pairs = []
    for x, y in zip(xs, ys):
        x, y = np.asarray(x, float), np.asarray(y, float)
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        if len(x) >= 2 and len(y) >= 2:
            pairs.append((x, y))
    if not pairs:
        return None
    return dict(stats.welch_strat([p[0] for p in pairs], [p[1] for p in pairs]), n_tasks=len(pairs))


def _episodes(r: dict) -> list[dict]:
    return [] if r.get("out") is None else [e for s in r["out"]["snapshots"] for e in s["episodes"]]


def _per_episode(r: dict, fn) -> float:
    eps = _episodes(r)
    return fn([e["cost"] for e in eps]) if eps else NAN


def contrast(X: list, Y: list, r_ref: list, tasks: list, mod=stats) -> dict:
    """Secondary contrast x - y (section 14): per task and pooled 95% Welch intervals of the normalised
    penalised score s (linearised), r = R / R_ref, c = C / 25 and the seed-level per-run hinge, all on the
    imputed endpoints; violation share and CVaR10 from the observed held-out episodes (runs without them
    are left out and counted); collapses with a Fisher exact test."""
    imp = impute(mod, X, Y)
    per, cols = {}, {k: ([], []) for k in ("s", "r", "c", "seed_level", "violation", "cvar10")}
    nx = ny = cx = cy = 0
    for task, xs, ys, i, rr in zip(tasks, X, Y, imp, r_ref):
        v = {"s": (mod.linearised(i["Rx"], i["Cx"], rr), mod.linearised(i["Ry"], i["Cy"], rr)),
             "r": (i["Rx"] / rr, i["Ry"] / rr), "c": (i["Cx"] / stats.COST_LIMIT, i["Cy"] / stats.COST_LIMIT),
             "seed_level": (mod.penalised(i["Rx"], i["Cx"]) / rr, mod.penalised(i["Ry"], i["Cy"]) / rr),
             "violation": (np.array([_per_episode(r, stats.violation_share) for r in xs]),
                           np.array([_per_episode(r, stats.violation_share) for r in ys])),
             "cvar10": (np.array([_per_episode(r, stats.cvar_upper) for r in xs]),
                        np.array([_per_episode(r, stats.cvar_upper) for r in ys]))}
        colx = mod.collapsed(i["Rx"], rr) | i["div_x"]
        coly = mod.collapsed(i["Ry"], rr) | i["div_y"]
        nx, ny, cx, cy = nx + len(xs), ny + len(ys), cx + int(colx.sum()), cy + int(coly.sum())
        per[task] = {"n_x": len(xs), "n_y": len(ys), "R_x": float(i["Rx"].mean()), "C_x": float(i["Cx"].mean()),
                     "R_y": float(i["Ry"].mean()), "C_y": float(i["Cy"].mean()),
                     "S_x": mod.arm_score(i["Rx"], i["Cx"], rr), "S_y": mod.arm_score(i["Ry"], i["Cy"], rr),
                     "collapses_x": int(colx.sum()), "collapses_y": int(coly.sum()),
                     "imputed_x": int(i["imputed_x"].sum()), "imputed_y": int(i["imputed_y"].sum())}
        for k, (a, b) in v.items():
            per[task][f"d{k}"] = _welch([a], [b])
            if k in ("violation", "cvar10"):
                per[task][f"{k}_x"] = float(np.nanmean(a)) if np.isfinite(a).any() else None
                per[task][f"{k}_y"] = float(np.nanmean(b)) if np.isfinite(b).any() else None
                per[task][f"{k}_missing"] = [int((~np.isfinite(a)).sum()), int((~np.isfinite(b)).sum())]
            cols[k][0].append(a)
            cols[k][1].append(b)
    pooled = {f"d{k}": _welch(a, b) for k, (a, b) in cols.items()}
    pooled.update(collapses_x=cx, collapses_y=cy, n_x=nx, n_y=ny, fisher_collapse=fisher_exact(cx, nx - cx, cy, ny - cy))
    return {"per_task": per, "pooled": pooled}


def contrast_or_reason(X: list, Y: list, r_ref: list, tasks: list, mod=stats, **extra) -> dict:
    """contrast(), or why it cannot be computed (an arm without runs in a task, nothing finite to impute)."""
    if not all(X) or not all(Y):
        return {"not_computable": "an arm has no runs in some task", **extra}
    try:
        return dict(contrast(X, Y, r_ref, tasks, mod), **extra)
    except ValueError as e:
        return {"not_computable": str(e), **extra}


# --- unblinding: the tables ---------------------------------------------------------------------------
def _value(r: dict, step: int, key: str) -> float:
    """An outcome column of a run at a step: decrypted sealed rows (seeds >= 300), else the public row."""
    for row in r.get("sealed_rows", []):
        if row.get("step") == step:
            return _num(row.get(key))
    row = _row(r, step)
    return _num(row.get(key)) if row is not None and key in row else NAN


def _arm(runs: dict, arm: str, task: str, stage: str = "conf", seed_list=None, mode: str = "deployed",
         as_arm: str | None = None) -> list[dict]:
    rs = [r for r in runs.values() if r["arm"] == arm and r["env"] == task and r["stage"] == stage
          and (seed_list is None or r["seed"] in seed_list)]
    return [view(r, mode, as_arm) for r in sorted(rs, key=lambda r: r["seed"])]


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 and math.isfinite(num) and math.isfinite(den) else NAN


def training_cost(X: list, Y: list, tasks: list) -> dict:
    """Cumulative training cost at 1M and over 200k-1M (cum_cost, cum_cost_200k at the 1M row), x - y,
    per task and pooled (95% Welch); runs without a 1M row are left out and counted."""
    out = {}
    for key in ("cum_cost", "cum_cost_200k"):
        xs = [np.array([_value(r, STEPS, key) for r in rs]) for rs in X]
        ys = [np.array([_value(r, STEPS, key) for r in rs]) for rs in Y]
        out[key] = {"per_task": {t: {"x": float(np.nanmean(a)) if np.isfinite(a).any() else None,
                                     "y": float(np.nanmean(b)) if np.isfinite(b).any() else None,
                                     "missing": [int((~np.isfinite(a)).sum()), int((~np.isfinite(b)).sum())],
                                     "diff": _welch([a], [b])} for t, a, b in zip(tasks, xs, ys)},
                    "pooled": _welch(xs, ys)}
    return out


def iqm_score(X: list, Y: list, r_ref: list, mod=stats) -> dict:
    """Sensitivity: Delta_s with the interquartile mean in place of the arm mean (imputed endpoints)."""
    imp = impute(mod, X, Y)
    per = [float(mod.penalised(iqm(i["Rx"]), iqm(i["Cx"])) - mod.penalised(iqm(i["Ry"]), iqm(i["Cy"]))) / rr
           for i, rr in zip(imp, r_ref)]
    return {"per_task": per, "ds_iqm": float(np.mean(per))}


def try_decide(mod, X: list, Y: list, r_ref: list, n_perm: int, keep=None) -> dict:
    """decide() on the runs that `keep` accepts; 'not computable' when an arm and task keeps < 2 runs."""
    if keep is not None:
        X = [[r for r in rs if keep(r)] for rs in X]
        Y = [[r for r in rs if keep(r)] for rs in Y]
    if any(len(rs) < 2 for rs in X + Y):
        return {"not_computable": "an arm keeps fewer than 2 runs in some task"}
    try:
        res = decide(mod, X, Y, r_ref, n_perm)
    except ValueError as e:
        return {"not_computable": str(e)}
    return {"ds": res["ds"], "p_perm": res["p_perm"], "verdict": res["verdict"], "guards": res["guards"],
            "welch_p": res["welch"]["p"], "welch_ci95": res["welch"]["ci95"], "n_x": res["n_x"], "n_y": res["n_y"]}


def _clean_run(r: dict) -> bool:
    """A run that is not imputed: finished with a finite held-out endpoint (a diverged run that finished
    keeps its endpoint and is not imputed; protocol section 11)."""
    return r["state"] == "finished" and r.get("out") is not None and all(
        math.isfinite(_num(r["out"]["pooled"].get(k))) for k in ("R", "C"))


def _nansum_or_nan(xs) -> float:
    """Sum of the finite values; NaN (reported as null) when none is finite, e.g. a counter not logged."""
    a = np.asarray(xs, float)
    return float(np.nansum(a)) if np.isfinite(a).any() else NAN


def mechanism(arm_runs: dict, rc: configs.RunConfig) -> dict:
    """Section 14.4 for one BVS-family arm ({task: runs}): training counters at 1M (pooled per task),
    held-out shield statistics (deployed), V calibration, the training-average margin bound against the
    realised training average, the window bound on each 50k interval from 50k, and the saturation gap
    pooled over the arm's runs."""
    per, gaps = {}, []
    for task, rs in arm_runs.items():
        tot = {k: _nansum_or_nan([_value(r, STEPS, k) for r in rs])
               for k in ("int_cum", "shield_steps_cum", "empty_cum", "F_cum", "ep_cum", "entries_cum", "misses_cum",
                         "rho_num", "rho_den", "trig_num")}
        n1m = sum(math.isfinite(_value(r, STEPS, "ep_cum")) for r in rs)
        eps = [e for r in rs for e in _episodes(r)]
        steps = sum(e["length"] for e in eps)
        vp = np.array([e.get("v_pred", NAN) for e in eps], float)
        vr = np.array([e.get("v_real", NAN) for e in eps], float)
        ok = np.isfinite(vp) & np.isfinite(vr)
        allowed = [e["n_allowed_mean"] for e in eps if "n_allowed_mean" in e]
        bound = []
        for r in rs:
            e50, e1m = _row(r, 50_000), _row(r, STEPS)
            K = (e1m["episodes"] - e50["episodes"]) if e50 and e1m else NAN
            realised = _ratio(_value(r, STEPS, "cum_cost") - _value(r, 50_000, "cum_cost"), K)
            jmax = float(r.get("horizon") or 1000)
            b = (stats.COST_LIMIT + (M_SAFE + rc.eta_m * (jmax - stats.COST_LIMIT) - rc.m0) / (rc.eta_m * K)
                 if K and math.isfinite(K) and K > 0 else NAN)
            win = []                     # window bound (A.6, projected integrator) on each 50k interval
            for a, z in zip(range(50_000, STEPS, 50_000), range(100_000, STEPS + 1, 50_000)):
                ra, rz = _row(r, a), _row(r, z)
                W = (rz["episodes"] - ra["episodes"]) if ra and rz else NAN
                if W and math.isfinite(W) and W > 0:
                    win.append({"from": a, "to": z, "W": W,
                                "realised": _ratio(_value(r, z, "cum_cost") - _value(r, a, "cum_cost"), W),
                                "bound": stats.COST_LIMIT + (M_SAFE + rc.eta_m * (jmax - stats.COST_LIMIT)
                                                             + stats.COST_LIMIT / 2) / (rc.eta_m * W)})
            bound.append({"run": r["id"], "K": K, "bound": b, "realised": realised, "windows": win})
        per[task] = {
            "training": {"runs_with_1m_row": n1m, "runs": len(rs),
                         "intervention_rate": _ratio(tot["int_cum"], tot["shield_steps_cum"]),
                         "empty_set_steps": tot["empty_cum"], "mean_F": _ratio(tot["F_cum"], tot["ep_cum"]),
                         "miss_rate": _ratio(tot["misses_cum"], tot["entries_cum"]), "entries": tot["entries_cum"],
                         "rho_achieved": _ratio(tot["rho_num"], tot["rho_den"]),
                         "trigger_rate": _ratio(tot["trig_num"], tot["rho_den"])},
            "heldout": {"episodes": len(eps),
                        "intervention_rate": _ratio(sum(e.get("interventions", 0) for e in eps), steps),
                        "empty_set_rate": _ratio(sum(e.get("empty_steps", 0) for e in eps), steps),
                        "mean_F": float(np.mean([e.get("F", NAN) for e in eps])) if eps else NAN,
                        "miss_rate": _ratio(sum(e.get("misses", 0) for e in eps),
                                            sum(e.get("entries", 0) for e in eps)),
                        "allowed_set_size": float(np.mean(allowed)) if allowed else "not logged"},
            "v_calibration": {"episodes": int(ok.sum()), "v_pred": float(vp[ok].mean()) if ok.any() else NAN,
                              "v_real": float(vr[ok].mean()) if ok.any() else NAN,
                              "corr": float(np.corrcoef(vp[ok], vr[ok])[0, 1]) if ok.sum() > 2 and vp[ok].std() > 0
                              and vr[ok].std() > 0 else NAN},
            "margin_bound": {"runs": bound, "mean_bound": float(np.nanmean([b["bound"] for b in bound])),
                             "mean_realised": float(np.nanmean([b["realised"] for b in bound]))}}
        gaps += [g for r in rs if r.get("out") for g in r["out"].get("gaps", [])]
    g = np.asarray(gaps, float)
    gap = {"n": int(g.size)}
    if g.size:
        gap.update(mean=float(g.mean()), p95=float(np.percentile(g, 95)))
        gap["approximately_saturated"] = bool(gap["mean"] <= INV_MEAN and gap["p95"] <= INV_P95)
        gap["wording"] = ("approximately saturated (not sufficient for recursive feasibility; never "
                          "'invariant')" if gap["approximately_saturated"] else "not saturated")
    return {"per_task": per, "saturation_gap": gap}


def _param_count(path: str) -> int:
    import torch
    snap = torch.load(path, weights_only=True)
    return int(sum(t.numel() for sd in snap.values() if isinstance(sd, dict) for t in sd.values()
                   if isinstance(t, torch.Tensor)))


def compute_table(runs: dict) -> dict:
    """Section 14.6: wall-clock hours and updates (last public row) and deployed parameter counts per arm."""
    from safe2 import evaluate
    out = {}
    for r in runs.values():
        if r["state"] != "finished":
            continue
        k = f"{r['arm']} {r['env']} {r['stage']}"
        last = r["rows"][-1]
        d = out.setdefault(k, {"runs": 0, "wall_h": [], "updates": [], "params": None})
        d["runs"] += 1
        d["wall_h"].append(last["wall_s"] / 3600)
        d["updates"].append(last["updates"])
        if d["params"] is None:
            try:
                snap = evaluate.snapshot_file(os.path.join(r["dir"], r["stem"] + ".snap"), r["steps"])
                d["params"] = _param_count(snap)
            except (OSError, KeyError, RuntimeError):
                d["params"] = "unavailable"
    return {k: {"runs": v["runs"], "wall_h_mean": float(np.mean(v["wall_h"])),
                "updates_mean": float(np.mean(v["updates"])), "params": v["params"]} for k, v in sorted(out.items())}


def arm_table(runs: dict, r3: list, r_ref: dict) -> dict:
    """Section 18: every arm, task and stage (B*'s PHS mode as arm PHS, R3 on PointGoal1) with n, the
    mean R and C over the runs with a held-out endpoint, S_t, r and c, violation share and CVaR10 (means
    of the per-run values), collapses, and the runs without an endpoint (imputed in the contrasts),
    diverged, unfinished and resumed. Observed values only: no imputation here."""
    groups = {}
    for r in runs.values():
        groups.setdefault((r["arm"], r["env"], r["stage"]), []).append(view(r))
        if r["arm"] == "B*" and "phs" in r.get("heldout", {}):
            groups.setdefault(("PHS", r["env"], r["stage"]), []).append(view(r, "phs", "PHS"))
    if r3:
        groups[("R3", PG1, "r3")] = r3
    out = {}
    for (arm, task, stage), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        a, rr = arrays(rs), float(r_ref[task])
        ok = np.isfinite(a["R"]) & np.isfinite(a["C"])
        R, C = a["R"][ok], a["C"][ok]
        viol = np.array([_per_episode(r, stats.violation_share) for r in rs])
        cvar = np.array([_per_episode(r, stats.cvar_upper) for r in rs])
        coll = (stats.collapsed(a["R"], rr) & ok) | a["diverged"]
        out[f"{arm} | {task} | {stage}"] = {
            "n": len(rs), "n_endpoint": int(ok.sum()),
            "mean_R": float(R.mean()) if R.size else None, "mean_C": float(C.mean()) if C.size else None,
            "S": stats.arm_score(R, C, rr) if R.size else None,
            "r": float(R.mean() / rr) if R.size else None, "c": float(C.mean() / stats.COST_LIMIT) if C.size else None,
            "violation_share": float(np.nanmean(viol)) if np.isfinite(viol).any() else None,
            "cvar10": float(np.nanmean(cvar)) if np.isfinite(cvar).any() else None,
            "collapses": int(coll.sum()),
            "no_endpoint": [r["id"] for r, k in zip(rs, ok) if not k],
            "diverged": [r["id"] for r in rs if r["diverged"]],
            "unfinished": [r["id"] for r in rs if r["state"] == "unfinished"],
            "resumed": [r["id"] for r in rs if r["n_resumes"] > 0]}
    return {"label": DESCRIPTIVE, "notes": [NOTE_COST_OBS, NOTE_LIMIT], "arms": out}


def run_status(runs: dict) -> dict:
    """Imputed, diverged, unfinished and resumed runs by arm (section 18)."""
    out = {}
    for r in runs.values():
        d = out.setdefault(f"{r['arm']} ({r['stage']})",
                           {"runs": 0, "diverged": [], "unfinished": [], "resumed": []})
        d["runs"] += 1
        if r["diverged"]:
            d["diverged"].append(r["id"])
        if r["state"] == "unfinished":
            d["unfinished"].append(r["id"])
        if r["n_resumes"] > 0:
            d["resumed"].append(r["id"])
    return out


def tables(root: str, plan: dict, runs: dict, r3: list, phs_quality: dict, mod, v0mod, deviated: list,
           n_perm: int) -> dict:
    """Every table of sections 12 and 14 in one pass."""
    v05 = _read_json(os.path.join(root, "v05.json"))
    v1 = _read_json(os.path.join(root, "v1.json"))
    r_ref = [float(plan["r_ref"][t]) for t in TASKS]
    bvs, shl = plan["bvs_star"], plan["shl_star"]
    c8, c12 = seeds.CONFIRMATORY_SEEDS[:8], seeds.CONFIRMATORY_SEEDS[:12]
    Bs = [_arm(runs, "B*", t) for t in TASKS]
    out = {"arms": arm_table(runs, r3, plan["r_ref"])}       # every arm, whether or not BVS was killed
    notes = [NOTE_COST_OBS, NOTE_LIMIT]
    if plan.get("b_star") and v05.get("b_star_base") == "omni":
        notes.append("BVS (literal TD3 learner) was compared with a baseline using a different base learner "
                     "(B* uses the OmniSafe TD3 base); BVS* - L* is reported next to it.")
    if not plan["family"]:
        out["primary"] = {"label": CONFIRMATORY, "family": [], "verdict": "no confirmatory claim: the primary family is "
                          f"empty (BVS killed: {v1.get('bvs_killed')})", "notes": notes}
    else:
        X = [_arm(runs, "BVS*", t) for t in TASKS]
        dmod = v0mod if deviated else mod
        res = decide(dmod, X, Bs, r_ref, n_perm)
        h = confirmed(dmod, res, plan["family"])
        out["primary"] = {"label": CONFIRMATORY, "hypothesis": "H_BVS: Delta_s > 0 (BVS* vs B*, arm-mean penalised "
                          "score, three tasks, stratified permutation test)", "family": plan["family"],
                          "bvs_star": bvs, "b_star": plan["b_star"], "verdict": res["verdict"], "beats": h["beats"],
                          "ds": res["ds"], "p_perm": res["p_perm"], "p_holm": h["p_holm"], "alpha_test": dmod.ALPHA_TEST,
                          "n_perm": n_perm, "decided_by": ("safe2/stats.py at method-v0 (deviations: "
                                                           + ", ".join(deviated) + ")") if deviated else
                          "safe2/stats.py (decision functions unchanged since method-v0)",
                          "result": res, "notes": notes}
        if deviated:
            cur = decide(mod, X, Bs, r_ref, n_perm)
            out["primary"]["sensitivity_current_stats"] = {"label": DESCRIPTIVE, "functions": deviated,
                                                           "alpha_test": mod.ALPHA_TEST,
                                                           "verdict": cur["verdict"], "ds": cur["ds"],
                                                           "p_perm": cur["p_perm"], "guards": cur["guards"]}
        sec = {"label": DESCRIPTIVE, "notes": [NOTE_COST_OBS], "bvs_vs_bstar": contrast_or_reason(X, Bs, r_ref, list(TASKS), mod),
               "training_cost": training_cost(X, Bs, list(TASKS))}
        dc = sec["bvs_vs_bstar"].get("pooled", {}).get("dc")
        if dc:          # "at matched cost" only if the pooled 90% interval of dc lies inside +-0.20 (+-5 cost)
            sec["bvs_vs_bstar"]["pooled"]["matched_cost_wording"] = bool(
                -MATCHED_C < dc["ci90"][0] and dc["ci90"][1] < MATCHED_C)
        larm = "L*" if plan["run_l_star"] else "B*" if plan["l_star"] == plan["b_star"] else \
            "Saute*" if plan["run_saute_star"] and plan["l_star"] == plan["saute_star"] else None
        Ls = [_arm(runs, larm, t, seed_list=c8) for t in TASKS] if larm else None     # seeds 300-307 of L*
        sec["bvs_vs_lstar"] = {"label": EXPLORATORY, "l_star": plan["l_star"], "arm": larm,
                               "unadjusted": try_decide(mod, X, Ls, r_ref, n_perm) if Ls else "no L* runs"}
        out["secondary"] = sec
        out["sensitivity"] = {
            "label": DESCRIPTIVE, "role": "sensitivity of the primary (none decisive)", "notes": [NOTE_COST_OBS],
            "iqm": iqm_score(X, Bs, r_ref, mod), "welch": {"p": res["welch"]["p"], "ci95": res["welch"]["ci95"]},
            "drop_resumed": try_decide(mod, X, Bs, r_ref, n_perm, keep=lambda r: r["n_resumes"] == 0),
            "drop_imputed": try_decide(mod, X, Bs, r_ref, n_perm, keep=_clean_run),
            "without_CarGoal1": try_decide(mod, [X[0], X[2]], [Bs[0], Bs[2]], [r_ref[0], r_ref[2]], n_perm),
            "without_PointCircle1": try_decide(mod, X[:2], Bs[:2], r_ref[:2], n_perm)}
        ctl = {"label": DESCRIPTIVE, "notes": [NOTE_COST_OBS]}
        X12 = [_arm(runs, "BVS*", PG1, seed_list=c12)]
        rc = configs.CONFIGS[bvs]
        for arm, key in (("BVSU", "bvs_vs_bvsu"), ("BVSD", "bvs_vs_bvsdisc")):
            Y = _arm(runs, arm, PG1)
            if Y:
                ctl[key] = contrast_or_reason([X12[0]], [Y], [r_ref[0]], [PG1], mod,
                                              mechanism_x=mechanism({PG1: X12[0]}, rc)["per_task"][PG1],
                                              mechanism_y=mechanism({PG1: Y}, rc)["per_task"][PG1])
        U = _arm(runs, "BVSU", PG1)
        if U:
            t007 = [_value(r, STEPS, "trig007_num") for r in U]
            ctl["bvs_vs_bvsu"].update(
                rho_star=plan["rho_star"], note=NOTE_BVSU,
                counterfactual_trigger_rate_T007=_ratio(sum(t007), sum(_value(r, STEPS, "rho_den") for r in U))
                if all(math.isfinite(v) for v in t007) else "not logged")
        P = [_arm(runs, "B*", t, mode="phs", as_arm="PHS") for t in TASKS]
        if any(r["out"] is not None for rs in P for r in rs):
            ctl["phs_vs_bstar"] = contrast_or_reason(P, Bs, r_ref, list(TASKS), mod)
            ctl["bvs_vs_phs"] = contrast_or_reason(X, P, r_ref, list(TASKS), mod, note=NOTE_PHS)
            ctl["phs_quality"] = phs_quality
            ctl["m_phs"] = plan["m_phs"]
        for m in ("shield_off", "strict", "s_near"):
            M = [_arm(runs, "BVS*", t, mode=m) for t in TASKS]
            if any(r["out"] is not None for rs in M for r in rs):
                ctl[f"bvs_{m}_vs_deployed"] = contrast_or_reason(M, X, r_ref, list(TASKS), mod)
        ctl["bvs_vs_r1"] = contrast_or_reason(X, [_arm(runs, "R1", t) for t in TASKS], r_ref, list(TASKS), mod)
        # B* is a Saute config: no Saute* arm. Saute* then runs as the L* arm when L* is Saute* (and not B*),
        # else B* stands for it
        sarm = ("Saute*" if plan["run_saute_star"] else
                "L*" if plan["run_l_star"] and plan["l_star"] == plan["saute_star"] else "B*")
        ctl["bvs_vs_saute"] = contrast_or_reason(X, [_arm(runs, sarm, t, seed_list=c8) for t in TASKS], r_ref,
                                                 list(TASKS), mod, saute_star=plan["saute_star"], arm=sarm)
        if r3:
            ctl["bvs_vs_r3"] = contrast_or_reason([X[0]], [r3], [r_ref[0]], [PG1], mod,
                                                  note="R3 is descriptive only: different seeds, an untuned "
                                                       "over-limit method")
        out["controls"] = ctl
        out["mechanism"] = dict(mechanism({t: x for t, x in zip(TASKS, X)}, rc), label=DESCRIPTIVE,
                                notes=[NOTE_COST_OBS])
    if shl:
        Ss = [_arm(runs, "SHL*", t) for t in TASKS]
        gain = shl.rsplit("-", 1)[1]
        a = v1["p1"]["SHL"]["table"][shl]
        b = v05["tables"]["p1_baselines"][gain]
        s04 = os.path.join(root, "stage0", "s04.json")
        out["shl"] = {"label": EXPLORATORY, "shl_star": shl, "notes": [NOTE_COST_OBS],
                      "shl_vs_bstar_unadjusted": try_decide(mod, Ss, Bs, r_ref, n_perm),
                      "same_gain_p1": {"control": gain, "d_mean_R": a["mean_R"] - b["mean_R"],
                                       "d_mean_C": a["mean_C"] - b["mean_C"], "d_score": a["score"] - b["score"]},
                      "gradient_share": kill_84(_read_json(s04), shl)["shares"] if os.path.exists(s04) else "missing"}
    cont = {}
    for arm in ("B*", "BVS*", "SHL*"):
        C3 = _arm(runs, arm, PG1, stage="cont3m")
        if C3:
            C1 = {r["seed"]: r for r in _arm(runs, arm, PG1)}
            rows = [{"run": r["id"], "state": r["state"], "R_3m": r["out"]["pooled"]["R"] if r.get("out") else None,
                     "C_3m": r["out"]["pooled"]["C"] if r.get("out") else None,
                     "R_1m": C1[r["seed"]]["out"]["pooled"]["R"] if C1.get(r["seed"], {}).get("out") else None,
                     "C_1m": C1[r["seed"]]["out"]["pooled"]["C"] if C1.get(r["seed"], {}).get("out") else None}
                    for r in C3]
            cont[arm] = rows
    out["other"] = {"label": DESCRIPTIVE, "compute": compute_table(runs), "continuations_3m": cont or "none run",
                    "runs_by_arm": run_status(runs),
                    "selection_records": {"v05": v05, "v1": v1},
                    "s01": {"label": EXPLORATORY, "files": sorted(os.path.relpath(p, root) for p in glob.glob(
                        os.path.join(root, "stage0", "s01", "**", "*"), recursive=True) if os.path.isfile(p))}}
    return out


def _dump_decrypted(root: str, runs: dict, r3_out: dict, phs_quality: dict) -> None:
    """Decrypted sealed records and held-out files go to analysis/decrypted/ (reported, section 18)."""
    base = os.path.join(root, "analysis", "decrypted")
    for r in runs.values():
        if r.get("sealed_rows"):
            path = os.path.join(base, "rows", r["stage"], r["stem"] + ".csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["step"] + train.OUTCOME_FIELDS, extrasaction="ignore", restval="")
                w.writeheader()
                w.writerows(r["sealed_rows"])
        for mode, o in r["heldout"].items():
            _write_atomic(os.path.join(base, "heldout", r["stage"], f"{r['stem']}.{mode}.json"), _dump(o))
    for name, o in r3_out.items():
        _write_atomic(os.path.join(base, "heldout", "r3", os.path.basename(name) + ".json"), _dump(o))
    for rid, o in phs_quality.items():
        _write_atomic(os.path.join(base, "phs", os.path.basename(rid) + ".quality.json"), _dump(o))


def unblind(root: str, key: str, n_perm: int = stats.N_PERM, stats_mod=None, deadline: bool = False) -> dict:
    """SPEC 9.6: verify tags, code, decision functions, records and completeness in order (refusing at the
    first failing step), then write UNBLINDED.txt, decrypt and compute every table in one pass."""
    mod = stats_mod or stats
    repo = _repo_of(root)
    dev_path = os.path.join(root, "deviations.json")
    deviations = _read_json(dev_path) if os.path.exists(dev_path) else []
    probs = check_tags(repo)
    if probs:
        raise Refusal(["step 1 (tags):"] + probs)
    probs, entries, dev_files = check_code(repo, root, deviations)
    probs += check_deviations(repo, deviations)
    if probs:
        raise Refusal(["step 2 (code hashes):"] + probs)
    probs, dev_funcs = check_functions(repo, root, deviations, mod)
    if probs:
        raise Refusal(["step 3 (decision functions):"] + probs)
    probs = check_records(repo, root, entries, deviations)
    if probs:
        raise Refusal(["step 4 (records):"] + probs)
    plan = _read_json(os.path.join(root, "plan.json"))
    cuts = _read_json(os.path.join(root, "cuts.json"))
    cut_set = cut_ids(cuts)
    if deadline:
        probs, runs, need, late = deadline_completeness(root, plan, cut_set)
    else:
        (probs, runs, need), late = check_completeness(root, plan, cut_set), None
    if probs:
        raise Refusal(["step 5 (completeness):"] + probs)
    # a listed change of safe2/stats.py (module constants such as ALPHA_TEST, or helpers that are not
    # function-hashed) also decides with the method-v0 module, as a listed decision-function change does
    deviated = dev_funcs + (["safe2/stats.py"] if any(f.startswith("safe2/stats.py:") for f in dev_files) else [])
    with tempfile.TemporaryDirectory() as tmp:
        v0mod = load_v0_stats(repo, tmp) if deviated else None
        try:
            uns = Unsealer(key)             # unlocking the key decrypts no outcome: done before UNBLINDED.txt
        except (subprocess.CalledProcessError, OSError) as e:
            raise Refusal(f"OpenSSL could not read the private key {key} ({type(e).__name__})")
        ublog = os.path.join(root, "UNBLINDED.txt")
        earlier = open(ublog).read().splitlines() if os.path.exists(ublog) else []
        code = (f"analyze={_sha(os.path.join(repo, 'safe2', 'analyze.py'))}\t"
                f"stats={_sha(os.path.join(repo, 'safe2', 'stats.py'))}\tdeviations={_sha(dev_path) or 'none'}")
        with open(ublog, "a") as f:
            f.write(f"{_utc()}\t{_git(repo, 'rev-parse', 'HEAD').strip()}\t{code}\n")
            f.flush()
            os.fsync(f.fileno())
        for r in runs.values():
            r["heldout"] = {}
            r["sealed_rows"] = uns.records(os.path.join(root, "sealed", r["stage"], r["stem"] + ".rows"))
        r3_out, phs_quality = {}, {}
        for e in need:
            path = os.path.join(root, e["output"])
            if e["kind"] == "heldout":
                o = uns.file(path)
                _check_heldout(o, e, os.path.basename(e["source"]) if e["run"] is None else runs[e["run"]]["stem"])
                if e["run"] is None:
                    r3_out[e["id"]] = o
                else:
                    runs[e["run"]]["heldout"][e["mode"]] = o
            elif e["kind"] == "phs_quality":
                phs_quality[e["run"]] = uns.file(path)
        _dump_decrypted(root, runs, r3_out, phs_quality)
        r3 = [{"id": i, "stem": os.path.basename(i), "arm": "R3", "env": PG1, "seed": int(i.rsplit("_s", 1)[1]),
               "stage": "r3", "state": "finished", "diverged": False, "n_resumes": 0, "out": o, "mode": "r3",
               "heldout": {"r3": o}} for i, o in sorted(r3_out.items())]
        out = tables(root, plan, runs, r3, phs_quality, mod, v0mod, deviated, n_perm)
    hist = seal.history(root)
    out["verification"] = {"label": DESCRIPTIVE, "deviated_files": dev_files, "deviated_functions": dev_funcs,
                           "cuts": cuts, "planned_runs": len(plan["runs"]), "analysed_runs": len(runs),
                           "resealed_files": sorted(k[0] for k, v in hist.items() if k[1] == "file" and len(v) > 1),
                           "relogged_records": dict(sorted(_count([k[0] for k, v in hist.items()
                                                                   if k[1] == "record" and len(v) > 1]).items()))}
    # a re-run never replaces the first output (main writes analysis/unblind.<n>); its primary is labelled
    # a re-run when the code or the deviations differ from the first unblinding
    n = len(earlier) + 1
    out["unblinding"] = {"label": DESCRIPTIVE, "n": n, "code": code, "first": earlier[0] if earlier else None}
    if late is not None:
        out["deadline"] = late
        if "primary" in out:
            out["primary"]["deadline_note"] = DEADLINE
    if earlier and earlier[0].split("\t")[2:] != code.split("\t") and "primary" in out:
        out["primary"]["label"] = RERUN
        out["primary"]["rerun_note"] = ("computed after the first unblinding with different analysis code or "
                                        "deviations; the first output (analysis/unblind.json) stands")
    return out


def _count(xs: list) -> dict:
    return {x: xs.count(x) for x in set(xs)}


# --- output -----------------------------------------------------------------------------------------
def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool, type(None))) for x in v):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def render(obj, indent: int = 0) -> list[str]:
    """Indented text of a cleaned result; a dict carrying a label shows it next to its key."""
    pad, lines = "  " * indent, []
    for k, v in obj.items():
        if k == "label":
            continue
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:" + (f"  [{v['label']}]" if isinstance(v.get("label"), str) else ""))
            lines += render(v, indent + 1)
        elif isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v):
            lines.append(f"{pad}{k}:")
            for x in v:
                if isinstance(x, dict):
                    lines.append(f"{pad}  -")
                    lines += render(x, indent + 2)
                else:
                    lines.append(f"{pad}  - {_fmt(x)}")
        else:
            lines.append(f"{pad}{k}: {_fmt(v)}")
    return lines


def report(root: str, name: str, obj: dict) -> str:
    """Write analysis/<name>.json and analysis/<name>.txt; returns the text."""
    obj = _clean(obj)
    d = os.path.join(root, "analysis")
    _write_atomic(os.path.join(d, name + ".json"), json.dumps(obj, indent=1) + "\n")
    head = obj.get("label") or "sections labelled CONFIRMATORY / DESCRIPTIVE / EXPLORATORY"
    text = f"safe2.analyze {name} [{head}]\n" + "\n".join(render(obj)) + "\n"
    _write_atomic(os.path.join(d, name + ".txt"), text)
    return text


def plan_summary(plan: dict) -> dict:
    count = lambda xs: {k: xs.count(k) for k in sorted(set(xs), key=str)}
    return {"label": DESCRIPTIVE, "b_star": plan["b_star"], "bvs_star": plan["bvs_star"], "shl_star": plan["shl_star"],
            "runs_by_arm_and_stage": count([f"{r['arm']} {r['stage']}" for r in plan["runs"]]),
            "runs_by_cut_group": count([str(r["cut_group"]) for r in plan["runs"]]),
            "evaluations_by_kind": count([f"{e['kind']} {e.get('mode', '')}".strip() for e in plan["evaluations"]]),
            "extra_continuations": plan["extra_continuations"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", nargs="?", choices=["v0-hashes", "p1-baselines", "bstar", "candidates", "plan", "cut"])
    p.add_argument("item", nargs="?", help="cut: the cut-list item (" + ", ".join(CUT_ORDER) + ")")
    p.add_argument("--unblind", action="store_true")
    p.add_argument("--key", help="--unblind: the lead's private key (passphrase typed into OpenSSL's prompt)")
    p.add_argument("--deadline", action="store_true",
                   help="--unblind at the analysis deadline: incomplete runs are imputed as unfinished (section 10)")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--stage", choices=["p1", "final"], default="final", help="candidates: p1 = before P2")
    p.add_argument("--extra-continuations", action="store_true", help="plan: seeds 303-304 (freed by a kill)")
    p.add_argument("--overwrite", action="store_true", help="replace a decision record that differs (log it in 22)")
    a = p.parse_args(argv)
    if (a.unblind == bool(a.cmd) or (a.cmd == "cut") != (a.item is not None) or (a.unblind and not a.key)
            or (a.deadline and not a.unblind)):
        p.error("give one subcommand (cut takes ITEM), or --unblind [--deadline] --key PRIVATE.pem")
    root = os.path.abspath(a.root)
    try:
        if a.cmd in ("p1-baselines", "bstar", "candidates", "plan"):
            probs = check_selection_code(root)
            if probs:
                raise Refusal(["selection code is not the method-v0 code:"] + probs)
        if a.unblind:
            res = unblind(root, a.key, deadline=True) if a.deadline else unblind(root, a.key)
            n = res["unblinding"]["n"]
            text = report(root, "unblind" if n == 1 else f"unblind.{n}", res)
        elif a.cmd == "v0-hashes":
            text = report(root, "v0-hashes", v0_hashes(root))
        elif a.cmd == "p1-baselines":
            res = p1_baselines(root)
            res["record"] = write_record(os.path.join(root, "p1", "top3.json"), res, a.overwrite, root)
            text = report(root, "p1-baselines", res)
        elif a.cmd == "bstar":
            res = bstar(root)
            path = os.path.join(root, "v05.json")
            status = write_record(path, res, a.overwrite, root)
            text = report(root, "bstar", dict(res, record=status, hashed=append_hashes(root, "method-v0.5", [path])))
        elif a.cmd == "candidates" and a.stage == "p1":
            res = candidates_p1(root)
            res["record"] = write_record(os.path.join(root, "p1", "candidates.json"), res, a.overwrite, root)
            text = report(root, "candidates-p1", res)
        elif a.cmd == "candidates":
            res = candidates(root)
            text = report(root, "candidates", dict(res, record=write_record(os.path.join(root, "v1.json"), res,
                                                                                a.overwrite, root)))
        elif a.cmd == "plan":
            plan = make_plan(root, a.extra_continuations)
            cpath = os.path.join(root, "cuts.json")
            if os.path.exists(cpath) and _read_json(cpath) != []:
                raise Refusal("cuts.json already holds cuts; the plan is frozen at method-v1")
            status = write_record(os.path.join(root, "plan.json"), plan, a.overwrite, root)
            if not os.path.exists(cpath):
                _write_atomic(cpath, "[]\n")
            hashed = append_hashes(root, "method-v1", [os.path.join(root, "v1.json"), os.path.join(root, "plan.json")])
            text = report(root, "plan", dict(plan_summary(plan), record=status, hashed=hashed))
        else:
            text = report(root, f"cut-{a.item}", cut(root, a.item))
    except Refusal as e:
        for line in e.problems:
            print(f"refusing: {line}")
        return 1
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
