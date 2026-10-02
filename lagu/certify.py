"""Certificate for Proposition 1 (plan W0 / CC4): python -m lagu.certify <dir> [--pairs pairs.json]

Each declared pair (A, B, seed) is expected to be "equal" or "different". For both runs the final
<stem>.pt (agent.state_dict()) is hashed per tensor with SHA-256 over actor, actor_targ, members,
members_targ, cost, cost_targ (and cost2/cost2_targ when present), the pi_opt/q_opt/c_opt states
(and lam_opt when present), lam_param and the final lambda. Counters, RNG state and cfg are not
hashed.

  equal      every hash matches, and the eval_*, lambda, qc_traj and mc_cost_traj CSV columns
             are identical row by row (as written, so NaN == NaN)
  different  at least one hash differs

Every pair must also be live: in each run sum(n_risk) > 0 and lambda > 0 in some row, and in each
gated run (cfg.use_gate) engagement > 0 in some row. Without liveness, "equal" would be vacuous.

Every pair must also be a like-for-like comparison, or either verdict could be vacuous:
  - each .pt carries every hashed component (a malformed .pt would otherwise hash nothing);
  - each CSV ends at the .pt's env_step (so a stale .pt beside a rerun's CSV cannot be certified);
  - both runs end at the same step;
  - the two cfgs differ in use_gate and/or gate_grad and in nothing else;
  - when both <stem>.json exist, their trainer args agree apart from arm, tag, gate_grad, and the
    bookkeeping args out, resume, drop_ckpt, ckpt_every and steps.
A positive control that differs only because of a config or step mismatch would otherwise pass.

Writes <dir>/certificate.json and <dir>/certificate.md and exits 1 if any expectation fails.
pairs.json is a list of {"a": "lagu-fix", "b": "lagu_nogate-fix", "seed": 0, "expect": "equal"}
with an optional "env" (default --env). Labels are <arm>[-<tag>] as in the run stems.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch

# plan W0: the inert pairs under the fixed and the realised-cost (episodic) duals, and two
# positive controls (Eq. 18 statewise dual; gradient through delta_ada)
DEFAULT_PAIRS = (
    [{"a": "lagu-fix", "b": "lagu_nogate-fix", "seed": s, "expect": "equal"} for s in (0, 1)]
    + [{"a": "lagu-epi", "b": "lagu_nogate-epi", "seed": s, "expect": "equal"} for s in (0, 1)]
    + [{"a": "lagu-sw", "b": "lagu_nogate-sw", "seed": 0, "expect": "different"},
       {"a": "lagu-epigg", "b": "lagu_nogate-epi", "seed": 0, "expect": "different"}])
NETS = ("actor", "actor_targ", "members", "members_targ", "cost", "cost_targ", "cost2", "cost2_targ")
OPTS = ("pi_opt", "q_opt", "c_opt", "lam_opt")
SCALARS = ("lam_param", "lambda")
CSV_COLS = ("lambda", "qc_traj", "mc_cost_traj")      # plus every eval_* column
EVAL_COLS = ("eval_return", "eval_cost", "eval_violation")   # required even if both CSVs lack them
GATE_KEYS = ("use_gate", "gate_grad")                 # the only cfg keys a pair may differ in
ARGS_IGNORED = ("arm", "tag", "gate_grad", "out", "resume", "drop_ckpt", "ckpt_every", "steps")


def _sha(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256(f"{t.dtype}{tuple(t.shape)}".encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def _sha_value(v) -> str:
    if torch.is_tensor(v):
        return _sha(v)
    if isinstance(v, float):
        return hashlib.sha256(b"float64" + struct.pack("<d", v)).hexdigest()
    return hashlib.sha256(repr(v).encode()).hexdigest()


def tensor_hashes(sd: dict) -> dict:
    """Per-tensor SHA-256 of a LagU.state_dict() (see the module docstring for what is included)."""
    out = {}
    for name in NETS:
        for k, v in (sd.get(name) or {}).items():
            out[f"{name}.{k}"] = _sha(v)
    for name in OPTS:
        if name in sd:
            for pid, st in sd[name]["state"].items():
                for k, v in st.items():
                    out[f"{name}.{pid}.{k}"] = _sha_value(v)
    for name in SCALARS:
        if name in sd:
            out[name] = _sha_value(float(sd[name]))
    return out


def missing_parts(sd: dict) -> list[str]:
    """Hashed components a complete LagU.state_dict() carries but `sd` lacks (or holds empty)."""
    cfg = sd.get("cfg") or {}
    episodic = cfg.get("dual") == "episodic"
    nets = NETS[:6] + (NETS[6:] if cfg.get("cost_twin") else ())
    out = [n for n in nets if not sd.get(n)]
    out += [n for n in OPTS[:3] if not (sd.get(n) or {}).get("state")]
    out += [n for n in ("lambda",) + (("lam_param", "lam_opt") if episodic else ()) if n not in sd]
    return out


def digest(hashes: dict) -> str:
    return hashlib.sha256("".join(f"{k}={v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()


def read_csv(path: str):
    with open(path) as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def load_run(dirpath: str, stem: str) -> dict:
    run = {"stem": stem}
    pt, csv_path = os.path.join(dirpath, stem + ".pt"), os.path.join(dirpath, stem + ".csv")
    missing = [os.path.basename(p) for p in (pt, csv_path) if not os.path.exists(p)]
    if missing:
        run["missing"] = missing
        return run
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    run["hashes"] = tensor_hashes(sd)
    run["digest"] = digest(run["hashes"])
    run["cfg"] = sd.get("cfg", {})
    run["lacks"] = missing_parts(sd)
    run["env_step"] = sd.get("env_step")
    js = os.path.join(dirpath, stem + ".json")
    run["args"] = json.load(open(js)).get("args") if os.path.exists(js) else None
    run["fields"], run["rows"] = read_csv(csv_path)
    col = lambda k: np.array([_num(r.get(k)) for r in run["rows"]], float)
    run["last_step"] = col("step")[-1] if run["rows"] else math.nan
    n_risk, lam, eng = col("n_risk"), col("lambda"), col("engagement")
    run["live"] = {"sum_n_risk": float(np.nansum(n_risk)),
                   "max_lambda": float(np.nanmax(lam)) if np.isfinite(lam).any() else math.nan,
                   "max_engagement": float(np.nanmax(eng)) if np.isfinite(eng).any() else math.nan,
                   "gated": bool(run["cfg"].get("use_gate", False)),
                   "resumed": bool(np.nansum(col("resumed")) > 0)}
    return run


def compare_csv(a: dict, b: dict) -> list[str]:
    """Columns (step, eval_*, lambda, qc_traj, mc_cost_traj) that are not identical row by row."""
    cols = (["step"] + sorted(set(EVAL_COLS) | {f for f in a["fields"] + b["fields"] if f.startswith("eval_")})
            + list(CSV_COLS))
    if len(a["rows"]) != len(b["rows"]):
        return ["row count"]
    return [c + (" (missing)" if c not in a["fields"] or c not in b["fields"] else "") for c in cols
            if c not in a["fields"] or c not in b["fields"]
            or any(ra[c] != rb[c] for ra, rb in zip(a["rows"], b["rows"]))]


def liveness(runs: list[dict]) -> list[str]:
    bad = []
    for r in runs:
        lv = r["live"]
        if not lv["sum_n_risk"] > 0:
            bad.append(f"{r['stem']}: no risk-sensitive updates (sum n_risk = 0)")
        if not lv["max_lambda"] > 0:
            bad.append(f"{r['stem']}: lambda never > 0")
        if lv["gated"] and not lv["max_engagement"] > 0:
            bad.append(f"{r['stem']}: the gate never engaged")
    if not any(r["live"]["gated"] for r in runs):
        bad.append("neither run has the gate on (cfg.use_gate)")
    return bad


def comparable(a: dict, b: dict, res: dict) -> list[str]:
    """Reasons the two runs are not a like-for-like gate comparison (see the module docstring)."""
    bad = []
    for r in (a, b):
        if r["lacks"]:
            bad.append(f"{r['stem']}: the .pt lacks {', '.join(r['lacks'])}")
        if r["env_step"] is None or r["last_step"] != r["env_step"]:
            bad.append(f"{r['stem']}: the CSV ends at step {r['last_step']:g} but the .pt is at "
                       f"env_step {r['env_step']} (stale .pt or unfinished run)")
    if a["env_step"] != b["env_step"]:
        bad.append(f"the runs end at different steps ({a['env_step']} vs {b['env_step']})")
    other = [k for k in res["cfg_diff"] if k not in GATE_KEYS]
    if other or not res["cfg_diff"]:
        bad.append("the cfgs must differ in use_gate/gate_grad and nothing else, but "
                   + (f"they also differ in {', '.join(other)}" if other else "they are identical"))
    if a["args"] is not None and b["args"] is not None:
        res["args_diff"] = sorted(k for k in set(a["args"]) | set(b["args"])
                                  if k not in ARGS_IGNORED and a["args"].get(k) != b["args"].get(k))
        if res["args_diff"]:
            bad.append(f"the trainer args differ in {', '.join(res['args_diff'])}")
    return bad


def check_pair(dirpath: str, pair: dict, env: str, cache: dict) -> dict:
    env = pair.get("env", env)
    stems = [f"{pair[k]}_{env}_s{pair['seed']}" for k in ("a", "b")]
    for s in stems:
        if s not in cache:
            cache[s] = load_run(dirpath, s)
    a, b = cache[stems[0]], cache[stems[1]]
    res = {"a": pair["a"], "b": pair["b"], "seed": pair["seed"], "env": env, "expect": pair["expect"],
           "stems": stems, "reasons": []}
    if pair["expect"] not in ("equal", "different"):
        res["reasons"].append(f"unknown expectation {pair['expect']!r}")
    for r in (a, b):
        if "missing" in r:
            res["reasons"].append(f"{r['stem']}: missing {', '.join(r['missing'])}")
    if res["reasons"]:
        res["passed"] = False
        return res
    keys = sorted(set(a["hashes"]) | set(b["hashes"]))
    diff = [k for k in keys if a["hashes"].get(k) != b["hashes"].get(k)]
    res.update(n_tensors=len(keys), n_differ=len(diff), differ=diff[:20],
               digests=[a["digest"], b["digest"]], csv_differ=compare_csv(a, b),
               cfg_diff=sorted(k for k in set(a["cfg"]) | set(b["cfg"]) if a["cfg"].get(k) != b["cfg"].get(k)),
               live={r["stem"]: r["live"] for r in (a, b)})
    if pair["expect"] == "equal":
        if diff:
            res["reasons"].append(f"{len(diff)} of {len(keys)} tensors differ (first: {diff[0]})")
        if res["csv_differ"]:
            res["reasons"].append(f"CSV columns differ: {', '.join(res['csv_differ'])}")
        resumed = [r["stem"] for r in (a, b) if r["live"]["resumed"]]
        if (diff or res["csv_differ"]) and resumed:
            res["reasons"].append(f"{', '.join(resumed)} was resumed from a checkpoint, and a resumed run is "
                                  "not bit-identical to an uninterrupted one: rerun the pair from scratch "
                                  "before reading this as a gate channel")
    elif pair["expect"] == "different" and not diff:
        res["reasons"].append(f"all {len(keys)} tensors are identical")
    res["reasons"] += comparable(a, b, res)
    res["reasons"] += liveness([a, b])
    res["passed"] = not res["reasons"]
    return res


def certify(dirpath: str, pairs: list[dict], env: str = "SafetyPointGoal1-v0") -> dict:
    cache: dict = {}
    results = [check_pair(dirpath, p, env, cache) for p in pairs]
    return {"dir": os.path.abspath(dirpath), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "passed": all(r["passed"] for r in results), "pairs": results,
            "runs": {s: {"digest": r["digest"], "tensors": r["hashes"]} for s, r in sorted(cache.items())
                     if "hashes" in r}}


def markdown(cert: dict) -> str:
    g = lambda x: f"{x:.4g}" if isinstance(x, float) and math.isfinite(x) else "-"
    lines = [f"# Gate-channel certificate: {'PASS' if cert['passed'] else 'FAIL'}", "",
             f"`{cert['dir']}`, {cert['time']}. Per-tensor SHA-256 of the final `.pt`; equal pairs also need "
             "identical `eval_*`, `lambda`, `qc_traj`, `mc_cost_traj` columns.", "",
             "| A | B | seed | expect | tensors differing | CSV columns differing | sum n_risk | "
             "max engagement | max lambda | result |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in cert["pairs"]:
        if "n_tensors" in r:
            lv = list(r["live"].values())
            eng = max((v["max_engagement"] for v in lv if v["gated"]), default=math.nan)
            cells = [f"{r['n_differ']} / {r['n_tensors']}", ", ".join(r["csv_differ"]) or "none",
                     " / ".join(g(v["sum_n_risk"]) for v in lv), g(eng), " / ".join(g(v["max_lambda"]) for v in lv)]
        else:
            cells = ["-"] * 5
        verdict = "pass" if r["passed"] else "**FAIL**: " + "; ".join(r["reasons"])
        lines.append(f"| {r['a']} | {r['b']} | {r['seed']} | {r['expect']} | " + " | ".join(cells) + f" | {verdict} |")
    lines += ["", "| run | digest (SHA-256 over the per-tensor hashes) |", "|---|---|"]
    lines += [f"| {s} | `{v['digest']}` |" for s, v in cert["runs"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dir")
    p.add_argument("--pairs", help="JSON list of pairs (default: the plan's W0 pairs)")
    p.add_argument("--env", default="SafetyPointGoal1-v0", help="env of pairs that do not name one")
    args = p.parse_args(argv)
    pairs = json.load(open(args.pairs)) if args.pairs else DEFAULT_PAIRS
    cert = certify(args.dir, pairs, args.env)
    with open(os.path.join(args.dir, "certificate.json"), "w") as f:
        json.dump(cert, f, indent=1)
    md = markdown(cert)
    with open(os.path.join(args.dir, "certificate.md"), "w") as f:
        f.write(md)
    print(md)
    return 0 if cert["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
