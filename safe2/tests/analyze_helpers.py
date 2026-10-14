"""Synthetic study files for the safe2.analyze tests: public run files (json, CSV rows every 50k, final
.pt), fake 1M snapshots, and sealed held-out outputs and outcome records."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

import numpy as np

from safe2 import configs, seal, seeds, train


def _val(v, step):
    return v(step) if callable(v) else v


def write_run(d, config: str, env: str, seed: int, R=10.0, C=20.0, steps: int = 1_000_000, finished: bool = True,
              diverge_at: int | None = None, upto: int | None = None, n_resumes: int = 0, cols=None, rc=None,
              horizon: int = 1000) -> str:
    """Public files of one fake run. Rows every 50k up to `upto` (default `steps`). eval_return and
    eval_cost are R and C (constants or functions of the step); every finite flag is 0 from `diverge_at`;
    cols(step) adds outcome columns. Blinded seeds (>= 300) get the health columns only."""
    d = str(d)
    os.makedirs(d, exist_ok=True)
    stem = train.stem_of(config, env, seed)
    rc = rc or configs.make(config)
    meta = {"args": {"config": config, "env": env, "seed": seed, "steps": steps, "out": d, "drop_ckpt": 0,
                     "argv": []}, "config": asdict(rc), "horizon": horizon, "obs_dim": 60, "act_dim": 2,
            "blind": train.blinded(seed), "resumes": [], "n_resumes": n_resumes}
    with open(os.path.join(d, stem + ".json"), "w") as f:
        json.dump(meta, f)
    blind = train.blinded(seed)
    fields = train.HEALTH_FIELDS if blind else train.HEALTH_FIELDS + train.OUTCOME_FIELDS
    with open(os.path.join(d, stem + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        for step in range(50_000, (upto or steps) + 1, 50_000):
            ok = int(diverge_at is None or step < diverge_at)
            row = {"step": step, "episodes": step // horizon, "updates": step, "sps": 100.0, "wall_s": step / 100,
                   "finite_out": ok, "finite_net": ok, "finite_loss": ok, "finite_dual": ok, "nonfinite_actions": 0,
                   "resumed": int(n_resumes > 0), "n_resumes": n_resumes}
            if not blind:
                row.update(eval_return=_val(R, step), eval_cost=_val(C, step), eval_violation=0.0)
                if cols:
                    row.update(cols(step))
            w.writerow(row)
    if finished:
        open(os.path.join(d, stem + ".pt"), "w").close()
    return stem


def write_snapshot(d, stem: str, step: int, **content) -> str:
    import torch
    path = os.path.join(str(d), stem + ".snap", f"{step}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(content), path)
    return path


def heldout_output(stem: str, mode: str, R: float, C: float, rng: np.random.Generator, sd_ep: float = 5.0,
                   end_step: int = 1_000_000, shielded: bool = False) -> dict:
    """An evaluate-style held-out output: 10 snapshots x 20 episodes on layouts 70000-70199 whose pooled
    means are exactly R and C (R3, mode "r3": lagu snapshots 800k, ..., 1M x 40 episodes, blocks 2j and
    2j + 1, as safe2.evaluate.r3_plan)."""
    rets = R + sd_ep * rng.standard_normal(200)
    costs = np.maximum(0.0, C + sd_ep * rng.standard_normal(200))
    rets += R - rets.mean()
    costs = costs * (C / costs.mean()) if costs.mean() > 0 else costs
    if mode == "r3":
        plan = [(s, [(2 * j, k) for k in range(20)] + [(2 * j + 1, k) for k in range(20)])
                for j, s in enumerate((800_000, 850_000, 900_000, 950_000, 1_000_000))]
    else:
        plan = [(s, [(i, k) for k in range(20)]) for i, s in enumerate(seeds.heldout_targets(end_step))]
    snaps = []
    for i, (step, idx) in enumerate(plan):
        eps = []
        for b, k in idx:
            j = 20 * b + k
            e = {"layout": seeds.heldout_layout(b, k), "return": float(rets[j]), "cost": float(costs[j]),
                 "length": 1000, "nonfinite_actions": 0}
            if shielded:
                e.update(interventions=10, empty_steps=1, F=0.5, entries=2, misses=1, v_pred=1.0, v_real=1.2)
            eps.append(e)
        snaps.append({"i": i, "step": step, "sha256": "0" * 64, "episodes": eps})
    out = {"stem": stem, "mode": mode, "end_step": end_step, "snapshots": snaps,
           "pooled": {"R": float(np.mean(rets)), "C": float(np.mean(costs))}}
    if shielded and mode == "deployed":
        out["gaps"] = [0.01, 0.02, 0.03]
    return out


def seal_heldout(root, rel_enc: str, out: dict) -> None:
    """Seal `out` at <root>/<rel_enc> (rel_enc ends in .json.enc, as plan.json declares it)."""
    assert rel_enc.startswith("sealed" + os.sep) and rel_enc.endswith(".enc")
    seal.write_file(str(root), rel_enc[len("sealed" + os.sep):-len(".enc")], out)


def seal_rows(root, stage: str, stem: str, rows: list[dict]) -> None:
    rf = seal.RecordFile(str(root), stage, stem)
    for row in rows:
        rf.append(row["step"], {**{k: "" for k in train.OUTCOME_FIELDS}, **row})
