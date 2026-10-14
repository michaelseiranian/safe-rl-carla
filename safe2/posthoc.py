"""Offline fits of the viability critic V and the backup actor pi_b on a finished run's replay, and
their entry recall (safe2/SPEC.md section 7; prereg_method.md section 5 (G0b) and Appendix A.9 (PHS)).

    python -m safe2.posthoc fit --ckpt RUN.ckpt.pt --episodes A-B --out OUT.posthoc.pt [--updates 100000] [--seed S]
    python -m safe2.posthoc recall --fit OUT.posthoc.pt --ckpt RUN.ckpt.pt --episodes A-B [--sealed]

* Data: the replay rows whose global episode index `ep` lies in [A-1, B-1] (episodes are 1-based and
  inclusive on the CLI). Rows are read in insertion order (ptr, n as lagu.train.Replay.load). Every
  requested episode must be stored whole (its `t` column is 0, 1, ..., len-1); otherwise the command
  refuses. G0b fits 201-900 and holds out 901-1000; PHS fits 1-900 and holds out 901-1000.
* fit: safe2.bvs.fit_offline (V twin with H = 32 heads and pi_b, exactly SPEC 2.4 steps 2-3: batch 256,
  Adam 3e-4, pi_b and Polyak 0.005 every 2nd update) with the checkpoint's own actor, frozen and
  deterministic, as the proposer (input cat(o2, b2) with budget_input, o2 otherwise). The executed
  action is the `a_exec` column when the run has one (BVS family) and `a_pi` otherwise. Hidden width =
  the run's. Seeds: torch.manual_seed(S), batch indices from default_rng(S), the target-candidate
  generator seeded with S; S defaults to the run seed. The output (torch.save, weights-only loadable)
  is fit_offline's dict plus metadata: ckpt (path), ckpt_sha256, episodes [A, B], rows, env, config,
  run_seed.
* recall (G0b(i)): on the held-out episodes, an alarm at t is vmax(o_t, a_t) > 0.5 with a_t the
  replay's executed action; an entry (cost > 0, no cost in the 12 steps before it, t_e >= 12) is caught
  if an alarm occurs at some t in [t_e - 12, t_e - 7]; a false-alarm step is an alarm with no cost in
  steps t..t+31 of the same episode (truncated at its end). Output: entries, caught, recall,
  false_alarm_steps, episodes, fa_per_episode, nonfinite_pred. A non-finite vmax is never an alarm, so
  any non-finite prediction makes recall and fa_per_episode NaN (a gate on them fails).
  With --sealed (PHS runs, seed >= 300) the output is sealed to <root>/sealed/phs/<stem>.recall.json.enc
  and the console prints only `<stem>: done`; an unsealed recall of a blinded seed is refused.

The Python interface safe2.probe uses (probe.POSTHOC_API): fit, fit_meta, load_fit, recall.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ALARM = 0.5          # an alarm: max_j V^j_H(o, a) > ALARM
LOOKBACK = 12        # an entry: no cost in the 12 steps before it, t_e >= 12
CATCH = (12, 7)      # an entry is caught by an alarm at t in [t_e - 12, t_e - 7]
FA_HORIZON = 32      # a false alarm: no cost in steps t..t+31
CHUNK = 65_536       # rows per vmax batch


# --- inputs -----------------------------------------------------------------------------------------
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_episodes(text) -> tuple[int, int]:
    """"A-B" (or a pair) -> (A, B), 1-based and inclusive, 1 <= A <= B."""
    a, b = (int(x) for x in (text.split("-") if isinstance(text, str) else text))
    if not 1 <= a <= b:
        raise SystemExit(f"bad episode range {text!r} (1-based, A <= B)")
    return a, b


def _np(x) -> np.ndarray:
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def load_ckpt(path: str) -> dict:
    return torch.load(path, weights_only=True, mmap=True)


def select_rows(rep: dict, episodes: tuple[int, int]) -> dict:
    """The replay columns (numpy, insertion order) of the rows with ep in [A-1, B-1]. Refuses when a
    requested episode is not stored whole."""
    A, B = episodes
    n, ptr, size = int(rep["n"]), int(rep["ptr"]), int(_np(rep["o"]).shape[0])
    order = np.arange(n) if n < size else (np.arange(ptr, ptr + size) % size)
    ep = _np(rep["ep"])[order].astype(np.int64)
    keep = order[(ep >= A - 1) & (ep <= B - 1)]
    cols = {k: _np(rep[k])[keep] for k in ("o", "a_pi", "c", "o2", "b2", "d", "ep", "t")}
    if "a_exec" in rep:
        cols["a_exec"] = _np(rep["a_exec"])[keep]
    ep, t = cols["ep"].astype(np.int64), cols["t"].astype(np.int64)
    present, start, counts = np.unique(ep, return_index=True, return_counts=True)
    missing = sorted(set(range(A - 1, B)) - set(present.tolist()))
    if missing:
        raise SystemExit(f"episodes {A}-{B}: {len(missing)} not in the replay (first: {missing[0] + 1})")
    if (np.diff(ep) < 0).any():
        raise SystemExit(f"episodes {A}-{B}: rows of an episode are not contiguous in the replay")
    for e, s, k in zip(present, start, counts):
        if not np.array_equal(t[s:s + k], np.arange(k)):
            raise SystemExit(f"episode {int(e) + 1} is not stored whole in the replay")
    return cols


def executed(cols: dict) -> np.ndarray:
    """The executed action: a_exec (BVS family) or a_pi (base arms, where it is the executed action)."""
    return cols["a_exec"] if "a_exec" in cols else cols["a_pi"]


def proposer(ck: dict):
    """(the checkpoint's actor, frozen; budget_input)."""
    from lagu.agent import Actor
    from safe2.configs import RunConfig
    rc = RunConfig(**ck["config"])
    obs_dim = int(_np(ck["replay"]["o"]).shape[1])
    act_dim = int(_np(ck["replay"]["a_pi"]).shape[1])
    with torch.random.fork_rng():
        actor = Actor(obs_dim + (1 if rc.budget_input else 0), act_dim, rc.hidden)
    actor.load_state_dict(ck["agent"]["actor"])
    actor.requires_grad_(False)
    return actor, bool(rc.budget_input), rc


def _save_atomic(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --- fit ----------------------------------------------------------------------------------------------
def fit(ckpt: str, episodes, out: str, updates: int = 100_000, seed: int | None = None) -> dict:
    """SPEC 7 fit; writes `out` and returns its metadata (fit_meta)."""
    from safe2.bvs import fit_offline
    episodes = parse_episodes(episodes)
    ck = load_ckpt(ckpt)
    seed = int(ck["seed"]) if seed is None else int(seed)
    cols = select_rows(ck["replay"], episodes)
    actor, budget_input, rc = proposer(ck)
    data = {"o": cols["o"], "a": executed(cols), "c": cols["c"], "o2": cols["o2"], "b2": cols["b2"],
            "d": cols["d"]}
    res = fit_offline(data, actor, budget_input, seed, updates=int(updates), hidden=rc.hidden)
    res.update({"ckpt": os.path.abspath(ckpt), "ckpt_sha256": sha256(ckpt), "episodes": list(episodes),
                "rows": int(len(cols["o"])), "env": str(ck.get("env", "")), "config": rc.name,
                "run_seed": int(ck["seed"]) if "seed" in ck else None})
    _save_atomic(res, out)
    return _meta(res)


def _meta(f: dict) -> dict:
    return {"ckpt_sha256": f["ckpt_sha256"], "episodes": list(f["episodes"]), "updates": int(f["updates"]),
            "seed": int(f["seed"])}


def fit_meta(path: str) -> dict:
    """{ckpt_sha256, episodes [A, B], updates, seed} of a saved fit."""
    return _meta(torch.load(path, weights_only=True))


def load_fit(path: str):
    """The fit's V (a safe2.bvs.ViabilityNet, eval mode, no gradient); .vmax(o, a) -> (B,)."""
    from safe2.bvs import ViabilityNet
    f = torch.load(path, weights_only=True)
    with torch.random.fork_rng():
        net = ViabilityNet(int(f["obs_dim"]), int(f["act_dim"]), int(f["H"]), int(f["hidden"]))
    net.load_state_dict(f["V"])
    return net.eval().requires_grad_(False)


# --- recall (G0b(i)) ----------------------------------------------------------------------------------
@torch.no_grad()
def vmax_rows(net, o: np.ndarray, a: np.ndarray) -> np.ndarray:
    out = [net.vmax(torch.as_tensor(o[i:i + CHUNK], dtype=torch.float32),
                    torch.as_tensor(a[i:i + CHUNK], dtype=torch.float32)).numpy()
           for i in range(0, len(o), CHUNK)]
    return np.concatenate(out).astype(float) if out else np.zeros(0)


def episode_counts(c: np.ndarray, v: np.ndarray) -> tuple[int, int, int]:
    """(entries, caught, false-alarm steps) of one episode with step costs c and predictions v."""
    hit = c > 0
    alarm = np.isfinite(v) & (v > ALARM)
    entries = caught = 0
    for te in np.flatnonzero(hit):
        if te >= LOOKBACK and not hit[te - LOOKBACK:te].any():
            entries += 1
            caught += int(alarm[te - CATCH[0]:te - CATCH[1] + 1].any())
    fa = sum(1 for t in np.flatnonzero(alarm) if not hit[t:t + FA_HORIZON].any())
    return entries, caught, fa


def recall_from(c: np.ndarray, ep: np.ndarray, v: np.ndarray) -> dict:
    """G0b(i) counts over the episodes of rows (c, ep, v) in insertion order."""
    entries = caught = fa = 0
    eps = np.unique(ep)
    for e in eps:
        m = ep == e
        n_e, n_c, n_f = episode_counts(c[m], v[m])
        entries, caught, fa = entries + n_e, caught + n_c, fa + n_f
    nonfinite = int((~np.isfinite(v)).sum())
    n_ep = int(len(eps))
    ok = nonfinite == 0
    return {"entries": entries, "caught": caught,
            "recall": caught / entries if (entries and ok) else float("nan"),
            "false_alarm_steps": fa, "episodes": n_ep,
            "fa_per_episode": fa / n_ep if (n_ep and ok) else float("nan"), "nonfinite_pred": nonfinite}


def recall(fit: str, ckpt: str, episodes) -> dict:
    """G0b(i) of a fit on the held-out episodes of a checkpoint's replay."""
    episodes = parse_episodes(episodes)
    cols = select_rows(load_ckpt(ckpt)["replay"], episodes)
    v = vmax_rows(load_fit(fit), cols["o"], executed(cols))
    return recall_from(cols["c"].astype(float), cols["ep"].astype(np.int64), v)


def _stem(ckpt: str) -> str:
    name = os.path.basename(ckpt)
    return name[:-len(".ckpt.pt")] if name.endswith(".ckpt.pt") else os.path.splitext(name)[0]


# --- CLI ----------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit V and pi_b on replay episodes A-B")
    f.add_argument("--ckpt", required=True)
    f.add_argument("--episodes", required=True, help="A-B, 1-based, inclusive")
    f.add_argument("--out", required=True)
    f.add_argument("--updates", type=int, default=100_000)
    f.add_argument("--seed", type=int, default=None, help="default: the run seed")
    r = sub.add_parser("recall", help="entry recall and false alarms (G0b(i)) on episodes A-B")
    r.add_argument("--fit", required=True)
    r.add_argument("--ckpt", required=True)
    r.add_argument("--episodes", required=True, help="A-B, 1-based, inclusive")
    r.add_argument("--sealed", action="store_true", help="seal the output under sealed/phs/ (seeds >= 300)")
    a = p.parse_args(argv)
    stem = _stem(a.ckpt)
    if a.cmd == "fit":
        fit(a.ckpt, a.episodes, a.out, a.updates, a.seed)
        print(f"{stem}: fit done -> {a.out}")
        return 0
    from safe2 import seal
    from safe2.train import blinded
    seed = int(load_ckpt(a.ckpt).get("seed", -1))
    if blinded(seed) and not a.sealed:
        raise SystemExit(f"{stem}: seed {seed} is blinded; recall needs --sealed")
    if a.sealed:
        root = seal.find_root(os.path.dirname(os.path.abspath(a.fit)))
        seal.pub_key(root)                      # refuse before any work without the key
        out = recall(a.fit, a.ckpt, a.episodes)
        out.update({"stem": stem, "fit_sha256": sha256(a.fit), "ckpt_sha256": sha256(a.ckpt),
                    "episodes_range": list(parse_episodes(a.episodes))})
        seal.write_file(root, os.path.join("phs", stem + ".recall.json"), out)
        print(f"{stem}: done")
    else:
        print(json.dumps(recall(a.fit, a.ckpt, a.episodes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
