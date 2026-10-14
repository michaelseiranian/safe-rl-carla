"""A stand-in for safe2.posthoc with the Python interface safe2.probe.g0b uses (probe.POSTHOC_API), for
tests and dry runs only. fit() records its arguments (and the checkpoint's SHA-256, for fit_meta) instead
of fitting; recall() returns fixed counts
that depend on the run seed; load_fit() returns an object whose torch vmax (as safe2.bvs.ViabilityNet's)
is 32 x the largest hazard-lidar bin of o (PointGoal1's flat observation holds hazards_lidar at 28:44), so
a dry run's ranking has a non-trivial signal."""
from __future__ import annotations

import hashlib

import torch

LIDAR = slice(28, 44)


def fit(ckpt: str, episodes: tuple, out: str, updates: int = 100_000, seed: int = 0) -> dict:
    with open(ckpt, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    meta = {"ckpt": ckpt, "ckpt_sha256": digest, "episodes": list(episodes), "updates": int(updates),
            "seed": int(seed)}
    torch.save(meta, out)
    return meta


def fit_meta(path: str) -> dict:
    m = torch.load(path, weights_only=True)
    return {k: m[k] for k in ("ckpt_sha256", "episodes", "updates", "seed")}


def recall(fit: str, ckpt: str, episodes: tuple) -> dict:
    meta = torch.load(fit, weights_only=True)
    n_ep = int(episodes[1]) - int(episodes[0]) + 1
    entries, caught, fa = 10, 8 + meta["seed"] % 2, 3.0 * (meta["seed"] % 3)
    return {"entries": entries, "caught": caught, "recall": caught / entries, "false_alarm_steps": fa,
            "episodes": n_ep, "fa_per_episode": fa / n_ep, "fit_episodes": meta["episodes"]}


class _Fit:
    def __init__(self, meta: dict):
        self.meta = meta

    def vmax(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        assert o.dtype == a.dtype == torch.float32 and o.ndim == a.ndim == 2 and len(o) == len(a)
        return 32.0 * o[:, LIDAR].max(1).values


def load_fit(path: str) -> _Fit:
    return _Fit(torch.load(path, weights_only=True))
