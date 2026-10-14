"""Shared test helpers: a deterministic fake Safety-Gymnasium environment, synthetic batches, network
hashes and a temporary sealing key."""
from __future__ import annotations

import hashlib
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

OBS, ACT = 8, 2


def batch(i: int, n: int = 64, obs: int = OBS, act: int = ACT, term: bool = False):
    """Synthetic batch i of the fixed stream (as lagu/tests/golden_make.py), as numpy arrays."""
    g = torch.Generator().manual_seed(i)
    d = (torch.rand(n, generator=g) < 0.1).float() if term else torch.zeros(n)
    return (torch.randn(n, obs, generator=g).numpy(), (torch.rand(n, act, generator=g) * 2 - 1).numpy(),
            torch.randn(n, generator=g).numpy(), (torch.rand(n, generator=g) < 0.3).float().numpy(),
            torch.randn(n, obs, generator=g).numpy(), d.numpy())


def net_hash(*nets) -> str:
    h = hashlib.sha256()
    for net in nets:
        for t in net.state_dict().values():
            h.update(t.detach().numpy().tobytes())
    return h.hexdigest()


def same(a, b) -> bool:
    """Exact recursive equality of nested dicts / lists / tensors / arrays / scalars."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and \
            a.shape == b.shape and bool(torch.equal(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True                                         # NaN == NaN (e.g. rho_star)
    return a == b


class FakeEnv:
    """Deterministic episodic environment with the Safety-Gymnasium step signature.

    Episodes last `length` steps (truncation). The observation is (episode index, t, 1, 0, ...);
    the reward is `reward` and the cost of step t is cost_fn(episode, t) (ep_in_obs=False drops the
    episode index from the observation, so a fresh instance replays the same observations). Every reset seed is
    recorded in `self.seeds`; the dynamics ignore seeds and actions, so a run is a deterministic
    function of the agent's own RNG streams."""

    def __init__(self, length: int = 10, obs_dim: int = OBS, reward: float = 0.5, cost_fn=None,
                 term_at: int | None = None, ep_in_obs: bool = True):
        self.length, self.obs_dim, self.reward, self.ep_in_obs = length, obs_dim, reward, ep_in_obs
        self.cost_fn = cost_fn or (lambda ep, t: 0.0)
        self.term_at = term_at
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT,), np.float64)
        self.spec = SimpleNamespace(max_episode_steps=length)
        self.seeds: list = []
        self.actions: list = []
        self.ep, self.t = -1, 0

    def _obs(self):
        o = np.zeros(self.obs_dim)
        o[0], o[1], o[2] = (self.ep % 7 if self.ep_in_obs else 0.0), self.t / self.length, 1.0
        return o

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.ep += 1
        self.t = 0
        return self._obs(), {}

    def step(self, a):
        self.actions.append(np.array(a, copy=True))
        c = float(self.cost_fn(self.ep, self.t))
        self.t += 1
        term = self.term_at is not None and self.t == self.term_at
        trunc = self.t >= self.length
        return self._obs(), self.reward, c, term, trunc and not term, {}

    def close(self):
        pass


def make_key(root) -> str:
    """A temporary 2048-bit RSA key pair: <root>/keys/seal_pub.pem and the private key path."""
    from safe2 import seal
    root = str(root)
    os.makedirs(os.path.join(root, "keys"), exist_ok=True)
    priv = os.path.join(root, "test_priv.pem")
    subprocess.run([seal.openssl(), "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048",
                    "-out", priv], check=True, capture_output=True)
    subprocess.run([seal.openssl(), "pkey", "-in", priv, "-pubout", "-out",
                    os.path.join(root, "keys", "seal_pub.pem")], check=True, capture_output=True)
    return priv
