# safe-rl-carla\safe_rl\utils\replay_buffer.py:
import numpy as np
import torch
from safe_rl.env.carla_cmdp_env import MAX_ACTORS, ACTOR_FEATURES

class ReplayBuffer:
    """
    Stores vector-only transitions:
      (s_spd, s_vec, s_actr, a, r, c, s2_spd, s2_vec, s2_actr, done)
    where action `a` is [steer, accel].
    Rewards & costs are *scaled* on write so the learner sees stable magnitudes.
    Set `scale=1.0` to disable.
    """
    def __init__(self, obs_shape, action_dim, size=int(1e6),
                 device="cpu", scale: float = 1.0, vec_dim:int = 4):
        # image buffers are unused in vector‑only mode
        self.obs_buf  = None
        self.next_buf = None
        self.act_buf  = np.zeros((size, action_dim), dtype=np.float32)
        self.rew_buf  = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.bool_)
        self.spd_buf  = np.zeros(size, dtype=np.float32)
        self.spd2_buf = np.zeros_like(self.spd_buf)
        self.vec_buf     = np.zeros((size, vec_dim), dtype=np.float32)
        self.vec2_buf    = np.zeros_like(self.vec_buf)
        # NEW – actor tensors (flattened)
        self.actr_buf    = np.zeros((size, MAX_ACTORS * ACTOR_FEATURES), dtype=np.float32)
        self.actr2_buf   = np.zeros_like(self.actr_buf)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device
        self.scale = scale

    def store(self, s_spd, s_vec, s_actr, a, r, c, s2_spd, s2_vec, s2_actr, d):
        """
        Stores a transition.  `s_spd` and `s2_spd` are **scalar floats**
        here; they become (B,1) tensors only at sample-time.
        """
        if self.ptr == 0:
            print("[debug] first transition stored", flush=True)
        self.spd_buf[self.ptr]   = s_spd
        self.spd2_buf[self.ptr]  = s2_spd
        self.vec_buf [self.ptr]  = s_vec
        self.vec2_buf[self.ptr]  = s2_vec
        self.actr_buf[self.ptr]  = s_actr.reshape(-1)
        self.actr2_buf[self.ptr] = s2_actr.reshape(-1)
        self.act_buf[self.ptr]  = a
        # Scale **once on write** so Q‑values stay in ±10…20.
        self.rew_buf[self.ptr]  = r * self.scale
        self.cost_buf[self.ptr] = c * self.scale
        self.done_buf[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch):
        idxs = np.random.randint(0, self.size, batch)
        device = self.device
        a  = torch.as_tensor(self.act_buf[idxs],  dtype=torch.float32, device=device)
        # Already stored in scaled form – no second division
        r  = torch.as_tensor(self.rew_buf[idxs],  dtype=torch.float32, device=device)
        c  = torch.as_tensor(self.cost_buf[idxs], dtype=torch.float32, device=device)
        d  = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32, device=device)
        spd  = torch.as_tensor(self.spd_buf[idxs],  dtype=torch.float32, device=device).unsqueeze(-1)          # (B,1)
        spd2 = torch.as_tensor(self.spd2_buf[idxs], dtype=torch.float32, device=device).unsqueeze(-1)
        vec   = torch.as_tensor(self.vec_buf [idxs], dtype=torch.float32, device=device)
        vec2  = torch.as_tensor(self.vec2_buf[idxs], dtype=torch.float32, device=device)
        actr  = torch.as_tensor(self.actr_buf[idxs],  dtype=torch.float32, device=device)
        actr2 = torch.as_tensor(self.actr2_buf[idxs], dtype=torch.float32, device=device)
        return spd, vec, actr, a, r, c, spd2, vec2, actr2, d


# --------------------------------------------------------------------- #
#  NEW: helper to wipe the buffer between curriculum stages             #
# --------------------------------------------------------------------- #
    def clear(self):
        """Fast reset without reallocating arrays."""
        self.ptr  = 0
        self.size = 0


    # External collector helper (Ray workers push dicts).
    # helper to write transitions coming from remote workers (vector-only)
    def push_from_worker(self, pkg: dict):
        """
        Accept a transition dict produced by a worker.
        Expected keys (vector-only): s_spd, s_vec, s_actr, a, r, c, s2_spd, s2_vec, s2_actr, done
        """
        self.store(pkg["s_spd"],  pkg["s_vec"],  pkg["s_actr"],
                   pkg["a"],      pkg["r"],      pkg["c"],
                   pkg["s2_spd"], pkg["s2_vec"], pkg["s2_actr"],
                   pkg["done"])
