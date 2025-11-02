# safe-rl-carla\safe_rl\algorithms\td3.py
from dataclasses import dataclass
from .lagu import LagUAgent as TD3Agent
from .lagu import LagUHyper as _LagUHyper

@dataclass
class TD3Hyper(_LagUHyper):
    """
    Wrapper hyper-params for the reward-only TD3 baseline.
    Only difference from LagUHyper is the default `baseline` value.
    """
    baseline: str = "td3"

# Reuse the exact implementation; behavior is gated by `cfg.baseline`.
# Public names kept as TD3Agent / TD3Hyper to match eval script expectations.
