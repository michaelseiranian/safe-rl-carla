# safe-rl-carla\safe_rl\algorithms\lag.py:
from dataclasses import dataclass
from .lagu import LagUAgent as LagAgent
from .lagu import LagUHyper as _LagUHyper

@dataclass
class LagHyper(_LagUHyper):
    """
    Wrapper hyper-params for the fixed-budget Lagrangian baseline.
    Only difference from LagUHyper is the default `baseline` value.
    """
    baseline: str = "lag"

# Reuse the exact implementation; behavior is gated by `cfg.baseline`.
# Public names kept as LagAgent / LagHyper to match eval script expectations.
