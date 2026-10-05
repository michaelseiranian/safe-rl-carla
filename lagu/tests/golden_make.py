"""Record SHA-256 hashes of every network after 300 updates per arm (hidden 32, fixed batch
stream, both update branches exercised). test_golden_regression re-computes and compares, so
any change that alters training numerics is caught."""
import hashlib
import json
import sys

import numpy as np
import torch

from lagu.agent import Config, LagU
from lagu.train import ARMS

OBS, ACT = 8, 2


def batch(i, n=64):
    g = torch.Generator().manual_seed(i)
    return (torch.randn(n, OBS, generator=g), torch.rand(n, ACT, generator=g) * 2 - 1,
            torch.randn(n, generator=g), (torch.rand(n, generator=g) < 0.3).float(),
            torch.randn(n, OBS, generator=g), torch.zeros(n))


def run(arm, **extra):
    torch.manual_seed(0)
    cfg = Config(hidden=32, lambda_lr=1e-2, lambda_init=0.1, delta0=0.05, **{**ARMS[arm], **extra})
    ag = LagU(OBS, ACT, cfg, seed=0)
    for i in range(300):
        ag.env_step = 150_000 + i          # past 0.5*Tc: both exploratory and risk-sensitive updates occur
        ag.update(batch(i))
    h = hashlib.sha256()
    for net in (ag.actor, ag.members, ag.cost):
        for t in net.state_dict().values():
            h.update(t.detach().numpy().tobytes())
    return {"hash": h.hexdigest(), "lambda": ag.lmbda}


CASES = {arm: {} for arm in ARMS}
CASES["lagu_gategrad"] = {"_arm": "lagu", "gate_grad": True}

if __name__ == "__main__":
    out = {k: run(v.pop("_arm", k), **v) for k, v in {k: dict(v) for k, v in CASES.items()}.items()}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out, indent=1))
