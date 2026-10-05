"""Where can Lag-U's gate act? (Proposition 1 in the paper.) With delta_ada detached, the gate is a
constant in the actor loss, so under a fixed or realised-cost dual the gated and ungated agents
train bit-identically; under the paper's critic-unit dual (Eq. 18) it acts only through lambda,
exactly like a uniform budget equal to the batch mean of delta_ada."""
import json
import os

import numpy as np
import pytest
import torch

from lagu.agent import Config, LagU
from lagu.train import ARMS

OBS, ACT = 8, 2
HERE = os.path.dirname(__file__)


def batch(i, n=64):
    g = torch.Generator().manual_seed(i)
    return (torch.randn(n, OBS, generator=g), torch.rand(n, ACT, generator=g) * 2 - 1,
            torch.randn(n, generator=g), (torch.rand(n, generator=g) < 0.3).float(),
            torch.randn(n, OBS, generator=g), torch.zeros(n))


def make(arm, **kw):
    torch.manual_seed(0)
    return LagU(OBS, ACT, Config(hidden=32, **{**ARMS[arm], **kw}), seed=0)


def train(ag, n=300, jc_every=0):
    for i in range(n):
        ag.env_step = 150_000 + i                 # c(x) in (0.1, 1): both update branches occur
        ag.update(batch(i))
        if jc_every and i % jc_every == 0:
            ag.update_dual_episodic(40.0 + 10 * np.sin(i))
    return ag


def tensors(ag):
    out = {}
    for name in ("actor", "actor_targ", "members", "members_targ", "cost", "cost_targ"):
        for k, v in getattr(ag, name).state_dict().items():
            out[f"{name}.{k}"] = v
    for name in ("pi_opt", "q_opt", "c_opt"):
        for pid, st in getattr(ag, name).state_dict()["state"].items():
            for k, v in st.items():
                out[f"{name}.{pid}.{k}"] = v
    return out


def identical(a, b):
    ta, tb = tensors(a), tensors(b)
    return ta.keys() == tb.keys() and all(torch.equal(ta[k], tb[k]) for k in ta)


@pytest.mark.parametrize("dual,kw,jc", [("fixed", dict(lambda_init=0.3), 0),
                                         ("episodic", dict(lambda_init=0.1, lambda_lr=0.05), 7)])
def test_gate_is_inert_under_fixed_and_realised_cost_duals(dual, kw, jc):
    g = train(make("lagu", dual=dual, T=0.02, **kw), jc_every=jc)
    n = train(make("lagu_nogate", dual=dual, T=0.02, **kw), jc_every=jc)
    assert g.gate_hits > 0 and g.lmbda > 0          # liveness: the gate fired and the penalty is on
    assert identical(g, n)
    assert g.lmbda == n.lmbda


def test_positive_controls_differ():
    sw_g = train(make("lagu", lambda_init=0.1, lambda_lr=1e-2, delta0=0.05, T=0.02))
    sw_n = train(make("lagu_nogate", lambda_init=0.1, lambda_lr=1e-2, delta0=0.05, T=0.02))
    assert not identical(sw_g, sw_n) and sw_g.lmbda != sw_n.lmbda
    gg = train(make("lagu", dual="fixed", lambda_init=0.3, T=0.02, gate_grad=True))
    ng = train(make("lagu_nogate", dual="fixed", lambda_init=0.3, T=0.02))
    assert not identical(gg, ng)


def test_analytic_gate_gradient():
    """grad(gate_grad) - grad(detached) = grad of -lambda * mean(delta_ada(std/|Q_bar|.detach()))."""
    def actor_grad(gate_grad):
        ag = make("lagu", dual="fixed", lambda_init=0.7, T=1e-3, gate_grad=gate_grad, c0=0.0, Tc=1)
        ag.env_step = 10
        o = batch(3)[0]
        a = ag.actor(o)
        qbar, var = ag.ensemble(o, a)
        std = torch.sqrt(var + 1e-12)
        qc = ag.qc_pi(o, a)
        ratio = std / (qbar.detach().abs() + ag.cfg.ratio_eps)
        delta = ag.adaptive_delta(ratio if gate_grad else ratio.detach())
        loss = (-qbar + ag.lmbda * (qc - delta)).mean()
        grads = torch.autograd.grad(loss, list(ag.actor.parameters()), retain_graph=True)
        extra = torch.autograd.grad((-ag.lmbda * ag.adaptive_delta(ratio)).mean(), list(ag.actor.parameters()))
        return grads, extra
    g1, extra = actor_grad(True)
    g0, _ = actor_grad(False)
    diff = [a - b for a, b in zip(g1, g0)]
    assert all(torch.allclose(d, e, atol=1e-6) for d, e in zip(diff, extra))
    assert any(float(e.abs().max()) > 0 for e in extra)


def test_eq18_gate_equals_a_uniform_budget_at_its_batch_mean():
    ag = make("lagu", lambda_init=0.2, lambda_lr=1e-2, T=1e-3, c0=0.0, Tc=1)
    ag.env_step = 10
    o = batch(5)[0]
    lam0 = ag.lmbda
    stats = ag._policy_update(o)
    d_used = stats["delta_mean"]
    assert d_used < ag.cfg.delta0                   # the gate tightened this update
    ref = make("lagu_nogate", lambda_init=0.2, lambda_lr=1e-2, T=1e-3, c0=0.0, Tc=1, delta0=d_used)
    ref.env_step = 10
    ref._policy_update(o)
    assert abs((ag.lmbda - lam0) - (ref.lmbda - lam0)) < 1e-6


def test_shadow_lambda():
    n = train(make("lagu_nogate", lambda_init=0.1, lambda_lr=1e-2, delta0=0.05))
    assert abs(n.lmbda_shadow - n.lmbda) < 1e-8
    g = train(make("lagu", lambda_init=0.1, lambda_lr=1e-2, delta0=0.05, T=0.02))
    assert g.lmbda_shadow <= g.lmbda + 1e-8
    m = g.interval_means()
    assert np.isfinite(m["gap_avg"]) and 0.0 <= m["would_engage"] <= 1.0 and m["n_pi"] > 0


def test_would_engage_is_measured_without_the_gate():
    n = train(make("lagu_nogate", T=0.02))
    assert n.gate_total == 0 and n.acc["would_total"] > 0


def test_golden_regression():
    from lagu.tests.golden_make import CASES, run
    gold = json.load(open(os.path.join(HERE, "golden", "golden.json")))
    for k, v in {k: dict(v) for k, v in CASES.items()}.items():
        assert run(v.pop("_arm", k), **v)["hash"] == gold[k]["hash"], k
