"""safe2.base and safe2.configs (SPEC section 10, tests 1-11): the golden equivalences with
lagu.agent.LagU under the omnisafe recipe, SHL's cost discount and rate normalisation, the PID dual,
the dual schedule and the registry's config differences."""
from __future__ import annotations

import io
from collections import deque
from dataclasses import asdict

import numpy as np
import pytest
import torch

from lagu.agent import Config, LagU
from lagu.train import ARMS, RECIPES
from safe2 import configs
from safe2.base import Agent, PIDLagrangian, lagu_config
from safe2.tests.helpers import ACT, OBS, batch, net_hash

HORIZON = 1000


def lagu_omnisafe(cost_clamp: bool, hidden: int = 32, **extra) -> Config:
    """The Config lagu.train.main builds for --arm td3lag --recipe omnisafe [--cost-clamp 1]."""
    rec = RECIPES["omnisafe"]
    cfg = Config(hidden=hidden, gamma=0.99, delta0=25.0 / HORIZON / (1 - 0.99), dual=rec["dual"], cost_limit=25.0,
                 gate_grad=False, cost_twin=False, cost_clamp=cost_clamp, **ARMS["td3lag"])
    for k in ("lambda_lr", "actor_lr", "critic_lr", "max_grad_norm"):
        setattr(cfg, k, rec[k])
    for k in ("pi_norm", "cost_target_online"):
        setattr(cfg, k, bool(rec[k]))
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def jc_script(i: int) -> float:
    return 25.0 + 20.0 * np.sin(0.1 * i)


def run_lagu(cfg: Config, n: int = 300):
    torch.manual_seed(0)
    ag = LagU(OBS, ACT, cfg, seed=0)
    for i in range(n):
        ag.env_step = 150_000 + i
        ag.update(batch(i, term=True))
        ag.update_dual_episodic(jc_script(i))
    return ag


def run_safe2(rc: configs.RunConfig, n: int = 300, dual: bool = True, jc=jc_script):
    torch.manual_seed(0)
    ag = Agent(OBS, ACT, rc, horizon=HORIZON, seed=0)
    for i in range(n):
        ag.env_step = 150_000 + i
        ag.update(batch(i, term=True))
        if dual:
            ag.dual_step(jc(i))
    return ag


@pytest.mark.parametrize("name,clamp", [("R1", False), ("B1", True)])
@pytest.mark.parametrize("lambda_lr", [None, 1e-2])
def test_golden_R1_and_B1_reproduce_lagu_omnisafe(name, clamp, lambda_lr):
    """Tests 1-2: R1 (no clamp) and B1 (clamp) equal LagU built with --recipe omnisafe, after 300 updates
    with a scripted episodic dual every update (at the recipe's 5e-7 and at 1e-2, where lambda moves)."""
    extra = {} if lambda_lr is None else {"lambda_lr": lambda_lr}
    ref = run_lagu(lagu_omnisafe(clamp, **extra))
    rc = configs.make(name, hidden=32, **extra)
    if name == "B1":           # B1 has a lambda cap of 100; lagu's recipe has none. Lambda stays far below it.
        assert rc.lambda_max == 100.0
    ag = run_safe2(rc)
    assert ref.lmbda > 0.0
    assert net_hash(ag.actor, ag.members, ag.cost) == net_hash(ref.actor, ref.members, ref.cost)
    assert net_hash(ag.actor_targ, ag.members_targ, ag.cost_targ) == net_hash(ref.actor_targ, ref.members_targ,
                                                                              ref.cost_targ)
    assert ag.lmbda == ref.lmbda
    assert ag.rng.bit_generator.state == ref.rng.bit_generator.state
    assert ag.cfg == ref.cfg if name == "R1" else asdict(ag.cfg) == {**asdict(ref.cfg), "lambda_max": 100.0}


def test_learn_from_replay_columns_equals_update_on_the_tuple():
    rc = configs.make("B5", hidden=32)
    torch.manual_seed(0)
    a = Agent(OBS, ACT, rc, HORIZON, seed=0)
    torch.manual_seed(0)
    b = Agent(OBS, ACT, rc, HORIZON, seed=0)
    for i in range(20):
        o, act, r, c, o2, d = batch(i)
        torch.manual_seed(i)
        a.update((o, act, r, c, o2, d))
        torch.manual_seed(i)
        b.learn({"o": o, "a_pi": act, "r": r, "c": c, "o2": o2, "d": d}, step=i)
    assert net_hash(a.actor, a.members, a.cost) == net_hash(b.actor, b.members, b.cost)


@pytest.mark.parametrize("gains", ["B4", "B5"])
def test_shl_with_gamma_c_equal_gamma_is_bit_identical_to_its_gain_control(gains):
    """Test 3: SHL with cost_gamma = gamma and rate_norm off is B4 / B5 bit for bit (PID dual moving)."""
    shl = configs.make(f"SHL-g0.90-{gains}", hidden=32, cost_gamma=0.99, rate_norm=False, kp=0.5, ki=0.01)
    ref = configs.make(gains, hidden=32, kp=0.5, ki=0.01)
    above = lambda i: 40.0 + 10.0 * np.sin(0.1 * i)              # J above the target: lambda > 0
    a, b = run_safe2(shl, jc=above), run_safe2(ref, jc=above)
    assert a.lmbda == b.lmbda > 0.0
    assert net_hash(a.actor, a.members, a.cost) == net_hash(b.actor, b.members, b.cost)
    c = run_safe2(configs.make(f"SHL-g0.90-{gains}", hidden=32, kp=0.5, ki=0.01), jc=above)   # real SHL differs
    assert net_hash(c.actor, c.cost) != net_hash(b.actor, b.cost)


@pytest.mark.parametrize("gc", [0.90, 0.95])
def test_shl_cost_target_and_clamp_on_a_fixed_batch(gc):
    """Test 4: the cost loss of one update equals mse(Q_c(o, a), clamp(c + gc (1-d) Q_c^targ(o2, pi(o2)),
    0, 1/(1-gc))), with the clamp binding on part of the batch."""
    torch.manual_seed(0)
    ag = Agent(OBS, ACT, configs.make(f"SHL-g{gc:.2f}-B5", hidden=32), HORIZON, seed=0)
    with torch.no_grad():                         # Q_c^targ near 1/(1-gc) - 1: the clamp binds when c = 3
        ag.cost_targ[-1].bias.fill_(1.0 / (1.0 - gc) - 1.0)
    o, a, r, c, o2, d = batch(3, term=True)
    c = np.where(np.arange(len(c)) % 2 == 0, 3.0, 0.0).astype(np.float32)
    T = lambda x: torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        raw = T(c) + gc * (1 - T(d)) * ag.qc_target(T(o2), ag.actor(T(o2)))
        want = raw.clamp(0.0, 1.0 / (1.0 - gc))
        assert (raw > 1.0 / (1.0 - gc)).any() and (raw <= 1.0 / (1.0 - gc)).any()
        loss_want = float(torch.nn.functional.mse_loss(ag.cost(torch.cat([T(o), T(a)], -1)).squeeze(-1), want))
        loss_noclamp = float(torch.nn.functional.mse_loss(ag.cost(torch.cat([T(o), T(a)], -1)).squeeze(-1), raw))
    stats = ag.update((o, a, r, c, o2, d))
    assert stats["c_loss"] == pytest.approx(loss_want, rel=1e-6)
    assert stats["c_loss"] != pytest.approx(loss_noclamp, rel=1e-3)


def _sgd_probe(ag: Agent, o: torch.Tensor, lr: float) -> dict:
    """Run one _policy_update with plain SGD at lr; return the parameter change and the loss."""
    ag.pi_opt = torch.optim.SGD(ag.actor.parameters(), lr=lr)
    before = [p.detach().clone() for p in ag.actor.parameters()]
    stats = ag._policy_update(o)
    return {"delta": [p.detach() - q for p, q in zip(ag.actor.parameters(), before)], "loss": stats["pi_loss"]}


def test_shl_actor_gradient_is_the_baseline_gradient_with_qc_scaled_by_s():
    """Test 5: d/dtheta of SHL's actor loss (finite differences) equals the gradient of the baseline's
    loss with Q_c replaced by s * Q_c (s = (1 - 0.9) / (1 - 0.99) = 10)."""
    torch.manual_seed(0)
    shl = Agent(OBS, ACT, configs.make("SHL-g0.90-B5", hidden=16, max_grad_norm=0.0), HORIZON, seed=0)
    base = Agent(OBS, ACT, configs.make("B5", hidden=16, max_grad_norm=0.0), HORIZON, seed=0)
    base.actor.load_state_dict(shl.actor.state_dict())
    base.members.load_state_dict(shl.members.state_dict())
    base.cost.load_state_dict(shl.cost.state_dict())
    s = shl.rate_scale
    assert s == pytest.approx(10.0)
    with torch.no_grad():                       # baseline's Q_c := s * Q_c
        base.cost[-1].weight.mul_(s)
        base.cost[-1].bias.mul_(s)
    shl.lmbda = base.lmbda = 0.7
    o = torch.as_tensor(batch(5, term=True)[0])
    g_base = [-x for x in _sgd_probe(base, o, lr=1.0)["delta"]]          # SGD lr 1: delta = -grad
    p = next(shl.actor.parameters())
    h = 1e-2
    idx = np.unravel_index(int(g_base[0].abs().argmax()), g_base[0].shape)
    with torch.no_grad():
        p[idx] += h
    up = _sgd_probe(shl, o, lr=0.0)["loss"]
    with torch.no_grad():
        p[idx] -= 2 * h
    down = _sgd_probe(shl, o, lr=0.0)["loss"]
    with torch.no_grad():
        p[idx] += h
    fd = (up - down) / (2 * h)
    assert fd == pytest.approx(float(g_base[0][idx]), rel=2e-2, abs=1e-4)
    g_shl = [-x for x in _sgd_probe(shl, o, lr=1.0)["delta"]]
    for x, y in zip(g_shl, g_base):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("gc", [0.90, 0.95, 0.99])
def test_level_identity_on_a_constant_cost_rate_chain(gc):
    """Test 6: on a 10-state cycle with cost 1 in one state (rate 0.1), (1 - gc) * mean_s Q_c(s, pi(s))
    is within 5% of the rate after training the cost critic with the agent's own update."""
    n = 10

    def chain(rng, k=128):
        s = rng.integers(0, n, k)
        eye = np.eye(n, dtype=np.float32)
        return (eye[s], rng.uniform(-1, 1, (k, ACT)).astype(np.float32), np.zeros(k, np.float32),
                (s == 0).astype(np.float32), eye[(s + 1) % n], np.zeros(k, np.float32))
    torch.manual_seed(0)
    ag = Agent(n, ACT, configs.make("SHL-g0.90-B4", cost_gamma=gc, hidden=32, critic_lr=3e-3, tau=0.5,
                                    cost_clamp=False), HORIZON, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(4000):
        ag.update(chain(rng))
    with torch.no_grad():
        o = torch.eye(n)
        level = (1 - gc) * float(ag.qc_pi(o, ag.actor(o)).mean())
    assert level == pytest.approx(0.1, rel=0.05)


def _pid_reference(js, kp, ki, kd, delay, ep, ed, cap, target):
    """Independent implementation with an explicit history list (no deque)."""
    I = dp = cd = 0.0
    hist = [0.0]
    out = []
    for J in js:
        delta = J - target
        I = max(0.0, I + ki * delta)
        dp = ep * dp + (1 - ep) * delta
        cd = ed * cd + (1 - ed) * J
        D = max(0.0, cd - hist[max(0, len(hist) - delay)])
        lam = max(0.0, kp * dp + I + kd * D)
        out.append(min(lam, cap) if cap > 0 else lam)
        hist.append(cd)
    return out, I


def test_pid_matches_a_hand_computed_30_step_sequence():
    """Test 7: integral, EMA P, delayed D, anti-windup at 0 and the cap."""
    kw = dict(kp=0.5, ki=0.1, kd=0.5, d_delay=3, ema_p=0.9, ema_d=0.8, lambda_max=4.0, target=25.0)
    js = [10.0] * 10 + [30.0 + 5 * i for i in range(10)] + [200.0] * 10
    pid = PIDLagrangian(**kw)
    got = []
    for k, J in enumerate(js):
        got.append(pid.update(J))
        if k == 3:
            D4 = pid.last_D
    # by hand. step 1: delta=-15, I=max(0,-1.5)=0, P_ema=-1.5, cost_d=2, D=2-0 -> -0.75+0.5*2 = 0.25
    assert got[0] == pytest.approx(0.25, abs=1e-12)
    # step 2: P_ema=0.9*-1.5-1.5=-2.85, cost_d=0.8*2+2=3.6, D=3.6-0 -> -1.425+0.5*3.6 = 0.375
    assert got[1] == pytest.approx(0.375, abs=1e-12)
    # step 4: the delay line holds [2, 3.6, 4.88]: D reads 2 (three updates back), not 0 or 4.88;
    # P_ema=-5.1585, cost_d=5.904 -> 0.5*-5.1585 + 0.5*3.904 < 0 -> lambda floored at 0
    cd4 = 0.8 * (0.8 * 3.6 + 2) + 2
    assert D4 == pytest.approx(cd4 - 2.0, abs=1e-12) and got[3] == 0.0
    assert got[-1] == 4.0                                     # the cap binds at J = 200
    want, I_end = _pid_reference(js, 0.5, 0.1, 0.5, 3, 0.9, 0.8, 4.0, 25.0)
    assert got == pytest.approx(want, abs=1e-12)
    assert pid.I == pytest.approx(I_end)
    # anti-windup: during the first 10 steps (J < target) the integral never goes below 0
    pid2 = PIDLagrangian(**{**kw, "kd": 0.0, "kp": 0.0})
    for J in js[:10]:
        pid2.update(J)
        assert pid2.I == 0.0
    assert pid2.update(35.0) == pytest.approx(0.1 * 10.0)     # recovers at once: no accumulated debt
    nocap = PIDLagrangian(**{**kw, "lambda_max": 0.0})
    assert max(nocap.update(J) for J in js) > 4.0


def test_pid_state_round_trip():
    kw = dict(kp=0.5, ki=0.1, kd=2.0, d_delay=3, ema_p=0.9, ema_d=0.8, lambda_max=4.0, target=25.0)
    a, b = PIDLagrangian(**kw), PIDLagrangian(**kw)
    for J in (10, 40, 60, 5, 80):
        a.update(J)
    b.load_state_dict(a.state_dict())
    assert [a.update(J) for J in (30, 90, 12)] == [b.update(J) for J in (30, 90, 12)]


@pytest.mark.parametrize("name,start", [("B5", 200_000), ("B1", 200_000), ("B8", 50_000), ("B7c", 50_000),
                                        ("R1", 202_000)])
def test_dual_is_off_before_dual_start(name, start):
    """Test 8: post_update leaves lambda at 0 before dual_start, acts at dual_start."""
    rc = configs.make(name, hidden=16, lambda_lr=0.1) if configs.make(name).dual == "adam" else \
        configs.make(name, hidden=16)
    assert rc.dual_start == start
    ag = Agent(OBS, ACT, rc, HORIZON)
    jc = deque([80.0] * 5, maxlen=50)
    for step in range(start - 100, start):
        ag.post_update(step, jc)
    assert ag.lmbda == 0.0 and (ag.pid is None or ag.pid.I == 0.0)
    ag.post_update(start, jc)
    assert ag.lmbda > 0.0
    ag2 = Agent(OBS, ACT, rc, HORIZON)
    ag2.post_update(start + 1, deque(maxlen=50))       # empty window: no dual step
    assert ag2.lmbda == 0.0 and (ag2.pid is None or ag2.pid.cost_d == 0.0)


def test_b9_targets_20_and_none_dual_never_moves():
    b9, b5 = configs.make("B9", hidden=16), configs.make("B5", hidden=16)
    assert b9.dual_target == 20.0 and b5.dual_target == 25.0
    a9, a5 = Agent(OBS, ACT, b9, HORIZON), Agent(OBS, ACT, b5, HORIZON)
    assert a9.pid.target == 20.0 and a9.cfg.cost_limit == 20.0
    jc = deque([22.0] * 10, maxlen=50)                 # between the two targets
    for step in range(200_000, 200_050):
        a9.post_update(step, jc)
        a5.post_update(step, jc)
    assert a9.lmbda > 0.0 and a5.lmbda == 0.0
    for name in ("R2", "SauteA", "SauteB"):
        ag = Agent(OBS, ACT, configs.make(name, hidden=16), HORIZON)
        ag.post_update(500_000, deque([500.0], maxlen=50))
        assert ag.lmbda == 0.0 and ag.pid is None and "lambda" not in ag.log_row()


def test_agent_state_dict_round_trip_restores_the_pid_state():
    """Test 10 (agent level; the trainer-level resume is in test_train.py)."""
    rc = configs.make("B5", hidden=16, kp=0.5, ki=0.01)
    a = run_safe2(rc, n=20)
    b = Agent(OBS, ACT, rc, HORIZON, seed=123)
    buf = io.BytesIO()                                  # through torch.save / torch.load, as a checkpoint
    torch.save(a.state_dict(), buf)
    buf.seek(0)
    b.load_state_dict(torch.load(buf, weights_only=False))
    assert b.pid.state_dict() == a.pid.state_dict() and b.lmbda == a.lmbda
    for i in range(20, 30):
        for ag in (a, b):
            torch.manual_seed(i)
            ag.update(batch(i))
            ag.dual_step(jc_script(i))
    assert net_hash(a.actor, a.cost) == net_hash(b.actor, b.cost) and a.lmbda == b.lmbda


def test_config_diffs():
    """Test 11: each derived config differs from its parent only where the protocol says."""
    d = lambda x, y: set(configs.config_diff(asdict(configs.make(x)), asdict(configs.make(y)))) - {"name"}
    lit = configs.RunConfig(**configs.LIT)
    b7 = set(configs.config_diff(asdict(configs.make("B7")), asdict(lit))) - {"name"}
    assert b7 == {"pi_norm", "cost_clamp", "dual", "kp", "ki"}           # kd = 0 is the default
    assert d("B7b", "B7") == {"ki"}                                       # B4 vs B5 gains: kp equal
    assert d("B7c", "B7") == {"dual_start"}
    assert d("B8", "B5") == {"dual_start"} and configs.make("B8").dual_start == 50_000
    assert d("B9", "B5") == {"dual_target"} and configs.make("B9").dual_target == 20.0
    assert d("R1", "B1") == {"cost_clamp", "dual_start", "lambda_max"}
    assert d("B4", "B5") == {"ki"} and d("B6", "B5") == {"kp"}
    assert d("SauteB", "SauteA") == {"saute_n"} and d("R2", "SauteA") == {"budget_input", "saute_n"}
    for g in ("0.90", "0.95"):
        for gains in ("B4", "B5"):
            assert d(f"SHL-g{g}-{gains}", gains) == {"cost_gamma", "rate_norm"}
    assert d("BVS-SQ-k0.1", "BVS-SQ-k0.02") == {"kappa"} and d("BVS-SN-k0.1", "BVS-SQ-k0.1") == {"rule"}
    assert len(configs.BASELINE_POOL) == 13 and set(configs.BASELINE_POOL) <= set(configs.CONFIGS)
    om = configs.make("B1")
    assert (om.actor_lr, om.critic_lr, om.max_grad_norm, om.pi_norm, om.cost_target_online, om.cost_clamp,
            om.start_steps, om.replay) == (5e-6, 1e-3, 40.0, True, True, True, 25_000, 1_000_000)
    lit_b = configs.make("R2")
    assert (lit_b.lr, lit_b.actor_lr, lit_b.critic_lr, lit_b.max_grad_norm, lit_b.pi_norm, lit_b.cost_clamp,
            lit_b.start_steps, lit_b.dual) == (3e-4, 0.0, 0.0, 0.0, False, False, 10_000, "none")
    for name in configs.CONFIGS:
        rc = configs.make(name)
        if rc.dual == "pid":
            assert (rc.d_delay, rc.ema_p, rc.ema_d, rc.lambda_max) == (10, 0.95, 0.95, 100.0)
    u = configs.from_bvs("BVS-SN-k0.1", "bvs_u", 0.05)
    assert (u.name, u.kind, u.rho_star) == ("BVSU", "bvs_u", 0.05)
    assert set(configs.config_diff(asdict(u), asdict(configs.make("BVS-SN-k0.1")))) == {"name", "kind", "rho_star"}
    disc = configs.from_bvs("BVS-SQ-k0.02", "bvs_disc")
    assert disc.kind == "bvs_disc" and disc.gamma_b == 1 - 1 / 32
    with pytest.raises(ValueError):
        configs.make("BVSU")
    with pytest.raises(TypeError):
        configs.make("B5", no_such_field=1)
    assert configs.config_diff(asdict(disc), asdict(disc)) == {}         # NaN rho_star compares equal


def test_lagu_config_mapping():
    rc = configs.make("B9", hidden=64)
    cfg = lagu_config(rc, horizon=500)
    assert (cfg.M, cfg.use_bonus, cfg.use_gate, cfg.dual, cfg.cost_limit, cfg.lambda_max, cfg.lambda_init) == \
        (1, False, False, "fixed", 20.0, 100.0, 0.0)
    assert cfg.delta0 == pytest.approx(25.0 / 500 / 0.01) and cfg.hidden == 64 and cfg.action_noise == 0.1
    assert lagu_config(configs.make("B2"), 1000).dual == "episodic"
    sa = Agent(OBS, ACT, configs.make("SauteA", hidden=16), HORIZON)
    assert sa.actor.net[0].in_features == OBS + 1 and sa.uses_budget


def test_bvs_configs_need_the_shield_after_the_warmup():
    """Review (BVS F6): the trainer passes info = {} on warm-up steps, so a BVS-family config must have
    start_steps <= shield_start (<= u_start for BVS-U); base configs are unaffected."""
    for name in configs.BVS_CONFIGS:
        configs.make(name)
        configs.from_bvs(name, "bvs_u", 0.1)
    configs.make("B5", start_steps=30_000)                                # base: no shield
    with pytest.raises(ValueError, match="shield_start"):
        configs.make("BVS-SQ-k0.1", start_steps=30_000)
    with pytest.raises(ValueError, match="u_start"):
        configs.from_bvs("BVS-SQ-k0.1", "bvs_u", 0.1, u_start=20_000)
    configs.from_bvs("BVS-SQ-k0.1", "bvs_disc", u_start=20_000)          # u_start is BVS-U's only
