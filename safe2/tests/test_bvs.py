"""safe2.bvs (SPEC section 10, tests 13-24, and the BVS half of tests 25 and 27): the golden equivalence
with the literal TD3 base, viability targets, exact values on a toy corridor, the shield with mocked
V and Q_r, BVS-U, BVS-disc, the relabel, the margin dual, step accounting, gradient isolation and the
update schedule; then the trainer-level checks (whitelisted columns, resume, snapshot round trip) and
the post-hoc shield (PHSCore, fit_offline). Fake-environment tests take well under a second each;
the corridor takes about 20 s and the real-environment round trip about a minute."""
from __future__ import annotations

import csv
import io
import os
from collections import deque

import numpy as np
import pytest
import torch

from lagu.agent import Config, LagU
from safe2 import bvs as bvs_mod
from safe2 import configs, evaluate, seeds, train
from safe2.base import Agent
from safe2.bvs import BVS, DiscCritic, PHSCore, Viability, ViabilityNet, fit_offline
from safe2.tests.helpers import ACT, OBS, FakeEnv, batch, net_hash, same

HORIZON = 1000
ENV = "SafetyPointGoal1-v0"


# --- helpers ------------------------------------------------------------------------------------------
def sample(i: int, n: int = 64, I=None, a_exec=None) -> dict:
    """Replay-column batch i of the fixed synthetic stream (helpers.batch plus b, b2, a_exec and I)."""
    o, a, r, c, o2, d = batch(i, n, term=True)
    g = np.random.default_rng(1000 + i)
    return {"o": o, "b": g.uniform(-1, 1, (n, 1)).astype(np.float32), "a_pi": a,
            "a_exec": a if a_exec is None else a_exec, "r": r, "c": c, "o2": o2,
            "b2": g.uniform(-1, 1, (n, 1)).astype(np.float32), "d": d,
            "I": np.zeros(n, np.uint8) if I is None else np.asarray(I, np.uint8)}


def agent(name: str = "BVS-SQ-k0.1", seed: int = 0, kind: str | None = None, rho: float = 0.2, **kw) -> BVS:
    kw.setdefault("hidden", 32)
    if kind is not None:
        rc = configs.from_bvs(name, kind, rho if kind == "bvs_u" else float("nan"), **kw)
    else:
        rc = configs.make(name, **kw)
    return BVS(OBS, ACT, rc, HORIZON, seed=seed)


def small(name: str = "BVS-SQ-k0.1", kind: str | None = None, rho: float = 0.2, **kw) -> configs.RunConfig:
    """A config whose shield, margin dual and T schedule all act inside a few hundred fake steps."""
    base = dict(hidden=16, batch=8, start_steps=15, shield_start=20, margin_start=30, u_start=40, u_every=10,
                u_window=50, eval_every=10, eval_episodes=2, snap_every=10, snap_from=0.0, ckpt_every=40)
    base.update(kw)
    if kind is not None:
        return configs.from_bvs(name, kind, rho if kind == "bvs_u" else float("nan"), **base)
    return configs.make(name, **base)


def fake(envs: list | None = None, **kw):
    def make(env_id):
        e = FakeEnv(**kw)
        if envs is not None:
            envs.append(e)
        return e
    return make


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def quiet(line):
    pass


def tens(s: dict) -> dict:
    return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in s.items()}


# --- test 13: golden equivalence ----------------------------------------------------------------------
def _golden_pair(name: str, n: int = 300, kappa: float = 0.0, I_frac: float = 0.0):
    torch.manual_seed(0)
    ref = Agent(OBS, ACT, configs.make("R2", hidden=32, budget_input=True, cost_target_online=True), HORIZON, seed=0)
    torch.manual_seed(0)
    ag = BVS(OBS, ACT, configs.make(name, hidden=32, kappa=kappa), HORIZON, seed=0)
    assert net_hash(ref.actor, ref.members) == net_hash(ag.actor, ag.members)   # SPEC 2.4 construction order
    # V, pi_b (BVS) and the cost critic (base) take different numbers of init draws from the global torch
    # RNG; both runs restart it so the reward critics' smoothing noise is the same stream.
    torch.manual_seed(1)
    for i in range(n):
        s = sample(i)
        ref.update((np.concatenate([s["o"], s["b"]], 1), s["a_pi"], s["r"], s["c"],
                    np.concatenate([s["o2"], s["b2"]], 1), s["d"]))
    torch.manual_seed(1)
    for i in range(n):
        I = (np.arange(64) % 4 == 0) if I_frac else None
        ag.learn(sample(i, I=I), 20_000 + i)
    return ref, ag


@pytest.mark.parametrize("name", ["BVS-SQ-k0.02", "BVS-SN-k0.1"])
def test_shield_off_kappa_0_reproduces_the_literal_td3_base_bit_for_bit(name):
    """Test 13: with kappa = 0 and I = 0, 300 BVS updates (V and pi_b included) leave the actor, the
    reward critic and their targets bit-identical to base.Agent(LIT, budget_input=True, dual="none",
    cost_target_online=True); a relabel with kappa > 0 on some I = 1 rows does change them."""
    ref, ag = _golden_pair(name)
    assert net_hash(ref.actor, ref.members) == net_hash(ag.actor, ag.members)
    assert net_hash(ref.actor_targ, ref.members_targ) == net_hash(ag.actor_targ, ag.members_targ)
    assert ag.n_updates == 300 and ag.stats["v_loss"] > 0.0
    _, relabelled = _golden_pair(name, n=20, kappa=0.1, I_frac=0.25)
    ref20, _ = _golden_pair(name, n=20)
    assert net_hash(relabelled.members) != net_hash(ref20.members)
    o = np.random.default_rng(0).normal(size=OBS)
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    a_pi = np.array([0.3, -0.4], np.float32)
    a, I, info = ag.act(o, 0.2, 3.0, a_pi, m=5.0, rng=rng, mode="off")        # shield off: a_pi, no draw
    assert np.array_equal(a, a_pi) and I == 0 and not info["empty"] and rng.bit_generator.state == before


# --- test 14: viability targets -----------------------------------------------------------------------
def test_viability_targets_on_a_hand_built_batch():
    """Test 14: y_1 == c; y_h reads target head h-1 only (perturbing head h-1 changes column h and
    nothing else; the last head is read by none); no bootstrap on termination; the candidate set is
    [pi_b^targ(o2), pi2, 4 uniform actions from v_gen]; outputs lie in [0, h]."""
    H, B = 8, 16
    ag = agent(H=H)
    via = ag.via
    g = torch.Generator().manual_seed(3)
    o2, pi2 = torch.randn(B, OBS, generator=g), torch.rand(B, ACT, generator=g) * 2 - 1
    c = (torch.rand(B, generator=g) < 0.5).float()
    d = torch.zeros(B)
    d[::4] = 1.0
    state = ag.v_gen.get_state()
    y = via.targets(o2, pi2, c, d)
    ag.v_gen.set_state(state)
    with torch.no_grad():
        uni = torch.rand((4, B, ACT), generator=ag.v_gen) * 2 - 1
        C2 = torch.cat([via.pib_targ(o2)[None], pi2[None], uni])
        mn = via.crit_targ(o2.expand(6, B, OBS), C2).max(0).values.min(0).values
    want = torch.cat([c[:, None], c[:, None] + (1 - d)[:, None] * mn[:, :-1]], 1)
    assert torch.equal(y, want)
    assert torch.equal(y[:, 0], c)
    assert torch.equal(y[d == 1], c[d == 1][:, None].expand(-1, H))
    for k in range(H):                                 # perturb target head k + 1 (0-based column k)
        ag.v_gen.set_state(state)
        orig = [f[-1].bias.clone() for f in via.crit_targ.f]
        with torch.no_grad():
            for f in via.crit_targ.f:
                f[-1].bias[k] += 2.0
        y2 = via.targets(o2, pi2, c, d)
        with torch.no_grad():
            for f, b in zip(via.crit_targ.f, orig):
                f[-1].bias.copy_(b)
        assert (y2 != y).any(0).tolist() == [h == k + 1 for h in range(H)], k
    hs = torch.arange(1, H + 1, dtype=torch.float32)
    v = via.crit(torch.randn(4096, OBS) * 50, torch.rand(4096, ACT) * 2 - 1)
    assert v.shape == (2, 4096, H) and (v >= 0).all() and (v <= hs).all()
    assert (y <= hs).all() and (y >= 0).all()          # c <= 1: y_h <= 1 + (h - 1)
    assert torch.equal(via.crit.vmax(o2, pi2), via.crit(o2, pi2)[..., -1].max(0).values)


# --- test 15: exact values on a corridor ---------------------------------------------------------------
# positions 0..4, hazard cell 3: entering it costs 1 and holds the agent there for two more costly
# steps (stuck counter 2, 1); from (3, 0) back or forward leave it, stop stays (cost 1)
CORRIDOR = [(0, 0), (1, 0), (2, 0), (3, 2), (3, 1), (3, 0), (4, 0)]
CENTRES = (-2 / 3, 0.0, 2 / 3)                        # back, stop, forward: [-1, -1/3), [-1/3, 1/3], (1/3, 1]


def _corridor_step(s, k):
    p, stuck = s
    if p == 3 and stuck > 0:
        return (3, stuck - 1), 1.0
    q = (max(p - 1, 0), p, min(p + 1, 4))[k]
    if q == 3:
        return ((3, 0), 1.0) if p == 3 else ((3, 2), 1.0)
    return (q, 0), 0.0


def test_trained_viability_equals_the_exact_minimum_cost_on_a_corridor():
    """Test 15: on a deterministic 1-D corridor with one hazard cell and a stop action, V trained by the
    BVS update code (Viability; lr 1e-3 and Polyak 0.05 for speed) matches the exact minimum cost within
    h steps, within 0.1, for every h = 1..6, every state and every action, for both twins."""
    H, n = 6, len(CORRIDOR)
    idx = {s: i for i, s in enumerate(CORRIDOR)}
    nxt = np.array([[idx[_corridor_step(s, k)[0]] for k in range(3)] for s in CORRIDOR])
    cost = np.array([[_corridor_step(s, k)[1] for k in range(3)] for s in CORRIDOR], np.float32)
    exact = np.zeros((H + 1, n, 3))
    for h in range(1, H + 1):
        exact[h] = cost + exact[h - 1][nxt].min(-1)
    assert sorted(set(exact[H].ravel())) == [0.0, 1.0, 2.0, 3.0] and exact[2, idx[(2, 0)], 2] == 2.0
    torch.manual_seed(0)
    via = Viability(n, 1, 64, H, lr=1e-3, gen=torch.Generator().manual_seed(0))
    eye = np.eye(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    for k in range(1, 8001):
        s, a = rng.integers(0, n, 256), rng.uniform(-1, 1, 256)
        act = np.digitize(a, [-1 / 3, 1 / 3 + 1e-12])
        via.update(torch.as_tensor(eye[s]), torch.as_tensor(a, dtype=torch.float32)[:, None],
                   torch.as_tensor(cost[s, act]), torch.as_tensor(eye[nxt[s, act]]), torch.zeros(256),
                   torch.zeros(256, 1))
        if k % 2 == 0:
            via.update_pib(torch.as_tensor(eye[s]))
            via.polyak(0.05)
    with torch.no_grad():
        o = torch.as_tensor(eye).repeat_interleave(3, 0)
        a = torch.as_tensor(np.tile(CENTRES, n), dtype=torch.float32)[:, None]
        v = via.crit(o, a).numpy().reshape(2, n, 3, H)
    err = np.abs(v - exact[1:].transpose(1, 2, 0)[None])
    assert err.max() < 0.1, float(err.max())


# --- test 16: the shield with mocked V and Q_r ---------------------------------------------------------
def mocked(rule="SQ", vfun=None, qfun=None, **kw) -> BVS:
    ag = agent("BVS-SQ-k0.1" if rule == "SQ" else "BVS-SN-k0.1", **kw)
    ag._vmax = lambda o, A: (vfun or (lambda A: 20.0 * (A[:, 0] + 1.0)))(np.asarray(A, np.float32))
    ag._sq = lambda x, o, A: (qfun or (lambda A: -A[:, 1]))(np.asarray(A, np.float32))
    return ag


def test_tau_and_the_budget_spent_rule():
    ag = agent()
    assert ag.tau(0.0, 5.0) == 20.0 and ag.tau(10.0, 5.0) == 10.0
    assert ag.tau(19.8, 5.0) == 0.5                                  # max(eps, 0.2)
    assert ag.tau(24.9, -10.0) == pytest.approx(10.1)                # negative margin, budget not yet spent
    assert ag.tau(25.0, -10.0) == 0.5 and ag.tau(40.0, -12.5) == 0.5  # C >= d: eps, whatever m
    assert ag.tau(0.0, 5.0, "strict") == 0.5


def test_shield_logic_with_mocked_v_and_q():
    """Test 16: a_pi when allowed (no draw); otherwise the 32 candidates in SPEC order from rng; S-Q
    takes the allowed candidate with the highest Q_r, S-N the nearest allowed; with A empty the argmin
    of vmax; ties to the first index; the executed action is allowed whenever A is non-empty."""
    o = np.random.default_rng(1).normal(size=OBS)
    a_pi = np.array([0.5, 0.2], np.float32)
    cands_of = lambda ag: ag.candidates(o, a_pi, np.random.default_rng(7))
    ag = mocked("SQ")
    # C = 5, m = 5: tau = 15, allowed iff 20 (a0 + 1) <= 15, i.e. a0 <= -0.25
    rng = np.random.default_rng(7)
    st = rng.bit_generator.state
    a, I, info = ag.act(o, 0.8, 5.0, np.array([-0.5, 0.2], np.float32), 5.0, rng)
    assert a.tolist() == pytest.approx([-0.5, 0.2]) and I == 0 and info["vmax"] == pytest.approx(10.0)
    assert rng.bit_generator.state == st and not info["empty"]
    cands = cands_of(ag)
    r = np.random.default_rng(7)
    pert = np.clip(a_pi + r.normal(0.0, 0.3, (16, ACT)), -1, 1)
    uni = r.uniform(-1, 1, (15, ACT))
    with torch.no_grad():
        pb = ag.via.pib(torch.as_tensor(o, dtype=torch.float32)[None]).numpy()
    assert cands.shape == (32, ACT) and cands.dtype == np.float32
    assert np.array_equal(cands, np.concatenate([pert, uni, pb]).astype(np.float32))
    allowed = np.flatnonzero(20.0 * (cands[:, 0] + 1.0) <= 15.0)
    assert 0 < len(allowed) < 32
    a, I, info = ag.act(o, 0.8, 5.0, a_pi, 5.0, np.random.default_rng(7))
    assert I == 1 and np.array_equal(a, cands[allowed[np.argmax(-cands[allowed, 1])]])
    assert info["n_allowed"] == len(allowed) and not info["empty"]
    assert info["vmax"] == pytest.approx(20.0 * (a[0] + 1.0))
    a, I, _ = ag.act(o, 0.8, 5.0, a_pi, 5.0, np.random.default_rng(7), rule="SN")   # rule override (s_near)
    near = allowed[np.argmin(np.linalg.norm(cands[allowed] - a_pi, axis=1))]
    assert I == 1 and np.array_equal(a, cands[near])
    sn = mocked("SN")
    c_sn = cands_of(sn)
    al_sn = np.flatnonzero(20.0 * (c_sn[:, 0] + 1.0) <= 15.0)
    a, I, _ = sn.act(o, 0.8, 5.0, a_pi, 5.0, np.random.default_rng(7))               # rc.rule = SN
    assert I == 1 and np.array_equal(a, c_sn[al_sn[np.argmin(np.linalg.norm(c_sn[al_sn] - a_pi, axis=1))]])
    ag._sq = lambda x, o, A: np.zeros(len(A))                                        # Q ties: first allowed
    a, _, _ = ag.act(o, 0.8, 5.0, a_pi, 5.0, np.random.default_rng(7))
    assert np.array_equal(a, cands[allowed[0]])
    # A empty (budget spent: tau = eps = 0.5 and vmax >= 1): best-effort argmin of vmax over all 32
    ag._vmax = lambda o, A: 1.0 + 20.0 * (np.asarray(A)[:, 0] + 1.0)
    a, I, info = ag.act(o, -0.2, 30.0, a_pi, 5.0, np.random.default_rng(7))
    assert info["empty"] and I == 1 and info["n_allowed"] == 0
    assert np.array_equal(a, cands[np.argmin(cands[:, 0])])
    a, I, info = ag.act(o, 0.8, 5.0, a_pi, 5.0, np.random.default_rng(7), mode="strict")   # tau = eps
    assert info["empty"] and I == 1
    ag._vmax = lambda o, A: np.full(len(A), 3.0)                                     # vmax ties: index 0
    a, _, info = ag.act(o, -0.2, 30.0, a_pi, 5.0, np.random.default_rng(7))
    assert info["empty"] and np.array_equal(a, cands[0])
    # the executed action is allowed whenever A is non-empty (random states, budgets, margins, rules)
    vfun = lambda A: 30.0 * (A[:, 0] + 1.0) * (1.0 + A[:, 1] ** 2)
    agents = {rule: mocked(rule, vfun=vfun, qfun=lambda A: A.sum(1)) for rule in ("SQ", "SN")}
    g = np.random.default_rng(11)
    n_screened = 0
    for trial in range(200):
        ag = agents["SQ" if trial % 2 else "SN"]
        C, m = float(g.uniform(0, 40)), float(g.uniform(-12.5, 25))
        a_pi = g.uniform(-1, 1, ACT).astype(np.float32)
        a, I, info = ag.act(g.normal(size=OBS), 0.0, C, a_pi, m, np.random.default_rng(trial))
        if not info["empty"]:
            assert vfun(a[None].astype(np.float32))[0] <= ag.tau(C, m)
        if I == 0:
            assert np.array_equal(a, a_pi) and vfun(a_pi[None])[0] <= ag.tau(C, m)
        n_screened += "n_allowed" in info and not info["empty"]
    assert n_screened > 20
    with pytest.raises(ValueError):
        ag.act(o, 0.0, 0.0, a_pi, 5.0, np.random.default_rng(0), mode="bad")


# --- test 17: BVS-U ---------------------------------------------------------------------------------------
def test_bvsu_trigger_rate_extremes_and_sn_on_an_allowed_a_pi():
    """Test 17: from u_start, rho* = 0 never triggers and rho* = 1 always does; a triggered step whose
    a_pi is allowed executes a_pi with I = 0 under S-N (candidate 32 is a_pi itself)."""
    g = np.random.default_rng(0)
    for rho, want in ((0.0, False), (1.0, True)):
        ag = agent("BVS-SN-k0.1", kind="bvs_u", rho=rho)
        ag.u_window.extend(g.uniform(0, 1, 100))
        ag.post_update(ag.rc.u_start, deque())
        assert ag.T == (float("inf") if rho == 0 else float("-inf"))
        for _ in range(50):
            o = g.normal(size=OBS)
            a_pi = g.uniform(-1, 1, ACT).astype(np.float32)
            _, I, info = ag.act(o, 0.5, 2.0, a_pi, 5.0, np.random.default_rng(1))
            assert info["triggered"] is want and (want or I == 0)
    ag = agent("BVS-SN-k0.1", kind="bvs_u", rho=1.0)
    ag.T = float("-inf")
    ag._vmax = lambda o, A: np.zeros(len(A))                       # every candidate allowed
    a_pi = np.array([0.3, -0.2], np.float32)
    a, I, info = ag.act(g.normal(size=OBS), 0.5, 2.0, a_pi, 5.0, np.random.default_rng(1))
    assert info["triggered"] and I == 0 and np.array_equal(a, a_pi) and info["n_allowed"] == 33
    ag._sq = lambda x, o, A: -np.linalg.norm(A - np.array([-0.9, 0.9]), axis=1)   # S-Q may take another
    a, I, _ = ag.act(g.normal(size=OBS), 0.5, 2.0, a_pi, 5.0, np.random.default_rng(1), rule="SQ")
    assert I == 1 and not np.array_equal(a, a_pi)



def test_corner_a_pi_duplicated_by_a_clipped_candidate_gives_I_0():
    """Review F1 (prereg A.4, I = 1[a_exec != a_pi]): a saturated a_pi = [1, -1] is reproduced exactly by
    clipped perturbations at lower indices; executing such a duplicate is not an intervention, for
    BVS-U under S-N and S-Q with a_pi allowed and for BVS's best-effort exit."""
    g = np.random.default_rng(3)
    a_pi = np.array([1.0, -1.0], np.float32)
    ag = agent("BVS-SN-k0.1", kind="bvs_u", rho=1.0)
    ag.T = float("-inf")
    ag._vmax = lambda o, A: np.zeros(len(A))                       # every candidate allowed
    ag._sq = lambda x, o, A: -np.linalg.norm(np.asarray(A) - a_pi, axis=1)
    for rule in ("SN", "SQ"):
        for i in range(50):
            a, I, info = ag.act(g.normal(size=OBS), 0.5, 2.0, a_pi, 5.0, np.random.default_rng(i), rule=rule)
            assert info["triggered"] and I == 0 and np.array_equal(a, a_pi)
    bvs = agent("BVS-SN-k0.1")                                      # BVS: a_pi rejected, nothing allowed
    bvs._vmax = lambda o, A: 1.0 + np.linalg.norm(np.asarray(A, np.float32) - a_pi, axis=1)
    hits = 0
    for i in range(50):
        a, I, info = bvs.act(g.normal(size=OBS), 0.5, 2.0, a_pi, 5.0, np.random.default_rng(i), mode="strict")
        assert info["empty"] and I == int(not np.array_equal(a, a_pi))
        hits += I == 0
    assert hits > 40                                               # P(duplicate among 16) ~ 0.99


def test_bvsu_counts_the_counterfactual_trigger_at_T0_after_35000():
    """Review F2 (prereg 14.3): trig007_num counts BVS-U steps > 35,000 with u >= T0 = 0.07, whatever T;
    the other kinds leave the column empty."""
    ag = agent("BVS-SN-k0.1", kind="bvs_u", rho=0.2)
    ag.begin_episode()
    us = [0.01, 0.07, 0.5, 0.06, 0.2, 0.08, 0.0]
    steps = [34_998, 34_999, 35_000, 35_001, 35_002, 35_003, 35_004]
    for st, u in zip(steps, us):
        ag.on_step(st, 20, 0.0, 0, {"vmax": 1.0, "u": u, "triggered": False, "empty": False})
    assert ag.counters["trig007_num"] == 2 and ag.counters["trig_num"] == 0
    assert ag.log_row()["trig007_num"] == 2 and "trig007_num" in train.OUTCOME_FIELDS
    plain = agent("BVS-SN-k0.1")
    assert "trig007_num" not in plain.log_row()

def test_bvsu_u_equals_lagu_ratio_and_members_2_3_stay_out_of_actor_targets_and_sq():
    """Test 17: u is LagU's ratio on the same members; members 2-3 get no actor gradient; perturbing them
    changes neither the actor, member 1, V, nor the S-Q choice (they enter u only)."""
    ag = agent("BVS-SQ-k0.1", kind="bvs_u", rho=0.1)
    assert len(ag.members) == 3 and ag.sq_critic is ag.members[0]
    lag = LagU(OBS + 1, ACT, Config(M=3, hidden=32))
    lag.members.load_state_dict(ag.members.state_dict())
    g = np.random.default_rng(0)
    for _ in range(20):
        x, a = g.normal(size=OBS + 1), g.uniform(-1, 1, ACT)
        xt, at = torch.as_tensor(x, dtype=torch.float32)[None], torch.as_tensor(a, dtype=torch.float32)[None]
        with torch.no_grad():
            qbar, var = lag.ensemble(xt, at)
            ratio = float(torch.sqrt(var + 1e-12) / (qbar.abs() + lag.cfg.ratio_eps))
        assert ag.u(x, a) == ratio
    ag.q_opt.zero_grad()
    ag._actor_update(torch.as_tensor(np.concatenate([sample(0)["o"], sample(0)["b"]], 1)))
    assert all(p.grad is None for m in ag.members[1:] for p in m.parameters())
    assert all(p.grad is not None for p in ag.members[0].q1.parameters())
    assert all(p.grad is None for p in ag.members[0].q2.parameters())
    # perturb members 2-3 and their targets: everything but those members is unchanged
    torch.manual_seed(0)
    a1 = agent("BVS-SQ-k0.1", kind="bvs_u", rho=0.1)
    torch.manual_seed(0)
    a2 = agent("BVS-SQ-k0.1", kind="bvs_u", rho=0.1)
    with torch.no_grad():
        for net in (a2.members[1:], a2.members_targ[1:]):
            for p in net.parameters():
                p.add_(torch.randn_like(p))
    for ag_ in (a1, a2):
        torch.manual_seed(3)
        for i in range(6):
            ag_.learn(sample(i, I=np.arange(64) % 3 == 0), 30_000 + i)
    assert net_hash(a1.actor, a1.actor_targ, a1.members[0], a1.via.crit, a1.via.pib) == \
        net_hash(a2.actor, a2.actor_targ, a2.members[0], a2.via.crit, a2.via.pib)
    assert net_hash(a1.members[1:]) != net_hash(a2.members[1:])
    o = g.normal(size=OBS)
    a_pi = np.array([0.9, 0.9], np.float32)
    for ag_ in (a1, a2):
        ag_.T = float("-inf")
        ag_._vmax = lambda o, A: 10.0 * (np.asarray(A)[:, 0] + 1.0)
    r1 = a1.act(o, 0.5, 2.0, a_pi, 5.0, np.random.default_rng(4))
    r2 = a2.act(o, 0.5, 2.0, a_pi, 5.0, np.random.default_rng(4))
    assert np.array_equal(r1[0], r2[0]) and r1[1] == r2[1] == 1 and r1[2]["u"] != r2[2]["u"]


def test_bvsu_threshold_schedule():
    """Test 17: T = 0.07 until u_start (35k); at 35k and every 1000 steps after, the (1 - rho*) linear
    quantile of the last 10,000 executed-step u values (steps > shield_start)."""
    ag = agent("BVS-SN-k0.1", kind="bvs_u", rho=0.25)
    rc = ag.rc
    assert (rc.T0, rc.u_start, rc.u_every, rc.u_window, rc.shield_start) == (0.07, 35_000, 1000, 10_000, 25_000)
    us = np.random.default_rng(0).uniform(0, 1, 12_000)
    for k, u in enumerate(us[:10_000]):                    # steps 25,001 .. 35,000
        step = rc.shield_start + 1 + k
        ag.on_step(step, k % 1000, 0.0, 0, {"u": float(u), "triggered": False, "vmax": 0.0})
        ag.post_update(step, deque())
        if step < rc.u_start:
            assert ag.T == 0.07
    assert ag.T == float(np.quantile(us[:10_000], 0.75))
    for k, u in enumerate(us[10_000:]):                    # steps 35,001 .. 37,000
        step = rc.u_start + 1 + k
        ag.on_step(step, k % 1000, 0.0, 0, {"u": float(u), "triggered": False, "vmax": 0.0})
        before = ag.T
        ag.post_update(step, deque())
        if step % 1000:
            assert ag.T == before
    assert len(ag.u_window) == 10_000 and ag.T == float(np.quantile(us[2_000:], 0.75))
    ag2 = agent("BVS-SN-k0.1", kind="bvs_u", rho=0.25)
    ag2.on_step(rc.shield_start, 0, 0.0, 0, {})           # steps <= shield_start never enter the window
    assert len(ag2.u_window) == 0


# --- test 18: BVS-disc -------------------------------------------------------------------------------------
def test_bvs_disc_critic_target_clamp_and_budget_test():
    """Test 18: Q_b replaces V; targets clamp(c + gamma_b (1-d) min_C' max_j Q_b^targ, 0, 32) with
    gamma_b = 1 - 1/32; the shield allows a iff max_j Q_b(o, a) <= tau."""
    ag = agent("BVS-SQ-k0.1", kind="bvs_disc")
    assert isinstance(ag.via.crit, DiscCritic) and ag.crit_name == "Qb" and ag.via.gamma_b == 1 - 1 / 32 == 0.96875
    assert ag.via.qb_max == 32.0
    B = 12
    g = torch.Generator().manual_seed(0)
    o2, pi2 = torch.randn(B, OBS, generator=g), torch.rand(B, ACT, generator=g) * 2 - 1
    c, d = torch.ones(B), torch.zeros(B)
    d[0] = 1.0
    st = ag.v_gen.get_state()
    for bias, want in ((None, None), (100.0, 32.0), (-100.0, 0.0)):
        if bias is not None:
            with torch.no_grad():
                for f in ag.via.crit_targ.f:
                    f[-1].bias.fill_(bias)
        ag.v_gen.set_state(st)
        y = ag.via.targets(o2, pi2, c, d)
        ag.v_gen.set_state(st)
        with torch.no_grad():
            uni = torch.rand((4, B, ACT), generator=ag.v_gen) * 2 - 1
            C2 = torch.cat([ag.via.pib_targ(o2)[None], pi2[None], uni])
            mn = ag.via.crit_targ(o2.expand(6, B, OBS), C2).max(0).values.min(0).values
        assert torch.equal(y, (c + 0.96875 * (1 - d) * mn).clamp(0.0, 32.0))
        assert y.shape == (B,) and y[0] == 1.0 and ((y >= 0) & (y <= 32)).all()       # row 0 terminal: y = c
        if want is not None:
            assert (y[1:] == want).all()
    o = np.random.default_rng(0).normal(size=OBS)
    A = np.random.default_rng(1).uniform(-1, 1, (32, ACT)).astype(np.float32)
    with torch.no_grad():
        q = ag.via.crit(torch.as_tensor(o, dtype=torch.float32)[None].expand(32, -1), torch.as_tensor(A))
    assert np.array_equal(ag._vmax(o, A), q.max(0).values.numpy())
    a_pi = A[0]
    v = float(ag._vmax(o, a_pi[None])[0])
    a, I, _ = ag.act(o, 1.0, 0.0, a_pi, m=25.0 - v - 1e-3, rng=np.random.default_rng(0))   # tau just above
    assert I == 0 and np.array_equal(a, a_pi)
    with torch.no_grad():
        for f in ag.via.crit.f:
            f[-1].bias.fill_(50.0)                         # every action far above any tau: best effort
    _, I, info = ag.act(o, 1.0, 0.0, a_pi, m=5.0, rng=np.random.default_rng(0))
    assert I == 1 and info["empty"] and np.isnan(ag.saturation_gap(o, a_pi))


# --- tests 19, 20, 22, 24: through the trainer on a fake environment ----------------------------------
def test_episode_state_resets_and_budget_input(tmp_path, monkeypatch):
    """Test 19: C and b = clip((25 - C)/25, -1, 1) restart at every episode, x = cat(o, b), the acting
    margin is fixed within an episode, and the replay's b and b2 follow C."""
    seen = []
    orig = BVS.select_action

    def spy(self, x, o, b, C, step):
        seen.append((step, x.copy(), o.copy(), b, C, self.m))
        return orig(self, x, o, b, C, step)
    monkeypatch.setattr(BVS, "select_action", spy)
    rc = small(start_steps=5, margin_start=0, eval_every=100, ckpt_every=100, snap_every=100)
    train.run(rc, "Fake", 920, str(tmp_path), steps=60, make_env=fake(length=10, cost_fn=lambda ep, t: 5.0 + ep),
              console=quiet)
    for step, x, o, b, C, m in seen:
        t = (step - 1) % 10
        ep = (step - 1) // 10
        assert C == t * (5.0 + ep) and b == float(np.clip((25 - C) / 25, -1, 1)) and x[-1] == b
        assert np.array_equal(x[:-1], o)
    for ep in range(1, 6):
        assert len({m for step, *_, m in seen if (step - 1) // 10 == ep}) == 1      # fixed within an episode
    assert len({m for *_, m in seen}) > 1                                            # and moved between them
    ck = torch.load(tmp_path / "BVS-SQ-k0.1_Fake_s920.ckpt.pt", weights_only=True)
    rep = {k: v.numpy() for k, v in ck["replay"].items() if k not in ("ptr", "n")}
    C = rep["t"] * (5.0 + rep["ep"])
    assert np.allclose(rep["b"][:, 0], np.clip((25 - C) / 25, -1, 1))
    assert np.allclose(rep["b2"][:, 0], np.clip((25 - C - 5.0 - rep["ep"]) / 25, -1, 1))


def _always_reject(self, o, A):
    """A V mock: allowed iff a0 <= (tau - 1)/30 - 1; nothing is allowed once tau = eps."""
    return 30.0 * (np.asarray(A)[:, 0] + 1.0) + 1.0


def test_relabel_columns_and_the_relabelled_reward_target(tmp_path, monkeypatch):
    """Test 20: the replay stores a_pi and a_exec (different exactly on I = 1 rows), I = 0 up to
    shield_start; Q_r's target uses r - kappa I."""
    monkeypatch.setattr(BVS, "_vmax", _always_reject)
    rc = small("BVS-SQ-k0.1", eval_every=100, ckpt_every=100, snap_every=100)
    train.run(rc, "Fake", 921, str(tmp_path), steps=100, make_env=fake(length=20, cost_fn=lambda ep, t: 2.0),
              console=quiet)
    ck = torch.load(tmp_path / "BVS-SQ-k0.1_Fake_s921.ckpt.pt", weights_only=True)
    rep = ck["replay"]
    I, a_pi, a_exec = rep["I"].numpy(), rep["a_pi"].numpy(), rep["a_exec"].numpy()
    assert rep["I"].dtype == torch.uint8 and not I[:20].any() and I[20:].sum() > 10
    assert np.array_equal(a_pi[I == 0], a_exec[I == 0])
    assert (np.abs(a_pi[I == 1] - a_exec[I == 1]).max(1) > 0).all()
    monkeypatch.undo()
    ag = agent("BVS-SQ-k0.1")
    s = sample(0, I=np.arange(64) % 2 == 0)
    t = tens(s)
    with torch.no_grad():
        x, x2 = torch.cat([t["o"], t["b"]], 1), torch.cat([t["o2"], t["b2"]], 1)
        pi2 = ag.actor_targ(x2)
        torch.manual_seed(5)
        eps = (torch.randn_like(t["a_pi"]) * 0.2).clamp(-0.5, 0.5)
        a2 = (pi2 + eps).clamp(-1, 1)
        boot = 0.99 * (1 - t["d"]) * torch.min(*ag.members_targ[0](x2, a2))
        q1, q2 = ag.members[0](x, t["a_pi"])
        mse = lambda y: float(torch.nn.functional.mse_loss(q1, y) + torch.nn.functional.mse_loss(q2, y))
        want, plain = mse((t["r"] - 0.1 * t["I"]) + boot), mse(t["r"] + boot)
    torch.manual_seed(5)
    ag.learn(s, 20_000)
    assert ag.stats["q_loss"] == pytest.approx(want, rel=1e-6) and ag.stats["q_loss"] != pytest.approx(plain, rel=1e-3)


def test_step_accounting_and_counters(tmp_path, monkeypatch):
    """Test 22 and the SPEC 2.4 counters: every executed step (warm-up, shielded, best-effort) is one
    step and one replay row; int, shield-step, empty, F, episode, entry, miss and rho* counters equal an
    independent recount of the training trace."""
    monkeypatch.setattr(BVS, "_vmax", _always_reject)
    monkeypatch.setattr(bvs_mod, "RHO_FROM", 30)
    trace = []
    orig = BVS.on_step

    def spy(self, step, t, c, I, info):
        trace.append((step, t, c, int(I), bool(info.get("empty", False))))
        return orig(self, step, t, c, I, info)
    monkeypatch.setattr(BVS, "on_step", spy)

    def cost(ep, t):                                    # entries at 13 and 26; C reaches 31 at t = 28
        return 1.0 if t in (13, 39) else (10.0 if t in (26, 27, 28) else 0.0)
    rc = small("BVS-SN-k0.02", eval_every=200, ckpt_every=200, snap_every=200)
    train.run(rc, "Fake", 922, str(tmp_path), steps=200, make_env=fake(length=40, cost_fn=cost), console=quiet)
    ck = torch.load(tmp_path / "BVS-SN-k0.02_Fake_s922.ckpt.pt", weights_only=True)
    rep, k = ck["replay"], ck["agent"]["counters"]
    assert [s for s, *_ in trace] == list(range(1, 201)) and int(rep["n"]) == 200 and ck["updates"] == 185
    assert rep["t"].tolist() == list(range(40)) * 5 and rep["ep"].tolist() == [e for e in range(5) for _ in range(40)]
    assert rep["I"].tolist() == [I for *_, I, _ in trace]
    sh = [x for x in trace if x[0] > 20]
    assert k["shield_steps_cum"] == 180 and k["int_cum"] == sum(x[3] for x in sh)
    assert k["empty_cum"] == sum(x[4] for x in sh) > 0
    assert k["rho_den"] == 170 and k["rho_num"] == sum(x[3] for x in trace if x[0] > 30)
    F = entries = misses = 0
    for e in range(5):
        ep = trace[40 * e:40 * (e + 1)]
        c = [x[2] for x in ep]
        I = [x[3] for x in ep]
        empty = [x[4] for x in ep]
        if any(empty):
            F += sum(c[empty.index(True):])
        for t in range(12, 40):
            if ep[t][0] > 20 and c[t] > 0 and not any(c[t - 12:t]):
                entries += 1
                misses += not any(I[t - 12:t])
    assert k["ep_cum"] == 5 and k["F_cum"] == F > 0
    assert k["entries_cum"] == entries == 9 and k["misses_cum"] == misses
    assert "trig_cum" in k and k["trig_cum"] == 0                       # BVS-U only


def test_update_schedule_and_one_shared_batch(tmp_path, monkeypatch):
    """Test 24: Q_r and V update at every step > start_steps; the actor, pi_b and every Polyak update on
    every 2nd update; one replay sample per step, shared by every loss of that step."""
    calls = []

    def spy(cls, name):
        orig = getattr(cls, name)

        def f(self, *a, **kw):
            calls.append((name, a))
            return orig(self, *a, **kw)
        monkeypatch.setattr(cls, name, f)
    for cls, name in ((BVS, "learn"), (BVS, "_critic_update"), (Viability, "update"), (BVS, "_actor_update"),
                      (Viability, "update_pib"), (BVS, "_polyak"), (train.Replay, "sample")):
        spy(cls, name)
    rc = small(start_steps=15, eval_every=100, ckpt_every=100, snap_every=100)
    train.run(rc, "Fake", 923, str(tmp_path), steps=40, make_env=fake(length=10), console=quiet)
    names = [n for n, _ in calls]
    assert names.count("learn") == names.count("sample") == names.count("_critic_update") == 25
    assert names.count("update") == 25 and names.count("_actor_update") == names.count("update_pib") == 12
    assert names.count("_polyak") == 12
    steps = [a[1] for n, a in calls if n == "learn"]
    assert steps == list(range(16, 41))
    per_step, cur = [], None
    for n, a in calls:
        if n == "sample":
            cur = []
            per_step.append(cur)
        elif n != "learn":
            cur.append(n)
    assert per_step[0] == ["_critic_update", "update"]
    assert per_step[1] == ["_critic_update", "update", "_actor_update", "update_pib", "_polyak"]
    for n, a in calls:                                  # V's batch is the learn() batch
        if n == "learn":
            s = a[0]
        if n == "update":
            assert torch.equal(a[0], torch.as_tensor(s["o"])) and torch.equal(a[1], torch.as_tensor(s["a_exec"]))


# --- test 21: margin dual ---------------------------------------------------------------------------------
def test_margin_dual():
    """Test 21: m_tilde projected onto [-12.5, inf) (anti-windup, prereg A.6), acting margin clipped to
    [-12.5, 25] and fixed per episode; no update for episodes ending at steps <= 50k; the telescoping
    identity while the projection does not bind; m_dep = mean of the last 20 acting margins (clip(m0)
    before any)."""
    ag = agent()
    assert (ag.m_tilde, ag.m, ag.m_dep, ag.rc.margin_start, ag.rc.eta_m) == (5.0, 5.0, 5.0, 50_000, 0.1)
    for step in (10_000, 49_999, 50_000):
        ag.begin_episode()
        ag.end_episode(step, 0.0, 400.0)
    assert ag.m_tilde == 5.0 and ag.m_dep == 5.0 and len(ag.margins) == 3
    ag.begin_episode()
    ag.end_episode(50_001, 0.0, 400.0)
    assert ag.m_tilde == pytest.approx(5.0 + 37.5) and ag.m == 5.0           # m is set at the next start
    ag.begin_episode()
    assert ag.m == 25.0 and ag.m_tilde > 25.0
    for _ in range(30):
        ag.end_episode(60_000, 0.0, 0.0)
        ag.begin_episode()
    assert ag.m_tilde == -12.5 and ag.m == -12.5                              # projected, no windup below
    ag = agent()
    g = np.random.default_rng(0)
    J, acting, mt = g.uniform(15, 35, 500), [], []     # around d: the margin stays mostly unclipped
    m0 = ag.m_tilde
    for j in J:
        ag.begin_episode()
        acting.append(ag.m)
        ag.end_episode(100_000, 0.0, float(j))
        mt.append(ag.m_tilde)
        assert ag.m_dep == pytest.approx(np.mean(acting[-20:]), abs=1e-12)
    assert min(mt) > -12.5                             # the projection never bound on this stream: identity
    assert J.mean() == pytest.approx(25.0 + (ag.m_tilde - m0) / (0.1 * len(J)), abs=1e-9)
    assert len(set(acting)) > 100 and min(acting) >= -12.5 and max(acting) <= 25.0
    assert agent(m0=40.0).m_dep == 25.0 and agent(m0=-20.0).m_dep == -12.5


def test_margin_dual_projection_keeps_the_run_bound_and_bounds_every_window():
    """Prereg A.6 (pre-v0 amendment of 2026-10-14). The projection only raises m_tilde, so
    sum (J - d) <= (m_tilde_K - m_tilde_0) / eta always holds (equality while it does not bind), and
    under assumption A (acting margin >= 25 => J <= 25) the full-run average stays <= 26.24 (Goal,
    K = 950) and every window of W counted episodes averages <= 25 + 1350 / W. The unprojected
    integrator lets 475 safe episodes pay for 475 episodes at J = 50 (the math audit's example)."""
    d, eta, jmax, K = 25.0, 0.1, 1000.0, 950
    # (1) the audit's credit-then-spend stream: with the projection the late episodes cannot all be at 50
    ag = agent()
    for j in [0.0] * 475:
        ag.begin_episode()
        ag.end_episode(100_000, 0.0, j)
    assert ag.m_tilde == -12.5                          # unprojected it would sit at 5 - 1187.5
    ag.begin_episode()
    n_over = 0
    while ag.m < 25.0:                                  # J = 50 raises m_tilde by 2.5 per episode
        ag.end_episode(100_000, 0.0, 50.0)
        ag.begin_episode()
        n_over += 1
    assert n_over == 15                                 # (25 + 12.5) / 2.5, not 475
    # (2) adversarial streams satisfying A: J_max whenever m < 25, any J <= 25 otherwise
    for seed in range(20):
        g, ag, Js, m0 = np.random.default_rng(seed), agent(), [], 5.0
        for k in range(K):
            ag.begin_episode()
            j = jmax if (ag.m < 25.0 and g.random() < 0.5) else float(g.uniform(0, 25) if ag.m >= 25.0
                                                                       else g.uniform(0, 60))
            Js.append(j)
            ag.end_episode(100_000, 0.0, j)
        Js = np.asarray(Js)
        assert (Js - d).sum() <= (ag.m_tilde - m0) / eta + 1e-6
        assert Js.mean() <= d + (25.0 + eta * (jmax - d) - m0) / (eta * K) + 1e-9      # 26.24
        for W in (100, 200, 475):
            win = np.convolve(Js, np.ones(W), "valid") / W
            assert win.max() <= d + 1350.0 / W + 1e-9


# --- test 23: gradient isolation ------------------------------------------------------------------------------
def test_gradient_isolation():
    """Test 23: V (and pi_b) get no gradient from the actor step; pi_b's step does not move V, and the
    gradient it leaves on V is discarded by V's next update; V's step moves V only."""
    ag = agent()
    for i in range(4):
        ag.learn(sample(i), 20_000 + i)
    t = tens(sample(9))
    x = torch.cat([t["o"], t["b"]], 1)
    ag.via.opt.zero_grad()
    ag.via.pib_opt.zero_grad()
    ag._actor_update(x)
    assert all(p.grad is None for p in ag.via.crit.parameters())
    assert all(p.grad is None for p in ag.via.pib.parameters())
    assert all(p.grad is not None for p in ag.actor.parameters())
    v_before = net_hash(ag.via.crit)
    ag.via.update_pib(t["o"])
    assert net_hash(ag.via.crit) == v_before
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ag.via.crit.parameters())
    clean = agent()
    buf = io.BytesIO()                                       # through torch.save, as a checkpoint: an
    torch.save(ag.state_dict(), buf)                         # in-memory optimizer state would be aliased
    buf.seek(0)
    clean.load_state_dict(torch.load(buf, weights_only=True))   # same state, no stale gradients on V
    assert all(p.grad is None for p in clean.via.crit.parameters())
    others = net_hash(ag.actor, ag.members, ag.via.pib)
    with torch.no_grad():
        pi2 = ag.actor_targ(torch.cat([t["o2"], t["b2"]], 1))
    for a in (ag, clean):
        a.via.update(t["o"], t["a_exec"], t["c"], t["o2"], t["d"], pi2)
    assert net_hash(ag.via.crit) == net_hash(clean.via.crit) != v_before
    assert net_hash(ag.actor, ag.members, ag.via.pib) == others


# --- trainer-level: whitelist, resume, snapshots ----------------------------------------------------------
KIND_CASES = [("BVS-SQ-k0.1", None), ("BVS-SN-k0.02", None), ("BVS-SQ-k0.02", "bvs_u"), ("BVS-SN-k0.1", "bvs_disc")]


@pytest.mark.parametrize("name,kind", KIND_CASES)
def test_every_whitelisted_column_is_written_once_learning_runs(tmp_path, name, kind):
    rc = small(name, kind)
    train.run(rc, "Fake", 924, str(tmp_path), steps=60, make_env=fake(length=10, cost_fn=lambda ep, t: float(t == 5)),
              console=quiet)
    last = rows(tmp_path / f"{rc.name}_Fake_s924.csv")[-1]
    for col in train.whitelist(rc):
        assert last[col] not in ("", "nan"), (rc.name, col)
    for col in ("qc", "c_loss", "lambda", "jc", "qc0", "mc_cost_traj", "pid_i"):
        assert last[col] == "", col                                       # base-only columns: empty
    for col in ("m", "m_dep", "int_cum", "shield_steps_cum", "ep_cum", "eval_int_steps", "eval_empty_steps"):
        assert last[col] != "", col
    assert (last["T"] != "") == (kind == "bvs_u") and (last["trig_cum"] != "") == (kind == "bvs_u")
    assert all(last[g] == "1" for g in train.GROUPS)
    ag = BVS(OBS, ACT, rc, HORIZON)
    assert set(ag.log_row()) <= set(train.OUTCOME_FIELDS)


@pytest.mark.parametrize("name,kind", [("BVS-SQ-k0.1", None), ("BVS-SN-k0.1", "bvs_u")])
def test_resume_reproduces_an_uninterrupted_bvs_run(tmp_path, name, kind):
    """Test 27 for the BVS family: a run that crashes after its checkpoint at 40 and resumes ends with
    every checkpoint field (networks, optimisers, shield_rng, v_gen, margin state, u window, T,
    counters, replay, RNGs) and every outcome row identical to an uninterrupted run."""
    rc = small(name, kind, ckpt_every=40)
    kw = dict(length=10, cost_fn=lambda ep, t: 9.0 * (t % 4 == 0), ep_in_obs=False)
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    stem = f"{rc.name}_Fake_s925"
    train.run(rc, "Fake", 925, str(a_dir), steps=80, make_env=fake(**kw), console=quiet)

    def crash(step, eps):
        if step == 60:
            raise RuntimeError("simulated crash")
    with pytest.raises(RuntimeError):
        train.run(rc, "Fake", 925, str(b_dir), steps=80, make_env=fake(**kw), console=quiet, on_eval=crash)
    train.run(rc, "Fake", 925, str(b_dir), steps=80, resume=True, make_env=fake(**kw), console=quiet)
    ra, rb = rows(a_dir / f"{stem}.csv"), rows(b_dir / f"{stem}.csv")
    skip = {"sps", "wall_s", "resumed", "n_resumes"}
    assert [int(r["step"]) for r in rb] == list(range(10, 81, 10))
    for x, y in zip(ra, rb):
        assert {k: v for k, v in x.items() if k not in skip} == {k: v for k, v in y.items() if k not in skip}
    ca = torch.load(a_dir / f"{stem}.ckpt.pt", weights_only=True)
    cb = torch.load(b_dir / f"{stem}.ckpt.pt", weights_only=True)
    for k in ca:
        if k not in ("wall_s", "n_resumes"):
            assert same(ca[k], cb[k]), k
    ag = ca["agent"]
    assert ag["counters"]["int_cum"] > 0 and ag["m_tilde"] != 5.0 and len(ag["margins"]) == 8
    if kind == "bvs_u":
        assert ag["T"] != 0.07 and len(ag["u_window"]) == 50


@pytest.mark.parametrize("name,kind", [("BVS-SQ-k0.1", None), ("BVS-SQ-k0.02", "bvs_u"), ("BVS-SN-k0.1", "bvs_disc")])
def test_snapshot_rebuild_equals_the_live_agent_on_every_mode(tmp_path, name, kind):
    """policy_from_snapshot rebuilds the live agent's deployed policy (m_dep, T, every network) without
    touching the torch RNG, and replays its episodes in every mode; shield_off never intervenes."""
    rc = small(name, kind, eval_every=60, ckpt_every=60, snap_every=60)
    agents = {}
    orig = BVS.make_policy

    def keep(self, mode):
        agents["live"] = self
        return orig(self, mode)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BVS, "make_policy", keep)
        train.run(rc, "Fake", 926, str(tmp_path), steps=60, make_env=fake(length=10, cost_fn=lambda ep, t: 3.0),
                  console=quiet)
    live = agents["live"]
    snap = str(tmp_path / f"{rc.name}_Fake_s926.snap" / "60.pt")
    before = torch.get_rng_state()
    rebuilt = evaluate.policy_from_snapshot(snap).core
    assert torch.equal(before, torch.get_rng_state())
    assert rebuilt.m_dep == live.m_dep and rebuilt.T == live.T
    assert net_hash(rebuilt.actor, rebuilt.members, rebuilt.via.crit, rebuilt.via.pib) == \
        net_hash(live.actor, live.members, live.via.crit, live.via.pib)
    modes = ["deployed", "shield_off", "strict"] + (["s_near"] if rc.rule == "SQ" else [])
    for mode in modes:
        e1 = evaluate.rollout(FakeEnv(length=10, cost_fn=lambda ep, t: 3.0), evaluate.policy_from_agent(live, mode),
                              "dev", [0, 1])
        e2 = evaluate.rollout(FakeEnv(length=10, cost_fn=lambda ep, t: 3.0),
                              evaluate.policy_from_snapshot(snap, mode), "dev", [0, 1])
        assert same(e1, e2), mode
        if mode == "shield_off":
            assert all(e["interventions"] == 0 for e in e1)
    s = torch.load(snap, weights_only=True)
    assert {"actor", "members", "pib", "m_dep", "Qb" if kind == "bvs_disc" else "V"} <= set(s)
    assert ("T" in s) == (kind == "bvs_u")


def test_snapshot_round_trip_reproduces_in_training_evaluation_on_the_real_task(tmp_path):
    """Test 25 (fast form for the BVS family; tests/test_bvs_smoke.py has the 30k-step form): the
    held-out evaluator's policy replays the in-training evaluation (shield on, m_dep, layout-seeded
    sampler) episode for episode, bit for bit, for BVS and BVS-U."""
    pytest.importorskip("safety_gymnasium")
    env = train._make_env(ENV)
    for name, kind in (("BVS-SN-k0.1", None), ("BVS-SQ-k0.02", "bvs_u")):
        rc = small(name, kind, hidden=32, batch=64, start_steps=300, shield_start=500, margin_start=500, u_start=700,
                   u_every=100, u_window=1000, eval_every=1000, eval_episodes=2, snap_every=1000, ckpt_every=2000)
        seen = {}
        out = tmp_path / rc.name
        train.run(rc, ENV, 927, str(out), steps=2000, console=quiet, on_eval=lambda step, eps: seen.__setitem__(step, eps))
        assert sorted(seen) == [1000, 2000]
        for step, eps in seen.items():
            policy = evaluate.policy_from_snapshot(str(out / f"{rc.name}_{ENV}_s927.snap" / f"{step}.pt"))
            again = evaluate.rollout(env, policy, "train_eval", range(2))
            assert same(eps, again), (rc.name, step)
        assert sum(e["interventions"] for e in seen[2000]) > 0, rc.name


# --- post-hoc shield ----------------------------------------------------------------------------------------
def _bstar(tmp_path, name):
    rc = configs.make(name, hidden=16, batch=8, start_steps=15, eval_every=40, eval_episodes=1, snap_every=40,
                      snap_from=0.0, ckpt_every=40)
    train.run(rc, "Fake", 928, str(tmp_path), steps=40, make_env=fake(length=10, cost_fn=lambda ep, t: float(t == 3)),
              console=quiet)
    snap = torch.load(tmp_path / f"{name}_Fake_s928.snap" / "40.pt", weights_only=True)
    ck = torch.load(tmp_path / f"{name}_Fake_s928.ckpt.pt", weights_only=True)
    rep = {k: v.numpy() for k, v in ck["replay"].items() if k not in ("ptr", "n")}
    data = {"o": rep["o"], "a": rep["a_pi"], "c": rep["c"], "o2": rep["o2"], "b2": rep["b2"], "d": rep["d"]}
    return snap, data


@pytest.mark.parametrize("name", ["B5", "SauteA"])
def test_fit_offline_and_phs_core(tmp_path, name):
    """PHS (prereg A.9): fit_offline trains V and pi_b with the BVS update code on fixed replay rows,
    deterministically in its seed and without touching the proposer; PHSCore proposes B*'s
    deterministic action (with b only for a budget-input B*), ranks S-Q by B*'s members[0], shields
    with the fitted V at m = m_PHS, and runs under ShieldPolicy."""
    snap, data = _bstar(tmp_path, name)
    bstar = Agent.from_snapshot(snap)
    actor_hash = net_hash(bstar.actor)
    kw = dict(updates=30, batch=16, hidden=16, H=8)
    f1 = fit_offline(data, bstar.actor, bstar.rc.budget_input, seed=4, **kw)
    f2 = fit_offline(data, bstar.actor, bstar.rc.budget_input, seed=4, **kw)
    f3 = fit_offline(data, bstar.actor, bstar.rc.budget_input, seed=5, **kw)
    assert net_hash(bstar.actor) == actor_hash
    assert same(f1["V"], f2["V"]) and same(f1["pib"], f2["pib"]) and not same(f1["V"], f3["V"])
    assert (f1["obs_dim"], f1["act_dim"], f1["H"], len(f1["losses"]["v"]), len(f1["losses"]["pib"])) == \
        (OBS, ACT, 8, 30, 15)
    torch.manual_seed(4)                                # the fit is the Viability loop on these rows
    via = Viability(OBS, ACT, 16, 8, gen=torch.Generator().manual_seed(4))
    rng = np.random.default_rng(4)
    for k in range(1, 31):
        i = rng.integers(0, len(data["o"]), 16)
        T = lambda key: torch.as_tensor(data[key][i], dtype=torch.float32)
        with torch.no_grad():
            pi2 = bstar.actor(torch.cat([T("o2"), T("b2")], 1) if bstar.rc.budget_input else T("o2"))
        via.update(T("o"), T("a"), T("c"), T("o2"), T("d"), pi2)
        if k % 2 == 0:
            via.update_pib(T("o"))
            via.polyak(0.005)
    assert same(via.crit.state_dict(), f1["V"])
    core = PHSCore(snap, f1, m_phs=3.25, rule="SQ")
    assert core.m_dep == 3.25 and core.rc.rule == "SQ" and core.rc.H == 8 and core.kind == "bvs"
    o = np.random.default_rng(0).normal(size=OBS)
    x = np.concatenate([o, [0.4]])
    base_pol = evaluate.BasePolicy(bstar.actor, bstar.rc.budget_input)
    base_pol.C = 15.0                                   # b = 0.4
    assert np.array_equal(core.propose(x), base_pol.act(o))
    A = np.random.default_rng(1).uniform(-1, 1, (32, ACT)).astype(np.float32)
    inp = torch.as_tensor(x if bstar.rc.budget_input else o, dtype=torch.float32)[None].expand(32, -1)
    with torch.no_grad():
        want = torch.min(*bstar.members[0](inp, torch.as_tensor(A))).numpy()
    assert np.array_equal(core._sq(x, o, A), want)
    pol = evaluate.ShieldPolicy(core, "deployed")
    eps = evaluate.rollout(FakeEnv(length=10, cost_fn=lambda ep, t: 4.0), pol, "dev", [0, 1])
    assert all("interventions" in e and "v_pred" in e for e in eps)
    evaluate.ShieldPolicy(core, "s_near")
    with pytest.raises(ValueError):
        evaluate.ShieldPolicy(PHSCore(snap, f1, 3.25, "SN"), "s_near")
    with pytest.raises(ValueError):
        core.propose(x, explore=True)


# --- construction --------------------------------------------------------------------------------------------
def test_construction_rngs_and_kinds():
    """SPEC 2.4: networks in order (actor, members, V or Q_b, pi_b, each with a frozen target); four Adam
    optimisers at lr 3e-4; self.rng = default_rng(seed), shield_rng and v_gen from
    SeedSequence(seed).spawn(2); M = 3 for BVS-U; base configs are refused."""
    ag = agent(seed=123)
    ss = np.random.SeedSequence(123).spawn(2)
    assert ag.rng.bit_generator.state == np.random.default_rng(123).bit_generator.state
    assert ag.shield_rng.bit_generator.state == np.random.default_rng(ss[0]).bit_generator.state
    assert torch.equal(ag.v_gen.get_state(),
                       torch.Generator().manual_seed(int(ss[1].generate_state(1)[0])).get_state())
    assert isinstance(ag.via.crit, ViabilityNet) and len(ag.members) == 1
    for targ in (ag.actor_targ, ag.members_targ, ag.via.crit_targ, ag.via.pib_targ):
        assert not any(p.requires_grad for p in targ.parameters())
    for opt in (ag.pi_opt, ag.q_opt, ag.via.opt, ag.via.pib_opt):
        assert isinstance(opt, torch.optim.Adam) and opt.param_groups[0]["lr"] == 3e-4
    assert ag.actor.net[0].in_features == OBS + 1 and ag.via.pib.net[0].in_features == OBS
    assert ag.via.crit.f[0][0].in_features == OBS + ACT and ag.members[0].q1[0].in_features == OBS + 1 + ACT
    assert ag.replay_extras == {"a_exec": ((ACT,), np.float32), "I": ((), np.uint8)} and ag.uses_budget
    torch.manual_seed(0)
    a1 = agent()
    torch.manual_seed(0)
    from lagu.agent import Actor, TwinQ, mlp
    actor = Actor(OBS + 1, ACT, 32)
    members = torch.nn.ModuleList([TwinQ(OBS + 1, ACT, 32)])
    f = [mlp(OBS + ACT, 32, 32), mlp(OBS + ACT, 32, 32)]
    pib = Actor(OBS, ACT, 32)
    assert net_hash(a1.actor, a1.members, a1.via.pib) == net_hash(actor, members, pib)
    assert net_hash(*a1.via.crit.f) == net_hash(*f)
    assert len(agent(kind="bvs_u").members) == 3
    with pytest.raises(ValueError):
        BVS(OBS, ACT, configs.make("B5"), HORIZON)


# --- review follow-ups: interval columns, shield inputs, non-finite values, blinding ------------------------
def test_interval_columns_equal_a_hand_computed_trace():
    """vH_exec_mean is the mean vmax of the shielded steps of the interval and u_p50 the median u; steps
    <= shield_start do not count, and reset_interval starts a new interval."""
    ag = agent("BVS-SN-k0.1", kind="bvs_u", rho=0.2)
    ag.begin_episode()
    s0 = ag.rc.shield_start
    trace = [(s0, 100.0, 9.0), (s0 + 1, 2.0, 0.3), (s0 + 2, 4.0, 0.1), (s0 + 3, 9.0, 0.2), (s0 + 4, 1.0, 0.7)]
    for st, v, u in trace:
        ag.on_step(st, 20, 0.0, 0, {"vmax": v, "u": u, "triggered": False, "empty": False})
    row = ag.log_row()
    assert row["vH_exec_mean"] == pytest.approx(4.0) and row["u_p50"] == pytest.approx(0.25)
    ag.reset_interval()
    assert ag.log_row()["vH_exec_mean"] == "" and ag.log_row()["u_p50"] == ""


def test_training_shields_with_m_and_shield_rng_and_evaluation_with_m_dep_and_the_layout_rng(monkeypatch):
    """The acting margin and shield_rng in training; m_dep and default_rng(layout id) in evaluation."""
    calls = []
    orig = BVS.act

    def spy(self, o, b, C, a_pi, m, rng, mode="on", rule=None):
        calls.append((m, rng, rng.bit_generator.state, mode))
        return orig(self, o, b, C, a_pi, m, rng, mode=mode, rule=rule)
    monkeypatch.setattr(BVS, "act", spy)
    ag = agent("BVS-SQ-k0.1")
    ag.m, ag.m_dep = 3.0, -4.0
    o = np.random.default_rng(0).normal(size=OBS)
    x = np.concatenate([o, [0.5]])
    ag.select_action(x, o, 0.5, 1.0, ag.rc.shield_start)
    assert calls == []                                              # no shield at steps <= shield_start
    ag.select_action(x, o, 0.5, 1.0, ag.rc.shield_start + 1)
    assert calls[-1][0] == 3.0 and calls[-1][1] is ag.shield_rng and calls[-1][3] == "on"
    pol = evaluate.ShieldPolicy(ag, "deployed")
    pol.begin(17)
    pol.act(o)
    assert calls[-1][0] == -4.0 and calls[-1][1] is not ag.shield_rng
    assert calls[-1][2] == np.random.default_rng(17).bit_generator.state


@pytest.mark.parametrize("name,kind", KIND_CASES[1:])
def test_nan_weights_do_not_stop_a_bvs_family_run(tmp_path, monkeypatch, name, kind):
    """Test 28 for the BVS family: the actor, V (Q_b) and the reward members set to NaN at step 30 ->
    the run reaches its end, every executed action is finite, later rows flag finite_net = 0."""
    orig = BVS.post_update

    def poison(self, step, jc):
        if step == 30:
            with torch.no_grad():
                for net in (self.actor, self.via.crit, self.members):
                    for p in net.parameters():
                        p.fill_(float("nan"))
        return orig(self, step, jc)
    monkeypatch.setattr(BVS, "post_update", poison)
    envs = []
    rc = small(name, kind)
    train.run(rc, "Fake", 926, str(tmp_path), steps=60, make_env=fake(envs, length=10), console=quiet)
    r = rows(tmp_path / f"{rc.name}_Fake_s926.csv")
    assert [int(x["step"]) for x in r] == [10, 20, 30, 40, 50, 60]
    assert all(x["finite_net"] == "0" for x in r[3:]) and int(r[-1]["nonfinite_actions"]) > 0
    assert all(np.all(np.isfinite(a)) for e in envs for a in e.actions)
    assert os.path.exists(tmp_path / f"{rc.name}_Fake_s926.pt")


def test_a_blinded_bvs_u_run_seals_its_rows_and_keeps_only_health_columns(tmp_path):
    """Test 29 for the BVS family: a BVS-U run at seed 300 writes only HEALTH_FIELDS in clear, prints
    only the blinded console line, and its sealed rows decrypt with every outcome column, T included."""
    import re
    from safe2 import seal
    from safe2.tests.helpers import make_key
    key = make_key(tmp_path)
    run_dir = tmp_path / "conf"
    lines = []
    rc = small("BVS-SN-k0.02", "bvs_u")
    train.run(rc, "Fake", 300, str(run_dir), steps=60, make_env=fake(length=10), console=lines.append)
    with open(run_dir / "BVSU_Fake_s300.csv") as f:
        assert next(csv.reader(f)) == train.HEALTH_FIELDS
    recs = seal.read_records(str(tmp_path / "sealed" / "conf" / "BVSU_Fake_s300.rows"), key)
    assert [r["step"] for r in recs] == [10, 20, 30, 40, 50, 60]
    assert set(recs[0]) == {"step"} | set(train.OUTCOME_FIELDS)
    assert recs[-1]["T"] not in ("", None) and recs[-1]["trig007_num"] not in ("", None)
    pat = re.compile(r"^\[BVSU_Fake_s300\] step=\d+ sps=\d+ finite=[01]$")
    assert lines and all(pat.match(x) for x in lines), lines
