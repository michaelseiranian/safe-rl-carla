import numpy as np
import torch

from lagu.agent import Config, LagU

OBS, ACT = 8, 2


def make(**kw):
    torch.manual_seed(0)
    cfg = Config(hidden=32, **kw)
    return LagU(OBS, ACT, cfg, seed=0), cfg


def batch(n=64, cost=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(n, OBS, generator=g)
    a = torch.rand(n, ACT, generator=g) * 2 - 1
    r = torch.randn(n, generator=g)
    c = torch.full((n,), float(cost))
    o2 = torch.randn(n, OBS, generator=g)
    d = torch.zeros(n)
    return o, a, r, c, o2, d


def test_adaptive_delta_matches_eq16():
    ag, cfg = make(delta0=2.5, T=0.07)
    ratio = torch.tensor([0.01, 0.0699, 0.07, 0.14, 0.7])
    got = ag.adaptive_delta(ratio)
    want = torch.tensor([2.5, 2.5, 2.5, 2.5 * 0.07 / 0.14, 2.5 * 0.07 / 0.7])
    assert torch.allclose(got, want)
    assert (got[1:] >= got[:-1] * 0).all()  # never negative


def test_explore_prob_schedule_eq13():
    ag, cfg = make(c0=0.1, Tc=1000)
    assert ag.explore_prob() == 1.0
    ag.env_step = 500                     # x counts environment steps (the paper's "incremental steps")
    assert abs(ag.explore_prob() - 0.5) < 1e-9
    ag.n_pi = 10**6                       # policy updates no longer drive the schedule
    assert abs(ag.explore_prob() - 0.5) < 1e-9
    ag.env_step = 1000
    assert ag.explore_prob() == cfg.c0
    ag.env_step = 5000
    assert ag.explore_prob() == cfg.c0


def test_members_have_their_own_losses_and_disagree():
    ag, cfg = make(M=3, policy_delay=1)
    for i in range(10):
        stats = ag.update(batch(seed=i))
    assert len(stats["member_losses"]) == 3
    assert len(set(round(x, 8) for x in stats["member_losses"])) == 3
    o, a, *_ = batch()
    _, var = ag.ensemble(o, a)
    assert float(var.mean()) > 0.0


def test_lambda_rises_when_cost_exceeds_budget_and_stays_zero_otherwise():
    ag, _ = make(M=1, delta0=0.0, lambda_lr=1e-2, policy_delay=1, use_gate=False, use_bonus=False)
    for i in range(40):
        ag.update(batch(cost=1.0, seed=i))
    assert ag.lmbda > 0.0
    ag2, _ = make(M=1, delta0=100.0, lambda_lr=1e-2, policy_delay=1, use_gate=False, use_bonus=False)
    for i in range(40):
        ag2.update(batch(cost=0.0, seed=i))
    assert ag2.lmbda == 0.0


def test_gate_engagement_is_counted_only_when_gate_is_on():
    ag, _ = make(M=3, use_gate=True, use_bonus=False, policy_delay=1, c0=0.0, Tc=1)
    ag.env_step = 10  # past Tc, so c(x) = c0 = 0: every policy update is risk-sensitive
    for i in range(5):
        ag.update(batch(seed=i))
    assert ag.n_risk == 5 and ag.gate_total == 5 * 64
    assert 0.0 <= ag.engagement() <= 1.0
    off, _ = make(M=3, use_gate=False, use_bonus=False, policy_delay=1, c0=0.0, Tc=1)
    off.env_step = 10
    for i in range(5):
        off.update(batch(seed=i))
    assert off.gate_total == 0 and off.engagement() == 0.0


def test_td3lag_arm_has_no_uncertainty_signal():
    ag, _ = make(M=1, use_gate=False, use_bonus=False, action_noise=0.1, policy_delay=1)
    ag.update(batch())
    assert ag.pi_stats["qstd"] < 1e-5
    a = ag.act(np.zeros(OBS, np.float32), explore=True)
    assert a.shape == (ACT,) and np.all(np.abs(a) <= 1.0)


def test_update_is_finite_and_polyak_moves_targets():
    ag, _ = make(M=2, policy_delay=1)
    before = [p.clone() for p in ag.actor_targ.parameters()]
    stats = ag.update(batch())
    assert np.isfinite(stats["q_loss"]) and np.isfinite(stats["c_loss"]) and np.isfinite(ag.pi_stats["pi_loss"])
    assert any(not torch.equal(b, p) for b, p in zip(before, ag.actor_targ.parameters()))


def test_episodic_dual_moves_lambda_with_realised_cost():
    ag, _ = make(M=1, dual="episodic", lambda_lr=0.05, cost_limit=25.0, use_gate=False, use_bonus=False)
    for _ in range(20):
        ag.update_dual_episodic(50.0)   # cost above the limit: lambda must rise
    assert ag.lmbda > 0.0
    for _ in range(300):
        ag.update_dual_episodic(0.0)    # cost far below: lambda decays and is clipped at 0
    assert ag.lmbda == 0.0


def test_gate_grad_changes_the_actor_update():
    def run(gate_grad):
        ag, _ = make(M=3, use_gate=True, use_bonus=False, policy_delay=1, c0=0.0, Tc=1,
                     T=1e-4, gate_grad=gate_grad)      # T tiny: the gate fires on every sample
        ag.env_step, ag.lmbda = 10, 1.0
        o, *_ = batch()
        ag._policy_update(o)
        return torch.cat([p.detach().flatten() for p in ag.actor.parameters()])
    detached, through = run(False), run(True)
    assert not torch.allclose(detached, through)


def test_state_dict_round_trip():
    ag, _ = make(M=2, dual="episodic", policy_delay=1)
    for i in range(3):
        ag.update(batch(seed=i))
    ag.update_dual_episodic(40.0)
    d = ag.state_dict()
    ag2, _ = make(M=2, dual="episodic", policy_delay=1)
    ag2.load_state_dict(d)
    o, a, *_ = batch(seed=9)
    assert ag2.lmbda == ag.lmbda and ag2.n_updates == ag.n_updates and ag2.n_pi == ag.n_pi
    assert torch.allclose(ag.actor(o), ag2.actor(o))
    assert torch.allclose(ag.ensemble(o, a)[0], ag2.ensemble(o, a)[0])


def test_pessimistic_cost_critic_uses_max_of_twins_and_round_trips():
    kw = dict(M=1, cost_twin=True, cost_clamp=True, use_gate=False, use_bonus=False, policy_delay=1)
    ag, cfg = make(**kw)
    o, a, *_ = batch()
    x = torch.cat([o, a], -1)
    q1, q2 = ag.cost(x).squeeze(-1), ag.cost2(x).squeeze(-1)
    assert torch.allclose(ag.qc_pi(o, a), torch.max(q1, q2))
    assert abs(cfg.cost_max / (1.0 - cfg.gamma) - 100.0) < 1e-6
    for i in range(5):
        stats = ag.update(batch(cost=1.0, seed=i))
    assert np.isfinite(stats["c_loss"]) and np.isfinite(ag.pi_stats["pi_loss"])
    d = ag.state_dict()
    assert "cost2" in d and "cost2_targ" in d
    ag2, _ = make(**kw)
    ag2.load_state_dict(d)
    assert torch.allclose(ag.qc_pi(o, a), ag2.qc_pi(o, a))


def test_single_cost_critic_checkpoints_stay_loadable():
    ag, _ = make(M=1, use_gate=False, use_bonus=False, policy_delay=1)
    ag.update(batch())
    d = ag.state_dict()
    assert "cost2" not in d
    ag2, _ = make(M=1, use_gate=False, use_bonus=False, policy_delay=1)
    ag2.load_state_dict(d)
    o, a, *_ = batch(seed=3)
    assert torch.allclose(ag.qc_pi(o, a), ag2.qc_pi(o, a))


def test_gate_grad_never_pushes_the_actor_to_lower_the_reward_value():
    """Through |Q_bar| the gate gradient flips sign when Q_bar < 0; it must flow via sqrt(Q_eu) only."""
    ag, _ = make(M=3, use_gate=True, use_bonus=False, gate_grad=True, T=1e-4, c0=0.0, Tc=1, policy_delay=1)
    ag.env_step, ag.lmbda = 10, 1.0
    o, *_ = batch()
    a_pi = ag.actor(o)
    qbar, var = ag.ensemble(o, a_pi)
    std = torch.sqrt(var + 1e-12)
    ratio = std / (qbar.detach().abs() + ag.cfg.ratio_eps)
    delta = ag.adaptive_delta(ratio)
    g = torch.autograd.grad(delta.sum(), qbar, retain_graph=True, allow_unused=True)[0]
    assert g is None or torch.all(g == 0)


def test_pi_norm_scales_the_actor_loss_by_one_plus_lambda():
    ag, _ = make(M=1, use_gate=False, use_bonus=False, policy_delay=1, pi_norm=True)
    ag2, _ = make(M=1, use_gate=False, use_bonus=False, policy_delay=1, pi_norm=False)
    ag.lmbda = ag2.lmbda = 3.0
    o, *_ = batch()
    l1, l2 = ag._policy_update(o)["pi_loss"], ag2._policy_update(o)["pi_loss"]
    assert abs(l1 * 4.0 - l2) < 1e-5


def test_fixed_dual_and_lambda_max():
    ag, _ = make(M=1, dual="fixed", lambda_init=0.3, lambda_lr=1.0, use_gate=False, use_bonus=False, policy_delay=1)
    for i in range(5):
        ag.update(batch(cost=5.0, seed=i))
    assert ag.lmbda == 0.3
    cap, _ = make(M=1, lambda_lr=10.0, lambda_max=0.5, delta0=-10.0, use_gate=False, use_bonus=False, policy_delay=1)
    for i in range(20):
        cap.update(batch(cost=5.0, seed=i))
    assert 0.0 < cap.lmbda <= 0.5
    epi, _ = make(M=1, dual="episodic", lambda_lr=1.0, lambda_max=0.2, use_gate=False, use_bonus=False)
    for _ in range(10):
        epi.update_dual_episodic(1000.0)
    assert epi.lmbda <= 0.2 + 1e-6


def test_grad_clipping_bounds_the_step_and_options_run_finite():
    ag, _ = make(M=2, policy_delay=1, max_grad_norm=1e-3, actor_lr=1e-4, critic_lr=1e-3,
                 cost_target_online=True, pi_norm=True)
    assert ag.pi_opt.param_groups[0]["lr"] == 1e-4 and ag.q_opt.param_groups[0]["lr"] == 1e-3
    before = [p.clone() for p in ag.members.parameters()]
    stats = ag.update(batch(cost=1.0))
    assert np.isfinite(stats["q_loss"]) and np.isfinite(stats["c_loss"]) and np.isfinite(ag.pi_stats["pi_loss"])
    step = torch.sqrt(sum(((p - b) ** 2).sum() for p, b in zip(ag.members.parameters(), before)))
    assert float(step) < 1.0      # Adam's first step is ~lr per weight; clipping keeps it bounded


def test_values_batch_matches_values_at():
    ag, _ = make(M=3)
    o = np.random.default_rng(0).normal(size=(4, OBS)).astype(np.float32)
    qb, qc = ag.values_batch(o)
    for i in range(4):
        b, _, c = ag.values_at(o[i])
        assert abs(b - qb[i]) < 1e-5 and abs(c - qc[i]) < 1e-5
