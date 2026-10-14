"""safe2.probe (SPEC section 10, tests 36-39 and 42, plus the G0a / G0b / S0-1 plumbing).

Every rollout here uses the development layouts 90500+ ("dev") and tiny scratch networks: no Stage-0
gate is run and no study output is read."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import mujoco
import numpy as np
import pytest
import torch

from lagu.agent import Config, LagU
from lagu.train import Replay as LaguReplay
from lagu.train import save_snapshot
from safe2 import configs, evaluate, probe, seeds
from safe2.base import Agent
from safe2.train import Replay

ENV = "SafetyPointGoal1-v0"
FAKE = "safe2.tests.fake_posthoc"


# --- helpers -------------------------------------------------------------------------------------------
def place(task, xy, vel=(0.0, 0.0)) -> None:
    """Put the Point agent at world xy with world velocity vel (its slide joints follow the body's
    initial heading, so the joint frame is solved for)."""
    d = task.data
    J = np.stack([d.xaxis[0][:2], d.xaxis[1][:2]], 1)
    d.qpos[0:2] += np.linalg.solve(J, np.asarray(xy, float) - task.agent.pos[:2])
    d.qvel[:] = 0.0
    d.qvel[0:2] = np.linalg.solve(J, np.asarray(vel, float))
    mujoco.mj_forward(task.model, d)


def steer(task, t: int, rng) -> np.ndarray:
    """Scripted Point actions that reach hazards: 60 steps at the nearest hazard (privileged), 40 random."""
    if t % 100 >= 60:
        return rng.uniform(-1, 1, 2)
    hz = np.array(task.hazards.pos)
    v = (hz[np.argmin([task.agent.dist_xy(p) for p in hz])] - task.agent.pos) @ task.agent.mat
    th = np.arctan2(v[1], v[0])
    return np.clip([np.cos(th), 3.0 * th], -1, 1) + rng.normal(0, 0.1, 2)


class Seeker:
    """A deterministic policy (evaluate.BasePolicy's interface) that drives at the nearest hazard (privileged)."""

    def __init__(self, env):
        self.task = env.unwrapped.task

    def begin(self, lid) -> None:
        pass

    def observe(self, c) -> None:
        pass

    def act(self, o) -> np.ndarray:
        t = self.task
        hz = np.array(t.hazards.pos)
        v = (hz[np.argmin([t.agent.dist_xy(p) for p in hz])] - t.agent.pos) @ t.agent.mat
        th = np.arctan2(v[1], v[0])
        return np.clip([np.cos(th), 3.0 * th], -1, 1).astype(np.float32)


def rng_state(task) -> tuple:
    return task.random_generator.random_generator.get_state()


def b1_pt(path: str, seed: int, zero: bool = False) -> None:
    """A safe2 final <stem>.pt of a tiny B1 agent (random init, or an actor that outputs 0)."""
    rc = configs.make("B1", hidden=32)
    torch.manual_seed(seed)
    agent = Agent(60, 2, rc, 1000, seed=seed)
    if zero:
        with torch.no_grad():
            agent.actor.net[-1].weight.zero_()
            agent.actor.net[-1].bias.zero_()
    torch.save({"config": asdict(rc), "obs_dim": 60, "act_dim": 2, "horizon": 1000, "step": 1000,
                "agent": agent.state_dict()}, path)


def fill(rep, total: int, rng) -> list:
    """Insert `total` random transitions; returns them in insertion order."""
    rows = []
    for g in range(total):
        row = dict(o=rng.normal(size=60), a=rng.uniform(-1, 1, 2), r=rng.normal(), c=float(rng.random() < 0.2),
                   o2=rng.normal(size=60), d=0.0)
        rows.append(row)
        if isinstance(rep, Replay):
            rep.add(o=row["o"], b=1.0, a_pi=row["a"], a_exec=row["a"], r=row["r"], c=row["c"], o2=row["o2"], b2=1.0,
                    d=row["d"], ep=g // 10, t=g % 10, I=0)
        else:
            rep.add(row["o"], row["a"], row["r"], row["c"], row["o2"], row["d"])
    return rows


# --- test 36: the oracle step reproduces env.step -----------------------------------------------------------
def test_step_physics_and_step_cost_reproduce_env_step():
    """SPEC test 36: on dev layout 90500, step_physics + step_cost on a second environment reproduce env.step's
    next observation (task.obs()) and cost bit for bit at every step up to the first goal_met (<= 300).
    step_cost runs before task.obs(), as in env.step: its mj_forward moves the solver's warm start."""
    env_a, env_b = probe.make_env(ENV), probe.make_env(ENV)
    oa, _ = seeds.reset(env_a, "dev", 0)
    ob, _ = seeds.reset(env_b, "dev", 0)
    ta, tb = env_a.unwrapped.task, env_b.unwrapped.task
    assert np.array_equal(oa, ob)
    rng = np.random.default_rng(3)
    costs = steps = 0
    for t in range(300):
        a = steer(ta, t, rng)
        oa, r, c, term, trunc, info = env_a.step(a)
        if info.get("goal_met"):
            break
        probe.step_physics(tb, a)
        assert c == probe.step_cost(tb), f"cost differs at step {steps}"
        assert np.array_equal(oa, tb.obs()), f"observation differs at step {steps}"
        costs += int(c > 0)
        steps += 1
        if term or trunc:
            break
    assert steps >= 100 and costs > 0           # the comparison covered hazard costs


def test_saving_exploring_and_restoring_leaves_the_episode_unchanged():
    """The G0a / G0b oracles explore from a saved state and restore it before env.step: the episode is then
    identical, bit for bit (observation, reward, cost, flags), to one that never explored. The oracle never
    draws from task.random_generator, so goal resampling would stay in step as well."""
    env_a, env_c = probe.make_env(ENV), probe.make_env(ENV)
    seeds.reset(env_a, "dev", 1)
    seeds.reset(env_c, "dev", 1)
    tc = env_c.unwrapped.task
    acts = probe.Actions(ENV)
    rng = np.random.default_rng(0)
    costs = 0
    for t in range(300):
        a = steer(tc, t, rng)
        s = probe.save(tc)
        probe.viable(tc, s, rng.uniform(-1, 1, 2), 3, acts.evasive)
        probe.oracle_label(tc, s, a, 4, acts.evasive)
        probe.load(tc, s)
        ra, rc_ = env_a.step(a), env_c.step(a)
        assert np.array_equal(ra[0], rc_[0]) and ra[1:5] == rc_[1:5], f"step {t} differs"
        costs += int(ra[2] > 0)
    assert costs > 0


def test_oracle_helpers_never_draw_from_the_task_rng():
    """Review F6 / T4: viable, oracle_label, value_score and task.obs() leave task.random_generator untouched
    (goal resampling after an explored goal_met would otherwise desynchronise the episode)."""
    env = probe.make_env(ENV)
    seeds.reset(env, "dev", 0)
    task = env.unwrapped.task
    acts, cfg = probe.Actions(ENV), probe.G0aConfig()
    pol = Seeker(env)
    for _ in range(40):                                           # near a hazard
        env.step(pol.act(None))
    before = rng_state(task)
    s0 = probe.save(task)
    probe.viable(task, s0, np.array([1.0, 0.0]), 25, acts.evasive)
    probe.oracle_label(task, s0, np.array([1.0, 0.0]), 31, acts.evasive)
    probe.value_score(task, s0, acts.value_cands[0], pol, cfg)
    task.obs()
    probe.load(task, s0)
    after = rng_state(task)
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


# --- test 37: viable on constructed states ---------------------------------------------------------------
def test_viable_on_constructed_states():
    env = probe.make_env(ENV)
    seeds.reset(env, "dev", 0)
    task = env.unwrapped.task
    acts = probe.Actions(ENV)
    hz = np.array(task.hazards.pos)[:, :2]
    h = hz[3]                                                     # its nearest neighbour is 0.69 away
    fwd = task.agent.mat[:2, 0]                                   # the body's forward axis
    stay = np.zeros(2)
    place(task, h - fwd * (task.hazards.size + 0.05))             # 0.05 outside, facing the hazard
    s0 = probe.save(task)
    assert probe.hazard_distance(task) == pytest.approx(0.05)
    assert probe.viable(task, s0, stay, 25, [np.array([1.0, 0.0])]) is False    # forward only: drives in
    assert probe.viable(task, s0, stay, 25, [np.array([-1.0, 0.0])]) is True    # backward only: escapes
    assert probe.viable(task, s0, stay, 25, acts.evasive) is True
    assert probe.viable(task, s0, stay, 0, [np.array([1.0, 0.0])]) is True      # H = 0: only the action counts
    place(task, h - fwd * (task.hazards.size + 0.005), vel=fwd * 3.0)          # fast into it: the action enters
    s1 = probe.save(task)
    assert probe.viable(task, s1, stay, 25, acts.evasive) is False
    place(task, h)                                                # at the centre: nothing is viable
    s2 = probe.save(task)
    assert probe.hazard_distance(task) < 0
    assert not any(probe.viable(task, s2, a, 25, acts.evasive) for a in acts.value_cands)


def test_candidate_sets_follow_the_design_script():
    assert len(probe.EVASIVE) == 8 and probe.EVASIVE[0] == (-1, 0)
    assert len(probe.GRID) == 25 and probe.GRID[0] == (1, -1) and probe.GRID[12] == (0, 0) and probe.GRID[24] == (-1, 1)
    assert probe.VALUE_CANDS == probe.GRID[::3] + (probe.GRID[12],) and len(set(probe.VALUE_CANDS)) == 9
    assert np.array_equal(probe.action("SafetyCarGoal1-v0", (0.5, 1)), [-0.5, 1.0])     # car wheels f - w, f + w
    assert np.array_equal(probe.action(ENV, (0.5, 1)), [0.5, 1.0])


def test_oracle_filter_selection(monkeypatch):
    """Keep a viable proposal; otherwise the best-scoring viable candidate (ties to the first); EVASIVE[0]
    when none is viable; the state is restored in every case."""
    env = probe.make_env(ENV)
    seeds.reset(env, "dev", 0)
    task = env.unwrapped.task
    acts = probe.Actions(ENV)
    cfg = probe.G0aConfig()
    a_pi = np.array([0.3, 0.3])
    ok = {}
    scores = {}

    def fake_viable(task_, s0, a, H, evasive):
        task_.data.qpos[0] += 1.0                                  # explore (the filter must restore)
        return ok.get(tuple(np.round(a, 3)), False)

    def fake_score(task_, s0, c, policy, cfg_):
        return scores[tuple(np.round(c, 3))]

    monkeypatch.setattr(probe, "viable", fake_viable)
    monkeypatch.setattr(probe, "value_score", fake_score)
    qpos = task.data.qpos.copy()
    ok[(0.3, 0.3)] = True
    assert probe.oracle_filter(task, None, a_pi, cfg, acts)[1:] == (False, False)
    assert np.array_equal(task.data.qpos, qpos)
    ok.clear()
    c = [tuple(np.round(x, 3)) for x in acts.value_cands]
    for i, k in enumerate(c):
        scores[k] = -float(i)
    ok[c[2]] = ok[c[5]] = ok[c[7]] = True
    scores[c[5]] = scores[c[7]] = 1.0                              # tie: the first (index 5) wins
    a, hit, fb = probe.oracle_filter(task, None, a_pi, cfg, acts)
    assert np.array_equal(a, acts.value_cands[5]) and hit and not fb
    assert np.array_equal(task.data.qpos, qpos)
    ok.clear()
    a, hit, fb = probe.oracle_filter(task, None, a_pi, cfg, acts)
    assert np.array_equal(a, acts.evasive[0]) and hit and fb


def test_oracle_filter_on_real_physics(tmp_path):
    """A forward proposal at a hazard's edge: kept when viable; replaced by the viable candidate with the
    best privileged value when not; EVASIVE[0] when no candidate is viable. The state is restored."""
    b1_pt(str(tmp_path / "a.pt"), 900)
    policy, _ = probe.load_policy(str(tmp_path / "a.pt"))
    env = probe.make_env(ENV)
    seeds.reset(env, "dev", 0)
    task = env.unwrapped.task
    acts, cfg = probe.Actions(ENV), probe.G0aConfig()
    h, fwd = np.array(task.hazards.pos)[3, :2], task.agent.mat[:2, 0]
    go = np.array([1.0, 0.0])
    for gap, speed, expect in ((0.1, 0.5, "keep"), (0.05, 0.5, "pick"), (0.02, 1.0, "fallback")):
        place(task, h - fwd * (task.hazards.size + gap), vel=fwd * speed)
        s0 = probe.save(task)
        a, hit, fb = probe.oracle_filter(task, policy, go, cfg, acts)
        s1 = probe.save(task)
        assert all(np.array_equal(x, y) for x, y in zip(s0, s1))      # restored
        if expect == "keep":
            assert a is go and not hit and not fb
            continue
        ok = [c for c in acts.value_cands if probe.viable(task, s0, c, cfg.H, acts.evasive)]
        if expect == "fallback":
            assert not ok and np.array_equal(a, acts.evasive[0]) and hit and fb
            continue
        scores = [probe.value_score(task, s0, c, policy, cfg) for c in ok]
        assert 0 < len(ok) < len(acts.value_cands) and hit and not fb
        assert np.array_equal(a, ok[int(np.argmax(scores))])
        probe.load(task, s0)


# --- G0a ------------------------------------------------------------------------------------------------
def test_g0a_episode_with_exploring_filter_equals_a_plain_rollout(tmp_path, monkeypatch):
    """With a filter that explores (viable and value rollouts) but keeps the proposal, a G0a episode returns
    exactly what evaluate.rollout gives for the same actor and layout."""
    b1_pt(str(tmp_path / "a.pt"), 900)
    policy, _ = probe.load_policy(str(tmp_path / "a.pt"))
    cfg = probe.G0aConfig(block="dev", B=0.0, near=float("inf"), H=3)

    def exploring(task, policy_, a, cfg_, acts):
        s0 = probe.save(task)
        probe.viable(task, s0, a, cfg_.H, acts.evasive)
        probe.value_score(task, s0, acts.value_cands[0], policy_, cfg_)
        probe.load(task, s0)
        return a, False, False

    monkeypatch.setattr(probe, "oracle_filter", exploring)
    env = probe.make_env(ENV)
    ep = probe.g0a_episode(env, policy, 2, cfg, probe.Actions(ENV))
    ref = evaluate.rollout(probe.make_env(ENV), probe.load_policy(str(tmp_path / "a.pt"))[0], "dev", [2])[0]
    assert (ep["return"], ep["cost"], ep["length"], ep["layout"]) == (ref["return"], ref["cost"], ref["length"],
                                                                     ref["layout"])


def test_g0a_episode_with_the_real_filter_intervenes_and_lowers_cost():
    """Review T3: the real oracle_filter inside g0a_episode, with a hazard-seeking proposer and B = 0, intervenes
    and lowers the episode cost below the unfiltered episode's (dev layout 90507)."""
    acts = probe.Actions(ENV)
    env = probe.make_env(ENV)
    plain = probe.g0a_episode(env, Seeker(env), 7, probe.G0aConfig(block="dev", B=1e9), acts)
    env = probe.make_env(ENV)
    filt = probe.g0a_episode(env, Seeker(env), 7, probe.G0aConfig(block="dev", B=0.0), acts)
    assert plain["interventions"] == 0 and plain["cost"] > 0
    assert filt["interventions"] > 0 and filt["cost"] < plain["cost"] and filt["length"] == plain["length"] == 1000


def test_g0a_writes_per_episode_results_and_input_hashes(tmp_path):
    runs, out = tmp_path / "p1", tmp_path / "stage0"
    runs.mkdir()
    for s in (900, 901):
        b1_pt(str(runs / f"B1_{ENV}_s{s}.pt"), s, zero=True)       # stands still: no interventions
    cfg = probe.G0aConfig(block="dev", episodes=1, B=0.0)
    res = probe.g0a(str(runs), str(out), cfg, run_seeds=(900, 901), workers=1)
    with open(out / "g0a.json") as f:
        saved = json.load(f)
    assert saved["n_episodes"] == 2 and len(saved["actors"]) == 2
    assert set(saved["inputs"]) == {str(runs / f"B1_{ENV}_s{s}.pt") for s in (900, 901)}
    assert all(len(h) == 64 for h in saved["inputs"].values()) and "safe2/probe.py" in saved["code"]
    ep = saved["actors"][0]["episodes"][0]
    assert ep["layout"] == seeds.layout("dev", 0) and ep["interventions"] == 0 and ep["length"] == 1000
    assert res["mean_return"] == pytest.approx(np.mean([a["episodes"][0]["return"] for a in saved["actors"]]))
    assert saved["label"].startswith("PRIVILEGED")
    with pytest.raises(SystemExit, match="missing input"):
        probe.g0a(str(runs), str(tmp_path / "other"), cfg, run_seeds=(902,), workers=1)


def test_gate_guard_refuses_dev_settings_under_results_method_and_silent_overwrites(tmp_path):
    """Review F5: under results/method/ only the preregistered settings and selection seeds run; an existing
    gate json is never overwritten silently, and a forced overwrite there is logged in deviations.json."""
    out = tmp_path / "results" / "method" / "stage0"
    with pytest.raises(SystemExit, match="non-default"):
        probe.guard_gate(str(out), "g0a", probe.G0aConfig(block="dev"), (0, 1, 2))
    with pytest.raises(SystemExit, match="non-default"):
        probe.guard_gate(str(out), "g0b", probe.G0bConfig(env="SafetyCarGoal1-v0"), (0, 1, 2))
    for bad in ((0, 1, 300), (900,), ()):
        with pytest.raises(SystemExit, match="run seeds"):
            probe.guard_gate(str(out), "g0a", probe.G0aConfig(), bad)
    probe.guard_gate(str(out), "g0a", probe.G0aConfig(), (0, 1, 2))           # allowed, nothing to overwrite
    probe.write_json(str(out / "g0a.json"), {"old": 1})
    old = probe.sha256(str(out / "g0a.json"))
    with pytest.raises(SystemExit, match="--force"):
        probe.guard_gate(str(out), "g0a", probe.G0aConfig(), (0, 1, 2))
    probe.guard_gate(str(out), "g0a", probe.G0aConfig(), (0, 1, 2), force=True)
    with open(tmp_path / "results" / "method" / "deviations.json") as f:
        dev = json.load(f)
    assert dev == [{"overwritten": os.path.join("results", "method", "stage0", "g0a.json"), "old_sha256": old,
                    "by": "safe2.probe g0a --force", "entry": None}]
    elsewhere = tmp_path / "scratch"                                     # outside results/method: dev settings ok
    probe.guard_gate(str(elsewhere), "g0a", probe.G0aConfig(block="dev"), (900,))
    probe.write_json(str(elsewhere / "g0a.json"), {})
    with pytest.raises(SystemExit, match="--force"):
        probe.guard_gate(str(elsewhere), "g0a", probe.G0aConfig(block="dev"), (900,))


def test_g0a_decision_uses_max_18_and_rbstar_plus_6():
    """SPEC test 42."""
    assert probe.g0a_decision(18.0, 25.0, 5.0)["proceed"]                 # threshold max(18, 11) = 18
    assert not probe.g0a_decision(17.9, 10.0, 5.0)["proceed"]
    assert probe.g0a_decision(19.9, 10.0, 14.0)["return_threshold"] == 20.0
    assert not probe.g0a_decision(19.9, 10.0, 14.0)["proceed"]            # 14 + 6 = 20 > 18
    assert probe.g0a_decision(20.0, 10.0, 14.0)["proceed"]
    assert not probe.g0a_decision(30.0, 25.01, 5.0)["proceed"]            # cost above 25
    assert not probe.g0a_decision(float("nan"), 10.0, 5.0)["proceed"]


# --- G0b ------------------------------------------------------------------------------------------------
def test_insertion_index_follows_lagu_replay_order():
    rep = {"o": np.zeros((30, 1)), "ptr": 10, "n": 30}             # 40 insertions into 30 rows: 0..9 overwritten
    assert probe.insertion_index(rep, 40, 0, 5) is None
    assert list(probe.insertion_index(rep, 40, 10, 15)) == [10, 11, 12, 13, 14]
    assert list(probe.insertion_index(rep, 40, 35, 40)) == [5, 6, 7, 8, 9]   # insertion g lives at g % 30
    assert probe.insertion_index(rep, 40, 35, 41) is None
    assert list(probe.insertion_index({"o": np.zeros((50, 1)), "ptr": 20, "n": 20}, 20, 0, 3)) == [0, 1, 2]


def test_replay_identity_reports_identical_and_the_first_differing_row(tmp_path):
    rng = np.random.default_rng(0)
    a, b = Replay(60, 2, 40, extras={"a_exec": ((2,), np.float32), "I": ((), np.uint8)}), LaguReplay(60, 2, 40)
    rows = fill(a, 35, rng)
    for row in rows:
        b.add(row["o"], row["a"], row["r"], row["c"], row["o2"], row["d"])
    pa, pb = str(tmp_path / "b1.ckpt.pt"), str(tmp_path / "lagu.ckpt.pt")
    torch.save({"replay": a.state(), "step": 35}, pa)
    torch.save({"replay": b.state(), "step": 35, "ep_rets": [np.float64(1.5)]}, pb)   # lagu ckpts hold numpy scalars
    assert probe.replay_identity(pa, pb, 30) == {"result": "identical", "rows": 30}
    b.r[7] += 0.5
    b.o2[7, 3] = -b.o2[7, 3]
    b.c[12] = 1.0 - b.c[12]
    torch.save({"replay": b.state(), "step": 35}, pb)
    res = probe.replay_identity(pa, pb, 30)
    assert res["result"] == "differs" and res["first_row"] == 7 and set(res["columns"]) == {"r", "o2"}
    assert probe.replay_identity(pa, pb, 7)["result"] == "identical"
    assert probe.replay_identity(pa, pb, 36)["result"] == "unavailable"


def test_replay_identity_reads_legacy_lagu_checkpoints_with_numpy_columns(tmp_path):
    """Older lagu.train checkpoints (pareto td3lag-omni s0, s1) pickle the replay columns as numpy arrays."""
    rng = np.random.default_rng(1)
    a = Replay(60, 2, 40, extras={"a_exec": ((2,), np.float32), "I": ((), np.uint8)})
    rows = fill(a, 35, rng)
    cols = {k: np.zeros((40,) + s, np.float32) for k, s in (("o", (60,)), ("a", (2,)), ("r", ()), ("c", ()),
                                                             ("o2", (60,)), ("d", ()))}
    for g, row in enumerate(rows):
        for k in cols:
            cols[k][g] = row[k]
    pa, pb = str(tmp_path / "b1.ckpt.pt"), str(tmp_path / "legacy.ckpt.pt")
    torch.save({"replay": a.state(), "step": 35}, pa)
    torch.save({"replay": {**cols, "ptr": 35, "n": 35}, "step": 35, "ep_rets": [np.float64(1.0)]}, pb)
    assert probe.replay_identity(pa, pb, 30) == {"result": "identical", "rows": 30}
    cols["c"][9] = 1.0 - cols["c"][9]
    torch.save({"replay": {**cols, "ptr": 35, "n": 35}, "step": 35, "ep_rets": [np.float64(1.0)]}, pb)
    assert probe.replay_identity(pa, pb, 30) == {"result": "differs", "rows": 30, "first_row": 9, "columns": ["c"]}


def test_state_sampler_rule():
    """SPEC test 38 (sampling): at most 20 states per episode, at least 10 steps apart, near hazards only."""
    s = probe.StateSampler(0.6, 20, 10)
    taken = [t for t in range(1000) if s.take(t, 0.1)]
    assert taken == list(range(0, 200, 10))
    s.begin()
    hd = lambda t: 0.1 if t % 7 else 0.9
    taken = [t for t in range(300) if s.take(t, hd(t))]
    assert all(b - a >= 10 for a, b in zip(taken, taken[1:])) and all(hd(t) < 0.6 for t in taken)
    assert taken[:3] == [1, 11, 22]                                  # 21 is a multiple of 7: not near
    s.begin()
    assert not s.take(5, 0.6) and s.take(6, 0.59)


def test_g0b_decision_fails_with_fewer_than_1000_states():
    """SPEC test 38 (gate): fewer than 1000 pooled states fails (ii), whatever the correlation."""
    good = {"recall": 0.9, "fa_per_episode": 10.0}
    assert probe.g0b_decision(good, 1000, 0.7)["proceed"]
    assert not probe.g0b_decision(good, 999, 1.0)["pass_ii"]
    assert not probe.g0b_decision(good, 5000, 0.69)["proceed"]
    assert not probe.g0b_decision(good, 5000, float("nan"))["proceed"]
    assert not probe.g0b_decision({"recall": 0.79, "fa_per_episode": 1.0}, 5000, 0.9)["pass_i"]
    assert not probe.g0b_decision({"recall": 0.9, "fa_per_episode": 30.01}, 5000, 0.9)["pass_i"]
    assert probe.g0b_decision({"recall": 0.8, "fa_per_episode": 30.0}, 5000, 0.9)["pass_i"]


def test_spearman_with_ties_against_a_hand_computed_case():
    """SPEC test 39: x ranks (1, 2.5, 2.5, 4), y ranks (1, 3, 2, 4): rho = 4.5 / sqrt(4.5 * 5) = sqrt(0.9)."""
    assert probe.spearman([1, 2, 2, 3], [10, 30, 20, 40]) == pytest.approx(np.sqrt(0.9), abs=1e-12)
    assert probe.spearman([3, 1, 2], [1, 3, 2]) == pytest.approx(-1.0)
    assert np.isnan(probe.spearman([1, 1, 1], [1, 2, 3])) and np.isnan(probe.spearman([1], [2]))
    assert list(probe.avg_ranks(np.array([5.0, 1.0, 5.0, 3.0]))) == [3.5, 1.0, 3.5, 2.0]


def test_non_finite_predictions_fail_the_ranking_gate():
    """A NaN would rank as the largest value and still give a finite, high rho (review F3)."""
    assert np.isnan(probe.spearman([1, 2, np.nan, np.nan, 3, 4], [1, 2, 5, 6, 3, 4]))
    assert np.isnan(probe.spearman([1, 2, 3], [1, np.inf, 3]))
    good = {"recall": 0.9, "fa_per_episode": 10.0}
    assert probe.g0b_decision(good, 5000, 0.9, nonfinite_pred=0)["pass_ii"]
    assert not probe.g0b_decision(good, 5000, 0.9, nonfinite_pred=1)["pass_ii"]


def test_pool_recall_pools_counts_not_rates():
    per = [{"entries": 10, "caught": 9, "false_alarm_steps": 300.0, "episodes": 100},
           {"entries": 30, "caught": 21, "false_alarm_steps": 900.0, "episodes": 100}]
    p = probe.pool_recall(per)
    assert p["recall"] == pytest.approx(30 / 40) and p["fa_per_episode"] == pytest.approx(6.0)
    assert np.isnan(probe.pool_recall([{"entries": 0, "caught": 0, "false_alarm_steps": 0, "episodes": 1}])["recall"])


def _brute_label(task, s0, a, steps, evasive) -> float:
    """oracle_label written out without its early exits."""
    probe.load(task, s0)
    probe.step_physics(task, a)
    c0 = probe.step_cost(task)
    s1 = probe.save(task)
    totals = []
    for e in evasive:
        probe.load(task, s1)
        tot = 0.0
        for _ in range(steps):
            probe.step_physics(task, e)
            tot += probe.step_cost(task)
        totals.append(tot)
    return c0 + min(totals)


def test_oracle_label_is_the_step_cost_plus_the_best_evasive_cost():
    env = probe.make_env(ENV)
    seeds.reset(env, "dev", 0)
    task = env.unwrapped.task
    acts = probe.Actions(ENV)
    h = np.array(task.hazards.pos)[3, :2]
    fwd = task.agent.mat[:2, 0]
    fwd_only = [np.array([1.0, 0.0])]
    labels = []
    for xy, vel in ((h, (0, 0)), (h - fwd * 0.1, (0, 0)), (h - fwd * (task.hazards.size + 0.05), fwd * 1.0),
                    (h - fwd * (task.hazards.size + 0.3), (0, 0))):
        place(task, xy, vel)
        s0 = probe.save(task)
        for ev in (acts.evasive, fwd_only):
            for a in (np.zeros(2), np.array([1.0, 0.0])):
                lab = probe.oracle_label(task, s0, a, 31, ev)
                assert lab == _brute_label(task, s0, a, 31, ev) and 0 <= lab <= 32
                labels.append(lab)
    assert labels[0] > 0 and labels[-4] == 0.0 and len(set(labels)) > 3


def test_g0b_rank_seed_ties_prediction_and_label_to_the_recorded_state():
    """Review T2: replaying the deterministic actor on each recorded layout up to t gives the recorded
    prediction vmax(o_t, a_t) and label _brute_label(s_t, a_t) (no off-by-one in o or in the label)."""
    from safe2.tests import fake_posthoc
    vmax = probe.vmax_fn(fake_posthoc._Fit({}))
    env = probe.make_env(ENV)
    acts = probe.Actions(ENV)
    cfg = probe.G0bConfig(block="dev", states_per_seed=12, max_episodes=1)
    rec = probe.g0b_rank_seed(env, Seeker(env), vmax, cfg, acts)
    assert rec["states"] >= 5 and len(set(rec["label"])) > 1
    env2 = probe.make_env(ENV)
    task, pol = env2.unwrapped.task, Seeker(env2)
    o, _ = seeds.reset(env2, "dev", 0)
    t, k = 0, 0
    while k < rec["states"]:
        a = pol.act(o)
        if rec["t"][k] == t:
            assert rec["layout"][k] == seeds.layout("dev", 0)
            assert rec["pred"][k] == float(vmax(o[None], a[None])[0])
            s0 = probe.save(task)
            assert rec["label"][k] == _brute_label(task, s0, a, cfg.label_steps, acts.evasive)
            probe.load(task, s0)
            k += 1
        o, *_ = env2.step(a)
        t += 1


def _g0b_inputs(tmp_path, seeds_=(900, 901)):
    runs, pareto = tmp_path / "p1", tmp_path / "pareto"
    runs.mkdir()
    pareto.mkdir()
    for s in seeds_:
        st = f"B1_{ENV}_s{s}"
        b1_pt(str(runs / f"{st}.pt"), s)
        rng = np.random.default_rng(s)
        a, b = Replay(60, 2, 60), LaguReplay(60, 2, 60)
        for row in fill(a, 60, rng):
            b.add(row["o"], row["a"], row["r"], row["c"], row["o2"], row["d"])
        if s == seeds_[-1]:
            b.d[20] = 1.0
        torch.save({"replay": a.state(), "step": 60}, runs / f"{st}.ckpt.pt")
        torch.save({"replay": b.state(), "step": 60}, pareto / f"td3lag-omni_{ENV}_s{s}.ckpt.pt")
    return runs, pareto


@pytest.mark.parametrize("workers", [1, 2])
def test_g0b_end_to_end_with_a_fake_posthoc(tmp_path, workers):
    runs, pareto = _g0b_inputs(tmp_path)
    out = tmp_path / "stage0"
    cfg = probe.G0bConfig(block="dev", fit_episodes=(2, 4), holdout_episodes=(5, 6), updates=7, identity_rows=50,
                          states_per_seed=25, max_episodes=2, min_states=10)
    res = probe.g0b(str(runs), str(pareto), str(out), cfg, run_seeds=(900, 901), workers=workers, backend=FAKE)
    with open(out / "g0b.json") as f:
        saved = json.load(f)
    st0, st1 = f"B1_{ENV}_s900", f"B1_{ENV}_s901"
    assert saved["identity"][st0]["result"] == "identical"
    assert saved["identity"][st1] == {"result": "differs", "rows": 50, "first_row": 20, "columns": ["d"]}
    for st, s in ((st0, 900), (st1, 901)):
        meta = torch.load(out / "g0b" / f"{st}.posthoc.pt", weights_only=True)
        assert meta == {"ckpt": str(runs / f"{st}.ckpt.pt"), "ckpt_sha256": probe.sha256(runs / f"{st}.ckpt.pt"),
                        "episodes": [2, 4], "updates": 7, "seed": s}
        assert saved["recall"]["per_run"][st]["episodes"] == 2
    pooled = saved["recall"]["pooled"]
    assert (pooled["entries"], pooled["caught"], pooled["episodes"]) == (20, 17, 4)
    assert pooled["fa_per_episode"] == pytest.approx((0.0 + 3.0) / 4)
    for r in saved["ranking"]["per_run"]:
        assert r["states"] == len(r["pred"]) == len(r["label"]) <= 25 and r["episodes"] <= 2
        for lid in set(r["layout"]):
            ts = [t for t, l in zip(r["t"], r["layout"]) if l == lid]
            assert len(ts) <= 20 and all(b - a >= 10 for a, b in zip(ts, ts[1:]))
        assert all(0.0 <= x <= 32.0 and x == int(x) for x in r["label"])
    n = sum(r["states"] for r in saved["ranking"]["per_run"])
    assert saved["ranking"]["states"] == n and n > 0
    assert saved["ranking"]["nonfinite_pred"] == 0
    assert saved["decision"] == probe.g0b_decision(pooled, n, saved["ranking"]["spearman"], cfg)
    assert len(saved["inputs"]) == 6 and len(saved["fits"]) == 2
    assert res["decision"]["pass_ii"] == (n >= 10 and saved["ranking"]["spearman"] >= 0.7)


def test_g0b_reuse_fits_and_the_cli_success_path(tmp_path, monkeypatch, capsys):
    """Review T6: --reuse-fits uses the fits already in <out>/g0b/ without refitting (and needs them); a re-run
    needs --force; main() prints the identity, (i), (ii) and the decision."""
    runs, pareto = _g0b_inputs(tmp_path)
    out = tmp_path / "stage0"
    cfg = probe.G0bConfig(block="dev", fit_episodes=(2, 4), holdout_episodes=(5, 6), updates=7, identity_rows=50,
                          states_per_seed=5, max_episodes=1, min_states=1)
    kw = dict(run_seeds=(900, 901), workers=1, backend=FAKE)
    with pytest.raises(SystemExit, match="missing input"):
        probe.g0b(str(runs), str(pareto), str(out), cfg, reuse_fits=True, **kw)
    first = probe.g0b(str(runs), str(pareto), str(out), cfg, **kw)
    with pytest.raises(SystemExit, match="--force"):
        probe.g0b(str(runs), str(pareto), str(out), cfg, reuse_fits=True, **kw)
    from safe2.tests import fake_posthoc
    monkeypatch.setattr(fake_posthoc, "fit", lambda *a, **k: pytest.fail("refitted despite reuse_fits"))
    again = probe.g0b(str(runs), str(pareto), str(out), cfg, reuse_fits=True, force=True, **kw)
    assert again["fits"] == first["fits"] and again["decision"] == first["decision"]
    monkeypatch.setattr(probe, "g0b", lambda *a, **k: first)
    assert probe.main(["g0b", "--runs", str(runs), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "identity" in text and "g0b (i)" in text and "g0b (ii)" in text and ("PROCEED" in text or "KILL" in text)


def test_g0b_refuses_missing_inputs_and_an_incomplete_backend(tmp_path):
    runs, pareto = _g0b_inputs(tmp_path, seeds_=(900,))
    with pytest.raises(SystemExit):
        probe.g0b(str(runs), str(pareto), str(tmp_path / "o"), probe.G0bConfig(block="dev"), run_seeds=(900, 901),
                  workers=1, backend=FAKE)
    with pytest.raises(SystemExit, match="load_fit"):
        probe.g0b(str(runs), str(pareto), str(tmp_path / "o"), probe.G0bConfig(block="dev"), run_seeds=(900,),
                  workers=1, backend="safe2.seeds")


def test_g0b_refuses_fits_that_do_not_match_their_run(tmp_path):
    """Review stage0 F4: every fit (fresh or reused) must carry this checkpoint's SHA-256, the fit episodes,
    the updates and seed = run seed (fit_meta); a copied or dev fit is refused."""
    from safe2.tests import fake_posthoc
    runs, pareto = _g0b_inputs(tmp_path)
    out = tmp_path / "stage0"
    cfg = probe.G0bConfig(block="dev", fit_episodes=(2, 4), holdout_episodes=(5, 6), updates=7, identity_rows=50,
                          states_per_seed=2, max_episodes=1, min_states=1)
    kw = dict(run_seeds=(900, 901), workers=1, backend=FAKE)
    probe.g0b(str(runs), str(pareto), str(out), cfg, **kw)
    fits = out / "g0b"
    st0, st1 = f"B1_{ENV}_s900", f"B1_{ENV}_s901"
    for bad in (dict(seed=5), dict(updates=8), dict(episodes=(1, 4))):
        args = dict(updates=7, seed=900, episodes=(2, 4)) | bad
        fake_posthoc.fit(str(runs / f"{st0}.ckpt.pt"), args["episodes"], str(fits / f"{st0}.posthoc.pt"),
                         updates=args["updates"], seed=args["seed"])
        with pytest.raises(SystemExit, match="does not match"):
            probe.g0b(str(runs), str(pareto), str(out), cfg, reuse_fits=True, force=True, **kw)
    fake_posthoc.fit(str(runs / f"{st1}.ckpt.pt"), (2, 4), str(fits / f"{st0}.posthoc.pt"), updates=7, seed=900)
    with pytest.raises(SystemExit, match="ckpt_sha256"):
        probe.g0b(str(runs), str(pareto), str(out), cfg, reuse_fits=True, force=True, **kw)


def _phs_inputs(tmp_path, seed=905):
    from safe2.tests import fake_posthoc
    from safe2.tests.helpers import make_key
    root = tmp_path / "root"
    priv = make_key(root)
    (root / "conf").mkdir()
    (root / "phs").mkdir()
    st = f"B1_{ENV}_s{seed}"
    run = root / "conf" / st
    b1_pt(str(run) + ".pt", seed)
    rep = Replay(60, 2, 60)
    fill(rep, 60, np.random.default_rng(seed))
    torch.save({"replay": rep.state(), "step": 60, "seed": seed, "env": ENV}, str(run) + ".ckpt.pt")
    fake_posthoc.fit(str(run) + ".ckpt.pt", (1, 900), str(root / "phs" / f"{st}.posthoc.pt"), updates=7, seed=seed)
    cfg = probe.G0bConfig(block="dev", fit_episodes=(1, 900), updates=7, states_per_seed=3, max_episodes=1)
    return root, run, st, priv, cfg


def test_phs_quality_is_sealed_and_checks_its_fit(tmp_path):
    """Stage0 F2b: phs-quality runs G0b(i) and (ii) for one B* run with its PHS fit and writes only the
    sealed file; a fit made with other settings is refused; a re-run needs force."""
    from safe2 import seal
    root, run, st, priv, cfg = _phs_inputs(tmp_path)
    path = probe.phs_quality(str(run), cfg=cfg, backend=FAKE)
    assert path == str(root / "sealed" / "phs" / f"{st}.quality.json.enc")
    out = seal.read_file(path, priv)
    assert out["probe"] == "phs-quality" and out["stem"] == st and out["settings"]["fit_episodes"] == [1, 900]
    assert out["recall"]["run"]["episodes"] == 100 and out["recall"]["pooled"]["entries"] == 10
    assert out["ranking"]["episodes"] == 1 and out["ranking"]["states"] == len(out["ranking"]["pred"]) <= 3
    assert out["ranking"]["nonfinite_pred"] == 0 and len(out["inputs"]) == 3
    assert seal.verify(str(root)) == []
    assert sorted(p.name for p in (root / "sealed").rglob("*") if p.is_file()) == [f"{st}.quality.json.enc"]
    with pytest.raises(SystemExit, match="--force"):
        probe.phs_quality(str(run), cfg=cfg, backend=FAKE)
    with pytest.raises(SystemExit, match="does not match"):
        probe.phs_quality(str(run), cfg=probe.G0bConfig(block="dev", fit_episodes=(1, 900)), backend=FAKE, force=True)


def test_phs_quality_cli_prints_no_number(tmp_path, monkeypatch, capsys):
    root, run, st, priv, cfg = _phs_inputs(tmp_path)
    real = probe.phs_quality
    monkeypatch.setattr(probe, "phs_quality", lambda r, root=None, force=False: real(r, root, cfg, FAKE, force))
    assert probe.main(["phs-quality", str(run)]) == 0
    assert capsys.readouterr().out == f"{st}: done\n"


# --- S0-1 statistics ------------------------------------------------------------------------------------
def _brute_auroc(x, y):
    pos, neg = x[y], x[~y]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def _brute_ap(x, y):
    """AUPRC with each distinct score as one threshold, written out directly."""
    P = y.sum()
    out, prev = 0.0, 0
    for thr in sorted(set(x.tolist()), reverse=True):
        sel = x >= thr
        tp = int((sel & y).sum())
        out += (tp - prev) / P * tp / sel.sum()
        prev = tp
    return out


def test_auroc_and_average_precision_against_brute_force_with_ties():
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.integers(0, 6, 80).astype(float)
        y = rng.random(80) < 0.3
        assert probe.auroc(x, y) == pytest.approx(_brute_auroc(x, y))
        assert probe.average_precision(x, y) == pytest.approx(_brute_ap(x, y))
    assert probe.average_precision(np.array([3.0, 2.0, 1.0]), np.array([True, False, True])) == pytest.approx(
        (1 + 2 / 3) / 2)
    assert np.isnan(probe.average_precision(np.ones(3), np.zeros(3, bool)))


def test_cluster_bootstrap_of_auprc_is_exact():
    """boot_ap with run multiplicities w equals the AUPRC of the data with each run repeated w times."""
    rng = np.random.default_rng(2)
    xs = [rng.integers(0, 8, n).astype(float) for n in (30, 50, 40)]
    ys = [rng.random(len(x)) < p for x, p in zip(xs, (0.2, 0.4, 0.1))]
    p, q = probe.ap_counts(xs, ys)
    W = np.array([[1, 1, 1], [2, 0, 1], [0, 0, 3], [0, 3, 0], [1, 0, 0]], float)
    got = probe.boot_ap(p, q, W, chunk=2)
    for w, g in zip(W, got):
        xr = np.concatenate([x for x, k in zip(xs, w) for _ in range(int(k))])
        yr = np.concatenate([y for y, k in zip(ys, w) for _ in range(int(k))])
        assert g == pytest.approx(probe.average_precision(xr, yr))
    U, P, N = probe.pair_matrix(xs, ys)
    bt = probe.boot_auroc(U, P, N, W[:1])
    assert bt[0] == pytest.approx(probe.auroc(np.concatenate(xs), np.concatenate(ys)))
    Wb = probe.weights(3, 7)
    assert Wb.shape == (7, 3) and np.all(Wb.sum(1) == 3)


def test_targets_entries_and_aligned_curves_on_a_hand_built_run():
    cost = np.zeros(40)
    cost[[15, 16, 30]] = 1.0                                       # entries at 15 and 30 (14 quiet steps before)
    cost2 = np.zeros(20)
    cost2[[5, 18]] = 1.0                                           # 5 < 12: not an entry; 18 is one
    ep = np.r_[np.zeros(40, int), np.ones(20, int)]
    use, y = probe.targets(np.r_[cost, cost2], ep, 5)
    assert use[:35].all() and not use[35:40].any() and use[40:55].all() and not use[55:].any()
    assert y[10] and not y[9] and y[29] and not y[16]
    assert probe.entry_steps(cost) == [15, 30] and probe.entry_steps(cost2) == [18]
    run = {"ep": ep, "cost": np.r_[cost, cost2], "v": np.r_[np.arange(40.0), 100 + np.arange(20.0)]}
    cv = probe.aligned_curve([run], "v", -2, 2)
    assert cv["entries"] == 3 and cv["offsets"] == [-2, -1, 0, 1, 2]
    assert cv["n"] == [3, 3, 3, 3, 2]                              # 18 + 2 = 20 is past episode 2's end
    assert cv["mean"][2] == pytest.approx((15 + 30 + 118) / 3)
    assert cv["mean"][4] == pytest.approx((17 + 32) / 2)


def test_lidar_slice_matches_the_unflattened_observation():
    from gymnasium.spaces.utils import unflatten
    env = probe.make_env(ENV)
    o, _ = seeds.reset(env, "dev", 0)
    task = env.unwrapped.task
    sl = probe.lidar_slice(task)
    assert np.array_equal(o[sl], unflatten(task.obs_info.obs_space_dict, o)["hazards_lidar"]) and sl == slice(28, 44)
    h, fwd = np.array(task.hazards.pos)[3, :2], task.agent.mat[:2, 0]
    for gap in (0.05, 0.5):                                       # review T5: against the task's own hazard lidar
        place(task, h - fwd * (task.hazards.size + gap))
        o = task.obs()
        assert np.array_equal(o[sl], task._obs_lidar(task.hazards.pos, task.hazards.group))
        near = o[sl].max() if gap == 0.05 else near
    assert near > o[sl].max() > 0
    assert probe.lidar_slice(probe.make_env("SafetyPointCircle1-v0").unwrapped.task) is None


def test_s01_plan_refuses_missing_snapshots_and_repeated_run_names(tmp_path):
    """Review F8: an incomplete run (no snapshot at N) or a snapshot far from 0.3N / 0.6N is refused, as is
    the same run name in two directories (their traces would overwrite each other)."""
    root = tmp_path / "main"
    _lagu_run(str(root / "results" / "a"), "td3lag-tt", "SafetyPointCircle1-v0", 900, 1, [300, 600, 1000])
    ok = [("td3lag-tt", "results/a", "SafetyPointCircle1-v0", (900,))]
    assert len(probe.s01_plan(str(root), ok)[0]) == 1
    _lagu_run(str(root / "results" / "b"), "td3lag-tt", "SafetyPointCircle1-v0", 900, 1, [300, 600, 1000])
    with pytest.raises(SystemExit, match="repeat"):
        probe.s01_plan(str(root), ok + [("td3lag-tt", "results/b", "SafetyPointCircle1-v0", (900,))])
    os.remove(root / "results" / "a" / "td3lag-tt_SafetyPointCircle1-v0_s900.snap" / "1000.pt")
    with pytest.raises(SystemExit, match="1N"):
        probe.s01_plan(str(root), ok)
    _lagu_run(str(root / "results" / "c"), "td3lag-tt", "SafetyPointCircle1-v0", 901, 1, [600, 100_000])
    with pytest.raises(SystemExit, match="0.3N"):                       # nearest to 30,000 is 600
        probe.s01_plan(str(root), [("td3lag-tt", "results/c", "SafetyPointCircle1-v0", (901,))])


def test_nearest_snapshot_ties_go_earlier():
    assert probe.nearest([100, 200, 300], 150) == 100 and probe.nearest([100, 200, 300], 260) == 300


# --- S0-1 end to end on scratch Lag-U snapshots ------------------------------------------------------------
def _lagu_run(base, label, env, seed, M, steps, heldout_return=None) -> None:
    os.makedirs(base, exist_ok=True)
    e = probe.make_env(env)
    obs_dim, act_dim = int(np.prod(e.observation_space.shape)), int(np.prod(e.action_space.shape))
    torch.manual_seed(seed)
    agent = LagU(obs_dim, act_dim, Config(M=M, hidden=16, use_bonus=M > 1, use_gate=M > 1))
    stem = os.path.join(base, f"{label}_{env}_s{seed}")
    for st in steps:
        save_snapshot(stem + ".snap", agent, st)
    with open(stem + ".json", "w") as f:
        json.dump({"args": {"env": env, "steps": steps[-1]}, "obs_dim": obs_dim, "act_dim": act_dim, "horizon": 1000}, f)
    if heldout_return is not None:
        with open(stem + ".heldout.json", "w") as f:
            json.dump({"pooled": {"eval_return": heldout_return}}, f)


def test_s01_end_to_end_on_scratch_snapshots(tmp_path):
    root, out = tmp_path / "main", tmp_path / "s01"
    gate = root / "results" / "g"
    for s, ret in ((900, 6.0), (901, 1.0)):
        _lagu_run(str(gate), "lagu-tt", ENV, s, 3, [300, 450, 600, 1000], heldout_return=ret)
    _lagu_run(str(root / "results" / "gx"), "td3lag-tt", "SafetyPointCircle1-v0", 900, 1, [300, 600, 1000])
    arms = [("lagu-tt", "results/g", ENV, (900, 901)), ("td3lag-tt", "results/gx", "SafetyPointCircle1-v0", None)]
    res = probe.s01(str(root), str(out), workers=1, arms=arms, block="dev", episodes=1, n_boot=50)
    healthy = f"lagu-tt@{ENV} (held-out return >= 5)"
    assert res["arms"] == {f"lagu-tt@{ENV}": [f"lagu-tt_{ENV}_s900", f"lagu-tt_{ENV}_s901"],
                           healthy: [f"lagu-tt_{ENV}_s900"],
                           "td3lag-tt@SafetyPointCircle1-v0": ["td3lag-tt_SafetyPointCircle1-v0_s900"]}
    snaps = {r["stem"].split(os.sep)[-1]: [s["step"] for s in r["snapshots"]] for r in res["runs"]}
    assert snaps[f"lagu-tt_{ENV}_s900"] == [300, 600, 1000]          # nearest 300, 600 and 1000
    assert len(res["inputs"]) == 2 * (1 + 3 + 1) + (1 + 3)            # jsons, snapshots, heldout.json
    assert res["healthy_runs"] == {"stems": [f"lagu-tt_{ENV}_s900"], "count": 1, "of": 2}
    key = f"lagu-tt@{ENV}|0.3N|k=10"
    assert set(res["scores"][key]) == {"ratio", "std", "qc", "negqbar", "lidar"}
    assert set(res["scores"]["td3lag-tt@SafetyPointCircle1-v0|1N|k=5"]) == {"qc", "negqbar"}
    for name in ("s01.json", "s01.txt", "s01_scores.csv", "s01_curves.csv"):
        assert (out / name).exists()
    with open(out / "s01.txt") as f:
        assert f.readline().startswith("EXPLORATORY")
        assert "runs (held-out return >= 5): 1 of 2" in f.read()
    traces = sorted(os.listdir(out / "traces"))
    assert len(traces) == 9
    with np.load(out / "traces" / traces[0]) as z:
        assert int(z["episodes"]) == 1 and str(z["block"]) == "dev"
    mtimes = {t: os.path.getmtime(out / "traces" / t) for t in traces}
    probe.s01(str(root), str(out), workers=1, arms=arms, block="dev", episodes=1, n_boot=50)
    assert {t: os.path.getmtime(out / "traces" / t) for t in traces} == mtimes     # cached by snapshot hash


# --- CLI ------------------------------------------------------------------------------------------------
def test_cli_refuses_missing_inputs(tmp_path):
    with pytest.raises(SystemExit):
        probe.main(["g0a", "--runs", str(tmp_path), "--out", str(tmp_path / "o"), "--workers", "1"])
    assert not (tmp_path / "o").exists()


# --- S0-4 (review F2) -----------------------------------------------------------------------------------------
def test_s04_labels_follow_the_design_script():
    c = np.zeros(1000)
    c[[100, 520]] = 1.0
    lab = probe.s04_labels(c)
    assert len(lab) == 500 and lab[100] == 2
    assert (lab[90:100] == 1).all() and lab[89] == -1 and lab[69] == 0 and lab[70] == -1   # 70 + 30 = 100
    assert lab[101] == 0 and lab[489] == 0 and lab[490] == -1 and lab[499] == -1      # 520 is past 499 + 10


def _s04_ckpt(path, kind: str, lengths, seed: int = 0) -> list:
    """A checkpoint with a tiny random agent and a replay of chained episodes of the given lengths (o2[i] = o[i+1]
    inside an episode). Returns the insertion indices of every episode's first row."""
    rng = np.random.default_rng(seed)
    total = sum(lengths)
    torch.manual_seed(seed)
    if kind == "safe2":
        rc = configs.make("SHL-g0.90-B4", hidden=16)
        agent = Agent(60, 2, rc, 1000, seed=seed)
        agent.lmbda = 0.7
        rep = Replay(60, 2, total + 5, extras={"a_exec": ((2,), np.float32), "I": ((), np.uint8)})
    else:
        agent = LagU(60, 2, Config(M=1, hidden=16, use_bonus=False, use_gate=False))
        rep = LaguReplay(60, 2, total + 5)
    starts, g = [], 0
    for e, n in enumerate(lengths):
        starts.append(g)
        o = rng.normal(size=60)
        for t in range(n):
            o2 = rng.normal(size=60)
            a, c = rng.uniform(-1, 1, 2), float(rng.random() < 0.05)
            if kind == "safe2":
                rep.add(o=o, b=1.0, a_pi=a, a_exec=a, r=0.0, c=c, o2=o2, b2=1.0, d=0.0, ep=e, t=t, I=0)
            else:
                rep.add(o, a, 0.0, c, o2, 0.0)
            o, g = o2, g + 1
    ck = {"replay": rep.state(), "step": total, "agent": agent.state_dict()}
    if kind == "safe2":
        ck["config"] = asdict(rc)
    torch.save(ck, path)
    return starts


@pytest.mark.parametrize("kind", ["lagu", "safe2"])
def test_s04_run_on_a_scratch_checkpoint(tmp_path, kind):
    """The last nep complete 1000-step episodes are used (a partial episode at the end is skipped); share is the
    quiet states' part of the |grad_a Q_c| mass, recomputed here by hand."""
    p = str(tmp_path / f"x_{kind}.ckpt.pt")
    starts = _s04_ckpt(p, kind, [1000, 1000, 1000, 1000, 300])
    ck = torch.load(p, weights_only=True)
    rows, lab = probe.s04_states(ck["replay"], 2)
    assert list(rows) == list(range(starts[2], starts[2] + 500)) + list(range(starts[3], starts[3] + 500))
    res = probe.s04_run(p, nep=2)
    assert res["episodes"] == 2 and res["n"] == 1000 and 0.0 <= res["share"] <= 1.0
    agent, lam, scale = probe.s04_agent(ck, 60, 2)
    assert (lam, scale) == ((0.7, (1 - 0.90) / (1 - 0.99)) if kind == "safe2" else (0.0, 1.0))
    ot = torch.as_tensor(probe._np(ck["replay"]["o"])[rows], dtype=torch.float32)
    at = agent.actor(ot).detach().requires_grad_(True)
    nc = torch.autograd.grad(agent.qc_pi(ot, at).sum(), at)[0].norm(dim=1).numpy()
    assert res["share"] == pytest.approx(nc[lab == 0].sum() / nc.sum(), rel=1e-6)
    assert res["share_states_quiet"] == pytest.approx((lab == 0).mean())


def test_s04_merges_by_stem_and_refuses_a_silent_rerun(tmp_path, capsys):
    a, b = str(tmp_path / f"SHL-g0.90-B4_{ENV}_s0.ckpt.pt"), str(tmp_path / f"td3lag-omni_{ENV}_s0.ckpt.pt")
    _s04_ckpt(a, "safe2", [1000, 1000, 1000])
    _s04_ckpt(b, "lagu", [1000, 1000, 1000], seed=1)
    out = tmp_path / "stage0"
    assert probe.main(["s04", "--runs", b, "--nep", "1", "--out", str(out)]) == 0
    probe.s04([a], str(out), nep=1)
    with open(out / "s04.json") as f:
        saved = json.load(f)
    assert set(saved["runs"]) == {f"SHL-g0.90-B4_{ENV}_s0", f"td3lag-omni_{ENV}_s0"} and set(saved["inputs"]) == {a, b}
    assert "share" in saved["runs"][f"SHL-g0.90-B4_{ENV}_s0"] and "safe2/probe.py" in saved["code"]
    with pytest.raises(SystemExit, match="--force"):
        probe.s04([a], str(out), nep=1)
    probe.s04([a], str(out), nep=1, force=True)
    with pytest.raises(SystemExit, match="missing input"):
        probe.s04([str(tmp_path / "nope.ckpt.pt")], str(out))
