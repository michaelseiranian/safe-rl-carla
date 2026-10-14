"""safe2.train (SPEC section 10, tests 9, 10, 12, 25-28, 35, plus the trainer half of tests 1-2):
lagu.train trajectory equivalence, replay, Saute, the dual window, snapshots, resume, non-finite
handling and the results/method refusals. Fake-environment tests take well under a second each;
real-environment tests a few seconds."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from safe2 import configs, evaluate, seal, seeds, train
from safe2.base import Agent
from safe2.tests.helpers import ACT, OBS, FakeEnv, make_key, net_hash, same

ENV = "SafetyPointGoal1-v0"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fake_factory(envs: list, **kw):
    def make(env_id):
        e = FakeEnv(**kw)
        envs.append(e)
        return e
    return make


def quiet(lines: list):
    return lambda line: lines.append(line)


def small(name: str, **kw) -> configs.RunConfig:
    base = dict(hidden=16, batch=8, start_steps=15, eval_every=10, eval_episodes=2, snap_every=10, snap_from=0.0,
                ckpt_every=40)
    base.update(kw)
    return configs.make(name, **base)


# --- trajectory equivalence with lagu.train -------------------------------------------------------
@pytest.mark.parametrize("name,clamp", [("R1", 0), ("B1", 1)])
def test_r1_and_b1_reproduce_the_lagu_train_trajectory(tmp_path, name, clamp):
    """R1 (B1 with --cost-clamp 1) replays lagu.train --arm td3lag --recipe omnisafe with the same seed:
    identical replay rows, networks, lambda and in-training evaluation (real environment, 1500 steps,
    a dual that moves)."""
    pytest.importorskip("safety_gymnasium")
    lag = tmp_path / "lagu"
    cmd = [sys.executable, "-m", "lagu.train", "--env", ENV, "--arm", "td3lag", "--recipe", "omnisafe",
           "--seed", "900", "--steps", "1500", "--start-steps", "300", "--dual-start", "400", "--lambda-lr", "0.01",
           "--hidden", "32", "--eval-every", "1500", "--eval-episodes", "1", "--out", str(lag),
           "--cost-clamp", str(clamp), "--cost-limit", "-1", "--delta0", "2.5"]   # J > -1: lambda grows
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    rc = configs.make(name, hidden=32, start_steps=300, dual_start=400, lambda_lr=0.01, dual_target=-1.0,
                      eval_every=1500, eval_episodes=1, ckpt_every=1500)
    train.run(rc, ENV, 900, str(tmp_path / "s2"), steps=1500, console=quiet([]))
    a = torch.load(lag / f"td3lag_{ENV}_s900.ckpt.pt", weights_only=False)     # lagu's own file
    b = torch.load(tmp_path / "s2" / f"{name}_{ENV}_s900.ckpt.pt", weights_only=True)
    for x, y in (("o", "o"), ("a", "a_pi"), ("r", "r"), ("c", "c"), ("o2", "o2"), ("d", "d")):
        assert torch.equal(a["replay"][x], b["replay"][y]), x
    for net in ("actor", "members", "cost", "actor_targ", "members_targ", "cost_targ"):
        for k, v in a["agent"][net].items():
            assert torch.equal(v, b["agent"][net][k]), (net, k)
    assert a["agent"]["lambda"] == b["agent"]["lambda"] > 0.0
    ra, rb = rows(lag / f"td3lag_{ENV}_s900.csv")[-1], rows(tmp_path / "s2" / f"{name}_{ENV}_s900.csv")[-1]
    for col in ("train_return", "train_cost", "eval_return", "eval_cost", "qc0", "mc_cost0", "qc_traj",
                "mc_cost_traj", "qbar_traj", "mc_ret_traj", "cum_cost", "lambda", "jc"):
        assert float(ra[col]) == float(rb[col]), col


# --- replay ---------------------------------------------------------------------------------------------
def test_replay_columns_extras_sampling_and_state():
    r = train.Replay(3, 2, 5, extras={"a_exec": ((2,), np.float32), "I": ((), np.uint8)})
    for i in range(7):
        r.add(o=np.full(3, i), b=[0.5], a_pi=np.zeros(2), a_exec=np.ones(2), r=i, c=0, o2=np.full(3, i + 1),
              b2=[0.4], d=0, ep=i // 3, t=i % 3, I=i % 2, unused=1)
    assert r.n == 5 and sorted(r.data["r"].tolist()) == [2, 3, 4, 5, 6]
    assert [int(x) for x in r.data["r"][r.order()]] == [2, 3, 4, 5, 6]
    assert r.data["I"].dtype == np.uint8 and r.data["ep"].dtype == np.int32 and r.data["b"].shape == (5, 1)
    s = r.sample(np.random.default_rng(0), 4, keys=("o", "r"))
    assert set(s) == {"o", "r"} and s["o"].shape == (4, 3)
    idx = np.random.default_rng(0).integers(0, 5, 4)             # one integers() call, as lagu.train.Replay
    assert np.array_equal(s["r"], r.data["r"][idx])
    base = train.Replay(3, 2, 5)
    assert "a_exec" not in base.data and "I" not in base.data   # BVS-only columns are not allocated
    for size in (3, 5, 9):
        dst = train.Replay(3, 2, size, extras={"a_exec": ((2,), np.float32), "I": ((), np.uint8)})
        dst.load(r.state())
        assert sorted(dst.data["r"][:dst.n].tolist()) == list(range(7 - min(5, size), 7))


# --- dual window, Saute ----------------------------------------------------------------------------
def test_warmup_episodes_never_enter_the_dual_window(tmp_path):
    """Test 9: episodes ending at steps <= start_steps (here 25) are not in jc."""
    rc = small("B5", start_steps=25, ckpt_every=60)
    envs = []
    train.run(rc, "Fake", 900, str(tmp_path), steps=60, make_env=fake_factory(envs, cost_fn=lambda ep, t: float(ep)
                                                                                * (t == 0)), console=quiet([]))
    ck = torch.load(tmp_path / f"B5_Fake_s900.ckpt.pt", weights_only=True)
    assert ck["jc"] == [2.0, 3.0, 4.0, 5.0]             # episodes 0-5 end at 10..60; 0 and 1 are warm-up
    assert ck["ep_costs"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_saute_reward_replacement_budget_input_and_episode_reset(tmp_path):
    """Test 12: r_learn = -n once C_t >= 25 (C_t = cost before step t), b = clip((25 - C)/25, -1, 1),
    both reset at every episode start; the actor sees obs + 1 inputs."""
    rc = small("SauteB", start_steps=200, ckpt_every=80, eval_every=80)
    train.run(rc, "Fake", 901, str(tmp_path), steps=80, make_env=fake_factory([], length=40,
                                                                             cost_fn=lambda ep, t: 1.0),
              console=quiet([]))
    ck = torch.load(tmp_path / f"SauteB_Fake_s901.ckpt.pt", weights_only=True)
    rep = ck["replay"]
    t = rep["t"].numpy()
    C = t.astype(float)                                   # one unit of cost per step
    assert np.array_equal(t, np.r_[np.arange(40), np.arange(40)])
    want_r = np.where(C >= 25, -0.2, 0.5).astype(np.float32)
    assert np.array_equal(rep["r"].numpy(), want_r)
    assert np.allclose(rep["b"].numpy()[:, 0], np.clip((25 - C) / 25, -1, 1))
    assert np.allclose(rep["b2"].numpy()[:, 0], np.clip((25 - C - 1) / 25, -1, 1))
    assert rep["ep"].numpy().tolist() == [0] * 40 + [1] * 40
    assert ck["agent"]["actor"]["net.0.weight"].shape[1] == OBS + 1
    assert ck["cum"]["cost"] == 80.0 and ck["cum"]["viol_eps"] == 2


def test_budget_input_reaches_the_actor_during_training(tmp_path, monkeypatch):
    seen = []
    orig = Agent.select_action

    def spy(self, x, o, b, C, step):
        seen.append((x.copy(), b, C))
        return orig(self, x, o, b, C, step)
    monkeypatch.setattr(Agent, "select_action", spy)
    train.run(small("SauteA", start_steps=5, ckpt_every=40, eval_every=40), "Fake", 902, str(tmp_path), steps=20,
              make_env=fake_factory([], length=10, cost_fn=lambda ep, t: 5.0), console=quiet([]))
    for x, b, C in seen:
        assert x[-1] == b == float(np.clip((25 - C) / 25, -1, 1)) and len(x) == OBS + 1
    assert {C for _, _, C in seen} >= {0.0, 25.0, 45.0}


# --- snapshots ---------------------------------------------------------------------------------------
def test_snapshot_cadence_and_never_overwritten(tmp_path):
    """Snapshots at multiples of snap_every from snap_from * N; a rewritten step gets <step>.r<k>.pt."""
    rc = configs.make("R2", hidden=16, batch=8, start_steps=15, eval_every=50, eval_episodes=1, snap_every=20,
                      ckpt_every=50)
    train.run(rc, "Fake", 903, str(tmp_path), steps=100, make_env=fake_factory([]), console=quiet([]))
    snap_dir = tmp_path / "R2_Fake_s903.snap"
    assert sorted(os.listdir(snap_dir), key=lambda n: int(n.split(".")[0])) == ["80.pt", "100.pt"]
    (snap_dir / "60.pt").write_bytes(b"x")
    assert train.snapshot_path(str(snap_dir), 60, 2) == str(snap_dir / "60.r2.pt")
    (snap_dir / "60.r2.pt").write_bytes(b"x")
    assert train.snapshot_path(str(snap_dir), 60, 2) == str(snap_dir / "60.r3.pt")
    assert evaluate.snapshot_file(str(snap_dir), 60).endswith("60.r2.pt")
    assert evaluate.snapshot_file(str(snap_dir), 80).endswith("80.pt")
    with pytest.raises(FileNotFoundError):
        evaluate.snapshot_file(str(snap_dir), 70)
    snap = torch.load(snap_dir / "100.pt", weights_only=True)
    assert {"step", "config", "obs_dim", "act_dim", "horizon", "actor", "members", "cost", "lambda"} <= set(snap)


def test_heldout_targets_and_r3_mapping(tmp_path):
    """Test 26."""
    assert seeds.heldout_targets(1_000_000) == list(range(820_000, 1_000_001, 20_000))
    assert seeds.heldout_targets(3_000_000) == list(range(2_460_000, 3_000_001, 60_000))
    stem = tmp_path / "run"
    os.makedirs(str(stem) + ".snap")
    for s in evaluate.R3_STEPS:
        (tmp_path / "run.snap" / f"{s}.pt").write_bytes(b"x")
    plan = evaluate.r3_plan(str(stem))
    assert [p[1] for p in plan] == [800_000, 850_000, 900_000, 950_000, 1_000_000]
    for j, _, _, idx in plan:
        assert idx == list(range(40 * j, 40 * j + 40))           # 20 (2j) + k and 20 (2j + 1) + k
    assert sorted(i for p in plan for i in p[3]) == list(range(200))
    for i, step in enumerate(seeds.heldout_targets(1_000_000)):
        (tmp_path / "run.snap" / f"{step}.pt").write_bytes(b"x")
    hp = evaluate.heldout_plan(str(stem), 1_000_000)
    assert [p[3] for p in hp] == [list(range(20 * i, 20 * i + 20)) for i in range(10)]


@pytest.mark.parametrize("name", ["B5", "SauteA"])
def test_snapshot_round_trip_reproduces_in_training_evaluation(tmp_path, name):
    """Test 25 (fast form; tests/test_smoke.py has the 30k-step form): the held-out evaluator's policy
    (policy_from_snapshot) replays the in-training evaluation episode for episode, bit for bit."""
    pytest.importorskip("safety_gymnasium")
    rc = configs.make(name, hidden=32, start_steps=300, dual_start=500, eval_every=1000, eval_episodes=2,
                      snap_every=1000, snap_from=0.0, ckpt_every=2000)
    seen = {}
    train.run(rc, ENV, 904, str(tmp_path), steps=2000, console=quiet([]),
              on_eval=lambda step, eps: seen.__setitem__(step, eps))
    env = train._make_env(ENV)
    for step, eps in seen.items():
        policy = evaluate.policy_from_snapshot(str(tmp_path / f"{name}_{ENV}_s904.snap" / f"{step}.pt"))
        again = evaluate.rollout(env, policy, "train_eval", range(2))
        for e1, e2 in zip(eps, again):
            assert {k: v for k, v in e1.items() if k != "traj"} == e2
    assert len(seen) == 2


def test_policy_from_snapshot_equals_policy_from_agent(tmp_path):
    rc = small("SauteA")
    train.run(rc, "Fake", 905, str(tmp_path), steps=40, make_env=fake_factory([]), console=quiet([]))
    snap = str(tmp_path / "SauteA_Fake_s905.snap" / "40.pt")
    p1 = evaluate.policy_from_snapshot(snap)
    final = torch.load(tmp_path / "SauteA_Fake_s905.pt", weights_only=True)
    torch.manual_seed(0)
    ag = Agent(OBS, ACT, configs.RunConfig(**final["config"]), final["horizon"])
    ag.load_state_dict(final["agent"])
    p2 = evaluate.policy_from_agent(ag)
    assert type(p1) is type(p2) and p1.budget_input == p2.budget_input
    assert net_hash(p1.actor) == net_hash(p2.actor)
    before = torch.get_rng_state()
    evaluate.policy_from_snapshot(snap)
    assert torch.equal(before, torch.get_rng_state())           # rebuilding never consumes the torch RNG
    with pytest.raises(ValueError):
        evaluate.policy_from_snapshot(snap, "strict")           # a baseline has no shield modes


# --- resume -----------------------------------------------------------------------------------------
def _crash_at(step):
    def cb(s, eps):
        if s == step:
            raise RuntimeError("simulated crash")
    return cb


def test_resume_reproduces_an_uninterrupted_run_with_a_deterministic_env(tmp_path):
    """Test 10 (trainer level) and test 27: a run that crashes after its checkpoint at 40 and resumes ends
    with the same networks, PID state, replay and outcome rows as an uninterrupted run; its CSV is
    truncated to the checkpoint step, never duplicated; the resume layout is seed * 1000 + start."""
    rc = small("B5", kp=0.5, ki=0.01, dual_start=20, start_steps=15)
    kw = dict(length=10, cost_fn=lambda ep, t: float(t % 3 == 0) * 10, ep_in_obs=False)
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    train.run(rc, "Fake", 906, str(a_dir), steps=80, make_env=fake_factory([], **kw), console=quiet([]))
    with pytest.raises(RuntimeError):
        train.run(rc, "Fake", 906, str(b_dir), steps=80, make_env=fake_factory([], **kw), console=quiet([]),
                  on_eval=_crash_at(60))
    assert [int(r["step"]) for r in rows(b_dir / "B5_Fake_s906.csv")] == [10, 20, 30, 40, 50]
    envs = []
    train.run(rc, "Fake", 906, str(b_dir), steps=80, resume=True, make_env=fake_factory(envs, **kw),
              console=quiet([]))
    assert envs[0].seeds[0] == seeds.resume_reseed(906, 41) == 906 * 1000 + 41
    ra, rb = rows(a_dir / "B5_Fake_s906.csv"), rows(b_dir / "B5_Fake_s906.csv")
    assert [int(r["step"]) for r in rb] == [10, 20, 30, 40, 50, 60, 70, 80]
    skip = {"sps", "wall_s", "resumed", "n_resumes"}
    for x, y in zip(ra, rb):
        assert {k: v for k, v in x.items() if k not in skip} == {k: v for k, v in y.items() if k not in skip}
    assert [r["resumed"] for r in rb] == ["0"] * 4 + ["1"] * 4 and rb[-1]["n_resumes"] == "1"
    ca = torch.load(a_dir / "B5_Fake_s906.ckpt.pt", weights_only=True)
    cb = torch.load(b_dir / "B5_Fake_s906.ckpt.pt", weights_only=True)
    assert ca["agent"]["pid"]["I"] > 0.0 and ca["agent"]["lambda"] > 0.0
    for k in ca:
        if k not in ("wall_s", "n_resumes"):
            assert same(ca[k], cb[k]), k
    assert (ca["n_resumes"], cb["n_resumes"]) == (0, 1)
    meta = json.load(open(b_dir / "B5_Fake_s906.json"))
    assert meta["n_resumes"] == 1 and meta["resumes"][0]["from_step"] == 40


def test_resume_abandons_a_partial_episode_with_a_new_episode_index(tmp_path):
    rc = small("R2", ckpt_every=35, eval_every=35)
    with pytest.raises(RuntimeError):
        train.run(rc, "Fake", 907, str(tmp_path), steps=70, make_env=fake_factory([]), console=quiet([]),
                  on_eval=_crash_at(70))
    train.run(rc, "Fake", 907, str(tmp_path), steps=70, resume=True, make_env=fake_factory([]), console=quiet([]))
    ck = torch.load(tmp_path / "R2_Fake_s907.ckpt.pt", weights_only=True)
    ep, t = ck["replay"]["ep"].numpy(), ck["replay"]["t"].numpy()
    assert ep[34] == 3 and t[34] == 4 and ep[35] == 4 and t[35] == 0     # step 36 opens episode 4
    assert ck["n_resumes"] == 1


def test_resume_refuses_a_changed_config_and_a_foreign_csv(tmp_path):
    rc = small("B5")
    train.run(rc, "Fake", 908, str(tmp_path), steps=40, make_env=fake_factory([]), console=quiet([]))
    os.remove(tmp_path / "B5_Fake_s908.pt")                     # as if it had crashed after the checkpoint
    with pytest.raises(SystemExit, match="refusing to resume"):
        train.run(small("B5", kp=0.01), "Fake", 908, str(tmp_path), steps=80, resume=True,
                  make_env=fake_factory([]), console=quiet([]))
    p = tmp_path / "B5_Fake_s908.csv"
    p.write_text("step,other\n10,1\n")
    with pytest.raises(SystemExit, match="refusing to resume"):
        train.run(rc, "Fake", 908, str(tmp_path), steps=80, resume=True, make_env=fake_factory([]),
                  console=quiet([]))


def test_finished_runs_drop_ckpt_and_overwrite_guard(tmp_path):
    rc = small("R2")
    train.run(rc, "Fake", 909, str(tmp_path), steps=40, drop_ckpt=True, make_env=fake_factory([]),
              console=quiet([]))
    assert os.path.exists(tmp_path / "R2_Fake_s909.pt") and not os.path.exists(tmp_path / "R2_Fake_s909.ckpt.pt")
    final = torch.load(tmp_path / "R2_Fake_s909.pt", weights_only=True)
    assert final["step"] == 40 and final["config"]["name"] == "R2" and "actor" in final["agent"]
    lines = []
    train.run(rc, "Fake", 909, str(tmp_path), steps=40, resume=True, make_env=fake_factory([]),
              console=quiet(lines))
    assert lines == ["[R2_Fake_s909] already complete"]
    with pytest.raises(SystemExit, match="checkpoint dropped"):
        train.run(rc, "Fake", 909, str(tmp_path), steps=80, resume=True, make_env=fake_factory([]),
                  console=quiet([]))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train.run(rc, "Fake", 909, str(tmp_path), steps=40, make_env=fake_factory([]), console=quiet([]))


# --- non-finite values ---------------------------------------------------------------------------------
def test_nan_actor_weights_do_not_stop_the_run(tmp_path, monkeypatch):
    """Test 28: actor weights set to NaN at step 30 -> the run reaches its end, non-finite actions are
    replaced by zeros and counted, the finite flags of later rows are 0, nothing raises."""
    orig = Agent.post_update

    def poison(self, step, jc):
        if step == 30:
            with torch.no_grad():
                for p in self.actor.parameters():
                    p.fill_(float("nan"))
        return orig(self, step, jc)
    monkeypatch.setattr(Agent, "post_update", poison)
    envs = []
    train.run(small("B5"), "Fake", 910, str(tmp_path), steps=60, make_env=fake_factory(envs), console=quiet([]))
    r = rows(tmp_path / "B5_Fake_s910.csv")
    assert [int(x["step"]) for x in r] == [10, 20, 30, 40, 50, 60]
    assert all(x["finite_loss"] == "1" and x["finite_net"] == "1" for x in r[:2])
    assert all(x["finite_net"] == "0" for x in r[3:]) and int(r[-1]["nonfinite_actions"]) == 30
    assert all(np.all(np.isfinite(a)) for a in envs[0].actions)
    assert os.path.exists(tmp_path / "B5_Fake_s910.pt")


def test_finite_flags_and_whitelists():
    wl = train.whitelist(configs.make("B5"))
    assert "lambda" in wl and "lambda" not in train.whitelist(configs.make("R2"))
    assert set(train.whitelist(configs.from_bvs("BVS-SQ-k0.1", "bvs_u", 0.1))) >= {"T", "u_p50", "v_loss",
                                                                                   "m_tilde"}
    row = {"train_return": 1.0, "train_cost": 2.0, "eval_return": 3.0, "eval_cost": float("nan"), "q_loss": "",
           "lambda": float("inf"), "u_p50": float("nan")}
    f = train.finite_flags(row, wl)
    assert f == {"finite_out": 0, "finite_net": 1, "finite_loss": 1, "finite_dual": 0}   # u_p50 not whitelisted
    assert set(train.OUTCOME_FIELDS).isdisjoint(train.HEALTH_FIELDS)
    for cols in train.GROUPS.values():
        assert set(cols) <= set(train.OUTCOME_FIELDS)
    for wl in train.WHITELIST.values():
        assert set(wl) <= set(train.OUTCOME_FIELDS)
    ag = Agent(OBS, ACT, configs.make("B5", hidden=16), 1000)
    assert set(ag.log_row()) <= set(train.OUTCOME_FIELDS)


def test_every_whitelisted_column_is_written_once_learning_runs(tmp_path):
    for name in ("B5", "B1", "R2", "SauteA", "SHL-g0.90-B4"):
        rc = small(name, dual_start=20)
        train.run(rc, "Fake", 911, str(tmp_path / name), steps=40, make_env=fake_factory([]), console=quiet([]))
        last = rows(tmp_path / name / f"{name}_Fake_s911.csv")[-1]
        for col in train.whitelist(rc):
            assert last[col] not in ("", "nan"), (name, col)
        for col in ("vH_exec_mean", "v_loss", "m_tilde", "T", "int_cum"):
            assert last[col] == "", (name, col)               # BVS-only columns: empty, never NaN
        assert all(last[g] == "1" for g in train.GROUPS)


# --- blinding (in-process parts; the subprocess test is in test_blinding.py) -----------------------------
def test_blinded_runs_seal_rows_and_keep_only_health_columns(tmp_path):
    key = make_key(tmp_path)
    run_dir = tmp_path / "conf"
    lines = []
    rc = small("B5")
    with pytest.raises(RuntimeError):
        train.run(rc, "Fake", 300, str(run_dir), steps=80, make_env=fake_factory([]), console=quiet(lines),
                  on_eval=_crash_at(60))
    train.run(rc, "Fake", 300, str(run_dir), steps=80, resume=True, make_env=fake_factory([]),
              console=quiet(lines))
    with open(run_dir / "B5_Fake_s300.csv") as f:
        assert next(csv.reader(f)) == train.HEALTH_FIELDS
    recs = seal.read_records(str(tmp_path / "sealed" / "conf" / "B5_Fake_s300.rows"), key)
    assert [r["step"] for r in recs] == [10, 20, 30, 40, 50, 60, 70, 80]
    assert set(recs[0]) == {"step"} | set(train.OUTCOME_FIELDS)
    assert seal.verify(str(tmp_path)) == []
    import re
    pat = re.compile(r"^\[B5_Fake_s300\] step=\d+ sps=\d+ finite=[01]$")
    assert len(lines) == 9 and all(pat.match(x) for x in lines), lines     # rows 10-50, then 50-80 again


def test_dev_seeds_are_not_blinded_and_confirmatory_seeds_are():
    assert train.blinded(300) and train.blinded(315) and not train.blinded(299)
    assert not train.blinded(0) and not train.blinded(900) and not train.blinded(999)


def test_blinded_run_without_a_key_refuses_to_start(tmp_path):
    with pytest.raises(FileNotFoundError):
        train.run(small("B5"), "Fake", 301, str(tmp_path / "x"), steps=20, make_env=fake_factory([]),
                  console=quiet([]))


# --- refusals under results/method ------------------------------------------------------------------------
def test_trainer_refuses_dev_flags_and_unregistered_seeds_under_results_method(tmp_path):
    """Test 35."""
    out = str(tmp_path / "results" / "method" / "p1")
    base = ["--config", "B5", "--env", ENV, "--out", out]
    for extra, why in ((["--seed", "0", "--steps", "1000"], "--steps"),
                       (["--seed", "0", "--steps", "3000000"], "--steps"),
                       (["--seed", "0", "--hidden", "64"], "--hidden"),
                       (["--seed", "0", "--eval-every", "1000"], "dev-only"),
                       (["--seed", "0", "--snap-from", "0"], "dev-only"),
                       (["--seed", "900"], "not a study training seed"),
                       (["--seed", "7"], "not a study training seed")):
        with pytest.raises(SystemExit, match=why):
            train.main(base + extra)
    ok = train.parse(base + ["--seed", "300"])
    assert train.refusal(ok) is None
    dev = train.parse(["--config", "B5", "--env", ENV, "--out", str(tmp_path / "dev"), "--seed", "900",
                       "--steps", "1000", "--hidden", "16", "--eval-every", "100"])
    assert train.refusal(dev) is None
    with pytest.raises(SystemExit, match="from-bvs"):
        train.build_config(train.parse(["--config", "BVSU", "--env", ENV, "--out", "x", "--seed", "900"]))
    u = train.build_config(train.parse(["--config", "BVSU", "--env", ENV, "--out", "x", "--seed", "900",
                                        "--from-bvs", "BVS-SN-k0.02", "--rho-star", "0.04"]))
    assert (u.name, u.kind, u.rho_star, u.rule) == ("BVSU", "bvs_u", 0.04, "SN")



def test_3m_only_as_a_continuation_in_cont3m_and_dev_seeds_stay_short(tmp_path):
    """Review (base 6, ops H1): under results/method a 3M run is allowed only as a --resume in cont3m/
    (PointGoal1, seeds 300-304), and cont3m/ holds nothing else; review (base 8): a dev seed (not
    blinded) never trains more than DEV_MAX_STEPS steps; --resume and --restart exclude each other."""
    m = tmp_path / "results" / "method"
    cont = ["--config", "B5", "--env", ENV, "--out", str(m / "cont3m"), "--steps", "3000000", "--resume"]
    assert train.refusal(train.parse(cont + ["--seed", "300"])) is None
    assert train.refusal(train.parse(cont + ["--seed", "304"])) is None
    for argv, why in (
            (["--config", "B5", "--env", ENV, "--seed", "301", "--out", str(m / "conf"), "--steps", "3000000",
              "--resume"], "only with --resume in cont3m"),          # in place in conf/: the primary endpoint
            (["--config", "B5", "--env", ENV, "--seed", "301", "--out", str(m / "p1"), "--steps", "3000000",
              "--resume"], "only with --resume in cont3m"),
            (cont[:-1] + ["--seed", "300"], "only 3M continuations"),                     # no --resume
            (["--config", "B5", "--env", ENV, "--seed", "300", "--out", str(m / "cont3m")], "only 3M continuations"),
            (cont + ["--seed", "305"], "is not a 3M continuation"),
            (["--config", "B5", "--env", "SafetyCarGoal1-v0", "--seed", "300", "--out", str(m / "cont3m"),
              "--steps", "3000000", "--resume"], "is not a 3M continuation"),
            (["--config", "B5", "--env", ENV, "--seed", "300", "--out", str(m / "conf"), "--resume", "--restart"],
             "exclude each other")):
        why_not = train.refusal(train.parse(argv))
        assert why_not is not None and why in why_not, (argv, why_not)
    dev = ["--config", "B5", "--env", ENV, "--out", str(tmp_path / "dev"), "--hidden", "16"]
    assert train.refusal(train.parse(dev + ["--seed", "900", "--steps", str(train.DEV_MAX_STEPS)])) is None
    assert "smoke runs only" in train.refusal(train.parse(dev + ["--seed", "999", "--steps", "1000000"]))
    assert train.refusal(train.parse(dev + ["--seed", "0", "--steps", "1000000"])) is None    # selection seeds: public
    with pytest.raises(SystemExit, match="dev seeds are not blinded"):                     # in-process too
        train.run(small("R2"), "Fake", 950, str(tmp_path / "dev"), steps=train.DEV_MAX_STEPS + 1,
                  make_env=fake_factory([]), console=quiet([]))


def test_a_3m_continuation_without_its_checkpoint_never_starts_under_results_method(tmp_path):
    """Change request (ops 1): --steps 3000000 --resume with no checkpoint does not train a fresh 3M run."""
    out = tmp_path / "results" / "method" / "cont3m"
    make_key(tmp_path / "results" / "method")
    with pytest.raises(SystemExit, match="no checkpoint to continue from"):
        train.run(small("R2", replay=200), "Fake", 300, str(out), steps=3_000_000, resume=True,
                  make_env=fake_factory([]), console=quiet([]))
    assert not os.path.exists(out / "R2_Fake_s300.json")


def test_the_resume_count_survives_crashes_before_the_next_checkpoint(tmp_path):
    """Review (base 2): a run that crashes again before its next checkpoint still counts every resume,
    so the 3-resume cap of the launcher applies."""
    rc = small("R2")                                            # checkpoint at 40, crash at 60
    with pytest.raises(RuntimeError):
        train.run(rc, "Fake", 912, str(tmp_path), steps=80, make_env=fake_factory([]), console=quiet([]),
                  on_eval=_crash_at(60))
    for k in (1, 2, 3):
        with pytest.raises(RuntimeError):
            train.run(rc, "Fake", 912, str(tmp_path), steps=80, resume=True, make_env=fake_factory([]),
                      console=quiet([]), on_eval=_crash_at(60))
        meta = json.load(open(tmp_path / "R2_Fake_s912.json"))
        assert meta["n_resumes"] == k == len(meta["resumes"])
        assert rows(tmp_path / "R2_Fake_s912.csv")[-1]["n_resumes"] == str(k)
    import safe2.launch as L
    assert L.n_resumes(str(tmp_path), "R2_Fake_s912") == 3


def test_a_no_op_resume_of_a_finished_run_writes_nothing(tmp_path):
    """Review (base 3, ops M1): --resume on a finished run that kept its checkpoint (B*, continuation
    seeds) changes no file: not the json (no spurious resume), the CSV or the sealed records."""
    key = make_key(tmp_path)
    out = tmp_path / "conf"
    rc = small("B5")
    train.run(rc, "Fake", 302, str(out), steps=40, make_env=fake_factory([]), console=quiet([]))
    names = ["B5_Fake_s302.json", "B5_Fake_s302.csv", "B5_Fake_s302.pt", "B5_Fake_s302.ckpt.pt"]
    before = {n: (out / n).read_bytes() for n in names}
    rows_before = (tmp_path / "sealed" / "conf" / "B5_Fake_s302.rows").read_bytes()
    lines = []
    for _ in range(2):
        assert train.run(rc, "Fake", 302, str(out), steps=40, resume=True, make_env=fake_factory([]),
                         console=quiet(lines)) == "B5_Fake_s302"
    assert {n: (out / n).read_bytes() for n in names} == before and lines == []        # blinded: silent
    assert (tmp_path / "sealed" / "conf" / "B5_Fake_s302.rows").read_bytes() == rows_before
    assert seal.verify(str(tmp_path)) == [] and len(seal.read_records(str(tmp_path / "sealed" / "conf" /
                                                                          "B5_Fake_s302.rows"), key)) == 4
    os.remove(out / "B5_Fake_s302.pt")                         # crashed between its last checkpoint and the .pt
    train.run(rc, "Fake", 302, str(out), steps=40, resume=True, drop_ckpt=True, make_env=fake_factory([]),
              console=quiet([]))
    assert (out / "B5_Fake_s302.pt").exists() and not (out / "B5_Fake_s302.ckpt.pt").exists()
    assert json.load(open(out / "B5_Fake_s302.json"))["n_resumes"] == 0


def test_a_started_run_is_never_restarted_silently(tmp_path):
    """Review (base 5, ops L3): a plain start over a run that started before is refused; with a
    checkpoint it is resumed, without one --restart starts it again, keeps its history and counts as a
    resume; snapshot names stay monotone, so the evaluator reads the newest trajectory."""
    rc = small("R2", snap_every=10, snap_from=0.0, ckpt_every=40)
    with pytest.raises(RuntimeError):                          # crashes at 30, before its first checkpoint
        train.run(rc, "Fake", 913, str(tmp_path), steps=80, make_env=fake_factory([]), console=quiet([]),
                  on_eval=_crash_at(30))
    for resume in (False, True):
        with pytest.raises(SystemExit, match="only with --restart"):
            train.run(rc, "Fake", 913, str(tmp_path), steps=80, resume=resume, make_env=fake_factory([]),
                      console=quiet([]))
    with pytest.raises(RuntimeError):                          # restarted; crashes again at 60, after its checkpoint
        train.run(rc, "Fake", 913, str(tmp_path), steps=80, restart=True, make_env=fake_factory([]),
                  console=quiet([]), on_eval=_crash_at(60))
    meta = json.load(open(tmp_path / "R2_Fake_s913.json"))
    assert meta["n_resumes"] == 1 and meta["resumes"] == [{"from_step": 0, "restart": True, "steps": 80, "argv": None}]
    assert [r["resumed"] for r in rows(tmp_path / "R2_Fake_s913.csv")] == ["1"] * 5
    snaps = sorted(os.listdir(tmp_path / "R2_Fake_s913.snap"))
    assert {"10.pt", "10.r1.pt", "20.pt", "20.r1.pt", "30.pt"} <= set(snaps)    # the first attempt's files are kept
    for restart in (False, True):
        with pytest.raises(SystemExit, match="continue the run with --resume"):
            train.run(rc, "Fake", 913, str(tmp_path), steps=80, restart=restart, make_env=fake_factory([]),
                      console=quiet([]))
    train.run(rc, "Fake", 913, str(tmp_path), steps=80, resume=True, make_env=fake_factory([]), console=quiet([]))
    meta = json.load(open(tmp_path / "R2_Fake_s913.json"))
    assert meta["n_resumes"] == 2 and [r["from_step"] for r in meta["resumes"]] == [0, 40]
    assert evaluate.snapshot_file(str(tmp_path / "R2_Fake_s913.snap"), 50).endswith("50.r2.pt")
    snap_dir = str(tmp_path / "R2_Fake_s913.snap")
    assert train.snapshot_path(snap_dir, 10, 0) == os.path.join(snap_dir, "10.r2.pt")    # never below an old k


def test_resume_refuses_a_checkpoint_of_another_run(tmp_path):
    """Review (base 7): a checkpoint copied by hand under another run's stem is refused."""
    rc = small("R2")
    train.run(rc, "Fake", 931, str(tmp_path / "a"), steps=40, make_env=fake_factory([]), console=quiet([]))
    os.makedirs(tmp_path / "b")
    os.replace(tmp_path / "a" / "R2_Fake_s931.ckpt.pt", tmp_path / "b" / "R2_Fake_s930.ckpt.pt")
    with pytest.raises(SystemExit, match="belongs to Fake seed 931"):
        train.run(rc, "Fake", 930, str(tmp_path / "b"), steps=80, resume=True, make_env=fake_factory([]),
                  console=quiet([]))


def test_a_continuation_counts_only_its_own_resumes(tmp_path):
    """Review (ops L1, analysis L8): a continuation (here 40 -> 80 steps from the final checkpoint of the
    40-step run, as cont3m/ does 1M -> 3M) is not a resume; its resumes start at 0 and every crash
    afterwards counts, also one before the continuation's first own checkpoint."""
    rc = small("R2")
    src = tmp_path / "conf"
    with pytest.raises(RuntimeError):
        train.run(rc, "Fake", 932, str(src), steps=40, make_env=fake_factory([]), console=quiet([]),
                  on_eval=_crash_at(40))                       # the 40-step run crashed once (no checkpoint) ...
    with pytest.raises(SystemExit):
        train.run(rc, "Fake", 932, str(src), steps=40, make_env=fake_factory([]), console=quiet([]))
    train.run(rc, "Fake", 932, str(src), steps=40, restart=True, make_env=fake_factory([]), console=quiet([]))
    assert json.load(open(src / "R2_Fake_s932.json"))["n_resumes"] == 1           # ... and was restarted
    cont = tmp_path / "cont3m"
    os.makedirs(cont)
    import shutil
    shutil.copyfile(src / "R2_Fake_s932.ckpt.pt", cont / "R2_Fake_s932.ckpt.pt")
    with pytest.raises(RuntimeError):                          # crashes before its own first checkpoint (80)
        train.run(rc, "Fake", 932, str(cont), steps=80, resume=True, make_env=fake_factory([]), console=quiet([]),
                  on_eval=_crash_at(60))
    meta = json.load(open(cont / "R2_Fake_s932.json"))
    assert meta["n_resumes"] == 0 and meta["resumes"] == []
    assert meta["continuations"][0]["from_step"] == 40 and meta["continuations"][0]["previous_n_resumes"] == 1
    assert [r["resumed"] for r in rows(cont / "R2_Fake_s932.csv")] == ["0"]          # row 50
    train.run(rc, "Fake", 932, str(cont), steps=80, resume=True, make_env=fake_factory([]), console=quiet([]))
    meta = json.load(open(cont / "R2_Fake_s932.json"))
    assert meta["n_resumes"] == 1 and len(meta["continuations"]) == 1
    assert [int(r["step"]) for r in rows(cont / "R2_Fake_s932.csv")] == [50, 60, 70, 80]
    ck = torch.load(cont / "R2_Fake_s932.ckpt.pt", weights_only=True)
    assert ck["step"] == ck["steps"] == 80 and ck["n_resumes"] == 1
    mid = small("R2", ckpt_every=20)                           # only the final checkpoint of a run is continued
    train.run(mid, "Fake", 933, str(tmp_path / "m"), steps=40, make_env=fake_factory([]), console=quiet([]),
              on_eval=lambda s, e: None)
    ck = torch.load(tmp_path / "m" / "R2_Fake_s933.ckpt.pt", weights_only=True)
    ck["step"] = 20
    torch.save(ck, tmp_path / "m" / "R2_Fake_s933.ckpt.pt")
    os.remove(tmp_path / "m" / "R2_Fake_s933.pt")
    with pytest.raises(SystemExit, match="final checkpoint"):
        train.run(mid, "Fake", 933, str(tmp_path / "m"), steps=80, resume=True, make_env=fake_factory([]),
                  console=quiet([]))


def test_an_emptied_column_and_bvs_u_infinite_t_flags():
    """Review (base 11): once a whitelisted column has been written, an empty cell counts as
    non-finite; change request (BVS 2): BVS-U's T may be +-inf (never NaN) when rho* is 0 or 1."""
    wl = train.whitelist(configs.make("B5"))
    row = {"train_return": 1.0, "train_cost": 2.0, "eval_return": 3.0, "eval_cost": 4.0, "q_loss": "",
           "pi_loss": 0.1, "qbar": 1.0, "c_loss": 0.2, "qc": 0.3, "lambda": 0.0}
    assert all(train.finite_flags(row, wl).values())                          # not computed yet: finite
    assert train.finite_flags(row, wl, {"q_loss"})["finite_loss"] == 0        # written before, now empty
    for rho, ok in ((0.0, ("T",)), (1.0, ("T",)), (0.5, ())):
        assert train.inf_allowed(configs.from_bvs("BVS-SQ-k0.1", "bvs_u", rho)) == ok
    assert train.inf_allowed(configs.make("BVS-SQ-k0.1")) == ()
    wl_u = train.whitelist(configs.from_bvs("BVS-SQ-k0.1", "bvs_u", 0.0))
    for t, flag in ((float("inf"), 1), (float("-inf"), 1), (float("nan"), 0)):
        assert train.finite_flags({"T": t}, wl_u, inf_ok=("T",))["finite_dual"] == flag
    assert train.finite_flags({"T": float("inf")}, wl_u)["finite_dual"] == 0
    assert "trig007_num" in train.OUTCOME_FIELDS


def test_the_written_columns_are_checkpointed(tmp_path):
    rc = small("B5")
    train.run(rc, "Fake", 914, str(tmp_path), steps=40, make_env=fake_factory([]), console=quiet([]))
    ck = torch.load(tmp_path / "B5_Fake_s914.ckpt.pt", weights_only=True)
    assert set(ck["written"]) == set(train.whitelist(rc))


def test_the_final_file_of_a_blinded_seed_holds_network_weights_only(tmp_path):
    """Review (base 9): no outcome scalar (last training-cost mean, lambda, PID state) in the clear
    final file of a seed >= 300; the actor stays where tools read it ("agent"/"actor")."""
    make_key(tmp_path)
    rc = small("B5", kp=0.5, dual_start=20)
    train.run(rc, "Fake", 303, str(tmp_path / "conf"), steps=40, make_env=fake_factory([]), console=quiet([]))
    final = torch.load(tmp_path / "conf" / "B5_Fake_s303.pt", weights_only=True)
    assert set(final["agent"]) == {"actor", "members", "cost"} and final["step"] == 40
    snap = torch.load(tmp_path / "conf" / "B5_Fake_s303.snap" / "40.pt", weights_only=True)
    assert all(torch.equal(v, snap["actor"][k]) for k, v in final["agent"]["actor"].items())
    train.run(rc, "Fake", 915, str(tmp_path / "dev"), steps=40, make_env=fake_factory([]), console=quiet([]))
    assert "pid" in torch.load(tmp_path / "dev" / "B5_Fake_s915.pt", weights_only=True)["agent"]
