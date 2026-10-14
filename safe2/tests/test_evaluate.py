"""safe2.evaluate (SPEC section 10, tests 26 and 31, and the rollout / policy contract the BVS module
fills in): every held-out evaluation is sealed whatever the training seed (R3 included), the console
prints only done / ERROR, BVS per-episode statistics, and the ShieldPolicy wrapper's contract."""
from __future__ import annotations

import glob
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lagu.agent import Config, LagU
from lagu.train import save_snapshot
from safe2 import configs, evaluate, seal, seeds, train
from safe2.tests.helpers import ACT, OBS, FakeEnv, make_key


def test_shield_stats_on_a_hand_built_episode():
    costs = [0] * 12 + [1, 1] + [0] * 13 + [1] + [0] * 4      # entries at t = 12 and t = 27
    n = len(costs)
    I = [0] * n
    I[20] = 1                                                   # inside 15..26: the entry at 27 is caught
    empty = [False] * n
    empty[26] = True
    vmax = list(np.linspace(0, 1, n))
    st = evaluate.shield_stats(I, empty, vmax, costs, H=4)
    assert st["entries"] == 2 and st["misses"] == 1 and st["interventions"] == 1 and st["empty_steps"] == 1
    assert st["F"] == 1.0                                       # cost from the first empty step (26) on
    full = n - 4 + 1
    assert st["v_pred"] == pytest.approx(np.mean(vmax[:full]))
    assert st["v_real"] == pytest.approx(np.mean([sum(costs[t:t + 4]) for t in range(full)]))
    st0 = evaluate.shield_stats([0] * 3, [False] * 3, [0.0] * 3, [1, 0, 0], H=32)
    assert st0["entries"] == 0 and st0["F"] == 0.0 and np.isnan(st0["v_pred"])


class MockCore:
    """A stand-in for a safe2.bvs agent: the ShieldPolicy contract of safe2/HOOKS.md."""

    def __init__(self, rule="SQ"):
        self.rc = configs.make("BVS-SQ-k0.1" if rule == "SQ" else "BVS-SN-k0.1")
        self.m_dep = 3.5
        self.calls = []

    def propose(self, x, explore):
        assert explore is False
        return np.full(ACT, 0.5, np.float32)

    def act(self, o, b, C, a_pi, m, rng, mode, rule):
        u = float(rng.uniform())
        self.calls.append(dict(b=b, C=C, m=m, mode=mode, rule=rule, u=u))
        I = int(C >= 1.0)
        return (-a_pi if I else a_pi), I, {"empty": C >= 3.0, "vmax": C}


def test_shield_policy_contract_and_layout_seeded_sampler():
    core = MockCore()
    env = FakeEnv(length=6, cost_fn=lambda ep, t: 1.0)
    eps = evaluate.rollout(env, evaluate.ShieldPolicy(core, "deployed"), "dev", [3, 3])
    first, second = core.calls[:6], core.calls[6:]
    assert [c["u"] for c in first] == [c["u"] for c in second]       # default_rng(layout id) at each begin
    assert first[0]["u"] == np.random.default_rng(seeds.layout("dev", 3)).uniform()
    assert [c["C"] for c in first] == [0, 1, 2, 3, 4, 5] and all(c["m"] == 3.5 for c in first)
    assert [c["b"] for c in first] == [np.clip((25 - C) / 25, -1, 1) for C in range(6)]
    assert all(c["mode"] == "on" and c["rule"] == "SQ" for c in first)
    assert eps[0]["interventions"] == 5 and eps[0]["empty_steps"] == 3 and eps[0]["F"] == 3.0
    assert env.actions[0].tolist() == [0.5, 0.5] and env.actions[1].tolist() == [-0.5, -0.5]
    for mode, want in (("shield_off", ("off", "SQ")), ("strict", ("strict", "SQ")), ("s_near", ("on", "SN"))):
        core.calls.clear()
        evaluate.rollout(FakeEnv(length=2), evaluate.ShieldPolicy(core, mode), "dev", [0])
        assert (core.calls[0]["mode"], core.calls[0]["rule"]) == want
    with pytest.raises(ValueError):
        evaluate.ShieldPolicy(MockCore("SN"), "s_near")             # s_near only for an S-Q config


def _dev_run(tmp_path, seed, name="B5"):
    rc = configs.make(name, hidden=16, batch=8, start_steps=15, eval_every=1000, eval_episodes=1, snap_every=20,
                      ckpt_every=1000)
    out = tmp_path / "runs"
    train.run(rc, "Fake", seed, str(out), steps=1000, make_env=lambda e: FakeEnv(length=10),
              console=lambda line: None)
    return out


@pytest.mark.parametrize("seed", [900, 300])
def test_heldout_evaluation_is_always_sealed(tmp_path, monkeypatch, capsys, seed):
    """Test 31: a dev seed's and a confirmatory seed's held-out results go only to sealed/heldout, with
    the snapshot hashes in clear; the console says only '<stem>: done'; re-runs are skipped."""
    key = make_key(tmp_path)
    out = _dev_run(tmp_path, seed)
    monkeypatch.setattr(evaluate, "_make_env", lambda env_id: FakeEnv(length=10))
    capsys.readouterr()
    assert evaluate.main([str(out), "--workers", "1"]) == 0
    stem = f"B5_Fake_s{seed}"
    assert capsys.readouterr().out == f"{stem}: done\n"
    enc = tmp_path / "sealed" / "heldout" / f"{stem}.deployed.n1000.json.enc"     # N != 1M: ".n<N>"
    assert evaluate.output_paths(str(tmp_path), stem, "deployed", 1000)[1] == str(enc)
    res = seal.read_file(str(enc), key)
    assert [s["step"] for s in res["snapshots"]] == seeds.heldout_targets(1000)
    layouts = [e["layout"] for s in res["snapshots"] for e in s["episodes"]]
    assert layouts == list(range(70_000, 70_200))
    assert res["pooled"]["C"] == 0.0 and res["pooled"]["R"] == pytest.approx(5.0)   # 10 steps x 0.5
    assert seal.verify(str(tmp_path)) == []
    assert sorted(os.listdir(tmp_path / "sealed" / "heldout")) == [f"{stem}.deployed.n1000.json.enc",
                                                                  f"{stem}.deployed.n1000.sha"]   # no plaintext
    sha = json.load(open(tmp_path / "sealed" / "heldout" / f"{stem}.deployed.n1000.sha"))
    assert sha == [s["sha256"] for s in res["snapshots"]]
    assert evaluate.main([str(out), "--workers", "1"]) == 0
    assert capsys.readouterr().out == f"{stem}: skipped (sealed output exists)\n"
    snap = out / f"{stem}.snap" / "1000.pt"
    torch.save(torch.load(snap, weights_only=True) | {"lambda": 1.0}, snap)
    assert evaluate.main([str(out), "--workers", "1"]) == 1
    assert capsys.readouterr().out == f"{stem}: ERROR SnapshotsChanged\n"


def test_heldout_errors_print_the_type_only(tmp_path, monkeypatch, capsys):
    make_key(tmp_path)
    out = _dev_run(tmp_path, 901)
    monkeypatch.setattr(evaluate, "_make_env", lambda env_id: FakeEnv(length=10))
    capsys.readouterr()
    assert evaluate.main([str(out), "--workers", "1", "--mode", "strict"]) == 0
    assert capsys.readouterr().out == "B5_Fake_s901: skipped (no strict mode)\n"   # a baseline has no shield mode
    monkeypatch.setattr(evaluate, "rollout", lambda *a, **k: 1 / 0)
    assert evaluate.main([str(out), "--workers", "1"]) == 1
    assert capsys.readouterr().out == "B5_Fake_s901: ERROR ZeroDivisionError\n"
    monkeypatch.undo()
    os.remove(out / "B5_Fake_s901.snap" / "900.pt")                         # no nearest-step fallback
    assert evaluate.main([str(out), "--workers", "1"]) == 1
    assert capsys.readouterr().out == "B5_Fake_s901: ERROR FileNotFoundError\n"
    assert not os.path.exists(tmp_path / "sealed" / "heldout")


def test_r3_reevaluation_is_sealed(tmp_path, monkeypatch):
    """R3: Lag-U snapshots 800k-1M of seeds 100-109 on held-out indices 20 (2j) + k and 20 (2j + 1) + k."""
    key = make_key(tmp_path / "method")
    lag = tmp_path / "gate"
    lag.mkdir()
    torch.manual_seed(0)
    ag = LagU(OBS, ACT, Config(hidden=8, M=3))
    for arm in ("lagu-lit", "lagu_nogate-lit", "td3lag-lit"):     # results/gate_literal holds all three arms
        stem = lag / f"{arm}_Fake_s100"
        json.dump({"args": {"seed": 100, "env": "Fake"}, "obs_dim": OBS, "act_dim": ACT, "horizon": 10},
                  open(str(stem) + ".json", "w"))
        for s in evaluate.R3_STEPS:
            save_snapshot(str(stem) + ".snap", ag, s)
    json.dump({"args": {"seed": 120, "env": "Fake"}}, open(lag / "other_s120.json", "w"))
    monkeypatch.setattr(evaluate, "_make_env", lambda env_id: FakeEnv(length=10))
    assert evaluate.main(["--r3", str(lag), "--seeds", "100-109", "--root", str(tmp_path / "method"),
                          "--workers", "1"]) == 0
    # review (base 4): R3 is lagu-lit only; the other arms of the directory are never evaluated
    assert sorted(os.listdir(tmp_path / "method" / "sealed" / "heldout")) == ["lagu-lit_Fake_s100.r3.json.enc",
                                                                             "lagu-lit_Fake_s100.r3.sha"]
    res = seal.read_file(str(tmp_path / "method" / "sealed" / "heldout" / "lagu-lit_Fake_s100.r3.json.enc"), key)
    assert [s["step"] for s in res["snapshots"]] == list(evaluate.R3_STEPS)
    for j, s in enumerate(res["snapshots"]):
        assert [e["layout"] for e in s["episodes"]] == list(range(70_000 + 40 * j, 70_000 + 40 * j + 40))
    assert not glob.glob(str(lag / "*.heldout.json"))


def test_rollout_records_traj_and_counts_nonfinite_actions():
    class NanPolicy(evaluate.Policy):
        def act(self, o):
            self.info = {"x": o}
            return np.array([np.nan, 0.0]) if len(self.seen) % 2 == 0 else np.array([0.1, 0.1])

        def observe(self, c):
            super().observe(c)
            self.seen.append(c)

    pol = NanPolicy()
    pol.seen = []
    env = FakeEnv(length=4, cost_fn=lambda ep, t: 2.0)
    ep = evaluate.rollout(env, pol, "train_eval", [5], keep_traj=True)[0]
    assert ep["layout"] == 10_005 and ep["nonfinite_actions"] == 2 and ep["cost"] == 8.0 and pol.C == 8.0
    assert len(ep["traj"]["x"]) == 4 and ep["traj"]["c"] == [2.0] * 4
    assert all(np.all(np.isfinite(a)) for a in env.actions)


# --- review and change-request fixes ------------------------------------------------------------------
def test_shield_stats_allowed_set_size_and_bvs_disc_calibration():
    """Change request (analysis 3, BVS F3): the allowed-set size of the screening steps; review (BVS
    F7): BVS-disc's v_real is its critic's quantity, the gamma_b-discounted cost-to-go clamped at qb_max."""
    costs = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    st = evaluate.shield_stats([0] * 6, [False] * 6, [0.5] * 6, costs, H=2, n_allowed=[3, 0, 9])
    assert st["screen_steps"] == 3 and st["n_allowed_mean"] == 4.0
    none = evaluate.shield_stats([0] * 6, [False] * 6, [0.5] * 6, costs, H=2)
    assert none["screen_steps"] == 0 and np.isnan(none["n_allowed_mean"])
    g = 0.5
    disc = evaluate.shield_stats([0] * 6, [False] * 6, [0.5] * 6, costs, H=2, disc=(g, 1.2))
    togo = [sum(g ** k * c for k, c in enumerate(costs[t:])) for t in range(6)]
    assert disc["v_real"] == pytest.approx(np.mean(np.minimum(togo[:5], 1.2)))
    assert none["v_real"] == pytest.approx(np.mean([sum(costs[t:t + 2]) for t in range(5)]))


def test_the_worst_case_heldout_output_fits_the_fixed_sealed_size():
    """Review (base 1): sealed files are padded to seal.PAD["file"], so the largest possible output (200
    shielded episodes, the gaps of every step of 20 episodes, long floats) must fit."""
    x = -1.2345678901234567e-05
    ep = {"layout": 70_199, "return": x, "cost": x, "length": 1000, "nonfinite_actions": 1000, "interventions": 1000,
          "empty_steps": 1000, "F": x, "entries": 1000, "misses": 1000, "v_pred": x, "v_real": x,
          "screen_steps": 1000, "n_allowed_mean": x}
    out = {"stem": "BVS-SQ-k0.02_SafetyPointGoal1-v0_s315", "mode": "s_near", "end_step": 3_000_000,
           "snapshots": [{"i": i, "step": 3_000_000, "sha256": "f" * 64, "episodes": [dict(ep)] * 20}
                         for i in range(10)],
           "pooled": {"R": x, "C": x}, "gaps": [x] * 20 * 1000,
           "phs": {"fit_sha256": "f" * 64, "m": x, "rule": "SQ"}}
    assert len(json.dumps(out).encode()) < seal.PAD["file"]


def _copy_snapshots(src_snap: str, stem_path, steps: int, meta: dict) -> None:
    """A run json and snapshot files at every held-out target of `steps`, each a copy of src_snap."""
    os.makedirs(str(stem_path) + ".snap", exist_ok=True)
    for t in seeds.heldout_targets(steps):
        torch.save(torch.load(src_snap, weights_only=True), f"{stem_path}.snap/{t}.pt")
    json.dump(meta, open(str(stem_path) + ".json", "w"))


def _one_snapshot(tmp_path, name="B5", **kw):
    rc = configs.make(name, hidden=16, batch=8, start_steps=15, eval_every=40, eval_episodes=1, snap_every=40,
                      snap_from=0.0, ckpt_every=40, **kw)
    out = tmp_path / "src"
    train.run(rc, "Fake", 902, str(out), steps=40, make_env=lambda e: FakeEnv(length=10), console=lambda line: None)
    meta = json.load(open(out / f"{name}_Fake_s902.json"))
    return str(out / f"{name}_Fake_s902.snap" / "40.pt"), meta


def test_3m_outputs_never_take_the_1m_name_and_end_steps_are_checked(tmp_path, monkeypatch, capsys):
    """Change request (analysis 1, ops 2) and review (ops H1): a 3M continuation's output is
    <stem>.<mode>.n3000000.json.enc beside the 1M <stem>.<mode>.json.enc; under results/method the end
    step is 1M, and 3M only in cont3m/."""
    m = tmp_path / "results" / "method"
    key = make_key(m)
    snap, meta = _one_snapshot(tmp_path)
    stem = "B5_Fake_s300"
    for stage, steps in (("conf", 1_000_000), ("cont3m", 3_000_000)):
        _copy_snapshots(snap, m / stage / stem, steps, meta | {"args": {**meta["args"], "seed": 300, "steps": steps}})
    monkeypatch.setattr(evaluate, "_make_env", lambda env_id: FakeEnv(length=2))
    capsys.readouterr()
    assert evaluate.main([str(m / "conf"), "--end-step", "3000000", "--workers", "1"]) == 1
    assert evaluate.main([str(m / "cont3m"), "--end-step", "1000000", "--workers", "1"]) == 1
    assert capsys.readouterr().out == f"{stem}: ERROR EndStepRefused\n" * 2
    assert evaluate.main([str(m / "conf"), "--workers", "1"]) == 0
    assert evaluate.main([str(m / "cont3m"), "--workers", "1"]) == 0
    assert capsys.readouterr().out == f"{stem}: done\n" * 2
    h = m / "sealed" / "heldout"
    assert sorted(os.listdir(h)) == [f"{stem}.deployed.json.enc", f"{stem}.deployed.n3000000.json.enc",
                                     f"{stem}.deployed.n3000000.sha", f"{stem}.deployed.sha"]
    assert seal.read_file(str(h / f"{stem}.deployed.json.enc"), key)["end_step"] == 1_000_000
    res = seal.read_file(str(h / f"{stem}.deployed.n3000000.json.enc"), key)
    assert res["end_step"] == 3_000_000 and [s["step"] for s in res["snapshots"]] == seeds.heldout_targets(3_000_000)
    from safe2 import analyze
    assert analyze.heldout_rel(stem, "deployed", 3_000_000) == os.path.relpath(
        evaluate.output_paths(str(m), stem, "deployed", 3_000_000)[1], m)


def test_the_sha_side_file_is_written_first_so_a_crash_never_blocks_a_rerun(tmp_path, monkeypatch, capsys):
    """Review (base 10): a crash while sealing leaves no sealed file without its .sha; a sealed file whose
    .sha is gone is reported as such."""
    make_key(tmp_path)
    out = _dev_run(tmp_path, 903)
    monkeypatch.setattr(evaluate, "_make_env", lambda env_id: FakeEnv(length=10))
    real = seal.write_file
    monkeypatch.setattr(seal, "write_file", lambda *a, **k: 1 / 0)
    capsys.readouterr()
    assert evaluate.main([str(out), "--workers", "1"]) == 1
    h = tmp_path / "sealed" / "heldout"
    assert os.listdir(h) == ["B5_Fake_s903.deployed.n1000.sha"]
    monkeypatch.setattr(seal, "write_file", real)
    assert evaluate.main([str(out), "--workers", "1"]) == 0          # re-evaluated, not refused
    assert capsys.readouterr().out == "B5_Fake_s903: ERROR ZeroDivisionError\nB5_Fake_s903: done\n"
    os.remove(h / "B5_Fake_s903.deployed.n1000.sha")
    assert evaluate.main([str(out), "--workers", "1"]) == 1
    assert capsys.readouterr().out == "B5_Fake_s903: ERROR ShaMissing\n"
    assert evaluate.main([str(out), "--workers", "1", "--force"]) == 0
    assert sorted(os.listdir(h)) == ["B5_Fake_s903.deployed.n1000.json.enc", "B5_Fake_s903.deployed.n1000.sha"]
    assert seal.verify(str(tmp_path)) == []


def test_secondary_modes_skip_the_runs_they_do_not_apply_to(tmp_path, monkeypatch, capsys):
    """Change request (ops 4): shield modes skip baselines, s_near skips S-N configs; they print
    'skipped (no <mode> mode)', never an error."""
    make_key(tmp_path)
    d = tmp_path / "runs"
    for name in ("B5", "BVS-SN-k0.1", "BVS-SQ-k0.1"):
        snap, meta = _one_snapshot(tmp_path / name, name)
        _copy_snapshots(snap, d / f"{name}_Fake_s902", 1000, meta | {"args": {**meta["args"], "steps": 1000}})
    monkeypatch.setattr(evaluate, "_make_env", lambda env_id: FakeEnv(length=2))
    capsys.readouterr()
    assert evaluate.main([str(d), "--mode", "s_near", "--workers", "1"]) == 0
    assert capsys.readouterr().out == ("B5_Fake_s902: skipped (no s_near mode)\n"
                                       "BVS-SN-k0.1_Fake_s902: skipped (no s_near mode)\n"
                                       "BVS-SQ-k0.1_Fake_s902: done\n")
    assert evaluate.main([str(d), "--mode", "strict", "--workers", "1"]) == 0
    assert capsys.readouterr().out == ("B5_Fake_s902: skipped (no strict mode)\nBVS-SN-k0.1_Fake_s902: done\n"
                                       "BVS-SQ-k0.1_Fake_s902: done\n")


def test_phs_evaluates_every_bstar_snapshot_behind_the_bvs_star_shield(tmp_path, monkeypatch, capsys):
    """Change requests (BVS 1, analysis 2): --phs plays each of B*'s 10 snapshots behind A.4's shield
    with the run's offline fit, m_PHS(task) and BVS*'s rule, sealed as <stem>.phs.json.enc; the .sha
    also covers the fit, and the other arms of the directory are left alone."""
    from safe2 import bvs
    key = make_key(tmp_path)
    snap, meta = _one_snapshot(tmp_path)
    conf = tmp_path / "conf"
    stem = "B5_Fake_s300"
    _copy_snapshots(snap, conf / stem, 1_000_000, meta | {"args": {**meta["args"], "seed": 300, "steps": 1_000_000}})
    _copy_snapshots(snap, conf / "R1_Fake_s300", 1_000_000,
                    meta | {"args": {**meta["args"], "config": "R1", "seed": 300, "steps": 1_000_000}})
    json.dump({"b_star": "B5", "bvs_star": "BVS-SN-k0.02", "m_phs": {"Fake": 2.5}}, open(tmp_path / "v1.json", "w"))
    rng = np.random.default_rng(0)
    data = {"o": rng.normal(size=(50, OBS)).astype(np.float32), "a": rng.uniform(-1, 1, (50, ACT)).astype(np.float32),
            "c": (rng.uniform(size=50) < 0.3).astype(np.float32), "o2": rng.normal(size=(50, OBS)).astype(np.float32),
            "b2": np.ones((50, 1), np.float32), "d": np.zeros(50, np.float32)}
    actor = evaluate.policy_from_snapshot(snap).actor
    fit = bvs.fit_offline(data, actor, False, seed=300, updates=4, batch=8, hidden=16, H=4)
    os.makedirs(tmp_path / "phs")
    torch.save(fit, tmp_path / "phs" / f"{stem}.posthoc.pt")
    monkeypatch.setattr(evaluate, "_make_env", lambda env_id: FakeEnv(length=3, cost_fn=lambda ep, t: 1.0))
    capsys.readouterr()
    assert evaluate.main(["--phs", str(conf), "--workers", "1"]) == 0
    assert capsys.readouterr().out == f"{stem}: done\n"
    h = tmp_path / "sealed" / "heldout"
    res = seal.read_file(str(h / f"{stem}.phs.json.enc"), key)
    assert res["mode"] == "phs" and res["phs"]["m"] == 2.5 and res["phs"]["rule"] == "SN"
    assert [s["step"] for s in res["snapshots"]] == seeds.heldout_targets(1_000_000)
    assert [e["layout"] for s in res["snapshots"] for e in s["episodes"]] == list(range(70_000, 70_200))
    assert {"interventions", "n_allowed_mean", "v_pred"} <= set(res["snapshots"][0]["episodes"][0])
    sha = json.load(open(h / f"{stem}.phs.sha"))
    assert sha[-1] == evaluate.digest(str(tmp_path / "phs" / f"{stem}.posthoc.pt")) and len(sha) == 11
    assert evaluate.main(["--phs", str(conf), "--workers", "1"]) == 0
    assert capsys.readouterr().out == f"{stem}: skipped (sealed output exists)\n"
    policy = evaluate.phs_policy(torch.load(snap, weights_only=True), fit, 2.5, "SN")
    assert policy.shielded and policy.core.m_dep == 2.5 and policy.rule == "SN"
