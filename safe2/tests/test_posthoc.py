"""safe2.posthoc (SPEC section 7; section 10 tests 40 and 41, plus fit_meta, the NaN guard of recall, the
episode checks, the sealed recall and a PHS fit that safe2.bvs.PHSCore accepts). Tiny scratch agents and
synthetic replays only: no study data is read."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from safe2 import configs, posthoc, seal
from safe2.base import Agent
from safe2.tests.helpers import make_key
from safe2.train import Replay

ENV = "SafetyPointGoal1-v0"
OBS, ACT = 6, 2


def make_ckpt(path, lengths, seed=950, cost=None, obs0=None, size=None, config="B1"):
    """A safe2-style checkpoint of a tiny agent whose replay holds episodes of the given lengths (ep from 0).
    cost(e, t) and obs0(e, t) set c and o[:, 0] (default: no cost, random). Returns the rows in insertion
    order as (ep, t, o)."""
    rc = configs.make(config, hidden=16)
    torch.manual_seed(seed)
    agent = Agent(OBS, ACT, rc, 1000, seed=seed)
    total = sum(lengths)
    rep = Replay(OBS, ACT, size or total)
    rng = np.random.default_rng(seed)
    rows = []
    for e, L in enumerate(lengths):
        for t in range(L):
            o = rng.normal(size=OBS).astype(np.float32)
            if obs0 is not None:
                o[0] = obs0(e, t)
            rep.add(o=o, b=1.0, a_pi=rng.uniform(-1, 1, ACT), r=0.0, c=cost(e, t) if cost else 0.0,
                    o2=rng.normal(size=OBS), b2=1.0, d=0.0, ep=e, t=t)
            rows.append((e, t, o))
    torch.save({"config": asdict(rc), "env": ENV, "seed": seed, "steps": total, "agent": agent.state_dict(),
                "replay": rep.state(), "step": total}, path)
    return agent, rows


class ObsNet:
    """A stand-in V whose vmax is o[:, 0] (the synthetic replays encode the predictions there)."""

    def vmax(self, o, a):
        return o[:, 0]


# --- test 41: the fit reads only the requested episodes ------------------------------------------------
def test_fit_uses_only_the_requested_episodes(tmp_path, monkeypatch):
    """SPEC test 41: the rows given to fit_offline are exactly those with ep in [A-1, B-1], in insertion order,
    with the executed action, and the run's own actor as the proposer."""
    ck = tmp_path / "run.ckpt.pt"
    agent, rows = make_ckpt(ck, [5, 7, 6, 4, 8])
    seen = {}

    def spy(data, proposer, budget_input, seed, updates=100_000, hidden=256, **kw):
        seen.update(data=data, proposer=proposer, budget_input=budget_input, seed=seed, updates=updates, hidden=hidden)
        return {"V": {}, "pib": {}, "obs_dim": OBS, "act_dim": ACT, "hidden": hidden, "H": 32, "updates": updates,
                "batch": 256, "seed": seed, "losses": {"v": [], "pib": []}}
    import safe2.bvs
    monkeypatch.setattr(safe2.bvs, "fit_offline", spy)
    posthoc.fit(str(ck), "2-4", str(tmp_path / "f.pt"), updates=3, seed=11)
    want = np.stack([o for e, t, o in rows if 1 <= e <= 3])
    assert np.array_equal(seen["data"]["o"], want) and len(want) == 7 + 6 + 4
    rep = torch.load(ck, weights_only=True)["replay"]
    assert np.array_equal(seen["data"]["a"], rep["a_pi"].numpy()[5:5 + 17])
    assert (seen["seed"], seen["updates"], seen["hidden"], seen["budget_input"]) == (11, 3, 16, False)
    for p, q in zip(seen["proposer"].parameters(), agent.actor.parameters()):
        assert torch.equal(p, q) and not p.requires_grad


def test_fit_ignores_poisoned_rows_outside_the_episodes(tmp_path):
    """A real (short) fit: NaN observations outside the requested episodes never reach V or pi_b; the file is
    weights-only loadable, carries the metadata, and the same seed gives the same weights."""
    ck = tmp_path / "run.ckpt.pt"
    make_ckpt(ck, [10, 10, 10, 10], obs0=lambda e, t: np.nan if e in (0, 3) else 0.1)
    a, b = tmp_path / "a.posthoc.pt", tmp_path / "b.posthoc.pt"
    meta = posthoc.fit(str(ck), (2, 3), str(a), updates=6, seed=5)
    posthoc.fit(str(ck), (2, 3), str(b), updates=6, seed=5)
    fa, fb = torch.load(a, weights_only=True), torch.load(b, weights_only=True)
    assert meta == posthoc.fit_meta(str(a)) == {"ckpt_sha256": posthoc.sha256(str(ck)), "episodes": [2, 3],
                                               "updates": 6, "seed": 5}
    assert fa["rows"] == 20 and fa["config"] == "B1" and fa["env"] == ENV and fa["run_seed"] == 950
    assert fa["H"] == 32 and fa["hidden"] == 16 and len(fa["losses"]["v"]) == 6 and len(fa["losses"]["pib"]) == 3
    assert all(np.isfinite(fa["losses"]["v"])) and all(torch.isfinite(v).all() for v in fa["V"].values())
    assert all(torch.equal(fa["V"][k], fb["V"][k]) for k in fa["V"])
    net = posthoc.load_fit(str(a))
    assert net.vmax(torch.zeros(3, OBS), torch.zeros(3, ACT)).shape == (3,)


def test_the_seed_defaults_to_the_run_seed(tmp_path):
    ck = tmp_path / "run.ckpt.pt"
    make_ckpt(ck, [6, 6], seed=957)
    assert posthoc.fit(str(ck), "1-2", str(tmp_path / "f.pt"), updates=2)["seed"] == 957


def test_episode_ranges_must_be_stored_whole(tmp_path):
    ck = tmp_path / "run.ckpt.pt"
    make_ckpt(ck, [5, 5, 5])
    with pytest.raises(SystemExit, match="not in the replay"):
        posthoc.fit(str(ck), "2-4", str(tmp_path / "f.pt"), updates=1)
    with pytest.raises(SystemExit):
        posthoc.parse_episodes("3-2")
    wrapped = tmp_path / "wrapped.ckpt.pt"
    make_ckpt(wrapped, [5, 5, 5], size=12)          # the first 3 rows (episode 1's t = 0..2) are overwritten
    with pytest.raises(SystemExit, match="episode 1 is not stored whole"):
        posthoc.fit(str(wrapped), "1-2", str(tmp_path / "f.pt"), updates=1)
    assert len(posthoc.select_rows(torch.load(wrapped, weights_only=True)["replay"], (2, 3))["o"]) == 10


# --- test 40: recall and false alarms on a synthetic replay with known entries ---------------------------
# (episode length, cost steps, {t: vmax}); everything else has vmax 0.
EPISODES = [
    (40, (5, 20, 21), {10: 0.6, 0: 0.9}),   # entry 20 caught (10 in [8, 13]); 5 (t < 12) and 21 are not entries
    (50, (45,), {39: 0.7, 2: 0.6, 48: 1.0, 3: 0.5}),   # entry 45 missed (39 > 38); false alarms at 2 and 48; 0.5 is no alarm
    (30, (25,), {18: 0.6}),                 # entry 25 caught at the window's late edge t_e - 7
    (30, (25,), {12: 0.6}),                 # entry 25 missed: 12 = t_e - 13; not a false alarm (cost at 25)
]


def _synthetic(tmp_path, nan_at=None, seed=950):
    ck = tmp_path / f"run{seed}.ckpt.pt"
    cost = lambda e, t: float(t in EPISODES[e][1])
    obs0 = lambda e, t: np.nan if (e, t) == nan_at else EPISODES[e][2].get(t, 0.0)
    make_ckpt(ck, [L for L, _, _ in EPISODES], seed=seed, cost=cost, obs0=obs0)
    return ck


def test_recall_and_false_alarms_on_a_synthetic_replay(tmp_path, monkeypatch):
    """SPEC test 40."""
    ck = _synthetic(tmp_path)
    monkeypatch.setattr(posthoc, "load_fit", lambda path: ObsNet())
    out = posthoc.recall("unused", str(ck), "1-4")
    assert out == {"entries": 4, "caught": 2, "recall": 0.5, "false_alarm_steps": 2, "episodes": 4,
                   "fa_per_episode": 0.5, "nonfinite_pred": 0}
    sub = posthoc.recall("unused", str(ck), "2-2")
    assert (sub["entries"], sub["caught"], sub["false_alarm_steps"], sub["episodes"]) == (1, 0, 2, 1)


def test_a_non_finite_prediction_makes_recall_nan(tmp_path, monkeypatch):
    """Review stage0 F3 item 4: a NaN vmax is never an alarm, so it would silently lower recall; any non-finite
    prediction makes recall and the false-alarm rate NaN (the gate then fails), and pooling keeps it NaN."""
    from safe2 import probe
    ck = _synthetic(tmp_path, nan_at=(0, 10))
    monkeypatch.setattr(posthoc, "load_fit", lambda path: ObsNet())
    out = posthoc.recall("unused", str(ck), "1-4")
    assert out["nonfinite_pred"] == 1 and np.isnan(out["recall"]) and np.isnan(out["fa_per_episode"])
    assert out["caught"] == 1
    clean = {"entries": 10, "caught": 10, "false_alarm_steps": 0, "episodes": 10}
    pooled = probe.pool_recall([clean, out])
    assert np.isnan(pooled["recall"]) and pooled["nonfinite_pred"] == 1
    assert not probe.g0b_decision(pooled, 2000, 0.9)["pass_i"]


def test_sealed_recall_prints_no_number_and_blinded_seeds_need_it(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    priv = make_key(root)
    ck = _synthetic(root, seed=300)
    fit = root / "phs" / "run300.posthoc.pt"
    fit.parent.mkdir()
    fit.write_bytes(b"stand-in")
    monkeypatch.setattr(posthoc, "load_fit", lambda path: ObsNet())
    with pytest.raises(SystemExit, match="blinded"):
        posthoc.main(["recall", "--fit", str(fit), "--ckpt", str(ck), "--episodes", "1-4"])
    assert posthoc.main(["recall", "--fit", str(fit), "--ckpt", str(ck), "--episodes", "1-4", "--sealed"]) == 0
    assert capsys.readouterr().out == "run300: done\n"
    out = seal.read_file(str(root / "sealed" / "phs" / "run300.recall.json.enc"), priv)
    assert (out["entries"], out["caught"], out["recall"], out["episodes"]) == (4, 2, 0.5, 4)
    assert out["fit_sha256"] == posthoc.sha256(str(fit)) and out["episodes_range"] == [1, 4]
    assert seal.verify(str(root)) == []


def test_the_unsealed_cli_prints_the_counts_for_a_dev_seed(tmp_path, monkeypatch, capsys):
    ck = _synthetic(tmp_path)
    monkeypatch.setattr(posthoc, "load_fit", lambda path: ObsNet())
    assert posthoc.main(["recall", "--fit", "x", "--ckpt", str(ck), "--episodes", "1-4"]) == 0
    assert json.loads(capsys.readouterr().out)["recall"] == 0.5


def test_a_saved_fit_drives_the_phs_shield(tmp_path):
    """The fit file is what evaluate --phs loads (weights only) and bvs.PHSCore accepts."""
    from safe2 import evaluate
    ck = tmp_path / "run.ckpt.pt"
    agent, _ = make_ckpt(ck, [8, 8])
    out = tmp_path / "run.posthoc.pt"
    posthoc.fit(str(ck), "1-2", str(out), updates=2)
    snap = {"step": 16, "config": asdict(agent.rc), "obs_dim": OBS, "act_dim": ACT, "horizon": 1000,
            **agent.snapshot()}
    pol = evaluate.phs_policy(snap, torch.load(out, weights_only=True), 1.0, "SQ")
    pol.begin(90500)
    a = pol.act(np.zeros(OBS, np.float32))
    assert a.shape == (ACT,) and np.all(np.abs(a) <= 1.0)
