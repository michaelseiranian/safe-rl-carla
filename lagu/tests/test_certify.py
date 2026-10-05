"""lagu.certify: per-tensor hashes, the equal/different/liveness expectations, and an end-to-end
certificate on three short real trainings (fixed dual: lagu == lagu_nogate; gate_grad=1 differs)."""
import csv
import json
import os
import subprocess
import sys

import pytest
import torch

from lagu import certify as C
from lagu.agent import Config, LagU

OBS, ACT, ENV = 6, 2, "SafetyPointGoal1-v0"
COLS = ["step", "eval_return", "eval_cost", "eval_violation", "lambda", "qc_traj", "mc_cost_traj",
        "n_risk", "engagement", "sps", "resumed"]


def agent(use_gate=True, **kw):
    torch.manual_seed(0)
    ag = LagU(OBS, ACT, Config(hidden=8, M=3, use_gate=use_gate, **kw), seed=0)
    g = torch.Generator().manual_seed(1)
    for i in range(4):                               # populate every optimizer state
        ag.env_step = 10
        ag.update((torch.randn(16, OBS, generator=g), torch.rand(16, ACT, generator=g) * 2 - 1,
                   torch.randn(16, generator=g), torch.ones(16), torch.randn(16, OBS, generator=g),
                   torch.zeros(16)))
    if ag.cfg.dual == "episodic":
        ag.update_dual_episodic(40.0)
    ag.env_step = 2000                               # the step of the last CSV row in rows()
    return ag


def nudged(sd, key="0.q1.0.weight"):
    """sd with one member weight moved by 1e-7: same cfg, different tensors."""
    w = sd["members"][key].clone()
    w[0, 0] += 1e-7
    return {**sd, "members": {**sd["members"], key: w}}


def write_run(d, label, seed, sd, rows, cols=COLS, args=None):
    stem = f"{label}_{ENV}_s{seed}"
    torch.save(sd, os.path.join(d, stem + ".pt"))
    with open(os.path.join(d, stem + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if args is not None:
        with open(os.path.join(d, stem + ".json"), "w") as f:
            json.dump({"args": args}, f)


def rows(engagement=0.5, eval_cost=3.0, sps=100.0, resumed=0, steps=(1000, 2000)):
    return [{"step": s, "eval_return": 1.5, "eval_cost": eval_cost, "eval_violation": 0.0, "lambda": 0.3,
             "qc_traj": float("nan"), "mc_cost_traj": 0.25, "n_risk": 5, "engagement": engagement, "sps": sps,
             "resumed": resumed} for s in steps]


def train_args(**kw):
    return {"env": ENV, "arm": "lagu", "tag": "x", "seed": 0, "steps": 2000, "start_steps": 300,
            "gate_grad": 0, "out": "/somewhere", "resume": False, "drop_ckpt": 0, **kw}


def test_hashes_cover_networks_optimizers_and_dual_but_not_counters_or_cfg():
    sd = agent(dual="episodic", lambda_init=0.1).state_dict()
    h = C.tensor_hashes(sd)
    prefixes = {k.split(".")[0] for k in h}
    assert prefixes == {"actor", "actor_targ", "members", "members_targ", "cost", "cost_targ",
                        "pi_opt", "q_opt", "c_opt", "lam_opt", "lam_param", "lambda"}
    assert any(k.startswith("q_opt.") and k.endswith("exp_avg_sq") for k in h)
    base = C.digest(h)
    for k, v in (("counters", (9, 9, 9, 9)), ("n_updates", 12345), ("cfg", {}), ("lambda_shadow", 7.0)):
        assert C.digest(C.tensor_hashes({**sd, k: v})) == base, k
    h2 = C.tensor_hashes(nudged(sd))
    assert [k for k in h if h[k] != h2[k]] == ["members.0.q1.0.weight"]
    assert C.tensor_hashes({**sd, "lam_param": 0.1000001})["lam_param"] != h["lam_param"]


def test_twin_cost_heads_are_hashed():
    h = C.tensor_hashes(agent(cost_twin=True).state_dict())
    assert any(k.startswith("cost2.") for k in h) and any(k.startswith("cost2_targ.") for k in h)


def synthetic(d):
    """lagu-x / lagu_nogate-x share every tensor; lagu-y has lagu-x's cfg and one weight moved."""
    g, n = agent(True), agent(False)
    write_run(d, "lagu-x", 0, g.state_dict(), rows())
    write_run(d, "lagu_nogate-x", 0, n.state_dict(), rows(engagement=0.0, sps=55.0))
    write_run(d, "lagu-y", 0, nudged(g.state_dict()), rows())


def test_equal_and_different_pairs_pass(tmp_path):
    synthetic(tmp_path)
    # the .json args may differ in arm/tag/gate_grad and bookkeeping (out, resume, drop_ckpt, steps)
    write_run(tmp_path, "lagu-x", 0, agent(True).state_dict(), rows(), args=train_args())
    write_run(tmp_path, "lagu_nogate-x", 0, agent(False).state_dict(), rows(engagement=0.0, sps=55.0),
              args=train_args(arm="lagu_nogate", out="/elsewhere", resume=True, drop_ckpt=1, steps=9))
    cert = C.certify(str(tmp_path), [{"a": "lagu-x", "b": "lagu_nogate-x", "seed": 0, "expect": "equal"},
                                     {"a": "lagu-y", "b": "lagu_nogate-x", "seed": 0, "expect": "different"}])
    assert cert["passed"], cert["pairs"]
    eq, diff = cert["pairs"]
    assert eq["n_differ"] == 0 and eq["csv_differ"] == [] and eq["cfg_diff"] == ["use_gate"]
    assert eq["args_diff"] == []
    assert diff["n_differ"] > 0


@pytest.mark.parametrize("case,expect,reason", [
    ("eval", "equal", "CSV columns differ: eval_cost"),
    ("tensor", "equal", "tensors differ"),
    ("same", "different", "tensors are identical"),
    ("dead_gate", "equal", "the gate never engaged"),
    ("missing", "equal", "missing"),
])
def test_failed_expectations_exit_nonzero(tmp_path, case, expect, reason):
    synthetic(tmp_path)
    b = "lagu_nogate-x"
    if case == "eval":
        write_run(tmp_path, b, 0, agent(False).state_dict(), rows(engagement=0.0, eval_cost=3.5))
    elif case == "tensor":
        write_run(tmp_path, b, 0, nudged(agent(False).state_dict()), rows(engagement=0.0))
    elif case == "dead_gate":
        write_run(tmp_path, "lagu-x", 0, agent(True).state_dict(), rows(engagement=0.0))
    elif case == "missing":
        os.remove(tmp_path / f"lagu_nogate-x_{ENV}_s0.pt")
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps([{"a": "lagu-x", "b": b, "seed": 0, "expect": expect}]))
    assert C.main([str(tmp_path), "--pairs", str(pairs)]) == 1
    cert = json.loads((tmp_path / "certificate.json").read_text())
    assert not cert["passed"] and any(reason in r for r in cert["pairs"][0]["reasons"]), cert["pairs"][0]
    assert "FAIL" in (tmp_path / "certificate.md").read_text()


@pytest.mark.parametrize("case,a,b,expect,reason", [
    ("malformed_pt", "lagu-x", "lagu_nogate-x", "equal", "the .pt lacks actor, actor_targ"),
    ("same_stem", "lagu-x", "lagu-x", "equal", "they are identical"),
    ("cfg", "lagu-y", "lagu_nogate-x", "different", "they also differ in lr"),
    ("steps", "lagu-y", "lagu_nogate-x", "different", "the runs end at different steps (3000 vs 2000)"),
    ("stale", "lagu-x", "lagu_nogate-x", "equal", "stale .pt or unfinished run"),
    ("args", "lagu-y", "lagu_nogate-x", "different", "the trainer args differ in start_steps"),
    ("no_eval_cols", "lagu-x", "lagu_nogate-x", "equal", "eval_return (missing)"),
])
def test_vacuous_comparisons_fail(tmp_path, case, a, b, expect, reason):
    """Regression: each of these used to certify (0 tensors hashed, a same-stem pair, or a positive
    control that differed only through its config, its step count or its trainer args)."""
    synthetic(tmp_path)
    g, n = agent(True), agent(False)
    if case == "malformed_pt":                      # hashes nothing, so every hash "matched"
        write_run(tmp_path, "lagu-x", 0, {"cfg": g.state_dict()["cfg"], "env_step": 2000}, rows())
        write_run(tmp_path, "lagu_nogate-x", 0, {"cfg": n.state_dict()["cfg"], "env_step": 2000},
                  rows(engagement=0.0))
    elif case == "cfg":
        sd = nudged(g.state_dict())
        write_run(tmp_path, "lagu-y", 0, {**sd, "cfg": {**sd["cfg"], "lr": 1e-3}}, rows())
    elif case == "steps":                           # e.g. the control rerun at 300k, its partner not
        write_run(tmp_path, "lagu-y", 0, {**nudged(g.state_dict()), "env_step": 3000},
                  rows(steps=(1000, 2000, 3000)))
    elif case == "stale":                           # a rerun's CSV beside the previous run's .pt
        write_run(tmp_path, "lagu_nogate-x", 0, {**n.state_dict(), "env_step": 1000}, rows(engagement=0.0))
    elif case == "args":
        write_run(tmp_path, "lagu-y", 0, nudged(g.state_dict()), rows(), args=train_args(start_steps=10_000))
        write_run(tmp_path, "lagu_nogate-x", 0, n.state_dict(), rows(engagement=0.0),
                  args=train_args(arm="lagu_nogate"))
    elif case == "no_eval_cols":                    # neither CSV has eval_*: nothing was compared
        cols = [c for c in COLS if not c.startswith("eval_")]
        write_run(tmp_path, "lagu-x", 0, g.state_dict(), rows(), cols=cols)
        write_run(tmp_path, "lagu_nogate-x", 0, n.state_dict(), rows(engagement=0.0), cols=cols)
    cert = C.certify(str(tmp_path), [{"a": a, "b": b, "seed": 0, "expect": expect}])
    assert not cert["passed"] and any(reason in r for r in cert["pairs"][0]["reasons"]), cert["pairs"][0]


def test_resumed_run_is_named_when_an_equal_pair_differs(tmp_path):
    synthetic(tmp_path)
    write_run(tmp_path, "lagu_nogate-x", 0, agent(False).state_dict(),
              rows(engagement=0.0, eval_cost=3.5, resumed=1))
    cert = C.certify(str(tmp_path), [{"a": "lagu-x", "b": "lagu_nogate-x", "seed": 0, "expect": "equal"}])
    reasons = cert["pairs"][0]["reasons"]
    assert not cert["passed"] and any(f"lagu_nogate-x_{ENV}_s0 was resumed" in r for r in reasons), reasons
    # an identical resumed pair (both resumed at the same checkpoint) still certifies
    write_run(tmp_path, "lagu-x", 0, agent(True).state_dict(), rows(eval_cost=3.5, resumed=1))
    assert C.certify(str(tmp_path), [{"a": "lagu-x", "b": "lagu_nogate-x", "seed": 0, "expect": "equal"}])["passed"]


def test_default_pairs_are_the_plan_w0_pairs():
    got = {(p["a"], p["b"], p["seed"], p["expect"]) for p in C.DEFAULT_PAIRS}
    assert got == {("lagu-fix", "lagu_nogate-fix", 0, "equal"), ("lagu-fix", "lagu_nogate-fix", 1, "equal"),
                   ("lagu-epi", "lagu_nogate-epi", 0, "equal"), ("lagu-epi", "lagu_nogate-epi", 1, "equal"),
                   ("lagu-sw", "lagu_nogate-sw", 0, "different"),
                   ("lagu-epigg", "lagu_nogate-epi", 0, "different")}


@pytest.fixture(scope="module")
def trained(tmp_path_factory, request):
    """lagu and lagu_nogate (fixed lambda 0.3, T 0.02, 3k steps, hidden 16), plus lagu with gate_grad=1."""
    pytest.importorskip("safety_gymnasium")
    root, out = request.config.rootpath, str(tmp_path_factory.mktemp("certify"))
    base = [sys.executable, "-m", "lagu.train", "--env", ENV, "--seed", "0", "--steps", "3000",
            "--hidden", "16", "--start-steps", "300", "--eval-every", "1000", "--eval-episodes", "1",
            "--dual", "fixed", "--lambda-init", "0.3", "--T", "0.02", "--out", out]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    jobs = [["--arm", "lagu", "--tag", "fix"], ["--arm", "lagu_nogate", "--tag", "fix"]]
    procs = [subprocess.Popen(base + j, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
             for j in jobs]                                  # two at a time
    for p in procs:
        assert p.wait(timeout=600) == 0, p.stdout.read()[-2000:]
    gg = subprocess.run(base + ["--arm", "lagu", "--tag", "gg", "--gate-grad", "1"], cwd=root, env=env,
                        capture_output=True, text=True, timeout=600)
    assert gg.returncode == 0, gg.stdout[-2000:] + gg.stderr[-2000:]
    return root, out


def run_certify(root, out, pairs):
    path = os.path.join(out, "pairs.json")
    with open(path, "w") as f:
        json.dump(pairs, f)
    res = subprocess.run([sys.executable, "-m", "lagu.certify", out, "--pairs", path], cwd=root,
                         capture_output=True, text=True, timeout=300)
    with open(os.path.join(out, "certificate.json")) as f:
        return res.returncode, json.load(f)


def test_real_runs_certify_equal_and_gate_grad_certifies_different(trained):
    root, out = trained
    code, cert = run_certify(root, out, [{"a": "lagu-fix", "b": "lagu_nogate-fix", "seed": 0, "expect": "equal"},
                                         {"a": "lagu-gg", "b": "lagu_nogate-fix", "seed": 0, "expect": "different"}])
    assert code == 0 and cert["passed"], cert["pairs"]
    eq, gg = cert["pairs"]
    assert eq["n_differ"] == 0 and eq["n_tensors"] > 100 and eq["digests"][0] == eq["digests"][1]
    live = eq["live"][f"lagu-fix_{ENV}_s0"]
    assert live["sum_n_risk"] > 0 and live["max_engagement"] > 0 and live["max_lambda"] == 0.3
    assert gg["n_differ"] > 0 and any(k.startswith("actor.") for k in gg["differ"])
    assert os.path.exists(os.path.join(out, "certificate.md"))


def test_real_gate_grad_run_fails_an_equal_expectation(trained):
    root, out = trained
    code, cert = run_certify(root, out, [{"a": "lagu-gg", "b": "lagu_nogate-fix", "seed": 0, "expect": "equal"}])
    assert code == 1 and not cert["passed"]
