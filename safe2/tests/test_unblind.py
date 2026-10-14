"""safe2.analyze --unblind end to end (SPEC section 10, test 33), on a temporary git repository with a bare
`origin`, a temporary key and a small synthetic confirmatory stage (B* and BVS* on three tasks, PHS, two R3
runs). It refuses when a planned run or held-out file is missing, when a hashed file or a decision function
differs without a deviation entry, when plan.json or v1.json differ, when cuts.json is uncommitted or not
pushed, and when a tag is missing; with a listed decision-function change it decides with the method-v0
source. On success it writes UNBLINDED.txt, the decrypted files and every table with its label."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from safe2 import analyze as an
from safe2 import seal, stats, train
from safe2.tests.analyze_helpers import heldout_output, seal_heldout, seal_rows, write_run
from safe2.tests.helpers import make_key

PG1 = an.PG1
REAL = os.path.dirname(os.path.dirname(os.path.abspath(an.__file__)))
PROTOCOL_TEXT = ("# Protocol\n\n## 12. Primary hypothesis\nH_BVS.\n\n## 21. Decision log\n- TBD\n\n## 22. Deviations\n"
                 "None yet.\n\n---\n\n## Appendix A. BVS\nA.1\n")
GIT = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo)] + GIT + list(args), check=True, capture_output=True, text=True).stdout


def commit_all(repo, msg):
    git(repo, "add", "-A", "-f")
    git(repo, "commit", "-q", "-m", msg)


def push(repo, *paths, msg="push"):
    """Commit (force-adding, as results/ is ignored) and push `paths`, as the lead does with HASHLOG.tsv,
    cuts.json and deviations.json."""
    git(repo, "add", "-f", *[str(p) for p in paths])
    git(repo, "commit", "-q", "-m", msg)
    git(repo, "push", "-q", "origin", "method")


def add_deviations(repo, root, entries):
    """deviations.json with `entries`, each entry text logged in protocol section 22, committed and pushed."""
    prot = repo / an.PROTOCOL
    text = prot.read_text()
    sec22 = text[text.index(an.DEVIATIONS_HEAD):text.index("---")]
    new = "".join(f"- {d['entry']}\n" for d in entries if d["entry"] not in sec22)
    prot.write_text(text.replace("None yet.\n", "").replace("\n---", new + "\n---", 1))
    path = os.path.join(root, "deviations.json")
    json.dump(entries, open(path, "w"))
    push(repo, path, prot, msg="deviations")


def small_plan(root):
    """make_plan's records for a v1, kept to B* and BVS* seeds 300-302, their PHS and R3 seeds 100-101."""
    plan = an.make_plan(root)
    keep = {r["id"] for r in plan["runs"] if r["stage"] == "conf" and r["arm"] in ("B*", "BVS*") and r["seed"] <= 302}
    plan["runs"] = [r for r in plan["runs"] if r["id"] in keep]
    plan["evaluations"] = [e for e in plan["evaluations"] if e.get("run") in keep or
                           e["id"] in (f"r3/lagu-lit_{PG1}_s100", f"r3/lagu-lit_{PG1}_s101")]
    return plan


V1 = {"b_star": "B5", "bvs_star": "BVS-SQ-k0.1", "shl_star": None, "shl_killed": "8.4", "bvs_killed": None,
      "saute_star": "SauteA", "l_star": "B5", "run_saute_star": False, "run_l_star": False, "rho_star": 0.1,
      "m_phs": {t: 2.0 for t in an.TASKS}, "r_ref": {t: 20.0 for t in an.TASKS}, "family": ["H_BVS"]}
V05 = {"b_star": "B5", "b_star_base": "omni", "saute_star": "SauteA", "l_star": "B5", "tables": {}}
ENDPOINT = {"B*": (15.0, 22.0), "BVS*": (19.0, 21.0)}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("unblind")
    priv = make_key(base / "keydir")
    repo = base / "repo"
    root = repo / "results" / "method"
    for f in an.V0_FILES + ("safe2/pause.sh",):
        os.makedirs(repo / os.path.dirname(f), exist_ok=True)
        if f == "safe2/stats.py":
            shutil.copy(os.path.join(REAL, f), repo / f)
        else:
            (repo / f).write_text(PROTOCOL_TEXT if f == an.PROTOCOL else f"# placeholder {f}\n")
    os.makedirs(root / "keys")
    shutil.copy(base / "keydir" / "keys" / "seal_pub.pem", root / "keys" / "seal_pub.pem")
    git(repo, "init", "-q", "-b", "method")
    an.v0_hashes(str(root))
    commit_all(repo, "v0")
    git(repo, "tag", "method-v0")
    (root / "v05.json").write_text(json.dumps(V05))
    an.append_hashes(str(root), "method-v0.5", [str(root / "v05.json")])
    commit_all(repo, "v0.5")
    git(repo, "tag", "method-v0.5")
    (root / "v1.json").write_text(json.dumps(V1))
    plan = small_plan(str(root))
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "cuts.json").write_text("[]\n")
    an.append_hashes(str(root), "method-v1", [str(root / "v1.json"), str(root / "plan.json")])
    commit_all(repo, "v1")
    git(repo, "tag", "method-v1")
    origin = base / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-q", "origin", "method", "--tags")
    # the confirmatory stage's files (never committed, as under results/)
    rng = np.random.default_rng(0)
    for r in plan["runs"]:
        R, C = ENDPOINT[r["arm"]]
        R, C = R + rng.normal(0, 2), C + rng.normal(0, 3)
        stem = write_run(root / "conf", r["config"], r["env"], r["seed"], finished=True)
        seal_rows(root, "conf", stem, [
            {"step": 50_000, "cum_cost": 1000.0, "ep_cum": 25.0, "int_cum": 100.0},
            {"step": 1_000_000, "cum_cost": 30_000.0 + 1000 * rng.normal(), "cum_cost_200k": 20_000.0,
             "int_cum": 9_000.0, "shield_steps_cum": 975_000.0, "empty_cum": 50.0, "F_cum": 400.0, "ep_cum": 975.0,
             "entries_cum": 300.0, "misses_cum": 30.0, "rho_num": 9000.0, "rho_den": 965_000.0}])
        for e in plan["evaluations"]:
            if e.get("run") != r["id"]:
                continue
            if e["kind"] == "phs_fit":
                os.makedirs(root / "phs", exist_ok=True)
                (root / e["output"]).write_bytes(b"weights")
            elif e["kind"] == "phs_quality":
                seal_heldout(root, e["output"], {"recall": 0.9, "fa_per_episode": 3.0, "spearman": 0.8,
                                                 "n_states": 1200})
            else:
                shift = {"deployed": 0.0, "shield_off": -1.0, "strict": -2.0, "s_near": 0.5, "phs": 1.0}[e["mode"]]
                seal_heldout(root, e["output"], heldout_output(stem, e["mode"], R + shift, C, rng,
                                                               shielded=r["arm"] == "BVS*" or e["mode"] == "phs"))
    for e in plan["evaluations"]:
        if e["mode"] == "r3" if "mode" in e else False:
            seal_heldout(root, e["output"], heldout_output(os.path.basename(e["source"]), "r3", 8.0, 50.0, rng))
    push(repo, root / seal.HASHLOG, msg="hash log")
    return base, priv


@pytest.fixture
def study(built, tmp_path):
    """A private copy of the built study (repository, origin and files) for one test."""
    base, priv = built
    dst = tmp_path / "s"
    shutil.copytree(base, dst, symlinks=True)
    repo = dst / "repo"
    git(repo, "remote", "set-url", "origin", str(dst / "origin.git"))
    return repo, str(repo / "results" / "method"), priv


def run(root, priv, **kw):
    return an.unblind(root, priv, n_perm=4000, **kw)


def refused(root, priv, pattern, **kw):
    with pytest.raises(an.Refusal) as e:
        run(root, priv, **kw)
    text = "\n".join(e.value.problems)
    assert pattern in text, text
    assert not os.path.exists(os.path.join(root, "UNBLINDED.txt"))
    return text


def modified_stats(path, monkeypatch):
    """A copy of safe2/stats.py whose `label` source differs (a decision-function change), imported as a
    module (registered in sys.modules, as an imported safe2.stats is, so source_hashes() finds it)."""
    src = open(os.path.join(REAL, "safe2", "stats.py")).read()
    new = src.replace('    return "Inconclusive"\n', '    return "Inconclusive"   # edited\n')
    assert new != src
    with open(path, "w") as f:
        f.write(new)
    spec = importlib.util.spec_from_file_location("stats_modified", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "stats_modified", mod)
    spec.loader.exec_module(mod)
    return mod, new


def test_unblind_succeeds_and_labels_every_table(study, monkeypatch):
    repo, root, priv = study
    out = run(root, priv)
    assert open(os.path.join(root, "UNBLINDED.txt")).read().split("\t")[1].strip() == git(repo, "rev-parse", "HEAD").strip()
    prim = out["primary"]
    assert prim["label"] == an.CONFIRMATORY and prim["family"] == ["H_BVS"]
    assert prim["decided_by"].startswith("safe2/stats.py (decision functions unchanged")
    assert any("different base learner" in n for n in prim["notes"])
    # the verdict equals a direct computation on the decrypted endpoints
    R = {a: [[] for _ in an.TASKS] for a in ("BVS*", "B*")}
    C = {a: [[] for _ in an.TASKS] for a in ("BVS*", "B*")}
    plan = json.load(open(os.path.join(root, "plan.json")))
    for r in plan["runs"]:
        o = seal.read_file(os.path.join(root, an.heldout_rel(r["id"].split("/")[1], "deployed")), priv)
        t = an.TASKS.index(r["env"])
        R[r["arm"]][t].append(o["pooled"]["R"])
        C[r["arm"]][t].append(o["pooled"]["C"])
    direct = stats.primary(R["BVS*"], C["BVS*"], R["B*"], C["B*"], [20.0] * 3, n_perm=4000)
    assert prim["ds"] == pytest.approx(direct["ds"]) and prim["p_perm"] == direct["p_perm"]
    assert prim["verdict"] == direct["verdict"] and prim["p_holm"] == direct["p_perm"]
    assert out["secondary"]["label"] == an.DESCRIPTIVE and out["sensitivity"]["label"] == an.DESCRIPTIVE
    assert out["secondary"]["bvs_vs_lstar"]["label"] == an.EXPLORATORY and out["secondary"]["bvs_vs_lstar"]["arm"] == "B*"
    assert out["controls"]["label"] == an.DESCRIPTIVE and out["mechanism"]["label"] == an.DESCRIPTIVE
    assert out["other"]["s01"]["label"] == an.EXPLORATORY
    ctl = out["controls"]
    assert set(ctl) >= {"phs_vs_bstar", "bvs_vs_phs", "phs_quality", "bvs_shield_off_vs_deployed",
                        "bvs_strict_vs_deployed", "bvs_s_near_vs_deployed", "bvs_vs_r3"}
    assert ctl["bvs_shield_off_vs_deployed"]["pooled"]["dr"]["stat"] == pytest.approx(-1.0 / 20.0)
    assert out["mechanism"]["saturation_gap"]["n"] == 9 * 3 and out["mechanism"]["saturation_gap"]["approximately_saturated"]
    tc = out["secondary"]["training_cost"]["cum_cost_200k"]["pooled"]
    assert tc["stat"] == pytest.approx(0.0)
    txt = an.report(root, "unblind", out)
    assert json.load(open(os.path.join(root, "analysis", "unblind.json")))["primary"]["label"] == an.CONFIRMATORY
    assert open(os.path.join(root, "analysis", "unblind.txt")).read() == txt
    assert "primary:  [CONFIRMATORY]" in txt and "bvs_vs_lstar:  [EXPLORATORY]" in txt
    stem = f"B5_{PG1}_s300"
    assert os.path.exists(os.path.join(root, "analysis", "decrypted", "heldout", "conf", f"{stem}.phs.json"))
    assert os.path.exists(os.path.join(root, "analysis", "decrypted", "rows", "conf", f"{stem}.csv"))
    assert out["unblinding"]["n"] == 1 and out["verification"]["resealed_files"] == []
    # a second run appends to UNBLINDED.txt, computes the same primary and keeps the first output
    again = run(root, priv)
    assert again["primary"]["p_perm"] == prim["p_perm"] and again["primary"]["label"] == an.CONFIRMATORY
    lines = open(os.path.join(root, "UNBLINDED.txt")).read().splitlines()
    assert len(lines) == 2 and lines[0].split("\t")[2:] == lines[1].split("\t")[2:]
    assert lines[0].split("\t")[2].startswith("analyze=") and again["unblinding"]["n"] == 2
    full = an.unblind
    monkeypatch.setattr(an, "unblind", lambda r, k: full(r, k, n_perm=4000))
    assert an.main(["--unblind", "--key", priv, "--root", root]) == 0     # the CLI's third run
    assert json.load(open(os.path.join(root, "analysis", "unblind.json")))["primary"]["p_perm"] == prim["p_perm"]
    assert os.path.exists(os.path.join(root, "analysis", "unblind.3.json"))


def test_refuses_a_missing_planned_run_or_heldout_file(study):
    repo, root, priv = study
    stem = f"BVS-SQ-k0.1_{PG1}_s301"
    os.remove(os.path.join(root, an.heldout_rel(stem, "strict")))
    text = refused(root, priv, "step 5")
    assert f"missing planned output {an.heldout_rel(stem, 'strict')}" in text
    os.remove(os.path.join(root, "conf", stem + ".pt"))            # now the run itself is not finished
    text = refused(root, priv, f"conf/{stem}: incomplete")
    assert "missing planned output" not in text                    # outputs are required of finished runs only


def test_deadline_mode_imputes_incomplete_runs_and_labels_the_output(study):
    """Protocol sections 10 and 16 (M8): at the analysis deadline the primary is still computed; a planned
    run that is not finished, or finished without its deployed output, is imputed as unfinished and listed
    by arm, other missing outputs are dropped and listed, and the output is labelled."""
    repo, root, priv = study
    bvs, bst = f"BVS-SQ-k0.1_{PG1}_s301", f"B5_{PG1}_s302"
    os.remove(os.path.join(root, an.heldout_rel(bvs, "strict")))
    os.remove(os.path.join(root, "conf", bvs + ".pt"))                     # BVS* run not finished
    os.remove(os.path.join(root, an.heldout_rel(bst, "deployed")))       # B* run without its endpoint
    refused(root, priv, "step 5")                                          # the ordinary mode refuses
    out = run(root, priv, deadline=True)
    late = out["deadline"]
    assert late["note"] == an.DEADLINE and out["primary"]["deadline_note"] == an.DEADLINE
    assert late["imputed_as_unfinished"]["BVS*"] == [{"run": f"conf/{bvs}", "was": "incomplete"}]
    assert late["imputed_as_unfinished"]["B*"] == [{"run": f"conf/{bst}",
                                                    "was": "finished without its deployed held-out output"}]
    assert an.heldout_rel(bvs, "strict") not in late["dropped_outputs"]       # its run is now unfinished
    assert out["primary"]["label"] == an.CONFIRMATORY and math.isfinite(out["primary"]["ds"])
    with pytest.raises(SystemExit):
        an.main(["--deadline", "--root", root])                            # only with --unblind


def test_refuses_a_modified_hash_log_or_record_file(study):
    repo, root, priv = study
    log = os.path.join(root, seal.HASHLOG)
    lines = open(log).read().splitlines()
    open(log, "w").write("\n".join(lines[1:]) + "\n")
    refused(root, priv, "HASHLOG.tsv has uncommitted changes")
    push(repo, log)
    refused(root, priv, "hash log")


def test_refuses_an_unpushed_hash_log(study):
    repo, root, priv = study
    log = os.path.join(root, seal.HASHLOG)
    git(repo, "rm", "-q", "--cached", log)
    git(repo, "commit", "-q", "-m", "untrack")
    refused(root, priv, "HASHLOG.tsv is not committed")
    git(repo, "add", "-f", log)
    git(repo, "commit", "-q", "-m", "log")
    refused(root, priv, "HASHLOG.tsv: its last commit is not on origin")


def test_refuses_a_changed_code_file_unless_listed_as_a_deviation(study):
    repo, root, priv = study
    with open(repo / "safe2" / "train.py", "a") as f:
        f.write("# changed after v0\n")
    refused(root, priv, "safe2/train.py: differs from its method-v0 hash")
    dev = [{"file": "safe2/train.py", "entry": "section 22, D1 (2026-10-20)"}]
    path = os.path.join(root, "deviations.json")
    json.dump(dev, open(path, "w"))
    refused(root, priv, "is not in protocol section 22")
    prot = repo / an.PROTOCOL
    prot.write_text(prot.read_text().replace("None yet.", "- section 22, D1 (2026-10-20)"))
    refused(root, priv, "deviations.json is not committed")
    add_deviations(repo, root, dev)
    out = run(root, priv)
    assert out["verification"]["deviated_files"] == ["safe2/train.py: differs from its method-v0 hash"]
    assert out["primary"]["label"] == an.CONFIRMATORY
    # a re-run with other deviations is labelled as such; a later edit must be committed and pushed again
    add_deviations(repo, root, dev + [{"file": "safe2/seal.py", "entry": "section 22, D3"}])
    assert run(root, priv)["primary"]["label"] == an.RERUN
    json.dump(dev, open(path, "w"))
    os.remove(os.path.join(root, "UNBLINDED.txt"))
    refused(root, priv, "deviations.json has uncommitted changes")


def test_protocol_decision_log_may_change_but_not_the_frozen_sections(study):
    repo, root, priv = study
    p = repo / an.PROTOCOL
    p.write_text(PROTOCOL_TEXT.replace("- TBD", "- method-v0 pushed 2026-10-12T08:00Z"))
    run(root, priv)
    os.remove(os.path.join(root, "UNBLINDED.txt"))
    p.write_text(PROTOCOL_TEXT.replace("H_BVS.", "H_BVS, two-sided."))
    refused(root, priv, "sections 0-20 or the appendices changed")


def test_decision_function_change_refuses_or_decides_with_the_v0_source(study, tmp_path, monkeypatch):
    repo, root, priv = study
    mod, src = modified_stats(str(tmp_path / "stats_mod.py"), monkeypatch)
    text = refused(root, priv, "decision function label changed", stats_mod=mod)
    assert text.count("changed since method-v0") == 1               # only label's source differs
    (repo / "safe2" / "stats.py").write_text(src)
    add_deviations(repo, root, [{"function": "label", "entry": "section 22, D2"},
                                {"file": "safe2/stats.py", "entry": "section 22, D2"}])
    out = run(root, priv, stats_mod=mod)
    prim = out["primary"]
    assert "method-v0" in prim["decided_by"] and "label" in prim["decided_by"]
    assert prim["sensitivity_current_stats"]["functions"] == ["label", "safe2/stats.py"]
    assert out["verification"]["deviated_functions"] == ["label"]


def test_refuses_changed_plan_or_v1_records(study):
    repo, root, priv = study
    for name in ("plan.json", "v1.json"):
        path = os.path.join(root, name)
        orig = open(path).read()
        d = json.loads(orig)
        d["edited"] = True
        open(path, "w").write(json.dumps(d))
        refused(root, priv, f"{name} differs from its recorded hash")
        open(path, "w").write(orig)


def test_refuses_uncommitted_or_unpushed_cuts(study):
    repo, root, priv = study
    plan = json.load(open(os.path.join(root, "plan.json")))
    assert not [r for r in plan["runs"] if r["cut_group"]]          # this small plan has nothing to cut
    path = os.path.join(root, "cuts.json")
    open(path, "w").write("[]\n\n")
    refused(root, priv, "cuts.json has uncommitted changes")
    git(repo, "add", "-f", path)
    git(repo, "commit", "-q", "-m", "cuts")
    refused(root, priv, "cuts.json: its last commit is not on origin")
    git(repo, "push", "-q", "origin", "method")
    run(root, priv)


def test_refuses_a_missing_or_moved_tag(study, capsys):
    repo, root, priv = study
    git(repo, "push", "-q", "origin", ":refs/tags/method-v1")
    refused(root, priv, "tag method-v1 is not on origin")
    assert an.main(["--unblind", "--key", priv, "--root", root]) == 1          # the CLI prints why, step by step
    assert capsys.readouterr().out.splitlines()[:2] == ["refusing: step 1 (tags):",
                                                        "refusing: tag method-v1 is not on origin"]
    git(repo, "push", "-q", "origin", "method-v1")
    git(repo, "tag", "-d", "method-v0.5")
    refused(root, priv, "tag method-v0.5 is missing locally")


@pytest.mark.parametrize("stem,mode,end_step,why", [
    ("SOME_OTHER_RUN", "deployed", 1_000_000, "not the planned"),
    (None, "strict", 1_000_000, "not the planned"),
    (None, "deployed", 3_000_000, "not the planned"),
])
def test_refuses_a_heldout_output_of_another_run_mode_or_end_step(study, stem, mode, end_step, why):
    repo, root, priv = study
    own = f"B5_{PG1}_s300"
    seal_heldout(root, an.heldout_rel(own, "deployed"),
                 heldout_output(stem or own, mode, 30.0, 5.0, np.random.default_rng(1), end_step=end_step))
    push(repo, os.path.join(root, seal.HASHLOG))
    with pytest.raises(an.Refusal, match=why):
        run(root, priv)


def test_heldout_check_covers_steps_layouts_and_pooled_means():
    rng = np.random.default_rng(0)
    e = {"output": "x", "mode": "deployed", "end_step": 1_000_000}
    o = heldout_output("s", "deployed", 10.0, 5.0, rng)
    an._check_heldout(o, e, "s")
    bad = json.loads(json.dumps(o))
    bad["snapshots"][0]["step"] = 800_000
    with pytest.raises(an.Refusal, match="snapshot steps or layouts"):
        an._check_heldout(bad, e, "s")
    bad = json.loads(json.dumps(o))
    a, b = bad["snapshots"][0]["episodes"][0], bad["snapshots"][1]["episodes"][0]
    a["layout"], b["layout"] = b["layout"], a["layout"]          # every layout once, in the wrong snapshot
    with pytest.raises(an.Refusal, match="snapshot steps or layouts"):
        an._check_heldout(bad, e, "s")
    bad = json.loads(json.dumps(o))
    bad["pooled"]["R"] += 1.0
    with pytest.raises(an.Refusal, match="pooled R"):
        an._check_heldout(bad, e, "s")
    r3 = heldout_output("lagu-lit_x_s100", "r3", 8.0, 50.0, rng)
    an._check_heldout(r3, dict(e, mode="r3"), "lagu-lit_x_s100")
    with pytest.raises(an.Refusal):
        an._check_heldout(o, dict(e, mode="r3"), "s")


def test_refuses_a_v05_rewritten_after_its_tag(study):
    repo, root, priv = study
    p = os.path.join(root, "v05.json")
    d = json.load(open(p))
    d["l_star"] = "B7"
    with pytest.raises(an.Refusal, match="frozen by tag method-v0.5"):     # the CLI path refuses first
        an.write_record(p, d, overwrite=True, root=root)
    an.write_record(p, d, overwrite=True)                                   # done by hand, then re-tagged
    an.append_hashes(root, "method-v0.5", [p])
    commit_all(repo, "v05 rewritten")
    git(repo, "tag", "-f", "method-v1")
    git(repo, "push", "-q", "-f", "origin", "method", "--tags")
    text = refused(root, priv, "step 4")
    assert "v05.json differs from its method-v0.5 version" in text
    assert "v05.json was hashed 2 times" in text


def test_a_listed_change_of_a_stats_constant_decides_with_the_v0_module(study, tmp_path, monkeypatch):
    """ALPHA_TEST is not part of any function hash (M5): listing safe2/stats.py as a changed file still
    decides with the method-v0 module, and the current module is the sensitivity."""
    repo, root, priv = study
    src = open(os.path.join(REAL, "safe2", "stats.py")).read()
    new = src.replace(f"ALPHA_TEST = {stats.ALPHA_TEST!r}", "ALPHA_TEST = 0.9")
    assert new != src
    path = tmp_path / "stats_alpha.py"
    path.write_text(new)
    spec = importlib.util.spec_from_file_location("stats_alpha", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "stats_alpha", mod)
    spec.loader.exec_module(mod)
    (repo / "safe2" / "stats.py").write_text(new)
    add_deviations(repo, root, [{"file": "safe2/stats.py", "entry": "section 22, D4"}])
    prim = run(root, priv, stats_mod=mod)["primary"]
    assert prim["alpha_test"] == stats.ALPHA_TEST and "method-v0" in prim["decided_by"]
    assert prim["sensitivity_current_stats"]["alpha_test"] == 0.9


# --- the tables on a full plan (every arm), without encryption ------------------------------------------
def full_tables(root, **v1_over):
    """tables() on make_plan's full plan: every arm, SHL alive, L* and Saute* arms, BVS-U, BVS-disc, PHS,
    the secondary modes, 3M continuations and R3, with synthetic decrypted outputs."""
    from safe2 import configs
    v1 = {"b_star": "B5", "bvs_star": "BVS-SQ-k0.1", "shl_star": "SHL-g0.90-B4", "saute_star": "SauteB",
          "l_star": "B7", "run_saute_star": True, "run_l_star": True, "rho_star": 0.12,
          "m_phs": {t: 1.0 for t in an.TASKS}, "r_ref": {t: 20.0 for t in an.TASKS}, "family": ["H_BVS"],
          "bvs_killed": None, "shl_killed": None,
          "p1": {"SHL": {"table": {"SHL-g0.90-B4": {"mean_R": 15.0, "mean_C": 22.0, "score": 15.0}}}}}
    v1.update(v1_over)
    v05 = {"b_star": v1["b_star"], "b_star_base": "omni",
           "tables": {"p1_baselines": {"B4": {"mean_R": 14.0, "mean_C": 23.0, "score": 14.0}}}}
    json.dump(v1, open(os.path.join(root, "v1.json"), "w"))
    json.dump(v05, open(os.path.join(root, "v05.json"), "w"))
    plan = an.make_plan(root)
    rng = np.random.default_rng(0)
    runs = {}
    for p in plan["runs"]:
        d = os.path.join(root, p["stage"])
        rc = configs.from_bvs(v1["bvs_star"], configs.DERIVED[p["config"]], 0.12) if p["config"] in configs.DERIVED else None
        write_run(d, p["config"], p["env"], p["seed"], steps=p["steps"], rc=rc)
        r = an.load_run(d, p["config"], p["env"], p["seed"], p["steps"], None, p["stage"])
        r.update(arm=p["arm"], heldout={}, sealed_rows=[
            {"step": 50_000, "cum_cost": 100.0},
            {"step": 1_000_000, "cum_cost": 3e4, "cum_cost_200k": 2e4, "int_cum": 9e3, "shield_steps_cum": 9.75e5,
             "empty_cum": 5.0, "F_cum": 400.0, "ep_cum": 975.0, "entries_cum": 300.0, "misses_cum": 30.0,
             "rho_num": 9e3, "rho_den": 9.65e5, "trig_num": 1e5 if p["config"] == "BVSU" else ""}])
        runs[p["id"]] = r
    for e in plan["evaluations"]:
        if e["kind"] == "heldout" and e["run"] is not None:
            r = runs[e["run"]]
            sh = r["arm"] in ("BVS*", "BVSU", "BVSD") or e["mode"] == "phs"
            r["heldout"][e["mode"]] = heldout_output(r["stem"], e["mode"], 15 + rng.normal(), 22 + rng.normal(), rng,
                                                     end_step=e["end_step"], shielded=sh)
    r3 = [{"id": f"r3/lagu-lit_{PG1}_s{s}", "stem": "x", "arm": "R3", "env": PG1, "seed": s, "stage": "r3",
           "state": "finished", "diverged": False, "n_resumes": 0, "mode": "r3",
           "out": heldout_output("x", "r3", 8.0, 50.0, rng)} for s in range(100, 110)]
    return an.tables(root, plan, runs, r3, {}, stats, None, [], n_perm=200), plan


def test_tables_on_a_full_plan(tmp_path):
    out, plan = full_tables(str(tmp_path))
    an.report(str(tmp_path), "unblind", out)
    arms = out["arms"]["arms"]
    assert out["arms"]["label"] == an.DESCRIPTIVE and an.NOTE_COST_OBS in out["arms"]["notes"]
    for arm in ("B*", "BVS*", "SHL*", "R1", "Saute*", "L*", "PHS"):
        for t in an.TASKS:
            assert arms[f"{arm} | {t} | conf"]["n_endpoint"] > 0, (arm, t)
    assert arms[f"BVSU | {PG1} | conf"]["n"] == 12 and arms[f"R3 | {PG1} | r3"]["n"] == 10
    assert arms[f"B* | {PG1} | cont3m"]["n"] == 3
    assert {"violation_share", "cvar10", "collapses", "no_endpoint", "diverged", "unfinished", "resumed"} <= \
        set(arms[f"B* | {PG1} | conf"])
    for sec in ("secondary", "controls", "mechanism", "sensitivity", "shl"):
        assert an.NOTE_COST_OBS in out[sec]["notes"], sec
    assert isinstance(out["secondary"]["bvs_vs_bstar"]["pooled"]["matched_cost_wording"], bool)
    assert out["controls"]["bvs_vs_saute"]["arm"] == "Saute*"
    tr = out["mechanism"]["per_task"][PG1]["training"]
    assert tr["trigger_rate"] != tr["trigger_rate"]          # not logged for BVS*: NaN (null), not 0.0
    assert out["controls"]["bvs_vs_bvsu"]["mechanism_y"]["training"]["trigger_rate"] > 0
    assert {k: len(v) for k, v in out["other"]["continuations_3m"].items()} == {"B*": 3, "BVS*": 3, "SHL*": 3}


def test_saute_contrast_uses_the_l_star_arm_when_l_star_is_saute_star(tmp_path):
    """B* is a Saute config (SauteB), Saute* = L* = SauteA: the L* arm runs SauteA and is the Saute* arm."""
    out, plan = full_tables(str(tmp_path), b_star="SauteB", saute_star="SauteA", l_star="SauteA",
                            run_saute_star=False, run_l_star=True)
    assert {r["config"] for r in plan["runs"] if r["arm"] == "L*"} == {"SauteA"}
    s = out["controls"]["bvs_vs_saute"]
    assert s["arm"] == "L*" and s["saute_star"] == "SauteA"


def test_killed_bvs_still_tabulates_every_arm(tmp_path):
    out, _ = full_tables(str(tmp_path), bvs_star=None, bvs_killed="8.2", family=[], m_phs=None)
    assert out["primary"]["family"] == [] and "B* | " + PG1 + " | conf" in out["arms"]["arms"]
    assert "SHL* | " + PG1 + " | conf" in out["arms"]["arms"]


def test_refuses_a_section_inserted_among_the_unfrozen_ones(study):
    repo, root, priv = study
    p = repo / an.PROTOCOL
    p.write_text(PROTOCOL_TEXT.replace("---\n", "## 23. Late rule\nanything\n\n---\n"))
    refused(root, priv, "sections 0-20 or the appendices changed")


def test_refuses_tags_out_of_order(study):
    repo, root, priv = study
    tree = git(repo, "rev-parse", "method-v0.5^{tree}").strip()
    orphan = subprocess.run(["git", "-C", str(repo)] + GIT + ["commit-tree", tree, "-m", "orphan"], check=True,
                            capture_output=True, text=True).stdout.strip()
    git(repo, "tag", "-f", "method-v0.5", orphan)
    git(repo, "push", "-q", "-f", "origin", "method-v0.5")
    refused(root, priv, "method-v0 is not an ancestor of method-v0.5")


def test_cut_entries_cover_their_group_in_list_order(tmp_path, monkeypatch):
    root = str(tmp_path)
    full_tables(root)                                        # writes v1.json and the plan's run files
    json.dump(an.make_plan(root), open(os.path.join(root, "plan.json"), "w"))
    json.dump([], open(os.path.join(root, "cuts.json"), "w"))
    with pytest.raises(an.Refusal, match="every run under cut item 1 has finished"):
        an.cut(root, "1")
    os.remove(os.path.join(root, "cont3m", f"B5_{PG1}_s300.pt"))
    entry = an.cut(root, "1")["cut"]
    assert entry["states"][f"B5_{PG1}_s300"] == "incomplete" and entry["states"][f"B5_{PG1}_s301"] == "finished"
    monkeypatch.setattr(an, "pushed", lambda repo, rel: [])
    cuts = json.load(open(os.path.join(root, "cuts.json")))
    plan = json.load(open(os.path.join(root, "plan.json")))
    shl = [r["id"].split("/", 1)[1] for r in plan["runs"] if r["cut_group"] == "2"]
    bad = [dict(cuts[0], stems=cuts[0]["stems"][1:]),
           {"item": "3", "stage": "conf", "reason": "throughput", "stems": []}]
    json.dump(bad, open(os.path.join(root, "cuts.json"), "w"))
    probs = an.check_records(an._repo_of(root), root, {})
    assert any("cut item 1 does not cover its whole group" in p for p in probs)
    assert any("cut item 3 precedes item 2" in p for p in probs)
    json.dump(cuts + [{"item": "2", "stage": "conf", "reason": "throughput", "stems": shl}],
              open(os.path.join(root, "cuts.json"), "w"))
    assert not [p for p in an.check_records(an._repo_of(root), root, {}) if "cuts.json" in p]
