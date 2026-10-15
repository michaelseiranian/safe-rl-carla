"""VERIFICATION (DESCRIPTIVE) of results/paper/e3_td3lag_e9: an independent re-derivation from the raw files.

Every analysis here is DESCRIPTIVE. None is confirmatory. This script re-derives the numbers in
results/paper/e3_td3lag_e9/{tables.md,results.json} from the raw run files with its own code. It
does NOT import lagu.analyze or the audited make.py; it reads only CSV / JSON files and the
listing of results/pareto. It then compares its numbers with the audited results.json and prints
the largest absolute difference per block.

    cd /Users/michaelseiranian/Desktop/safe-rl-carla
    .venv/bin/python results/paper/e3-td3lag-verify/make.py

Pre-specified definitions (prereg.md sections 5, 6, 9, 11; nothing chosen after looking at outcomes):
- Primary endpoint: mean held-out eval_return / eval_cost over the 5 snapshots x 20 layouts in
  <stem>.heldout.json (re-computed here as the mean of the 5 per-snapshot values and checked against
  the file's "pooled" entry, its end_step, snapshot steps 0.8N..N and seed bases 11000 + 20 i).
- Collapse: held-out return < 5, or the run diverged (a non-finite eval_return, eval_cost, qc_traj,
  qbar_traj, q_loss, c_loss or lambda after the warm-up, or a non-finite held-out value).
- Paired bootstrap: resample seed pairs with replacement, 10,000 resamples, mean paired difference,
  percentile 2.5 / 97.5, numpy.random.default_rng(0) (return and cost share the resampled indices).
- Fisher exact, two-sided: sum of hypergeometric probabilities <= that of the observed table.
- Engagement: sum(engagement * n_risk) / sum(n_risk) over rows 25k < step <= 300k (and over all rows),
  pooled over seeds. Mean budget D: sum(delta_avg * n_pi) / sum(n_pi) over rows with n_pi > 0, pooled.
- Lag-U vs TD3-Lag: IQM = mean of the middle 50% (trim floor(n/4) from each end); stratified
  bootstrap = resample each arm independently, 10,000 resamples, default_rng(0) per endpoint;
  P(return) = P(X > Y) + P(X = Y)/2, P(cost) = P(X < Y) + P(X = Y)/2, over all cross-arm pairs.
- E9: in-training final window = mean of eval_return / eval_cost over rows with step >= 0.8N (N = 1M),
  seeds 0, 1, 2; mean and sd (ddof 1) over seeds.
No run is excluded anywhere.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
RES = os.path.join(ROOT, "results")
AUDITED = os.path.join(RES, "paper", "e3_td3lag_e9", "results.json")
N_BOOT = 10_000
PG1, CAR, CIRC = "SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyPointCircle1-v0"
SHORT = {PG1: "PointGoal1", CAR: "CarGoal1", CIRC: "PointCircle1"}
DIVERGE_COLS = ("eval_return", "eval_cost", "qc_traj", "qbar_traj", "q_loss", "c_loss", "lambda")


# ----------------------------------------------------------------------------- raw loading

def fnum(s):
    try:
        return float(s) if s not in ("", None) else float("nan")
    except ValueError:
        return float("nan")


def load_run(d, stem):
    base = os.path.join(RES, d, stem)
    with open(base + ".csv", newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        data = [[fnum(x) for x in row] for row in rdr if row]
    rows = {h: np.array([r[i] if i < len(r) else float("nan") for r in data]) for i, h in enumerate(header)}
    with open(base + ".json") as fh:
        meta = json.load(fh)
    held = None
    if os.path.exists(base + ".heldout.json"):
        with open(base + ".heldout.json") as fh:
            held = json.load(fh)
    return {"stem": stem, "dir": d, "header": header, "rows": rows, "args": meta["args"], "cfg": meta["cfg"],
            "held": held, "steps": int(meta["args"]["steps"]), "warmup": int(meta["args"]["start_steps"])}


def run(d, arm, env, seed):
    return load_run(d, f"{arm}_{env}_s{seed}")


def check_heldout(r):
    """Return (return, cost, problems): mean of the 5 snapshot values, with integrity checks."""
    h, N, probs = r["held"], r["steps"], []
    if h is None:
        return float("nan"), float("nan"), ["no heldout.json"]
    if int(h["end_step"]) != N:
        probs.append(f"end_step {h['end_step']} != steps {N}")
    snaps = h["per_snapshot"]
    if len(snaps) != 5:
        probs.append(f"{len(snaps)} snapshots")
    for i, s in enumerate(snaps):
        if s["step"] != round(N * (0.8 + 0.05 * i)):
            probs.append(f"snapshot {i} at step {s['step']}")
        if s["seed_base"] != 10000 + 1000 + 20 * i:
            probs.append(f"snapshot {i} seed_base {s['seed_base']}")
    ret = float(np.mean([s["eval_return"] for s in snaps]))
    cost = float(np.mean([s["eval_cost"] for s in snaps]))
    if abs(ret - h["pooled"]["eval_return"]) > 1e-9 or abs(cost - h["pooled"]["eval_cost"]) > 1e-9:
        probs.append("pooled != mean of snapshots")
    if int(h.get("episodes", 0)) != 20:
        probs.append(f"episodes {h.get('episodes')}")
    return ret, cost, probs


def diverged(r):
    st = r["rows"]["step"]
    m = st > r["warmup"]
    bad = [c for c in DIVERGE_COLS if c in r["rows"] and not np.all(np.isfinite(r["rows"][c][m]))]
    return bad


def resumed(r):
    return bool("resumed" in r["rows"] and np.nansum(r["rows"]["resumed"] == 1) > 0)


def complete(r):
    return int(np.nanmax(r["rows"]["step"])) >= r["steps"]


# ----------------------------------------------------------------------------- statistics

def boot_pair(d, seed=0):
    d = np.asarray(d, float)
    n = len(d)
    idx = np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))
    m = d[idx].mean(axis=1)                      # (N_BOOT, 2)
    return np.percentile(m, [2.5, 97.5], axis=0).T  # (2, 2): rows return, cost


def fisher(a, n1, c, n2):
    """Two-sided Fisher exact for collapsed a of n1 vs c of n2."""
    k = a + c
    N = n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, k - n2), min(n1, k) + 1) if p(x) <= p0 * (1 + 1e-12))


def iqm(x):
    x = np.sort(np.asarray(x, float), axis=-1)
    n = x.shape[-1]
    k = n // 4
    return x[..., k:n - k].mean(axis=-1)


def p_better(x, y, higher=True):
    x = np.asarray(x, float)[..., :, None]
    y = np.asarray(y, float)[..., None, :]
    if not higher:
        x, y = -x, -y
    return ((x > y) + 0.5 * (x == y)).mean(axis=(-2, -1))


def strat_boot(xa, xb, higher, seed=0):
    rng = np.random.default_rng(seed)
    ba = xa[rng.integers(0, len(xa), size=(N_BOOT, len(xa)))]
    bb = xb[rng.integers(0, len(xb), size=(N_BOOT, len(xb)))]
    q = lambda v: np.percentile(v, [2.5, 97.5])
    ia, ib = iqm(ba), iqm(bb)
    return {"iqm_a": float(iqm(xa)), "iqm_b": float(iqm(xb)), "ci_a": q(ia), "ci_b": q(ib),
            "diff": float(iqm(xa) - iqm(xb)), "ci_diff": q(ia - ib),
            "p": float(p_better(xa, xb, higher)), "ci_p": q(p_better(ba, bb, higher))}


def pooled_w(rs, col, wcol, lo=-np.inf, hi=np.inf):
    num = den = 0.0
    for r in rs:
        st, x, w = r["rows"]["step"], r["rows"][col], r["rows"][wcol]
        m = (st >= lo) & (st <= hi) & np.isfinite(x) & np.isfinite(w) & (w > 0)
        num += float((x[m] * w[m]).sum())
        den += float(w[m].sum())
    return num / den if den > 0 else float("nan")


fmt = lambda x, p=2: "nan" if x is None or not np.isfinite(x) else f"{x:.{p}f}"
fci = lambda c, p=2: f"[{fmt(c[0], p)}, {fmt(c[1], p)}]"


# ----------------------------------------------------------------------------- (1) E3

def e3(d, env, seeds):
    A = [run(d, "lagu-tt", env, s) for s in seeds]
    B = [run(d, "lagu_nogate-tt", env, s) for s in seeds]
    per, probs = [], []
    for s, a, b in zip(seeds, A, B):
        ra, ca, pa = check_heldout(a)
        rb, cb, pb = check_heldout(b)
        probs += [f"{a['stem']}: {x}" for x in pa] + [f"{b['stem']}: {x}" for x in pb]
        for r in (a, b):
            if not complete(r):
                probs.append(f"{r['stem']}: incomplete")
        dva, dvb = diverged(a), diverged(b)
        per.append({"seed": s, "ra": ra, "rb": rb, "ca": ca, "cb": cb,
                    "cola": bool(dva or ra < 5 or not np.isfinite(ra)), "colb": bool(dvb or rb < 5 or not np.isfinite(rb)),
                    "div": dva + dvb, "resumed": resumed(a) or resumed(b)})
    dr = np.array([p["ra"] - p["rb"] for p in per])
    dc = np.array([p["ca"] - p["cb"] for p in per])
    ci = boot_pair(np.c_[dr, dc])
    na, nb = sum(p["cola"] for p in per), sum(p["colb"] for p in per)
    res = {"n": len(per), "mean_ra": np.mean([p["ra"] for p in per]), "mean_rb": np.mean([p["rb"] for p in per]),
           "mean_ca": np.mean([p["ca"] for p in per]), "mean_cb": np.mean([p["cb"] for p in per]),
           "dr": dr.mean(), "dc": dc.mean(), "ci_r": ci[0], "ci_c": ci[1], "col_a": na, "col_b": nb,
           "fisher": fisher(na, len(per), nb, len(per)),
           "eng_early": pooled_w(A, "engagement", "n_risk", 25_001, 300_000),
           "eng_run": pooled_w(A, "engagement", "n_risk"),
           "D": pooled_w(A, "delta_avg", "n_pi"), "delta0": float(A[0]["cfg"]["delta0"]),
           "per": per, "probs": probs, "resumed_seeds": [p["seed"] for p in per if p["resumed"]],
           "diverged": [p["seed"] for p in per if p["div"]]}
    keep = [i for i, p in enumerate(per) if not p["resumed"]]
    if res["resumed_seeds"] and len(keep) >= 2:
        ci2 = boot_pair(np.c_[dr[keep], dc[keep]])
        res["sens"] = {"n": len(keep), "dr": dr[keep].mean(), "dc": dc[keep].mean(), "ci_r": ci2[0], "ci_c": ci2[1]}
    # Monte Carlo spread of the 95% bounds over 20 other bootstrap RNG seeds (sensitivity only)
    mc = np.array([boot_pair(np.c_[dr, dc], seed=k) for k in range(1, 21)])  # (20, 2, 2)
    res["mc_spread"] = float((mc.max(0) - mc.min(0)).max())
    # decomposition of the mean return difference by collapse status of the pair
    groups = {}
    for p in per:
        key = ("gate only" if p["cola"] and not p["colb"] else "no-gate only" if p["colb"] and not p["cola"]
               else "both" if p["cola"] else "neither")
        groups.setdefault(key, []).append(p["ra"] - p["rb"])
    res["decomp"] = {k: (len(v), float(np.sum(v)) / len(per)) for k, v in groups.items()}
    return res


def sgn(x):
    return "+" if x > 0 else "-" if x < 0 else "0"


# ----------------------------------------------------------------------------- (2) Lag-U vs TD3-Lag

def td3(d, env, a, b, seeds_a, seeds_b):
    RA = [run(d, a, env, s) for s in seeds_a]
    RB = [run(d, b, env, s) for s in seeds_b]
    probs, cfgs = [], {}
    for lab, R in ((a, RA), (b, RB)):
        cset = {(r["cfg"]["action_noise"], r["cfg"]["M"], r["cfg"]["use_bonus"], r["cfg"]["use_gate"]) for r in R}
        cfgs[lab] = sorted(cset)
        for r in R:
            _, _, p = check_heldout(r)
            probs += [f"{r['stem']}: {x}" for x in p]
            if not complete(r):
                probs.append(f"{r['stem']}: incomplete")
    xa = np.array([check_heldout(r)[0] for r in RA]), np.array([check_heldout(r)[1] for r in RA])
    xb = np.array([check_heldout(r)[0] for r in RB]), np.array([check_heldout(r)[1] for r in RB])
    return {"ret": strat_boot(xa[0], xb[0], True), "cost": strat_boot(xa[1], xb[1], False),
            "n_a": len(RA), "n_b": len(RB), "cfg": cfgs, "probs": probs,
            "resumed": [r["stem"] for r in RA + RB if resumed(r)],
            "lr": {lab: sorted({(r["cfg"]["actor_lr"], r["cfg"]["critic_lr"], r["cfg"]["dual"], r["cfg"]["lambda_lr"])
                                for r in R}) for lab, R in ((a, RA), (b, RB))}}


# ----------------------------------------------------------------------------- (3) E9

VARIANTS = ("fix005", "fix01", "fix02", "omni-fix03", "omni-fix1", "omni")


def e9():
    inv, summ = [], []
    listing = sorted(os.listdir(os.path.join(RES, "pareto")))
    for v in VARIANTS:
        rs = [run("pareto", f"td3lag-{v}", PG1, s) for s in (0, 1, 2)]
        r2 = rs[2]
        stem = r2["stem"]
        files = sorted(x[len(stem):] for x in listing if x.startswith(stem + ".") or x == stem)
        snapdir = os.path.join(RES, "pareto", stem + ".snap")
        snaps = sorted(int(x[:-3]) for x in os.listdir(snapdir) if x.endswith(".pt")) if os.path.isdir(snapdir) else []
        diff_args = sorted(k for k in set(rs[0]["args"]) | set(r2["args"])
                           if rs[0]["args"].get(k) != r2["args"].get(k))
        diff_cfg = sorted(k for k in set(rs[0]["cfg"]) | set(r2["cfg"]) if rs[0]["cfg"].get(k) != r2["cfg"].get(k))
        diff_args01 = sorted(k for k in set(rs[0]["args"]) | set(rs[1]["args"])
                             if rs[0]["args"].get(k) != rs[1]["args"].get(k))
        st = r2["rows"]["step"]
        inv.append({"variant": v, "files": files, "rows": len(st), "last": int(st.max()), "ncol": len(r2["header"]),
                    "ncol01": (len(rs[0]["header"]), len(rs[1]["header"])),
                    "snaps": (len(snaps), snaps[0] if snaps else None, snaps[-1] if snaps else None,
                              sorted(set(np.diff(snaps))) if len(snaps) > 1 else []),
                    "resumed": resumed(r2), "diverged": diverged(r2), "heldout": r2["held"] is not None,
                    "diff_args_s0_s2": diff_args, "diff_cfg_s0_s2": diff_cfg, "diff_args_s0_s1": diff_args01,
                    "cum_cost": float(r2["rows"]["cum_cost"][np.argmax(st)]),
                    "recipe": r2["args"].get("recipe"), "dual": r2["cfg"]["dual"], "lam0": r2["cfg"]["lambda_init"],
                    "noise": r2["cfg"]["action_noise"], "M": r2["cfg"]["M"]})
        wr, wc, nw, lam = [], [], [], []
        for r in rs:
            st = r["rows"]["step"]
            m = st >= 0.8 * 1_000_000
            nw.append(int(m.sum()))
            wr.append(float(r["rows"]["eval_return"][m].mean()))
            wc.append(float(r["rows"]["eval_cost"][m].mean()))
            lam.append(float(r["rows"]["lambda"][np.argmax(st)]))
        wr, wc = np.array(wr), np.array(wc)
        lams = [r["rows"]["lambda"] for r in rs]
        summ.append({"variant": v, "ret": wr, "cost": wc, "n_window": nw, "lam": lam,
                     "rmean": wr.mean(), "rsd": wr.std(ddof=1), "cmean": wc.mean(), "csd": wc.std(ddof=1),
                     "collapsed": int((wr < 5).sum()), "le25": int((wc <= 25).sum()),
                     "lam_nondecr": all(bool(np.all(np.diff(x[np.isfinite(x)]) >= -1e-12)) for x in lams),
                     "lam_step_last": float(np.mean([np.diff(x[-6:]).mean() for x in lams]))})
    return inv, summ


# ----------------------------------------------------------------------------- comparison with audited json

def cmp(blocks):
    """blocks: list of (name, mine, theirs) scalar or array; prints max abs diff."""
    worst = 0.0
    lines = []
    for name, m, t in blocks:
        dd = float(np.max(np.abs(np.asarray(m, float) - np.asarray(t, float))))
        worst = max(worst, dd)
        if dd > 1e-9:
            lines.append(f"  - {name}: max |mine - audited| = {dd:.3g}")
    return worst, lines


def main():
    os.chdir(ROOT)
    out = []
    P = out.append
    P("# Verification of results/paper/e3_td3lag_e9 (DESCRIPTIVE; independent re-derivation, "
      "results/paper/e3-td3lag-verify/make.py)\n")
    with open(AUDITED) as fh:
        T = json.load(fh)

    # ---------------- (1) E3
    E = {"E1": e3("gate", PG1, list(range(100, 115))),
         CAR: e3("gate_x", CAR, list(range(100, 110))),
         CIRC: e3("gate_x", CIRC, list(range(100, 110)))}
    P("## V1. E3 per task, lagu-tt minus lagu_nogate-tt, held-out endpoint (DESCRIPTIVE re-derivation; "
      "E3 NOT BLIND, first computed by an audit after G3)\n")
    P("| Task | Pairs | Return gate / no gate | Δreturn [95%] | Cost gate / no gate | Δcost [95%] | Collapsed (Fisher p) "
      "| Engagement 25k-300k / run | D (D/δ0) | Signs | MC spread of 95% bounds (20 RNG seeds) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, lab in (("E1", "PointGoal1 (E1 ref)"), (CAR, "CarGoal1"), (CIRC, "PointCircle1")):
        e = E[k]
        P(f"| {lab} | {e['n']} | {fmt(e['mean_ra'])} / {fmt(e['mean_rb'])} | {fmt(e['dr'])} {fci(e['ci_r'])} "
          f"| {fmt(e['mean_ca'])} / {fmt(e['mean_cb'])} | {fmt(e['dc'])} {fci(e['ci_c'])} "
          f"| {e['col_a']} / {e['col_b']} ({fmt(e['fisher'], 3)}) | {fmt(e['eng_early'], 3)} / {fmt(e['eng_run'], 3)} "
          f"| {fmt(e['D'], 3)} ({fmt(e['D'] / e['delta0'], 3)}, δ0 {e['delta0']:g}) | ({sgn(e['dr'])}, {sgn(e['dc'])}) "
          f"| {fmt(e['mc_spread'], 2)} |")
    P("")
    for k in (CAR, CIRC):
        e = E[k]
        holds = sgn(e["dr"]) == sgn(E["E1"]["dr"]) and sgn(e["dc"]) == sgn(E["E1"]["dc"])
        P(f"- {SHORT[k]} section 9 sign check vs E1 ({sgn(E['E1']['dr'])}, {sgn(E['E1']['dc'])}): "
          f"{'holds' if holds else 'does not hold'}; resumed seeds {e['resumed_seeds'] or 'none'}; "
          f"diverged {e['diverged'] or 'none'}; integrity problems {e['probs'] or 'none'}.")
    e1 = E["E1"]
    P(f"- E1 reference: resumed seeds {e1['resumed_seeds']}; without them ({e1['sens']['n']} pairs) "
      f"Δreturn {fmt(e1['sens']['dr'])} {fci(e1['sens']['ci_r'])}, Δcost {fmt(e1['sens']['dc'])} {fci(e1['sens']['ci_c'])}; "
      f"integrity problems {e1['probs'] or 'none'}.")
    pr = ("-10.17", "[-16.08, -3.80]", "-3.31", "[-18.75, 16.70]", "5.42", "15.59", "34.46", "37.77", "9", "3")
    mine = (fmt(e1["dr"]), fci(e1["ci_r"]), fmt(e1["dc"]), fci(e1["ci_c"]), fmt(e1["mean_ra"]), fmt(e1["mean_rb"]),
            fmt(e1["mean_ca"]), fmt(e1["mean_cb"]), str(e1["col_a"]), str(e1["col_b"]))
    P(f"- E1 reference vs prereg.md section 15 (H3 entry): {'identical at 2 decimals' if pr == mine else 'DIFFERS: ' + str(mine)}"
      f"; G2 entry D = 1.12158, engagement 0.8497 vs here {e1['D']:.5f}, {e1['eng_early']:.4f}.")
    for k in (CAR, CIRC):
        dec = E[k]["decomp"]
        P(f"- {SHORT[k]} Δreturn decomposed by pair collapse status (n pairs, contribution to the mean): "
          + "; ".join(f"{g} {n} ({c:+.2f})" for g, (n, c) in sorted(dec.items())) + ".")
    P("")
    P("Per-seed held-out values (gate / no gate):\n")
    P("| Task | Seed | Return | Cost | Collapsed |")
    P("|---|---|---|---|---|")
    for k in (CAR, CIRC):
        for p in E[k]["per"]:
            P(f"| {SHORT[k]} | {p['seed']} | {fmt(p['ra'])} / {fmt(p['rb'])} | {fmt(p['ca'])} / {fmt(p['cb'])} "
              f"| {'yes' if p['cola'] else 'no'} / {'yes' if p['colb'] else 'no'} |")
    P("")

    blocks = []
    for k, tk in (("E1", "E1_reference"), (CAR, CAR), (CIRC, CIRC)):
        e, t = E[k], T["E3"][tk]
        c = t["contrast"]
        blocks += [(f"E3 {tk} return means/diff/ci95", [e["mean_ra"], e["mean_rb"], e["dr"], *e["ci_r"]],
                    [c["return"]["mean_a"], c["return"]["mean_b"], c["return"]["diff"], *c["return"]["ci95"]]),
                   (f"E3 {tk} cost means/diff/ci95", [e["mean_ca"], e["mean_cb"], e["dc"], *e["ci_c"]],
                    [c["cost"]["mean_a"], c["cost"]["mean_b"], c["cost"]["diff"], *c["cost"]["ci95"]]),
                   (f"E3 {tk} collapses/fisher", [e["col_a"], e["col_b"], e["fisher"]],
                    [c["collapses_a"], c["collapses_b"], c["fisher_p"]]),
                   (f"E3 {tk} engagement/D", [e["eng_early"], e["eng_run"], e["D"], e["D"] / e["delta0"]],
                    [t["gate"]["engagement_25k_300k"], t["gate"]["engagement_run"], t["gate"]["D"], t["gate"]["D_over_delta0"]])]
        if k != "E1":
            blocks.append((f"E3 {tk} per-seed", [[p["ra"], p["rb"], p["ca"], p["cb"], p["cola"], p["colb"]] for p in e["per"]],
                           [[p["return_a"], p["return_b"], p["cost_a"], p["cost_b"], p["collapsed_a"], p["collapsed_b"]]
                            for p in t["per_seed"]]))
        else:
            s = c["sensitivity"]
            blocks.append(("E3 E1 sensitivity", [e["sens"]["dr"], *e["sens"]["ci_r"], e["sens"]["dc"], *e["sens"]["ci_c"]],
                           [s["return"]["diff"], *s["return"]["ci95"], s["cost"]["diff"], *s["cost"]["ci95"]]))
    w1, l1 = cmp(blocks)

    # ---------------- (2) Lag-U vs TD3-Lag
    S10, S5 = list(range(100, 110)), list(range(100, 105))
    comps = (("gate", PG1, "lagu-tt", "td3lag-tt", S10, S10), ("gate_literal", PG1, "lagu-lit", "td3lag-lit", S10, S10),
             ("gate_x", CAR, "lagu-tt", "td3lag-tt", S10, S5), ("gate_x", CIRC, "lagu-tt", "td3lag-tt", S10, S5))
    present = sorted(s for s in S10 if os.path.exists(os.path.join(RES, "gate_literal", f"td3lag-lit_{PG1}_s{s}.heldout.json")))
    gx_extra = sorted(x for x in os.listdir(os.path.join(RES, "gate_x")) if x.startswith("td3lag") and x.endswith(".csv")
                      and int(x.split("_s")[-1][:-4]) > 104)
    D2 = []
    P("## V2. Lag-U vs TD3-Lag, held-out endpoint, unpaired IQM (DESCRIPTIVE only; CONFOUNDED: td3lag action noise 0.1, "
      "M = 1, no bonus; lagu action noise 0, M = 3, bonus)\n")
    P("| Dir / task | n | Return IQM Lag-U | Return IQM TD3-Lag | Diff | P(Lag-U return >) | Cost IQM Lag-U | Cost IQM TD3-Lag "
      "| Diff | P(Lag-U cost <) |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for d, env, a, b, sa, sb in comps:
        r = td3(d, env, a, b, sa, sb)
        D2.append(r)
        x, y = r["ret"], r["cost"]
        P(f"| {d} {SHORT[env]} | {r['n_a']} / {r['n_b']} | {fmt(x['iqm_a'])} {fci(x['ci_a'])} | {fmt(x['iqm_b'])} {fci(x['ci_b'])} "
          f"| {fmt(x['diff'])} {fci(x['ci_diff'])} | {fmt(x['p'])} {fci(x['ci_p'])} | {fmt(y['iqm_a'])} {fci(y['ci_a'])} "
          f"| {fmt(y['iqm_b'])} {fci(y['ci_b'])} | {fmt(y['diff'])} {fci(y['ci_diff'])} | {fmt(y['p'])} {fci(y['ci_p'])} |")
    P("")
    for (d, env, a, b, _, _), r in zip(comps, D2):
        P(f"- {d} {SHORT[env]}: cfg (action_noise, M, bonus, gate) {r['cfg']}; (actor_lr, critic_lr, dual, lambda_lr) "
          f"{r['lr']}; resumed {r['resumed'] or 'none'}; integrity problems {r['probs'] or 'none'}.")
    P(f"- td3lag-lit held-out files present for seeds {present} ({len(present)} of 10).")
    P(f"- gate_x td3lag runs beyond s104: {gx_extra or 'none'}.")
    P("")
    blocks = []
    for r, t in zip(D2, T["lagu_vs_td3lag"]):
        for k, tk in (("ret", "return"), ("cost", "cost")):
            m, tt = r[k], t[tk]
            blocks.append((f"TD3 {t['dir']} {t['env']} {tk}",
                           [m["iqm_a"], m["iqm_b"], *m["ci_a"], *m["ci_b"], m["diff"], *m["ci_diff"], m["p"], *m["ci_p"]],
                           [tt["iqm_a"], tt["iqm_b"], *tt["ci_a"], *tt["ci_b"], tt["diff"], *tt["ci_diff"],
                            tt["prob_improvement"], *tt["ci_prob_improvement"]]))
    w2, l2 = cmp(blocks)

    # ---------------- (3) E9
    inv, summ = e9()
    P("## V3. E9 results/pareto seed 2 inventory and 3-seed in-training final window (DESCRIPTIVE)\n")
    P("| Variant | Recipe / dual / λ0 | noise, M | s2 files | rows / last | cols s2 (s0, s1) | snapshots (n, first, last, step) "
      "| resumed / diverged / heldout | args differing s0 vs s2 | cfg differing s0 vs s2 | cum_cost 1M |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in inv:
        P(f"| td3lag-{r['variant']} | {r['recipe']} / {r['dual']} / {r['lam0']:g} | {r['noise']:g}, {r['M']} "
          f"| {', '.join(r['files'])} | {r['rows']} / {r['last']} | {r['ncol']} ({r['ncol01'][0]}, {r['ncol01'][1]}) "
          f"| {r['snaps'][0]}, {r['snaps'][1]}, {r['snaps'][2]}, {r['snaps'][3]} | {r['resumed']} / {r['diverged'] or False} / {r['heldout']} "
          f"| {r['diff_args_s0_s2']} | {r['diff_cfg_s0_s2'] or 'none'} | {r['cum_cost']:.0f} |")
    P("")
    P("| Variant | window rows (s0, s1, s2) | Return s0 / s1 / s2 | Return mean ± sd | Cost s0 / s1 / s2 | Cost mean ± sd "
      "| Collapsed (window return < 5) | Cost ≤ 25 | λ at 1M | λ never decreases (all seeds) / mean λ rise per 50k over 750k-1M |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for r in summ:
        P(f"| td3lag-{r['variant']} | {r['n_window']} | {' / '.join(fmt(x, 1) for x in r['ret'])} | {fmt(r['rmean'], 1)} ± {fmt(r['rsd'], 1)} "
          f"| {' / '.join(fmt(x, 1) for x in r['cost'])} | {fmt(r['cmean'], 1)} ± {fmt(r['csd'], 1)} | {r['collapsed']}/3 "
          f"| {r['le25']}/3 | {' / '.join(fmt(x, 4) for x in r['lam'])} | {r['lam_nondecr']} / {fmt(r['lam_step_last'], 4)} |")
    P("")
    blocks = []
    for r, t in zip(summ, T["E9"]["summary"]):
        blocks.append((f"E9 {r['variant']}", [*r["ret"], *r["cost"], r["rmean"], r["rsd"], r["cmean"], r["csd"],
                                               r["collapsed"], r["le25"], *r["lam"]],
                       [*[t["return"][s] for s in "012"], *[t["cost"][s] for s in "012"], t["return_mean"], t["return_sd"],
                        t["cost_mean"], t["cost_sd"], t["collapsed"], t["at_or_below_limit"],
                        *[t["final_lambda"][s] for s in "012"]]))
    for r, t in zip(inv, T["E9"]["seed2_inventory"]):
        blocks.append((f"E9 inventory {r['variant']}", [r["rows"], r["last"], r["ncol"], r["snaps"][0], r["cum_cost"],
                                                        r["ncol01"][0]],
                       [t["rows"], t["last_step"], t["n_columns"], t["snapshots"], t["cum_cost"], t["seed01_columns"]]))
    w3, l3 = cmp(blocks)

    P("## V4. Agreement with the audited results.json\n")
    for name, w, ls in (("E3 (Table 1, 1b, E1 reference and sensitivity)", w1, l1), ("Lag-U vs TD3-Lag (Table 2)", w2, l2),
                        ("E9 (Tables 3-4)", w3, l3)):
        P(f"- {name}: largest absolute difference {w:.3g}" + (" (agree)" if w < 1e-9 else ""))
        out.extend(ls)
    leftovers = {"verdict": 0, "tost_equivalent": 0}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in leftovers:
                    leftovers[k] += 1
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(T["E3"])
    P(f"- Section 7 quantities left in the audited results.json E3 block (prereg.md 9 says the margins are not applied "
      f"to E3; they are not printed in tables.md): {leftovers}.")
    text = "\n".join(out) + "\n"
    print(text, end="")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_tables.md"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    sys.exit(main())
