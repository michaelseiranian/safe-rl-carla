"""Descriptive follow-up to the 2026-11-16 addendum (prereg.md section 16, items 17 and 18).

NOT PLANNED. This file was written on 2026-11-16 after k.txt (the declared output of make.py) had
been read, to show what stands behind the declared differences. Nothing here is a test, nothing
enters the Holm family, and no declared number is replaced by a number from this file.

  X1  Per-seed table: held-out return and cost, prediction ratio, the multiplier's average over
      the run (lagu.analyze.lambda_time) and its largest logged value, for the two new arms and
      the two October arms with the same learning rates.
  X2  Per-seed paired differences behind K2 and K3, and the same contrasts with seed 101, seed 109
      or both left out (lagu.analyze.paired_bootstrap, 10,000 resamples, seed 0).
  X3  Pooled prediction ratio with seed 101 left out (lagu.analyze.pooled_calib).
  X4  The multiplier: median over runs of its average over the run, and the median paired
      difference with a percentile bootstrap interval of the median (10,000 resamples,
      numpy default_rng(0)), as in paper/figures/make_v3_numbers.py V3.15.
  X5  Start of training: the first logged step at which the training return reached 10, per run;
      the rows of the two runs on seed 101.
  X6  The policy network of the runs on seed 101: spread of its output over 4096 fixed inputs at
      the saved steps 50k, 100k and 1M (needs the .snap directories, which are not in git; skipped
      with a note when they are absent).
  X7  The signed gate here next to the |Q-bar| gate of October with the same learning rates:
      firing share and budget(all) on the windows of K3 (make.py's dose()).
  X8  How the two return intervals that exclude zero depend on the bootstrap's random numbers: the
      bound nearer zero for 1,000 other bootstrap seeds (10,000 resamples each).
  X9  The in-training evaluation return over the last fifth of training (the secondary endpoint
      window_return of k.txt), per seed, for K3 and K2, and with seeds left out; the number of
      non-zero differences behind each Wilcoxon p of k.txt.

X8 and X9 were added on 2026-11-17, after an independent reader of the paper asked for them.

    .venv/bin/python results/addendum2/extra.py      # writes extra.txt and extra.json here
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import lagu.analyze as A  # noqa: E402

spec = importlib.util.spec_from_file_location("addendum2_make", os.path.join(HERE, "make.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

PG, SEEDS, N_BOOT = M.PG, M.SEEDS, M.N_BOOT
D, G, O, OG = M.DESC_NG, M.DESC_G, M.OCT_NG, "lagu-lit"


def ci95(d):
    return [float(v) for v in A.paired_bootstrap(np.asarray(d, float), N_BOOT, 0)[95][0]]


def med_ci(d):
    d = np.asarray(d, float)
    idx = np.random.default_rng(0).integers(0, len(d), (N_BOOT, len(d)))
    return [float(v) for v in np.percentile(np.median(d[idx], axis=1), [2.5, 97.5])]


def fmt(c, p=2):
    return f"[{c[0]:+.{p}f}, {c[1]:+.{p}f}]"


def main():
    new, old = A.load("results/addendum2"), A.load("results/gate_literal")
    arms = {D: M.get(new, D), G: M.get(new, G), O: M.get(old, O), OG: M.get(old, OG)}
    E = {k: [A.endpoints(r, None, True) for r in rs] for k, rs in arms.items()}
    lam = {k: np.array([A.lambda_time(r) for r in rs]) for k, rs in arms.items()}
    lam_max = {k: np.array([np.nanmax(A.col(r, "lambda")) for r in rs]) for k, rs in arms.items()}
    res: dict = {"seeds": SEEDS, "arms": list(arms)}
    print("NOT PLANNED: descriptive follow-up to the addendum of 2026-11-16, written after k.txt was read "
          "(prereg.md section 16, item 18).")

    # ------------------------------------------------------------------ X1
    print("\nX1. Per seed: held-out return, held-out cost, prediction ratio, multiplier's average over the run, "
          "multiplier's largest logged value")
    res["X1"] = {}
    for k in arms:
        print(f"  {k}")
        res["X1"][k] = {}
        for i, s in enumerate(SEEDS):
            e = E[k][i]
            row = {"return": e["return"], "cost": e["cost"], "ratio": e["calib"], "lambda_time": float(lam[k][i]),
                   "lambda_max": float(lam_max[k][i])}
            res["X1"][k][s] = row
            print(f"    s{s}  return {row['return']:6.2f}  cost {row['cost']:6.2f}  ratio {row['ratio']:.3f}  "
                  f"multiplier average {row['lambda_time']:8.4f}  largest {row['lambda_max']:7.3f}")
        c = np.array([e["cost"] for e in E[k]])
        r = np.array([e["return"] for e in E[k]])
        print(f"    mean return {r.mean():.2f}, mean cost {c.mean():.2f}, lowest cost {c.min():.2f}, highest cost "
              f"{c.max():.2f}, runs with cost <= 25: {int((c <= A.COST_LIMIT).sum())}, runs with return < 5: "
              f"{int((r < A.COLLAPSE).sum())}")

    # ------------------------------------------------------------------ X2
    print("\nX2. Paired differences per seed (first arm minus second), and with seeds left out")
    res["X2"] = {}
    for name, a, b in (("K2", D, O), ("K3", G, D)):
        res["X2"][name] = {}
        for key in ("return", "cost"):
            d = np.array([x[key] - y[key] for x, y in zip(E[a], E[b])])
            print(f"  {name} {a} - {b}, {key}: " + " ".join(f"{v:+.2f}" for v in d))
            out = {"per_seed": dict(zip(SEEDS, map(float, d)))}
            for drop in ((), (101,), (109,), (101, 109)):
                keep = np.array([s not in drop for s in SEEDS])
                c = ci95(d[keep])
                out["without_" + "_".join(map(str, drop)) if drop else "all"] = {"n": int(keep.sum()),
                                                                              "mean": float(d[keep].mean()), "ci95": c}
                print(f"      {'all 10 pairs' if not drop else 'without seed(s) ' + ', '.join(map(str, drop)):26s} "
                      f"n {int(keep.sum()):2d}  mean {d[keep].mean():+.2f}  95% {fmt(c)}")
            res["X2"][name][key] = out

    # ------------------------------------------------------------------ X3
    print("\nX3. Pooled prediction ratio with seed 101 left out")
    res["X3"] = {}
    for k in arms:
        keep = [e for e, s in zip(E[k], SEEDS) if s != 101]
        full = A.pooled_calib(E[k], N_BOOT, 0)
        part = A.pooled_calib(keep, N_BOOT, 0)
        per = [e["calib"] for e in keep]
        res["X3"][k] = {"all": list(full), "without_101": list(part), "per_run_without_101": [min(per), max(per)]}
        print(f"  {k:22s} all 10 runs {full[0]:.3f} [{full[1]:.3f}, {full[2]:.3f}];  without seed 101 {part[0]:.3f} "
              f"[{part[1]:.3f}, {part[2]:.3f}], per run {min(per):.3f} to {max(per):.3f}")
    for name, a, b in (("K1", D, O),):
        d = np.array([x["calib"] - y["calib"] for x, y, s in zip(E[a], E[b], SEEDS) if s != 101])
        c = ci95(d)
        res["X3"]["paired_without_101"] = {"mean": float(d.mean()), "ci95": c}
        print(f"  per-run ratio, {a} minus {b}, without seed 101: {d.mean():+.3f} {fmt(c, 3)}")

    # ------------------------------------------------------------------ X4
    print("\nX4. The multiplier's average over the run")
    res["X4"] = {}
    for k in arms:
        print(f"  {k:22s} median over runs {np.median(lam[k]):.4f}  mean {lam[k].mean():.4f}  largest logged value in any "
              f"run {lam_max[k].max():.3f}")
        res["X4"][k] = {"median": float(np.median(lam[k])), "mean": float(lam[k].mean()), "largest": float(lam_max[k].max())}
    for name, a, b in (("K2", D, O), ("K3", G, D), ("October", OG, O)):
        d = lam[a] - lam[b]
        c = med_ci(d)
        res["X4"][name] = {"median_diff": float(np.median(d)), "ci95_median": c, "per_seed": dict(zip(SEEDS, map(float, d)))}
        print(f"  {a} - {b}: median paired difference {np.median(d):+.4f} {fmt(c, 4)}")

    # ------------------------------------------------------------------ X5
    print("\nX5. Start of training: first logged step with training return >= 10 (None: never)")
    res["X5"] = {}
    for k in arms:
        first = []
        for r in arms[k]:
            step, tr = A.col(r, "step"), A.col(r, "train_return")
            hit = np.flatnonzero(tr >= 10)
            first.append(int(step[hit[0]]) if len(hit) else None)
        res["X5"][k] = dict(zip(SEEDS, first))
        print(f"  {k:22s} " + " ".join(f"{'never' if v is None else f'{v // 1000}k':>6s}" for v in first))
    for k in (D, G):
        r = arms[k][SEEDS.index(101)]
        tr, er, ec, lm = (A.col(r, c) for c in ("train_return", "eval_return", "eval_cost", "lambda"))
        step = A.col(r, "step")
        late = step >= 100_000
        res["X5"][f"{k}_s101"] = {"train_return_min": float(tr.min()), "train_return_max": float(tr.max()),
                                  "eval_return_from_100k": sorted(set(map(float, er[late]))),
                                  "eval_cost_from_100k": sorted(set(map(float, ec[late]))), "lambda_end": float(lm[-1])}
        print(f"  {k} seed 101: training return between {tr.min():+.2f} and {tr.max():+.2f} at all {len(tr)} logged rows; "
              f"in-training evaluation from 100k on: return {sorted(set(np.round(er[late], 4)))}, cost "
              f"{sorted(set(np.round(ec[late], 1)))}; multiplier at the end {lm[-1]:.2f}")
    for k in (D, G):
        r = arms[k][SEEDS.index(109)]
        tr, lm, step = A.col(r, "train_return"), A.col(r, "lambda"), A.col(r, "step")
        late = step >= 500_000
        print(f"  {k} seed 109, rows from 500k: training return {tr[late].min():.1f} to {tr[late].max():.1f}, multiplier "
              f"{lm[late].min():.3f} to {lm[late].max():.3f}")
        res["X5"][f"{k}_s109_from_500k"] = {"train_return": [float(tr[late].min()), float(tr[late].max())],
                                            "lambda": [float(lm[late].min()), float(lm[late].max())]}

    # ------------------------------------------------------------------ X6
    print("\nX6. Policy network: standard deviation of each action component over 4096 fixed inputs")
    res["X6"] = {}
    try:
        import torch

        from lagu.agent import Config, LagU
        gen = torch.Generator().manual_seed(0)
        for k, s in ((D, 101), (G, 101), (D, 100), (G, 100)):
            stem = os.path.join("results/addendum2", f"{k}_{PG}_s{s}")
            with open(stem + ".json") as f:
                meta = json.load(f)
            row = {}
            for step in (50_000, 100_000, 1_000_000):
                snap = torch.load(os.path.join(stem + ".snap", f"{step}.pt"), map_location="cpu", weights_only=False)
                agent = LagU(meta["obs_dim"], meta["act_dim"], Config(**snap["cfg"]))
                agent.actor.load_state_dict(snap["actor"])
                gen.manual_seed(0)
                obs = torch.rand(4096, meta["obs_dim"], generator=gen)
                obs[:, :12] = torch.randn(4096, 12, generator=gen)
                with torch.no_grad():
                    act = agent.actor(obs)
                row[step] = {"mean": act.mean(0).tolist(), "std": act.std(0).tolist()}
            res["X6"][f"{k}_s{s}"] = row
            print(f"  {k:22s} s{s}  " + "   ".join(
                f"{st // 1000}k: mean ({v['mean'][0]:+.3f}, {v['mean'][1]:+.3f}) sd ({v['std'][0]:.4f}, {v['std'][1]:.4f})"
                for st, v in row.items()))
    except FileNotFoundError as err:
        print(f"  skipped: snapshot files are not here ({err.filename})")
        res["X6"] = None

    # ------------------------------------------------------------------ X7
    print("\nX7. Firing share and budget(all) as a fraction of delta0: the signed gate here and the |Q-bar| gate of October")
    res["X7"] = {}
    for k in (G, OG):
        dz = M.dose(arms[k])
        res["X7"][k] = dz
        for name, row in dz.items():
            e, b = row["engagement"], row["budget_all"]
            print(f"  {k:16s} {name:10s} firing share {e['value']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]   budget(all) "
                  f"{b['value']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]")
    # ------------------------------------------------------------------ X8
    print("\nX8. The 95% interval of the mean paired difference in held-out return under 1,000 other bootstrap seeds "
          "(10,000 resamples each)")
    res["X8"] = {}
    for name, a, b, side in (("K3", G, D, 0), ("K2", D, O, 1)):
        d = np.array([x["return"] - y["return"] for x, y in zip(E[a], E[b])])
        bounds = np.array([A.paired_bootstrap(d, N_BOOT, seed)[95][0][side] for seed in range(1, 1001)])
        excl = float(np.mean(bounds > 0) if side == 0 else np.mean(bounds < 0))
        res["X8"][name] = {"bound": "lower" if side == 0 else "upper", "min": float(bounds.min()),
                           "max": float(bounds.max()), "share_excluding_zero": excl}
        print(f"  {name} {a} - {b}: {'lower' if side == 0 else 'upper'} bound between {bounds.min():+.3f} and "
              f"{bounds.max():+.3f}; the interval excludes zero for {100 * excl:.1f}% of the seeds")

    # ------------------------------------------------------------------ X9
    print("\nX9. In-training evaluation return over the last fifth of training (window_return), paired differences")
    res["X9"] = {}
    for name, a, b in (("K3", G, D), ("K2", D, O)):
        d = np.array([x["window_return"] - y["window_return"] for x, y in zip(E[a], E[b])])
        out = {"per_seed": dict(zip(SEEDS, map(float, d)))}
        print(f"  {name} {a} - {b}: " + " ".join(f"{v:+.2f}" for v in d))
        for drop in ((), (101,), (109,), (101, 109)):
            keep = np.array([s not in drop for s in SEEDS])
            c = ci95(d[keep])
            w = A.wilcoxon(d[keep])
            out["without_" + "_".join(map(str, drop)) if drop else "all"] = {"n": int(keep.sum()), "mean": float(d[keep].mean()),
                                                                          "ci95": c, "wilcoxon_p": w["p"], "wilcoxon_n": w["n"]}
            print(f"      {'all 10 pairs' if not drop else 'without seed(s) ' + ', '.join(map(str, drop)):26s} "
                  f"n {int(keep.sum()):2d}  mean {d[keep].mean():+.2f}  95% {fmt(c)}  Wilcoxon p {w['p']:.3f} "
                  f"(on {w['n']} non-zero differences)")
        res["X9"][name] = out
    print("  Non-zero differences behind the Wilcoxon p-values of k.txt (held-out endpoint):")
    for name, a, b in (("K3", G, D), ("K2", D, O)):
        for key in ("return", "cost"):
            d = np.array([x[key] - y[key] for x, y in zip(E[a], E[b])])
            w = A.wilcoxon(d)
            print(f"      {name} {key}: p {w['p']:.4f} on {w['n']} non-zero differences; positive {int((d > 0).sum())}, "
                  f"negative {int((d < 0).sum())}, zero {int((d == 0).sum())}")
            res["X9"][f"{name}_{key}_wilcoxon"] = {"p": w["p"], "n": w["n"], "positive": int((d > 0).sum()),
                                                  "negative": int((d < 0).sum()), "zero": int((d == 0).sum())}
    return res


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = main()
    text = buf.getvalue()
    print(text, end="")
    with open(os.path.join(HERE, "extra.txt"), "w") as f:
        f.write(text)
    A.write_json(out, os.path.join(HERE, "extra.json"))
