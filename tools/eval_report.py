#!/usr/bin/env python3
"""
Report generator for policy_eval.csv with strict ordering and grouped plots.

New features vs the original:
- Orders best_by_row.csv and topK_by_row.csv by (algo-major) × (seen+unseen manifest row order).
- Builds Top‑K grouped bar charts per algorithm across ALL labels (x-axis = route label order).
- Adds a cross‑algorithm comparison plot: per label, bars for the best (Top‑1) of lagu, lag, td3.

Ranking rule (unchanged):
  1) route_completion (desc)
  2) cost_per_km (asc)
  3) episode_time_s (asc)
  4) collisions_per_km (asc)

Usage example:
  python tools/eval_report.py \
    --csv eval_logs/policy_eval.csv \
    --manifest-seen curricula/eval/manifest_seen.csv \
    --manifest-unseen curricula/eval/manifest_unseen.csv \
    --outdir eval_logs \
    --best-per-row eval_logs/best_by_row.csv \
    --topk 3

Outputs (in --outdir):
  - best_by_row.csv (ordered)
  - top3_by_row.csv (ordered; name reflects --topk)
  - grouped_top3_per_algo_lagu.png (and for lag, td3)
  - best_across_algos_by_label.png
  - Plus the original summary/plots (hist/box/scatter)
"""

import argparse
import os
import re
from typing import List, Optional, Sequence, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------ utils ------------------------------

ALG_ORDER = ["lagu", "lag", "td3"]


def _infer_baseline(row: pd.Series) -> str:
    """Heuristic baseline extraction from 'baseline' / 'label' / 'ckpt'."""
    for key in ("baseline", "label", "ckpt"):
        if key in row and isinstance(row[key], str):
            s = row[key].lower()
            if "lagu" in s:
                return "lagu"
            # match 'lag' but avoid 'lagu'
            if re.search(r"(^|[^a-z])lag([^a-z]|$)", s):
                return "lag"
            if "td3" in s:
                return "td3"
    return "unknown"


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _fig_path(outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, name)


def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)


# ------------------------------ manifest ordering ------------------------------

def _load_manifest_labels(manifest_path: str, split: str) -> List[str]:
    """Load a manifest CSV and return labels *as they appear in eval CSV*.

    The eval runner appended `_SPLIT` to the `label` field when writing results.
    So we mirror that here: manifest label + '_' + split.
    """
    if not manifest_path:
        return []
    mf = pd.read_csv(manifest_path)
    if "label" not in mf.columns:
        raise ValueError(f"Manifest {manifest_path} has no 'label' column")
    labels = [str(x).strip() for x in mf["label"].tolist()]
    return [f"{lab}_{split}" for lab in labels]


def build_label_order(manifest_seen: Optional[str], manifest_unseen: Optional[str]) -> List[str]:
    order: List[str] = []
    order += _load_manifest_labels(manifest_seen, "seen") if manifest_seen else []
    order += _load_manifest_labels(manifest_unseen, "unseen") if manifest_unseen else []
    return order


def _strip_trailing_split(label: str) -> str:
    """Return label without a final _seen/_unseen if present."""
    s = str(label)
    if s.endswith("_seen"):
        return s[:-5]
    if s.endswith("_unseen"):
        return s[:-7]
    return s


# ------------------------------ plotting ------------------------------

def boxplot_by_baseline(df: pd.DataFrame, metric: str, outdir: str) -> Optional[str]:
    if metric not in df.columns or "baseline" not in df.columns:
        return None

    order = [b for b in ("lagu", "lag", "td3") if b in set(df["baseline"])]
    if not order:
        order = sorted(df["baseline"].astype(str).unique().tolist())

    data = [df.loc[df["baseline"] == b, metric].dropna().values for b in order]
    if not any(len(d) for d in data):
        return None

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=order, showfliers=False, whis=(5, 95))
    plt.xlabel("baseline")
    plt.ylabel(metric)
    plt.title(f"{metric} by baseline")
    out = _fig_path(outdir, f"{metric}_by_baseline.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def hist_plot(df: pd.DataFrame, metric: str, outdir: str, bins: int = 30) -> Optional[str]:
    if metric not in df.columns:
        return None
    vals = df[metric].dropna().values
    if len(vals) == 0:
        return None

    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=bins)
    plt.xlabel(metric)
    plt.ylabel("count")
    plt.title(f"{metric} histogram")
    out = _fig_path(outdir, f"{metric}_hist.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def scatter_completion_vs_cost(df: pd.DataFrame, outdir: str) -> Optional[str]:
    if "route_completion" not in df.columns or "cost_per_km" not in df.columns:
        return None

    plt.figure(figsize=(7, 5))
    if "baseline" in df.columns:
        baselines = df["baseline"].fillna("unknown").astype(str)
        uniq = sorted(baselines.unique().tolist())
        cmap = plt.get_cmap("tab10")
        for i, b in enumerate(uniq):
            sub = df[baselines == b]
            if len(sub) == 0:
                continue
            plt.scatter(sub["route_completion"], sub["cost_per_km"], label=b, alpha=0.7, color=cmap(i % cmap.N))
        plt.legend(title="baseline")
    else:
        plt.scatter(df["route_completion"], df["cost_per_km"], alpha=0.7)

    plt.xlabel("route_completion (%)")
    plt.ylabel("cost_per_km")
    plt.title("Completion vs Cost")
    out = _fig_path(outdir, "progress_vs_cost_per_km.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ------------------------------ summary table ------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if "baseline" not in df.columns:
        df = df.copy()
        df["baseline"] = "all"
    else:
        df = df.copy()
        df["baseline"] = df["baseline"].fillna("unknown")

    def _agg(g: pd.DataFrame) -> pd.Series:
        comp = pd.to_numeric(g.get("route_completion", pd.Series(dtype=float)), errors="coerce")
        cpk = pd.to_numeric(g.get("cost_per_km", pd.Series(dtype=float)), errors="coerce")
        cpk_coll = pd.to_numeric(g.get("collisions_per_km", pd.Series(dtype=float)), errors="coerce")
        cpk_lane = pd.to_numeric(g.get("lane_marks_per_km", pd.Series(dtype=float)), errors="coerce")
        cpk_red = pd.to_numeric(g.get("red_ticks_per_km", pd.Series(dtype=float)), errors="coerce")
        tsec = pd.to_numeric(g.get("episode_time_s", pd.Series(dtype=float)), errors="coerce")

        return pd.Series({
            "episodes": len(g),
            "completion_median": float(np.nanmedian(comp.values)) if len(comp) else np.nan,
            "completion_mean": float(np.nanmean(comp.values)) if len(comp) else np.nan,
            "cost_per_km_median": float(np.nanmedian(cpk.values)) if len(cpk) else np.nan,
            "cost_per_km_mean": float(np.nanmean(cpk.values)) if len(cpk) else np.nan,
            "success_rate": float(np.nanmean((comp >= 99.0).astype(float))) if len(comp) else np.nan,
            "collisions_per_km_mean": float(np.nanmean(cpk_coll.values)) if len(cpk_coll) else np.nan,
            "lane_marks_per_km_mean": float(np.nanmean(cpk_lane.values)) if len(cpk_lane) else np.nan,
            "red_ticks_per_km_mean": float(np.nanmean(cpk_red.values)) if len(cpk_red) else np.nan,
            "time_s_median": float(np.nanmedian(tsec.values)) if len(tsec) else np.nan,
        })

    agg = df.groupby("baseline").apply(_agg).reset_index()
    return agg


# ------------------------------ ranking helpers ------------------------------

RANK_KEYS = [
    ("route_completion", False),  # False => descending
    ("cost_per_km", True),
    ("episode_time_s", True),
    ("collisions_per_km", True),
]


def _normalize_for_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    need = [k for k, _ in RANK_KEYS]
    for k in need:
        if k not in df.columns:
            df[k] = np.nan
        df[k] = pd.to_numeric(df[k], errors="coerce")
    return df


def _sorted_for_rank(df: pd.DataFrame) -> pd.DataFrame:
    by = [k for k, _ in RANK_KEYS]
    ascending = [asc for _, asc in RANK_KEYS]
    return df.sort_values(by=by, ascending=ascending)


def best_per(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if not group_cols:
        return df
    df = _normalize_for_rank(df)
    sdf = _sorted_for_rank(df)
    out = sdf.groupby(group_cols, as_index=False).head(1).reset_index(drop=True)
    return out


def topk_per(df: pd.DataFrame, group_cols: List[str], k: int = 3) -> pd.DataFrame:
    if not group_cols:
        return df
    df = _normalize_for_rank(df)
    sdf = _sorted_for_rank(df)
    out = sdf.groupby(group_cols, as_index=False).head(max(1, int(k))).reset_index(drop=True)
    out["rank"] = out.groupby(group_cols).cumcount().add(1)
    return out


# ------------------------------ ordered CSV outputs ------------------------------

def order_rows_best(best_df: pd.DataFrame, label_order: Sequence[str]) -> pd.DataFrame:
    """Return best_df ordered by ALG_ORDER, then by label_order."""
    parts = []
    for algo in ALG_ORDER:
        for lab in label_order:
            sub = best_df[(best_df["algo"].astype(str) == algo) & (best_df["label"].astype(str) == lab)]
            if len(sub):
                parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else best_df.iloc[0:0]


def order_rows_topk(topk_df: pd.DataFrame, label_order: Sequence[str], k: int) -> pd.DataFrame:
    """Return topk_df ordered by ALG_ORDER, then label_order, then rank 1..k."""
    parts = []
    for algo in ALG_ORDER:
        for lab in label_order:
            sub = topk_df[(topk_df["algo"].astype(str) == algo) & (topk_df["label"].astype(str) == lab)]
            if len(sub):
                sub = sub.sort_values(["rank", "route_completion"], ascending=[True, False])
                # Ensure exactly 1..k ordering (some may be missing)
                for r in range(1, k + 1):
                    sr = sub[sub["rank"] == r]
                    if len(sr):
                        parts.append(sr)
    return pd.concat(parts, ignore_index=True) if parts else topk_df.iloc[0:0]


# ------------------------------ new grouped plots ------------------------------

def plot_topk_ckpt_frequency(topk_df: pd.DataFrame, outdir: str, k: int) -> List[str]:
    """Create one bar chart per algorithm: x = checkpoint name, y = count of appearances in Top‑K."""
    saved: List[str] = []
    if not len(topk_df) or "ckpt" not in topk_df.columns:
        return saved
    for algo in ALG_ORDER:
        sub = topk_df[topk_df["algo"].astype(str) == algo]
        if not len(sub):
            continue
        counts = (
            sub["ckpt"].astype(str)
            .value_counts()
            .sort_values(ascending=False)
        )
        labels = counts.index.tolist()
        vals = counts.values.tolist()
        if not labels:
            continue
        plt.figure(figsize=(max(10, int(len(labels) * 0.5)), 5))
        xpos = np.arange(len(labels))
        plt.bar(xpos, vals)
        plt.xticks(xpos, labels, rotation=30, ha="right")
        plt.ylabel("count (Top‑K appearances)")
        plt.title(f"Top-{k} checkpoint frequency — {algo}")
        out = _fig_path(outdir, f"top{k}_ckpt_hist_{algo}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        saved.append(out)
    return saved

# existing grouped plot function below

def plot_grouped_topk_per_algo(topk_df: pd.DataFrame, label_order: Sequence[str], outdir: str, k: int) -> List[str]:
    """For each algo, one plot with x-axis over label_order, with K bars per label (rank 1..K)."""
    saved: List[str] = []
    if not len(topk_df):
        return saved

    # Precompute rank colors using tab10 (no manual color spec if you prefer defaults)
    cmap = plt.get_cmap("tab10")
    for algo in ALG_ORDER:
        sub_a = topk_df[topk_df["algo"].astype(str) == algo]
        if not len(sub_a):
            continue

        # Prepare data arrays per rank
        heights_by_rank: Dict[int, List[float]] = {r: [] for r in range(1, k + 1)}
        cpk_by_rank: Dict[int, List[Optional[float]]] = {r: [] for r in range(1, k + 1)}
        ckpt_by_rank: Dict[int, List[str]] = {r: [] for r in range(1, k + 1)}
        xlabels: List[str] = []

        for lab in label_order:
            xlabels.append(_strip_trailing_split(lab))
            sub_l = sub_a[sub_a["label"].astype(str) == lab]
            if len(sub_l):
                sub_l = sub_l.sort_values(["rank", "route_completion"], ascending=[True, False]).head(k)
            # Map rank → row
            rows_by_rank = {int(r): rrow for r, rrow in sub_l.set_index("rank").iterrows()} if len(sub_l) else {}
            for r in range(1, k + 1):
                row = rows_by_rank.get(r)
                if row is None:
                    heights_by_rank[r].append(np.nan)
                    cpk_by_rank[r].append(np.nan)
                    ckpt_by_rank[r].append("")
                else:
                    heights_by_rank[r].append(pd.to_numeric(row["route_completion"], errors="coerce"))
                    cpk_by_rank[r].append(pd.to_numeric(row.get("cost_per_km", np.nan), errors="coerce"))
                    ckpt_by_rank[r].append(str(row.get("ckpt", "")))

        # Plot
        n = len(label_order)
        width = 0.8 / max(1, k)
        xpos = np.arange(n)

        plt.figure(figsize=(max(10, int(n * 0.6)), 6))
        bars_by_rank = {}
        for r in range(1, k + 1):
            offs = (r - 1 - (k - 1) / 2) * width
            heights = np.array(heights_by_rank[r], dtype=float)
            bars = plt.bar(xpos + offs, heights, width=width, label=f"rank {r}", alpha=0.9)
            bars_by_rank[r] = bars
            # annotate
            for i, b in enumerate(bars):
                h = heights[i]
                if not np.isnan(h) and h != 0:
                    ck = ckpt_by_rank[r][i]
                    cpk = cpk_by_rank[r][i]
                    ann = ck
                    if not (cpk is None or np.isnan(cpk)):
                        ann += f"\ncpk={cpk:.3f}"
                    plt.text(b.get_x() + b.get_width() / 2, h, ann, ha="center", va="bottom", fontsize=8)

        plt.xticks(xpos, xlabels, rotation=30, ha="right")
        plt.ylabel("route_completion (%)")
        plt.title(f"Top-{k} per label — {algo}")
        plt.legend(title="Top‑K rank")
        out = _fig_path(outdir, f"grouped_top{k}_per_algo_{algo}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        saved.append(out)

    return saved


def plot_best_across_algos_by_label(best_df: pd.DataFrame, label_order: Sequence[str], outdir: str) -> Optional[str]:
    """Per label, show the best (Top‑1) of each algorithm side‑by‑side, without per‑bar text labels."""
    if not len(best_df):
        return None

    xlabels = [_strip_trailing_split(l) for l in label_order]
    n = len(label_order)
    width = 0.8 / max(1, len(ALG_ORDER))
    xpos = np.arange(n)

    plt.figure(figsize=(max(10, int(n * 0.6)), 6))

    for j, algo in enumerate(ALG_ORDER):
        heights = []
        for lab in label_order:
            row = best_df[(best_df["algo"].astype(str) == algo) & (best_df["label"].astype(str) == lab)]
            if len(row):
                rc = pd.to_numeric(row["route_completion"].values[0], errors="coerce")
            else:
                rc = np.nan
            heights.append(rc)
        plt.bar(xpos + (j - (len(ALG_ORDER) - 1)/2) * width, heights, width=width, label=algo)

    plt.xticks(xpos, xlabels, rotation=30, ha="right")
    plt.ylabel("route_completion (%)")
    plt.title("Best across algorithms per label (Top‑1)")
    plt.legend(title="algo")
    out = _fig_path(outdir, "best_across_algos_by_label.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to policy_eval.csv")
    ap.add_argument("--outdir", default=None, help="Where to save plots (default: alongside CSV)")
    ap.add_argument("--manifest-seen", default=None, help="Path to manifest_seen.csv (to enforce ordering)")
    ap.add_argument("--manifest-unseen", default=None, help="Path to manifest_unseen.csv (to enforce ordering)")
    ap.add_argument("--best-per-row", default=None, help="Path to save best per (label, algo) [Top-1]")
    ap.add_argument("--best-overall", default=None, help="(unused here; kept for compat)")
    ap.add_argument("--topk", type=int, default=3, help="Top-K size for additional rankings (default: 3)")
    ap.add_argument("--boxplot-scope", choices=["all","top1","top3"], default="all",
                    help="Which runs to use for by-baseline boxplots: all runs, Top-1 per (label,algo), or Top-3 per (label,algo). Default: all")
    args = ap.parse_args()

    csv_path = args.csv
    outdir = args.outdir or os.path.dirname(os.path.abspath(csv_path)) or "."

    # paths for optional scoped summaries
    summary_top1_path = None
    summary_top3_path = None

    # Load data
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Infer baseline if missing
    if "baseline" not in df.columns:
        df["baseline"] = df.apply(_infer_baseline, axis=1)

    # Ensure numeric metrics
    numeric_cols = [
        "route_completion", "episode_time_s", "cost", "cost_per_km",
        "collisions_total", "lane_marks_total", "red_ticks_total",
        "collisions_per_km", "lane_marks_per_km", "red_ticks_per_km",
        "km_travelled", "timeout_s", "terminated", "truncated",
        "collision_fail", "stagnation",
    ]
    _ensure_numeric(df, numeric_cols)

    # Build label order from manifests (seen first, then unseen)
    label_order = build_label_order(args.manifest_seen, args.manifest_unseen)
    if not label_order:
        # Fallback to whatever order appears in the CSV, grouped by split semantics
        uniq = df["label"].astype(str).unique().tolist()
        # push those with _seen ahead of _unseen
        label_order = sorted(uniq, key=lambda s: (0 if str(s).endswith("_seen") else 1, str(s)))

        # Plots (original set)
    saved = []

    # Scope selection helper for by-baseline boxplots (cost_per_km and route_completion)
    scope = args.boxplot_scope
    if scope == "top1" and _has_cols(df, ["label","algo"]):
        df_for_box = best_per(df, ["label","algo"])  # Top-1 per (label,algo)
    elif scope == "top3" and _has_cols(df, ["label","algo"]):
        df_for_box = topk_per(df, ["label","algo"], k=3)  # Top-3 per (label,algo)
    else:
        df_for_box = df

    # By-baseline boxplots (scoped)
    p = boxplot_by_baseline(df_for_box, "cost_per_km", outdir)
    if p: saved.append(p)
    p = boxplot_by_baseline(df_for_box, "route_completion", outdir)
    if p: saved.append(p)

    # Histograms on full df
    for metric in ("cost_per_km", "route_completion"):
        p = hist_plot(df, metric, outdir)
        if p: saved.append(p)
    p = scatter_completion_vs_cost(df, outdir)
    if p: saved.append(p)

    # Summary by baseline (always over full df)
    summary_df = summarize(df)
    print("=== Evaluation Summary (by baseline) ===")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(summary_df.to_string(index=False))
    summary_path = _fig_path(outdir, "summary_by_baseline.csv")
    summary_df.to_csv(summary_path, index=False)

    # Also write summaries computed over Top-1 and Top-3 per (label, algo)
    if _has_cols(df, ["label","algo"]):
        # Top‑1
        df_top1 = best_per(df, ["label","algo"])
        summary_top1 = summarize(df_top1)
        summary_top1_path = _fig_path(outdir, "summary_by_baseline_top1.csv")
        summary_top1.to_csv(summary_top1_path, index=False)
        # Top‑3
        df_top3 = topk_per(df, ["label","algo"], k=3)
        summary_top3 = summarize(df_top3)
        summary_top3_path = _fig_path(outdir, "summary_by_baseline_top3.csv")
        summary_top3.to_csv(summary_top3_path, index=False)

    # -------- selections --------
    topk = max(1, int(args.topk))

    # Per (label, algo) best and Top‑K
    have_la_cols = _has_cols(df, ["label", "algo"])
    if have_la_cols and args.best_per_row:
        best_la = best_per(df, ["label", "algo"])  # Top‑1 within each (label, algo)
        best_la_ordered = order_rows_best(best_la, label_order)
        best_la_ordered.to_csv(args.best_per_row, index=False)
        print(f"Saved ordered best per (label, algo): {args.best_per_row}")

        topk_la = topk_per(df, ["label", "algo"], k=topk)
        topk_la_ordered = order_rows_topk(topk_la, label_order, k=topk)
        topk_la_path = _fig_path(os.path.dirname(args.best_per_row) or outdir, f"top{topk}_by_row.csv")
        topk_la_ordered.to_csv(topk_la_path, index=False)
        print(f"Saved ordered Top-{topk} per (label, algo): {topk_la_path}")        # New grouped plots
        saved += plot_grouped_topk_per_algo(topk_la_ordered, label_order, outdir, k=topk)
        saved += plot_topk_ckpt_frequency(topk_la_ordered, outdir, k=topk)
        p2 = plot_best_across_algos_by_label(best_la_ordered, label_order, outdir)
        if p2:
            saved.append(p2)

    # Report saved artifacts
    print("Saved files:")
    for p in saved:
        print(" -", p)
    print(" -", summary_path)
    if summary_top1_path: print(" -", summary_top1_path)
    if summary_top3_path: print(" -", summary_top3_path)
    if args.best_per_row:    print(" -", args.best_per_row)
    if args.best_per_row:    print(" -", os.path.join(os.path.dirname(args.best_per_row) or outdir, f"top{topk}_by_row.csv"))


if __name__ == "__main__":
    main()
