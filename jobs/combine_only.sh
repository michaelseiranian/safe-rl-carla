#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
COMBINED="eval_logs/policy_eval.csv"
OUTDIR="eval_logs"
BEST_PER_ROW="${OUTDIR}/best_by_row.csv"
TOPK=3

# Manifests (same paths used by the SLURM manifest runner)
MANIFEST_SEEN="curricula/eval/manifest_seen.csv"
MANIFEST_UNSEEN="curricula/eval/manifest_unseen.csv"

# sanity checks (optional but helpful)
[[ -f "$MANIFEST_SEEN" ]] || { echo "Missing $MANIFEST_SEEN"; exit 1; }
[[ -f "$MANIFEST_UNSEEN" ]] || { echo "Missing $MANIFEST_UNSEEN"; exit 1; }

# ---- combine individual eval csvs into one ----
rm -f "$COMBINED"
first=true
while IFS= read -r -d '' f; do
  if $first; then sed -e '1s/^\xEF\xBB\xBF//' "$f" > "$COMBINED"; first=false
  else tail -n +2 "$f" >> "$COMBINED"
  fi
done < <(find eval_logs -type f -name 'eval_*.csv' -print0 | sort -z)

echo "Wrote $COMBINED"

# ---- generate ordered reports & plots ----
python -u tools/eval_report.py \
  --csv "$COMBINED" \
  --manifest-seen "$MANIFEST_SEEN" \
  --manifest-unseen "$MANIFEST_UNSEEN" \
  --outdir "$OUTDIR" \
  --best-per-row "$BEST_PER_ROW" \
  --topk "$TOPK" \
  --boxplot-scope top3
