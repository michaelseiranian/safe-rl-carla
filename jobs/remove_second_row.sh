#!/usr/bin/env bash
set -euo pipefail

DIR="eval_logs/lagu/seen"   # change if your path is different

shopt -s nullglob
for f in "$DIR"/*.csv; do
  # only touch files that have at least 3 lines (header + 2 data rows)
  if [ "$(wc -l < "$f")" -ge 3 ]; then
    # remove the 3rd line (second data row), keep everything else
    awk 'NR!=3' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
    echo "Fixed: $f"
  else
    # leave single-row CSVs alone
    echo "OK (unchanged): $f"
  fi
done
