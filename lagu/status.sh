#!/usr/bin/env bash
# lagu/status.sh results/lr_sweep   -> live process count and the last line of every run log
d=${1:-results/pilot}
echo "lagu.train processes (all, incl. tests and scratch runs): $(ps -axo command | grep -Ec '^[^ ]*python[0-9.]* -m lagu[.]train ' || true)"
for f in "$d"/*.log; do
  [ -e "$f" ] || continue
  printf '%-44s %s\n' "$(basename "$f" .log)" "$(grep -v Adroit "$f" | tail -1 | cut -c1-150)"
done
