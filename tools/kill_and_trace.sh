#!/usr/bin/env bash
# Snap stack traces from trainer + CARLA, then kill them (TERM→KILL).
# Works even if the job is wedged. Requires: bash, ps, pgrep. Optional: gstack/gdb/pstack, ss/lsof, nvidia-smi.

set -euo pipefail

PATTERN="${1:-safe_rl/train_lagu.py}"   # process-match for your trainer
LOGROOT="${2:-logs}"                    # where to put artifacts
TS="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${LOGROOT}/hangdump-${TS}"
mkdir -p "${OUTDIR}"

echo "[info] writing diagnostics to ${OUTDIR}"

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }
bt_native() {
  local pid="$1" out="$2"
  if have gstack; then
    gstack "${pid}" > "${out}" 2>&1 || true
  elif have pstack; then
    pstack "${pid}" > "${out}" 2>&1 || true
  elif have gdb; then
    gdb -q -batch -p "${pid}" \
        -ex "set pagination off" \
        -ex "thread apply all bt" \
        -ex "detach" -ex "quit" > "${out}" 2>&1 || true
  else
    printf "no gdb/gstack/pstack; native backtrace unavailable\n" > "${out}"
  fi
}

children_of() {
  # recursively collect all descendants of a PID
  local root="$1"
  local queue=("$root")
  local all=()
  while ((${#queue[@]})); do
    local p="${queue[0]}"; queue=("${queue[@]:1}")
    all+=("$p")
    mapfile -t kids < <(pgrep -P "$p" || true)
    if ((${#kids[@]})); then queue+=("${kids[@]}"); fi
  done
  printf "%s\n" "${all[@]}" | sort -u
}

# ---------- locate processes ----------
mapfile -t TRAIN_PIDS < <(pgrep -u "$USER" -f "${PATTERN}" || true)
if ((${#TRAIN_PIDS[@]}==0)); then
  echo "[err] no trainer process matching '$PATTERN' found for user '$USER'." >&2
  exit 1
fi

echo "[info] trainer PIDs: ${TRAIN_PIDS[*]}"

# Collect the full process trees (trainer + children)
TREE_PIDS=()
for p in "${TRAIN_PIDS[@]}"; do
  while read -r q; do TREE_PIDS+=("$q"); done < <(children_of "$p")
done
# Add UE4/Carla if present (sometimes parented to init)
mapfile -t UE4_PIDS < <(pgrep -u "$USER" -f 'CarlaUE4|UE4|CarlaUE4-Linux-Shipping' || true)
TREE_PIDS+=("${UE4_PIDS[@]:-}")
# Make unique
mapfile -t TREE_PIDS < <(printf "%s\n" "${TREE_PIDS[@]}" | awk 'NF' | sort -u)

# ---------- snapshot context ----------
{
  echo "== meta =="
  date -Is
  echo "host=$(hostname)"
  echo "user=$USER"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
  echo
  echo "== process tree =="
  if have pstree; then
    for p in "${TRAIN_PIDS[@]}"; do
      echo "-- pstree for $p --"
      pstree -ap "$p" || true
    done
  else
    ps -o pid,ppid,stime,etime,stat,pcpu,pmem,cmd --forest -p "${TRAIN_PIDS[@]}" || true
  fi
} > "${OUTDIR}/summary.txt" 2>&1

# Ports/GPU/FDs snapshot (best-effort)
have ss       && ss -ltnp > "${OUTDIR}/ports.txt" 2>&1 || true
have nvidia-smi && nvidia-smi -q > "${OUTDIR}/nvidia_smi.txt" 2>&1 || true
for p in "${TREE_PIDS[@]}"; do
  have lsof && lsof -p "$p" > "${OUTDIR}/lsof_${p}.txt" 2>&1 || true
done

# ---------- trigger Python faulthandler dumps ----------
# Your train_lagu.py registered SIGUSR1 with faulthandler; dumps go to the process' STDERR
# (in Slurm: logs/<job>.e<id>).
echo "[info] sending SIGUSR1 to Python processes for faulthandler dumps…"
for p in "${TREE_PIDS[@]}"; do
  if [[ -r "/proc/${p}/cmdline" ]] && tr '\0' ' ' < "/proc/${p}/cmdline" | grep -q 'python'; then
    echo "  -> USR1 to PID $p"
    kill -USR1 "$p" 2>/dev/null || true
  fi
done
sleep 2

# ---------- grab native backtraces for everything ----------
echo "[info] capturing native backtraces…"
for p in "${TREE_PIDS[@]}"; do
  comm="$(basename "$(tr '\0' ' ' < /proc/$p/comm 2>/dev/null || echo "?")")"
  bt_native "$p" "${OUTDIR}/bt_${p}_${comm}.txt"
  # kernel thread stack (sometimes useful)
  if [[ -r "/proc/${p}/stack" ]]; then
    cp "/proc/${p}/stack" "${OUTDIR}/kstack_${p}.txt" || true
  fi
done

# ---------- extra: CARLA UE4 backtrace focus ----------
if ((${#UE4_PIDS[@]})); then
  echo "[info] UE4 PIDs: ${UE4_PIDS[*]}"
  for p in "${UE4_PIDS[@]}"; do
    bt_native "$p" "${OUTDIR}/bt_ue4_${p}.txt"
  done
fi

# ---------- terminate ----------
echo "[info] terminating trainer + UE4 (TERM → 8s → KILL)…"
for sig in TERM KILL; do
  for p in "${TREE_PIDS[@]}"; do
    echo "  -> $sig $p"
    kill -"${sig}" "$p" 2>/dev/null || true
  done
  [[ "$sig" == "TERM" ]] && sleep 8
done

echo
echo "[done] Snapshots saved under: ${OUTDIR}"
echo "      Python faulthandler stack dumps are in your Slurm stderr file (e.g., logs/safe_rl_train.e<JOBID>)."
