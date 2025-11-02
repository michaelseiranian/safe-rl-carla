#!/bin/bash --login
#SBATCH -p gpuV
#SBATCH --gres=gpu:1
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
#SBATCH -J safe_rl_eval_manifest
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -euo pipefail

module purge
module load apps/python/carla/2022

source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate tfuse
export PATH="$CONDA_PREFIX/bin:$PATH"

REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_ROOT"
echo "Running in $PWD"

CARLA_PY="$CARLAROOT/carla/PythonAPI"
PYV=$(python -c 'import sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}")')
CARLA_EGG=$(ls "$CARLA_PY"/carla/dist/*${PYV}*.egg | head -n1)
export PYTHONPATH="$CARLA_EGG:$CARLA_PY:$CARLA_PY/carla:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---------- start CARLA (offscreen) ----------
export CARLA_PORT
CARLA_PORT=$(shuf -i 15000-39999 -n1)
export CUDA_VISIBLE_DEVICES=0
export LC_ALL=C
DISPLAY= \
CarlaUE4.sh -carla-rpc-port=$CARLA_PORT \
            -carla-streaming-port=0 \
            -carla-traffic-manager-port=$((CARLA_PORT+600)) \
            -opengl -nosound -quality-level=Low -RenderOffScreen &
UE4_PID=$!
trap "kill $UE4_PID 2>/dev/null || true" EXIT INT TERM

for i in {1..180}; do
  if ss -ltn | grep -q ":$CARLA_PORT"; then echo "CARLA up on :$CARLA_PORT"; break; fi
  sleep 1
done

# ---------- knobs ----------
EPISODES=${EPISODES:-1}
SEEDS="${SEEDS:-42}"
COST_WEIGHTS="${COST_WEIGHTS:-1.0,0.5,0.2}"
# Allow env overrides like: ALGOS_OVERRIDE="td3" SPLITS_OVERRIDE="seen"
if [[ -n "${ALGOS_OVERRIDE:-}" ]]; then
  read -r -a ALGOS <<<"${ALGOS_OVERRIDE}"
else
  ALGOS=("lagu" "lag" "td3")
fi
if [[ -n "${SPLITS_OVERRIDE:-}" ]]; then
  read -r -a SPLITS <<<"${SPLITS_OVERRIDE}"
else
  SPLITS=("seen" "unseen")
fi
MANIFEST_DIR="curricula/eval"

# Return a sorted list of all checkpoints for an algo.
# Accepts any file starting with <algo>_ and ending in .pth, sorted by numeric step if found.
list_ckpts_for_algo() {
  local algo="$1"
  shopt -s nullglob
  local files=(checkpoints/${algo}_*.pth)
  if ((${#files[@]} == 0)); then
    echo "ERROR: No checkpoints found for algo=${algo} in checkpoints/" >&2
    exit 1
  fi
  # Sort "naturally": prefer anything that has a trailing number
  printf "%s\n" "${files[@]}" \
  | awk -F'[_.]' '
      {
        step=-1;
        for(i=NF;i>=1;i--) if($i ~ /^[0-9]+$/){ step=$i; break }
        print step "\t" $0
      }
    ' \
  | sort -k1,1n -k2,2 \
  | cut -f2-
}

run_manifest_for_algo_and_split() {
  local ALGO="$1"
  local SPLIT="$2"
  local MANIFEST="${MANIFEST_DIR}/manifest_${SPLIT}.csv"

  echo "------------------------------------------------------------"
  echo " Running: algo=$ALGO | split=$SPLIT | manifest=$MANIFEST"
  echo "------------------------------------------------------------"

  if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
  fi

  local OUT_DIR="eval_logs/${ALGO}/${SPLIT}"
  mkdir -p "$OUT_DIR"

  # Gather all checkpoints for this algo
  mapfile -t CKPTS < <(list_ckpts_for_algo "$ALGO")

  local idx=0
  # strip BOM, skip header/comments/blanks; keep all columns
  sed -e '1s/^\xEF\xBB\xBF//' "$MANIFEST" \
  | awk -F',' '
      /^[[:space:]]*#/ {next}
      /^[[:space:]]*$/ {next}
      tolower($1) ~ /^[[:space:]]*routes_xml[[:space:]]*$/ {next}
      {print}
  ' | while IFS=, read -r \
      RXML SJSON RIDX STEPS LABEL STAG REWARD_MODE SPEED_BONUS STEER_PENALTY SPEED_SCALE PROG_SCALE \
      COST_LIMIT_INIT LAMBDA_ALPHA0 LAMBDA_ALPHA0_AFTER LAMBDA_ALPHA0_RAMP_AFTER DELTA_CLIP_MULT \
      ACTOR_DROPOUT_START ACTOR_DROPOUT_END ACTOR_DROPOUT_DECAY_STEPS \
      M_ENABLE_TRAFFIC M_N_TRAFFIC BONUS_COEF M_RED_MOVING_THRESH_MS PROGRESS_WARMUP_STEPS SUCCESS_COMPLETION_PCT
  do
    LABEL_CLEAN=$(echo "$LABEL" | tr -cd '[:alnum:]_.-')
    # bake split into label so the CSV has it explicitly
    LABEL_WITH_SPLIT="${LABEL_CLEAN}_${SPLIT}"

    # optional args propagated to Python evaluator
    OPTS=()
    [[ -n "${SPEED_BONUS}" ]]            && OPTS+=(--speed-bonus "$SPEED_BONUS")
    [[ -n "${PROG_SCALE}" ]]             && OPTS+=(--prog-scale "$PROG_SCALE")
    [[ -n "${M_ENABLE_TRAFFIC}" ]]       && OPTS+=(--enable-traffic "$M_ENABLE_TRAFFIC")
    [[ -n "${M_N_TRAFFIC}" ]]            && OPTS+=(--n-traffic "$M_N_TRAFFIC")
    [[ -n "${M_RED_MOVING_THRESH_MS}" ]] && OPTS+=(--red-moving-thresh-ms "$M_RED_MOVING_THRESH_MS")

    echo "=== evaluating $ALGO :: ${LABEL_WITH_SPLIT} (route_index=$RIDX) on ${#CKPTS[@]} checkpoints ==="
    echo "    routes_xml=$RXML"

    for CKPT in "${CKPTS[@]}"; do
      CKPT_BASE=$(basename "$CKPT")
      # Try to parse trailing step number for naming
      CKPT_STEP=$(echo "$CKPT_BASE" | grep -Eo '[0-9]+' | tail -n1)

      for SEED in $SEEDS; do
        OUT="${OUT_DIR}/eval_${idx}_${LABEL_WITH_SPLIT}_ckpt${CKPT_STEP:-x}_seed${SEED}.csv"
        if [[ -s "$OUT" ]]; then
          echo "[resume] skipping existing $OUT"
          continue
        fi
        python -u -m safe_rl.eval.run_policy_eval \
          --algo "$ALGO" \
          --ckpt "$CKPT" \
          --routes "$RXML" \
          --scenarios "$SJSON" \
          --route-index "$RIDX" \
          --episodes "$EPISODES" \
          --carla-host localhost \
          --carla-port "$CARLA_PORT" \
          --tm-port $((CARLA_PORT+600)) \
          --cost-weights "$COST_WEIGHTS" \
          --seed "$SEED" \
          --label "$LABEL_WITH_SPLIT" \
          --out-csv "$OUT" \
          --disable-guard \
          "${OPTS[@]}"
      done
    done

    idx=$((idx+1))
  done
}

# ---------- run everything ----------
for ALGO in "${ALGOS[@]}"; do
  for SPLIT in "${SPLITS[@]}"; do
    run_manifest_for_algo_and_split "$ALGO" "$SPLIT"
  done
done

# ---------- consolidate ----------
COMBINED="eval_logs/policy_eval.csv"
rm -f "$COMBINED"
first=true
while IFS= read -r -d '' f; do
  if $first; then
    sed -e '1s/^\xEF\xBB\xBF//' "$f" > "$COMBINED"
    first=false
  else
    tail -n +2 "$f" >> "$COMBINED"
  fi
done < <(find eval_logs -type f -name 'eval_*.csv' -print0 | sort -z)

echo "Wrote consolidated CSV: $COMBINED"

# ---------- best-per-row reports ----------
# (best per (label,algo) and best overall per label across algos)
python -u tools/eval_report.py \
  --csv "$COMBINED" \
  --manifest-seen "curricula/eval/manifest_seen.csv" \
  --manifest-unseen "curricula/eval/manifest_unseen.csv" \
  --outdir "eval_logs" \
  --best-per-row "eval_logs/best_by_row.csv" \
  --topk 3

echo "done."
