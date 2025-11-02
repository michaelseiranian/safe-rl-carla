#!/bin/bash --login
#SBATCH -p gpuV
#SBATCH --gres=gpu:2
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 3-00:00
#SBATCH -J safe_rl_train
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

# ==============================================================
#  Robust curriculum runner for CARLA + LagU training
#  - Restarts CarlaUE4 BETWEEN stages (prevents map-change leaks)
#  - Uses random RPC port per stage
#  - Enables dynamic streaming port allocation
#  - Keeps learner checkpoints across stages (--resume)
# ==============================================================

set -euo pipefail

# ---------------- cleanup / traps ----------------
UE4_PID=""
cleanup () {
    echo "[cleanup] stopping training + server"
    if [[ -n "${UE4_PID:-}" ]] && ps -p "$UE4_PID" > /dev/null 2>&1; then
        echo "[cleanup] killing UE4 $UE4_PID"
        kill "$UE4_PID" 2>/dev/null || true
        wait "$UE4_PID" 2>/dev/null || true
    fi
    pkill -u "$USER" -f safe_rl/train_lagu.py 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

# ---------------- environment --------------------
export SDL_VIDEODRIVER=offscreen
export UE4_FORCE_LOG_FLUSH=1
export UE_LOG_SEVERITY="Error"    # UE4 prints only Error/Fatal
ulimit -n 65536 || true

module purge
module load apps/python/carla/2022

# conda env
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate tfuse
export PATH="$CONDA_PREFIX/bin:$PATH"

# ---------------- repo + pythonpath --------------
REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_ROOT"
echo "Running in $PWD"

CARLA_PY="$CARLAROOT/carla/PythonAPI"
PYV=$(python -c 'import sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}")')
CARLA_EGG=$(ls "$CARLA_PY"/carla/dist/*${PYV}*.egg | head -n1)
export PYTHONPATH="$CARLA_EGG:$CARLA_PY:$CARLA_PY/carla:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---------------- paths --------------------------
LOG_DIR="logs"
CKPT_DIR="checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Curriculum manifest (CSV with header):
# routes_xml,scenarios_json,route_index,steps,label,stagnation[,overrides...]
CURRIC="curricula/simple_first/manifest.csv"

# Grab the full header (preserve all columns)
CURRIC_HEADER=$(sed -e '1s/^\xEF\xBB\xBF//' "$CURRIC" \
  | awk 'BEGIN{FS=","} /^[[:space:]]*#/ {next} /^[[:space:]]*$/ {next} {print; exit}')

# ---------------- helpers ------------------------
wait_for_port () {
    local port="$1" tries="${2:-300}"
    echo -n "waiting for CARLA on :$port … "
    for _ in $(seq 1 "$tries"); do
        if command -v ss >/dev/null 2>&1; then
            if ss -ltn | grep -q ":$port"; then
                echo "up."
                return 0
            fi
        elif command -v nc >/dev/null 2>&1; then
            if nc -z localhost "$port" >/dev/null 2>&1; then
                echo "up."
                return 0
            fi
        else
            # Bash /dev/tcp fallback
            if (echo > /dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
                echo "up."
                return 0
            fi
        fi
        sleep 1
    done
    echo "timeout."
    return 1
}

start_carla () {
    local MAP_NAME="${1:-}"
    export CARLA_PORT
    CARLA_PORT=$(shuf -i 15000-39999 -n1)

    # GPU 0 for the simulator
    CUDA_VISIBLE_DEVICES=0 DISPLAY= \
    CarlaUE4.sh \
        -carla-rpc-port="$CARLA_PORT" \
        -carla-streaming-port=0 \
        -carla-traffic-manager-port=$((CARLA_PORT+600)) \
        ${MAP_NAME:+-carla-map="$MAP_NAME"} \
        -opengl -nosound -quality-level=Low -RenderOffScreen &

    UE4_PID=$!
    wait_for_port "$CARLA_PORT" 300

    echo "[startup] CARLA port is up. Waiting 15s for assets to load..."
    sleep 15
    echo "[startup] issuing a dry tick"
    python - <<'PY'
import carla, os, time
c = carla.Client("localhost", int(os.environ["CARLA_PORT"]))
c.set_timeout(10.0)
w = c.get_world()
w.tick()
time.sleep(1.0)
PY
}

stop_carla () {
    if [[ -n "${UE4_PID:-}" ]]; then
        kill "$UE4_PID" 2>/dev/null || true
        wait "$UE4_PID" 2>/dev/null || true
        UE4_PID=""
    fi
}

run_stage () {
    local routes_xml="$1"
    local scenarios_json="$2"
    local route_index="$3"
    local steps="$4"
    local label="$5"
    local stagnation="$6"
    local rest="$7"
    local stage_idx="$8"

    echo
    echo "[stage $stage_idx] routes='${routes_xml}' scenarios='${scenarios_json}' index=${route_index} steps=${steps} label='${label}' stagn=${stagnation}"
    echo

    # Determine desired Town from routes XML (first 'town="..."')
    local MAP_NAME
    MAP_NAME=$(awk 'match($0,/town="([^"]+)"/,a){print a[1]; exit}' "${routes_xml}")
    echo "[stage ${stage_idx}] will start CARLA with map: ${MAP_NAME:-<default>}"

    # Start a fresh simulator for THIS stage (preload correct map)
    start_carla "${MAP_NAME}"

    # Create a one-line manifest so the trainer can read 'stagnation' etc.
    local stage_manifest
    stage_manifest="$(mktemp -t stage_manifest.XXXXXX.csv)"
    # Build one CSV row and pad to header column count
    local NUM_COLS
    NUM_COLS=$(echo "$CURRIC_HEADER" | awk -F',' 'NR==1{print NF}')

    local extra
    extra="${rest#,}"; extra="${extra## }"

    local row
    row="${routes_xml},${scenarios_json},${route_index},${steps},${label},${stagnation}"
    if [[ -n "$extra" ]]; then
      row="${row},${extra}"
    fi

    # Pad with trailing commas so NF(row) == NF(header)
    local present_cols
    present_cols=$(awk -F',' '{print NF}' <<<"$row")
    if (( present_cols < NUM_COLS )); then
      row="${row}$(printf ',%.0s' $(seq $((NUM_COLS - present_cols))))"
    fi

    {
      # Write the original header (all columns)
      echo "$CURRIC_HEADER"
      # Write the padded row
      echo "$row"
    } > "$stage_manifest"

    # GPU 1 for the learner
    export CUDA_VISIBLE_DEVICES=1



    # Run training for this stage only
    set +e

    export CUDA_LAUNCH_BLOCKING=1
    export NCCL_DEBUG=INFO         # if you ever move to multi‑GPU DDP
    export WANDB_RUN_ID=${WANDB_RUN_ID:-$(uuidgen)}
    # Optional: choose baseline at job level (can still be overridden per-row in CSV)
    : "${TRAIN_BASELINE:=lagu}"   # one of lagu|lag|td3

    python -u safe_rl/train_lagu.py \
        --host localhost \
        --port "$CARLA_PORT" \
        --tm-port $((CARLA_PORT+600)) \
        --manifest "$stage_manifest" \
        --resume \
        --seed 42 \
        --baseline "$TRAIN_BASELINE" \
        --reward-mode progress \
        --speed-bonus 0.3 \
        --steer-penalty 0.05 \
        --replay-size 150000 \
        --log-interval 100 \
        --save-dir "$CKPT_DIR"

    PY_STATUS=$?
    set -e

    echo "[train_curriculum] $(date --iso-8601=seconds)  stage ${stage_idx} python exited with $PY_STATUS"

    rm -f "$stage_manifest"

    # Always stop the simulator between stages to free GPU/heap
    stop_carla

    # Propagate non-zero status to Slurm (fail fast)
    if [[ "$PY_STATUS" -ne 0 ]]; then
        exit "$PY_STATUS"
    fi
}

# ---------------- main loop ----------------------
# One WandB run across the whole curriculum (and across requeues)
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
  export WANDB_RUN_ID=$(uuidgen)
  echo "$WANDB_RUN_ID" > "$CKPT_DIR/.wandb_run_id"
fi
export WANDB_RESUME=allow
STAGE=0

# Expect at least 6 columns: routes_xml,scenarios_json,route_index,steps,label,stagnation
# Robustly skip BOM, header, comments, blanks; allow extra columns (ignored via _rest)
while IFS=, read -r routes_xml scenarios_json route_index steps label stagnation _rest; do
    STAGE=$((STAGE+1))
    # Trim possible CRLF
    routes_xml="${routes_xml%$'\r'}"
    scenarios_json="${scenarios_json%$'\r'}"
    route_index="${route_index%$'\r'}"
    steps="${steps%$'\r'}"
    label="${label%$'\r'}"
    stagnation="${stagnation%$'\r'}"

    run_stage "$routes_xml" "$scenarios_json" "$route_index" "$steps" "$label" "$stagnation" "$_rest" "$STAGE"
done < <(
  # strip UTF-8 BOM, drop comments/blank lines, and drop ANY header line
  sed -e '1s/^\xEF\xBB\xBF//' "$CURRIC" \
  | awk -F',' '
      /^[[:space:]]*#/ {next}
      /^[[:space:]]*$/ {next}
      tolower($1) ~ /^[[:space:]]*routes_xml[[:space:]]*$/ {next}
      {print}
    '
)

echo "[train_curriculum] All stages completed successfully."
exit 0