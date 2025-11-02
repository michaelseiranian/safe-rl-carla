#!/bin/bash --login
#SBATCH -p gpuV
#SBATCH --gres=gpu:2
#SBATCH -n 12
#SBATCH --mem=48G
#SBATCH -t 3-00:00
#SBATCH -J safe_rl_parallel
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -euo pipefail

# ── modules & env ────────────────────────────────────────────────────────────
module purge
module load apps/python/carla/2022
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate tfuse
export PATH="$CONDA_PREFIX/bin:$PATH"
# — Ray tunable —
export RAY_DISABLE_IMPORT_EXPORT=1

# ── repo root (directory where you typed sbatch) ────────────────────────────
REPO_ROOT="${SLURM_SUBMIT_DIR:?}"      # abort if not set
cd "$REPO_ROOT"
echo "Running in $PWD  (job launched from $SLURM_SUBMIT_DIR)"

# ── PYTHONPATH: CARLA egg + repo ────────────────────────────────────────────
CARLA_PY=$CARLAROOT/carla/PythonAPI
PYV=$(python -c 'import sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}")')
CARLA_EGG=$(ls "$CARLA_PY"/carla/dist/*${PYV}*.egg | head -n1)
export PYTHONPATH="$CARLA_EGG:$CARLA_PY:$CARLA_PY/carla:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ── ensure local output directories exist (repo‑relative) ───────────────────
LOG_DIR=logs
CKPT_DIR=checkpoints_parallel
mkdir -p "$LOG_DIR" || { echo "❌ Cannot write to $LOG_DIR"; exit 1; }
mkdir -p "$CKPT_DIR" || { echo "❌ Cannot write to $CKPT_DIR"; exit 1; }

# ── CARLA servers on GPU 0 ──────────────────────────────────────────────────
export NUM_WORKERS=2                      # adjust ≤ SLURM CPUs
export BASE_PORT=$(shuf -i 15000-30000 -n1)

CUDA_VISIBLE_DEVICES=0
for i in $(seq 0 $((NUM_WORKERS-1))); do
    PORT=$((BASE_PORT + 2*i))
    CarlaUE4.sh -carla-rpc-port=$PORT -opengl -nosound \
                -carla-streaming-port=$((PORT+100)) \
                >"$LOG_DIR/ue4_${SLURM_JOB_ID}_${PORT}.log" 2>&1 &
    UE4_PIDS[$i]=$!
done

# graceful cleanup
cleanup() {
    for p in "${UE4_PIDS[@]}"; do
        kill "$p" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# wait for first server to accept connections
echo -n "waiting for CARLA..."
for _ in {1..60}; do
    ss -ltn | grep -q ":$BASE_PORT" && break
    sleep 1
done
echo "up."

# ── learner on GPU 1 ────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1

ROUTES_DIR=leaderboard/data/training/routes
SCEN_DIR=leaderboard/data/training/scenarios
STEPS_PER_ROUTE=200000      # env‑steps per route
REPLAY=200000               # replay buffer capacity

for ROUTE_XML in $(find "$ROUTES_DIR" -name '*.xml' | sort); do
    BASE=$(basename "$ROUTE_XML" .xml)
    SUBF=$(basename "$(dirname "$ROUTE_XML")")
    SCEN_JSON=$SCEN_DIR/$SUBF/${BASE}.json

    echo "=== training on route: $ROUTE_XML ==="

    python -u safe_rl/train_lagu_parallel.py \
        --host localhost \
        --base-port $BASE_PORT \
        --num-workers $NUM_WORKERS \
        --routes "$ROUTE_XML" \
        --scenarios "$SCEN_JSON" \
        --route-index 0 \
        --steps $STEPS_PER_ROUTE \
        --replay-size $REPLAY \
        --reward-mode blend \
        --prog-scale 1.0 \
        --speed-scale 0.1 \
        --save-dir "$CKPT_DIR"
done
