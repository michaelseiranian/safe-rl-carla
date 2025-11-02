#!/bin/bash --login
#SBATCH -p gpuV
#SBATCH --gres=gpu:1
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -J safe_rl_eval
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -euo pipefail

module purge
module load apps/python/carla/2022
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate tfuse

REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_ROOT"
mkdir -p logs eval_logs

CARLA_PY="$CARLAROOT/carla/PythonAPI"
PYV=$(python -c 'import sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}")')
CARLA_EGG=$(ls "$CARLA_PY"/carla/dist/*${PYV}*.egg | head -n1)
export PYTHONPATH="$CARLA_EGG:$CARLA_PY:$CARLA_PY/carla:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

start_carla () {
  export CARLA_PORT=$(shuf -i 15000-39999 -n1)
  CUDA_VISIBLE_DEVICES=0 DISPLAY= \
  CarlaUE4.sh -carla-rpc-port="$CARLA_PORT" -carla-streaming-port=0 \
              -opengl -nosound -quality-level=Low -RenderOffScreen &
  UE4_PID=$!
  # wait for port
  for _ in {1..240}; do ss -ltn | grep -q ":$CARLA_PORT" && break; sleep 1; done
  sleep 10
}

stop_carla () { kill "$UE4_PID" 2>/dev/null || true; wait "$UE4_PID" 2>/dev/null || true; }

OUT="eval_logs/policy_eval.csv"
LABEL_PREFIX="${LABEL_PREFIX:-grid}"
CKPT=${CKPT:?set CKPT=path/to/*.pth}
EPISODES=${EPISODES:-5}

declare -a ROUTES=(
  "curricula/simple/Town01_straight.xml,curricula/simple/none.json,0"
  "curricula/simple/Town02_tjunction.xml,curricula/simple/none.json,0"
  "curricula/simple/Town03_curve.xml,curricula/simple/none.json,0"
  "curricula/medium/Town03_s_bend_medium.xml,curricula/simple/none.json,0"
  "curricula/medium/Town02_turn_then_bend.xml,curricula/simple/none.json,0"
  "curricula/medium/Town05_two_curves_short.xml,curricula/simple/none.json,0"
)

for spec in "${ROUTES[@]}"; do
  IFS=, read -r RX SC RI <<<"$spec"
  # boot fresh CARLA with correct town preloaded (optional)
  start_carla
  python -u safe_rl/eval/run_policy_eval.py \
    --ckpt "$CKPT" \
    --routes "$RX" \
    --scenarios "$SC" \
    --route-index "$RI" \
    --carla-port "$CARLA_PORT" \
    --episodes "$EPISODES" \
    --label "${LABEL_PREFIX}_$(basename "$CKPT")_$(basename "$RX" .xml)" \
    --out-csv "$OUT" \
    --cost-weights "1.0,0.5,0.2"
  stop_carla
done
echo "Done → $OUT"
