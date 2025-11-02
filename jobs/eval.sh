#!/bin/bash --login
#SBATCH -p gpuV
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:30
#SBATCH --mem=12G
#SBATCH -J safe_rl_eval
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -euo pipefail

# ─── modules / env ─────────────────────────────────────────────────
module purge
module load apps/python/carla/2022
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate tfuse
export PATH="$CONDA_PREFIX/bin:$PATH"

REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_ROOT"
echo "Eval Running in $PWD"

CKPT=$1        # passed by sbatch
WANDB_RUN=$2   # training run name/id

# ─── CARLA PYTHONPATH ─────────────────────────────────────────────
CARLA_PY=$CARLAROOT/carla/PythonAPI
PYV=$(python -c 'import sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}")')
CARLA_EGG=$(ls "$CARLA_PY"/carla/dist/*${PYV}*.egg | head -n1)
export PYTHONPATH="$CARLA_EGG:$CARLA_PY:$CARLA_PY/carla:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"


PORT=$(shuf -i 15000-30000 -n1)

# headless server
CUDA_VISIBLE_DEVICES=0 \
"${CARLAROOT}"/carla/CarlaUE4.sh \
    -carla-rpc-port=$PORT \
    -carla-traffic-manager-port=$((PORT+600)) \
    -opengl -nosound &
SERVER_PID=$!
trap "kill $SERVER_PID" EXIT INT TERM

# wait until server listens
for t in {1..60}; do
    ss -ltn | grep -q ":$PORT" && break
    sleep 1
done

CUDA_VISIBLE_DEVICES=0 \
python -m safe_rl.eval.eval_ckpt \
    --ckpt "$CKPT" \
    --routes leaderboard/data/validation/routes.xml \
    --scenarios leaderboard/data/validation/eval_scenarios.json \
    --agent team_code_safe_rl/safe_rl_agent.py \
    --agent-config "" \
    --host localhost \
    --port $PORT \
    --wandb-run "$WANDB_RUN"
