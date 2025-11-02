#!/bin/bash --login
#SBATCH -p gpuV
#SBATCH --gres=gpu:1
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J safe_rl_eval
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

################ CARLA paths ################
CARLA_PY=$CARLAROOT/carla/PythonAPI
PYV=$(python -c 'import sys; print("py%d.%d" % (sys.version_info.major, sys.version_info.minor))')
CARLA_EGG=$(ls "$CARLA_PY"/carla/dist/*${PYV}*.egg | head -n1)
export PYTHONPATH="$CARLA_EGG:$CARLA_PY:$CARLA_PY/carla:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"


################ Launch CARLA ################
export CARLA_PORT=$(shuf -i 15000-39999 -n1)
export CUDA_VISIBLE_DEVICES=0     # UE4 only
DISPLAY= \
CarlaUE4.sh -carla-rpc-port=$CARLA_PORT -opengl -nosound \
            -quality-level=Low -RenderOffScreen \
            -carla-traffic-manager-port=$((CARLA_PORT+600)) &

cleanup () {
    echo "[CLEANUP] killing UE4 $UE4_PID"
    kill $UE4_PID 2>/dev/null || true
    pkill -u $USER -f safe_rl/train_lagu.py 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in {1..120}; do ss -ltn | grep -q ":$CARLA_PORT" && break; sleep 1; done
echo "up."

################ Run evaluation ################
python -m safe_rl.eval.run_policy_eval \
       --ckpt       checkpoints/lagu_ckpt_final_7500.pth \
       --routes     curricula/simple/Town01_straight.xml \
       --scenarios  curricula/simple/none.json \
       --episodes   10 \
       --carla-host localhost \
       --carla-port $CARLA_PORT \
       --out-csv    eval_results.csv
