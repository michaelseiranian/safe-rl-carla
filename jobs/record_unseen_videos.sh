#!/bin/bash --login
#SBATCH -p gpuV
#SBATCH --gres=gpu:1
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J record_unseen_videos
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
export CARLA_TM_PORT=$((CARLA_PORT+600))
export CUDA_VISIBLE_DEVICES=0
export LC_ALL=C
DISPLAY= \
CarlaUE4.sh -carla-rpc-port=$CARLA_PORT \
            -carla-streaming-port=0 \
            -carla-traffic-manager-port=$CARLA_TM_PORT \
            -opengl -nosound -quality-level=Low -RenderOffScreen &

UE4_PID=$!
trap "kill $UE4_PID 2>/dev/null || true" EXIT INT TERM

for i in {1..180}; do
  if ss -ltn | grep -q ":$CARLA_PORT"; then echo "CARLA up on :$CARLA_PORT"; break; fi
  sleep 1
done

# ---------- knobs ----------
CAMERA="${CAMERA:-chase}"   # topdown | chase | ego
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-20}"
EPISODES="${EPISODES:-1}"
BEST_CSV="${BEST_CSV:-eval_logs/best_by_row.csv}"
OUTDIR="${OUTDIR:-videos_seen}"

python -u -m safe_rl.eval.record_unseen_videos \
  --best-csv "$BEST_CSV" \
  --outdir "$OUTDIR" \
  --camera "$CAMERA" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --fps "$FPS" \
  --disable-guard \
  --episodes "$EPISODES"

echo "Videos written under $OUTDIR/"
