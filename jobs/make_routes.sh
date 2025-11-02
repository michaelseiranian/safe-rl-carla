#!/bin/bash --login
#SBATCH -p gpuV
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -t 0-00:10
#SBATCH -J make_routes
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -euo pipefail

# ───────── env / modules ──────────────────────────────────────────
module purge
module load apps/python/carla/2022               # same module you train with

source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate tfuse

export SDL_VIDEODRIVER=offscreen                 # head-less Unreal

# CARLAROOT is defined by the module (note the missing “_”)
if ! [ -d "${CARLAROOT}/carla" ]; then
  echo "ERROR: CARLAROOT not set by module; abort." >&2
  exit 1
fi

# ───────── PYTHONPATH identical to training script ───────────────
CARLA_PY="${CARLAROOT}/carla/PythonAPI"
PYV=$(python -c 'import sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}")')
CARLA_EGG=$(ls "$CARLA_PY"/carla/dist/*${PYV}*.egg | head -n1)
export PYTHONPATH="$CARLA_EGG:$CARLA_PY:$CARLA_PY/carla:${PWD}${PYTHONPATH:+:$PYTHONPATH}"

# ───────── launch CARLA server on GPU-0 ───────────────────────────
PORT=$(shuf -i 15000-30000 -n1)
CUDA_VISIBLE_DEVICES=0 DISPLAY= \
"${CARLAROOT}/carla/CarlaUE4.sh" -carla-rpc-port=$PORT -opengl -nosound \
                                -quality-level=Low -RenderOffScreen &

SERVER_PID=$!
trap "kill $SERVER_PID" EXIT INT TERM

echo -n "Waiting for CARLA ..."
for i in {1..60}; do
  ss -ltn | grep -q ":$PORT" && break
  sleep 0.5
done
if ! ss -ltn | grep -q ":$PORT"; then
  echo " Server failed to start." >&2
  exit 1
fi
echo " up."

# ───────── run the route-generation script ───────────────────────
python curricula/simple/make_routes.py --port $PORT
echo "Route XMLs generated under curricula/simple/"

# done – trap will kill server
