#!/bin/bash --login
#SBATCH -p gpuV
#SBATCH --gres=gpu:1
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -J carla_route_plots
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -euo pipefail

# ---------- modules & env ----------
module purge
module load apps/python/carla/2022

source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate tfuse
export PATH="$CONDA_PREFIX/bin:$PATH"

REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_ROOT"
echo "Running in $PWD"

# ---------- CARLA Python path ----------
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

# ---------- route list ----------
ROUTE_FILES=(
  "curricula/simple/Town01_straight.xml"
  "curricula/simple/Town02_tjunction.xml"
  "curricula/simple/Town03_curve.xml"
  "curricula/medium/Town02_turn_then_bend.xml"
  "curricula/medium/Town03_s_bend_medium.xml"
  "curricula/medium/Town05_two_curves_short.xml"
  "curricula/eval/Town04_eval_straight.xml"
  "curricula/eval/Town05_eval_turn.xml"
  "curricula/eval/Town06_eval_curve.xml"
)

OUT_DIR_TOPO="route_images/topology"
OUT_DIR_RGB="route_images/rgb"
mkdir -p "$OUT_DIR_TOPO" "$OUT_DIR_RGB"

# ---------- render all ----------
for RXML in "${ROUTE_FILES[@]}"; do
  [[ -f "$RXML" ]] || { echo "Missing: $RXML" >&2; continue; }
  base=$(basename "$RXML"); stem="${base%.*}"

  echo "Plotting (topology)  $RXML -> $OUT_DIR_TOPO/${stem}.png"
  python -u tools/plot_routes.py \
    --routes "$RXML" \
    --out "$OUT_DIR_TOPO/${stem}.png" \
    --host localhost --port "$CARLA_PORT" \
    --step 2.5 --dpi 220

  echo "Plotting (RGB)       $RXML -> $OUT_DIR_RGB/${stem}.png"
  python -u tools/plot_routes.py \
    --routes "$RXML" \
    --out "$OUT_DIR_RGB/${stem}.png" \
    --host localhost --port "$CARLA_PORT" \
    --rgb --rgb-fov 90 --rgb-w 1920 --rgb-h 1920
done

echo "Done. See:"
echo "  $OUT_DIR_TOPO  (road graph + route)"
echo "  $OUT_DIR_RGB   (photorealistic top-down + route)"
