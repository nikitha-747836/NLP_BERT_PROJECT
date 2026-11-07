#!/usr/bin/env bash
# One-command run:  ./run.sh            (full project)
#                   ./run.sh --tasks 1  (any run_all.py arguments)
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=nlp-project5-lines
mkdir -p outputs checkpoints cache dataset
docker build -t "$IMAGE" .
GPU_ARGS=()
if docker info 2>/dev/null | grep -qi nvidia || command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
  echo ">> NVIDIA runtime detected - running with GPU"
else
  echo ">> No NVIDIA runtime detected - running on CPU (slow)"
fi
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--config config.yaml)
docker run --rm "${GPU_ARGS[@]}" --shm-size=2g \
  -v "$PWD/outputs:/app/outputs" -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/cache:/app/cache" -v "$PWD/dataset:/app/dataset" \
  -v nlp_p5_hf_cache:/hf_cache \
  "$IMAGE" "${ARGS[@]}"
echo ">> Done. Open outputs/REPORT.md"
