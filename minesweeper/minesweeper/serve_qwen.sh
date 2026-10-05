#!/usr/bin/env bash
# Host Qwen3-4B-Instruct-2507 with vLLM's OpenAI-compatible server.
#
#   ./serve_qwen.sh                 # defaults below
#   VLLM_PORT=8100 ./serve_qwen.sh  # override any variable
#
# The server is ready once it logs "Application startup complete"; check with
#   curl http://localhost:${VLLM_PORT}/v1/models

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Hugging Face model directory}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-4B-Instruct-2507}"
# Prefixed names: a bare HOST is already exported by conda environments.
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --dtype bfloat16 \
  "$@"
