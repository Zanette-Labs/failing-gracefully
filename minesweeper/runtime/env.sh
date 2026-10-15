#!/usr/bin/env bash
# All writable runtime state belongs on the node's /tmp disk.
export MINESWEEPER_RUN_ROOT="${MINESWEEPER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# Reject ambiguous paths and keep all generated files on the node-local /tmp disk.
[[ "$MINESWEEPER_RUN_ROOT" =~ ^/tmp/[a-zA-Z0-9_./-]+$ ]] || { echo "Choose an absolute /tmp path without spaces" >&2; return 1; }
MINESWEEPER_RUN_ROOT=$(realpath -m "$MINESWEEPER_RUN_ROOT")
[[ "$MINESWEEPER_RUN_ROOT" == /tmp/* ]] || { echo "Run root must resolve under /tmp" >&2; return 1; }
export ENROOT_RUNTIME_PATH="$MINESWEEPER_RUN_ROOT/enroot/runtime"
export ENROOT_CACHE_PATH="$MINESWEEPER_RUN_ROOT/enroot/cache"
export ENROOT_DATA_PATH="$MINESWEEPER_RUN_ROOT/enroot/data"
export ENROOT_TEMP_PATH="$MINESWEEPER_RUN_ROOT/enroot/tmp"
export ENROOT_CONFIG_PATH="$MINESWEEPER_RUN_ROOT/enroot/config"
export ENROOT_MAX_PROCESSORS=32
export ENROOT_MOUNT_HOME=no
export XDG_CACHE_HOME="$MINESWEEPER_RUN_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$MINESWEEPER_RUN_ROOT/cache/config"
export XDG_DATA_HOME="$MINESWEEPER_RUN_ROOT/cache/data"
export XDG_STATE_HOME="$MINESWEEPER_RUN_ROOT/cache/state"
export HF_HOME="$MINESWEEPER_RUN_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR="$MINESWEEPER_RUN_ROOT/cache/pip"
export UV_CACHE_DIR="$MINESWEEPER_RUN_ROOT/cache/uv"
export TORCH_HOME="$MINESWEEPER_RUN_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$MINESWEEPER_RUN_ROOT/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$MINESWEEPER_RUN_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$MINESWEEPER_RUN_ROOT/cache/triton"
export CUDA_CACHE_PATH="$MINESWEEPER_RUN_ROOT/cache/cuda"
export FLASHINFER_WORKSPACE_BASE="$MINESWEEPER_RUN_ROOT/cache/flashinfer"
export WANDB_DIR="$MINESWEEPER_RUN_ROOT/wandb"
export WANDB_CACHE_DIR="$MINESWEEPER_RUN_ROOT/cache/wandb"
export WANDB_CONFIG_DIR="$MINESWEEPER_RUN_ROOT/cache/wandb_config"
export WANDB_DATA_DIR="$MINESWEEPER_RUN_ROOT/cache/wandb_data"
export PARALLEL_HOME="$MINESWEEPER_RUN_ROOT/cache/parallel"
export TMPDIR="$MINESWEEPER_RUN_ROOT/tmp"
export RAY_TMPDIR="$MINESWEEPER_RUN_ROOT/ray8"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export RAY_USAGE_STATS_ENABLED=0
export WORK="$MINESWEEPER_RUN_ROOT/work"
export HF_MODEL="$MINESWEEPER_RUN_ROOT/models/Qwen3-4B-Instruct-2507"
export MINESWEEPER_DIR="$MINESWEEPER_RUN_ROOT/minesweeper"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-maxrl}"
case "$ADVANTAGE_ESTIMATOR" in
    maxrl|rloo|gracefulrl) ;;
    *) echo "ADVANTAGE_ESTIMATOR must be maxrl, rloo, or gracefulrl" >&2; return 1 ;;
esac
export GRACEFULRL_EPS="${GRACEFULRL_EPS:-0}"
export GRACEFUL_REWARD="${GRACEFUL_REWARD:-0}"
export MINESWEEPER_SMOKE_TEST="${MINESWEEPER_SMOKE_TEST:-0}"
case "$MINESWEEPER_SMOKE_TEST" in
    0|1) ;;
    *) echo "MINESWEEPER_SMOKE_TEST must be 0 or 1" >&2; return 1 ;;
esac
python3 - "$GRACEFUL_REWARD" "$GRACEFULRL_EPS" <<'PY' || return 1
import math
import sys
try:
    grace = float(sys.argv[1])
    gracefulrl_eps = float(sys.argv[2])
except ValueError:
    raise SystemExit('GRACEFUL_REWARD and GRACEFULRL_EPS must be numbers')
if not math.isfinite(grace) or not 0 <= grace < 1:
    raise SystemExit('GRACEFUL_REWARD must satisfy 0 <= value < 1')
if not math.isfinite(gracefulrl_eps) or gracefulrl_eps < 0:
    raise SystemExit('GRACEFULRL_EPS must be nonnegative')
PY
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
export OMP_NUM_THREADS=8
mkdir -p "$TMPDIR" "$WANDB_DIR" "$RAY_TMPDIR" "$WORK" \
    "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" \
    "$ENROOT_TEMP_PATH" "$ENROOT_CONFIG_PATH" "$MINESWEEPER_RUN_ROOT/logs"
