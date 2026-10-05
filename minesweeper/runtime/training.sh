#!/usr/bin/env bash
# Qwen3-4B: 8-turn medium Minesweeper, 9,500 train + 500 validation puzzles.
# GPUs 0-3 train (TP=2, DP=2); GPUs 4-7 generate (two TP=2 engines).
# All runtime files use /tmp. Train for 250 steps, then save the final checkpoint.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"

SEED="${1:-72}"
if [[ "$MINESWEEPER_SMOKE_TEST" == 1 ]]; then
    NUM_ROLLOUT=1
    ROLLOUT_BATCH_SIZE=2
    N_SAMPLES_PER_PROMPT=2
    GLOBAL_BATCH_SIZE=4
    TRAIN_COUNT=8
    VAL_COUNT=2
    GAME_MAX_TURNS=2
    ROLLOUT_MAX_RESPONSE_LEN=4096
    ROLLOUT_MAX_CONTEXT_LEN=16384
    SAVE_INTERVAL=1
    EVAL_INTERVAL=1
    RUN_KIND=smoke
else
    NUM_ROLLOUT=250
    ROLLOUT_BATCH_SIZE=32
    N_SAMPLES_PER_PROMPT=16
    GLOBAL_BATCH_SIZE=512
    TRAIN_COUNT=9500
    VAL_COUNT=500
    GAME_MAX_TURNS=8
    ROLLOUT_MAX_RESPONSE_LEN=4096
    ROLLOUT_MAX_CONTEXT_LEN=32000
    SAVE_INTERVAL=250
    EVAL_INTERVAL=25
    RUN_KIND=full
fi
JOB_ID="minesweeper-qwen3-4b-${ADVANTAGE_ESTIMATOR}-eps${GRACEFULRL_EPS}-grace${GRACEFUL_REWARD}-medium-turns${GAME_MAX_TURNS}-samples${N_SAMPLES_PER_PROMPT}-8gpu-${RUN_KIND}-seed${SEED}-$(date +%Y%m%d-%H%M%S)"
REPO="$MINESWEEPER_RUN_ROOT/repo"
MEGATRON=/root/Megatron-LM
TORCH_DIST="$WORK/miles_models/Qwen3-4B-Instruct-2507_torch_dist"
CHECKPOINT_DIR="$WORK/checkpoints/minesweeper/$JOB_ID"
DATA="$WORK/data_for_minesweeper_miles/medium_turns${GAME_MAX_TURNS}_${RUN_KIND}_seed${SEED}"
TRAIN_DATA="$DATA/train.jsonl"
VAL_DATA="$DATA/val.jsonl"
SEED_FILE="$MINESWEEPER_DIR/game_analysis/turns8/heuristic_not_naive_10k_seeds.txt"
cd "$REPO"
export PYTHONPATH="$REPO:$MEGATRON"

for path in "$HF_MODEL/config.json" "$MINESWEEPER_DIR/minesweeper_env.py" "$SEED_FILE"; do
    [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
source "$REPO/scripts/models/qwen3-4B.sh"

# Convert pretrained input weights once; these are not training checkpoints.
if [[ ! -f "$TORCH_DIST/latest_checkpointed_iteration.txt" ]]; then
    torchrun --standalone --nproc-per-node 1 "$REPO/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" --hf-checkpoint "$HF_MODEL" \
        --rotary-base 5000000 --save "$TORCH_DIST"
fi

# One train set and one disjoint validation set. List format.
PREP="$REPO/examples/minesweeper/prepare_data.py"
python3 "$PREP" --seed-file "$SEED_FILE" --split medium \
    --start 0 --count "$TRAIN_COUNT" --format list --max-turns "$GAME_MAX_TURNS" \
    --output "$TRAIN_DATA" --shuffle --seed "$SEED"
python3 "$PREP" --seed-file "$SEED_FILE" --split medium_heldout \
    --start "$TRAIN_COUNT" --count "$VAL_COUNT" --format list --max-turns "$GAME_MAX_TURNS" \
    --output "$VAL_DATA"

# Start this run's local Ray head; never stop other jobs on the node.
export RAY_ADDRESS=127.0.0.1:16389
if ! timeout 10 ray status --address "$RAY_ADDRESS" >/dev/null 2>&1; then
    ray start --head --node-ip-address 127.0.0.1 --port=16389 --num-gpus 8 \
        --num-cpus=64 --object-store-memory=16000000000 \
        --min-worker-port=32002 --max-worker-port=33999 --ray-client-server-port=16390 \
        --dashboard-agent-listen-port=18276 --dashboard-agent-grpc-port=18277 \
        --runtime-env-agent-port=18278 --temp-dir "$RAY_TMPDIR" \
        --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=18275
fi

VENV_SITE=$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"$REPO:$MEGATRON:$VENV_SITE\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"MINESWEEPER_DIR\": \"$MINESWEEPER_DIR\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"FLASHINFER_DISABLE_VERSION_CHECK\": \"1\"
  }
}"

TRAIN_ARGS=(
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_MODEL"
    --load "$TORCH_DIST"
    --start-rollout-id 0
    --save "$CHECKPOINT_DIR"
    --save-interval "$SAVE_INTERVAL"
    --rotary-base 5000000

    # On-policy: 32 prompts x 16 samples = 512 fresh episodes, one optimizer step.
    --prompt-data "$TRAIN_DATA"
    --input-key prompt
    --metadata-key metadata
    --rollout-shuffle
    --num-rollout "$NUM_ROLLOUT"
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --num-steps-per-rollout 1
    --balance-data
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
    --rollout-max-context-len "$ROLLOUT_MAX_CONTEXT_LEN"
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --custom-generate-function-path examples.minesweeper.agent.generate
    --generate-max-turns 0
    --custom-rm-path examples.minesweeper.agent.reward_func
    --reward-key score
    --graceful-reward "$GRACEFUL_REWARD"

    # Validate on the same held-out 500 puzzles every 25 steps.
    --eval-prompt-data val "$VAL_DATA"
    --eval-interval "$EVAL_INTERVAL"
    --skip-eval-before-train
    --eval-input-key prompt
    --n-samples-per-eval-prompt 1
    --eval-temperature 1.0
    --eval-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"

    # Advantage estimator and AdamW.
    --advantage-estimator "$ADVANTAGE_ESTIMATOR"
    --gracefulrl-eps "$GRACEFULRL_EPS"
    --calculate-per-token-loss
    --eps-clip 0.2
    --eps-clip-high 0.28
    --entropy-coef 0.0
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    # Four training GPUs, with CPU optimizer offload for long-context headroom.
    --actor-num-nodes 1
    --actor-num-gpus-per-node 4
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 24576
    --optimizer-cpu-offload
    --optimizer-offload-fraction 1.0
    --use-precision-aware-optimizer
    --overlap-cpu-optimizer-d2h-h2d

    # Four rollout GPUs: two engines with safe inference memory limits.
    --rollout-num-gpus 4
    --rollout-num-gpus-per-engine 2
    --num-gpus-per-node 8
    --sglang-mem-fraction-static 0.7
    --sglang-max-running-requests 64
    --sglang-chunked-prefill-size 4096

    --seed "$SEED"
    --rollout-seed "$SEED"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --log-passrate
    --custom-rollout-log-function-path examples.minesweeper.log_utils.log_rollout_data
    --use-wandb
    --wandb-project minesweeper_online_rl
    --wandb-group "${ADVANTAGE_ESTIMATOR}_eps${GRACEFULRL_EPS}_disagg_list_medium_turns${GAME_MAX_TURNS}_samples${N_SAMPLES_PER_PROMPT}_grace${GRACEFUL_REWARD}_${RUN_KIND}_seed${SEED}"
    --disable-wandb-random-suffix
)

printf '%s\n' "$JOB_ID" > "$MINESWEEPER_RUN_ROOT/active_job_id"
ray job submit --address="http://127.0.0.1:18275" --submission-id "$JOB_ID" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python3 "$REPO/train.py" "${TRAIN_ARGS[@]}"
