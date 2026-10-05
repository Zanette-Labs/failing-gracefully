#!/bin/bash
set -euo pipefail

# Usage: ./run_idk_train_4B_orchard.sh <normal|brier|idk|reward_tuning> [--lr LR] [--advantage ADVANTAGE]

CMD=${1:-normal}
shift 1 2>/dev/null || true
LR_OVERRIDE=""
ADVANTAGE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lr) LR_OVERRIDE="$2"; shift 2 ;;
    --advantage) ADVANTAGE_OVERRIDE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ============ Configuration ============
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_DIR=${DATA_DIR:-$SCRIPT_DIR}
MODEL_PATH=${MODEL_PATH:-/tmp/Qwen3-4B-Base}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/tmp/qwen_4b_checkpoints}
SAVE_FREQ=${SAVE_FREQ:-200}
export PYTHONPATH="${SCRIPT_DIR%/*}${PYTHONPATH:+:$PYTHONPATH}"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$CMD" = "normal" ]; then
  TRAIN_DATA=$DATA_DIR/data/dapo/normal/train.parquet
  VAL_DATA="['$DATA_DIR/data/aime_combined/normal/test.parquet']"
  REWARD_MANAGER=multi_thread
  VALIDATION_DATA_DIR=$DATA_DIR/validation_rollouts_normal
  ADVANTAGE_ESTIMATOR=grpo
  EXPERIMENT_SUFFIX=normal
  LR=1e-6
elif [ "$CMD" = "idk" ]; then
  # To train IDK on one reward level, use a single-element array, e.g. IDK_C_VALUES=(0.4).
  IDK_C_VALUES=(0.0 0.2 0.4 0.6 0.8)
  _train_list=""
  _val_list=""
  for c in "${IDK_C_VALUES[@]}"; do
    _train_list+="'$DATA_DIR/data/dapo/idk/train_c_${c}.parquet',"
    for ds in aime_combined; do
      _val_list+="'$DATA_DIR/data/${ds}/idk/test_c_${c}.parquet',"
    done
  done
  TRAIN_DATA="[${_train_list%,}]"
  VAL_DATA="[${_val_list%,}]"
  REWARD_MANAGER=multi_thread_idk
  VALIDATION_DATA_DIR=$CHECKPOINT_DIR/qwen3_4b/validation_rollouts_idk
  # GracefulRL eps is hardcoded in verl/trainer/ppo/core_algos.py.
  ADVANTAGE_ESTIMATOR=gracefulrl
  EXPERIMENT_SUFFIX=idk_l02468_eps1
  LR=1e-6
elif [ "$CMD" = "brier" ]; then
  TRAIN_DATA="['$DATA_DIR/data/dapo/brier/train.parquet']"
  VAL_DATA="['$DATA_DIR/data/aime_combined/brier/test.parquet']"
  REWARD_MANAGER=multi_thread_brier
  VALIDATION_DATA_DIR=$CHECKPOINT_DIR/qwen3_4b/validation_rollouts_brier
  ADVANTAGE_ESTIMATOR=rloo
  EXPERIMENT_SUFFIX=brier
  LR=1e-6
elif [ "$CMD" = "reward_tuning" ]; then
  IDK_C_VALUES=(0.0 0.2 0.4 0.6 0.8)
  _train_list=""
  _val_list=""
  for c in "${IDK_C_VALUES[@]}"; do
    _train_list+="'$DATA_DIR/data/dapo/idk/train_c_${c}.parquet',"
    for ds in aime_combined; do
      _val_list+="'$DATA_DIR/data/${ds}/idk/test_c_${c}.parquet',"
    done
  done
  TRAIN_DATA="[${_train_list%,}]"
  VAL_DATA="[${_val_list%,}]"
  REWARD_MANAGER=multi_thread_reward_tuning
  VALIDATION_DATA_DIR=$CHECKPOINT_DIR/qwen3_4b/validation_rollouts_reward_tuning
  ADVANTAGE_ESTIMATOR=rloo
  EXPERIMENT_SUFFIX=reward_tuning_l02468
  LR=1e-6
else
  echo "Usage: $0 <normal|idk|brier|reward_tuning> [--lr LR] [--advantage ADVANTAGE]"
  exit 1
fi

[ -n "$LR_OVERRIDE" ] && LR="$LR_OVERRIDE"
[ -n "$ADVANTAGE_OVERRIDE" ] && ADVANTAGE_ESTIMATOR="$ADVANTAGE_OVERRIDE"

echo "Running with: experiment=${CMD}, lr=${LR}, advantage=${ADVANTAGE_ESTIMATOR}"
echo "TRAIN_DATA=${TRAIN_DATA}"
echo "VAL_DATA=${VAL_DATA}"

TRUNCATE_ORDER=64
N_ROLLOUTS=32
N_VAL=32

PROJECT_NAME=Reliability-4B
EXPERIMENT_NAME=${EXPERIMENT_SUFFIX}_${LR}_${ADVANTAGE_ESTIMATOR}

# ============ Ray Setup ============
ray stop --force 2>/dev/null || true
ray start --head --num-gpus ${N_GPUS}

# ============ Training ============
python3 -m verl.trainer.main_ppo \
  ray_init.ray_dir=/tmp/ray \
  algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR} \
  algorithm.use_kl_in_reward=False \
  algorithm.pass_k=${TRUNCATE_ORDER} \
  algorithm.truncate_order=${TRUNCATE_ORDER} \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${VAL_DATA} \
  data.train_batch_size=32 \
  data.filter_overlong_prompts=True \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_model_len=32000 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32000 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=${N_VAL} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  algorithm.kl_ctrl.kl_coef=0.0 \
  reward_model.reward_manager=${REWARD_MANAGER} \
  +reward_model.reward_kwargs.num_reward_actors=16 \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.max_actor_ckpt_to_keep=5 \
  trainer.validation_data_dir=${VALIDATION_DATA_DIR} \
  trainer.test_freq=50 \
  trainer.default_local_dir=${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
  trainer.total_epochs=10 \
  trainer.total_training_steps=1000
