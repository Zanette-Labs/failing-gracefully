# GracefulRL Safe Maze

This directory contains the Safe Maze environment, CNN policy, pretrained policy,
and reinforcement-learning training script used to run GracefulRL experiments.

## Setup

From the repository root, install the dependencies:

```bash
python -m pip install -r safe_maze/requirements.txt
```

## Demo

Run a small CPU-only GracefulRL training demo from the repository root:

```bash
WANDB_MODE=offline python -m safe_maze.rl_gpu \
  --advantage gracefulrl \
  --n_steps 1 \
  --n_envs 2 \
  --n_rollouts 2 \
  --device cpu \
  --no_save_ckpt
```

The script loads `safe_maze/pretrained_policy.pt` by default. Remove
`--no_save_ckpt` to save the resulting policy, or pass `--from_scratch` to skip
loading the pretrained weights.

## Full training runs

Run the following commands from the repository root. Each training step uses 32
maze environments with 64 policy rollouts per environment (2,048 trajectories
per step), and each run trains for 20,000 steps. Outputs, metrics, and the final
`policy.pt` checkpoint are written to the specified `runs/` directory.

Weights & Biases logging is enabled by default. Set `WANDB_MODE=offline` before
a command if you want to keep its logs local.

### Fixed safe reward

These runs compare RLOO, GRPO, and GracefulRL while holding the safe-terminal
reward fixed at `0.5`. They use the standard policy-gradient loss, so the
advantage estimator is the only algorithmic difference.

RLOO:

```bash
python -m safe_maze.rl_gpu \
  --advantage rloo \
  --loss pg \
  --safe_reward_mode constant \
  --safe_reward 0.5 \
  --n_steps 20000 \
  --n_envs 32 \
  --n_rollouts 64 \
  --wandb_project gracefulrl-safe-maze \
  --run_name rloo-fixed-safe-0.5 \
  --run_dir runs/rloo-fixed-safe-0.5
```

GRPO:

```bash
python -m safe_maze.rl_gpu \
  --advantage grpo \
  --loss pg \
  --safe_reward_mode constant \
  --safe_reward 0.5 \
  --n_steps 20000 \
  --n_envs 32 \
  --n_rollouts 64 \
  --wandb_project gracefulrl-safe-maze \
  --run_name grpo-fixed-safe-0.5 \
  --run_dir runs/grpo-fixed-safe-0.5
```

GracefulRL:

```bash
python -m safe_maze.rl_gpu \
  --advantage gracefulrl \
  --loss pg \
  --safe_reward_mode constant \
  --safe_reward 0.5 \
  --n_steps 20000 \
  --n_envs 32 \
  --n_rollouts 64 \
  --wandb_project gracefulrl-safe-maze \
  --run_name gracefulrl-fixed-safe-0.5 \
  --run_dir runs/gracefulrl-fixed-safe-0.5
```

### Delightful Policy Gradient

DPG changes the loss rather than the advantage estimator. This run uses RLOO
advantages, a fixed safe reward of `0.5`, and a DPG gate temperature of `0.9`.

```bash
python -m safe_maze.rl_gpu \
  --advantage rloo \
  --loss dpg \
  --dpg_eta 0.9 \
  --safe_reward_mode constant \
  --safe_reward 0.5 \
  --n_steps 20000 \
  --n_envs 32 \
  --n_rollouts 64 \
  --wandb_project gracefulrl-safe-maze \
  --run_name dpg-rloo-fixed-safe-0.5 \
  --run_dir runs/dpg-rloo-fixed-safe-0.5
```

### Online reward tuning

Online reward tuning sets the current safe reward to the previous batches'
exponential moving average goal rate multiplied by `REWARD_MULTIPLIER`. The EMA
starts at zero; `--online_ema_decay 0.9` retains 90% of its previous value each
step. This example uses RLOO advantages and a multiplier of `1.0`. Set the
multiplier anywhere from `0.0` to `1.0`; for example, `0.5` makes the safe reward
half of the estimated goal rate.

```bash
REWARD_MULTIPLIER=0.5

python -m safe_maze.rl_gpu \
  --advantage rloo \
  --loss pg \
  --safe_reward_mode online_scaled \
  --online_multiplier "$REWARD_MULTIPLIER" \
  --online_ema_decay 0.9 \
  --n_steps 20000 \
  --n_envs 32 \
  --n_rollouts 64 \
  --wandb_project gracefulrl-safe-maze \
  --run_name "rloo-online-scaled-x${REWARD_MULTIPLIER}" \
  --run_dir "runs/rloo-online-scaled-x${REWARD_MULTIPLIER}"
```
