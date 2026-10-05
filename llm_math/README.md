# LLM math experiments

This directory contains the VERL-based code used to train Qwen3-1.7B and
Qwen3-4B on mathematical reasoning with three reliability recipes:

- **Brier** trains calibrated answer confidence with a Brier-score reward and
  RLOO advantages.
- **IDK** teaches the model to emit an explicit “I don't know” response and uses
  the GracefulRL advantage estimator (named `tailrl` internally in this code).
- **Reward tuning** adjusts the reward for abstention online and uses RLOO
  advantages.

Run all commands below from this `llm_math/` directory. The launchers expect a
working CUDA/Slurm environment with the dependencies required by the included
VERL tree. Log in to Weights & Biases before training if you want online
experiment tracking.

## Installation

Install on the same GPU machine you will use for training, ideally one that can
build and run FlashAttention. The environment used for these experiments was
Python 3.10 with CUDA 12.4 and PyTorch 2.6:

```bash
conda create -n gracefulrl python==3.10
conda activate gracefulrl

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Install the FlashAttention build dependencies, then build FlashAttention from
source. Adjust `MAX_JOBS` to suit the machine's available CPU cores and memory:

```bash
pip install ninja
pip install packaging
pip install psutil

git clone https://github.com/Dao-AILab/flash-attention.git
pushd flash-attention
export MAX_JOBS=4
python setup.py install
popd
```

Install the rollout backend and remaining experiment dependencies:

```bash
pip install vllm==0.8.4
pip install wandb
pip install math-verify
```

Finally, from this `llm_math/` directory, install the included VERL fork in
editable mode:

```bash
pip install -e .
```

Package versions can conflict as upstream libraries evolve. If possible, keep
the versions above fixed and build the environment directly on the target GPU
machine.

## Prepare the datasets

The training recipes use DAPO for training and the combined AIME 2024–2026 set
for validation. The preprocessing script downloads the source datasets from
Hugging Face and writes `normal`, `brier`, and `idk` parquet variants. Generate
the five IDK reward levels consumed by the launchers with:

```bash
PYTHONPATH="$PWD" python examples/data_preprocess/math_dataset_combined.py \
  --dataset dapo \
  --local_dir scripts/data/dapo \
  --idk_c 0.0 0.2 0.4 0.6 0.8

PYTHONPATH="$PWD" python examples/data_preprocess/math_dataset_combined.py \
  --dataset aime_combined \
  --local_dir scripts/data/aime_combined \
  --idk_c 0.0 0.2 0.4 0.6 0.8
```

The launchers then read these paths:

```text
scripts/data/dapo/brier/train.parquet
scripts/data/dapo/idk/train_c_<level>.parquet
scripts/data/aime_combined/brier/test.parquet
scripts/data/aime_combined/idk/test_c_<level>.parquet
```

Set `DATA_DIR=/another/path` when launching training if the generated files live
under `/another/path/data/` instead of `scripts/data/`.

## Train Qwen3-1.7B

The 1.7B launcher runs directly on one node and uses every GPU visible to the
current shell. Activate the Python environment containing the VERL dependencies,
set the base-model and checkpoint locations, then start one of the three recipes:

```bash
export MODEL_PATH=/path/to/Qwen3-1.7B-Base
export CHECKPOINT_DIR=/path/to/checkpoints

./scripts/run_idk_train_1.7B_orchard.sh brier
./scripts/run_idk_train_1.7B_orchard.sh idk
./scripts/run_idk_train_1.7B_orchard.sh reward_tuning
```

The Brier and reward-tuning recipes use all required DAPO/AIME variants. The
1.7B IDK recipe trains jointly on reward levels `0.0`, `0.2`, `0.4`, `0.6`, and
`0.8`. Optional overrides are available as `--lr <value>` and
`--advantage <name>`.

## Train Qwen3-4B

The 4B launcher follows the same direct, single-node flow and uses every visible
GPU. Activate the desired Python environment and set the model and checkpoint
locations:

```bash
export MODEL_PATH=/path/to/Qwen3-4B-Base
export CHECKPOINT_DIR=/path/to/checkpoints

./scripts/run_idk_train_4B_orchard.sh brier
./scripts/run_idk_train_4B_orchard.sh idk
./scripts/run_idk_train_4B_orchard.sh reward_tuning
```

The 4B IDK recipe uses reward level `0.0`; reward tuning uses levels `0.0`,
`0.2`, `0.4`, `0.6`, and `0.8`. The launcher accepts `--lr <value>` and
`--advantage <name>` overrides.

Set `CUDA_VISIBLE_DEVICES` before either launcher if only a subset of the node's
GPUs should be used. Run only one launcher at a time on a node because each
script creates its own local Ray runtime.
