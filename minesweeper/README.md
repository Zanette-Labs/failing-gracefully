# Failing Gracefully: Minesweeper RL

This directory is a self-contained release of the eight-turn Minesweeper
reinforcement-learning experiment. It trains `Qwen3-4B-Instruct-2507` while
varying two experiment parameters:

- `GRACEFUL_REWARD`: reward for using all eight turns without solving and
  without making a harmful or invalid move.
- `ADVANTAGE_ESTIMATOR`: `maxrl`, `rloo`, or `gracefulrl`.

A solved board receives reward `1`. A safe turn-limit receives the configured
grace reward `c`, where `0 <= c < 1`. Mine reveals, malformed actions,
out-of-bounds actions, response truncation, and context exhaustion receive
reward `0`.

## Requirements

The supplied launcher was tested on one x86-64 Linux node with:

- 8 H100 GPUs;
- working NVIDIA drivers and Enroot GPU integration;
- approximately 250 GB of free node-local `/tmp` space;
- outbound access for the container and model downloads.

The launcher imports `docker://radixark/miles:test-20260601`, pins the SGLang
and Megatron-LM revisions checked by `runtime/check_runtime.py`, and downloads
`Qwen/Qwen3-4B-Instruct-2507` at a pinned Hugging Face revision. No Python
environment needs to be installed on the host.

### Enroot installation versus image setup

The release automates the **image setup**, but it does not install Enroot or
the NVIDIA container integration on the host. Those are cluster/system
prerequisites. Verify them first with:

```bash
enroot version
nvidia-smi
```

On the first `bash run.sh`, `runtime/setup.sh` places every Enroot cache,
runtime, and writable-container file under `MINESWEEPER_RUN_ROOT`, imports the
pinned Docker image, creates the `miles-minesweeper` container, checks the
pinned SGLang and Megatron revisions, performs tensor operations on all eight
GPUs, and downloads the base model. Later launches reuse those files.

To perform setup without starting training:

```bash
export MINESWEEPER_RUN_ROOT="/tmp/$(id -un)-minesweeper"
bash runtime/setup.sh
```

## Quick start

From this directory:

```bash
export MINESWEEPER_RUN_ROOT="/tmp/$(id -un)-minesweeper-gracefulrl-grace050"
ADVANTAGE_ESTIMATOR=gracefulrl \
GRACEFUL_REWARD=0.5 \
GRACEFULRL_EPS=0 \
bash run.sh 72
```

The final argument is the training and rollout seed; it defaults to `72`.
Setup can take a while on a new node because it imports the container and
downloads the model. Training is started in a detached process after setup.

Other examples:

```bash
# Binary reward with MaxRL.
MINESWEEPER_RUN_ROOT="/tmp/$(id -un)-ms-maxrl-g0" \
ADVANTAGE_ESTIMATOR=maxrl GRACEFUL_REWARD=0 bash run.sh 72

# Grace reward 0.25 with leave-one-out REINFORCE.
MINESWEEPER_RUN_ROOT="/tmp/$(id -un)-ms-rloo-g025" \
ADVANTAGE_ESTIMATOR=rloo GRACEFUL_REWARD=0.25 bash run.sh 72
```

To check the complete setup, rollout, reward, optimizer, evaluation, and
checkpoint path with one small update, use:

```bash
MINESWEEPER_RUN_ROOT="/tmp/$(id -un)-minesweeper-smoke" \
MINESWEEPER_SMOKE_TEST=1 WANDB_MODE=offline bash run.sh 72
```

The smoke mode uses the same eight-GPU model/runtime path as the full run, but
only eight training puzzles, two held-out puzzles, two samples per prompt,
two-turn episodes, and one optimizer update.

Use a distinct `MINESWEEPER_RUN_ROOT` for every configuration. A run consumes
all eight GPUs, and the launcher permits only one active job per run root.

### Configuration variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `GRACEFUL_REWARD` | `0` | Reward `c` for a safe turn-limit; must satisfy `0 <= c < 1`. |
| `ADVANTAGE_ESTIMATOR` | `maxrl` | `maxrl`, `rloo`, or `gracefulrl`. |
| `GRACEFULRL_EPS` | `0` | GracefulRL rank-smoothing epsilon; used only by GracefulRL and must be nonnegative. |
| `MINESWEEPER_RUN_ROOT` | `/tmp/$USER-minesweeper` | Runtime, model, data, log, and checkpoint root. |
| `WANDB_MODE` | automatic | Defaults to `offline` when no W&B credentials are found. |
| `MINESWEEPER_SMOKE_TEST` | `0` | Set to `1` for the one-update end-to-end smoke test. |

The exact model, optimizer, rollout, and GPU settings are in
[`runtime/training.sh`](runtime/training.sh). The fixed experiment uses:

- 6x6 boards with 6 mines and an eight-turn budget;
- 9,500 medium training seeds and 500 disjoint validation seeds;
- 32 prompts x 16 samples = 512 fresh episodes per optimizer update;
- one update per rollout batch, for 250 updates;
- four actor GPUs and four rollout GPUs;
- validation every 25 updates and one checkpoint after update 250;
- AdamW at a constant learning rate of `1e-6`.

## Monitoring and stopping

```bash
tail -f "$MINESWEEPER_RUN_ROOT/logs/training.log"
python3 "$MINESWEEPER_RUN_ROOT/control.py" status
python3 "$MINESWEEPER_RUN_ROOT/control.py" stop
```

Checkpoints are written under:

```text
$MINESWEEPER_RUN_ROOT/work/checkpoints/minesweeper/<run-id>/
```

Generated prompt data, the pretrained model, W&B files, and all caches also
remain below `MINESWEEPER_RUN_ROOT`. A new run starts from pretrained weights;
automatic checkpoint resume is intentionally not enabled.

## What the launcher does

`run.sh` delegates to the scripts in `runtime/`:

1. `setup.sh` stages this release into the run root, imports and validates the
   pinned Enroot image, and downloads the pinned base model.
2. `launch.py` starts a detached container process.
3. `training.sh` converts the HF model to Megatron `torch_dist`, creates the
   train/validation JSONL files from the committed seed split, starts Ray, and
   submits the training job.
4. `examples/minesweeper/agent.py` runs a complete multi-turn game for each
   rollout, and `env_bridge.py` computes its terminal outcome and reward.

The exact model-facing messages are recorded in
[`examples/minesweeper/PROMPTS.md`](examples/minesweeper/PROMPTS.md).

## Repository layout

```text
run.sh                         one-command entry point
runtime/                       Enroot setup, launch, training, and control
examples/minesweeper/          Miles rollout, reward, data, and logging adapter
minesweeper/                   environment, split builders, tests, and seed files
eval/                          fixed 12-turn evaluation seeds and evaluator
miles/ and miles_plugins/      required Miles training runtime
scripts/models/qwen3-4B.sh     model architecture arguments
tools/                         HF/Megatron checkpoint converters
train.py                       Miles synchronous training driver
```

## Exporting a checkpoint for inference

Training checkpoints are Megatron distributed checkpoints. To serve one with
SGLang, run the included converter inside the prepared container. Replace the
two paths with the desired `iter_*` directory and output directory:

```bash
bash "$MINESWEEPER_RUN_ROOT/container.sh" python3 \
  "$MINESWEEPER_RUN_ROOT/repo/tools/convert_torch_dist_to_hf.py" \
  --input-dir "$MINESWEEPER_RUN_ROOT/work/checkpoints/minesweeper/RUN/iter_0000249" \
  --output-dir "$MINESWEEPER_RUN_ROOT/work/hf_models/RUN" \
  --origin-hf-dir "$MINESWEEPER_RUN_ROOT/models/Qwen3-4B-Instruct-2507" \
  --vocab-size 151936
```

## 🤗 Released checkpoints

The four inference-ready checkpoints are collected at
[Graceful Failure: Minesweeper](https://huggingface.co/collections/daman1209arora/graceful-failure-minesweeper).
Repository names intentionally omit the checkpoint step:

| Grace reward | Training updates | Hugging Face model |
| ---: | ---: | --- |
| `0` | 249 | [`daman1209arora/Qwen3-4B-Minesweeper-GracefulRL-Grace0`](https://huggingface.co/daman1209arora/Qwen3-4B-Minesweeper-GracefulRL-Grace0) |
| `0.25` | 240 | [`daman1209arora/Qwen3-4B-Minesweeper-GracefulRL-Grace0.25`](https://huggingface.co/daman1209arora/Qwen3-4B-Minesweeper-GracefulRL-Grace0.25) |
| `0.5` | 243 | [`daman1209arora/Qwen3-4B-Minesweeper-GracefulRL-Grace0.5`](https://huggingface.co/daman1209arora/Qwen3-4B-Minesweeper-GracefulRL-Grace0.5) |
| `0.75` | 240 | [`daman1209arora/Qwen3-4B-Minesweeper-GracefulRL-Grace0.75`](https://huggingface.co/daman1209arora/Qwen3-4B-Minesweeper-GracefulRL-Grace0.75) |

The update counts record the exact source checkpoints and are not part of the
public repository names. New training runs made with this release use 250
updates and checkpoint once after the final update.

### Download and evaluate a released checkpoint

First prepare the runtime as described above, then download a model inside the
container. This example selects the grace-0.5 checkpoint:

```bash
export MINESWEEPER_RUN_ROOT="/tmp/$(id -un)-minesweeper-eval"
bash runtime/setup.sh

export MODEL_REPO="daman1209arora/Qwen3-4B-Minesweeper-GracefulRL-Grace0.5"
export MODEL_DIR="$MINESWEEPER_RUN_ROOT/work/hf_models/gracefulrl-grace0.5"
bash "$MINESWEEPER_RUN_ROOT/container.sh" python3 -c \
  'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])' \
  "$MODEL_REPO" "$MODEL_DIR"
```

Run a 10-puzzle smoke evaluation on two GPUs:

```bash
bash "$MINESWEEPER_RUN_ROOT/container.sh" python3 \
  "$MINESWEEPER_RUN_ROOT/repo/eval/run_eval.py" \
  --model "$MODEL_DIR" \
  --label gracefulrl-grace0.5 \
  --grace 0.5 \
  --training-updates 243 \
  --puzzles 10 \
  --gpus 0,1 \
  --output "$MINESWEEPER_RUN_ROOT/work/evaluations/gracefulrl-grace0.5-smoke.jsonl"
```

As a release sanity check, all four uploaded checkpoints were evaluated on the
first 10 fixed held-out puzzles with the command above (sampling seed `72`).
These tiny-sample results verify that the checkpoints load and complete the
full evaluation path; they are not statistically meaningful benchmark scores:

| Grace reward | Success | Safe turn limit | Terminal error | Mean training reward |
| ---: | ---: | ---: | ---: | ---: |
| `0` | 6/10 | 2/10 | 2/10 | `0.600` |
| `0.25` | 5/10 | 1/10 | 4/10 | `0.525` |
| `0.5` | 8/10 | 1/10 | 1/10 | `0.850` |
| `0.75` | 8/10 | 1/10 | 1/10 | `0.875` |

Each run passed the evaluator's trajectory replay audit. Use the complete
500-puzzle evaluation below for reportable results.

For the fixed 500-puzzle evaluation, omit `--puzzles 10`. Use a different
output path for each checkpoint. The evaluator resumes an interrupted JSONL;
pass `--overwrite` to intentionally replace it. It writes full trajectories,
an audited `*.summary.json`, a `*.manifest.json`, and an SGLang server log.

Run the fixed 500-puzzle transfer evaluation with:

```bash
bash "$MINESWEEPER_RUN_ROOT/container.sh" python3 \
  "$MINESWEEPER_RUN_ROOT/repo/eval/run_eval.py" \
  --model "$MINESWEEPER_RUN_ROOT/work/hf_models/RUN" \
  --label gracefulrl-grace0.5 \
  --grace 0.5 \
  --training-updates 250 \
  --gpus 0,1 \
  --output "$MINESWEEPER_RUN_ROOT/work/evaluations/gracefulrl-grace0.5.jsonl"
```

The evaluator uses the exact ordered seeds and inference protocol from the
12-turn comparison, records full trajectories, audits them by replaying every
move, and writes a summary. See [`eval/README.md`](eval/README.md) for details.
`examples/minesweeper/eval_openai.py` remains available for ad hoc evaluation
against an existing OpenAI-compatible server.

## License

The bundled Miles-derived code retains its Apache-2.0 license in `LICENSE`.
