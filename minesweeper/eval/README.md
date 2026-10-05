# Twelve-turn transfer evaluation

`medium12_heldout_500_seeds.txt` is the exact ordered seed set used for the
reported 12-turn transfer evaluation. It contains the last 500 seeds from the
12-turn `heuristic_not_naive_10k` medium split, shuffled with Python
`random.Random(72)`. Its SHA-256 is:

```text
e9cc9500070e9672646f9e12434961bccbb7847a8a192fbf8635ef0e72e423ea
```

These are 12-turn boards; training and in-distribution validation use distinct
eight-turn boards. Changing the turn budget changes board generation, so the
two sets must not be treated as the same environment split.

`run_eval.py` launches one tensor-parallel SGLang server, evaluates the selected
seeds once each, saves full trajectories, replays every trajectory through the
environment as an audit, and writes summary and run-manifest JSON files. The
default parameters reproduce the final evaluation protocol:

- 500 puzzles, one attempt each;
- 12 turns per puzzle;
- temperature 1.0 and top-p 1.0;
- sampling seed 72;
- 6,144 generated tokens per turn;
- 81,920 total context tokens;
- two GPUs for tensor-parallel inference.

Run it inside the prepared Enroot container after exporting a training
checkpoint to Hugging Face format:

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

Stop training before evaluation so the requested GPUs are free. An interrupted
evaluation resumes from the completed puzzle indices in the output JSONL. Use
`--overwrite` to deliberately start that output path again from scratch.

For a quick end-to-end smoke test, add `--puzzles 10`. To use an already
running native SGLang server, pass `--server-url http://HOST:PORT`; `--model`
is still required locally for its tokenizer.

Released checkpoints can be downloaded from the
[Graceful Failure: Minesweeper collection](https://huggingface.co/collections/daman1209arora/graceful-failure-minesweeper).
The release-root README lists the four model IDs, their grace values, their
exact source update counts, and a complete download-plus-evaluation example.
