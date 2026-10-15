# Minesweeper integration for Miles

This package connects the bundled no-guess Minesweeper environment to Miles:

- `env_bridge.py` wraps one game and defines terminal outcomes and grace reward.
- `agent.py` implements the multi-turn rollout and reward function.
- `prepare_data.py` converts committed seed splits into Miles prompt JSONL.
- `log_utils.py` reports per-outcome and per-split training metrics.
- `eval_openai.py` evaluates an HF export through an OpenAI-compatible server.

The training recipe uses Python-list boards with zero-indexed actions. Revealing
an already visible cell wastes a turn but does not immediately lose. A complete
solve receives reward 1; a safe turn-limit receives `--graceful-reward`; every
harmful, malformed, truncated, or infrastructure terminal outcome receives 0.

See [`PROMPTS.md`](PROMPTS.md) for the exact model messages and the release-root
README for launch instructions.
