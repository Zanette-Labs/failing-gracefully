"""Evaluate an HF export of a minesweeper policy against an OpenAI-compatible
server (sglang / vLLM) using the SAME episode rules and prompts as training.

Reads a miles prompt jsonl (prepare_data.py output; ``metadata`` carries the
seed, prompt_format and board incl. max_turns) and plays every row through
``env_bridge.MinesweeperGame`` — so an 8-turn board is evaluated as an 8-turn
board, unlike the vLLM harness (play_with_model.py) which is fixed to the env
defaults. Output records / summary follow ``minesweeper/run_batch.py``.

    python3 examples/minesweeper/eval_openai.py \
        --data /path/to/medium_heldout_500.jsonl --puzzles 100 \
        --attempts 4 --model step49 --base-url http://localhost:30000/v1 \
        --output inference_rollouts/step49_medium8_100x4.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.minesweeper.env_bridge import GRACEFUL_OUTCOMES, MinesweeperGame, episode_reward  # noqa: E402


def play_episode(client: OpenAI, args, meta: dict, attempt: int) -> dict:
    game = MinesweeperGame(meta, max_tokens=args.max_tokens)
    messages = [
        {"role": "system", "content": game.system_prompt},
        {"role": "user", "content": game.user_prompt()},
    ]
    turns = []
    started = time.monotonic()
    try:
        for _ in range(game.max_turns):
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.model_seed + attempt,
                extra_body=({"chat_template_kwargs": args.chat_template_kwargs} if args.chat_template_kwargs else None),
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            finish = choice.finish_reason or "stop"
            turns.append({"reply": text, "finish_reason": finish,
                          "completion_tokens": resp.usage.completion_tokens if resp.usage else None})
            messages.append({"role": "assistant", "content": text})
            if game.act(text, finish):
                break
            messages.append({"role": "user", "content": game.user_prompt()})
        error = game.error
    except Exception as exc:  # context overflow / dropped connection: record, don't raise
        msg = f"{type(exc).__name__}: {exc}"
        if "context" in msg.lower() or "maximum" in msg.lower() or "length" in msg.lower():
            game.end_early("context_exhausted")
        else:
            game.end_early("aborted")
        error = msg
    return {
        "puzzle_seed": game.seed,
        "prompt_format": game.prompt_format,
        "max_turns": game.max_turns,
        "attempt": attempt,
        "success": game.success,
        "graceful": game.outcome in GRACEFUL_OUTCOMES,
        "reward": episode_reward(game.outcome, args.graceful_reward),
        "outcome": game.outcome,
        "steps": game.steps,
        "repeats": game.repeats,
        "n_turns": len(turns),
        "actions": game.actions,
        "error": error,
        "seconds": round(time.monotonic() - started, 2),
        "turns": turns,
    }


def summarize(records: list[dict], args) -> dict:
    by_puzzle: dict[int, list[dict]] = {}
    for r in records:
        by_puzzle.setdefault(r["puzzle_seed"], []).append(r)
    solved_counts = {s: sum(r["success"] for r in rs) for s, rs in by_puzzle.items()}
    n_att = args.attempts
    solved_steps = [r["steps"] for r in records if r["success"]]
    comp_tokens = [t["completion_tokens"] for r in records for t in r["turns"]
                   if t.get("completion_tokens") is not None]
    return {
        "model": args.model,
        "data": str(args.data),
        "puzzles": len(by_puzzle),
        "attempts_per_puzzle": n_att,
        "episodes": len(records),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_turns": sorted({r["max_turns"] for r in records}),
        "graceful_reward": args.graceful_reward,
        "attempt_success_rate": round(sum(r["success"] for r in records) / max(len(records), 1), 4),
        "graceful_rate": round(sum(r["graceful"] for r in records) / max(len(records), 1), 4),
        "harmful_rate": round(sum(r["outcome"] in ("mine", "illegal_move") for r in records) / max(len(records), 1), 4),
        "mean_reward": round(statistics.mean(r["reward"] for r in records), 4) if records else None,
        "pass_at_1": round(statistics.mean(c / n_att for c in solved_counts.values()), 4),
        f"pass_at_{n_att}": round(sum(c > 0 for c in solved_counts.values()) / max(len(solved_counts), 1), 4),
        "puzzles_solved_at_least_once": sum(c > 0 for c in solved_counts.values()),
        "puzzles_solved_every_attempt": sum(c == n_att for c in solved_counts.values()),
        "puzzles_never_solved": sum(c == 0 for c in solved_counts.values()),
        "outcomes": dict(Counter(r["outcome"] for r in records).most_common()),
        "solved_count_histogram": dict(sorted(Counter(solved_counts.values()).items())),
        "mean_steps_when_solved": round(statistics.mean(solved_steps), 2) if solved_steps else None,
        "mean_turns_per_episode": round(statistics.mean(r["n_turns"] for r in records), 2),
        "mean_completion_tokens_per_turn": round(statistics.mean(comp_tokens), 1) if comp_tokens else None,
        "mean_seconds_per_episode": round(statistics.mean(r["seconds"] for r in records), 2),
        "per_puzzle_solved": {str(s): c for s, c in sorted(solved_counts.items())},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="miles prompt jsonl")
    p.add_argument("--puzzles", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--attempts", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--base-url", default="http://localhost:30000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", required=True)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--graceful-reward", type=float, default=0.0,
                   help="reward c for a turn_limit episode with no harmful move (reported as mean_reward)")
    p.add_argument("--chat-template-kwargs", type=json.loads, default=None,
                   help='JSON passed as chat_template_kwargs, e.g. \'{"enable_thinking": false}\'')
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, default=None)
    args = p.parse_args()
    summary_path = args.summary or args.output.with_name(args.output.stem + "_summary.json")

    with open(args.data) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    metas = [r["metadata"] for r in rows[args.start:args.start + args.puzzles]]
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=3, timeout=1800)

    jobs = [(m, a) for m in metas for a in range(args.attempts)]
    records: list[dict] = []
    lock = threading.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with open(args.output, "w") as out, ThreadPoolExecutor(args.concurrency) as pool:
        futs = [pool.submit(play_episode, client, args, m, a) for m, a in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                records.append(rec)
                out.write(json.dumps(rec) + "\n")
                out.flush()
            if i % 10 == 0 or i == len(jobs):
                solved = sum(r["success"] for r in records)
                print(f"[{i}/{len(jobs)}] solved {solved} ({solved / i:.3f})  "
                      f"{time.monotonic() - t0:.0f}s", flush=True)

    summary = summarize(records, args)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_puzzle_solved"}, indent=2))


if __name__ == "__main__":
    main()
