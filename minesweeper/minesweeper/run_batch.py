"""Run many puzzles with several attempts each and report aggregate results.

Start the server with ``./serve_qwen.sh`` first, then for example::

    python run_batch.py --puzzles 100 --attempts 8
    python run_batch.py --puzzles 100 --attempts 8 --concurrency 32 \
        --output runs/qwen.jsonl --summary runs/qwen_summary.json

Every (puzzle, attempt) pair is one independent episode played by a fresh chat
session. Attempts of the same puzzle differ only in the sampling seed sent to
the server, so a puzzle solved 3 times out of 8 reports as such rather than
collapsing into one deterministic answer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import statistics
import sys
import threading
import time

from openai import OpenAI

from play_puzzle import DEFAULT_PROMPT_FORMAT, PROMPT_FORMATS
from play_with_model import (
    DEFAULT_SEED_FILE,
    Display,
    ModelPlayer,
    load_seeds,
    play_episode,
)


def select_seeds(
    seed_file: Path, puzzles: int, seed_index: int, sample_seed: int | None
) -> list[int]:
    """Take the puzzles to play, either consecutively or as a random sample."""
    seeds = load_seeds(seed_file)
    if sample_seed is not None:
        import random

        return random.Random(sample_seed).sample(seeds, min(puzzles, len(seeds)))
    return seeds[seed_index:seed_index + puzzles]


def run_one(
    client: OpenAI,
    args: argparse.Namespace,
    puzzle_seed: int,
    attempt: int,
) -> dict:
    """Play a single attempt; a server failure is recorded, not raised."""
    player = ModelPlayer(
        client,
        args.model,
        args.temperature,
        args.top_p,
        args.max_tokens,
        # Distinct per attempt so the eight tries are eight samples, not one.
        args.model_seed + attempt,
    )
    started = time.monotonic()
    try:
        record = play_episode(player, puzzle_seed, Display(), args.prompt_format)
    except Exception as exc:  # a dropped connection should not lose the batch
        record = {
            "puzzle_seed": puzzle_seed,
            "prompt_format": args.prompt_format,
            "success": False,
            "outcome": "error",
            "steps": 0,
            "turns": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    record["attempt"] = attempt
    record["seconds"] = round(time.monotonic() - started, 2)
    return record


def summarize(records: list[dict], puzzle_seeds: list[int], attempts: int) -> dict:
    """Aggregate attempt-level records into the batch's headline numbers."""
    by_puzzle: dict[int, list[dict]] = {seed: [] for seed in puzzle_seeds}
    for record in records:
        by_puzzle[record["puzzle_seed"]].append(record)

    wins = [r for r in records if r["success"]]
    solved_counts = {
        seed: sum(r["success"] for r in rs) for seed, rs in by_puzzle.items()
    }
    any_solved = sum(count > 0 for count in solved_counts.values())
    all_solved = sum(count == attempts for count in solved_counts.values())

    return {
        "puzzles": len(puzzle_seeds),
        "prompt_format": records[0].get("prompt_format") if records else None,
        "attempts_per_puzzle": attempts,
        "episodes": len(records),
        "attempt_success_rate": round(len(wins) / len(records), 4) if records else 0.0,
        f"pass_at_{attempts}": round(any_solved / len(puzzle_seeds), 4),
        "puzzles_solved_at_least_once": any_solved,
        "puzzles_solved_every_attempt": all_solved,
        "puzzles_never_solved": len(puzzle_seeds) - any_solved,
        "outcomes": dict(Counter(r["outcome"] for r in records).most_common()),
        "solved_count_histogram": {
            str(k): sum(1 for c in solved_counts.values() if c == k)
            for k in range(attempts + 1)
        },
        "mean_steps_when_solved": (
            round(statistics.mean(r["steps"] for r in wins), 2) if wins else None
        ),
        "mean_seconds_per_episode": (
            round(statistics.mean(r["seconds"] for r in records if "seconds" in r), 2)
            if records else None
        ),
        "per_puzzle_solved": {str(seed): solved_counts[seed] for seed in puzzle_seeds},
    }


def print_summary(summary: dict, elapsed: float) -> None:
    attempts = summary["attempts_per_puzzle"]
    print(f"\n========== {summary['episodes']} episodes in {elapsed:.1f}s ==========")
    print(f"puzzles                  {summary['puzzles']} x {attempts} attempts")
    print(f"attempt success rate     {summary['attempt_success_rate']:.1%}")
    print(f"pass@{attempts:<20} {summary[f'pass_at_{attempts}']:.1%} "
          f"({summary['puzzles_solved_at_least_once']}/{summary['puzzles']})")
    print(f"solved every attempt     {summary['puzzles_solved_every_attempt']}")
    print(f"never solved             {summary['puzzles_never_solved']}")
    if summary["mean_steps_when_solved"] is not None:
        print(f"mean steps when solved   {summary['mean_steps_when_solved']}")
    print("\noutcomes")
    for outcome, count in summary["outcomes"].items():
        print(f"  {outcome:<22} {count:>5}  ({count / summary['episodes']:.1%})")
    print("\npuzzles by number of solved attempts")
    for k in range(attempts + 1):
        count = summary["solved_count_histogram"][str(k)]
        bar = "#" * round(40 * count / max(summary["puzzles"], 1))
        print(f"  {k}/{attempts}  {count:>4}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--puzzles", type=int, default=100,
                        help="how many puzzle seeds to play")
    parser.add_argument("--attempts", type=int, default=8,
                        help="independent attempts per puzzle")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="episodes in flight at once against the server")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default="EMPTY",
                        help="unused by vLLM, but required by the client")
    parser.add_argument("--model", default="Qwen3-4B-Instruct-2507",
                        help="served model name given to serve_qwen.sh")
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE,
                        help="text file with one puzzle seed per line")
    parser.add_argument("--seed-index", type=int, default=0,
                        help="zero-based line number to start from in the seed file")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="pick puzzles randomly from the file using this RNG seed")
    parser.add_argument("--format", dest="prompt_format", choices=PROMPT_FORMATS,
                        default=DEFAULT_PROMPT_FORMAT,
                        help="grid: unlabeled text board, 1-indexed; "
                             "list: Python list of lists, 0-indexed")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--model-seed", type=int, default=0,
                        help="base sampling seed; attempt i is sent this seed plus i")
    parser.add_argument("--output", type=Path, default=None,
                        help="write one JSON record per episode to this file")
    parser.add_argument("--summary", type=Path, default=None,
                        help="write the aggregate summary to this JSON file")
    args = parser.parse_args()

    puzzle_seeds = select_seeds(
        args.seed_file, args.puzzles, args.seed_index, args.sample_seed
    )
    if not puzzle_seeds:
        parser.error(f"no seeds selected from {args.seed_file}")
    if len(puzzle_seeds) < args.puzzles:
        print(f"note: only {len(puzzle_seeds)} seeds available in {args.seed_file}")

    jobs = [(seed, attempt)
            for seed in puzzle_seeds
            for attempt in range(args.attempts)]
    print(f"running {len(jobs)} episodes "
          f"({len(puzzle_seeds)} puzzles x {args.attempts} attempts) "
          f"at concurrency {args.concurrency}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=3)
    records: list[dict] = []
    done = 0
    lock = threading.Lock()
    started = time.monotonic()

    # Episodes are written as they land, so an interrupted batch still keeps
    # the transcripts it already earned; the file is rewritten in seed order
    # once the run completes.
    output_handle = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w")

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(run_one, client, args, seed, attempt): (seed, attempt)
                for seed, attempt in jobs
            }
            for future in as_completed(futures):
                record = future.result()
                with lock:
                    records.append(record)
                    if output_handle is not None:
                        output_handle.write(json.dumps(record) + "\n")
                        output_handle.flush()
                    done += 1
                    wins = sum(r["success"] for r in records)
                    print(f"\r{done}/{len(jobs)} episodes, {wins} solved "
                          f"({wins / done:.1%})", end="", flush=True)
    finally:
        if output_handle is not None:
            output_handle.close()
    print()

    records.sort(key=lambda r: (r["puzzle_seed"], r["attempt"]))
    summary = summarize(records, puzzle_seeds, args.attempts)
    print_summary(summary, time.monotonic() - started)

    if args.output:
        # Replace the completion-ordered stream with the sorted final file.
        with args.output.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"\nwrote {len(records)} transcripts to {args.output}")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote summary to {args.summary}")

    # Lost episodes are results, not harness errors, so the exit code stays 0.
    errors = summary["outcomes"].get("error", 0)
    if errors:
        print(f"\nwarning: {errors} episodes failed with a client/server error",
              file=sys.stderr)


if __name__ == "__main__":
    main()
