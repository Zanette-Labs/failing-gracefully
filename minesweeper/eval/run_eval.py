#!/usr/bin/env python3
"""Run the exact 12-turn Minesweeper transfer evaluation on one HF model."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from examples.minesweeper.env_bridge import MinesweeperGame, episode_reward  # noqa: E402


DEFAULT_SEEDS = Path(__file__).with_name("medium12_heldout_500_seeds.txt")
DEFAULT_SEEDS_SHA256 = "e9cc9500070e9672646f9e12434961bccbb7847a8a192fbf8635ef0e72e423ea"
SCAFFOLD = [{"role": "user", "content": "dummy"}, {"role": "assistant", "content": "dummy"}]
ERROR_OUTCOMES = {
    "mine",
    "illegal_move",
    "unparseable",
    "truncated",
    "context_exhausted",
    "aborted",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an HF-exported policy on the fixed 500-puzzle, 12-turn transfer set."
    )
    parser.add_argument("--model", type=Path, required=True, help="Local Hugging Face model directory")
    parser.add_argument("--label", required=True, help="Model label stored in every result row")
    parser.add_argument(
        "--grace",
        type=float,
        default=0.0,
        help="Training grace value, used only to report reward_at_training_grace",
    )
    parser.add_argument("--training-updates", type=int, default=None)
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, required=True, help="Trajectory JSONL; an incomplete file resumes")
    parser.add_argument("--start", type=int, default=0, help="First seed index, for a subset smoke test")
    parser.add_argument("--puzzles", type=int, default=500, help="Number of seeds to evaluate")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPUs for the launched SGLang server")
    parser.add_argument("--port", type=int, default=31310)
    parser.add_argument(
        "--server-url",
        default=None,
        help="Use an existing native SGLang server instead of launching one, e.g. http://127.0.0.1:31310",
    )
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--max-context-tokens", type=int, default=81920)
    parser.add_argument("--sampling-seed", type=int, default=72)
    parser.add_argument("--server-startup-timeout", type=int, default=900)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing trajectory file instead of resuming it",
    )
    args = parser.parse_args()

    if not 0 <= args.grace < 1:
        parser.error("--grace must satisfy 0 <= value < 1")
    if args.start < 0 or args.puzzles < 1:
        parser.error("--start must be nonnegative and --puzzles must be positive")
    if args.concurrency < 1 or args.max_new_tokens < 1 or args.max_context_tokens < 1:
        parser.error("concurrency and token limits must be positive")
    if not args.model.joinpath("config.json").is_file():
        parser.error(f"--model is not a local HF model directory: {args.model}")
    return args


def sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(output.stem + suffix)


def load_seeds(path: Path) -> tuple[list[int], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    seeds = [int(token) for token in raw.split()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"seed file must contain unique integer seeds: {path}")
    if path.resolve() == DEFAULT_SEEDS.resolve():
        if digest != DEFAULT_SEEDS_SHA256 or len(seeds) != 500:
            raise ValueError(f"bundled evaluation seeds failed integrity check: {path}")
    return seeds, digest


def metadata(seed: int) -> dict:
    return {
        "seed": seed,
        "split": "medium12_heldout_500",
        "prompt_format": "list",
        "rows": 6,
        "cols": 6,
        "mines": 6,
        "min_steps": 6,
        "max_steps": 12,
        "max_turns": 12,
    }


def user_delta(content: str, tokenizer) -> list[int]:
    """Tokenize an appended user turn exactly as the training agent does."""
    before = tokenizer.apply_chat_template(
        SCAFFOLD, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    after = tokenizer.apply_chat_template(
        SCAFFOLD + [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if after[: len(before)] != before:
        raise ValueError("chat template is not append-only for user turns")
    return after[len(before) :]


def revealed(game: MinesweeperGame) -> int:
    return sum(value is not None for row in game.env.observation().visible for value in row)


def play_episode(
    index: int,
    meta: dict,
    args: argparse.Namespace,
    tokenizer,
    server_url: str,
) -> dict:
    started = time.monotonic()
    game = MinesweeperGame(meta, max_tokens=args.max_new_tokens)
    initial_visible = revealed(game)
    initial_messages = [
        {"role": "system", "content": game.system_prompt},
        {"role": "user", "content": game.user_prompt()},
    ]
    rendered = tokenizer.apply_chat_template(initial_messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    turns = []
    request_errors = []
    user_text = initial_messages[-1]["content"]

    with requests.Session() as session:
        for turn_index in range(game.max_turns):
            limit = min(args.max_new_tokens, args.max_context_tokens - len(token_ids))
            if limit <= 0:
                game.end_early("context_exhausted")
                break
            payload = {
                "input_ids": token_ids,
                "sampling_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": -1,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": limit,
                    "skip_special_tokens": False,
                    "no_stop_trim": True,
                    "spaces_between_special_tokens": False,
                    "sampling_seed": args.sampling_seed,
                },
                "return_logprob": True,
            }
            response = None
            for retry in range(3):
                try:
                    http_response = session.post(
                        f"{server_url}/generate", json=payload, timeout=(15, 1800)
                    )
                    http_response.raise_for_status()
                    response = http_response.json()
                    break
                except requests.RequestException as error:
                    request_errors.append(
                        {"turn": turn_index + 1, "retry": retry, "error": str(error)}
                    )
                    if retry < 2:
                        time.sleep(2**retry)
            if response is None:
                raise RuntimeError(
                    f"server request failed for puzzle {index}: {request_errors[-1]['error']}"
                )

            info = response["meta_info"]
            new_ids = [entry[1] for entry in info["output_token_logprobs"]]
            if len(new_ids) != info["completion_tokens"]:
                raise ValueError("SGLang token logprob response has inconsistent length")
            finish_reason = info["finish_reason"]["type"]
            if finish_reason == "abort":
                raise RuntimeError(f"SGLang aborted puzzle {index}: {info['finish_reason']}")

            turns.append(
                {
                    "turn": turn_index + 1,
                    "user": user_text,
                    "reply": response["text"],
                    "finish_reason": finish_reason,
                    "input_tokens": len(token_ids),
                    "completion_tokens": len(new_ids),
                    "max_new_tokens": limit,
                }
            )
            token_ids.extend(new_ids)
            if not token_ids or token_ids[-1] != tokenizer.eos_token_id:
                token_ids.append(tokenizer.eos_token_id)
            token_ids.extend(tokenizer.encode("\n", add_special_tokens=False))

            if game.act(response["text"], finish_reason):
                break
            user_text = game.user_prompt()
            token_ids.extend(user_delta(user_text, tokenizer))

    if not game.done:
        raise RuntimeError(f"puzzle {index} did not reach a terminal state")
    return {
        "model": args.label,
        "grace": args.grace,
        "training_updates": args.training_updates,
        "puzzle_index": index,
        "puzzle_seed": game.seed,
        "metadata": meta,
        "attempt": 0,
        "outcome": game.outcome,
        "success": game.success,
        "graceful": game.outcome == "turn_limit",
        "reward_at_training_grace": episode_reward(game.outcome, args.grace),
        "steps": game.steps,
        "n_turns": len(turns),
        "repeats": game.repeats,
        "actions": game.actions,
        "initial_visible_safe_cells": initial_visible,
        "final_visible_safe_cells": revealed(game),
        "total_safe_cells": game.env.rows * game.env.cols - game.env.mine_count,
        "completion_tokens": sum(turn["completion_tokens"] for turn in turns),
        "seconds": time.monotonic() - started,
        "initial_messages": initial_messages,
        "error": game.error,
        "request_errors": request_errors,
        "turns": turns,
    }


def load_existing(path: Path, selected: dict[int, dict], args: argparse.Namespace) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_index = {row["puzzle_index"]: row for row in rows}
    if len(rows) != len(by_index):
        raise ValueError(f"duplicate puzzle indices in {path}")
    if not set(by_index).issubset(selected):
        raise ValueError(f"existing output contains indices outside the requested seed slice: {path}")
    for index, row in by_index.items():
        if row["model"] != args.label or float(row["grace"]) != args.grace:
            raise ValueError(f"existing output belongs to another model or grace value: {path}")
        if row["puzzle_seed"] != selected[index]["seed"]:
            raise ValueError(f"existing output seed mismatch at puzzle index {index}")
    return by_index


def wait_for_server(url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"SGLang server exited with status {process.returncode}")
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.ok:
                return
        except requests.RequestException:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"SGLang server did not become ready at {url}")
        time.sleep(3)


def launch_server(args: argparse.Namespace) -> tuple[str, subprocess.Popen, object, list[str]]:
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must name at least one GPU")
    url = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--served-model-name",
        args.label,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp-size",
        str(len(gpu_ids)),
        "--context-length",
        str(args.max_context_tokens),
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        "64",
        "--chunked-prefill-size",
        "4096",
        "--random-seed",
        str(args.sampling_seed),
        "--sampling-defaults",
        "openai",
        "--sampling-backend",
        "pytorch",
        "--dist-init-addr",
        f"127.0.0.1:{args.port + 500}",
    ]
    log_path = sidecar(args.output, ".server.log")
    log = log_path.open("a")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpu_ids))
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        wait_for_server(url, process, args.server_startup_timeout)
    except BaseException:
        stop_server(process)
        log.close()
        raise
    return url, process, log, command


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def evaluate_pending(
    selected: dict[int, dict],
    completed: dict[int, dict],
    args: argparse.Namespace,
    tokenizer,
    server_url: str,
) -> None:
    pending = [(index, meta) for index, meta in selected.items() if index not in completed]
    if not pending:
        return
    counts = Counter(row["outcome"] for row in completed.values())
    with args.output.open("a") as output, futures.ThreadPoolExecutor(args.concurrency) as pool:
        jobs = {
            pool.submit(play_episode, index, meta, args, tokenizer, server_url): index
            for index, meta in pending
        }
        for number, future in enumerate(futures.as_completed(jobs), len(completed) + 1):
            row = future.result()
            output.write(json.dumps(row) + "\n")
            output.flush()
            counts[row["outcome"]] += 1
            if number % 25 == 0 or number == len(selected):
                print(
                    json.dumps(
                        {
                            "complete": number,
                            "total": len(selected),
                            "outcomes": dict(counts),
                        }
                    ),
                    flush=True,
                )


def audit(records: list[dict], args: argparse.Namespace) -> None:
    for record in records:
        game = MinesweeperGame(record["metadata"], max_tokens=args.max_new_tokens)
        if game.system_prompt != record["initial_messages"][0]["content"]:
            raise AssertionError(f"system prompt mismatch at puzzle {record['puzzle_index']}")
        for turn in record["turns"]:
            if game.user_prompt() != turn["user"]:
                raise AssertionError(f"user prompt mismatch at puzzle {record['puzzle_index']}")
            game.act(turn["reply"], turn["finish_reason"])
        if record["outcome"] == "context_exhausted" and not game.done:
            game.end_early("context_exhausted")
        if game.outcome != record["outcome"]:
            raise AssertionError(f"outcome mismatch at puzzle {record['puzzle_index']}")
        if game.steps != record["steps"] or game.repeats != record["repeats"]:
            raise AssertionError(f"environment metric mismatch at puzzle {record['puzzle_index']}")
        if [list(action) for action in game.actions] != record["actions"]:
            raise AssertionError(f"action replay mismatch at puzzle {record['puzzle_index']}")
        if any(
            turn["input_tokens"] + turn["completion_tokens"] > args.max_context_tokens
            for turn in record["turns"]
        ):
            raise AssertionError(f"context limit exceeded at puzzle {record['puzzle_index']}")


def summarize(records: list[dict], args: argparse.Namespace, seed_digest: str) -> dict:
    counts = Counter(record["outcome"] for record in records)
    n = len(records)
    return {
        "model": args.label,
        "model_path": str(args.model),
        "training_grace": args.grace,
        "training_updates": args.training_updates,
        "seed_file": str(args.seed_file),
        "seed_file_sha256": seed_digest,
        "puzzles": n,
        "start_index": args.start,
        "attempts_per_puzzle": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens_per_turn": args.max_new_tokens,
        "max_context_tokens": args.max_context_tokens,
        "max_turns": 12,
        "sampling_seed": args.sampling_seed,
        "success_rate": sum(record["success"] for record in records) / n,
        "safe_turn_limit_rate": counts["turn_limit"] / n,
        "terminal_error_rate": sum(counts[name] for name in ERROR_OUTCOMES) / n,
        "mean_reward_at_training_grace": statistics.mean(
            record["reward_at_training_grace"] for record in records
        ),
        "mean_turns": statistics.mean(record["n_turns"] for record in records),
        "mean_completion_tokens": statistics.mean(record["completion_tokens"] for record in records),
        "outcomes": dict(counts.most_common()),
        "audit": "PASS",
    }


def main() -> None:
    global requests

    args = parse_args()
    import requests
    from transformers import AutoTokenizer

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()

    seeds, seed_digest = load_seeds(args.seed_file)
    end = args.start + args.puzzles
    if end > len(seeds):
        raise ValueError(f"requested seed slice [{args.start}:{end}] exceeds {len(seeds)} seeds")
    selected = {index: metadata(seeds[index]) for index in range(args.start, end)}
    completed = load_existing(args.output, selected, args)
    pending_count = len(selected) - len(completed)

    manifest_path = sidecar(args.output, "_manifest.json")
    summary_path = sidecar(args.output, "_summary.json")
    manifest = {
        "status": "starting",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "label": args.label,
        "grace": args.grace,
        "training_updates": args.training_updates,
        "output": str(args.output),
        "seed_file": str(args.seed_file),
        "seed_file_sha256": seed_digest,
        "seed_slice": [args.start, end],
        "pending_at_start": pending_count,
        "protocol": "native SGLang /generate with append-only token history matching training",
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "max_context_tokens": args.max_context_tokens,
            "sampling_seed": args.sampling_seed,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    server_process = None
    server_log = None
    try:
        if pending_count:
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            if args.server_url:
                server_url = args.server_url.rstrip("/")
                wait_for_server(server_url, None, 30)
                manifest["server"] = {"external_url": server_url}
            else:
                server_url, server_process, server_log, command = launch_server(args)
                manifest["server"] = {
                    "url": server_url,
                    "pid": server_process.pid,
                    "gpus": args.gpus,
                    "command": command,
                    "log": str(sidecar(args.output, ".server.log")),
                }
            manifest["status"] = "running"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            evaluate_pending(selected, completed, args, tokenizer, server_url)

        final_rows = load_existing(args.output, selected, args)
        if len(final_rows) != len(selected):
            raise RuntimeError(f"evaluation incomplete: {len(final_rows)}/{len(selected)} puzzles")
        records = [final_rows[index] for index in sorted(final_rows)]
        audit(records, args)
        summary = summarize(records, args, seed_digest)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        manifest.update(
            status="complete",
            completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            summary=str(summary_path),
            audit="PASS",
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    except BaseException:
        manifest.update(status="error", error=traceback.format_exc())
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    finally:
        if server_process is not None:
            stop_server(server_process)
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    main()
