"""Play a dataset puzzle against a vLLM-hosted model over multiple turns.

Start the server with ``./serve_qwen.sh`` first, then for example::

    python play_with_model.py --seed-index 0
    python play_with_model.py --seed 57 --verbose
    python play_with_model.py --seed 57 --debug    # step through turn by turn

Seeds come from the comparison dataset (one integer per line); the default file
is the split the naive baseline fails and the heuristic baseline solves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import random
import re
import sys

from openai import OpenAI

from minesweeper_env import MinesweeperEnv
from play_puzzle import (
    repeat_notice,
    DEFAULT_PROMPT_FORMAT,
    PROMPT_FORMATS,
    model_prompt,
    parse_action,
    system_prompt,
    to_env_coords,
    translate_error,
)

DEFAULT_SEED_FILE = (
    Path(__file__).parent / "game_analysis" / "heuristic_not_naive_10k_seeds.txt"
)
# Fallback for a chatty last line: the last reveal(...) it contains.
LOOSE_ACTION_PATTERN = re.compile(
    r"reveal\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE
)

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def color_supported() -> bool:
    """Colorize only for a real terminal that has not opted out."""
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


class Display:
    """Episode output: quiet by default, or a stepped, colored debug view.

    Prompts sent to the model print in green and its replies print in red.
    In debug mode every turn waits for confirmation before it runs.
    """

    def __init__(self, verbose: bool = False, debug: bool = False,
                 color: bool = True) -> None:
        self.debug = debug
        self.verbose = verbose or debug
        self.color = color

    def _paint(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    def prompt(self, title: str, text: str) -> None:
        if self.verbose:
            print(self._paint(f"\n=== {title} ===\n{text}", GREEN))

    def reply(self, text: str) -> None:
        if self.verbose:
            print(self._paint(f"\n--- MODEL REPLY ---\n{text}", RED))

    def note(self, text: str) -> None:
        if self.verbose:
            print(text)

    def confirm(self, question: str) -> bool:
        """Ask whether to run the next turn; only debug mode ever pauses."""
        if not self.debug:
            return True
        while True:
            try:
                answer = input(f"\n{question} [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return False
            if answer in {"", "y", "yes"}:
                return True
            if answer in {"n", "no", "q", "quit"}:
                return False
            print("please answer y or n")


def load_seeds(path: Path) -> list[int]:
    """Read one integer seed per line, ignoring blank lines."""
    seeds = [int(line) for line in path.read_text().split() if line.strip()]
    if not seeds:
        raise ValueError(f"no seeds found in {path}")
    return seeds


def extract_action(text: str) -> tuple[int, int]:
    """Parse the action from the reply's last line, tolerating trailing prose.

    Reasoning above the final line is ignored, so only the line the model
    finishes on counts as its move.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    try:
        return parse_action(last)
    except ValueError:
        matches = LOOSE_ACTION_PATTERN.findall(last)
        if not matches:
            raise ValueError(
                "expected reveal(row, col) on the last line, for example reveal(2, 5)"
            ) from None
        row, col = matches[-1]
        return int(row), int(col)


class ModelPlayer:
    """A chat client that keeps the full episode in its message history."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.messages: list[dict[str, str]] = []

    def start(self, prompt: str) -> None:
        self.messages = [{"role": "system", "content": prompt}]

    def respond(self, prompt: str) -> tuple[str, str]:
        """Send one user message; return the reply and its finish reason."""
        self.messages.append({"role": "user", "content": prompt})
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        choice = completion.choices[0]
        reply = (choice.message.content or "").strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply, choice.finish_reason or "stop"


def play_episode(
    player: ModelPlayer,
    puzzle_seed: int,
    display: Display,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
) -> dict:
    """Run one full episode and return a JSON-serializable record.

    There are no retries: an unparseable reply or an out-of-bounds reveal ends
    the episode as a loss, exactly as a mine would. Revealing an already-visible
    cell is not an error: it uncovers nothing, spends a turn, and the next prompt
    says so above the board. Actions in the record are in the model's own coordinates, so a ``list``
    episode stores 0-indexed pairs.
    """
    env = MinesweeperEnv()
    _, info = env.reset(seed=puzzle_seed)
    player.start(system_prompt(env, prompt_format, getattr(player, "max_tokens", None)))
    display.prompt("SYSTEM PROMPT", player.messages[0]["content"])

    turns: list[dict] = []
    prompt = model_prompt(env, prompt_format)

    def record(outcome: str, success: bool) -> dict:
        return {
            "puzzle_seed": puzzle_seed,
            "prompt_format": prompt_format,
            "success": success,
            "outcome": outcome,
            "steps": env.observation().steps,
            "turns": turns,
            "final_board": env.render(coordinates=False),
        }

    while True:
        display.prompt(f"USER PROMPT (turn {len(turns) + 1})", prompt)
        if not display.confirm("Run this turn?"):
            display.note("Stopped before the turn ran.")
            return record("aborted", False)

        reply, finish_reason = player.respond(prompt)
        display.reply(reply)

        def fail(outcome: str, message: str, action: list[int] | None = None) -> dict:
            """End the episode as a loss, naming which way the reply went wrong."""
            turn = {"reply": reply, "finish_reason": finish_reason, "error": message}
            if action is not None:
                turn["action"] = action
            turns.append(turn)
            display.note(f"\nInvalid action ({outcome}), episode lost: {message}")
            return record(outcome, False)

        # Each way a reply can fail is its own outcome: the three are diagnosed
        # differently, so collapsing them into one bucket hides what went wrong.
        if finish_reason == "length":
            # The model reasoned past its budget, so any action here is partial.
            return fail(
                "truncated",
                "the response was cut off before a complete action; "
                "reason briefly and end with the action",
            )
        try:
            row, col = extract_action(reply)
        except ValueError as exc:
            return fail("unparseable", str(exc))
        try:
            observation, reward, done, result = env.reveal(
                *to_env_coords(row, col, prompt_format)
            )
        except ValueError as exc:
            # An out-of-bounds action is a lost episode, not a chance to try
            # again. The message is phrased in the model's own coordinates.
            return fail(
                "illegal_move", translate_error(str(exc), row, col, prompt_format), [row, col]
            )

        repeated = bool(result.get("already_revealed"))
        turns.append({
            "reply": reply,
            "finish_reason": finish_reason,
            "action": [row, col],
            "reward": reward,
            "outcome": result["outcome"],
            "already_revealed": repeated,
        })
        display.note(
            f"\n{'Repeated' if repeated else 'Accepted'} reveal({row}, {col}) -> reward {reward:g}, "
            f"{result['turns_remaining']} turns left"
        )
        if done:
            display.note("\n=== FINAL BOARD ===")
            display.note(env.render(coordinates=False))
            display.note(f"\nOutcome: {result['outcome']} after {result['steps']} moves.")
            return record(result["outcome"], observation.success)
        prompt = model_prompt(env, prompt_format)
        if repeated:
            prompt = f"{repeat_notice(row, col, prompt_format)}\n\n{prompt}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default="EMPTY", help="unused by vLLM, but required by the client")
    parser.add_argument("--model", default="Qwen3-4B-Instruct-2507",
                        help="served model name given to serve_qwen.sh")
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE,
                        help="text file with one puzzle seed per line")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="puzzle seed to play, bypassing the seed file")
    group.add_argument("--seed-index", type=int, help="zero-based line number in the seed file")
    parser.add_argument("--episodes", type=int, default=1,
                        help="number of consecutive seeds to play from --seed-index")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="pick puzzles randomly from the file using this RNG seed")
    parser.add_argument("--format", dest="prompt_format", choices=PROMPT_FORMATS,
                        default=DEFAULT_PROMPT_FORMAT,
                        help="grid: unlabeled text board, 1-indexed; "
                             "list: Python list of lists, 0-indexed")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="nucleus sampling cutoff sent to the server")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="the model reasons at length before acting; too low truncates the action")
    parser.add_argument("--model-seed", type=int, default=0,
                        help="sampling seed passed to the server for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="print every prompt and reply")
    parser.add_argument("--debug", action="store_true",
                        help="step through the rollout: prompts in green, replies in "
                             "red, and a confirmation before every turn")
    parser.add_argument("--no-color", action="store_true",
                        help="disable ANSI color even on a terminal")
    parser.add_argument("--output", type=Path, default=None,
                        help="write one JSON record per episode to this file")
    args = parser.parse_args()

    if args.seed is not None:
        puzzle_seeds = [args.seed]
    else:
        seeds = load_seeds(args.seed_file)
        if args.sample_seed is not None:
            picks = random.Random(args.sample_seed).sample(
                seeds, min(args.episodes, len(seeds))
            )
            puzzle_seeds = picks
        else:
            start = args.seed_index or 0
            puzzle_seeds = seeds[start:start + args.episodes]
        if not puzzle_seeds:
            parser.error(f"no seeds selected from {args.seed_file}")

    display = Display(
        verbose=args.verbose,
        debug=args.debug,
        color=not args.no_color and color_supported(),
    )
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    player = ModelPlayer(
        client, args.model, args.temperature, args.top_p, args.max_tokens,
        args.model_seed,
    )

    records = []
    for puzzle_seed in puzzle_seeds:
        print(f"\n########## puzzle seed {puzzle_seed} ##########")
        record = play_episode(player, puzzle_seed, display, args.prompt_format)
        records.append(record)
        print(f"seed {puzzle_seed}: {record['outcome']} in {record['steps']} steps")
        if record["outcome"] == "aborted":
            print("stopped at your request; skipping any remaining puzzles")
            break

    wins = sum(record["success"] for record in records)
    print(f"\nsolved {wins}/{len(records)} puzzles")

    if args.output:
        with args.output.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"wrote transcripts to {args.output}")

    # A lost episode is a result, not a harness error, so the exit code stays 0.


if __name__ == "__main__":
    main()
