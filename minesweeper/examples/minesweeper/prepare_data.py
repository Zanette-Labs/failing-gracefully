"""Build a miles prompt-data jsonl from Minesweeper puzzle-seed files.

Each output row:
    {"prompt": <system prompt>,                         # cosmetic; agent.py rebuilds it
     "metadata": {"seed": <puzzle seed>, "split": <name>, "prompt_format": <grid|list>,
                  "rows": 6, "cols": 6, "mines": 6, "min_steps": 6, "max_steps": 12, "max_turns": 12}}

``--max-turns`` is the per-episode turn budget. The env caps the generator's
longest accepted solution (``max_steps``) at the budget, and the shortest
(``min_steps``) at ``max_steps``, so a budget below 12 changes the board each
seed produces; use seed files built for that budget
(``build_comparison_dataset.py --max-turns N``, e.g. ``game_analysis/turns8/``
or ``game_analysis/turns4/``) and the same ``--min-steps`` / ``--max-steps``.

``--format`` picks how the board is shown to the policy (see env_bridge.py):
``grid`` is the original unlabeled text board with 1-indexed actions, ``list``
prints a Python list of lists and takes 0-indexed actions. Rows in one file may
mix formats, but a run normally uses one throughout.

Seeds come from the one-integer-per-line splits under
``../minesweeper/game_analysis/`` (easy_100 / heuristic_not_naive_10k /
heuristic_failures_1k). ``--start/--count`` slice a file so a held-out tail can
serve as eval; ``--repeat`` duplicates rows to up-weight a small split inside a
mixed train file; ``--append`` adds to an existing jsonl so a mixed set is built
with several calls. GRPO groups the n samples of each ROW, so duplicated rows
are simply extra prompt-groups of that seed.

Examples:
    python examples/minesweeper/prepare_data.py --seed-file $MS/game_analysis/easy_100_seeds.txt \
        --split easy --repeat 5 --output train.jsonl
    python examples/minesweeper/prepare_data.py --seed-file $MS/game_analysis/heuristic_not_naive_10k_seeds.txt \
        --split medium --count 9500 --output train.jsonl --append --shuffle
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.minesweeper.env_bridge import (  # noqa: E402
    DEFAULT_BOARD,
    DEFAULT_PROMPT_FORMAT,
    PROMPT_FORMATS,
    make_env,
    system_prompt,
)


def load_seeds(path: str) -> list[int]:
    with open(path) as f:
        seeds = [int(tok) for tok in f.read().split() if tok.strip()]
    if not seeds:
        raise ValueError(f"no seeds found in {path}")
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(description="Minesweeper seeds -> miles prompt-data jsonl")
    ap.add_argument("--seed-file", required=True, help="text file, one puzzle seed per line")
    ap.add_argument("--split", required=True, help="label stored in metadata.split (per-split wandb columns)")
    ap.add_argument("--start", type=int, default=0, help="first line to take (0-based)")
    ap.add_argument("--count", type=int, default=0, help="how many seeds from --start (0 = to the end)")
    ap.add_argument("--repeat", type=int, default=1, help="duplicate every row this many times")
    ap.add_argument("--rows", type=int, default=DEFAULT_BOARD["rows"])
    ap.add_argument("--cols", type=int, default=DEFAULT_BOARD["cols"])
    ap.add_argument("--mines", type=int, default=DEFAULT_BOARD["mines"])
    ap.add_argument("--max-turns", type=int, default=DEFAULT_BOARD["max_turns"])
    ap.add_argument("--min-steps", type=int, default=None,
                    help="shortest accepted no-guess solution (default min(6, max_steps))")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="longest accepted no-guess solution; default min(12, --max-turns)")
    ap.add_argument("--format", dest="prompt_format", choices=PROMPT_FORMATS,
                    default=DEFAULT_PROMPT_FORMAT,
                    help="grid: unlabeled text board, 1-indexed; list: Python list of lists, 0-indexed")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="per-turn reply budget quoted in the stored prompt (cosmetic: agent.py "
                         "rebuilds the prompt from the run's max response len); 0 omits it")
    ap.add_argument("--output", required=True)
    ap.add_argument("--append", action="store_true", help="append to --output instead of overwriting")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle the WHOLE output file after writing (use on the last call)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --shuffle")
    args = ap.parse_args()

    max_steps = min(DEFAULT_BOARD["max_steps"], args.max_turns) if args.max_steps is None else args.max_steps
    min_steps = min(DEFAULT_BOARD["min_steps"], max_steps) if args.min_steps is None else args.min_steps
    board = {"rows": args.rows, "cols": args.cols, "mines": args.mines,
             "min_steps": min_steps, "max_steps": max_steps, "max_turns": args.max_turns}
    # depends only on board params + format, not the seed
    prompt = system_prompt(make_env(board), args.prompt_format, args.max_tokens or None)

    seeds = load_seeds(args.seed_file)
    end = len(seeds) if args.count <= 0 else args.start + args.count
    seeds = seeds[args.start:end]
    if not seeds:
        raise ValueError(f"empty slice [{args.start}:{end}] of {args.seed_file}")

    rows = [
        {"prompt": prompt,
         "metadata": {"seed": s, "split": args.split, "prompt_format": args.prompt_format, **board}}
        for s in seeds
        for _ in range(args.repeat)
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "a" if args.append else "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    print(f"[prepare] {args.split} ({args.prompt_format}): {len(seeds)} seeds x{args.repeat} "
          f"= {len(rows)} rows ({'appended to' if args.append else 'wrote'} {args.output})")

    if args.shuffle:
        with open(args.output) as f:
            lines = [ln for ln in f if ln.strip()]
        random.Random(args.seed).shuffle(lines)
        with open(args.output, "w") as out:
            out.writelines(lines)
        print(f"[prepare] shuffled {len(lines)} total rows in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
