"""Build reproducible seed-only datasets comparing the two baselines."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path

from minesweeper_env import MinesweeperEnv

from .heuristic_solver import HeuristicSolver
from .naive_solver import NaiveSolver

HEURISTIC_NOT_NAIVE = "heuristic_not_naive"
HEURISTIC_FAILURE = "heuristic_failure"
EASY = "easy"

# An easy board must survive repeated tries by the weakest baseline, so luck in
# a single random walk cannot make a hard board look easy.
EASY_TRIALS = 8
EASY_MAX_STEPS = 5

# Board the splits are built on. ``max_turns`` is the turn budget every solver
# plays under; the env requires ``min_steps <= max_steps`` (shortest/longest
# accepted no-guess solution) and ``max_steps <= max_turns``, so a shorter
# budget also changes which boards a seed generates. Splits built for different
# boards are therefore unrelated.
DEFAULT_BOARD = {"rows": 6, "cols": 6, "mines": 6, "min_steps": 6, "max_steps": 12, "max_turns": 12}


def easy_trial_seed(puzzle_seed: int, trial: int) -> int:
    """Solver seed for one easy-split trial, kept distinct across puzzles."""
    return puzzle_seed * 1_000 + trial


def is_easy(initial_env: MinesweeperEnv, seed: int, easy_max_steps: int = EASY_MAX_STEPS) -> bool:
    """Report whether the naive baseline wins quickly however it chooses.

    The naive solver picks uniformly among the currently provable safe cells, so
    replaying it under several seeds walks several different paths through the
    same board.  A board counts as easy only when every one of those walks wins
    inside ``EASY_MAX_STEPS`` of the twelve available turns, which leaves the
    board both unambiguous at every turn and far short of the turn limit.
    """
    for trial in range(EASY_TRIALS):
        result = NaiveSolver(seed=easy_trial_seed(seed, trial)).solve(
            deepcopy(initial_env)
        )
        if not result.success:
            if result.outcome != "turn_limit":
                raise RuntimeError(f"naive solver hit an unsafe cell for seed {seed}")
            return False
        # Short-circuit: most boards blow the budget on their first trial.
        if len(result.moves) > easy_max_steps:
            return False
    return True


def classify_seed(
    seed: int, board: dict | None = None, easy_max_steps: int = EASY_MAX_STEPS
) -> str | None:
    """Classify a seed, using it for both solvers' random generators."""
    initial_env = MinesweeperEnv(**(board or DEFAULT_BOARD))
    initial_env.reset(seed=seed)

    heuristic = HeuristicSolver(seed=seed).solve(deepcopy(initial_env))
    if not heuristic.success:
        if heuristic.outcome != "turn_limit":
            raise RuntimeError(f"heuristic hit an unsafe cell for seed {seed}")
        return HEURISTIC_FAILURE

    naive = NaiveSolver(seed=seed).solve(deepcopy(initial_env))
    if not naive.success:
        if naive.outcome != "turn_limit":
            raise RuntimeError(f"naive solver hit an unsafe cell for seed {seed}")
        return HEURISTIC_NOT_NAIVE
    # Both baselines already win here; the easy split is the subset they win
    # comfortably, so it never overlaps the two harder splits.
    return EASY if is_easy(initial_env, seed, easy_max_steps) else None


def evaluate_seed(seed: int, board: dict | None = None) -> dict | None:
    """Return the compact comparison outcome for one seed."""
    classification = classify_seed(seed, board)
    if classification is None:
        return None
    return {
        "puzzle_seed": seed,
        "solver_seed": seed,
        "classification": classification,
    }


def build_datasets(
    wanted: dict[str, int],
    workers: int,
    batch_size: int,
    board: dict | None = None,
    easy_max_steps: int = EASY_MAX_STEPS,
) -> tuple[dict[str, list[int]], dict[str, int], int]:
    """Scan nonnegative seeds until every requested split is full.

    Returns the seeds per split, the number of seeds scanned when each split
    filled, and the total scanned.  Splits fill at very different rates, so the
    per-split figure is what makes a single split reproducible on its own.
    """
    collected: dict[str, list[int]] = {name: [] for name in wanted}
    filled_at: dict[str, int] = {}
    next_seed = 0
    classify = partial(classify_seed, board=board or DEFAULT_BOARD, easy_max_steps=easy_max_steps)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        while any(len(collected[name]) < count for name, count in wanted.items()):
            seeds = range(next_seed, next_seed + batch_size)
            classifications = executor.map(classify, seeds, chunksize=32)
            for seed, classification in zip(seeds, classifications):
                if classification is None:
                    continue
                bucket = collected.get(classification)
                if bucket is not None and len(bucket) < wanted[classification]:
                    bucket.append(seed)
                    if len(bucket) == wanted[classification]:
                        filled_at.setdefault(classification, seed + 1)
            next_seed += batch_size
            progress = "; ".join(
                f"{name} {len(collected[name])}/{wanted[name]}" for name in wanted
            )
            print(f"scanned {next_seed}; {progress}", flush=True)

    for name in wanted:
        filled_at.setdefault(name, next_seed)
    return collected, filled_at, next_seed


def _write_seeds(path: Path, seeds: list[int]) -> None:
    path.write_text("".join(f"{seed}\n" for seed in seeds))


SPLITS = {
    HEURISTIC_NOT_NAIVE: {
        "default_count": 10_000,
        "file": "heuristic_not_naive_10k_seeds.txt",
        "qualification": "heuristic success and naive turn_limit",
    },
    HEURISTIC_FAILURE: {
        "default_count": 1_000,
        "file": "heuristic_failures_1k_seeds.txt",
        "qualification": "heuristic turn_limit",
    },
    EASY: {
        "default_count": 100,
        "file": "easy_100_seeds.txt",
        "qualification": (
            f"naive success in at most {EASY_MAX_STEPS} steps on all "
            f"{EASY_TRIALS} trial seeds"
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heuristic-not-naive-count", type=int,
                        default=SPLITS[HEURISTIC_NOT_NAIVE]["default_count"])
    parser.add_argument("--heuristic-failure-count", type=int,
                        default=SPLITS[HEURISTIC_FAILURE]["default_count"])
    parser.add_argument("--easy-count", type=int,
                        default=SPLITS[EASY]["default_count"],
                        help="seeds the naive baseline wins quickly and reliably")
    parser.add_argument("--only", choices=sorted(SPLITS), default=None,
                        help="build just this split, leaving the other files alone")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_BOARD["max_turns"],
                        help="turn budget the solvers play under (default 12)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="longest accepted no-guess solution; default min(12, --max-turns). "
                             "Changing it changes the board every seed generates.")
    parser.add_argument("--min-steps", type=int, default=None,
                        help="shortest accepted no-guess solution; default min(6, max_steps), so a "
                             "budget below 6 (e.g. --max-turns 4) keeps only boards whose solution "
                             "is exactly max_steps long")
    parser.add_argument("--easy-max-steps", type=int, default=None,
                        help=f"an easy board must be won by the naive solver within this many turns "
                             f"(default min({EASY_MAX_STEPS}, --max-turns))")
    args = parser.parse_args()

    board = dict(DEFAULT_BOARD)
    board["max_turns"] = args.max_turns
    board["max_steps"] = min(DEFAULT_BOARD["max_steps"], args.max_turns) if args.max_steps is None else args.max_steps
    board["min_steps"] = min(DEFAULT_BOARD["min_steps"], board["max_steps"]) if args.min_steps is None else args.min_steps
    if args.easy_max_steps is None:
        args.easy_max_steps = min(EASY_MAX_STEPS, board["max_turns"])
    if not 1 <= args.easy_max_steps <= board["max_turns"]:
        parser.error("--easy-max-steps must be between 1 and --max-turns")
    # Fail fast on an inconsistent board instead of inside the worker pool.
    MinesweeperEnv(**board)
    SPLITS[EASY]["qualification"] = (
        f"naive success in at most {args.easy_max_steps} steps on all {EASY_TRIALS} trial seeds"
    )

    wanted = {
        HEURISTIC_NOT_NAIVE: args.heuristic_not_naive_count,
        HEURISTIC_FAILURE: args.heuristic_failure_count,
        EASY: args.easy_count,
    }
    if args.only:
        wanted = {args.only: wanted[args.only]}
    if min(*wanted.values(), args.workers, args.batch_size) < 1:
        parser.error("counts, workers, and batch-size must be positive")

    collected, filled_at, scanned = build_datasets(
        wanted, args.workers, args.batch_size, board, args.easy_max_steps
    )

    manifest_path = args.output_dir / "comparison_dataset_manifest.json"
    # Rebuilding one split must not drop the others from the manifest.
    manifest = {"splits": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("splits", {})
        # Splits recorded before the range moved per-split inherit the old
        # whole-file range, which is the only bound their build actually proved.
        legacy_range = manifest.get("seed_range_scanned")
        for split in manifest["splits"].values():
            if "seed_range_scanned" not in split and legacy_range:
                split["seed_range_scanned"] = legacy_range
    manifest["solver_seed_rule"] = (
        "solver_seed equals puzzle_seed, except the easy split, whose trial t "
        "uses puzzle_seed * 1000 + t"
    )
    if manifest.get("board", board) != board:
        parser.error(f"{manifest_path} was built for board {manifest['board']}, not {board}; "
                     "use a different --output-dir")
    manifest["board"] = board
    for name, seeds in collected.items():
        path = args.output_dir / SPLITS[name]["file"]
        _write_seeds(path, seeds)
        manifest["splits"][name] = {
            "seeds": len(seeds),
            "qualification": SPLITS[name]["qualification"],
            "data_file": path.name,
            "seed_range_scanned": [0, filled_at[name] - 1],
            "board": board,
        }
        print(f"wrote {len(seeds)} seeds to {path}")
    manifest["seed_range_scanned"] = [
        0,
        max(split["seed_range_scanned"][1]
            for split in manifest["splits"].values()
            if "seed_range_scanned" in split),
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
