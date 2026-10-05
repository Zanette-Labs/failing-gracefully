"""Write reproducible starting-board samples from the simulator as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minesweeper_env import MinesweeperEnv


def sample_puzzles(seeds: list[int]) -> list[dict]:
    samples = []
    for seed in seeds:
        env = MinesweeperEnv()
        _, info = env.reset(seed=seed)
        samples.append(
            {
                "seed": seed,
                "opening": list(info["opening"]),
                "rows": env.rows,
                "cols": env.cols,
                "mines": env.mine_count,
                "board": env.render(coordinates=False).splitlines(),
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("sample_puzzles.json"),
    )
    args = parser.parse_args()
    args.output.write_text(json.dumps(sample_puzzles(args.seeds), indent=2) + "\n")
    print(f"wrote {len(args.seeds)} puzzles to {args.output}")


if __name__ == "__main__":
    main()
