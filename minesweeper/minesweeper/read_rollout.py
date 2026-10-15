"""Print a rollout from a run_batch.py transcript file in readable form.

    python read_rollout.py runs/easy_50x1.jsonl              # list every episode
    python read_rollout.py runs/easy_50x1.jsonl --seed 403   # read one in full
    python read_rollout.py runs/easy_50x1.jsonl --outcome success
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def show(record: dict) -> None:
    head = (f"===== seed {record['puzzle_seed']} attempt {record['attempt']}: "
            f"{record['outcome']} in {record['steps']} steps =====")
    print(head)
    for index, turn in enumerate(record["turns"], 1):
        print(f"\n--- turn {index} (finish_reason={turn.get('finish_reason')}) ---")
        print(turn["reply"])
        if "action" in turn:
            print(f">>> action reveal{tuple(turn['action'])} "
                  f"reward={turn.get('reward')} {turn.get('outcome', '')}")
        if "error" in turn:
            print(f">>> REJECTED: {turn['error']}")
    print(f"\n--- final board ---\n{record.get('final_board', '')}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    parser.add_argument("--seed", type=int, help="show every episode for this puzzle seed")
    parser.add_argument("--outcome", help="show only episodes with this outcome")
    args = parser.parse_args()

    records = load(args.path)
    if args.outcome:
        records = [r for r in records if r["outcome"] == args.outcome]
    if args.seed is not None:
        records = [r for r in records if r["puzzle_seed"] == args.seed]
        for record in records:
            show(record)
        return

    # No seed given: an index, so you can pick one to read in full.
    for record in records:
        turns = len(record["turns"])
        print(f"seed {record['puzzle_seed']:>6}  attempt {record['attempt']}  "
              f"{record['outcome']:<13} {turns} turn(s)  {record['seconds']}s")
    print(f"\n{len(records)} episodes; re-run with --seed <n> to read one in full")


if __name__ == "__main__":
    main()
