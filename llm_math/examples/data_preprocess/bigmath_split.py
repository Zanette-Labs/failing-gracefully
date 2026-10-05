"""
Build a deduplicated train/test split of SynthLabsAI/Big-Math-RL-Verified and
push it to the Hub.

The source ships a single `train` split of 251,122 rows in which 7,526 problems
are repeated (up to 7x each). Duplicates are collapsed before splitting, so the
held-out problems cannot leak back into train.

    python examples/data_preprocess/bigmath_split.py
    python examples/data_preprocess/bigmath_split.py --dry_run
"""

import argparse
import re

import datasets

SOURCE_REPO = "SynthLabsAI/Big-Math-RL-Verified"


def norm_key(problem: str) -> str:
    """Dedup key: collapse whitespace and case. Catches 57 problems that differ
    from an earlier copy only in formatting."""
    return re.sub(r"\s+", " ", problem).strip().lower()


def dedup(dataset):
    """Drop repeated problems, keeping the first occurrence with its original text."""
    seen = set()
    keep = []
    for i, problem in enumerate(dataset["problem"]):
        k = norm_key(problem)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return dataset.select(keep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="daman1209arora/bigmath", help="Target Hub dataset repo")
    parser.add_argument("--n_test", type=int, default=200, help="Number of held-out problems")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the split")
    parser.add_argument("--private", action="store_true", help="Push as a private dataset")
    parser.add_argument("--dry_run", action="store_true", help="Build and report, but do not push")
    args = parser.parse_args()

    raw = datasets.load_dataset(SOURCE_REPO, trust_remote_code=True)["train"]
    print(f"source rows: {len(raw)}", flush=True)

    deduped = dedup(raw)
    print(f"deduped rows: {len(deduped)} (removed {len(raw) - len(deduped)})", flush=True)

    split = deduped.train_test_split(test_size=args.n_test, seed=args.seed, shuffle=True)
    dataset_dict = datasets.DatasetDict({"train": split["train"], "test": split["test"]})
    print(dataset_dict)

    if args.dry_run:
        print("--dry_run set, skipping push")
    else:
        dataset_dict.push_to_hub(args.repo, private=args.private)
        print(f"pushed to https://huggingface.co/datasets/{args.repo}")
