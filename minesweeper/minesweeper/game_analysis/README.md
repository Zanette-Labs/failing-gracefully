# Game analysis

`sample_puzzles.json` contains five reproducible initial observations sampled
from the simulator with puzzle seeds 0 through 4. Regenerate it with:

```sh
python -m game_analysis.sample_puzzles
```

The naïve solver uses only the public observation and the public mine count. On
each turn it finds every hidden square proven safe by the basic adjacent-number
rules, counts those squares, and uniformly selects one using a seeded random
number generator:

```sh
python -m game_analysis.naive_solver --puzzle-seed 0 --solver-seed 0
```

It deliberately does not inspect `_mines` or call `solution_for_testing()`.

## Slightly smarter baseline

The heuristic solver makes the same safe deductions, but ranks the choices. It
uses nearby visible clues to estimate the mine pressure around each candidate,
prefers the candidate most likely to be a zero-cell flood reveal, and favors the
least-explored neighborhood as a tie-breaker:

```sh
python -m game_analysis.heuristic_solver --puzzle-seed 0 --solver-seed 0
```

On puzzle seeds 0 through 999, with each solver seeded by the puzzle seed, this
improved wins from 978 to 996 and reduced mean turns from 8.55 to 7.92. Both
baselines reveal only logically proven-safe cells; failures are turn-limit
failures, not mine hits.

## Comparison dataset

The dataset has three plain-text splits, with one integer seed per line. They
form a difficulty ladder over the same 6x6 board with 6 mines and 12 turns, so
any of them drops straight into the harness with `--seed-file`:

- `easy_100_seeds.txt`: 100 seeds the naïve solver wins in at most 5 of the 12
  turns on all 8 trial seeds. The easiest rung.
- `easy_2000_seeds.txt`: the same rule carried on to 2,000 seeds, for RL
  training that needs more distinct easy boards. Its first 100 lines are exactly
  `easy_100_seeds.txt`, because both come from the same ascending scan.
- `heuristic_not_naive_10k_seeds.txt`: 10,000 seeds on which the naïve solver
  reaches the turn limit and the heuristic solver succeeds.
- `heuristic_failures_1k_seeds.txt`: 1,000 seeds on which the heuristic solver
  reaches the turn limit. The hardest rung.

The splits are disjoint: a seed is only classified `easy` when *both* baselines
already succeed on it, which is exactly the case the other two splits exclude.

### What makes the easy split easy

The naïve solver chooses uniformly among the cells it can currently prove safe,
so re-running it under different seeds walks different paths through the same
board. A seed qualifies only when all 8 of those walks win, and none needs more
than 5 turns. That rules out boards that merely happen to suit one lucky walk,
and it leaves at least 7 of the 12 turns unused, so an agent has room to waste a
move. In practice these boards open with a large flood, and the hidden cells that
remain sit in one small cluster:

```
0 0 0 0 0 0        # # # # # #
0 0 0 0 1 1        # # # # # #
0 0 1 2 3 #        # 2 1 2 # #
0 0 1 # # #        # 2 0 1 2 #
2 3 3 # # #        1 1 0 0 1 #
# # # # # #        0 0 0 0 1 #

easy seed 532      heuristic_not_naive seed 57
```

About 1 seed in 317 qualifies, so the split is built by scanning far more seeds
than it keeps.

Boards, mine locations, and traces are not stored. A heuristic failure means a
failure by this particular seeded policy within the turn budget, not that the
underlying board is logically unsolvable.

The solver seed equals the puzzle seed for a simple reproducibility rule; the
easy split's trial `t` uses `puzzle_seed * 1000 + t`. Each split was obtained by
scanning puzzle seeds from 0 in ascending order and retaining the requested
number of each class. Rebuild everything, or just one split, with:

```sh
python -m game_analysis.build_comparison_dataset --workers 8
python -m game_analysis.build_comparison_dataset --only easy --workers 8
```

The builder always writes the easy split to `easy_100_seeds.txt`, so the
2,000-seed file was produced by building into a separate directory and copying
the result. At about 1 easy seed per 317 scanned it takes roughly 8 minutes on
16 cores:

```sh
python -m game_analysis.build_comparison_dataset --only easy --easy-count 2000 \
    --workers 16 --batch-size 20000 --output-dir /some/scratch
cp /some/scratch/easy_100_seeds.txt game_analysis/easy_2000_seeds.txt
```

Generation is deterministic; `comparison_dataset_manifest.json` records the
selection rules and, because the splits fill at very different rates, the seed
range scanned for each one.

### Other turn budgets

The splits above are for the 12-turn board. `--max-turns N` builds the same
three splits under an N-turn budget. The environment caps the longest accepted
no-guess solution (`max_steps`) at the budget, so for N < 12 the generator
rejects boards it would otherwise keep and every seed yields a *different*
board: seed files for different budgets are unrelated, and each build records
its board in the manifest. `turns8/` holds the 8-turn build (medium 10,000,
hard 1,000, easy 2,000 with `easy_100_seeds.txt` copied to
`easy_2000_seeds.txt` as before; the easy rule is still "naive wins within 5
turns on all 8 trials"). The classes fill much faster than at 12 turns because
the naive solver runs out of turns far more often:

```sh
python -m game_analysis.build_comparison_dataset --max-turns 8 --easy-count 2000 \
    --workers 32 --batch-size 20000 --output-dir game_analysis/turns8
cp game_analysis/turns8/easy_100_seeds.txt game_analysis/turns8/easy_2000_seeds.txt
```

Below 6 turns the default shortest accepted solution (`min_steps` 6) no longer
fits, so the builder caps it at `max_steps` (override with `--min-steps`): a
4-turn board keeps only seeds whose no-guess solution is exactly 4 reveals,
i.e. no slack at all inside the budget. `turns4/` holds only the medium split
(10,000 seeds, ~11% of scanned seeds qualify); the easy rule "naive wins
within 5 turns" is meaningless there, so no easy or hard split was built:

```sh
python -m game_analysis.build_comparison_dataset --max-turns 4 --only heuristic_not_naive \
    --workers 32 --batch-size 20000 --output-dir game_analysis/turns4
```

To inspect any puzzle manually, take a line from the seed file and pass it to the
interactive player. For example, the first seed is 57:

```sh
python play_puzzle.py --seed 57
```
