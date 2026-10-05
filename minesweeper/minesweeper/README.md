# No-guess Minesweeper environment

A dependency-free Python environment intended for LLM agents. Every generated
episode:

- begins after a safe zero-cell opening has already been revealed;
- is certified solvable by ordinary adjacent-number deductions without guessing;
- takes one known logical solution between 6 and 12 `reveal(row, col)` actions;
- uses standard zero-cell flood fill; and
- succeeds as soon as all non-mine cells are visible.

```python
from minesweeper_env import MinesweeperEnv

env = MinesweeperEnv()
observation, info = env.reset(seed=42)
print(env.render())

observation, reward, terminated, info = env.reveal(3, 4)  # 1-indexed
print(observation.at(3, 4))
```

Coordinates are 1-indexed: the top-left cell is `(1, 1)` and the bottom-right cell
of the default board is `(6, 6)`. This applies to `reveal`, `step`, `info["opening"]`,
`solution_for_testing()`, and the labels printed by `render()`. `observation.visible`
stays a plain top-to-bottom grid, so address it with `observation.at(row, col)`
rather than subscripting it directly.

Cells in `observation.visible` are `None` when hidden and integers from 0 to 8
when revealed. A safe move rewards `0`, winning rewards `1`, and revealing a mine
terminates with reward `-1`. Invalid or repeated reveals raise `ValueError` and do
not consume a step. The opening move does not count as an agent step.
For Gym-style loops, `step((row, col))` is an equivalent spelling of `reveal`.
The default episode limit is 12 agent turns. Reaching it without clearing every
safe cell ends the episode with reward `-1`; `info["turns_remaining"]` reports the
remaining budget after every action.

The default board is 6x6 with 6 mines. Generation is reproducible by seed. The
`solution_for_testing()` method exposes the validator's reveal trace for automated
environment checks; training agents should not receive it.

Run tests with:

```sh
python -m unittest -v
```

The interactive player presents an unlabeled grid, tells the model the board size,
the mine count, and the 1-indexed coordinate convention in its starting prompt, and
includes the remaining turn budget in every turn:

```sh
python play_puzzle.py --seed 42
```

## Play as the model

Run the interactive prompt simulator:

```sh
python play_puzzle.py --seed 42
```

It displays the system prompt once and the model's user prompt after every move.
Respond exactly as the model would, for example `reveal(3, 4)`. Type `quit` to
stop. Using the same seed reproduces the same puzzle.

## Prompt formats

The prompt can show the board two ways; every entry point takes `--format`:

| format | board in the prompt | coordinates the model uses |
| --- | --- | --- |
| `grid` (default) | unlabeled text rows, `#` for hidden | 1-indexed, `reveal(1, 1)` is top-left |
| `list` | a Python `board = [[...], ...]` literal, `"#"` for hidden | 0-indexed, `reveal(r, c)` reveals `board[r][c]` |

```sh
python play_puzzle.py --seed 57 --format list
python run_batch.py --seed-file game_analysis/easy_100_seeds.txt \
    --puzzles 50 --attempts 1 --max-tokens 8192 --format list
```

The harness translates `list` actions to the environment's 1-indexed
coordinates; transcripts record actions in the model's own convention and tag
each episode with its `prompt_format`.

## Difficulty splits

`game_analysis/` ships three seed-only splits over this same 6x6 board, ordered
easiest to hardest. Any of them can be pointed at the model harness:

```sh
python run_batch.py --seed-file game_analysis/easy_100_seeds.txt \
    --puzzles 100 --attempts 8
```

| split | file | how hard |
| --- | --- | --- |
| easy | `easy_100_seeds.txt` | naive baseline wins in <=5 of 12 turns, on all 8 trials |
| easy (large) | `easy_2000_seeds.txt` | same rule, 2,000 seeds; the first 100 are the split above |
| medium | `heuristic_not_naive_10k_seeds.txt` | naive baseline hits the turn limit, heuristic wins |
| hard | `heuristic_failures_1k_seeds.txt` | heuristic baseline hits the turn limit |

The medium split is the default seed file. See `game_analysis/README.md` for how
each split is selected and rebuilt.
