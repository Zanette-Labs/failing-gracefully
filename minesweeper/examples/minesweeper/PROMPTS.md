# Minesweeper training prompts

These are the messages sent to the model by the [eight-turn medium training recipe](../../runtime/training.sh). Each episode begins with one system message and one user message. After each valid reveal, the agent appends the model's reply and a new user message with the updated board. The model's native chat template wraps these messages; there are no example moves in the prompt.

The training boards have 6 rows, 6 columns, 6 mines, and an eight-turn limit. The recipe selects the `list` format, so actions use zero-indexed `board[row][col]` coordinates. The agent reconstructs the messages from each row's metadata; the `prompt` field in the training JSONL is not sent as a separate message. The [prompt formatter](../../minesweeper/play_puzzle.py) supplies the exact text below.

## System message

The following is the full message for `prompt_format="list"`, `max_turns=8`, and a 4,096-token per-turn reply limit:

```text
You are playing Minesweeper on a 6 by 6 board that contains 6 mines.

Your only valid action is reveal(row, col). The board is given as a Python list of
lists named board, and board[row][col] is the cell you reveal with reveal(row, col).
Indices start at 0: board[0] is the top row, board[5] is the bottom row,
board[row][0] is the leftmost cell of a row, and board[row][5] is the
rightmost. There is no row 6 and no column 6.

A "#" entry is an unrevealed cell. An integer is the number of mines in its
neighboring cells. The initial reveal has already been made and zero cells use normal flood fill.
You have at most 8 turns to reveal every non-mine cell. Each turn you cannot spend more than 4096 tokens,
reasoning included: a reply cut off at that limit loses the episode.

Keep your reasoning brief. The last line of your reply must be exactly one action,
in this format:
reveal(row, col)

Only that last line is read, so nothing may follow it. There are no second chances:
a reply with no action on its final line or a cell outside the board loses the
episode outright, exactly as revealing a mine would. Revealing a cell that is
already visible does not lose the episode, but it uncovers nothing and still
uses up one of your turns.
```

## User message

Here is an actual initial user message for seed `12438`, the first row in the shuffled training JSONL produced with seed `72`:

```text
Turns remaining: 8

board = [
    ["#", "#", "#", "#", "#", "#"],
    ["#", 2, 1, 2, 3, "#"],
    ["#", 2, 0, 0, 1, 1],
    ["#", 2, 1, 1, 0, 0],
    ["#", "#", "#", 1, 0, 0],
    ["#", "#", "#", 1, 0, 0],
]

You must uncover all the safe cells before time runs out. Answer with the 0-indexed row and column, so reveal(row, col) reveals board[row][col]; rows and columns each run from 0 to 5.
```

Every later user message has the same layout: `Turns remaining: N`, a blank line, the current `board = [...]`, a blank line, and the same final sentence. `N` decreases after each valid reveal. If the model reveals an already visible cell, the next user message starts with this notice, followed by a blank line and the usual board message (shown here for `reveal(2, 3)`):

```text
Your action reveal(2, 3) named board[2][3], which was already revealed: nothing was uncovered and that turn is spent. Pick an unrevealed ("#") cell.
```

The [training agent](agent.py) assembles the conversation, and [the environment bridge](env_bridge.py) chooses the prompt format and adds the repeated-reveal notice.
