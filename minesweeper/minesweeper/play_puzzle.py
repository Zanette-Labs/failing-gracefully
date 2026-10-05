"""Play the environment while seeing the same prompts an LLM would receive."""

from __future__ import annotations

import argparse
import re

from minesweeper_env import MinesweeperEnv


ACTION_PATTERN = re.compile(
    r"^\s*reveal\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$",
    re.IGNORECASE,
)

# Two ways of showing the model the same board. ``grid`` is the original
# unlabeled text board with 1-indexed coordinates. ``list`` prints the board as
# a Python list of lists and takes 0-indexed coordinates, so the model can
# address a cell as board[row][col] exactly as it would in code.
PROMPT_FORMATS = ("grid", "list")
DEFAULT_PROMPT_FORMAT = "grid"

# How the model's coordinates map onto the environment's 1-indexed ones.
_ORIGIN = {"grid": 1, "list": 0}


def _check_format(prompt_format: str) -> None:
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(
            f"unknown prompt format {prompt_format!r}; expected one of {PROMPT_FORMATS}"
        )


def render_board(env: MinesweeperEnv, prompt_format: str = DEFAULT_PROMPT_FORMAT) -> str:
    """Render the visible board in the requested prompt format."""
    _check_format(prompt_format)
    if prompt_format == "grid":
        return env.render(coordinates=False)
    rows = [
        "[" + ", ".join('"#"' if n is None else str(n) for n in row) + "]"
        for row in env.observation().visible
    ]
    return "board = [\n" + "".join(f"    {row},\n" for row in rows) + "]"


def to_env_coords(row: int, col: int, prompt_format: str = DEFAULT_PROMPT_FORMAT) -> tuple[int, int]:
    """Translate a model action into the environment's 1-indexed coordinates."""
    _check_format(prompt_format)
    shift = 1 - _ORIGIN[prompt_format]
    return row + shift, col + shift


def translate_error(message: str, row: int, col: int, prompt_format: str = DEFAULT_PROMPT_FORMAT) -> str:
    """Rewrite an environment error so it names the cell the way the model did.

    The environment reports 1-indexed ``(row, col)``; a ``list`` transcript
    should instead say ``board[row][col]`` in the model's 0-indexed terms.
    """
    _check_format(prompt_format)
    if prompt_format == "list":
        env_row, env_col = to_env_coords(row, col, prompt_format)
        message = message.replace(f"({env_row}, {env_col})", f"board[{row}][{col}]")
        # "rows are 1 to 6 and columns are 1 to 6" -> 0-indexed ranges
        message = re.sub(
            r"rows are 1 to (\d+) and columns are 1 to (\d+)",
            lambda m: f"rows are 0 to {int(m.group(1)) - 1} and columns are 0 to {int(m.group(2)) - 1}",
            message,
        )
    return message


def cell_name(row: int, col: int, prompt_format: str = DEFAULT_PROMPT_FORMAT) -> str:
    """Name a cell the way the model addresses it in this prompt format."""
    _check_format(prompt_format)
    return f"board[{row}][{col}]" if prompt_format == "list" else f"cell ({row}, {col})"


def repeat_notice(row: int, col: int, prompt_format: str = DEFAULT_PROMPT_FORMAT) -> str:
    """Feedback for a reveal of an already-visible cell (prepended to the next board)."""
    return (
        f"Your action reveal({row}, {col}) named {cell_name(row, col, prompt_format)}, which was "
        f'already revealed: nothing was uncovered and that turn is spent. Pick an unrevealed ("#") cell.'
    )


def system_prompt(
    env: MinesweeperEnv,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
    max_tokens: int | None = None,
) -> str:
    """Build the starting prompt, including fixed episode parameters.

    ``max_tokens`` is the per-turn reply budget the caller samples with; when
    given, the prompt states it so the model knows a cut-off reply is a loss.
    """
    _check_format(prompt_format)
    token_budget = (
        f" Each turn you cannot spend more than {max_tokens} tokens,\n"
        f"reasoning included: a reply cut off at that limit loses the episode."
        if max_tokens is not None else ""
    )
    if prompt_format == "grid":
        coordinates = f"""Your only valid action is reveal(row, col). Rows and columns are numbered starting
at 1: row 1 is the top row, row {env.rows} is the bottom row, column 1 is the leftmost
column, and column {env.cols} is the rightmost. There is no row 0 and no column 0.
The board is printed without row or column labels, so count the rows and columns
yourself before answering.

A # is an unrevealed cell. A digit is the number of mines in its neighboring
cells."""
    else:
        coordinates = f"""Your only valid action is reveal(row, col). The board is given as a Python list of
lists named board, and board[row][col] is the cell you reveal with reveal(row, col).
Indices start at 0: board[0] is the top row, board[{env.rows - 1}] is the bottom row,
board[row][0] is the leftmost cell of a row, and board[row][{env.cols - 1}] is the
rightmost. There is no row {env.rows} and no column {env.cols}.

A "#" entry is an unrevealed cell. An integer is the number of mines in its
neighboring cells."""
    return f"""You are playing Minesweeper on a {env.rows} by {env.cols} board that \
contains {env.mine_count} mines.

{coordinates} The initial reveal has already been made and zero cells use normal flood fill.
You have at most {env.max_turns} turns to reveal every non-mine cell.{token_budget}

Keep your reasoning brief. The last line of your reply must be exactly one action,
in this format:
reveal(row, col)

Only that last line is read, so nothing may follow it. There are no second chances:
a reply with no action on its final line or a cell outside the board loses the
episode outright, exactly as revealing a mine would. Revealing a cell that is
already visible does not lose the episode, but it uncovers nothing and still
uses up one of your turns."""


def model_prompt(env: MinesweeperEnv, prompt_format: str = DEFAULT_PROMPT_FORMAT) -> str:
    """Build the user message supplied to the model for the current turn."""
    _check_format(prompt_format)
    obs = env.observation()
    if prompt_format == "grid":
        reminder = (
            "You must uncover all the safe cells before time runs out. Rows are numbered 1 "
            f"(top) to {env.rows} and columns 1 (left) to {env.cols}."
        )
    else:
        reminder = (
            "You must uncover all the safe cells before time runs out. Answer with the "
            f"0-indexed row and column, so reveal(row, col) reveals board[row][col]; "
            f"rows and columns each run from 0 to {env.rows - 1}."
        )
    return (
        f"Turns remaining: {env.max_turns - obs.steps}\n\n"
        f"{render_board(env, prompt_format)}\n\n"
        f"{reminder}"
    )


def parse_action(text: str) -> tuple[int, int]:
    """Parse exactly the action syntax advertised in the prompt."""
    match = ACTION_PATTERN.fullmatch(text)
    if not match:
        raise ValueError("expected reveal(row, col), for example reveal(2, 5)")
    return int(match.group(1)), int(match.group(2))


def play(seed: int | None = None, prompt_format: str = DEFAULT_PROMPT_FORMAT) -> bool:
    env = MinesweeperEnv()
    _, info = env.reset(seed=seed)

    print("=== SYSTEM PROMPT ===")
    print(system_prompt(env, prompt_format))

    while True:
        print("\n=== USER PROMPT ===")
        print(model_prompt(env, prompt_format))
        try:
            raw = input("\nYour model response: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGame stopped.")
            return False

        if raw.strip().lower() in {"quit", "exit"}:
            print("Game stopped.")
            return False

        try:
            model_row, model_col = parse_action(raw)
            row, col = to_env_coords(model_row, model_col, prompt_format)
            observation, reward, done, result = env.reveal(row, col)
        except ValueError as exc:
            print(f"\nInvalid response: {exc}")
            continue

        if done:
            print("\n=== FINAL BOARD ===")
            print(env.render(coordinates=False))
            if observation.success:
                print(f"\nSuccess! You cleared the board in {result['steps']} moves.")
                return True
            if result["outcome"] == "turn_limit":
                print("\nOut of turns. The board was not cleared.")
            else:
                print(f"\nBoom! reveal({model_row}, {model_col}) was a mine.")
            return False
        print(f"\nAction accepted (reward: {reward:g}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Minesweeper using the exact prompts and actions of an LLM agent."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="reproduce a puzzle with a particular integer seed",
    )
    parser.add_argument(
        "--format", dest="prompt_format", choices=PROMPT_FORMATS,
        default=DEFAULT_PROMPT_FORMAT,
        help="grid: unlabeled text board, 1-indexed; list: Python list of lists, 0-indexed",
    )
    args = parser.parse_args()
    play(args.seed, args.prompt_format)


if __name__ == "__main__":
    main()
