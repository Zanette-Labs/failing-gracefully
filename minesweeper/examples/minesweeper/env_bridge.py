"""Bridge between miles and the standalone no-guess Minesweeper environment.

The environment is bundled at ``minesweeper/`` in this release (override with
``$MINESWEEPER_DIR``) and is imported unchanged:
``minesweeper_env.MinesweeperEnv`` plus the exact prompt/action-parsing helpers
from ``play_puzzle.py`` (``system_prompt``, ``model_prompt``, ``parse_action``,
``to_env_coords``, ``translate_error``), so training sees byte-identical prompts
to the vLLM harness (``play_with_model.py``). The only thing re-implemented here
is that harness's ``extract_action`` (its module imports ``openai``, which is
not in the miles image).

The board can be shown in either of the harness's prompt formats, chosen per
row through ``metadata.prompt_format`` (written by prepare_data.py ``--format``):
``grid`` (unlabeled text board, 1-indexed actions) or ``list`` (a Python list
of lists, 0-indexed actions where ``reveal(r, c)`` reveals ``board[r][c]``).
Actions are converted to the env's 1-indexed coordinates at the boundary and
recorded in the model's own convention.

``MinesweeperGame`` wraps one episode with the SAME rules as
``play_with_model.play_episode``: an unparseable reply, an out-of-bounds reveal,
or a reply cut off by the per-turn length cap ends the episode as a loss,
exactly like revealing a mine. Revealing a cell that is ALREADY visible is
not a loss: the env spends the turn without uncovering anything, and the next
user turn carries a notice above the (unchanged) board.
"""

from __future__ import annotations

import os
import re
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MINESWEEPER_DIR = os.environ.get(
    "MINESWEEPER_DIR", os.path.join(_REPO_ROOT, "minesweeper")
)
if MINESWEEPER_DIR not in sys.path:
    sys.path.append(MINESWEEPER_DIR)

from minesweeper_env import MinesweeperEnv  # noqa: E402
from play_puzzle import (  # noqa: E402
    DEFAULT_PROMPT_FORMAT,
    PROMPT_FORMATS,
    model_prompt,
    parse_action,
    repeat_notice,
    system_prompt,
    to_env_coords,
    translate_error,
)

# Same fallback as play_with_model.py: a chatty last line still counts if it
# contains a reveal(...); the LAST one on that line wins.
_LOOSE_ACTION_RE = re.compile(r"reveal\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE)

# Board defaults (== MinesweeperEnv defaults); prepare_data.py writes them into
# each row's metadata so a run can change the board without touching code.
# min_steps/max_steps bound the no-guess solution the generator accepts; the env
# requires min_steps <= max_steps <= max_turns, and they change which board a
# seed yields, so a shorter turn budget (e.g. 8 or 4) needs seed splits built
# for that board. Rows without min_steps/max_steps (older files) get them capped
# at the budget, matching the seed builder's defaults.
DEFAULT_BOARD = {"rows": 6, "cols": 6, "mines": 6, "min_steps": 6, "max_steps": 12, "max_turns": 12}

# Every way an episode can end, split out so wandb shows WHICH failure mode
# dominates (see log_utils.py). success earns 1; turn_limit earns the graceful
# reward (see episode_reward); everything else earns 0.
OUTCOMES = (
    "success",
    "mine",
    "turn_limit",
    "illegal_move",
    "unparseable",
    "truncated",
    "context_exhausted",
    "aborted",
)

# Outcomes that count as "failed gracefully": the turn budget ran out and no
# harmful move was made. The env only reports turn_limit when the final reveal
# was safe, so mine / illegal_move are excluded by construction. Wasted turns
# (already-visible reveals) are not harmful, so they do not disqualify.
# Invalid replies (unparseable / truncated) and framework endings stay at 0 so
# the policy cannot earn the graceful reward by stalling.
GRACEFUL_OUTCOMES = ("turn_limit",)


def episode_reward(outcome: str, graceful_reward: float = 0.0) -> float:
    """Terminal reward: 1 for a full solve, ``graceful_reward`` (0 <= c < 1) for
    running out of turns without a harmful move, 0 for everything else."""
    if outcome == "success":
        return 1.0
    if outcome in GRACEFUL_OUTCOMES:
        return float(graceful_reward)
    return 0.0


def extract_action(text: str) -> tuple[int, int]:
    """Parse the action from the reply's last non-empty line (port of
    play_with_model.extract_action)."""
    lines = [line for line in text.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    try:
        return parse_action(last)
    except ValueError:
        matches = _LOOSE_ACTION_RE.findall(last)
        if not matches:
            raise ValueError(
                "expected reveal(row, col) on the last line, for example reveal(2, 5)"
            ) from None
        row, col = matches[-1]
        return int(row), int(col)


def make_env(meta: dict) -> MinesweeperEnv:
    board = {k: int(meta.get(k, v)) for k, v in DEFAULT_BOARD.items()}
    # Older rows carry only max_turns; keep them valid when it is below 12.
    if "max_steps" not in meta:
        board["max_steps"] = min(board["max_steps"], board["max_turns"])
    if "min_steps" not in meta:
        board["min_steps"] = min(board["min_steps"], board["max_steps"])
    return MinesweeperEnv(
        rows=board["rows"], cols=board["cols"], mines=board["mines"],
        min_steps=board["min_steps"], max_steps=board["max_steps"], max_turns=board["max_turns"],
    )


class MinesweeperGame:
    """One episode: holds the env, renders prompts, judges replies."""

    def __init__(self, meta: dict, max_tokens: int | None = None):
        # Per-turn reply budget the caller samples with; stated in the system prompt.
        self.max_tokens = max_tokens
        self.seed = int(meta["seed"])
        self.prompt_format = str(meta.get("prompt_format", DEFAULT_PROMPT_FORMAT))
        if self.prompt_format not in PROMPT_FORMATS:
            raise ValueError(
                f"metadata.prompt_format={self.prompt_format!r}; expected one of {PROMPT_FORMATS}"
            )
        self.env = make_env(meta)
        self.env.reset(seed=self.seed)
        self.max_turns = self.env.max_turns
        # Reveals of already-visible cells (each spends a turn, uncovers nothing).
        self.repeats = 0
        self.outcome = "ongoing"
        self.error: str | None = None
        # Feedback to prepend to the next user turn (set by a repeated reveal).
        self.notice: str | None = None
        # In the model's coordinates (0-indexed for ``list``), like the harness.
        self.actions: list[tuple[int, int]] = []

    @property
    def system_prompt(self) -> str:
        return system_prompt(self.env, self.prompt_format, self.max_tokens)

    def user_prompt(self) -> str:
        board = model_prompt(self.env, self.prompt_format)
        if self.notice is None:
            return board
        notice, self.notice = self.notice, None
        return f"{notice}\n\n{board}"

    @property
    def done(self) -> bool:
        return self.outcome != "ongoing"

    @property
    def success(self) -> bool:
        return self.outcome == "success"

    @property
    def steps(self) -> int:
        return self.env.observation().steps

    def act(self, reply: str, finish_reason: str) -> bool:
        """Apply one model reply; return True iff the episode is over.

        Mirrors play_with_model.play_episode turn-for-turn: length cut ->
        ``truncated``; no action on the last line -> ``unparseable``; env
        ValueError (out of bounds) -> ``illegal_move``; otherwise the env's own
        outcome (``ongoing`` / ``success`` / ``mine`` / ``turn_limit``). An
        already-visible cell is accepted by the env as a wasted turn; a notice
        is queued for the next user prompt.
        """
        assert not self.done, "episode is over"
        if finish_reason == "length":
            self.outcome, self.error = "truncated", "response cut off before a complete action"
            return True
        try:
            row, col = extract_action(reply)
        except ValueError as exc:
            self.outcome, self.error = "unparseable", str(exc)
            return True
        try:
            _obs, _reward, done, info = self.env.reveal(
                *to_env_coords(row, col, self.prompt_format)
            )
        except ValueError as exc:
            self.outcome = "illegal_move"
            self.error = translate_error(str(exc), row, col, self.prompt_format)
            return True
        self.actions.append((row, col))
        if info.get("already_revealed"):
            self.repeats += 1
            self.notice = repeat_notice(row, col, self.prompt_format)
        if done:
            self.outcome = info["outcome"]
        return done

    def end_early(self, outcome: str) -> None:
        """Terminate for a reason outside the game (context exhausted / abort)."""
        assert outcome in OUTCOMES
        self.outcome = outcome
