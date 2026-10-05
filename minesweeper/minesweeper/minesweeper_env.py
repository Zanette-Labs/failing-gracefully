"""A small, dependency-free Minesweeper environment for language-model agents.

Boards are accepted only when a basic deterministic solver can finish them using
reveals alone.  The solver may *infer* mines, but flagging is not an environment
action.  The initial safe reveal is performed before the first observation.

Public coordinates are 1-indexed: the top-left cell is ``(1, 1)`` and the
bottom-right cell of a 6x6 board is ``(6, 6)``.  Cells are stored internally as
0-indexed pairs and converted at the public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Optional

Cell = tuple[int, int]


def _to_public(cell: Cell) -> Cell:
    """Convert an internal 0-indexed cell to the public 1-indexed pair."""
    return (cell[0] + 1, cell[1] + 1)


@dataclass(frozen=True)
class Observation:
    """A snapshot of the visible board.

    ``visible`` is a plain top-to-bottom grid, so ``visible[0][0]`` is the
    top-left cell.  Address cells by 1-indexed coordinates with ``at()``.
    """

    rows: int
    cols: int
    visible: tuple[tuple[Optional[int], ...], ...]
    steps: int
    terminated: bool
    success: bool

    def at(self, row: int, col: int) -> Optional[int]:
        """Return the clue at a 1-indexed cell, or ``None`` when it is hidden."""
        if not (1 <= row <= self.rows and 1 <= col <= self.cols):
            raise ValueError(f"cell ({row}, {col}) is outside the board")
        return self.visible[row - 1][col - 1]


class MinesweeperEnv:
    """No-guess Minesweeper with ``reveal(row, col)`` as its only action.

    ``reset(seed)`` generates a reproducible board, performs the first reveal,
    and returns an observation.  Revealing a cell that is already visible is
    not an error: it uncovers nothing but still spends a turn (``info`` then
    carries ``already_revealed=True``).  ``step((row, col))`` returns the familiar
    ``(observation, reward, terminated, info)`` tuple.  Coordinates are
    1-indexed, so valid rows run from 1 to ``rows`` and columns from 1 to
    ``cols``.
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 6,
        mines: int = 6,
        min_steps: int = 6,
        max_steps: int = 12,
        max_turns: Optional[int] = None,
        max_generation_attempts: int = 20_000,
    ) -> None:
        if rows < 2 or cols < 2:
            raise ValueError("rows and cols must each be at least 2")
        if not 0 < mines < rows * cols - 1:
            raise ValueError("mines must leave at least two safe cells")
        if not 1 <= min_steps <= max_steps:
            raise ValueError("expected 1 <= min_steps <= max_steps")
        if max_turns is not None and max_turns < max_steps:
            raise ValueError("max_turns must be at least max_steps")
        self.rows, self.cols, self.mine_count = rows, cols, mines
        self.min_steps, self.max_steps = min_steps, max_steps
        self.max_turns = max_steps if max_turns is None else max_turns
        self.max_generation_attempts = max_generation_attempts
        self._mines: set[Cell] = set()
        self._numbers: list[list[int]] = []
        self._revealed: set[Cell] = set()
        self._terminated = False
        self._success = False
        self._steps = 0
        self._outcome = "ongoing"
        self._solution: tuple[Cell, ...] = ()
        self._opening: Cell = (0, 0)

    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict]:
        rng = random.Random(seed)
        cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        for _ in range(self.max_generation_attempts):
            opening = rng.choice(cells)
            # A zero opening gives normal Minesweeper's useful initial flood.
            protected = {opening, *self._neighbors(opening)}
            candidates = [cell for cell in cells if cell not in protected]
            if len(candidates) < self.mine_count:
                continue
            mines = set(rng.sample(candidates, self.mine_count))
            numbers = self._make_numbers(mines)
            initially_revealed = self._flood(opening, mines, numbers)
            solution = self._logical_solution(initially_revealed, mines, numbers)
            if solution is not None and self.min_steps <= len(solution) <= self.max_steps:
                self._opening, self._mines, self._numbers = opening, mines, numbers
                self._revealed = set(initially_revealed)
                self._solution = tuple(solution)
                self._steps, self._terminated, self._success = 0, False, False
                self._outcome = "ongoing"
                return self.observation(), self._info()
        raise RuntimeError(
            "could not generate a no-guess board in the requested step range; "
            "increase max_generation_attempts or adjust board parameters"
        )

    def step(self, action: Cell) -> tuple[Observation, float, bool, dict]:
        """Apply the sole action type, as a 1-indexed ``(row, col)`` tuple."""
        if not (isinstance(action, tuple) and len(action) == 2):
            raise ValueError("action must be a (row, col) tuple")
        return self.reveal(action[0], action[1])

    def reveal(self, row: int, col: int) -> tuple[Observation, float, bool, dict]:
        """Reveal one 1-indexed cell; return ``(observation, reward, done, info)``."""
        if self._terminated:
            raise RuntimeError("episode is over; call reset()")
        if not isinstance(row, int) or not isinstance(col, int):
            raise ValueError("row and col must be integers")
        cell = (row - 1, col - 1)
        if not self._in_bounds(cell):
            raise ValueError(
                f"cell ({row}, {col}) is outside the board; rows are 1 to "
                f"{self.rows} and columns are 1 to {self.cols}"
            )

        self._steps += 1
        if cell in self._revealed:
            # A wasted turn, not an error: nothing is uncovered, the turn is
            # spent, and the turn limit applies as usual.
            turn_limit_reached = self._steps >= self.max_turns
            self._terminated = turn_limit_reached
            if turn_limit_reached:
                self._outcome = "turn_limit"
            reward = -1.0 if turn_limit_reached else 0.0
            return self.observation(), reward, self._terminated, self._info() | {"already_revealed": True}
        if cell in self._mines:
            self._terminated, self._success = True, False
            self._outcome = "mine"
            reward = -1.0
        else:
            self._revealed.update(self._flood(cell, self._mines, self._numbers))
            safe_count = self.rows * self.cols - self.mine_count
            self._success = len(self._revealed) == safe_count
            turn_limit_reached = self._steps >= self.max_turns
            self._terminated = self._success or turn_limit_reached
            if self._success:
                self._outcome = "success"
            elif turn_limit_reached:
                self._outcome = "turn_limit"
            reward = 1.0 if self._success else (-1.0 if turn_limit_reached else 0.0)
        return self.observation(), reward, self._terminated, self._info()

    def observation(self) -> Observation:
        visible = tuple(
            tuple(self._numbers[r][c] if (r, c) in self._revealed else None
                  for c in range(self.cols))
            for r in range(self.rows)
        )
        return Observation(self.rows, self.cols, visible, self._steps,
                           self._terminated, self._success)

    def render(self, coordinates: bool = True) -> str:
        """Return a text board; ``#`` means unrevealed."""
        obs = self.observation()
        if not coordinates:
            return "\n".join(
                " ".join("#" if n is None else str(n) for n in row)
                for row in obs.visible
            )
        header = "    " + " ".join(str(c) for c in range(1, self.cols + 1))
        lines = [header]
        for r, row in enumerate(obs.visible, start=1):
            lines.append(f"{r:>2}  " + " ".join("#" if n is None else str(n) for n in row))
        return "\n".join(lines)

    @property
    def opening(self) -> Cell:
        """The pre-revealed opening cell, as 1-indexed coordinates."""
        return _to_public(self._opening)

    def solution_for_testing(self) -> tuple[Cell, ...]:
        """Return one no-guess reveal trace, 1-indexed (used by evaluation tests)."""
        return tuple(_to_public(cell) for cell in self._solution)

    def _info(self) -> dict:
        return {
            "steps": self._steps,
            "turns_remaining": max(0, self.max_turns - self._steps),
            "success": self._success,
            "outcome": self._outcome,
            "opening": _to_public(self._opening),
        }

    def _in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def _neighbors(self, cell: Cell) -> set[Cell]:
        r, c = cell
        return {(rr, cc) for rr in range(max(0, r - 1), min(self.rows, r + 2))
                for cc in range(max(0, c - 1), min(self.cols, c + 2))
                if (rr, cc) != cell}

    def _make_numbers(self, mines: set[Cell]) -> list[list[int]]:
        return [[sum(n in mines for n in self._neighbors((r, c)))
                 for c in range(self.cols)] for r in range(self.rows)]

    def _flood(self, start: Cell, mines: set[Cell], numbers: list[list[int]]) -> set[Cell]:
        """Reveal a cell and recursively reveal zero regions plus their border."""
        if start in mines:
            return set()
        revealed, stack = set(), [start]
        while stack:
            cell = stack.pop()
            if cell in revealed or cell in mines:
                continue
            revealed.add(cell)
            if numbers[cell[0]][cell[1]] == 0:
                stack.extend(self._neighbors(cell) - revealed)
        return revealed

    def _logical_solution(
        self, initial: Iterable[Cell], mines: set[Cell], numbers: list[list[int]]
    ) -> Optional[list[Cell]]:
        """Solve with local number constraints; return safe clicks or None.

        Mine locations are used only to confirm that deductions are sound, never
        to choose a move.  Inferred mines are bookkeeping, not player actions.
        """
        revealed, inferred, moves = set(initial), set(), []
        safe_total = self.rows * self.cols - len(mines)
        while len(revealed) < safe_total:
            changed = True
            safe_candidates: set[Cell] = set()
            while changed:
                changed = False
                for cell in tuple(revealed):
                    hidden = self._neighbors(cell) - revealed - inferred
                    remaining = numbers[cell[0]][cell[1]] - len(self._neighbors(cell) & inferred)
                    if remaining < 0 or remaining > len(hidden):
                        return None
                    if hidden and remaining == len(hidden):
                        new_mines = hidden - inferred
                        if not new_mines <= mines:
                            return None
                        if new_mines:
                            inferred.update(new_mines)
                            changed = True
                    elif hidden and remaining == 0:
                        safe_candidates.update(hidden)
            safe_candidates -= revealed | inferred
            if not safe_candidates:
                return None
            # One reveal per environment step; flood fill may reveal other candidates.
            click = min(safe_candidates)
            if click in mines:
                return None
            moves.append(click)
            revealed.update(self._flood(click, mines, numbers))
        return moves
