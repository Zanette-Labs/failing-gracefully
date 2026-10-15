"""A simple solver that randomly reveals one provably safe cell at a time."""

from __future__ import annotations

from dataclasses import dataclass
import random

from minesweeper_env import Cell, MinesweeperEnv, Observation


class NoSafeMoveError(RuntimeError):
    """Raised when the visible clues do not prove that any hidden cell is safe."""


@dataclass(frozen=True)
class Move:
    """One solver decision, including how many safe choices were available."""

    safe_square_count: int
    chosen: Cell


@dataclass(frozen=True)
class SolveResult:
    success: bool
    outcome: str
    moves: tuple[Move, ...]


def _neighbors(cell: Cell, rows: int, cols: int) -> set[Cell]:
    """Neighbors of a 1-indexed cell, clipped to the board."""
    row, col = cell
    return {
        (r, c)
        for r in range(max(1, row - 1), min(rows, row + 1) + 1)
        for c in range(max(1, col - 1), min(cols, col + 1) + 1)
        if (r, c) != cell
    }


def provably_safe_squares(
    observation: Observation, mine_count: int
) -> tuple[set[Cell], set[Cell]]:
    """Return safe hidden cells and mines inferred solely from visible clues.

    This applies the two elementary Minesweeper rules until stable:

    * if a clue's remaining hidden neighbors must all be mines, mark them mines;
    * if all of a clue's mines are accounted for, its other neighbors are safe.

    The total mine count is also used: once every mine has been inferred, every
    other hidden square is safe.
    """
    hidden = {
        (r, c)
        for r in range(1, observation.rows + 1)
        for c in range(1, observation.cols + 1)
        if observation.at(r, c) is None
    }
    inferred_mines: set[Cell] = set()
    safe: set[Cell] = set()

    changed = True
    while changed:
        changed = False
        for r in range(1, observation.rows + 1):
            for c in range(1, observation.cols + 1):
                clue = observation.at(r, c)
                if clue is None:
                    continue
                neighbors = _neighbors((r, c), observation.rows, observation.cols)
                unknown = (neighbors & hidden) - inferred_mines
                remaining = clue - len(neighbors & inferred_mines)
                if remaining < 0 or remaining > len(unknown):
                    raise ValueError("visible board contains inconsistent clues")
                if unknown and remaining == len(unknown):
                    new_mines = unknown - inferred_mines
                    if new_mines:
                        inferred_mines.update(new_mines)
                        safe.difference_update(new_mines)
                        changed = True
                elif unknown and remaining == 0:
                    safe.update(unknown)

        if len(inferred_mines) > mine_count:
            raise ValueError("visible board implies more mines than mine_count")
        if len(inferred_mines) == mine_count:
            safe.update(hidden - inferred_mines)

    return safe - inferred_mines, inferred_mines


class NaiveSolver:
    """Choose uniformly at random among every currently provable safe square."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose(self, observation: Observation, mine_count: int) -> Move:
        safe, _ = provably_safe_squares(observation, mine_count)
        if not safe:
            raise NoSafeMoveError("no hidden square can currently be proven safe")
        choices = sorted(safe)
        return Move(len(choices), self._rng.choice(choices))

    def solve(self, env: MinesweeperEnv) -> SolveResult:
        moves: list[Move] = []
        while not env.observation().terminated:
            move = self.choose(env.observation(), env.mine_count)
            moves.append(move)
            _, _, _, info = env.step(move.chosen)
        return SolveResult(bool(info["success"]), str(info["outcome"]), tuple(moves))


def main() -> None:
    """Run the solver on a reproducible default puzzle."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle-seed", type=int, default=0)
    parser.add_argument("--solver-seed", type=int, default=0)
    args = parser.parse_args()

    env = MinesweeperEnv()
    env.reset(seed=args.puzzle_seed)
    print(env.render())
    result = NaiveSolver(seed=args.solver_seed).solve(env)
    for index, move in enumerate(result.moves, 1):
        print(
            f"move {index}: {move.safe_square_count} safe square(s); "
            f"reveal{move.chosen}"
        )
    print(f"{result.outcome} in {len(result.moves)} moves")


if __name__ == "__main__":
    main()
