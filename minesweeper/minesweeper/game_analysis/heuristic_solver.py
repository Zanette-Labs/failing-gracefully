"""A safer-cell ranking baseline built on top of the naïve deductions."""

from __future__ import annotations

import argparse

from minesweeper_env import Cell, MinesweeperEnv, Observation

from .naive_solver import Move, NaiveSolver, NoSafeMoveError, _neighbors, provably_safe_squares


def estimated_mine_risks(
    observation: Observation, mine_count: int, inferred_mines: set[Cell]
) -> dict[Cell, float]:
    """Estimate mine pressure for hidden, not-yet-classified cells.

    This is deliberately a cheap heuristic, not an exact probability model. For
    each cell it takes the strongest adjacent clue estimate. Unconstrained cells
    receive the board-wide remaining-mine rate.
    """
    hidden = {
        (r, c)
        for r in range(1, observation.rows + 1)
        for c in range(1, observation.cols + 1)
        if observation.at(r, c) is None
    }
    unknown = hidden - inferred_mines
    remaining_mines = mine_count - len(inferred_mines)
    prior = remaining_mines / len(unknown) if unknown else 0.0
    risks: dict[Cell, float] = {}

    for cell in unknown:
        local_estimates = []
        for clue_cell in _neighbors(cell, observation.rows, observation.cols):
            clue = observation.at(*clue_cell)
            if clue is None:
                continue
            neighbors = _neighbors(clue_cell, observation.rows, observation.cols)
            local_unknown = (neighbors & hidden) - inferred_mines
            if local_unknown:
                mines_left = clue - len(neighbors & inferred_mines)
                local_estimates.append(mines_left / len(local_unknown))
        risks[cell] = max(local_estimates, default=prior)
    return risks


class HeuristicSolver(NaiveSolver):
    """Reveal a proven-safe cell likely to expose the most useful territory."""

    def choose(self, observation: Observation, mine_count: int) -> Move:
        safe, inferred_mines = provably_safe_squares(observation, mine_count)
        if not safe:
            raise NoSafeMoveError("no hidden square can currently be proven safe")

        hidden = {
            (r, c)
            for r in range(1, observation.rows + 1)
            for c in range(1, observation.cols + 1)
            if observation.at(r, c) is None
        }
        risks = estimated_mine_risks(observation, mine_count, inferred_mines)

        def score(cell: Cell) -> tuple[float, int]:
            neighbors = _neighbors(cell, observation.rows, observation.cols)
            expected_adjacent_mines = len(neighbors & inferred_mines) + sum(
                risks.get(neighbor, 0.0)
                for neighbor in neighbors - inferred_mines
            )
            # Low mine pressure increases the chance of revealing a zero and
            # triggering flood fill. Favor less-explored areas on a tie.
            return -expected_adjacent_mines, -len(neighbors - hidden)

        best_score = max(score(cell) for cell in safe)
        best_cells = sorted(cell for cell in safe if score(cell) == best_score)
        return Move(len(safe), self._rng.choice(best_cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle-seed", type=int, default=0)
    parser.add_argument("--solver-seed", type=int, default=0)
    args = parser.parse_args()

    env = MinesweeperEnv()
    env.reset(seed=args.puzzle_seed)
    result = HeuristicSolver(seed=args.solver_seed).solve(env)
    for index, move in enumerate(result.moves, 1):
        print(
            f"move {index}: {move.safe_square_count} safe square(s); "
            f"reveal{move.chosen}"
        )
    print(f"{result.outcome} in {len(result.moves)} moves")


if __name__ == "__main__":
    main()
