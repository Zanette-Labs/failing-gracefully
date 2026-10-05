from copy import deepcopy
import unittest

from game_analysis.naive_solver import NaiveSolver, provably_safe_squares
from game_analysis.heuristic_solver import HeuristicSolver
from game_analysis.build_comparison_dataset import (
    EASY,
    EASY_MAX_STEPS,
    EASY_TRIALS,
    classify_seed,
    easy_trial_seed,
    evaluate_seed,
)
from game_analysis.sample_puzzles import sample_puzzles
from minesweeper_env import MinesweeperEnv


class NaiveSolverTests(unittest.TestCase):
    def test_reported_count_and_choice_match_safe_set(self):
        env = MinesweeperEnv()
        observation, _ = env.reset(seed=0)
        safe, inferred_mines = provably_safe_squares(observation, env.mine_count)
        move = NaiveSolver(seed=123).choose(observation, env.mine_count)
        self.assertEqual(move.safe_square_count, len(safe))
        self.assertIn(move.chosen, safe)
        self.assertTrue(safe.isdisjoint(inferred_mines))

    def test_solver_never_hits_a_mine_on_sampled_puzzles(self):
        for puzzle_seed in range(5):
            with self.subTest(puzzle_seed=puzzle_seed):
                env = MinesweeperEnv()
                env.reset(seed=puzzle_seed)
                result = NaiveSolver(seed=99).solve(env)
                self.assertIn(result.outcome, {"success", "turn_limit"})
                self.assertEqual(result.success, result.outcome == "success")
                self.assertLessEqual(len(result.moves), env.max_turns)

    def test_samples_are_reproducible(self):
        first = sample_puzzles([0])[0]
        second = sample_puzzles([0])[0]
        self.assertEqual(first, second)
        self.assertEqual(len(first["board"]), 6)

    def test_heuristic_solver_never_hits_a_mine(self):
        for puzzle_seed in range(20):
            with self.subTest(puzzle_seed=puzzle_seed):
                env = MinesweeperEnv()
                env.reset(seed=puzzle_seed)
                result = HeuristicSolver(seed=99).solve(env)
                self.assertIn(result.outcome, {"success", "turn_limit"})

    def test_known_comparison_seed_qualifies(self):
        row = evaluate_seed(57)
        self.assertIsNotNone(row)
        self.assertEqual(row["classification"], "heuristic_not_naive")

    def test_known_heuristic_failure_seed_qualifies(self):
        self.assertEqual(classify_seed(50), "heuristic_failure")

    def test_known_easy_seed_qualifies(self):
        self.assertEqual(classify_seed(403), EASY)

    def test_easy_seeds_are_won_quickly_by_the_naive_baseline(self):
        """The split's promise: the weakest baseline wins fast however it walks."""
        for puzzle_seed in (403, 532, 1901):
            env = MinesweeperEnv()
            env.reset(seed=puzzle_seed)
            for trial in range(EASY_TRIALS):
                with self.subTest(puzzle_seed=puzzle_seed, trial=trial):
                    solver = NaiveSolver(seed=easy_trial_seed(puzzle_seed, trial))
                    result = solver.solve(deepcopy(env))
                    self.assertTrue(result.success)
                    self.assertLessEqual(len(result.moves), EASY_MAX_STEPS)

    def test_large_easy_split_extends_the_small_one(self):
        """easy_2000 comes from the same ascending scan, so easy_100 is its prefix."""
        from pathlib import Path
        here = Path(__file__).parent / "game_analysis"
        small = here.joinpath("easy_100_seeds.txt").read_text().split()
        large = here.joinpath("easy_2000_seeds.txt").read_text().split()
        self.assertEqual(len(large), 2000)
        self.assertEqual(large[:len(small)], small)
        self.assertEqual(len(set(large)), len(large))  # no duplicates

    def test_easy_split_does_not_overlap_the_harder_splits(self):
        """Every seed lands in at most one split, so the ladder stays disjoint."""
        for puzzle_seed in (403, 57, 50):
            with self.subTest(puzzle_seed=puzzle_seed):
                classification = classify_seed(puzzle_seed)
                self.assertEqual(classification == EASY, puzzle_seed == 403)


if __name__ == "__main__":
    unittest.main()
