import unittest

from minesweeper_env import MinesweeperEnv
from play_puzzle import model_prompt, parse_action, system_prompt


class MinesweeperEnvTests(unittest.TestCase):
    def test_generated_board_has_preplayed_opening_and_valid_solution(self):
        env = MinesweeperEnv()
        obs, info = env.reset(seed=7)
        self.assertEqual(obs.steps, 0)
        self.assertIsNotNone(obs.at(*info["opening"]))
        solution = env.solution_for_testing()
        self.assertGreaterEqual(len(solution), 6)
        self.assertLessEqual(len(solution), 12)
        for cell in solution:
            obs, reward, done, info = env.reveal(*cell)
        self.assertTrue(done)
        self.assertTrue(obs.success)
        self.assertEqual(reward, 1.0)

    def test_seed_is_reproducible(self):
        a, b = MinesweeperEnv(), MinesweeperEnv()
        oa, ia = a.reset(seed=123)
        ob, ib = b.reset(seed=123)
        self.assertEqual(oa, ob)
        self.assertEqual(ia, ib)
        self.assertEqual(a.solution_for_testing(), b.solution_for_testing())

    def test_only_reveal_semantics_and_errors(self):
        env = MinesweeperEnv()
        obs, _ = env.reset(seed=1)
        revealed = next((r, c) for r, row in enumerate(obs.visible, start=1)
                        for c, value in enumerate(row, start=1) if value is not None)
        # Re-revealing a visible cell is a wasted turn, not an error.
        obs2, reward, done, info = env.step(revealed)
        self.assertEqual((obs2.visible, reward, done, obs2.steps), (obs.visible, 0.0, False, 1))
        self.assertTrue(info["already_revealed"])
        with self.assertRaises(ValueError):
            env.step((-1, 1))
        # Coordinates start at 1, so the old zero-indexed corner is off the board.
        with self.assertRaises(ValueError):
            env.step((0, 0))
        with self.assertRaises(ValueError):
            env.step((env.rows + 1, 1))

    def test_coordinates_are_one_indexed(self):
        env = MinesweeperEnv()
        obs, info = env.reset(seed=7)
        self.assertTrue(all(1 <= r <= env.rows and 1 <= c <= env.cols
                            for r, c in env.solution_for_testing()))
        row, col = info["opening"]
        self.assertTrue(1 <= row <= env.rows and 1 <= col <= env.cols)
        self.assertEqual(env.opening, info["opening"])
        # at() and the raw grid agree once the one-cell offset is applied.
        self.assertEqual(obs.at(row, col), obs.visible[row - 1][col - 1])
        self.assertEqual(env.render().splitlines()[0].split(),
                         [str(c) for c in range(1, env.cols + 1)])

    def test_llm_action_parser_and_prompt(self):
        self.assertEqual(parse_action("reveal(2, 5)"), (2, 5))
        self.assertEqual(parse_action(" REVEAL ( 0, 1 ) "), (0, 1))
        for invalid in ("2, 5", "flag(2, 5)", "reveal(2)", "reveal(2, 5) extra"):
            with self.assertRaises(ValueError):
                parse_action(invalid)

        env = MinesweeperEnv()
        env.reset(seed=42)
        prompt = model_prompt(env)
        self.assertIn("Turns remaining: 12", prompt)
        self.assertIn("#", prompt)
        self.assertNotIn("solution", prompt.lower())
        self.assertNotIn("0 1 2 3 4 5", prompt)
        self.assertIn("numbered 1 (top) to 6", prompt)
        self.assertIn("contains 6 mines", system_prompt(env))
        self.assertIn("at most 12 turns", system_prompt(env))
        self.assertNotIn("without guessing", system_prompt(env))
        self.assertIn("numbered starting\nat 1", system_prompt(env))
        self.assertIn("no row 0 and no column 0", system_prompt(env))
        self.assertNotIn("tokens", system_prompt(env))
        self.assertIn("cannot spend more than 4096 tokens", system_prompt(env, max_tokens=4096))

    def test_turn_limit_is_enforced(self):
        env = MinesweeperEnv(max_turns=12)
        env.reset(seed=42)
        safe_trace = env.solution_for_testing()
        # Consume otherwise valid turns by choosing safe cells in a deliberately
        # inefficient order is impossible due to repeated-reveal validation, so
        # use a board whose certified solution reaches its configured limit.
        env.max_turns = 1
        obs, reward, done, info = env.reveal(*safe_trace[0])
        self.assertTrue(done)
        self.assertFalse(obs.success)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info["turns_remaining"], 0)
        self.assertEqual(info["outcome"], "turn_limit")


if __name__ == "__main__":
    unittest.main()


class PromptFormatTests(unittest.TestCase):
    """The list format shows the same board 0-indexed and maps actions back."""

    def setUp(self):
        from play_puzzle import model_prompt, render_board, system_prompt, to_env_coords
        self.model_prompt = model_prompt
        self.render_board = render_board
        self.system_prompt = system_prompt
        self.to_env_coords = to_env_coords
        self.env = MinesweeperEnv()
        self.env.reset(seed=0)

    def test_list_board_is_valid_python_matching_observation(self):
        namespace = {}
        exec(self.render_board(self.env, "list"), namespace)
        board = namespace["board"]
        visible = self.env.observation().visible
        self.assertEqual(len(board), self.env.rows)
        for r, row in enumerate(board):
            self.assertEqual(len(row), self.env.cols)
            for c, cell in enumerate(row):
                expected = visible[r][c]
                self.assertEqual(cell, "#" if expected is None else expected)

    def test_list_coordinates_shift_to_one_indexed(self):
        self.assertEqual(self.to_env_coords(0, 0, "list"), (1, 1))
        self.assertEqual(self.to_env_coords(5, 3, "list"), (6, 4))
        self.assertEqual(self.to_env_coords(1, 1, "grid"), (1, 1))
        with self.assertRaises(ValueError):
            self.to_env_coords(0, 0, "nope")

    def test_translate_error_uses_model_coordinates(self):
        from play_puzzle import translate_error
        message = "cell (6, 4) is outside the board"
        self.assertEqual(
            translate_error(message, 5, 3, "list"), "cell board[5][3] is outside the board"
        )
        self.assertEqual(translate_error(message, 6, 4, "grid"), message)
        outside = "cell (7, 7) is outside the board; rows are 1 to 6 and columns are 1 to 6"
        self.assertEqual(
            translate_error(outside, 6, 6, "list"),
            "cell board[6][6] is outside the board; rows are 0 to 5 and columns are 0 to 5",
        )

    def test_list_prompts_describe_zero_indexing(self):
        system = self.system_prompt(self.env, "list")
        self.assertIn("board[0] is the top row", system)
        self.assertIn(f"board[{self.env.rows - 1}] is the bottom row", system)
        user = self.model_prompt(self.env, "list")
        self.assertIn("board = [", user)
        self.assertIn("0-indexed", user)
        grid = self.model_prompt(self.env, "grid")
        self.assertIn(self.env.render(coordinates=False), grid)
        self.assertNotIn("board = [", grid)

    def test_list_solution_trace_replays_through_conversion(self):
        # A 0-indexed transcript of the validator's trace must clear the board.
        for row, col in self.env.solution_for_testing():
            observation, _, done, _ = self.env.reveal(
                *self.to_env_coords(row - 1, col - 1, "list")
            )
            if done:
                break
        self.assertTrue(observation.success)
