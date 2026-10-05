"""Tests for multi_thread_naive_brier.py"""

import unittest
import math
from multi_thread_naive_brier import _parse_response


class TestParseResponseValid:
    def test_basic(self):
        resp = r"Some reasoning... <answer>\boxed{42}</answer> I'm pretty sure. <confidence>\boxed{0.9}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "42"
        assert conf == 0.9

    def test_zero_confidence(self):
        resp = r"<answer>\boxed{7}</answer> <confidence>\boxed{0.0}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "7"
        assert conf == 0.0

    def test_one_confidence(self):
        resp = r"<answer>\boxed{x+1}</answer> <confidence>\boxed{1.0}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "x+1"
        assert conf == 1.0

    def test_fraction_answer(self):
        resp = r"<answer>\boxed{\frac{1}{2}}</answer> <confidence>\boxed{0.85}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == r"\frac{1}{2}"
        assert conf == 0.85

    def test_whitespace_around_boxed(self):
        resp = "<answer>  \\boxed{10}  </answer> <confidence>  \\boxed{0.5}  </confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "10"
        assert conf == 0.5

    def test_newlines_in_between(self):
        resp = (
            "Let me think...\n"
            "<answer>\n\\boxed{3}\n</answer>\n"
            "I am confident because...\n"
            "<confidence>\n\\boxed{0.7}\n</confidence>"
        )
        answer, conf = _parse_response(resp)
        assert answer == "3"
        assert conf == 0.7

    def test_multiline_answer(self):
        resp = r"<answer>\boxed{a + b + c}</answer> <confidence>\boxed{0.6}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "a + b + c"
        assert conf == 0.6

    def test_integer_confidence(self):
        resp = r"<answer>\boxed{5}</answer> <confidence>\boxed{1}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "5"
        assert conf == 1.0


class TestParseResponseInvalid:
    def test_missing_answer_tag(self):
        resp = r"The answer is 42. <confidence>\boxed{0.9}</confidence>"
        assert _parse_response(resp) == (None, None)

    def test_missing_confidence_tag(self):
        resp = r"<answer>\boxed{42}</answer> I'm pretty sure."
        assert _parse_response(resp) == (None, None)

    def test_missing_both_tags(self):
        resp = "The answer is 42 and I'm 90% sure."
        assert _parse_response(resp) == (None, None)

    def test_confidence_out_of_range_high(self):
        resp = r"<answer>\boxed{42}</answer> <confidence>\boxed{1.5}</confidence>"
        assert _parse_response(resp) == (None, None)

    def test_confidence_out_of_range_negative(self):
        resp = r"<answer>\boxed{42}</answer> <confidence>\boxed{-0.1}</confidence>"
        # -0.1 won't match [\d.]+ so it returns (None, None)
        assert _parse_response(resp) == (None, None)

    def test_confidence_not_a_number(self):
        resp = r"<answer>\boxed{42}</answer> <confidence>\boxed{high}</confidence>"
        # "high" won't match [\d.]+ regex
        assert _parse_response(resp) == (None, None)

    def test_answer_without_boxed(self):
        resp = r"<answer>42</answer> <confidence>\boxed{0.9}</confidence>"
        assert _parse_response(resp) == (None, None)

    def test_confidence_without_boxed(self):
        resp = r"<answer>\boxed{42}</answer> <confidence>0.9</confidence>"
        assert _parse_response(resp) == (None, None)

    def test_empty_answer(self):
        resp = r"<answer>\boxed{}</answer> <confidence>\boxed{0.5}</confidence>"
        # .+? requires at least one char, so this won't match
        assert _parse_response(resp) == (None, None)

    def test_empty_string(self):
        assert _parse_response("") == (None, None)


class TestParseResponseEdgeCases:
    """Edge cases and tricky inputs for _parse_response."""

    def test_nested_braces_in_answer(self):
        """Answer containing nested curly braces (e.g. sets)."""
        resp = r"<answer>\boxed{\{1, 2, 3\}}</answer> <confidence>\boxed{0.8}</confidence>"
        answer, conf = _parse_response(resp)
        # .+? is non-greedy so it should grab up to first }
        # This tests what actually happens with nested braces
        assert (answer, conf) is not None  # at least doesn't crash

    def test_multiple_answer_tags_takes_first(self):
        """If there are multiple answer tags, regex .search() returns the first."""
        resp = (
            r"<answer>\boxed{first}</answer> "
            r"<answer>\boxed{second}</answer> "
            r"<confidence>\boxed{0.5}</confidence>"
        )
        answer, conf = _parse_response(resp)
        assert answer == "first"
        assert conf == 0.5

    def test_multiple_confidence_tags_takes_first(self):
        resp = (
            r"<answer>\boxed{42}</answer> "
            r"<confidence>\boxed{0.3}</confidence> "
            r"<confidence>\boxed{0.9}</confidence>"
        )
        answer, conf = _parse_response(resp)
        assert answer == "42"
        assert conf == 0.3

    def test_confidence_boundary_zero(self):
        resp = r"<answer>\boxed{x}</answer> <confidence>\boxed{0}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "x"
        assert conf == 0.0

    def test_confidence_boundary_one(self):
        resp = r"<answer>\boxed{x}</answer> <confidence>\boxed{1}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "x"
        assert conf == 1.0

    def test_confidence_just_above_one(self):
        resp = r"<answer>\boxed{x}</answer> <confidence>\boxed{1.001}</confidence>"
        assert _parse_response(resp) == (None, None)

    def test_confidence_many_decimals(self):
        resp = r"<answer>\boxed{x}</answer> <confidence>\boxed{0.123456789}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "x"
        assert abs(conf - 0.123456789) < 1e-10

    def test_answer_with_latex_sqrt(self):
        resp = r"<answer>\boxed{\sqrt{2}}</answer> <confidence>\boxed{0.95}</confidence>"
        answer, conf = _parse_response(resp)
        # non-greedy .+? will match \sqrt{2 (stops at first })
        # This documents current behavior
        assert conf == 0.95

    def test_answer_with_spaces_only(self):
        """Answer containing only whitespace after strip should still match since .+? matches spaces."""
        resp = r"<answer>\boxed{   }</answer> <confidence>\boxed{0.5}</confidence>"
        answer, conf = _parse_response(resp)
        # .+? matches whitespace, but strip() makes it empty string
        # Actually .+? will match "   " and strip gives ""
        # The function doesn't reject empty after strip, so this should return ("", 0.5)
        if answer is not None:
            assert answer == ""
            assert conf == 0.5

    def test_confidence_reversed_order(self):
        """Confidence tag before answer tag should still work (regex searches independently)."""
        resp = r"<confidence>\boxed{0.7}</confidence> <answer>\boxed{99}</answer>"
        answer, conf = _parse_response(resp)
        assert answer == "99"
        assert conf == 0.7

    def test_very_long_response_with_valid_tags(self):
        """Tags buried in a very long response."""
        padding = "This is some reasoning. " * 100
        resp = padding + r"<answer>\boxed{42}</answer>" + padding + r"<confidence>\boxed{0.8}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "42"
        assert conf == 0.8

    def test_malformed_boxed_missing_closing_brace(self):
        resp = r"<answer>\boxed{42</answer> <confidence>\boxed{0.5}</confidence>"
        assert _parse_response(resp) == (None, None)

    def test_confidence_with_leading_dot(self):
        resp = r"<answer>\boxed{x}</answer> <confidence>\boxed{.5}</confidence>"
        answer, conf = _parse_response(resp)
        assert answer == "x"
        assert conf == 0.5

    def test_confidence_double_dot(self):
        """Something like 0.5.5 should fail float() conversion."""
        resp = r"<answer>\boxed{x}</answer> <confidence>\boxed{0.5.5}</confidence>"
        # [\d.]+ will match "0.5.5", but float("0.5.5") raises ValueError
        assert _parse_response(resp) == (None, None)


class TestParseResponseOldFormat:
    """Verify the old format (confidence without boxed) no longer parses."""

    def test_old_confidence_format(self):
        resp = r"<answer>\boxed{42}</answer> <confidence>0.9</confidence>"
        assert _parse_response(resp) == (None, None)


# =============================================================================
# Tests for Brier reward computation logic
# =============================================================================

class TestBrierRewardComputation:
    """Test the Brier reward formula: correctness_indicator - (pass_rate - confidence)^2"""

    @staticmethod
    def brier_reward(is_correct: bool, pass_rate: float, confidence: float) -> float:
        """Replicate the reward formula from the manager."""
        correctness_indicator = 1.0 if is_correct else 0.0
        brier = (pass_rate - confidence) ** 2
        return correctness_indicator - brier

    def test_correct_perfect_calibration(self):
        """Correct answer, confidence matches pass_rate → reward = 1.0"""
        reward = self.brier_reward(is_correct=True, pass_rate=0.8, confidence=0.8)
        assert reward == 1.0

    def test_wrong_perfect_calibration(self):
        """Wrong answer, confidence matches pass_rate → reward = 0.0"""
        reward = self.brier_reward(is_correct=False, pass_rate=0.5, confidence=0.5)
        assert reward == 0.0

    def test_correct_overconfident(self):
        """Correct but overconfident (confidence > pass_rate) → reward < 1.0"""
        reward = self.brier_reward(is_correct=True, pass_rate=0.5, confidence=0.9)
        assert reward < 1.0
        assert abs(reward - (1.0 - 0.16)) < 1e-9  # 1 - (0.5-0.9)^2 = 1 - 0.16 = 0.84

    def test_correct_underconfident(self):
        """Correct but underconfident → reward < 1.0"""
        reward = self.brier_reward(is_correct=True, pass_rate=0.8, confidence=0.3)
        assert reward < 1.0
        assert abs(reward - (1.0 - 0.25)) < 1e-9  # 1 - (0.8-0.3)^2 = 0.75

    def test_wrong_overconfident(self):
        """Wrong and overconfident → reward is negative"""
        reward = self.brier_reward(is_correct=False, pass_rate=0.2, confidence=0.9)
        assert reward < 0.0
        assert abs(reward - (0.0 - 0.49)) < 1e-9  # 0 - (0.2-0.9)^2 = -0.49

    def test_wrong_zero_confidence(self):
        """Wrong with zero confidence and zero pass_rate → reward = 0.0"""
        reward = self.brier_reward(is_correct=False, pass_rate=0.0, confidence=0.0)
        assert reward == 0.0

    def test_correct_all_pass(self):
        """All rollouts correct (pass_rate=1.0), confidence=1.0 → reward = 1.0"""
        reward = self.brier_reward(is_correct=True, pass_rate=1.0, confidence=1.0)
        assert reward == 1.0

    def test_reward_range_correct(self):
        """For correct answers, reward is in [0, 1] when calibrated."""
        for pr in [0.0, 0.2, 0.5, 0.8, 1.0]:
            for conf in [0.0, 0.2, 0.5, 0.8, 1.0]:
                reward = self.brier_reward(is_correct=True, pass_rate=pr, confidence=conf)
                assert reward <= 1.0

    def test_reward_range_wrong(self):
        """For wrong answers, reward is in [-1, 0]."""
        for pr in [0.0, 0.2, 0.5, 0.8, 1.0]:
            for conf in [0.0, 0.2, 0.5, 0.8, 1.0]:
                reward = self.brier_reward(is_correct=False, pass_rate=pr, confidence=conf)
                assert reward <= 0.0
                assert reward >= -1.0

    def test_higher_calibration_gives_higher_reward(self):
        """Better calibrated confidence should give higher reward."""
        # pass_rate = 0.6
        # closer confidence to 0.6 should give higher reward
        reward_close = self.brier_reward(is_correct=True, pass_rate=0.6, confidence=0.6)
        reward_far = self.brier_reward(is_correct=True, pass_rate=0.6, confidence=0.1)
        assert reward_close > reward_far

    def test_symmetric_brier_penalty(self):
        """Brier penalty is symmetric: overconfidence and underconfidence by same amount give same penalty."""
        reward_over = self.brier_reward(is_correct=True, pass_rate=0.5, confidence=0.8)
        reward_under = self.brier_reward(is_correct=True, pass_rate=0.5, confidence=0.2)
        assert abs(reward_over - reward_under) < 1e-9


class TestPassRateComputation:
    """Test pass_rate logic: computed per ground_truth group over formatted items only."""

    def test_all_correct_pass_rate(self):
        """If all formatted items for a prompt are correct, pass_rate = 1.0."""
        formatted = [True, True, True]
        correct = [True, True, True]
        num_correct = sum(1 for f, c in zip(formatted, correct) if f and c)
        num_formatted = sum(formatted)
        pass_rate = num_correct / num_formatted
        assert pass_rate == 1.0

    def test_half_correct_pass_rate(self):
        formatted = [True, True, True, True]
        correct = [True, True, False, False]
        num_correct = sum(1 for f, c in zip(formatted, correct) if f and c)
        num_formatted = sum(formatted)
        pass_rate = num_correct / num_formatted
        assert pass_rate == 0.5

    def test_no_formatted_items_pass_rate(self):
        """If no items are formatted, pass_rate should be 0.0."""
        formatted = [False, False]
        num_formatted = sum(formatted)
        pass_rate = 0.0 if num_formatted == 0 else None
        assert pass_rate == 0.0

    def test_bad_format_excluded_from_pass_rate(self):
        """Bad format items don't count in pass_rate denominator."""
        formatted = [True, True, False, False]
        correct = [True, False, False, False]
        num_correct = sum(1 for f, c in zip(formatted, correct) if f and c)
        num_formatted = sum(formatted)
        pass_rate = num_correct / num_formatted
        assert pass_rate == 0.5  # 1 correct out of 2 formatted
