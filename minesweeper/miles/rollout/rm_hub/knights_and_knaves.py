"""Binary Knights-and-Knaves reward for miles.

A 1:1 port of verl's ``verl/utils/reward_score/knights_and_knaves.py``
(``~/exploration/verl/...``) wired as a miles ``--custom-rm-path`` batch reward.

Scoring is strictly BINARY: extract the model's ``\\boxed{...}`` answer, normalize
it and the gold answer to a set of ``(name, role)`` pairs via reasoning_gym's
``KnightsKnavesDataset._normalize_answer``, and return ``1.0`` iff the two sets are
equal, else ``0.0``. (The source HF dataset ``ftajwar/knights_and_knaves_fraction_reward``
ships a *fractional* / partial-credit reward via its ``data_source`` field; we ignore
that entirely and always score binary, as requested.)

The gold ``label`` for each sample is the inner ``ground_truth`` JSON blob produced by
reasoning_gym (a dict with an ``"answer"`` key, plus ``statements``/``solution``/...),
exactly as verl's ``compute_score`` consumes it. For robustness we also accept a label
that is already the bare answer string.

Wire in with:
  --custom-rm-path miles.rollout.rm_hub.knights_and_knaves.compute_knights_and_knaves_rewards

The verification is pure-Python string/set work (no sympy), so unlike the math-verify
reward it cannot hang; we score inline rather than in a kill-on-timeout process pool.
"""

from __future__ import annotations

import asyncio
import json

try:
    from reasoning_gym.logic.knights_knaves import KnightsKnavesDataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Please install reasoning-gym: pip install reasoning-gym") from exc


# --- boxed-answer extraction (copied verbatim from verl.utils.reward_score.math) ----
def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def extract_solution(solution_str: str):
    solution_substr = last_boxed_only_string(solution_str)
    if solution_substr is None:
        return None
    try:
        return remove_boxed(solution_substr)
    except Exception:
        return None


def _gold_answer(ground_truth: str) -> str:
    """The label is reasoning_gym's inner ground_truth JSON ({"answer": ...}); fall
    back to treating the label as the bare answer string if it isn't that JSON."""
    try:
        meta = json.loads(ground_truth)
        if isinstance(meta, dict) and "answer" in meta:
            return meta["answer"]
    except (json.JSONDecodeError, TypeError):
        pass
    return ground_truth


def compute_score(model_output: str, ground_truth: str) -> float:
    """Binary KnK score: 1.0 iff the boxed assignment matches the gold exactly."""
    answer = _gold_answer(ground_truth)

    model_answer = extract_solution(model_output)
    if model_answer is None:
        return 0.0

    try:
        oracle_assignments = KnightsKnavesDataset._normalize_answer(answer)
        model_assignments = KnightsKnavesDataset._normalize_answer(model_answer)
        return 1.0 if oracle_assignments == model_assignments else 0.0
    except Exception as e:
        print(f"[knights_and_knaves] exception triggered: {e}")
        return 0.0


def _label_to_str(label) -> str:
    return "" if label is None else str(label)


async def compute_knights_and_knaves_rewards(args, samples, **kwargs):
    """Custom rm entrypoint. Accepts a single Sample or a list of Samples."""
    single = not isinstance(samples, (list, tuple))
    sample_list = [samples] if single else list(samples)
    scores = [compute_score(s.response or "", _label_to_str(s.label)) for s in sample_list]
    return scores[0] if single else scores
