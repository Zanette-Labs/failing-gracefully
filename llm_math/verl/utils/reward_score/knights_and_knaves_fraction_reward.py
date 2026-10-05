from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from reasoning_gym.logic.knights_knaves import KnightsKnavesDataset
import json


def extract_solution(solution_str: str) -> str:
    solution_substr = last_boxed_only_string(solution_str)
    if solution_substr is None:
        return None
    try:
        box_removed = remove_boxed(solution_substr)
    except:
        box_removed = None
    return box_removed


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0):
    metadata = json.loads(ground_truth)
    answer = metadata["answer"]

    model_answer = extract_solution(model_output)
    if model_answer is None:
        return 0.0

    try:
        oracle_assignments = KnightsKnavesDataset._normalize_answer(answer)
        model_assignments = KnightsKnavesDataset._normalize_answer(model_answer)

        n = len(oracle_assignments)
        if n == 0:
            return 0.0

        k = len(oracle_assignments.intersection(model_assignments))
        return k / n

    except Exception:
        return 0.0