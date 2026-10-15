from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from reasoning_gym import get_score_answer_fn
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
    ret_score = 0.0

    entry = {}
    
    metadata = json.loads(ground_truth)
    entry["metadata"] = metadata
    entry["answer"] = metadata["answer"]

    scoring_fn = get_score_answer_fn(metadata["source_dataset"])
    model_output = extract_solution(model_output)

    try: 
        ret_score = scoring_fn(model_output, entry)

    except Exception as e:
        print(f"exception triggered: {e}")


    return ret_score