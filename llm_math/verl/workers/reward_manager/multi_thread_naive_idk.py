# Copyright 2024 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional
import time
import signal

import ray
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.utils.reward_score.math import last_boxed_only_string
from verl.utils.reward_score.math_verify import extract_solution

# -----------------------------------------------------------------------------
# math_verify imports
# -----------------------------------------------------------------------------
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    raise RuntimeError("Please install math-verify: pip install math-verify")


# =============================================================================
# Per-item timeout helpers (actor-local)
# =============================================================================

class _ItemTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ItemTimeout


_IDK_ANSWER = "\\text{I don't know}"


def _is_idk(response: str) -> bool:
    """Check if the model's boxed answer is \\text{I don't know} AND that box
    is the final content of the response (only trailing whitespace and an
    optional closing full stop after it).
    """
    extracted = extract_solution(response)
    if extracted is None:
        return False
    if extracted.strip() != _IDK_ANSWER:
        return False

    boxed_str = last_boxed_only_string(response)
    if boxed_str is None:
        return False

    box_end = response.rfind(boxed_str)
    if box_end == -1:
        return False
    box_end += len(boxed_str)

    return response[box_end:].strip() in ("", ".")


# =============================================================================
# MathVerify scorer (ONE per actor process)
# =============================================================================

class MathVerifyScorer:
    def __init__(self):
        self._verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )

    def compute_score(
        self,
        model_output: str,
        ground_truth_unboxed: str,
        idk_reward: float,
        timeout_score: float,
        per_item_timeout_s: int,
    ) -> Tuple[float, str]:
        """Returns (score, category) where category is 'idk', 'correct', or 'wrong'."""

        # Check IDK first
        if _is_idk(model_output):
            return float(idk_reward), "idk"

        # math_metric expects boxed GT
        gt_boxed = f"\\boxed{{{ground_truth_unboxed}}}"

        # hard per-item timeout
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(per_item_timeout_s)
        try:
            score, _ = self._verify_func([gt_boxed], [model_output])
            if float(score) > 0:
                return 1.0, "correct"
            else:
                return 0.0, "wrong"
        except _ItemTimeout:
            return 0.0, "wrong"
        except TimeoutException:
            return 0.0, "wrong"
        except Exception:
            return 0.0, "wrong"
        finally:
            signal.alarm(0)


# =============================================================================
# Ray actor
# =============================================================================

@ray.remote(
    max_restarts=0,
    max_task_retries=0,
)
class RewardScoreActor:
    def __init__(self):
        self._scorer = MathVerifyScorer()

    def compute_scores_batch(
        self,
        batch: List[Tuple[int, str, str, float]],  # (item_idx, response_str, ground_truth_unboxed, idk_reward)
        timeout_score: float,
        per_item_timeout_s: int,
    ) -> List[Tuple[int, Dict[str, Any]]]:
        out: List[Tuple[int, Dict[str, Any]]] = []
        for item_idx, response_str, ground_truth, idk_reward in batch:
            score, category = self._scorer.compute_score(
                response_str,
                ground_truth,
                idk_reward,
                timeout_score,
                per_item_timeout_s,
            )

            out.append((item_idx, {"score": float(score), "category": category}))
        return out


# Helper to get the number of reward actors
def get_num_reward_actors(
    cpu_per_actor: float = 1.0,
    max_actors: Optional[int] = None,
    min_actors: int = 1,
) -> int:
    resources = ray.available_resources()
    num_cpus = int(resources.get("CPU", 0))

    if num_cpus <= 0:
        return min_actors

    n = int(num_cpus // cpu_per_actor)

    if max_actors is not None:
        n = min(n, max_actors)

    return max(n, min_actors)


# =============================================================================
# Reward Manager
# =============================================================================

@register("multi_thread_idk")
class MultiThreadNaiveIdkRewardManager:
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        num_reward_actors: Optional[int] = 16,
        batch_size: int = 8,
        in_flight_batches_per_actor: int = 4,
        per_item_timeout_s: int = 1,
        per_batch_timeout_s: float = 10.0,
        poll_interval_s: float = 0.5,
        timeout_score: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        self._batch_size = int(batch_size)
        self._timeout_score = float(timeout_score)
        self._per_item_timeout_s = int(per_item_timeout_s)
        self._per_batch_timeout_s = float(per_batch_timeout_s)
        self._poll_interval_s = float(poll_interval_s)

        if isinstance(num_reward_actors, int):
            self.num_reward_actors = num_reward_actors
        else:
            self.num_reward_actors = int(get_num_reward_actors())

        print("\nReward model is using this many reward actors: ", self.num_reward_actors, "\n")

        self._actors = [RewardScoreActor.remote() for _ in range(self.num_reward_actors)]
        self._next_actor = 0
        self._max_inflight_batches = self.num_reward_actors * in_flight_batches_per_actor

    def _pick_actor(self):
        a = self._actors[self._next_actor]
        self._next_actor = (self._next_actor + 1) % len(self._actors)
        return a

    def _effective_idk_reward(self, c: float) -> float:
        return c

    def _after_scoring(self, items, results, scored_indices) -> Dict[str, float]:
        return {}

    def __call__(self, data: DataProto, return_dict: bool = False):
        # preserve shortcut behavior
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: Dict[str, list] = defaultdict(list)

        # logging counters
        num_batch_timeouts = 0
        num_batch_exceptions = 0
        num_batches_ok = 0

        already_print_data_sources: Dict[str, int] = {}

        n = len(data)

        # decode once
        items: List[Dict[str, Any]] = []
        for i in range(n):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            attn_mask = data_item.batch["attention_mask"]

            valid_prompt_len = attn_mask[:prompt_len].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = data_item.batch["responses"]
            valid_resp_len = attn_mask[prompt_len:].sum()
            valid_resp_ids = response_ids[:valid_resp_len]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)

            reward_model = data_item.non_tensor_batch["reward_model"]
            ground_truth = reward_model["ground_truth"]
            idk_reward = reward_model["idk_reward"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            items.append(
                dict(
                    i=i,
                    response=response_str,
                    ground_truth=ground_truth,
                    idk_reward=idk_reward,
                    effective_idk_reward=self._effective_idk_reward(float(idk_reward)),
                    data_source=data_source,
                    prompt_str=prompt_str,
                    valid_resp_len=valid_resp_len,
                )
            )

        # batching scheduler
        pending: Set[ray.ObjectRef] = set()
        start_time: Dict[ray.ObjectRef, float] = {}
        ref_to_batch: Dict[ray.ObjectRef, List[int]] = {}
        results: Dict[int, Dict[str, Any]] = {}
        scored_indices: Set[int] = set()

        next_i = 0

        def submit_one_batch():
            nonlocal next_i
            if next_i >= n:
                return

            batch: List[int] = []
            while next_i < n and len(batch) < self._batch_size:
                batch.append(next_i)
                next_i += 1

            payload = [(i, items[i]["response"], items[i]["ground_truth"], items[i]["effective_idk_reward"]) for i in batch]
            ref = self._pick_actor().compute_scores_batch.remote(
                payload,
                self._timeout_score,
                self._per_item_timeout_s,
            )

            pending.add(ref)
            start_time[ref] = time.time()
            ref_to_batch[ref] = batch

        # prime inflight
        while len(pending) < self._max_inflight_batches and next_i < n:
            submit_one_batch()

        # gather
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1, timeout=self._poll_interval_s)
            now = time.time()

            for ref in ready:
                pending.remove(ref)
                batch = ref_to_batch.pop(ref)
                start_time.pop(ref, None)

                try:
                    pairs = ray.get(ref)
                    num_batches_ok += 1
                    for idx, out in pairs:
                        results[idx] = out
                        scored_indices.add(idx)
                except Exception:
                    num_batch_exceptions += 1
                    for i in batch:
                        results[i] = {"score": 0.0, "category": "wrong"}

            # batch-level safety timeout
            for ref in list(pending):
                if now - start_time.get(ref, now) > self._per_batch_timeout_s:
                    pending.remove(ref)
                    batch = ref_to_batch.pop(ref)
                    start_time.pop(ref, None)
                    try:
                        ray.cancel(ref, force=True)
                    except Exception:
                        pass
                    num_batch_timeouts += 1
                    for i in batch:
                        results[i] = {"score": 0.0, "category": "wrong"}

            # backfill
            while len(pending) < self._max_inflight_batches and next_i < n:
                submit_one_batch()

        tuning_metrics = self._after_scoring(items, results, scored_indices)

        # postprocess + metrics
        total_idk = 0
        total_correct = 0
        total_wrong = 0

        # per-dataset counters
        ds_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0, "idk": 0})

        for i, info in enumerate(items):
            out = results.get(i, {"score": 0.0, "category": "wrong"})
            reward_tensor[i, info["valid_resp_len"] - 1] = float(out["score"])

            category = out["category"]
            if category == "idk":
                total_idk += 1
            elif category == "correct":
                total_correct += 1
            else:
                total_wrong += 1

            ds = info["data_source"]
            ds_counts[ds][category] += 1

            # examine prints
            if ds not in already_print_data_sources:
                already_print_data_sources[ds] = 0

            if already_print_data_sources[ds] < self.num_examine:
                already_print_data_sources[ds] += 1
                print("[data_source]", ds)
                print("[prompt]", info["prompt_str"])
                print("[response]", info["response"])
                print("[ground_truth]", info["ground_truth"])
                print("[score]", out["score"])
                print("[category]", out["category"])

        pct_idk = (total_idk / n) if n else 0.0
        pct_correct = (total_correct / n) if n else 0.0
        pct_wrong = (total_wrong / n) if n else 0.0

        print(f"\n[IDK Stats] total={n}, correct={total_correct} ({(pct_correct*100):.1f}%), "
              f"idk={total_idk} ({(pct_idk*100):.1f}%), wrong={total_wrong} ({(pct_wrong*100):.1f}%)\n")

        # per-dataset statistics
        statistics_per_ds: Dict[str, Dict[str, float]] = {}
        for ds, counts in ds_counts.items():
            ds_total = counts["correct"] + counts["wrong"] + counts["idk"]
            ds_correct = counts["correct"]
            ds_wrong = counts["wrong"]
            ds_idk = counts["idk"]
            answered = ds_correct + ds_wrong
            statistics_per_ds[ds] = {
                "correct": ds_correct / ds_total if ds_total else 0.0,
                "wrong": ds_wrong / ds_total if ds_total else 0.0,
                "idk": ds_idk / ds_total if ds_total else 0.0,
                "selective_accuracy": ds_correct / answered if answered else 0.0,
            }
            print(f"[IDK Stats][{ds}] total={ds_total}, correct={ds_correct} ({statistics_per_ds[ds]['correct']*100:.1f}%), "
                  f"idk={ds_idk} ({statistics_per_ds[ds]['idk']*100:.1f}%), wrong={ds_wrong} ({statistics_per_ds[ds]['wrong']*100:.1f}%), "
                  f"selective_accuracy={statistics_per_ds[ds]['selective_accuracy']*100:.1f}%")

        if return_dict:
            result = {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "statistics_per_ds": statistics_per_ds,
                "pct_idk": pct_idk,
                "pct_correct": pct_correct,
                "pct_wrong": pct_wrong,
            }
            if tuning_metrics:
                result["reward_tuning_metrics"] = tuning_metrics
            return result

        return reward_tensor
