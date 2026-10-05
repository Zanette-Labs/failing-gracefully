# Copyright 2024 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import re
import signal
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import ray
import torch

from verl import DataProto
from verl.workers.reward_manager import register

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


# =============================================================================
# Parsing helpers
# =============================================================================

BAD_FORMAT_PENALTY = -1.1

_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_CONFIDENCE_BLOCK_RE = re.compile(r"<confidence>(.*?)</confidence>", re.DOTALL)


def _last_boxed_content_with_tail(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (content, tail) for the last \\boxed{...} in text using brace matching.

    `tail` is everything in `text` after the closing brace of the boxed expression.
    """
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None, None
    i = idx + len("\\boxed{")
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None, None
    return text[idx + len("\\boxed{") : i - 1], text[i:]


def _last_boxed_content(text: str) -> Optional[str]:
    """Return the content of the last \\boxed{...} in text using brace matching."""
    content, _ = _last_boxed_content_with_tail(text)
    return content


_CONFIDENCE_TAIL_RE = re.compile(r"^\.?\s*$")


def _parse_response(response: str) -> Tuple[Optional[str], Optional[float]]:
    """Extract the boxed answer and confidence from the model response.

    Expected format:
        <answer>...\\boxed{...}...</answer> <confidence>\\boxed{float 0-1}[.]</confidence>

    Requires exactly one <answer> block and exactly one <confidence> block, and
    nothing after the confidence's \\boxed{...} except an optional '.' and whitespace.

    Returns (answer_content, confidence) or (None, None) if format is invalid.
    """
    ans_blocks = _ANSWER_BLOCK_RE.findall(response)
    conf_blocks = _CONFIDENCE_BLOCK_RE.findall(response)

    if len(ans_blocks) != 1 or len(conf_blocks) != 1:
        return None, None

    answer_content = _last_boxed_content(ans_blocks[0])
    if answer_content is None:
        return None, None
    answer_content = answer_content.strip()

    conf_raw, conf_tail = _last_boxed_content_with_tail(conf_blocks[0])
    if conf_raw is None:
        return None, None
    if not _CONFIDENCE_TAIL_RE.match(conf_tail):
        return None, None

    try:
        confidence = float(conf_raw.strip())
    except ValueError:
        return None, None

    if confidence < 0.0 or confidence > 1.0:
        return None, None

    return answer_content, confidence


# =============================================================================
# MathVerify scorer (ONE per actor process)
# =============================================================================

class MathVerifyScorer:
    def __init__(self):
        self._verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )

    def compute_correctness(
        self,
        answer_boxed: str,
        ground_truth_unboxed: str,
        per_item_timeout_s: int,
    ) -> bool:
        """Returns True if the answer is correct."""
        gt_boxed = f"\\boxed{{{ground_truth_unboxed}}}"
        pred_boxed = f"\\boxed{{{answer_boxed}}}"

        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(per_item_timeout_s)
        try:
            score, _ = self._verify_func([gt_boxed], [pred_boxed])
            return float(score) > 0
        except (_ItemTimeout, TimeoutException):
            return False
        except Exception:
            return False
        finally:
            signal.alarm(0)


# =============================================================================
# Ray actor
# =============================================================================

@ray.remote(max_restarts=0, max_task_retries=0)
class RewardScoreActor:
    def __init__(self):
        self._scorer = MathVerifyScorer()

    def compute_scores_batch(
        self,
        batch: List[Tuple[int, str, str]],  # (item_idx, answer_boxed, ground_truth_unboxed)
        per_item_timeout_s: int,
    ) -> List[Tuple[int, bool]]:
        """Returns list of (item_idx, is_correct)."""
        out: List[Tuple[int, bool]] = []
        for item_idx, answer_boxed, ground_truth in batch:
            correct = self._scorer.compute_correctness(
                answer_boxed, ground_truth, per_item_timeout_s,
            )
            out.append((item_idx, correct))
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

@register("multi_thread_brier")
class MultiThreadNaiveBrierRewardManager:
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
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        self._batch_size = int(batch_size)
        self._per_item_timeout_s = int(per_item_timeout_s)
        self._per_batch_timeout_s = float(per_batch_timeout_s)
        self._poll_interval_s = float(poll_interval_s)

        if isinstance(num_reward_actors, int):
            self.num_reward_actors = num_reward_actors
        else:
            self.num_reward_actors = int(get_num_reward_actors())

        print("\nBrier reward manager using", self.num_reward_actors, "reward actors\n")

        self._actors = [RewardScoreActor.remote() for _ in range(self.num_reward_actors)]
        self._next_actor = 0
        self._max_inflight_batches = self.num_reward_actors * in_flight_batches_per_actor

    def _pick_actor(self):
        a = self._actors[self._next_actor]
        self._next_actor = (self._next_actor + 1) % len(self._actors)
        return a

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: Dict[str, list] = defaultdict(list)

        num_batch_timeouts = 0
        num_batch_exceptions = 0
        num_batches_ok = 0

        already_print_data_sources: Dict[str, int] = {}

        n = len(data)

        # ---- Pass 0: decode all items and parse format ----
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

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            answer_content, confidence = _parse_response(response_str)

            items.append(dict(
                i=i,
                response=response_str,
                ground_truth=ground_truth,
                data_source=data_source,
                prompt_str=prompt_str,
                valid_resp_len=valid_resp_len,
                answer_content=answer_content,   # None if bad format
                confidence=confidence,            # None if bad format
                is_correct=False,                 # filled in pass 1
                format_ok=answer_content is not None,
            ))

        # ---- Pass 1: compute correctness via Ray actors (only for well-formatted items) ----
        valid_indices = [i for i, item in enumerate(items) if item["format_ok"]]

        pending: Set[ray.ObjectRef] = set()
        start_time: Dict[ray.ObjectRef, float] = {}
        ref_to_batch: Dict[ray.ObjectRef, List[int]] = {}
        correctness: Dict[int, bool] = {}

        next_vi = 0  # index into valid_indices

        def submit_one_batch():
            nonlocal next_vi
            if next_vi >= len(valid_indices):
                return
            batch: List[int] = []
            while next_vi < len(valid_indices) and len(batch) < self._batch_size:
                batch.append(valid_indices[next_vi])
                next_vi += 1

            payload = [
                (i, items[i]["answer_content"], items[i]["ground_truth"])
                for i in batch
            ]
            ref = self._pick_actor().compute_scores_batch.remote(
                payload, self._per_item_timeout_s,
            )
            pending.add(ref)
            start_time[ref] = time.time()
            ref_to_batch[ref] = batch

        # prime inflight
        while len(pending) < self._max_inflight_batches and next_vi < len(valid_indices):
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
                    for idx, is_correct in pairs:
                        correctness[idx] = is_correct
                except Exception:
                    num_batch_exceptions += 1
                    for i in batch:
                        correctness[i] = False

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
                        correctness[i] = False

            # backfill
            while len(pending) < self._max_inflight_batches and next_vi < len(valid_indices):
                submit_one_batch()

        # store correctness back into items
        for i, is_correct in correctness.items():
            items[i]["is_correct"] = is_correct

        # ---- Pass 2: compute brier reward ----
        total_correct = 0
        total_wrong = 0
        total_bad_format = 0
        brier_scores: List[float] = []
        ds_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0, "bad_format": 0})
        ds_brier_scores: Dict[str, List[float]] = defaultdict(list)
        ds_confidences: Dict[str, List[float]] = defaultdict(list)

        for i, item in enumerate(items):
            ds = item["data_source"]

            if not item["format_ok"]:
                reward = BAD_FORMAT_PENALTY
                category = "bad_format"
                total_bad_format += 1
            else:
                correctness_indicator = 1.0 if item["is_correct"] else 0.0
                confidence = item["confidence"]
                brier = (confidence - correctness_indicator) ** 2
                brier_scores.append(brier)
                ds_brier_scores[ds].append(brier)
                ds_confidences[ds].append(confidence)
                reward = correctness_indicator - brier
                category = "correct" if item["is_correct"] else "wrong"
                if item["is_correct"]:
                    total_correct += 1
                else:
                    total_wrong += 1

            reward_tensor[i, item["valid_resp_len"] - 1] = reward
            ds_counts[ds][category] += 1

            # examine prints
            if ds not in already_print_data_sources:
                already_print_data_sources[ds] = 0
            if already_print_data_sources[ds] < self.num_examine:
                already_print_data_sources[ds] += 1
                corr_ind = 1.0 if item["is_correct"] else 0.0
                brier_val = (item["confidence"] - corr_ind) * 2 if item["format_ok"] else "N/A"
                print("[data_source]", ds)
                print("[prompt]", item["prompt_str"])
                print("[response]", item["response"])
                print("[ground_truth]", item["ground_truth"])
                print("[confidence]", item["confidence"])
                print("[is_correct]", item["is_correct"])
                print("[brier_score]", brier_val)
                print("[reward]", reward)
                print("[category]", category)

        pct_correct = (total_correct / n) if n else 0.0
        pct_wrong = (total_wrong / n) if n else 0.0
        pct_bad_format = (total_bad_format / n) if n else 0.0

        avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0.0

        print(f"\n[Brier Stats] total={n}, correct={total_correct} ({pct_correct*100:.1f}%), "
              f"wrong={total_wrong} ({pct_wrong*100:.1f}%), "
              f"bad_format={total_bad_format} ({pct_bad_format*100:.1f}%), "
              f"avg_brier={avg_brier:.4f}\n")

        # per-dataset statistics
        statistics_per_ds: Dict[str, Dict[str, float]] = {}
        for ds, counts in ds_counts.items():
            ds_total = counts["correct"] + counts["wrong"] + counts["bad_format"]
            ds_correct = counts["correct"]
            ds_wrong = counts["wrong"]
            ds_bad_format = counts["bad_format"]
            answered = ds_correct + ds_wrong
            ds_briers = ds_brier_scores[ds]
            ds_confs = ds_confidences[ds]
            ds_avg_brier = sum(ds_briers) / len(ds_briers) if ds_briers else 0.0
            ds_avg_conf = sum(ds_confs) / len(ds_confs) if ds_confs else 0.0
            ds_std_conf = (sum((c - ds_avg_conf) ** 2 for c in ds_confs) / len(ds_confs)) ** 0.5 if ds_confs else 0.0
            statistics_per_ds[ds] = {
                "correct": ds_correct / ds_total if ds_total else 0.0,
                "wrong": ds_wrong / ds_total if ds_total else 0.0,
                "bad_format": ds_bad_format / ds_total if ds_total else 0.0,
                "avg_brier": ds_avg_brier,
                "avg_confidence": ds_avg_conf,
                "std_confidence": ds_std_conf,
            }
            print(f"[Brier Stats][{ds}] total={ds_total}, "
                  f"correct={ds_correct} ({statistics_per_ds[ds]['correct']*100:.1f}%), "
                  f"wrong={ds_wrong} ({statistics_per_ds[ds]['wrong']*100:.1f}%), "
                  f"bad_format={ds_bad_format} ({statistics_per_ds[ds]['bad_format']*100:.1f}%), "
                  f"avg_brier={ds_avg_brier:.4f}, "
                  f"avg_confidence={ds_avg_conf:.4f} (std={ds_std_conf:.4f})")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "statistics_per_ds": statistics_per_ds,
                "pct_correct": pct_correct,
                "pct_wrong": pct_wrong,
                "pct_bad_format": pct_bad_format,
                "avg_brier": avg_brier,
            }

        return reward_tensor
