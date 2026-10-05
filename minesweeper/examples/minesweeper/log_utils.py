"""Per-split / per-outcome training-rollout logging for the Minesweeper example.

Additive custom log function (wire with
``--custom-rollout-log-function-path examples.minesweeper.log_utils.log_rollout_data``).
miles calls it at the top of its own ``log_rollout_data``; we log the breakdown
and return ``False`` so the framework's default aggregate ``rollout/*`` + ``perf/*``
logging STILL runs.

Columns added:
    rollout/outcome/<o>                fraction of the batch ending as <o>, for every
                                       o in env_bridge.OUTCOMES (success, mine, turn_limit,
                                       illegal_move, unparseable, truncated, ...). The
                                       failure-mode canary: e.g. "truncated" dominating
                                       means the model reasons past the per-turn cap.
    rollout/<split>/reward_mean        mean episode reward (1 / graceful c / 0)
    rollout/<split>/success_rate       fraction fully solved (reward 1)
    rollout/<split>/graceful_rate      fraction ending at the turn limit with no harmful move
    rollout/<split>/turns_{mean,min,max}   model-generation turns per episode
    rollout/<split>/steps_mean         accepted reveals per episode
    rollout/<split>/repeats_mean       reveals of already-visible cells per episode (wasted turns)
    rollout/<split>/outcome/<o>        per-split outcome fractions
    rollout/<split>/response_len/*     generated tokens only (loss_mask==1)
    rollout/<split>/response_len_with_obs/*  generated + board-observation tokens
    rollout/<split>/total_trajectory_len/*   prompt + everything (context canary)
    rollout/<split>/count              # rollouts of this split in the batch
    rollout/<split>/zero_std_{all_zero,all_one,frac}
                                       prompt-groups with no reward variance (no
                                       GRPO gradient): too hard / too easy / either
"""

from __future__ import annotations

import logging

import numpy as np

from miles.utils import tracking_utils
from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import compute_rollout_step, compute_statistics, dict_add_prefix
from miles.utils.types import Sample

from examples.minesweeper.env_bridge import OUTCOMES

logger = logging.getLogger(__name__)


def _split_of(sample: Sample) -> str:
    return (sample.metadata or {}).get("split", "unknown")


def _outcome_fractions(samples: list[Sample]) -> dict[str, float]:
    outcomes = [(s.metadata or {}).get("outcome", "unknown") for s in samples]
    n = max(1, len(outcomes))
    return {o: outcomes.count(o) / n for o in OUTCOMES}


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    log_dict: dict[str, float | int] = {}

    log_dict |= dict_add_prefix(_outcome_fractions(samples), "rollout/outcome/")

    for split, ssamples in group_by(samples, _split_of).items():
        prefix = f"rollout/{split}/"
        rewards = [s.get_reward_value(args) for s in ssamples]
        successes = [float((s.metadata or {}).get("success", 0.0)) for s in ssamples]
        gracefuls = [float((s.metadata or {}).get("graceful", 0.0)) for s in ssamples]
        metrics = [(s.metadata or {}).get("agent_metrics", {}) for s in ssamples]
        turns = [int(m.get("turns", 0)) for m in metrics]
        steps = [int(m.get("steps", 0)) for m in metrics]
        repeats = [int(m.get("repeats", 0)) for m in metrics]

        log_dict[prefix + "reward_mean"] = float(np.mean(rewards))
        log_dict[prefix + "success_rate"] = float(np.mean(successes))
        log_dict[prefix + "graceful_rate"] = float(np.mean(gracefuls))
        log_dict[prefix + "turns_mean"] = float(np.mean(turns))
        log_dict[prefix + "turns_min"] = int(min(turns)) if turns else 0
        log_dict[prefix + "turns_max"] = int(max(turns)) if turns else 0
        log_dict[prefix + "steps_mean"] = float(np.mean(steps))
        log_dict[prefix + "repeats_mean"] = float(np.mean(repeats))
        log_dict |= dict_add_prefix(_outcome_fractions(ssamples), prefix + "outcome/")

        log_dict |= dict_add_prefix(
            compute_statistics([s.effective_response_length for s in ssamples]), prefix + "response_len/"
        )
        log_dict |= dict_add_prefix(
            compute_statistics([s.response_length for s in ssamples]), prefix + "response_len_with_obs/"
        )
        log_dict |= dict_add_prefix(
            compute_statistics([len(s.tokens) for s in ssamples]), prefix + "total_trajectory_len/"
        )
        log_dict[prefix + "truncated_ratio"] = float(
            np.mean([s.status == Sample.Status.TRUNCATED for s in ssamples])
        )
        log_dict[prefix + "count"] = len(ssamples)

        groups = group_by(ssamples, lambda s: s.group_index)
        if groups:
            all_zero = all_one = all_graceful = zero_var = 0
            for grp in groups.values():
                grp_rewards = [s.get_reward_value(args) for s in grp]
                if all(r == grp_rewards[0] for r in grp_rewards):
                    zero_var += 1
                    if grp_rewards[0] == 0.0:
                        all_zero += 1
                    elif grp_rewards[0] == 1.0:
                        all_one += 1
                    else:
                        all_graceful += 1  # every sample earned the graceful c
            log_dict[prefix + "zero_std_all_zero"] = all_zero / len(groups)
            log_dict[prefix + "zero_std_all_one"] = all_one / len(groups)
            log_dict[prefix + "zero_std_all_graceful"] = all_graceful / len(groups)
            log_dict[prefix + "zero_std_frac"] = zero_var / len(groups)

    log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
    tracking_utils.log(args, log_dict, step_key="rollout/step")

    # Also print the headline numbers, since the framework's own stdout line only
    # shows the mean reward (which is no longer the success rate once c > 0).
    for split in sorted({_split_of(s) for s in samples}):
        prefix = f"rollout/{split}/"
        outcomes = {o: round(log_dict[prefix + "outcome/" + o], 3) for o in OUTCOMES
                    if log_dict[prefix + "outcome/" + o] > 0}
        logger.info(
            f"minesweeper rollout {rollout_id} [{split}]: reward_mean={log_dict[prefix + 'reward_mean']:.3f} "
            f"success_rate={log_dict[prefix + 'success_rate']:.3f} "
            f"graceful_rate={log_dict[prefix + 'graceful_rate']:.3f} "
            f"steps_mean={log_dict[prefix + 'steps_mean']:.2f} repeats_mean={log_dict[prefix + 'repeats_mean']:.2f} "
            f"outcomes={outcomes}"
        )
    return False
