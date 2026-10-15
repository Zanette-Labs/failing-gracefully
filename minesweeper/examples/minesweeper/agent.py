"""Minesweeper multi-turn agentic rollout for miles (online RL / GRPO).

Drives one no-guess Minesweeper episode (``minesweeper/``) through miles'
rollout. The conversation is exactly what ``play_with_model.py`` sends to vLLM:
a SYSTEM message with the rules (``play_puzzle.system_prompt``), then per turn
a USER message with the turn budget + board (``play_puzzle.model_prompt``) and
the model's ASSISTANT reply whose last line must be ``reveal(row, col)``. The
board is rendered in the row's ``metadata.prompt_format`` (``grid``: unlabeled
text, 1-indexed; ``list``: Python list of lists, 0-indexed). The env's reply is
appended as a loss-masked user turn, so one trajectory = one whole episode.

Episode rules are the harness's rules (see env_bridge.py): success / mine /
turn_limit from the env, plus illegal_move (out of bounds) / unparseable /
truncated (per-turn length cap) which all end the episode as a loss. Revealing
an already-visible cell is NOT a loss: the env spends the turn without
uncovering anything, and the next (loss-masked) user turn says so above the
same board. Reward (``env_bridge.episode_reward``) is 1.0 iff every safe cell
was revealed, ``--graceful-reward`` c (default 0.0) if the turn budget ran out
without a harmful move (outcome turn_limit), else 0.0; it is surfaced by
``reward_func`` as {"score": ...} (use ``--reward-key score``). The outcome and
turn count are kept in ``sample.metadata`` for log_utils.py.

Wire with:
    --custom-generate-function-path examples.minesweeper.agent.generate
    --graceful-reward 0.25
    --custom-rm-path examples.minesweeper.agent.reward_func  --reward-key score
    --input-key prompt --metadata-key metadata
    --custom-rollout-log-function-path examples.minesweeper.log_utils.log_rollout_data
Requires MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 (GenerateFnInput interface).
"""

from __future__ import annotations

import logging
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample

from examples.minesweeper.env_bridge import GRACEFUL_OUTCOMES, MinesweeperGame, episode_reward

logger = logging.getLogger(__name__)

# Two-message scaffold used to tokenize the incremental delta of appending a
# user turn (same trick as examples/paprika/agent.py).
_DUMMY_SCAFFOLD = [
    {"role": "user", "content": "dummy"},
    {"role": "assistant", "content": "dummy"},
]


def _tokenize_user_turn(content: str, tokenizer) -> list[int]:
    """Token ids for appending one user turn + the next assistant prompt.

    Computed as the suffix delta over a fixed [user, assistant] scaffold so the
    running ``sample.tokens`` stays append-only (no full re-tokenization, no
    drift from the prompt the model actually saw).
    """
    without = tokenizer.apply_chat_template(
        _DUMMY_SCAFFOLD, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    with_ = tokenizer.apply_chat_template(
        _DUMMY_SCAFFOLD + [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    assert with_[: len(without)] == without, (
        "minesweeper: append-only user-turn tokenization failed (prefix mismatch). "
        "This chat template reformats earlier turns when a new one is added; "
        f"{with_=} {without=}"
    )
    return with_[len(without) :]


def _append_env_turn(sample: Sample, content: str, tokenizer) -> None:
    """Append the environment's next board as a masked (loss=0) user turn."""
    toks = _tokenize_user_turn(content, tokenizer)
    sample.response += tokenizer.decode(toks)
    sample.response_length += len(toks)
    sample.tokens += toks
    sample.loss_mask += [0] * len(toks)
    sample.rollout_log_probs += [0.0] * len(toks)


def _close_assistant_turn(sample: Sample, tokenizer) -> None:
    """End the running tokens with the canonical assistant-turn closer
    (``<|im_end|>\\n``) regardless of why generation stopped. A length-cut turn
    is missing eos, so inject it. Injected formatting tokens get loss_mask 0 /
    logprob 0.0 so the PG + importance-sampling objective stays exact."""
    closer: list[int] = []
    if not sample.tokens or sample.tokens[-1] != tokenizer.eos_token_id:
        closer.append(tokenizer.eos_token_id)
    closer += tokenizer.encode("\n", add_special_tokens=False)
    sample.response += tokenizer.decode(closer)
    sample.response_length += len(closer)
    sample.tokens += closer
    sample.loss_mask += [0] * len(closer)
    sample.rollout_log_probs += [0.0] * len(closer)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer
    assert not args.partial_rollout, "Partial rollout is not supported"

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    meta = dict(sample.metadata or {})
    # Tell the model its per-turn reply budget (--rollout/--eval-max-response-len).
    game = MinesweeperGame(meta, max_tokens=input.sampling_params.get("max_new_tokens"))

    # Optional global ceiling on turns (0 = the board's own max_turns).
    max_turns = game.max_turns
    if cap := getattr(args, "generate_max_turns", 0):
        max_turns = min(max_turns, cap)

    # system rules + first board, through the model's NATIVE chat template.
    sample.prompt = [
        {"role": "system", "content": game.system_prompt},
        {"role": "user", "content": game.user_prompt()},
    ]
    sample.tokens = compute_prompt_ids_from_sample(input.state, sample, tools=None).copy()
    # Restore a string prompt (the exact rendered text) so the framework's
    # `sample.prompt + sample.response` logging stays happy.
    sample.prompt = tokenizer.decode(sample.tokens)

    n_turns = 0
    status_override = None
    for _turn in range(max_turns):
        payload, halt = compute_request_payload(args, sample.tokens, input.sampling_params)
        if payload is None:
            # no room left to generate -> episode truncated by the context ceiling
            status_override = halt
            game.end_early("context_exhausted")
            break
        output = await post(url, payload)
        await update_sample_from_response(
            args, sample, payload=payload, output=output, update_loss_mask=True
        )
        n_turns += 1

        finish_reason = output["meta_info"]["finish_reason"]["type"]
        if finish_reason == "abort":
            # system-level abort (e.g. weight sync); cannot be retried in-trajectory
            status_override = Sample.Status.ABORTED
            game.end_early("aborted")
            break

        # Close the assistant turn canonically (inject <|im_end|>\n after a length
        # cut) so every turn boundary is well-formed even when the episode ends.
        _close_assistant_turn(sample, tokenizer)

        if game.act(output["text"], finish_reason):
            break  # success or any of the loss outcomes: no further turn
        _append_env_turn(sample, game.user_prompt(), tokenizer)

    # A per-turn length cut is a GAME loss, not a framework truncation, so the
    # sample is COMPLETED unless the context ceiling / an abort ended it.
    sample.status = status_override or Sample.Status.COMPLETED

    reward = episode_reward(game.outcome, getattr(args, "graceful_reward", 0.0))
    meta.update({
        "reward": reward,
        "success": game.success,
        "graceful": game.outcome in GRACEFUL_OUTCOMES,
        "outcome": game.outcome,
        "error": game.error,
        "actions": game.actions,
        "agent_metrics": {"turns": n_turns, "steps": game.steps, "repeats": game.repeats,
                          "max_turns": max_turns},
    })
    sample.metadata = meta
    return GenerateFnOutput(samples=sample)


def _add_arguments(parser):
    # 0 => use the board's own max_turns (metadata / env default 12); >0 caps it.
    parser.add_argument("--generate-max-turns", type=int, default=0)
    # Reward for running out of turns without a harmful move (0 <= c < 1).
    # 0 keeps the original strictly binary reward.
    parser.add_argument("--graceful-reward", type=float, default=0.0)
    # accepted for multi_turn-cli compatibility; this generate inlines the env.
    parser.add_argument("--generate-multi-samples", action="store_true")


generate.add_arguments = _add_arguments


async def reward_func(args, samples, **kwargs):
    """Surface the per-episode binary reward computed in generate()."""
    if isinstance(samples, list):
        return [{"score": float(s.metadata.get("reward", 0.0))} for s in samples]
    return {"score": float(samples.metadata.get("reward", 0.0))}
