"""
Simple multi-turn generation with tool calling.
"""

import argparse
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import (
    create_tool_call_parser,
    execute_tool_calls,
    update_sample_with_tool_responses,
)
from miles.utils.http_utils import post
from miles.utils.misc import load_function


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # ----------------------- Setup -------------------------

    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer
    assert not args.partial_rollout, "Partial rollout is not supported"

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    execute_tool_function = load_function(args.generate_execute_tool_function_path)

    tool_specs = load_function(args.generate_tool_specs_path)
    tool_call_parser = create_tool_call_parser(tool_specs, args.generate_tool_call_parser)

    multi_samples = []
    round_number = 0
    tool_call_count = 0

    # ----------------------- Initial prompts -------------------------

    prompt_tokens_ids = compute_prompt_ids_from_sample(input.state, sample, tools=tool_specs)

    sample.tokens = prompt_tokens_ids.copy()

    for _turn in range(args.generate_max_turns):
        # ----------------------- Call inference endpoint -------------------------

        payload, halt_status = compute_request_payload(args, sample.tokens, input.sampling_params)
        if payload is None:
            sample.status = halt_status
            if args.generate_multi_samples and multi_samples:
                multi_samples[-1].status = halt_status
            break

        if args.generate_multi_samples:
            sample = deepcopy(input.sample)

        output = await post(url, payload)
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)
        round_number += 1

        if args.generate_multi_samples:
            multi_samples.append(deepcopy(sample))

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            break

        # ----------------------- Execute tools -------------------------

        _, tool_calls = tool_call_parser.parse_non_stream(output["text"])
        if len(tool_calls) == 0:
            break

        tool_call_count += len(tool_calls)
        tool_messages = await execute_tool_calls(tool_calls, execute_tool_function)
        update_sample_with_tool_responses(
            sample, tool_messages, tokenizer=tokenizer, max_obs_tokens=args.generate_max_tool_obs_tokens
        )

    samples_out = multi_samples if args.generate_multi_samples else sample
    _record_multi_turn_metadata(samples_out, round_number=round_number, tool_call_count=tool_call_count)
    return GenerateFnOutput(samples=samples_out)


def _record_multi_turn_metadata(samples, *, round_number: int, tool_call_count: int) -> None:
    """Stash per-trajectory turn/tool-call counts in sample metadata.

    `round_number` is the number of model-generation turns (one per inference
    call); `tool_call_count` is the total tool calls executed across the
    trajectory. These are picked up by train_data_conversion into
    rollout_data["round_number" / "tool_call_count"] and logged by
    log_multi_turn_data when --log-multi-turn is set. tool_call_count is also
    consumed by reward functions that shape on tool usage (see retool_v2).
    """
    targets = samples if isinstance(samples, list) else [samples]
    for s in targets:
        s.metadata["round_number"] = round_number
        s.metadata["tool_call_count"] = tool_call_count


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true")
    parser.add_argument(
        "--generate-max-tool-obs-tokens",
        type=int,
        default=8192,
        help="Cap on tokens a single tool observation may add to the prompt; "
        "prevents a runaway tool print from overflowing the model context. <= 0 disables.",
    )


generate.add_arguments = _add_arguments
