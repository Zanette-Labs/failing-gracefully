# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


TABC_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its confidence about the solution and then provides the user with its confidence level. "
    "The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags."
    " The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags. The assistant should reason about its confidence in the solution and its uncertainty in the solution within these tags."
    "The final format that must be followed is : <think> reasoning process here </think><answer> final answer here </answer> <analysis> analysis about confidence and uncertainty here</analysis> <confidence> confidence level here (number between 0 and 1) </confidence>"
)


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data/repos/maxrl_idk/idk/data/math500")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=30)
    parser.add_argument("--idk", action="store_true", default=False,
                        help="Use IDK-aware prompt that allows the model to say 'I don't know'")
    parser.add_argument("--tabc", action="store_true", default=False,
                        help="Use TABC system prompt that asks the model to emit a confidence level")

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    # data_source = "DigitalLearningGmbH/MATH-lighteval"
    # data_source = "di-zhang-fdu/MATH12000"
    data_source = "di-zhang-fdu/MATH500"
    # data_source = "math-ai/aime25"
    # data_source = "self-label-zanette-lab/math_proper_train_test"
    # data_source = "ftajwar/srt_test_dataset"
    # data_source = "Maxwell-Jia/AIME_2024"
    # data_source = "POLARIS-Project/Polaris-Dataset-53K"
    # data_source = "open-r1/OpenR1-Math-220k"
    # data_source = "zwhe99/amc23"
    # data_source = "math-ai/aime25"
    # data_source = "math-ai/minervamath"
    # data_source = "math-ai/olympiadbench"
    # data_source = "ByteDance-Seed/BeyondAIME"
    # data_source = "ftajwar/deduplicated_dapo_dataset"
    # data_source = "MathArena/hmmt_feb_2025"
    # data_source = "MathArena/hmmt_nov_2025"
    # data_source = "daman1209arora/jeebench_numeric"

    # data_source_name = "math12k_idk" if args.idk else "math12k"
    # data_source = "aime25_idk" if args.idk else "aime25"
    # data_source = "Maxwell-Jia/AIME_2024"
    # data_source_name = "beyondaime_idk" if args.idk else "beyondaime"
    if args.tabc:
        data_source_name = "math500_tabc"
    elif args.idk:
        data_source_name = "math500_idk"
    else:
        data_source_name = "math500"

    # data_source = "self-label-zanette-lab/big_math_filtered_pass_rate_between_0.3_and_0.7"

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    # dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    # instruction_following = " " + "Let's think step by step and output the final answer within \\boxed{}."
    instruction_following_normal = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    instruction_following_idk = (
        "\nPlease reason step by step.\n\n"
        "If you are confident in the answer, output:\n"
        "\\boxed{<final answer>}\n\n"
        "If you are not confident or cannot solve the problem, output:\n"
        "\\boxed{\\text{I don't know}}"
    )

    if args.tabc:
        prompt_variants = {"tabc": instruction_following_normal}
    elif args.idk:
        prompt_variants = {
            "normal": instruction_following_normal,
            "idk": instruction_following_idk,
        }
    else:
        prompt_variants = {"normal": instruction_following_normal}

    # add a row to each data item that represents a unique id
    def make_map_fn(split, instruction_following):
        def process_fn(example, idx):
            # question = example.pop("question")   # amc23, minerva, olympiadbench
            question = example.pop("problem") # polaris, open-r1, aime25, beyondaime
            # question = example.pop("Problem") # math-12k + srt test dataset + aime24
            # question = example.pop("prompt") # big-math-rl-verified, dapo

            # question = question + " " + instruction_following
            question = question + " " + instruction_following

            # answer = example.pop("final_answer")[0]  # olympiadbench
            answer = example.pop("answer")  # polaris, open-r1, amc23, aime25, minerva, beyondaime, dapo
            # answer = example.pop("Answer") # math-12k + srt test dataset + aime24
            # answer = example.pop('answer') # big-math-rl-verified
            # solution = extract_solution(answer)

            answer = str(answer)

            if idx == 0:
                print("Question: ", question)
                print("Answer: ", type(answer), " ", answer)

            data = {
                "data_source": data_source_name,
                "id": idx,
                "prompt": (
                    [
                        {"role": "system", "content": TABC_PROMPT},
                        {"role": "user", "content": f"\n\nPROBLEM: {question}\n\n"},
                    ]
                    if args.tabc
                    else [
                        {"role": "user", "content": question},
                    ]
                ),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    for variant_name, instruction_following in prompt_variants.items():
        # Reload dataset for each variant since .map() with pop modifies examples
        dataset = datasets.load_dataset(data_source, "default", trust_remote_code=True)
        if data_source in ["di-zhang-fdu/MATH12000", "Maxwell-Jia/AIME_2024", "POLARIS-Project/Polaris-Dataset-53K"]:
            train_dataset = dataset["train"]
            test_dataset = dataset["train"] # srt test dataset, aime 2024
        elif data_source in ["di-zhang-fdu/MATH500", "math-ai/aime25", "ByteDance-Seed/BeyondAIME"]:
            train_dataset = dataset["test"]
            test_dataset = dataset["test"]

        train_dataset = train_dataset.map(function=make_map_fn("train", instruction_following), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn("test", instruction_following), with_indices=True)

        if args.idk:
            variant_dir = os.path.join(local_dir, variant_name)
        else:
            variant_dir = local_dir
        if args.tabc:
            variant_dir = os.path.join(local_dir, "tabc")

        os.makedirs(variant_dir, exist_ok=True)
        train_dataset.to_parquet(os.path.join(variant_dir, "train.parquet"))
        test_dataset.to_parquet(os.path.join(variant_dir, "test.parquet"))
        print(f"Saved {variant_name} variant to {variant_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
