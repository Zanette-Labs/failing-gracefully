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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--idk", action="store_true", default=False,
                        help="Use IDK-aware prompt that allows the model to say 'I don't know'")

    args = parser.parse_args()

    data_source = "openai/gsm8k_idk" if args.idk else "openai/gsm8k"

    # dataset = datasets.load_dataset("madrylab/gsm8k-platinum", "main")
    dataset = datasets.load_dataset("openai/gsm8k", "main")

    train_dataset = dataset["train"]
    # train_dataset = dataset["test"]
    test_dataset = dataset["test"]

    # instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    # instruction_following = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    instruction_following_normal = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    instruction_following_idk = (
        "\nPlease reason step by step.\n\n"
        "If you are confident in the answer, output:\n"
        "\\boxed{<final answer>}\n\n"
        "If you are not confident or cannot solve the problem, output:\n"
        "\\boxed{\\text{I don't know}}"
    )

    prompt_variants = {
        "normal": instruction_following_normal,
        "idk": instruction_following_idk,
    } if args.idk else {
        "normal": instruction_following_normal,
    }

    # add a row to each data item that represents a unique id
    def make_map_fn(split, instruction_following):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            # question = question_raw + instruction_following
            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)

            if idx == 0:
                print("Question: ", question)
                print("Answer: ", type(solution), " ", solution)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    for variant_name, instruction_following in prompt_variants.items():
        # Reload dataset for each variant since .map() with pop modifies examples
        dataset = datasets.load_dataset("openai/gsm8k", "main")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        train_dataset = train_dataset.map(function=make_map_fn("train", instruction_following), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn("test", instruction_following), with_indices=True)

        if args.idk:
            variant_dir = os.path.join(local_dir, variant_name)
        else:
            variant_dir = local_dir

        os.makedirs(variant_dir, exist_ok=True)
        train_dataset.to_parquet(os.path.join(variant_dir, "train.parquet"))
        test_dataset.to_parquet(os.path.join(variant_dir, "test.parquet"))
        print(f"Saved {variant_name} variant to {variant_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
