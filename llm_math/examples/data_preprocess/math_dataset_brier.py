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
Preprocess the MATH-lighteval dataset to parquet format (Brier variant)
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/data/repos/maxrl_idk/idk/data/")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_sources = [
        ("Maxwell-Jia/AIME_2024", "aime24"),
        ("math-ai/aime25", "aime25"),
    ]

    instruction_following_normal = (
        "\nPlease reason step by step, and put your final answer in the following format:\n"
        "<answer>\\boxed{<final answer>}</answer>\n"
        "Then, reason about how confident you are in your answer using the following scale:\n"
        "- 0.1: You are guessing or have very little idea\n"
        "- 0.2: You are quite unsure, the approach might be wrong\n"
        "- 0.3: You are somewhat unsure, there are likely errors\n"
        "- 0.4: You are slightly unsure, some steps feel shaky\n"
        "- 0.5: You could go either way\n"
        "- 0.6: You are slightly confident, the approach seems right\n"
        "- 0.7: You are fairly confident, most steps check out\n"
        "- 0.8: You are quite confident, only minor doubt remains\n"
        "- 0.9: You are very confident and have verified key steps\n"
        "- 1.0: You are certain the answer is correct\n"
        "Report your confidence as:\n"
        "<confidence>\\boxed{<your score>}</confidence>"
    )
    prompt_variants = {
        "normal": instruction_following_normal,
    }

    for data_source, data_source_name in data_sources:
        local_dir = os.path.join(args.base_dir, data_source_name, "brier")
        print(f"Loading the {data_source} dataset from huggingface...", flush=True)

        # add a row to each data item that represents a unique id
        def make_map_fn(split, instruction_following, data_source=data_source, data_source_name=data_source_name):
            def process_fn(example, idx):
                if data_source in ["math-ai/aime25"]:
                    question = example.pop("problem")
                    answer = example.pop("answer")
                elif data_source in ["Maxwell-Jia/AIME_2024"]:
                    question = example.pop("Problem")
                    answer = example.pop("Answer")

                question = question + " " + instruction_following

                answer = str(answer)

                if idx == 0:
                    print("Question: ", question)
                    print("Answer: ", type(answer), " ", answer)

                data = {
                    "data_source": data_source_name,
                    "id": idx,
                    "prompt": [
                        {
                            "role": "user",
                            "content": question,
                        },
                    ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {"split": split, "index": idx},
                }
                return data

            return process_fn

        for variant_name, instruction_following in prompt_variants.items():
            # Reload dataset for each variant since .map() with pop modifies examples
            dataset = datasets.load_dataset(data_source, "default", trust_remote_code=True)
            split_name = "train" if "train" in dataset else "test"
            train_dataset = dataset[split_name]
            test_dataset = dataset[split_name]

            train_dataset = train_dataset.map(function=make_map_fn("train", instruction_following), with_indices=True)
            test_dataset = test_dataset.map(function=make_map_fn("test", instruction_following), with_indices=True)

            os.makedirs(local_dir, exist_ok=True)
            train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
            test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
            print(f"Saved {variant_name} variant for {data_source_name} to {local_dir}")

        if args.hdfs_dir is not None:
            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
