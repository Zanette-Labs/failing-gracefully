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
Preprocess math datasets to parquet format, generating normal, idk, and brier variants.
"""

import argparse
import os
import re
import socket

import datasets

from verl.utils.hdfs_io import copy, makedirs

_HOSTNAME = socket.gethostname()
if _HOSTNAME.startswith("gh"):
    DATA_BASE_DIR = "/u/darora1/maxrl_idk/idk/data"
elif _HOSTNAME == "training-pod-8gpu":
    DATA_BASE_DIR = "/data/repos/maxrl_idk/idk/data"
elif _HOSTNAME.startswith("login") or _HOSTNAME.startswith("nid"):
    DATA_BASE_DIR = "/pscratch/sd/f/ftajwar7/data_for_idk_5_levels"
else:
    raise RuntimeError(f"Unexpected hostname '{_HOSTNAME}'. Set DATA_BASE_DIR manually.")

# Maps dataset name -> {hf_repo, hf_config, train_split, test_split, question_field, answer_field, answer_fn}
# answer_fn: optional callable to extract the answer string from the raw field value
DATASET_CONFIG = {
    "math12k": {
        "hf_repo": "di-zhang-fdu/MATH12000",
        "hf_config": "default",
        "train_split": "train",
        "test_split": "train",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_fn": None,
    },
    "math500": {
        "hf_repo": "di-zhang-fdu/MATH500",
        "hf_config": "default",
        "train_split": "test",
        "test_split": "test",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_fn": None,
    },
    "aime24": {
        "hf_repo": "Maxwell-Jia/AIME_2024",
        "hf_config": "default",
        "train_split": "train",
        "test_split": "train",
        "question_field": "Problem",
        "answer_field": "Answer",
        "answer_fn": None,
    },
    "aime25": {
        "hf_repo": "math-ai/aime25",
        "hf_config": "default",
        "train_split": "test",
        "test_split": "test",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_fn": None,
    },
    "aime26": {
        "hf_repo": "math-ai/aime26",
        "hf_config": "default",
        "train_split": "test",
        "test_split": "test",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_fn": None,
    },
    "gsm8k": {
        "hf_repo": "openai/gsm8k",
        "hf_config": "main",
        "train_split": "train",
        "test_split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "answer_fn": lambda s: re.search(r"#### (\-?[0-9\.,]+)", s).group(1).replace(",", ""),
    },
    "minerva": {
        "hf_repo": "math-ai/minervamath",
        "hf_config": "default",
        "train_split": "test",
        "test_split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "answer_fn": None,
    },
    "polaris": {
        "hf_repo": "POLARIS-Project/Polaris-Dataset-53K",
        "hf_config": "default",
        "train_split": "train",
        "test_split": "train",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_fn": None,
    },
}

INSTRUCTION_NORMAL = "\nPlease reason step by step, and put your final answer within \\boxed{}."

INSTRUCTION_IDK = (
    "\nPlease reason step by step.\n\n"
    "If you are confident in the answer, output:\n"
    "\\boxed{<final answer>}\n\n"
    "If you are not confident or cannot solve the problem, output:\n"
    "\\boxed{\\text{I don't know}}"
)

INSTRUCTION_BRIER = (
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

VARIANTS = [
    ("normal", INSTRUCTION_NORMAL, ""),
    ("idk", INSTRUCTION_IDK, "_idk"),
    ("brier", INSTRUCTION_BRIER, "_brier"),
]


def make_map_fn(split, instruction, data_source_name, question_field, answer_field, answer_fn, idk_reward=None):
    def process_fn(example, idx):
        question = example.pop(question_field)
        raw_answer = example.pop(answer_field)
        answer = str(answer_fn(raw_answer) if answer_fn is not None else raw_answer)
        question = question + " " + instruction

        if idx == 0:
            print("Question: ", question)
            print("Answer: ", type(answer), " ", answer)

        reward_model = {"style": "rule", "ground_truth": answer}
        if idk_reward is not None:
            reward_model["idk_reward"] = float(idk_reward)

        return {
            "data_source": data_source_name,
            "id": idx,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": reward_model,
            "extra_info": {"split": split, "index": idx},
        }

    return process_fn


IDK_C_VARIANTS = [
    (0.0, "Be optimistic of your approach."),
    (0.2, "Be optimistic of your approach but also don't be too careless."),
    (0.4, "Be careful but also take some risk."),
    (0.6, "Be very careful and try to avoid any mistakes."),
    (0.8, "Be extremely careful and only answer when you are highly confident."),
]


def build_idk_instruction(extra: str) -> str:
    return INSTRUCTION_IDK + "\n\n" + extra


AIME_COMBINED_SOURCES = ["aime24", "aime25", "aime26"]


if __name__ == "__main__":
    all_choices = list(DATASET_CONFIG.keys()) + ["aime_combined"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=all_choices,
                        help=f"Dataset name: {', '.join(all_choices)}")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    dataset_name = args.dataset
    local_dir = os.path.join(DATA_BASE_DIR, dataset_name)

    if dataset_name == "aime_combined":
        print("Building aime_combined from aime24, aime25, aime26", flush=True)
        for variant_name, instruction, suffix in VARIANTS:
            data_source_name = dataset_name
            variant_dir = os.path.join(local_dir, variant_name)

            combined_parts = []
            for src_name in AIME_COMBINED_SOURCES:
                cfg = DATASET_CONFIG[src_name]
                print(f"\nLoading {cfg['hf_repo']} ({src_name}) for '{variant_name}' variant...", flush=True)
                raw = datasets.load_dataset(cfg["hf_repo"], cfg["hf_config"], trust_remote_code=True)
                split = raw[cfg["test_split"]]
                # Normalize field names to lowercase problem/answer
                if cfg["question_field"] != "problem":
                    split = split.rename_column(cfg["question_field"], "problem")
                if cfg["answer_field"] != "answer":
                    split = split.rename_column(cfg["answer_field"], "answer")
                split = split.select_columns(["problem", "answer"])
                if split.features["answer"].dtype != "string":
                    split = split.cast_column("answer", datasets.Value("string"))
                combined_parts.append(split)

            merged = datasets.concatenate_datasets(combined_parts)
            os.makedirs(variant_dir, exist_ok=True)

            if variant_name == "idk":
                for c, extra in IDK_C_VARIANTS:
                    c_instruction = build_idk_instruction(extra)
                    c_data_source_name = f"{data_source_name}_c_{c}"
                    map_fn = make_map_fn("test", c_instruction, c_data_source_name,
                                         "problem", "answer", None, idk_reward=c)
                    processed = merged.map(function=map_fn, with_indices=True)
                    processed.to_parquet(os.path.join(variant_dir, f"train_c_{c}.parquet"))
                    processed.to_parquet(os.path.join(variant_dir, f"test_c_{c}.parquet"))
            else:
                map_fn = make_map_fn("test", instruction, data_source_name,
                                     "problem", "answer", None)
                processed = merged.map(function=map_fn, with_indices=True)
                processed.to_parquet(os.path.join(variant_dir, "train.parquet"))
                processed.to_parquet(os.path.join(variant_dir, "test.parquet"))
            print(f"Saved '{variant_name}' (data_source={data_source_name}) to {variant_dir}")
    else:
        cfg = DATASET_CONFIG[dataset_name]
        print(f"Dataset: {dataset_name} -> {cfg['hf_repo']}", flush=True)

        for variant_name, instruction, suffix in VARIANTS:
            data_source_name = dataset_name
            variant_dir = os.path.join(local_dir, variant_name)

            print(f"\nLoading {cfg['hf_repo']} for '{variant_name}' variant...", flush=True)
            raw = datasets.load_dataset(cfg["hf_repo"], cfg["hf_config"], trust_remote_code=True)

            os.makedirs(variant_dir, exist_ok=True)

            if variant_name == "idk":
                for c, extra in IDK_C_VARIANTS:
                    c_instruction = build_idk_instruction(extra)
                    c_data_source_name = f"{data_source_name}_c_{c}"
                    map_fn_train = make_map_fn("train", c_instruction, c_data_source_name,
                                               cfg["question_field"], cfg["answer_field"],
                                               cfg["answer_fn"], idk_reward=c)
                    map_fn_test = make_map_fn("test", c_instruction, c_data_source_name,
                                              cfg["question_field"], cfg["answer_field"],
                                              cfg["answer_fn"], idk_reward=c)
                    train_dataset = raw[cfg["train_split"]].map(function=map_fn_train, with_indices=True)
                    test_dataset = raw[cfg["test_split"]].map(function=map_fn_test, with_indices=True)
                    train_dataset.to_parquet(os.path.join(variant_dir, f"train_c_{c}.parquet"))
                    test_dataset.to_parquet(os.path.join(variant_dir, f"test_c_{c}.parquet"))
            else:
                map_fn_train = make_map_fn("train", instruction, data_source_name,
                                           cfg["question_field"], cfg["answer_field"], cfg["answer_fn"])
                map_fn_test = make_map_fn("test", instruction, data_source_name,
                                          cfg["question_field"], cfg["answer_field"], cfg["answer_fn"])
                train_dataset = raw[cfg["train_split"]].map(function=map_fn_train, with_indices=True)
                test_dataset = raw[cfg["test_split"]].map(function=map_fn_test, with_indices=True)
                train_dataset.to_parquet(os.path.join(variant_dir, "train.parquet"))
                test_dataset.to_parquet(os.path.join(variant_dir, "test.parquet"))
            print(f"Saved '{variant_name}' (data_source={data_source_name}) to {variant_dir}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)