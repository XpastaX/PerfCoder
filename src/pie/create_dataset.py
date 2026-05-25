from collections import defaultdict
import argparse
import json
import random
import traceback
from pathlib import Path
from typing import Optional

from datasets import Dataset
import os
import re

SYSTEM_PROMPT = """"""

# CODE_FORMAT = """**Optimized Code:**

# ```cpp
# {{  tgt_code  }}
# ```
# """
    

def split_dataset_by_application(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    # Create empty datasets with the same features
    train_set = dataset
    val_set = Dataset.from_dict({k: [] for k in dataset.features.keys()}, features=dataset.features)
    test_set = Dataset.from_dict({k: [] for k in dataset.features.keys()}, features=dataset.features)
    return {
    "train": train_set,
    "val": val_set,
    "test": test_set,
    }

    # split_1 = dataset.train_test_split(test_size = val_ratio + test_ratio, seed = random_state)
    # split_2 = split_1["test"].train_test_split(test_size = val_ratio / (val_ratio + test_ratio), seed = random_state)
    # return {
    #     "train": split_1["train"],
    #     "val": split_2["train"],
    #     "test": split_2["test"],
    # }

DATA_PATH = 'dataset/dataset_info.json'
TESTCASES_MAPPING_PATH = 'dataset/input_mapping.json'

def extract_code(instruction: str) -> str:
    pattern = re.compile(rf"```cpp\n(.*?)```", re.DOTALL)
    matches = pattern.findall(instruction)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer

def load_dataset(dataset_name):
    with open(TESTCASES_MAPPING_PATH, 'r') as f:
        problem2testcase_mapping = json.load(f)
    def add_testcases(example):
        problem_id = example['id'].split("-")[0]
        example["testcases"] = problem2testcase_mapping[problem_id]
        example['problem_id'] = problem_id
        return example
    # def add_code_output_format(example):
    #     prompt = f"""{example['instruction']}
    #     {CODE_FORMAT}
    #     """
    #     example['instruction'] = prompt
    #     return example
    def add_org_code(example):
        if "org_code" not in example:
            org_code = extract_code(example['instruction'])
            example['org_code'] = org_code
        return example
    with open(DATA_PATH, 'r') as f:
        json_data = json.load(f)
        file_path = json_data[dataset_name]['file_name']
        data = []
        with open(file_path, 'r', encoding='utf-8') as f2:
            for line in f2:
                data.append(json.loads(line))

        dataset = Dataset.from_list(data)
        column_mapping = { v : k for k, v in json_data[dataset_name]['columns'].items()}
        dataset = dataset.map(add_testcases)
        dataset = dataset.map(add_org_code)
        # dataset = dataset.map(add_code_output_format)
        dataset = dataset.rename_columns(column_mapping)
        return split_dataset_by_application(dataset)

def format_sft_example(example):
    return example

def format_grpo_example(example):
    example["system_prompt"] = SYSTEM_PROMPT
    return example

def format_to_input_ids(example, tokenizer, SFTConfig):
    full_text = example["prompt"] + example["response"]
    # tokenizer.padding_side = 'left'
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=SFTConfig.max_length,
        padding=True,
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]

    # Construct label, only keep response part
    prompt_len = len(tokenizer(example["prompt"])["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    
    # Truncate
    input_ids = input_ids[:SFTConfig.max_length]
    labels = labels[:SFTConfig.max_length]
    attention_mask = tokenized["attention_mask"][:SFTConfig.max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def format_to_input_ids_grpo(example, tokenizer, GRPOConfig):
    # tokenizer.padding_side = 'left'
    tokenized = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=GRPOConfig.max_prompt_length,
        padding=True,
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]
    
    # Construct label, only keep response part
    prompt_len = len(tokenizer(example["prompt"])["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    # Truncate
    input_ids = input_ids[: GRPOConfig.max_prompt_length]
    labels = labels[: GRPOConfig.max_prompt_length]
    attention_mask = tokenized["attention_mask"][: GRPOConfig.max_prompt_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == '__main__':
    d = load_dataset('', 'TEST_DATASET')
    print(d['train'].column_names)
    print(d['train'][0])