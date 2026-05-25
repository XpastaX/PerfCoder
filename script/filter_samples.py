from utils.common import load_data, save_data
import re
import matplotlib.pyplot as plt
from collections import Counter
import os


def markdown_to_list(markdown_text):
    # Match items starting with "- **["
    items = re.split(r'\n(?=- \*\*\[)', markdown_text.strip())
    cleaned_items = [item.lstrip('- ').strip() for item in items if item.strip()]
    return cleaned_items


def extract_title_content_dict(markdown_text):
    # Split on each new bullet that starts with - **[
    items = re.split(r'\n(?=- \*\*\[)', markdown_text.strip())
    result = {}

    for item in items:
        match = re.match(r'- \*\*\[(.*?)\]\*\*: (.+)', item.strip(), re.DOTALL)
        if match:
            title = match.group(1).strip()
            content = match.group(2).strip()
            result[title] = content

    return result


def extract_relevant_section(s):
    # Step 1: Keep from first "- **[" onward
    start_marker = "- **["
    start_index = s.find(start_marker)
    if start_index == -1:
        return ""

    trimmed = s[start_index:]

    # Step 2: Find last "]**" before the phrase and cut at "\n\nBy following these suggestions,"
    end_marker = "\n\nBy following these suggestions,"
    end_index = trimmed.rfind(end_marker)
    if end_index != -1:
        trimmed = trimmed[:end_index]  # Cut off from the phrase onward
    end_marker = "Here is how the student"
    end_index = trimmed.rfind(end_marker)
    if end_index != -1:
        trimmed = trimmed[:end_index]  # Cut off from the phrase onward

    return trimmed


def clean_data(_path, _resp_name):
    data = load_data(_path)

    for sample in data:
        sample[_resp_name] = extract_relevant_section(sample[_resp_name])
        sample['suggestion'] = sample[_resp_name]
        del sample[_resp_name]
        sample['strategy'] = extract_title_content_dict(sample['suggestion'])

    base, ext = os.path.splitext(_path)
    save_path = f"{base}_cleaned{ext}"

    save_data(data, save_path, jsonl=True)


if __name__ == "__main__":
    path = 'result/strategy/Qwen_32B_Inst_raw_strategy.jsonl'
    resp_name = 'Qwen_32B_Inst_resp'
    clean_data(path, resp_name)
    path = 'result/strategy/Qwen_32B_Inst_hq_raw_strategy_hq.jsonl'
    resp_name = 'Qwen_32B_Inst_hq_resp'
    clean_data(path, resp_name)

    #
    data_hq = load_data('result/strategy/Qwen_32B_Inst_hq_raw_strategy_hq_cleaned.jsonl')
    data_all = load_data('result/strategy/Qwen_32B_Inst_raw_strategy_cleaned.jsonl')
    data = data_hq + data_all

    word = {}
    kind = {}
    num_strategy = 0

    for sample in data:
        num_strategy += len(sample['strategy'])
        for strategy, suggestion in sample['strategy'].items():
            if strategy not in kind:
                kind[strategy] = {"frequency": 1, "suggestion": suggestion}
            else:
                kind[strategy]['frequency'] += 1
    kind = dict(sorted(kind.items(), key=lambda x: x[1]['frequency'], reverse=True))
    print(f"Number of strategy {num_strategy}")
    print(f"Number of Kind {len(kind)}")

    save_data(kind, "data/PIE/strategy/strategy_all.json")
