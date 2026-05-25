import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Points to CodeOpt/
from utils.common import load_data
from utils.common import save_data as save_data
import numpy as np
import matplotlib.pyplot as plt
import random



# Extract keys and values
def plot_dict(data_dict, total, path):
    # Extract keys and values
    keys = list(data_dict.keys())
    values = [round(v / total * 100, 2) for v in data_dict.values()]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(keys, values)

    # Rotate x-axis labels vertically
    plt.xticks(rotation=90)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height}', ha='center', va='bottom')

    # Add labels and title
    plt.xlabel('Keys')
    plt.ylabel('Percentage (%)')
    plt.title('Bar Plot of Dictionary with Values on Bars')
    plt.tight_layout()

    # Save the plot
    plt.savefig(path)

    # Show the plot


def plot_dataset(_data, _category, path):
    category_freq = {key: 0 for key in _category}
    for sample in _data:
        cate_set = set(sample['category'])
        for cate in cate_set:
            category_freq[cate] += 1

    plot_dict(category_freq, len(_data), path)
    return category_freq


def cal_avg_speedup(_data):
    speed_up = 0
    for sample in _data:
        speed_up += sample['speed_up']
    print(round(speed_up / len(_data), 2))


def print_speed_up_margin(_data):
    _data_sorted = sorted(_data, key=lambda x: x['speed_up'], reverse=True)
    margin = [10, 5, 4, 3, 2, 1.5, 1.1]
    p = 0
    prev = 0
    print('=============================')
    for index, sample in enumerate(_data_sorted):
        if sample['speed_up'] < margin[p]:
            print(f"{round((index - prev)/len(_data_sorted)*100, 2)}")
            prev = index
            p += 1
            if p >= len(margin): break


def extract_cate_names(text):
    pattern = r"- \*\*(.+?)\*\*:"
    return re.findall(pattern, text)


if __name__ == '__main__':
    """
    Target:
    1. Plot category distribution of different dataset collection.
    2. Regularize strategy name.
    3. Obtain best optimization collection
    4. Obtain strategy following training set
    5. Baseline: random strategy following and best optimization,
    
    """

    strategy_map = load_data("data/PIE/strategy/strategy_classified_Qwen32B-Inst.json")
    data_all = load_data('result/strategy/Qwen_Refine_F_RS.jsonl')
    category = load_data('data/PIE/strategy/category.json')
    del category['Code Cleaning and Readability Improvements']
    # ================================
    # collect code speed
    # ================================
    submission = {}
    for sample in data_all:
        pid, s1, s2 = sample['id'].split('-')
        submission[s1] = sample['cpu_time'][0]
        submission[s2] = sample['cpu_time'][1]
        sample["suggestion"] = sample["Qwen_Refine_resp"].split("\n\n[End of Optimization Suggestions]")[0]
        sample['category'] =[cate for cate in extract_cate_names(sample['suggestion']) if cate in category]
        sample['speedup'] = sample['cpu_time'][0] / sample['cpu_time'][1]

    data_hq = [sample for sample in data_all if sample['is_best']]
    save_data(random.sample(data_all,5000), 'data/PIE/decompose/data_hq_rand_5k.json')

    # ================================
    # collect baseline data, top 5k data with best speedup
    # ================================
    data_speed_sorted = sorted([sample for sample in data_all if not sample['cross_user']], key=lambda x: x['speedup'], reverse=True)

    problem = {}
    for sample in data_speed_sorted:
        pid, s1, s2 = sample['id'].split('-')
        if pid not in problem:
            problem[pid]=0
    data_hq_speedup = []
    index = 0
    while len(data_hq_speedup) < 5000:
        sample = data_speed_sorted[index]
        pid, s1, s2 = sample['id'].split('-')
        if problem[pid]<4:
            data_hq_speedup.append(sample)
        index+=1

    save_data(data_hq_speedup, 'data/PIE/decompose/data_speedup_T5k.json')
    
    # ================================
    # score and sort 5k
    # ================================
    category_freq = plot_dataset(data_hq, category, "plot/hq.png" )
    error_count = 0
    exclude_cate = {'Code Cleaning and Readability Improvements'}
    for sample in data_hq:
        score = 0
        cate_set = set(sample['category'])
        for cate in cate_set:
            freq = category_freq[cate] / len(data_hq)
            score += (1 / freq)
        if len(cate_set)> 0:
            score /= len(cate_set)
        else:
            score = -100
            error_count += 1
        sample['score'] = score
    print(error_count)

    data_hq_sorted = sorted(data_hq, key=lambda x: x['score'], reverse=True)[:5000]

    problem = {}
    for sample in sorted(data_hq, key=lambda x: x['score'], reverse=True):
        pid, s1, s2 = sample['id'].split('-')
        if pid not in problem:
            problem[pid]=[]
        problem[pid].append(sample)
    
    data_hq_selected = []
    candidate = []
    p=0
    while len(data_hq_selected)<5000:
        for pid in problem:
            try:
                candidate.append(problem[pid][p])
            except:
                pass
        if len(candidate)+len(data_hq_selected)< 5000:
            data_hq_selected+=candidate
        else:
            data_hq_selected+=random.sample(candidate,5000-len(data_hq_selected))
        p+=1

    # print_speed_up_margin(data_hq)
    print_speed_up_margin(data_hq_sorted)
    print_speed_up_margin(data_hq_selected)

    category_freq = plot_dataset(data_hq_sorted, category, "plot/hq_score_5k.png")
    category_freq = plot_dataset(data_hq_selected, category, "plot/hq_selected_5k.png")
    save_data(data_hq_selected, 'data/PIE/decompose/data_hq_selected_5k.json')
    save_data(data_hq, 'data/PIE/decompose/data_hq.json')


