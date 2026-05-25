import random
import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Points to CodeOpt/
from utils.common import load_data, save_data, read_file, replace_with_terms

random.seed(123)

train_path = 'data/PIE/SFT.jsonl'
prefix_path = 'prefix/direct_opt.txt'

prefix = read_file(prefix_path)

"""
=================================
obtain hq_sft
=================================
"""
root = "data/raw"
PIE = f'{root}/PIE'
cache = f'{PIE}/cache'
problem_dict_save_path = f'{cache}/problem_dict.jsonl'
problem_dict_save_path = 'data/problem_dict.jsonl'

problem_collection = load_data(problem_dict_save_path)
#  fist find user that has more than two submissions
selected = {}
counter = [0, 0]
for pid in problem_collection:
    solutions = problem_collection[pid]["solutions"]
    kept = {}
    for uid in solutions:
        counter[0] += len(solutions[uid])
        if len(solutions[uid]) > 1:
            kept[uid] = solutions[uid]
            counter[1] += len(kept[uid])
    if len(kept) > 0:
        selected[pid] = {'instruction': problem_collection[pid]['instruction'],
                         'solutions': kept}
print(f'Original Number of Problems:{len(problem_collection)}')
print(f'Kept Number of Problems: {len(selected)}')
print(f"Number of Solutions: {counter[0]} --> {counter[1]}")


# check number of submission sets and unique users
DPO_solution_sets = []
count, user = 0, {}
for pid in selected:
    solutions = selected[pid]['solutions']
    for uid in solutions:
        count += 1
        user[uid] = 0
        DPO_solution_sets.append([item for _, item in solutions[uid].items()])

print(f"Number of Submission Sets: {count}")
print(f"Number of Unique Users: {len(user)}")

train_data = []
for solution in DPO_solution_sets:
    tmp = sorted(solution, key=lambda x: x["runtime"], reverse=True)
    train_data.append(tmp)


path = 'data/PIE/SFT_hq_ours.jsonl'
sft_hg =[]
sft_hg_plain=[]
for sets in train_data:
    pid = sets[0]['problem_id']
    target = sets[-1]
    to_rep = {
        "{{  tgt_code  }}": target['solution'],
    }
    tmp = replace_with_terms(prefix, to_rep)
    for src in sets[:-1]:
        to_rep = {
            "{{  src_code  }}": src['solution'],
        }
        replaced = replace_with_terms(tmp, to_rep)
        # split the string into instruction and response
        index = replaced.find("**Optimized Code:**")
        if src['solution_id']==target['solution_id']: continue
        new_sample = {
            'id': f"{pid}-{target['user_id']}-{src['solution_id']}-{target['solution_id']}",
            'instruction': replaced[:index],
            'response': replaced[index:]
        }
        sft_hg.append(new_sample)
        sft_hg_plain.append({
            'id': f"{pid}-{target['user_id']}-{src['solution_id']}-{target['solution_id']}",
            'src_code':src['solution'],
            'tgt_code':target['solution']
        })
print(f"Total number of training samples: {len(sft_hg)}")
# save_data(sft_hg, 'data/PIE/SFT_hg_ours.jsonl', jsonl=True)
save_data(sft_hg_plain,'data/PIE/SFT_hg_ours_plain.jsonl', jsonl=True)