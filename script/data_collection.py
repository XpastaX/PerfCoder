"""
This code is to:
1. collect all unique solutions
2. map then to origianl users
3. gather all preferece learning pairs for DPO training
"""
from utils.common import load_data, save_data, check_file, read_file, check_dir
from glob import glob
import csv
from tqdm import tqdm

root = "data/raw"
PIE = f'{root}/PIE'
CodeNet = f'{root}/CodeNet/Project_CodeNet'
cache = f'{PIE}/cache'
train_path = f"{PIE}/train.jsonl"
metadata_solution_folder = f'{CodeNet}/metadata'
metadata_problem_folder = f'{CodeNet}/problem_descriptions'
# test_path = "data/raw/PIE/test.jsonl"
solution_save_path = f'{cache}/solutions.jsonl'
solution_dict_save_path = f'{cache}/solutions_all.jsonl'
problem_dict_save_path = f'{cache}/problem_dict.jsonl'
selected_matedata_path = f'data/PIE/selected.jsonl'
SFT_save_path = f'data/PIE/SFT.jsonl'
check_dir(cache)
check_dir(SFT_save_path)

# # ===============================================
# # step 1: collection all unique solutions in PIE
# # ===============================================
train = load_data(train_path)  # 77,967

print(f"Train Data Size: {len(train)}")
if not check_file(solution_save_path):
    cpp = {}
    for item in train:
        sid = item['src_id']
        tid = item["tgt_id"]
        st = item['src_agg_runtime']
        tt = item['tgt_agg_runtime']
        pid = item['problem_id']
        cpp[sid] = {
            'solution_id': sid, 'problem_id': pid, 'runtime': st, 'solution': item['src_code']
        }
        cpp[tid] = {
            'solution_id': tid, 'problem_id': pid, 'runtime': tt, 'solution': item['tgt_code']
        }
    save_data(cpp, solution_save_path)
else:
    cpp = load_data(solution_save_path)

print(f"Unique Solutions: {len(cpp)}")  # 40,625

# ===============================================
# step 2: Build meta data from CodeNet
# ===============================================

files = glob(metadata_solution_folder + '/p*.csv')

solutions_dict = {}

# Iterate over all files in the folder
if not check_file(solution_dict_save_path):
    for file_path in tqdm(files):
        # Read the content of the CSV file
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    # Extract the relevant fields
                    submission_id = row['submission_id']
                    problem_id = row['problem_id']
                    user_id = row['user_id']
                except:
                    print(row)
                    print(file_path)
                    continue
                # Build the dictionary entry
                solutions_dict[submission_id] = {
                    'submission_id': submission_id,
                    'problem_id': problem_id,
                    'user_id': user_id
                }
    save_data(solutions_dict, solution_dict_save_path)
else:
    solutions_dict = load_data(solution_dict_save_path)

print(f"Number of Solutions:{len(solutions_dict)}")  # 13,916,868

# ===============================================
# step 3: Collect problem-->(user, [solutions])
# ===============================================
if not check_file(problem_dict_save_path):
    problem_collection = {}
    for solution_id in tqdm(cpp):
        if solution_id not in solutions_dict:
            print(f"Solution:{solution_id} not found in meta data!")
        else:
            if cpp[solution_id]['problem_id'] != solutions_dict[solution_id]['problem_id']:
                print(f"Problem_id does not match for {solution_id}")
            else:
                uid = solutions_dict[solution_id]['user_id']
                pid = solutions_dict[solution_id]['problem_id']
                cpp[solution_id]['user_id'] = uid
                if pid not in problem_collection:
                    problem_collection[pid] = {'instruction': "", "solutions": {}}
                if uid not in problem_collection[pid]['solutions']:
                    problem_collection[pid]['solutions'][uid] = {}
                problem_collection[pid]['solutions'][uid][solution_id] = cpp[solution_id]
    # collect problem defination
    for pid in problem_collection:
        file_path = metadata_problem_folder + f'/{pid}.html'
        if check_file(file_path):
            p = read_file(file_path)
            problem_collection[pid]['instruction'] = p
        else:
            print(f'Problem {pid} not found!')

    save_data(problem_collection, problem_dict_save_path)
else:
    problem_collection = load_data(problem_dict_save_path)

# ===============================================
# step 4: Collect dataset
# ===============================================

#  first find user that has more than two submissions
selected = {}
counter = [0, 0]
for pid in problem_collection:
    solutions = problem_collection[pid]["solutions"]
    kept = {}
    for uid in solutions:
        counter[0] += len(solutions[uid])
        if len(solutions[uid]) > 2:
            kept[uid] = solutions[uid]
            counter[1] += len(kept[uid])
    if len(kept) > 0:
        selected[pid] = {'instruction': problem_collection[pid]['instruction'],
                         'solutions': kept}
print(f'Original Number of Problems:{len(problem_collection)}')
print(f'Kept Number of Problems: {len(selected)}')
print(f"Number of Solutions: {counter[0]} --> {counter[1]}")

save_data(selected, selected_matedata_path)


SFT = []
for item in train:
    sid = item['src_id']
    tid = item["tgt_id"]
    st = item['src_agg_runtime']
    tt = item['tgt_agg_runtime']
    pid = item['problem_id']
    SFT.append({
        'id': f"{pid}-{sid}-{tid}",
        'src_code': item['src_code'],
        'tgt_code': item['tgt_code'],
        'cpu_time': [st, tt]
    })
save_data(SFT, SFT_save_path)

# ===============================================
# step 5: Clean test set
# ===============================================
from utils.common import load_data, save_data, check_file, read_file, check_dir

test_path = "data/raw/PIE/test.jsonl"
save_path = 'data/PIE/test_c20.jsonl'
test = load_data(test_path)

# clean the test set, keep only the codes and ids

cleaned = []

for idx, item in enumerate(test):
    sample = {
        'id': f"pie_test_{idx}",
        'problem_id': item['problem_id'],
        'src_id': item['src_id'],
        'tgt_id': item['tgt_id'],
        'src_code': item['src_code'],
        'tgt_code': item['tgt_code'],
        'fastest_code': item['fastest_code'],
        'tests': item['tests'][:20]
    }
    cleaned.append(sample)

save_data(cleaned, save_path, jsonl=True)