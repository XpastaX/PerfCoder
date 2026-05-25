import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Points to CodeOpt/

from utils.common import load_data,save_data


def remove_cpp_comments(code):
    # Remove all multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove all single-line comments
    code = re.sub(r'//.*', '', code)
    return code.strip()


# path of the extracted strategies.
path = "result/strategy/Qwen_Refine_F_RS.jsonl"

data = load_data(path)
source = load_data('data/problem_dict.jsonl')
original_data=load_data('data/PIE/SFT.jsonl')

original_id = {sample['id'] for sample in original_data}
# # ===============================================
# # step 1: recollect data
# # ===============================================

full = []
best = {}

for pid in source:
    users = source[pid]['solutions']
    if pid not in best:
        best[pid]=None
    for u,solutions in users.items():
        sorted_list = sorted(solutions.items(), key=lambda x: x[1]['runtime'], reverse=False)
        for i in range(len(sorted_list)-1):
            tgt_id = sorted_list[i][0]
            tgt_code = sorted_list[i][1]['solution']
            tgt_runtime = sorted_list[i][1]['runtime']
            if best[pid] is None:
                best[pid]=[tgt_id, tgt_runtime, u, tgt_code]
            elif best[pid][1]>tgt_runtime:
                best[pid]=[tgt_id, tgt_runtime, u, tgt_code]
            else:
                pass
            for j in range(i+1,len(sorted_list)):
                src_id = sorted_list[j][0]
                src_code = sorted_list[j][1]['solution']
                src_runtime = sorted_list[j][1]['runtime']
                speed_up=src_runtime/tgt_runtime
                if speed_up<1.1:continue
                if f"{pid}-{src_id}-{tgt_id}" not in original_id: continue
                sample = {
                    "id":f"{pid}-{src_id}-{tgt_id}",
                    "src_code":src_code,
                    "tgt_code":tgt_code,
                    "cpu_time":[src_runtime, tgt_runtime],
                    "speed_up":speed_up,
                    "is_best":True if i==0 else False,
                    "cross_user":False,
                    "user":u
                }
 
                full.append(sample)
print(len(full),len([sample for sample in full if sample['is_best']]))


# save_data(collection, "data/PIE/full.jsonl", jsonl=True)

# # ===============================================
# # step 2: get best solution for each pid
# # ===============================================

tgt_best = [sample for sample in full if sample['is_best']]


stat={
    1.3:0,
    1.5:0,
    2:0,
    3:0,
    4:0,
    5:0,
    10000:0
}
count=0

for sample in tgt_best:
    pid,src_id,tgt_id = sample["id"].split('-')
    u = sample['user']
    best_speed = best[pid][1]
    if sample['cpu_time'][1]/best_speed < 2:
        new_sample = {
            "id":f"{pid}-{src_id}-{best[pid][0]}",
            "src_code":sample['src_code'],
            "tgt_code":best[pid][-1],
            "cpu_time":[src_runtime, best_speed],
            "speed_up":src_runtime/best_speed,
            "is_best":True,
            "cross_user":True,
            "user":f"{u}|{best[pid][2]}"
        }
        count+=1
        full.append(new_sample)

print(count)


# clean code
for sample in full:
    sample['src_code'] = remove_cpp_comments(sample['src_code'])
    sample['tgt_code'] = remove_cpp_comments(sample['tgt_code'])


# stra = load_data('result/strategy/Qwen_Refine_F_RS.jsonl')

# stra_map = {sample['id']:sample['Qwen_Refine_resp'] for sample in stra}
# for sample in full:
#     if sample['id'] in stra_map and not sample['cross_user']:
#         sample['Qwen_Refine_resp'] = stra_map[sample['id']]
#     else:
#         sample['Qwen_Refine_resp'] = None

save_data(full, "data/PIE/full.jsonl", jsonl=True)




# # ===============================================
# # step 3: merge back and clean code
# # ===============================================
# count=0
# data = {sample['id']:sample for sample in data}
# for sample in full:
#     if sample['id'] in data:
#         sample['Qwen_Refine_resp']=data[sample['id']]['Qwen_Refine_resp']
#     else:
#         sample['Qwen_Refine_resp']=None
#         if not sample['cross_user']:
#             count+=1

print(count)

# for key,val in stat.items():
#     print(key,val,round(val/len(tgt_best)*100, 2))