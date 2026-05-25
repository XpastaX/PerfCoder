import json
import os
try:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD=spawn"]='spawn'
except:
    pass
import sys
from pathlib import Path
import argparse
import torch
from vllm import LLM, SamplingParams
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))  # Points to CodeOpt/
from utils.common import load_data, save_data, read_file, replace_with_terms
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def classify_strategies(model_path, prefix, category, strategy):
        # reformat category
    category_txt=''
    for cate, content in category.items():
        category_txt+=f"{cate}: {content}\n\n"

    prompt_all = []
    for stra in tqdm(strategy):
        suggestion = strategy[stra]['suggestion']
        replace_dict = {
        '{{  category  }}':category_txt,
        '{{  strategy  }}':f"{stra}: {suggestion}"
        }
        prompt = replace_with_terms(prefix, replace_dict)
        prompt_all.append(prompt)
    print(prompt_all[0])

    # Set up vLLM
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,
        dtype="auto",
        # quantization="fp8",  # or "gptq" depending on the quantization used
        trust_remote_code=True,
        gpu_memory_utilization=0.85
    )

    outputs = llm.generate(prompt_all, sampling_params)
    
    torch.save(outputs, 'cache/classify_cache.torch')
    count = 0
    extra_cate=[]

    outputs = torch.load('cache/classify_cache.torch')
    strategy_list = list(strategy.keys())
    for opt,stra in zip(outputs, strategy_list):
        cate = opt.outputs[0].text.split("\n")[0].strip().strip()
        if cate not in category:
            _cate = "N/A"
            for key in category:
                if key in cate:
                    print(f"MATCH:{key}|{cate}")
                    _cate=key
                    break
            if _cate == "N/A":
                extra_cate.append(stra)
                print('==========================')
                print(opt.outputs[0].text.split("\n")[0].strip().strip())
                cate = _cate
        if cate == "N/A": count+=1
        strategy[stra]['category'] = cate
    save_data(strategy, 'data/PIE/strategy/strategy_classified_Qwen32B-Inst.json')
    save_data(extra_cate, 'data/PIE/strategy/strategy_unrecognized_Qwen32B-Inst.json')
    print(f"{count}/{len(strategy)}:{round(count/len(strategy),4)*100}%")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify strategies using a language model.")
    parser.add_argument("--model_path", type=str, default="/path/to/Qwen_32B_Inst", help="Path to the model.")
    parser.add_argument("--strategies_path", type=str, default="data/PIE/strategy/strategy_all.json", help="Path to the strategies JSON file.")
    parser.add_argument("--prefix_path", type=str, default="prefix/classification.txt", help="Path to the prefix text file.")
    parser.add_argument("--category_path", type=str, default="data/PIE/strategy/category.json", help="Path to the category JSON file.")
    args = parser.parse_args()

    prefix = read_file(args.prefix_path)
    category = load_data(args.category_path)
    strategy = load_data(args.strategies_path)



    classify_strategies(args.model_path, prefix, category, strategy)