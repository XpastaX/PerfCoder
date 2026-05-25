import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = 'spawn'
import argparse
import torch
import re
from tqdm import tqdm
from utils.common import load_data, save_data, read_file, replace_with_terms, extract_code_chunk, check_dir
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import multiprocessing
import json
from glob import glob

# Make sure the start method is 'spawn'
def set_start_method():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Avoid errors if the start method has already been set

def process_data_chunk(args, data_chunk, result_queue):
    prefix = read_file(args.prefix_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    parallel = args.parallel

    # Initialize the LLM model
    if not args.use_8bit:
        llm = LLM(model=args.model_path, trust_remote_code=True, tokenizer_mode='auto', dtype='bfloat16',
                    tensor_parallel_size=parallel, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=12288)
    else:
        print("Loading model in 8bits")
        llm = LLM(model=args.model_path, trust_remote_code=True, tokenizer_mode='auto', quantization="fp8", 
                    tensor_parallel_size=parallel, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=12288)

    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=16384)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    replace_key = args.replace_key.split(",")
    replace_map = {}
    # collect replace terms
    for item in replace_key:
        if len(item)!=0:    
            key_prefix,key_sample = item.split("|")
            replace_map["{{  "+key_prefix+"  }}"]=key_sample
    
    # Prepare prompts for the chunk
    prompts = []
    pri=True
    for sample in data_chunk:  # Add progress bar for preparing prompts
        to_rep = {key_prefix: sample[key_sample] for key_prefix,key_sample in replace_map.items()}
        instruction = replace_with_terms(prefix, to_rep)
        if pri: 
            print(instruction)
            pri=False
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False,
                                                add_generation_prompt=True)
        prompts.append((sample["id"], prompt))  # Store the sample ID along with its prompt

    # Generate responses in batches of 100 samples
    batch_size = 100
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses", unit="batch"):  # Progress bar for batch generation
        batch_prompts = [prompt for _, prompt in prompts[i:i + batch_size]]
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        
        # Collect the outputs from the batch
        batch_outputs = [opt.outputs[0].text for opt in batch_outputs]
        # Store response in sample
        new_batch = []
        for sample, opt in zip(data_chunk[i:i + batch_size], batch_outputs):
            sample[f"{args.model_name}_resp"] = opt
            if args.extract_code:
                sample[f"{args.model_name}"] = extract_code_chunk(opt)
            new_batch.append(sample)

        with open(args.temp_file, 'a') as f:  # Open in append mode
            for sample in tqdm(new_batch, desc="Saving results", unit="sample"):  # Progress bar for saving results
                json_line = json.dumps(sample)  # Convert each processed_id to a JSON string
                f.write(json_line + '\n')

def restore(_data, np, model_name):
    cache = []
    print("=================================")
    print("      Restore from cache")
    print("=================================")
    cache_list = glob(f"cache/{model_name}/temp_cache_process_*.json")
    print(f"Loading cache from {len(cache_list)} files")
    for temp_file in cache_list:  
        if os.path.exists(temp_file):
            try:
                _cache = load_data(temp_file)
                cache += _cache
                print(f"{len(_cache)} cache inference is loaded from {temp_file}")
            except Exception as e:
                print(f"Failed to load cache from {temp_file}: {str(e)}")
        else:
            print(f"{temp_file} does not exist!")

    id_map = set([sample['id'] for sample in cache])
    remain = []
    save_key = f"{model_name}_resp"
    for sample in _data:
        if save_key in sample:
            if sample[save_key] is not None:
                cache.append(sample)
                continue
        if sample['id'] not in id_map:
            remain.append(sample)
    print(f"{len(cache)} samples are restored from cache, {len(remain)} samples remain.")
    return cache, remain

from multiprocessing import Queue

def main(args):
    # Set the multiprocessing start method to 'spawn'
    set_start_method()

    # Create a queue for error reporting
    result_queue = Queue()

    # Load the full data
    data = load_data(args.input_path)

    gpu_ids = args.gpu_ids.split(',')

    num_processes = len(gpu_ids) // args.parallel  # Each process will use one or more GPUs

    # Create a temp cache file for each process to save its progress
    cache, remain = restore(data, True, args.model_name)

    if len(remain) == 0:
        save_data(cache, args.save_path, jsonl=True)
        print(f'Results saved at {args.save_path}')

    # Determine the number of processes (based on GPU availability or user input)
    chunk_size = len(remain) // num_processes

    # Split the data into chunks based on the number of processes
    data_chunks = [remain[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    
    # Handle the last chunk if the data size is not perfectly divisible
    if len(remain) % num_processes != 0:
        data_chunks[-1].extend(remain[num_processes * chunk_size:])

    # Define the GPU groups dynamically
    gpu_ids_list = [','.join([str(gpu_ids[i]) for i in range(start, start + args.parallel)]) 
                    for start in range(0, len(gpu_ids), args.parallel)]
    processes = []

    # Create a temp cache file for each process to save its progress
    for i, (gpu_ids, chunk) in enumerate(zip(gpu_ids_list, data_chunks)):
        tmp_path = f"cache/{args.model_name}/temp_cache_process_{i}.json"
        check_dir(tmp_path)
        # Prepare the arguments for each process
        process_args = argparse.Namespace(
            input_path=args.input_path,
            model_name=args.model_name,
            model_path=args.model_path,
            gpu_ids=gpu_ids,  # Each process gets different GPUs
            parallel=args.parallel,
            use_8bit=args.use_8bit,
            gpu_memory_utilization=args.gpu_memory_utilization,
            temp_file=f"cache/{args.model_name}/temp_cache_process_{i}.json",
            prefix_path=args.prefix_path,
            replace_key=args.replace_key,
            extract_code=args.extract_code,

        )

        # Create a process for each chunk of data
        p = multiprocessing.Process(target=process_data_chunk, args=(process_args, chunk, result_queue))
        processes.append(p)
        p.start()

    # Wait for all processes to finish and handle any errors
    for p in processes:
        p.join()

    # Check the result queue for any errors from the worker processes
    while not result_queue.empty():
        error_message = result_queue.get()
        print(f"Error: {error_message}")

    # Save the combined data
    cache, _ = restore(data, num_processes, args.model_name)
    save_data(cache, args.save_path, jsonl=True)
    print(f'Results saved at {args.save_path}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM on multiple sets of GPUs and generate responses.")
    parser.add_argument("--input_path", type=str, default='data/PIE/test_c20_eval.jsonl', help="Path to the file containing prompts.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results.")
    parser.add_argument("--model_name", type=str, required=True, help="Custom name of your model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM.")
    parser.add_argument("--use_8bit", action='store_true', help="Enable usage of 8-bit model for lower memory consumption.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="GPU memory utilization.")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="GPU ids.")
    parser.add_argument("--prefix_path", type=str, default='prefix/decompose/extract_stra.txt', help="Prefix path")
    parser.add_argument("--parallel", type=int, default = 1, help="number of gpu to use for each vLLM")
    parser.add_argument("--replace_key", type=str, default="src_code|src_code,tgt_code|tgt_code")
    parser.add_argument("--extract_code", action='store_true', help="Whether to extract code in response")

    args = parser.parse_args()

    # Run the main function
    main(args)
