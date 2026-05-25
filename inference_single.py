import argparse
import torch
import os
from tqdm import tqdm
from utils.common import load_data, save_data, read_file, replace_with_terms, extract_code_chunk
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main(args):
    input_path = args.input_path
    model_path = args.model_path
    save_path = args.save_path
    prefix = read_file(args.prefix)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    parallel = len(args.gpu_ids.split(','))

    if args.num_generations < 2:
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=args.max_tokens,
            stop=['[/SUGG]'],
            repetition_penalty=args.repetition_penalty
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=args.max_tokens,
            n=args.num_generations,
            stop=['[/SUGG]'],
            repetition_penalty=args.repetition_penalty
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    capability = torch.cuda.get_device_capability()
    compute_capability = capability[0] * 10 + capability[1]

    if compute_capability >= 80:
        quant_method = "fp8"
        dt = "bfloat16"
    else:
        quant_method = "awq"
        dt = 'float16'

    if not args.use_8bit:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tokenizer_mode='auto',
            dtype=dt,
            tensor_parallel_size=parallel,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    else:
        print("Loading model in 8bits")
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tokenizer_mode='auto',
            quantization=quant_method,
            tensor_parallel_size=parallel,
            gpu_memory_utilization=args.gpu_memory_utilization
        )

    if args.stage == 'stage1':
        data = load_data(input_path)
        prompts = []
        replace_key = args.replace_key.split(",")
        replace_map = {}
        for item in replace_key:
            if len(item) != 0:
                key_prefix, key_sample = item.split("|")
                replace_map["{{  " + key_prefix + "  }}"] = key_sample

        for sample in data:
            to_rep = {key_prefix: sample[key_sample] for key_prefix, key_sample in replace_map.items()}
            replaced = replace_with_terms(prefix, to_rep)
            if args.parse != '':
                cut_idx = replaced.find(args.parse)
                instruction = replaced[:cut_idx]
            else:
                instruction = replaced
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False,
                                                   add_generation_prompt=True)
            prompts.append(prompt)

        print(prompts[0])
        outputs = llm.generate(prompts, sampling_params)
        print('Finish generating suggestions')

        if args.num_generations < 2:
            outputs = [opt.outputs[0].text for opt in outputs]
            torch.save(outputs, f'cache/{args.model_name}_tmp.torch')
            for sample, opt in zip(data, outputs):
                sample[f"{args.model_name}_resp"] = opt
                sample[args.model_name] = extract_code_chunk(opt)
        else:
            outputs = [[opt.outputs[i].text for i in range(len(opt.outputs))] for opt in outputs]
            torch.save(outputs, f'cache/{args.model_name}_tmp_{args.num_generations}.torch')
            for sample, opt in zip(data, outputs):
                sample[f"{args.model_name}_resp"] = opt
                sample[args.model_name] = [extract_code_chunk(resp) for resp in opt]

        save_data(data, save_path, jsonl=True)
        print(f"Suggestions saved to {save_path}")

    elif args.stage == 'stage2':
        raw_data = load_data(input_path)
        processed_data = []
        prompts = []
        new_data = []

        suggestion_key = args.suggestion_key
        assert suggestion_key != "", "Must provide --suggestion_key for stage2"

        replace_key = args.replace_key.split(",")
        replace_map = {}
        for item in replace_key:
            if len(item) != 0:
                key_prefix, key_sample = item.split("|")
                replace_map["{{  " + key_prefix + "  }}"] = key_sample

        for sample in raw_data:
            suggestions = sample[suggestion_key]
            sub_prompts = []
            for sugg in suggestions:
                new_sample = sample.copy()
                new_sample['suggestion'] = sugg
                to_rep = {
                    key_prefix: new_sample[key_sample]
                    for key_prefix, key_sample in replace_map.items()
                }
                replaced = replace_with_terms(prefix, to_rep)
                if args.parse != '':
                    cut_idx = replaced.find(args.parse)
                    instruction = replaced[:cut_idx]
                else:
                    instruction = replaced

                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                sub_prompts.append(prompt)
            prompts.extend(sub_prompts)
            new_data.append((sample, len(suggestions)))

        outputs = llm.generate(prompts, sampling_params)
        print('Finish generating optimized code')

        all_outputs = []
        idx = 0
        for sample, count in new_data:
            suggestions_outputs = [outputs[idx + i].outputs[0].text for i in range(count)]
            idx += count
            sample[f"{args.model_name}_opt_resp"] = suggestions_outputs
            sample[f"{args.model_name}_final"] = [extract_code_chunk(txt) for txt in suggestions_outputs]
            all_outputs.append(sample)

        save_data(all_outputs, save_path, jsonl=True)
        print(f"Final results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM on a single GPU and generate responses.")
    parser.add_argument("--input_path", type=str, default='data/PIE/test_c20_eval.jsonl',
                        help="Path to the file containing prompts.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results.")
    parser.add_argument("--model_name", type=str, required=True, help="Custom name of your model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM.")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU index to use (default: 0).")
    parser.add_argument("--use_8bit", action='store_true', help="Enable usage of 8-bit model for lower memory consumption.")
    parser.add_argument("--num_generations", type=int, default=1, help="Number of generations to generate.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="GPU memory utilization.")
    parser.add_argument("--prefix", type=str, default="prefix/direct_opt.txt")
    parser.add_argument("--parse", type=str, default="**Optimized Code:**")
    parser.add_argument("--replace_key", type=str, default="src_code|src_code")
    parser.add_argument("--max_tokens", type=int, default=8192*2)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--stage", type=str, default="stage1", help="stage1 or stage2")
    parser.add_argument("--suggestion_key", type=str, default="", help="Key in JSON containing model1 suggestions")

    args = parser.parse_args()
    main(args)
