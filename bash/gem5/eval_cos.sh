# reference_file_path=data/PIE/test.jsonl
reference_file_path=data/PIE/test_c20_eval.jsonl

for model_path in result/cos/2step/*.jsonl
do
    model=$(basename "$model_path" .jsonl)  # extract the file name without extension
    model_generated_outputs_path="$model_path"
    output_dir="result/eval/cos/$model"
    
    # Set the column name based on whether "qwen2.5" is in the model name
    if [[ "$model" == *qwen2.5* ]]; then
        model_generated_potentially_faster_code_col="auto"
    else
        model_generated_potentially_faster_code_col="$model"
    fi

    echo =========================${model}=========================
    python speed_test.py --model_generated_outputs_path "${model_generated_outputs_path}" \
                         --reference_file_path "${reference_file_path}" \
                         --output_dir "${output_dir}" \
                         --model_generated_potentially_faster_code_col "${model_generated_potentially_faster_code_col}"
done
                        #  --redo_src_tgt \