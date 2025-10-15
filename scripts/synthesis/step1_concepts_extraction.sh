NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

model_path=meta-llama/Llama-3.3-70B-Instruct
python src/extract_concepts.py \
    --model_name_or_path ${model_path} \
    --input_path record/failure_cases.jsonl \
    --output_path record/failure_case_concepts.jsonl \
    --gpu_memory_utilization 0.9 \
    --top_p 0.95 \
    --temperature 0.0 \
    --n_sampling 1 \
    --tensor_parallel_size ${NUM_GPUS} \
    --max_tokens 4096  \
    --swap_space 64 \