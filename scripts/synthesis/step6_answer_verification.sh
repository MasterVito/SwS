set -ex 
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

input_path=record/filtered_questions.jsonl
output_path=record/filtered_questions_with_ref_answers.jsonl

python src/evaluation.py \
    --model_name_or_path Qwen/QwQ-32B \
    --input_path ${input_path} \
    --output_path ${output_path} \
    --gpu_memory_utilization 0.90 \
    --top_p 0.95 \
    --temperature 0.7 \
    --n_sampling 8 \
    --tensor_parallel_size ${NUM_GPUS} \
    --max_tokens 32768  \
    --swap_space 64 \
    --prompt_key question 

python PromptCoT/get_answer_consistency.py --infer_path record/filtered_questions_with_ref_answers.jsonl