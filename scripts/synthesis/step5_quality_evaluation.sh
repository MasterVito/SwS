set -ex
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

question_path=record/generated_questions.jsonl
llama_judge_path=record/llama_judgement.jsonl
qwen_judge_path=record/qwen_judgement.jsonl
filtered_path=record/filtered_questions.jsonl

python src/rejection_sampling_reward.py \
    --data_path ${question_path} \
    --output_path ${qwen_judge_path} \
    --model_path Qwen/Qwen2.5-72B-Instruct \
    --n_gpus ${NUM_GPUS} \
    --temperature 0.6 \
    --use_chat_template True \
    --seed 8000

python src/rejection_sampling_reward.py \
    --data_path ${question_path} \
    --output_path ${llama_judge_path} \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --n_gpus ${NUM_GPUS} \
    --temperature 0.6 \
    --use_chat_template True \
    --seed 8000

python src/filter_question_with_judgement_0419.py \
    --question_path ${question_path} \
    --llama_judge_path ${llama_judge_path} \
    --qwen_judge_path ${qwen_judge_path} \
    --output_path ${filtered_path} 