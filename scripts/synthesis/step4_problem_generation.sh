set -ex
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

concept_path=record/sampled_concepts.jsonl
question_path=record/generated_questions.jsonl

python src/problem_generation.py \
    --data_path ${concept_path} \
    --output_path  ${question_path} \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --n_gpus ${NUM_GPUS} \
    --temperature 0.6 \
    --max_len 4096 \
    --seed 8000 \
    --use_chat_template True 