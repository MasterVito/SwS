NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

python src/concept_encoding.py \
        --data_path record/failure_cases_concepts.jsonl \
        --output_path record/failure_cases_concepts_encodings.jsonl \
        --model_path meta-llama/Llama-3.1-8B \
        --n_gpus ${NUM_GPUS} \
        --concepts_key extract_concepts