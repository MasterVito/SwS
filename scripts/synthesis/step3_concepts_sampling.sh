set -ex
num_process=50
total_sample_size=1000000
sub_sample_size=$((total_sample_size / num_process))

### parallel concepts sampling for problem generation
# for process in $(seq 1 ${num_process}); do
#     python src/concept_sampling.py \
#         --data_path record/failure_cases_concepts.jsonl \
#         --output_path record/sampled_concepts/${process}-of-${num_process}.jsonl \
#         --data_size ${sub_sample_size} \
#         --embed_path record/failure_cases_concepts_encodings.jsonl \
#         --ratio_path record/fail_ratio_by_category.json &
# done
# wait

### gather all the sampled concepts
python src/combine_concepts_samplings.py --split_path record/sampled_concepts --num_process ${num_process} 