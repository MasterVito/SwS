import json
import os
import sys
import argparse
from glob import glob
from pdb import set_trace
from openai import AzureOpenAI
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, concatenate_datasets
sys.path.append(os.getcwd())
from utils.utils import  select_load_dataset


def extract_question(sample):
    completion = sample['completion']
    start_str_set = ["Question:\n\n", "Question:\n", "Question:", "question:", 'Problem:', "problem:", ":\n", "could be:", "Question\n", "question\n", "Question Development:\n", "problem\n", 'Problem\n', 'question.\n', 'problem.\n', 'example, "']
    for start_str in start_str_set:
        if start_str in completion:
            break
    try:
        question = completion.split(start_str)[1].split("\n\n")[0]
    except:
        question = "None"
        for line in completion.split("\n"):
            if line.startswith("Given") and '?' in line.split('Given')[-1]:
                question = line
            # if ":" in line and "Step " not in line:
            #     question = line.split(":")[1].strip(" ").strip("\n").strip(" ")
        if completion.count('"') == 2:
            question = completion.split('"')[1].split('"')[0]
    return {"question": question.strip(" ").strip("\n").strip(" ").strip('"').strip(" ")}


parser = argparse.ArgumentParser(description='描述你的脚本或程序的用途')
parser.add_argument('--question_path', '-i', type=str)
parser.add_argument('--judge_path', '-r', type=str)
parser.add_argument('--tokenizer_path', type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
PROBLEM_CLASS=['Intermediate_Algebra', 'Geometry', 'Precalculus', 'Number_Theory', 'Counting_Probability', 'Algebra', 'Prealgebra']

### Filterting generated questions with Llama3.3-70B and Qwen2.5-72B
all_filter_questions = []
for problem_type in PROBLEM_CLASS:
    print(f">>> Processing {problem_type}")
    # load the questions and judgements
    questions = select_load_dataset(os.path.join(args.question_path, f'{problem_type}_failed_zero_samplings.jsonl'))
    verifier_1 = select_load_dataset(os.path.join(args.judge_path, f'{problem_type}_failed_zero_reward0.jsonl'))
    verifier_2 = select_load_dataset(os.path.join(args.judge_path, f'{problem_type}_failed_zero_reward1.jsonl'))

    # filtering questions by judgement, we need llama judgement to be perfect since this model is prone to give 'perfect'
    questions = questions.add_column('judgement_llama', verifier_1['judgement']).add_column('judgement_qwen', verifier_2['judgement'])
    questions_judge_filtered = questions.filter(lambda x:x['judgement_llama'] in ['perfect'] and x['judgement_qwen'] in ['perfect', 'acceptable'] , num_proc=64)
    print(f"Qusetion Filtering Above Acceptable Rate: {round(len(questions_judge_filtered) / len(questions) * 100, 2)}")

    # filtering questions by whether extract
    questions_judge_filtered_extracted = questions_judge_filtered.map(extract_question, num_proc=64)
    questions_judge_filtered_extracted = questions_judge_filtered_extracted.filter(lambda x:x['question'] not in ["None"], num_proc = 64)
    print(f"Qusetion Filtering Extractable Rate: {round(len(questions_judge_filtered_extracted) / len(questions_judge_filtered) * 100, 2)}")

    # proof filtering
    questions_judge_filtered_extracted_no_proof = questions_judge_filtered_extracted.filter(lambda x:"prove" not in x['question'].lower(), num_proc = 64)
    print(f"Qusetion Filtering No Proof Rate: {round(len(questions_judge_filtered_extracted_no_proof) / len(questions_judge_filtered_extracted) * 100, 2)}")

    # prompt length filtering
    questions_judge_filtered_extracted_no_proof_limit_length = questions_judge_filtered_extracted_no_proof.filter(lambda x:len(tokenizer.encode(x['question'])) < 4000, num_proc = 64)
    print(f"Qusetion Filtering Shorter than Limit Rate: {round(len(questions_judge_filtered_extracted_no_proof_limit_length) / len(questions_judge_filtered_extracted_no_proof) * 100, 2)}")

    # add type
    questions_judge_filtered_extracted_no_proof_limit_length = questions_judge_filtered_extracted_no_proof_limit_length.add_column('type', [problem_type] * len(questions_judge_filtered_extracted_no_proof_limit_length))
    all_filter_questions.append(questions_judge_filtered_extracted_no_proof_limit_length)
    print(problem_type, len(questions_judge_filtered_extracted_no_proof_limit_length))


all_filter_questions = concatenate_datasets(all_filter_questions)
all_filter_questions.to_json(os.path.join(args.question_path, f"{int(len(all_filter_questions) / 1000)}k_initial_filtered_questions.jsonl"))