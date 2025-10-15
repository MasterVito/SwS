import os
import sys
import argparse
from functools import partial
from tqdm import tqdm
from pdb import set_trace
from concurrent import futures
from collections import Counter
from datasets import load_dataset
sys.path.append(os.getcwd())
from utils.utils import select_load_dataset
from verl.utils.reward_score.prime_math import math_normalize, match_answer
from math_verify import parse


def safe_prime_extract(pred):
    try:
        return math_normalize.normalize_answer(match_answer(pred)[1])
    except:
        return "None"


def safe_parse(pred):
    try:
        return parse(pred)[1]
    except:
        return "None"
    

def extract_answer_from_predictions(sample, completion_weight = 2, timeout = 5):
    '''
    Getting accuracy from predictions
    To avoid the hang due to the bugs in extraction function, we use the concurrent.futures to control the verifying time
    '''
    prime_extract_answers = []
    math_verify_extract_answers = []
    for pred in sample['prediction']:
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(safe_prime_extract, pred)
                prime_extract = future.result(timeout=timeout)
            except:
                prime_extract = "None"

        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(safe_parse, pred)
                math_verify_extract = future.result(timeout=timeout)
            except:
                math_verify_extract = "None"

        prime_extract_answers.append(prime_extract)
        math_verify_extract_answers.append(math_verify_extract)

    if completion_weight > 0:
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(safe_prime_extract, sample['completion'])
                prime_completion_answer = future.result(timeout=timeout)
            except:
                prime_completion_answer = "None"

        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(safe_parse, sample['completion'])
            try:
                math_verify_completion_answer = future.result(timeout=timeout)
                prime_extract_answers.append(prime_extract)
            except:
                math_verify_completion_answer = "None"
    if completion_weight > 0:
        prime_answer, prime_consistency = Counter(prime_extract_answers + [prime_completion_answer] * completion_weight).most_common(1)[0]
        math_verify_answer, math_verify_consistency = Counter(math_verify_extract_answers + [math_verify_completion_answer] * completion_weight).most_common(1)[0]
    else:
        prime_answer, prime_consistency = Counter(prime_extract_answers).most_common(1)[0]
        math_verify_answer, math_verify_consistency = Counter(math_verify_extract_answers).most_common(1)[0]
    return {
            "prime_extract_answers": prime_extract_answers, 
            "prime_answer": prime_answer, 
            "prime_consistency": prime_consistency, 
            "math_verify_extract_answers": math_verify_extract_answers,
            "math_verify_answer": math_verify_answer, 
            "math_verify_consistency": math_verify_consistency, 
            }


def extract_answer_from_predictions_updated(sample, timeout = 5):
    '''
    Getting accuracy from predictions
    To avoid the hang due to the bugs in extraction function, we use the concurrent.futures to control the verifying time
    '''
    prime_extract_answers = []
    math_verify_extract_answers = []
    for pred in sample['prediction']:
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(safe_prime_extract, pred)
                prime_extract = future.result(timeout=timeout)
            except:
                prime_extract = "None"
            try:
                math_verify_extract = parse(pred)[1]
            except:
                math_verify_extract = "None"
        prime_extract_answers.append(prime_extract)
        math_verify_extract_answers.append(math_verify_extract)
    prime_answer, prime_consistency = Counter(prime_extract_answers).most_common(1)[0]
    math_verify_answer, math_verify_consistency = Counter(math_verify_extract_answers).most_common(1)[0]
    return {
            "prime_extract_answers": prime_extract_answers, 
            "prime_answer": prime_answer, 
            "prime_consistency": prime_consistency, 
            "math_verify_extract_answers": math_verify_extract_answers,
            "math_verify_answer": math_verify_answer, 
            "math_verify_consistency": math_verify_consistency, 
            }

def filter_answer_consistency(sample, threshold=4):
    return sample['prime_consistency'] >= threshold or sample['math_verify_consistency'] >= threshold

def get_final_answer(sample):
    return {"ref_answer": sample['prime_answer'] if sample['prime_consistency'] >= sample['math_verify_consistency'] else sample['math_verify_answer']}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='描述你的脚本或程序的用途')
    parser.add_argument('--infer_path', '-i', type=str, default="None", help="The path of the inference result")
    parser.add_argument('--total_chunks', '-t', type=int, default=-1, help="The number of shards to verify the answer")
    parser.add_argument('--current_chunk', '-c', type=int, default=-1, help="The current shard to verify the answer")
    args = parser.parse_args()
    
    file_path = os.path.join(args.infer_path, f"{args.current_chunk}-of-{args.total_chunks}.jsonl")
    print(file_path)
    dataset = select_load_dataset(file_path)

    problems_with_predictions_extracted = dataset.map(extract_answer_from_predictions_updated, num_proc=int(os.cpu_count() // 4 * 3))
    problems_with_predictions_extracted = problems_with_predictions_extracted.remove_columns(["prediction"])
    problems_with_predictions_extracted.to_json(os.path.join(args.infer_path, f"{args.current_chunk}-of-{args.total_chunks}_answer_extracted.jsonl")) 
    
    problems_with_predictions_extracted = problems_with_predictions_extracted.filter(filter_answer_consistency, num_proc=64)
    problems_with_predictions_extracted = problems_with_predictions_extracted.map(get_final_answer, num_proc=64)
    problems_with_predictions_extracted.to_json(os.path.join(args.infer_path, f"{args.current_chunk}-of-{args.total_chunks}_answer_extracted_filtered.jsonl")) 