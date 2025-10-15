import argparse
from data_synthesis import FastConceptSampler as ConceptSampler
from multiprocessing import Pool, Manager, Lock
import random
import json
from tqdm import tqdm
from pdb import set_trace
import random
import os


def load_embeddings(embed_path):
    embeddings = {}
    with open(embed_path, encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Loading embeddings"):
            item = json.loads(line)
            embeddings[item['sentence']] = item['embedding']
    return embeddings


def load_concept_lists(input_file):
    concept_lists = []
    with open(input_file, encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            lst = item["extract_concepts"]
            if len(lst) == 1 and len(lst[0]) == 0:
                continue
            concept_lists.append(lst)
    return concept_lists


def domain_load_embeddings(embed_path, domain = None):
    embeddings = {}
    with open(embed_path, encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Loading embeddings"):
            item = json.loads(line)
            if item['type'] not in [domain]:
                continue
            embeddings[item['sentence']] = item['embedding']
    return embeddings


def domain_load_concept_lists(input_file, domain = None):
    concept_lists = []
    with open(input_file, encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            lst = item["extract_concepts"]
            if len(lst) == 1 and len(lst[0]) == 0:
                continue
            if item['type'] not in [domain]:
                continue
            concept_lists.append(lst)
    return concept_lists


def generate_samples(sampler, data_size, difficulty_levels, domain):
    results = []
    concept_text_pool = set()

    with tqdm(total=data_size, desc="Generating samples") as pbar:
        while len(results) < data_size:
            concept_list = sampler.sample_concept_list(size=5, temperature=0.2)
            concept_text = "\n".join(f"{i + 1}. {concept}" for i, concept in enumerate(concept_list))

            if concept_text in concept_text_pool:
                continue

            concept_text_pool.add(concept_text)
            level = random.choice(difficulty_levels)

            prompt = open("prompts/question_generation.txt", "r").read()
            prompt = prompt.replace("{CONCEPTS}", concept_text).replace("{DIFFICULTY}", level).replace("{CATEGORY}", domain)
            
            results.append({
                "foundational_concepts": concept_list,
                "level": level,
                "prompt": prompt,
                "type": domain
            })
            pbar.update(1)

    return results


def save_results(results, output_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate problem concepts dataset')
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to output JSONL file'
    )
    parser.add_argument(
        '--data_size',
        type=int,
        default=2000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--embed_path',
        type=str,
        default="data/embeddings.jsonl",
        help='Path to embeddings file'
    )
    parser.add_argument(
        '--ratio_path',
        type=str,
        default="data/train_fail_ratios.jsonl",
        help='Path to the file that records the ratio of different files'
    )
    args = parser.parse_args()

    # Difficulty levels configuration
    difficulty_levels = ["AMC12", "HMMT-Nov", "HMMT-Feb", "AIME", "USAJMO", "USAMO", "USOJMO", "USOMO"]
    domains = ["Geometry", "Algebra", "Intermediate Algebra", "Precalculus", "Counting & Probability", "Prealgebra", "Number Theory"]
    domain_ratio = json.load(open(args.ratio_path, "r"))
    domain_ratio_normalized = {key: value / sum(domain_ratio.values()) for key, value in domain_ratio.items()}

    ### domain specific processing
    results = []
    for domain in domains:
        # Load data
        embeddings = domain_load_embeddings(args.embed_path, domain=domain)
        concept_lists = domain_load_concept_lists(args.data_path, domain=domain)
        print(f"Finished loading data for domain {domain}")
        
        # Initialize sampler and generate samples
        domain_data_size = int(args.data_size * domain_ratio_normalized[domain])
        sampler = ConceptSampler(concept_lists=concept_lists, concept_embeddings=embeddings)
        print(f"Starting Concepts Extraction for domain {domain} with {domain_data_size} samples.")
        results.extend(generate_samples(sampler, domain_data_size, difficulty_levels, domain))

    save_results(results, args.output_path)


if __name__ == "__main__":
    main()