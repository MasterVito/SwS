import os
import sys
import json
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from pdb import set_trace
from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
NUM_PROC = os.cpu_count() // 4 * 3


def extract_concepts(sample):
    try:
        code_str = sample['prediction'][0].split("```python\n")[1].split("```")[0]
        local_vars = {}
        exec(code_str, {}, local_vars)
        for k in ['foundational_concepts', 'fundamental_knowledge_points', 'core_concepts', 'fundamental_concepts']:
            if k in local_vars.keys():
                foundational_concepts = local_vars[k]
        return {"extract_concepts": foundational_concepts}
    except:
        foundational_concepts = [""]
        return {"extract_concepts": foundational_concepts}
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--concepts_extract_prompt_path", default='prompts/concept_extraction.txt', type=str)
    parser.add_argument("--tensor_parallel_size", default=1, type=int, choices=[1, 2, 4, 8])
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--gpu_memory_utilization", default=1.0, type=float) 
    parser.add_argument("--swap_space", default=4, type=int)
    parser.add_argument("--max_tokens", default=512, type=int)
    args = parser.parse_args()
    available_gpus = torch.cuda.device_count()
    sample_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, n=args.n_sampling, max_tokens=args.max_tokens, stop=["<end>", "\] \] \]", "\n\n\n"])

    dataset = load_dataset('json', data_files=args.input_path, split='train')
    concepts_extract_prompt = open(args.concepts_extract_prompt_path, "r").read()
    dataset = dataset.map(lambda x: {'prompt': concepts_extract_prompt.replace(r"{REPLACE}", x['prompt'])}, num_proc=64)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    add_generation_prompt = False if "llama" in args.model_name_or_path.lower() else True
    dataset = dataset.map(lambda x: {"vllm_input": tokenizer.apply_chat_template([{"content": x['prompt'], 'role': 'user'}], add_generation_prompt=add_generation_prompt, tokenize=False)}, num_proc=64)

    ### testing inputs
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    test_inputs = dataset["vllm_input"][:5]
    for id in range(len(test_inputs)):
        print(f">>> Testing input {id + 1}: \n{test_inputs[id]}")
    
    ### testing outputs
    model = LLM(args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, swap_space=args.swap_space)
    test_outputs = model.generate(test_inputs, sample_params)
    test_outputs = [random.choice([_.outputs[i].text for i in range(len(_.outputs))]) for _ in test_outputs]
    for id in range(len(test_inputs)):
        print(f">>> Testing Output {id + 1}: \n{test_outputs[id]}")
    
    ### start inference
    print(f">>> Starting inference...")
    outputs = model.generate(dataset["vllm_input"], sample_params)
    outputs = [[_.outputs[i].text.split("<|endoftext|>")[0] for i in range(len(_.outputs))] for _ in outputs]

    ### Verify and save the results
    print(f">>> Finishing inference")
    dataset = dataset.add_column("prediction", outputs)
    dataset = dataset.remove_columns(["vllm_input"])
    dataset = dataset.map(extract_concepts, num_proc=64)
    dataset.to_json(args.output_path, num_proc = os.cpu_count() // 2)