import os
import sys
import argparse
from pdb import set_trace
from datasets import load_dataset

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_path', type=str)
    parser.add_argument('--num_process', type=int, default = 50)
    args = parser.parse_args()
    return args
args = parser()

all_sampled_concepts = load_dataset("json", data_files=[os.path.join(args.split_path, f"{x+1}-of-{args.num_process}.jsonl") for x in range(args.num_process)], num_proc=min(args.num_process, 64), split = "train")
all_sampled_concepts.to_json(args.split_path + ".jsonl")