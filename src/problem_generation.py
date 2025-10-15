import argparse
import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from str2bool import str2bool
from pdb import set_trace
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to store cached outputs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the pretrained model.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type to use for the model (e.g., fp16, bf16, etc.).")
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs to use for tensor parallelism.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling for generation.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum number of tokens to generate.")
    parser.add_argument("--use_chat_template", type=str2bool, default=False)
    parser.add_argument("--total_chunk", default=-1, type=int, help="Set to >0 to split the dataset to small chunks.")
    parser.add_argument("--current_chunk", default=-1, type=int, help="Set to >0 to choose the current chunk.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # Load the tokenizer for LLaMA or any model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load inference framework
    model = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path,
        tokenizer_mode="slow",
        dtype=args.dtype,
        tensor_parallel_size=args.n_gpus,
        enforce_eager=True,
    )

    # Setup sampling parameters for model generation
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_len,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    prompts = []
    items = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in tqdm(f.readlines(), total=len(f.readlines()), desc="Loading dataset"):
            item = json.loads(line)
            prompt = item["prompt"]
            if args.use_chat_template:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            items.append(item)
            
    if args.total_chunk > 0 and args.current_chunk >= 0:
        start = int(args.current_chunk * len(prompts) / args.total_chunk)
        end = min(int((args.current_chunk + 1) * len(prompts) / args.total_chunk), len(prompts))
    else:
        start = 0
        end = len(prompts)
    print(f">>> Current chunk start id: {start}, end id: {end}.")

    with torch.no_grad():
        completions = model.generate(prompts[start:end], sampling_params)
        completions = [completion.outputs[0].text for completion in completions]
        # from pdb import set_trace; set_trace()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item, completion in zip(items[start:end], completions):
            item["completion"] = completion
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
