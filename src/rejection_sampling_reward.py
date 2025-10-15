import fire
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import torch
from data_synthesis import rationale_judgement_prompt, extract_final_judgement


def main(
    data_path,
    output_path,
    model_path,
    dtype="bfloat16",
    n_gpus=8,
    temperature=0.0,
    max_len=2048,
    use_chat_template=False,
    seed=42,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = LLM(
        model=model_path,
        tokenizer=model_path,
        tokenizer_mode="slow",
        dtype=dtype,
        tensor_parallel_size=n_gpus,
        seed=seed,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_len,
    )

    prompts = []
    items = []
    rationales = []

    with open(data_path, encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            foundational_concepts = item["foundational_concepts"]
            level = item["level"]
            rationale_and_problem = item["completion"]
            prompt = rationale_judgement_prompt(
                concepts=foundational_concepts,
                level=level,
                rationale_and_problem=rationale_and_problem,
            )
            if use_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            items.append(item)
            rationales.append(rationale_and_problem)

    with torch.no_grad():
        completions = model.generate(prompts, sampling_params)
        completions = [completion.outputs[0].text for completion in completions]

    for item, rationale, completion in zip(items, rationales, completions):
        final_judgement = extract_final_judgement(completion)
        item["rationale_and_problem"] = rationale
        item["judgement"] = final_judgement
        item["completion"] = completion

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    fire.Fire(main)