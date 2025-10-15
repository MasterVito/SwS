import argparse
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from str2bool import str2bool


@dataclass
class ProcessedItem:
    prompt: str
    completion: str


def parse_judgement(judgement_text: str) -> str:
    judgement_text = judgement_text.lower()
    if judgement_text.startswith("perfect") or "perfect" in judgement_text:
        return "perfect"
    elif judgement_text.startswith("acceptable") or "acceptable" in judgement_text:
        return "acceptable"
    return "bad"


def load_and_process_file(file_path: str) -> Tuple[List[Dict], List[str]]:
    items = []
    ratings = []
    with open(file_path, encoding="utf-8") as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            judgement = parse_judgement(json_obj["judgement"])
            assert judgement in ["perfect", "acceptable", "bad"]
            ratings.append(judgement)
            items.append(json_obj)
    return items, ratings


def process_items(items: List[Dict], ratings_list: List[List[str]],
                  only_perfect: bool) -> List[ProcessedItem]:
    processed_items = []
    n_rewards = len(ratings_list)

    for idx in range(len(items)):
        all_perfect = all(
            ratings_list[reward_idx][idx] == "perfect"
            for reward_idx in range(n_rewards)
        )
        has_perfect_no_bad = (
                any(ratings_list[reward_idx][idx] == "perfect"
                    for reward_idx in range(n_rewards)) and
                all(ratings_list[reward_idx][idx] != "bad"
                    for reward_idx in range(n_rewards))
        )

        if (only_perfect and all_perfect) or (not only_perfect and has_perfect_no_bad):
            processed_items.append(ProcessedItem(
                prompt=items[idx]["prompt"],
                completion=items[idx]["rationale_and_problem"]
            ))

    return processed_items


def main():
    parser = argparse.ArgumentParser(description='Process and filter completion data')
    parser.add_argument('--template', type=str,
                        required=True,
                        help='Template for input files')
    parser.add_argument('--output_path', type=str,
                        required=True,
                        help='Path for output file')
    parser.add_argument('--only_perfect', type=str2bool,
                        default=True,
                        help='Only include items rated as perfect by all rewards')
    parser.add_argument('--n_rewards', type=int, default=2,
                        help='Number of reward models')
    args = parser.parse_args()

    items_list = []
    ratings_list = []

    # Load and process files for each reward model
    for reward_idx in range(args.n_rewards):
        file_path = args.template.format(reward_idx)
        items, ratings = load_and_process_file(file_path)
        if reward_idx == 0:
            items_list = items
        ratings_list.append(ratings)

    # Verify consistency
    assert all(len(ratings) == len(items_list)
               for ratings in ratings_list)

    # Process items
    processed_items = process_items(
        items_list, ratings_list,
        args.only_perfect
    )
    final_items = [vars(item) for item in processed_items]

    # Save results
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in final_items:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()