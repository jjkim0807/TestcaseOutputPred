import json
from collections import defaultdict
from itertools import combinations
from math import comb
from typing import List, Optional

import numpy as np


def acc_at_k(accs: list[float], k: int, scores: list[int]) -> float:
    # Mapping accuracies to their corresponding scores
    score_mapping = defaultdict(list)
    score_count = defaultdict(int)

    for score, acc in zip(scores, accs):
        score_mapping[score].append(acc)
        score_count[score] += 1

    # Sort scores in descending order
    sorted_scores = sorted(score_count.keys(), reverse=True)
    cumulative_sum = 0
    max_acc = 0

    for score in sorted_scores:
        current_count = score_count[score]
        cumulative_sum += current_count

        if cumulative_sum > k:
            # Handle the case where only part of the current score's accuracies are included
            excess = cumulative_sum - k
            acc_values = score_mapping[score]

            # Generate all combinations to exclude `excess` items
            acc_combinations = list(combinations(acc_values, current_count - excess))
            mean_combination_acc = np.mean([max(comb) for comb in acc_combinations])

            # Update the maximum accuracy so far
            max_acc = max(max_acc, mean_combination_acc)
            return max_acc
        else:
            # Include all accuracies for the current score
            max_acc = max(max_acc, max(score_mapping[score]))

            if cumulative_sum == k:
                return max_acc

    return max_acc


def selection_to_scores(raw_selection):
    scores = [0] * 5
    for k, v in raw_selection.items():
        first = int(k[0])
        second = int(k[2])
        if v == 1:
            scores[first] += 1
        elif v == 2:
            scores[second] += 1
        else:
            # raise ValueError("Invalid selection")
            pass

    return scores


def main():
    rank_path = "results/20241203-lcb-gpt4omini-zero/results_merged_1.json"
    tc_out_path = "results/20241203-lcb-gpt4omini-zero/results_merged_1.json"

    with open(rank_path, "r") as f:
        rank_data = json.load(f)

    with open(tc_out_path, "r") as f:
        tc_out_data = json.load(f)

    acc_log_dict = defaultdict(list)
    for r, t in zip(rank_data, tc_out_data):
        raw_selection = r["ranking_pc-parsed"]
        # raw_selection = r["ranking_pc"]
        gen = t["tc_output-pred"]
        gt = t["tc_output-gt"][0]

        scores = selection_to_scores(raw_selection)
        # scores = [1,1,1,1,1]
        for k in [1, 2, 5]:
            evals = []
            for g in gen:
                evals.append(
                    sum([g.strip() == t.strip() for g, t in zip(g, gt)]) / len(gt) # == 1
                )

            acc = acc_at_k(evals, k, scores)
            acc_log_dict[str(k)].append(acc)

    total_acc_dict = {}
    for k in [1, 2, 5]:
        total_acc_dict[str(k)] = sum(acc_log_dict[str(k)]) / len(acc_log_dict[str(k)])

    print(total_acc_dict)


if __name__ == "__main__":
    main()
