import ast
import json
from copy import deepcopy

import numpy as np
import tqdm
from apply_cutoff import apply_cutoff


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def compute_metrics_from_results(results, k_list=[1, 5]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist()
        for k in ks
        if (total >= k).all()
    }
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k


def extract_instance_results(results):
    instance_wise_grades = {}
    for task_id, res in results.items():
        instance_wise_grades[task_id] = []
        for generation in res:
            instance_wise_grades[task_id].append(all([g > 0 for g in generation]))

    instance_wise_grades = [
        v for _, v in sorted(instance_wise_grades.items(), key=lambda item: item[0])
    ]
    return instance_wise_grades


def parse_assert_statement(statement):
    """
    Parse a Python assert statement and extract the expected output
    from the right side of the '==' operator as a string.

    :param statement: A string containing the assert statement.
    :return: The expected output from the assert statement as a string.
    """
    try:
        parsed = ast.parse(statement, mode="exec")
    except SyntaxError:
        return "Invalid syntax"

    if len(parsed.body) == 0:
        return "Empty statement"

    if not isinstance(parsed.body[0], ast.Assert):
        return "Not an assert statement"

    comparison = parsed.body[0].test

    if not isinstance(comparison, ast.Compare) or not isinstance(
        comparison.ops[0], ast.Eq
    ):
        return "Not an equality assertion"

    # Extract and return the right side of the '==' operator as a string
    return ast.get_source_segment(statement, comparison.comparators[0])


def check_testcase_output(testcase_str, expected_output):

    if len(testcase_str.splitlines()) > 1:
        for line in testcase_str.splitlines():
            if line.startswith("#"):
                continue
            if "assert" in line:
                testcase_str = line
                break

    testcase_str = testcase_str.strip()

    if "assert" in testcase_str:
        testcase_output_str = str(parse_assert_statement(testcase_str))

    else:
        testcase_output_str = testcase_str

    global_result = None

    try:
        testcase_output_eval = eval(testcase_output_str)
    except:
        try:
            testcase_output_eval = json.loads(testcase_output_str)
        except:
            testcase_output_eval = testcase_output_str

    try:
        expected_output_eval = eval(expected_output)
    except:
        try:
            expected_output_eval = json.loads(expected_output)
        except:
            expected_output_eval = expected_output

    if global_result is None:
        global_result = testcase_output_eval == expected_output_eval

    return global_result


def test_output_metrics(
    samples,
    generations,
    k_list=[1, 2, 5],
):
    num_samples = len(samples)
    results = []
    for idx in tqdm.tqdm(list(range(num_samples))):
        idx_results = []
        sample = samples[idx]
        extracted_generation_list = generations[idx]
        for extracted_generation in extracted_generation_list:
            global_result = check_testcase_output(
                extracted_generation, sample["output"]
            )
            idx_results.append([global_result])
        results.append(idx_results)

    results = {result_idx: results[result_idx] for result_idx in range(len(results))}

    metrics = compute_metrics_from_results(results, k_list=k_list)

    return [metrics, results]


def self_consistency(gen, scores):
    gen_ith = [g.strip() for g in gen]
    candidates = list(set(gen_ith))
    votes = {c: 0 for c in candidates}
    for j, c in enumerate(gen_ith):
        votes[c] += scores[j]

    max_votes = max(votes.values())
    winners = [c for c, v in votes.items() if v == max_votes]

    return winners


def selection_to_scores(raw_selection):
    scores = [0] * 5
    for k, v in raw_selection.items():
        first = int(k[0])
        second = int(k[2])
        if v == 1:
            scores[first] += 2
        elif v == 2:
            scores[second] += 2
        elif v == "tie":
            scores[first] += 0.5
            scores[second] += 0.5
        else:
            # raise ValueError("Invalid selection")
            pass

    return scores


def main(path, rank_path=None):
    print(f"path: {path}")
    with open(path, "r") as f:
        data = json.load(f)

    rank_data = None
    if rank_path is not None:
        with open(rank_path, "r") as f:
            rank_data = json.load(f)

    samples = []
    generations = []
    sc_generations = []
    sc_rank_generations = []
    for i, datum in enumerate(data):
        gt = datum["tc_output-gt"][0]
        gen = datum["tc_output-pred"]
        if rank_data is None:
            scores = [1, 1, 1, 1, 1]
        else:
            selection = rank_data[i]["ranking_pc-parsed"]
            scores = selection_to_scores(selection)

        for i, _gt in enumerate(gt):
            samples.append({"output": _gt})
            flatten_gen = [g[i] for g in gen]
            generations.append(flatten_gen)
            sc_generations.append(self_consistency(flatten_gen, scores=[1, 1, 1, 1, 1]))
            sc_rank_generations.append(self_consistency(flatten_gen, scores=scores))

    metrics, results = test_output_metrics(samples, generations, k_list=[1, 5])
    print(f"\\num{{{metrics['pass@1']}}}")

    metrics, results = test_output_metrics(samples, sc_generations, k_list=[1, 5])

    print(f"\\num{{{metrics['pass@1']}}}")

    metrics, results = test_output_metrics(samples, sc_rank_generations, k_list=[1, 5])
    print(f"\\num{{{metrics['pass@1']}}}")
    print("=====================================")


if __name__ == "__main__":
    paths = [
        ("results/20241205-gpt4-best_ours-no_trace/results_merged_1.json", None),
    ]

    for path, rank_path in paths:
        main(path, rank_path)
        cutoff_path = apply_cutoff(path)
        cutoff_rank_path = apply_cutoff(rank_path) if rank_path is not None else None
        main(cutoff_path, cutoff_rank_path)
