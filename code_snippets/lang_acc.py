import json
from pathlib import Path

origin_path = Path("data/code_contests-test.json")

target_paths = [
    Path("results/20240924-tc_output-none/eval_result.json"),
    Path("results/20240924-tc_output-gold-code/eval_result.json"),
    Path("results/20240924-tc_output-gold-lbl/eval_result.json"),
    Path("results/20240924-tc_output-gold-func/eval_result.json"),
    Path("results/20240924-tc_output-gold-nl/eval_result.json"),
    Path("results/20240924-tc_output-pred-code/eval_result.json"),
    Path("results/20240924-tc_output-pred-lbl/eval_result.json"),
    Path("results/20240924-tc_output-pred-func/eval_result.json"),
    Path("results/20240924-tc_output-pred-nl/eval_result.json"),
    Path("results/20240924-tc_output-gold-pdb-lbl/eval_result.json"),
    Path("results/20240924-tc_output-gold-pdb-func/eval_result.json"),
]


def main(origin_path, target_path):

    with open(origin_path) as f:
        origin_data = json.load(f)

    with open(target_path) as f:
        target_data = json.load(f)

    total_acc = 0
    total_problem_wise_acc = 0
    python_acc = []
    cpp_acc = []
    python_problem_wise_acc = []
    cpp_problem_wise_acc = []
    python_high_difficulty_subset_difficulty = []
    python_high_difficulty_subset_acc = []
    for origin, target in zip(origin_data, target_data):
        total_acc += target["accuracy"]
        total_problem_wise_acc += target["accuracy"] == 1

        lang = origin["language"]
        if lang == 3:
            python_acc.append(target["accuracy"])
            python_problem_wise_acc.append(target["accuracy"] == 1)
            if origin["cf_rating"] >= 2500:
                python_high_difficulty_subset_difficulty.append(origin["cf_rating"])
                python_high_difficulty_subset_acc.append(target["accuracy"])

        elif lang == 2:
            cpp_acc.append(target["accuracy"])
            cpp_problem_wise_acc.append(target["accuracy"] == 1)
        else:
            raise ValueError("Invalid language")

    total_acc /= len(origin_data)
    total_problem_wise_acc /= len(origin_data)
    python_acc = sum(python_acc) / len(python_acc)
    cpp_acc = sum(cpp_acc) / len(cpp_acc)
    python_problem_wise_acc = sum(python_problem_wise_acc) / len(
        python_problem_wise_acc
    )
    cpp_problem_wise_acc = sum(cpp_problem_wise_acc) / len(cpp_problem_wise_acc)
    python_high_difficulty_subset_len = len(python_high_difficulty_subset_acc)
    python_high_difficulty_subset_difficulty = sum(
        python_high_difficulty_subset_difficulty
    ) / len(python_high_difficulty_subset_difficulty)
    python_high_difficulty_subset_acc = sum(python_high_difficulty_subset_acc) / len(
        python_high_difficulty_subset_acc
    )

    print(f"{total_problem_wise_acc * 100:.2f}")

    new_path = target_path.parent / "eval_result2.json"
    with open(new_path, "w") as f:
        json.dump(
            {
                "total": total_acc,
                "python": python_acc,
                "cpp": cpp_acc,
                "python_problem_wise": python_problem_wise_acc,
                "cpp_problem_wise": cpp_problem_wise_acc,
                # "python_high_difficulty_subset": python_high_difficulty_subset_acc,
            },
            f,
            indent=4,
        )


for target_path in target_paths:
    main(origin_path, target_path)
