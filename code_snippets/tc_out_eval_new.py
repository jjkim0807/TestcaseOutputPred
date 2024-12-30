import json
from pathlib import Path


def parse_gen_output(gen_output):
    if gen_output.startswith("```json"):
        gen_output = gen_output[7:]
    elif gen_output.startswith("```"):
        gen_output = gen_output[3:]

    if gen_output.endswith("```"):
        gen_output = gen_output[:-3]

    gen_output = gen_output.strip()

    if "\n```\n" in gen_output:
        gen_output = gen_output.replace("\n```\n", '"')
    elif "\n```" in gen_output:
        gen_output = gen_output.replace("\n```", '"')
    elif "```\n" in gen_output:
        gen_output = gen_output.replace("```\n", '"')
    elif "```" in gen_output:
        gen_output = gen_output.replace("```", '"')

    try:
        gen_output = json.loads(gen_output)["output"]

        if isinstance(gen_output, str):
            gen_output = gen_output.strip()
        else:
            gen_output = str(gen_output)
    except Exception:
        gen_output = gen_output

    return gen_output


def eval_or_str(s: str):
    if s.isdecimal():
        return float(s)
    elif s == "True":
        return True
    elif s == "False":
        return False
    elif s == "None":
        return None
    elif s.startswith('"') and s.endswith('"'):
        return eval_or_str(s[1:-1])
    elif s.startswith("'") and s.endswith("'"):
        return eval_or_str(s[1:-1])
    else:
        try:
            return eval(s)
        except Exception:
            return s


origin_path = Path("results/20240924-gpt-tc_output-none/results_merged_1.json")

target_paths = [
    Path("results/20241104-tc_output-vanilla-pc_lbl-pred/results_merged_1.json"),
    Path("results/20241104-tc_output-bp-pc_lbl-pred/results_merged_1.json"),
    Path("results/20241104-tc_output-vanilla-pc_lbl-gt/results_merged_1.json"),
    Path("results/20241104-tc_output-bp-pc_lbl-gt/results_merged_1.json"),
]


def main(origin_path, target_path):
    with open(origin_path) as f:
        origin_data = json.load(f)

    with open(target_path) as f:
        target_data = json.load(f)

    total_acc = 0
    total_problem_wise_acc = 0
    for origin, target in zip(origin_data, target_data):
        if "tc_output-pred" not in target:
            continue

        gen = target["tc_output-pred"]
        gen = [parse_gen_output(g) for g in gen]
        gt = origin["tc_output-gt"]

        acc = sum(
            [eval_or_str(g.strip()) == eval_or_str(t.strip()) for g, t in zip(gen, gt)]
        ) / len(gt)

        total_acc += acc
        total_problem_wise_acc += acc == 1

    total_acc /= len(origin_data)
    total_problem_wise_acc /= len(origin_data)

    print(f"{total_acc * 100:.2f},{total_problem_wise_acc * 100:.2f}")

    new_path = target_path.parent / "eval_result_summary.json"
    with open(new_path, "w") as f:
        json.dump(
            {
                "total": total_acc,
                "total_problem_wise": total_problem_wise_acc,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    for target_path in target_paths:
        main(origin_path, target_path)
