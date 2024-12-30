import json

from tc_out_eval import calc_acc, parse_gen_output

paths = {
    "lbl_bp": "results/20240930-gpt-tc_output-pred-lbl-bpt/results_merged_1.json",
    "lbl_vanilla": "results/20240924-gpt-tc_output-pred-lbl/results_merged_1.json",
    "nl_bp": "results/20240930-gpt-tc_output-pred-nl-bpt/results_merged_1.json",
    "nl_vanilla": "results/20240924-gpt-tc_output-pred-nl/results_merged_1.json",
}
output_path = "case_study-oxox.md"

data = {}
for key, path in paths.items():
    with open(path, "r") as f:
        data[key] = json.load(f)

# 4x4 matrix
matrix = {}
for a in [True, False]:
    for b in [True, False]:
        for c in [True, False]:
            for d in [True, False]:
                matrix[(a, b, c, d)] = 0

total_acc = {
    "lbl_bp": 0,
    "lbl_vanilla": 0,
    "nl_bp": 0,
    "nl_vanilla": 0,
}

output = []
for i in range(len(data["lbl_bp"])):
    lbl_bp = data["lbl_bp"][i]
    lbl_vanilla = data["lbl_vanilla"][i]
    nl_bp = data["nl_bp"][i]
    nl_vanilla = data["nl_vanilla"][i]

    assert lbl_bp["id"] == lbl_vanilla["id"]
    assert lbl_bp["id"] == nl_bp["id"]
    assert lbl_bp["id"] == nl_vanilla["id"]
    id = lbl_bp["id"]
    problem = lbl_bp["problem"]
    tc_output_gt = lbl_bp["tc_output-gt"]

    pseudocode = {
        "lbl_bp": lbl_bp["pseudocode"],
        "lbl_vanilla": lbl_vanilla["pseudocode"],
        "nl_bp": nl_bp["pseudocode"],
        "nl_vanilla": nl_vanilla["pseudocode"],
    }

    tc_output_gen = {
        "lbl_bp": lbl_bp["tc_output-gen"],
        "lbl_vanilla": list(map(parse_gen_output, lbl_vanilla["tc_output-gen"])),
        "nl_bp": nl_bp["tc_output-gen"],
        "nl_vanilla": list(map(parse_gen_output, nl_vanilla["tc_output-gen"])),
    }

    tc_output_gen_raw = {
        "lbl_bp": lbl_bp["tc_output-gen_raw"],
        "lbl_vanilla": lbl_vanilla["tc_output-gen_raw"],
        "nl_bp": nl_bp["tc_output-gen_raw"],
        "nl_vanilla": nl_vanilla["tc_output-gen_raw"],
    }

    acc_dict = {}
    for key in pseudocode.keys():
        acc_dict[key] = calc_acc(tc_output_gt, tc_output_gen[key])
        total_acc[key] += acc_dict[key]

    key = tuple([v > 0 for v in acc_dict.values()])
    matrix[key] += 1

    if key == (True, False, True, False):
        output_str = f"# ID\n{id}\n\n" f"# Problem\n{problem}\n\n"

        output_str += "# Pseudocode\n"
        for k, v in pseudocode.items():
            output_str += f"## {k}\n" f"```\n{v}```\n\n"

        for i in range(len(tc_output_gt)):
            for key in pseudocode.keys():
                output_str += f"# Test case output {i}th {key}\n"
                output_str += (
                    f"## GT\n{tc_output_gt[i]}\n\n"
                    f"## Generated\n{tc_output_gen[key][i]}\n\n"
                    f"## Raw\n{tc_output_gen_raw[key][i]}\n\n"
                )

        output.append(output_str)

with open(output_path, "w") as f:
    f.write("\n\n".join(output))

for key, value in matrix.items():
    for k in key:
        if k == True:
            print("o\t", end="")
        else:
            print("x\t", end="")

    print(value)

for key, value in total_acc.items():
    print(key, value / len(data[key]))
