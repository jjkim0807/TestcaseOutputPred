from pathlib import Path

import simplejson as json

path = Path("results/20240924-tc_output-none/results_merged_1.json")
data = json.load(open(path))
tc_output_key = "tc_output-pred-none"

# Adjust the parsing of generated output to handle non-JSON strings safely
logs = []
for problem in data:
    tc_gt = problem.get("tc_output-gt", [])
    tc_gen = problem.get(tc_output_key, [])

    assert len(tc_gt) == len(
        tc_gen
    ), f"Test case count mismatch for problem {problem['id']}"
    assert len(tc_gt) == 3

    correct_count = 0
    incorrect_count = 0
    log = []
    for i, (gt, gen) in enumerate(zip(tc_gt, tc_gen)):
        gt_output = gt.strip()  # Ground truth

        # if gen starts with ```json remove it
        gen_searched = gen
        if gen_searched.startswith("```json"):
            gen_searched = gen_searched[7:]
        elif gen_searched.startswith("```"):
            gen_searched = gen_searched[3:]

        # if gen ends with ``` remove it
        if gen_searched.endswith("```"):
            gen_searched = gen_searched[:-3]

        gen_searched = gen_searched.strip()

        # if ```found in middle, replace it with "
        if "\n```\n" in gen_searched:
            gen_searched = gen_searched.replace("\n```\n", '"')
        elif "\n```" in gen_searched:
            gen_searched = gen_searched.replace("\n```", '"')
        elif "```\n" in gen_searched:
            gen_searched = gen_searched.replace("```\n", '"')
        elif "```" in gen_searched:
            gen_searched = gen_searched.replace("```", '"')

        if "\n```\n" in gen_searched:
            gen_searched = gen_searched.replace("\n```\n", '"')
        elif "\n```" in gen_searched:
            gen_searched = gen_searched.replace("\n```", '"')
        elif "```\n" in gen_searched:
            gen_searched = gen_searched.replace("```\n", '"')
        elif "```" in gen_searched:
            gen_searched = gen_searched.replace("```", '"')

        try:
            gen_output = json.loads(gen_searched)["output"]
            if isinstance(gen_output, str):
                gen_output = gen_output.strip()
            else:
                gen_output = str(gen_output)

            if gt_output == gen_output:
                log.append(True)
                correct_count += 1
            else:
                log.append(False)
                incorrect_count += 1
        except json.JSONDecodeError:
            log.append(False)
            incorrect_count += 1

    logs.append(log)

# Calculate accuracy
acc_list = list(map(lambda x: sum(x) / len(x), logs))
avg_acc = sum(acc_list) / len(acc_list)

# save logs and acc_list in same file
new_data = []
for i, datum in enumerate(data):
    id = datum["id"]
    new_datum = {
        "id": id,
        "tc_output-gt": datum["tc_output-gt"],
        tc_output_key: datum[tc_output_key],
        "log": logs[i],
        "accuracy": acc_list[i],
    }

    new_data.append(new_datum)

new_path = path.parent / "eval_result.json"
with open(new_path, "w") as f:
    json.dump(new_data, f, indent=2)

print(f"Accuracy: {avg_acc:.2%}")
with open(path.parent / "accuracy.txt", "w") as f:
    f.write(f"Accuracy: {avg_acc:.2%}")
