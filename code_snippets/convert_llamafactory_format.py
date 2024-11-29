import json

path = "results/20241108-pc-train/results_merged_1.json"
with open(path) as f:
    data = json.load(f)

converted_list = []
for datum in data:
    id = datum["id"]
    problem = datum["problem"]
    chosen = datum["pc_lbl-gt"]
    rejected = datum["pc_lbl-pred"]

    prompt = """\
# Inputs
## Problem
```plaintext
{problem}
```

## Pseudocode 1
```plaintext
{pc_1}
```

## Pseudocode 2
```plaintext
{pc_2}
```

# Instruction
I provided you a coding problem and its two pseudocode solutions.
You need to choose the correct pseudocode solution.
The output should be "1" or "2". No other output is allowed."""

    converted_1 = {
        "conversations": [
            {
                "from": "human",
                "value": prompt.format(problem=problem, pc_1=chosen, pc_2=rejected),
            }
        ],
        "chosen": {"from": "gpt", "value": "1"},
        "rejected": {"from": "gpt", "value": "2"},
    }

    converted_2 = {
        "conversations": [
            {
                "from": "human",
                "value": prompt.format(problem=problem, pc_1=rejected, pc_2=chosen),
            }
        ],
        "chosen": {"from": "gpt", "value": "2"},
        "rejected": {"from": "gpt", "value": "1"},
    }

    converted_list.append(converted_1)
    converted_list.append(converted_2)

# Save the converted data
converted_path = path.replace(".json", "_llamafactory.json")
with open(converted_path, "w") as f:
    json.dump(converted_list, f, indent=2, ensure_ascii=False)
