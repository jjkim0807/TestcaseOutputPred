import json
from pathlib import Path

import regex as re

path = Path("results/20240924-gpt-pc_pred-func/results_merged_1.json")
key = "pc_pred-func"
with open(path, "r") as f:
    data = json.load(f)

for datum in data:
    pseudocodes: str = datum[key][0]

    try:
        pattern = re.compile(r"```[a-z]+\n(.*?)```", re.DOTALL)
        code_block = pattern.findall(pseudocodes)[0]
    except IndexError:
        if pseudocodes.startswith("```python"):
            code_block = pseudocodes[len("```python") :]
        elif pseudocodes.startswith("```plaintext"):
            code_block = pseudocodes[len("```plaintext") :]
        elif pseudocodes.startswith("```"):
            code_block = pseudocodes[len("```") :]
        else:
            code_block = pseudocodes

    assert code_block is not None and code_block.strip() != ""
    datum[key] = code_block

new_path = path.parent / "parsed.json"
if not new_path.parent.exists():
    new_path.parent.mkdir(parents=True)
with open(new_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
