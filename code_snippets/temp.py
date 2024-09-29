import json
from pathlib import Path

import regex as re

path = Path("results/20240924-gpt-pc_gold/results_merged_1.json")
with open(path, "r") as f:
    data = json.load(f)

for datum in data:
    pseudocodes: str = datum["pseudocodes"][0]

    # there are three code blocks with backticks.
    # ex) "1. line by line pseudocode\n```python\ndef solve():\n    n, m = map(int, input().split())  # Read the number of strings and their length\n    s = []  # Initialize an empty list to store the strings\n    d = {}  # Initialize an empty dictionary to store the original index of each string\n\n    for i in range(n):  # Loop over the number of strings\n        s.append(list(input()))  # Read a string and convert it into a list of characters\n\n    for j in range(n):  # Loop over the strings\n        for k in range(len(s[i])):  # Loop over the characters in each string\n            if (k + 1) % 2 == 0:  # Check if the character is at an even position\n                s[j][k] = chr(90 - (ord(s[j][k]) + 26 - 91))  # Convert the character to its corresponding descending order\n\n    for i in range(len(s)):  # Loop over the strings\n        d[''.join(s[i])] = i + 1  # Store the original index of each string in the dictionary\n\n    s.sort()  # Sort the list of strings based on their modified characters\n\n    for i in s:  # Loop over the sorted strings\n        print(d[''.join(i)], end=\" \")  # Print the original index of each string\n\n\ndef main():\n    solve()\n\n\nif __name__ == \"__main__\":\n    main()\n\n```\n2. middle level abstraction pseudocode\n```python\ndef solve():\n    n, m = read_input_parameters()  # Read the number of strings and their length\n    s = initialize_string_list(n)  # Initialize an empty list to store the strings\n    d = create_index_dictionary(s)  # Create a dictionary to store the original index of each string\n\n    modify_strings_for_descending_order(s)  # Modify the characters in each string for descending order\n    sort_strings(s)  # Sort the list of strings based on their modified characters\n\n    print_original_indices(d, s)  # Print the original indices of the sorted strings\n\n\ndef main():\n    solve()\n\n\nif __name__ == \"__main__\":\n    main()\n\n```\n3. Pure natural language pseudocode\n```plaintext\nFunction solve:\n    Read the number of strings and their length from input.\n    Initialize an empty list to store the strings.\n    Create a dictionary to store the original index of each string.\n\n    Loop over the strings, modifying characters at even positions for descending order.\n    Sort the list of strings based on their modified characters.\n\n    Print the original indices of the sorted strings.\n\n\nFunction main:\n    Call function solve to execute the program.\n\n\n```"
    # split the pseudocode into three parts
    pattern = re.compile(r"```[a-z]+\n(.*?)```", re.DOTALL)
    code_blocks = pattern.findall(pseudocodes)
    if len(code_blocks) != 3:
        code_blocks = pattern.findall(pseudocodes + "```")  # for the last code block

    datum["pc_lbl"] = code_blocks[0]
    datum["pc_func"] = code_blocks[1]
    datum["pc_nl"] = code_blocks[2]

    testcases = datum["testcases"]
    tc_input = []
    tc_output = []
    for testcase in testcases:
        tc_input.append(testcase["input"])
        tc_output.append(testcase["output"])

    datum["tc_input"] = tc_input
    datum["tc_output"] = tc_output

new_path = path.parent / "splitted.json"
if not new_path.parent.exists():
    new_path.parent.mkdir(parents=True)
with open(new_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
