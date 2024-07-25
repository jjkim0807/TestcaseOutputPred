import json

# read json file in path
path = "results/gold_code-gold_tc/results_merged_1.json"
with open(path, "r") as f:
    data = json.load(f)

# the data is a list of dictionaries
# for each dict, read exec_result field
# the field is a list of string
# calculate the percentage of "Exit Code: 0" in the list
# and calculate the average value of above percentage for all dicts
# print the average value
exit_code_0 = []
for d in data:
    exec_result = d["exec_result"]
    count = 0
    for e in exec_result:
        if "Exit Code: 0" in e:
            count += 1
    exit_code_0.append(count / len(exec_result))

print(sum(exit_code_0) / len(exit_code_0))

for i, percent in enumerate(exit_code_0):
    data[i]["exit_code_0_percent"] = percent

# write the data to a new json file
path = "results/gold_code-gold_tc/results_merged_1_new.json"
with open(path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

# calculate the percentage of dict that all exec_result are "Exit Code: 0"
# and print the percentage
count = 0
for percent in exit_code_0:
    if percent == 1:
        count += 1

print(count / len(exit_code_0))
