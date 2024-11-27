import json

import datasets


def main():
    hf_path = "livecodebench/test_generation"
    dataset = datasets.load_dataset(hf_path)["test"]

    data = {}
    for datum in dataset:
        question_id = datum["question_id"]
        test_id = datum["test_id"]
        question_title = datum["question_title"]
        question_content = datum["question_content"]
        starter_code = datum["starter_code"]
        function_name = datum["function_name"]
        difficulty = datum["difficulty"]
        test = datum["test"]

        id = f"{question_id}-{question_title}"
        if id not in data:
            data[id] = {
                "id": id,
                "question_id": question_id,
                "question_title": question_title,
                "question_content": question_content,
                "starter_code": starter_code,
                "function_name": function_name,
                "difficulty": difficulty,
                "test": [],
            }

        while len(data[id]["test"]) <= test_id:
            data[id]["test"].append(None)
        data[id]["test"][test_id] = json.loads(test)

    data = list(data.values())
    data = sorted(data, key=lambda x: x["id"])

    json_path = "data/livecodebench_test.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
