import json

from datasets import load_dataset


def sampling(dataset, seed, num_samples):
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(num_samples))

    return dataset


def main():
    dataset = load_dataset("deepmind/code_contests", split="train")

    def filter(datum):
        lang = datum["solutions"]["language"]
        return 3 in lang

    dataset = dataset.filter(filter)

    def map_fn(datum, id):
        lang = datum["solutions"]["language"]
        idx = [i for i, x in enumerate(lang) if x == 3]

        # filter the solutions based on the indices
        datum["solutions"] = {
            k: [v[i] for i in idx] for k, v in datum["solutions"].items()
        }

        result = {
            "task_id": f"code_contests/{id}/{datum['name']}",
            "description": datum["description"],
            "code": datum["solutions"]["solution"][0],
            "public_tests": datum["public_tests"],
            "private_tests": datum["private_tests"],
            "generated_tests": datum["generated_tests"],
        }

        return result

    dataset = dataset.map(map_fn, with_indices=True)

    dataset = sampling(dataset, seed=42, num_samples=4000)

    path = "data/code_contests-sampled.jsonl"
    with open(path, "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
