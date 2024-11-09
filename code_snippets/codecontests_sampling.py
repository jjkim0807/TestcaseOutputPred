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

    def map_fn(datum):
        # find all the indices of the language 3
        lang = datum["solutions"]["language"]
        idx = [i for i, x in enumerate(lang) if x == 3]

        # filter the solutions based on the indices
        datum["solutions"] = {
            k: [v[i] for i in idx] for k, v in datum["solutions"].items()
        }

        return datum

    dataset = dataset.map(map_fn)

    dataset = sampling(dataset, seed=42, num_samples=4000)

    path = "data/code_contests-sampled"
    dataset.save_to_disk(path)


if __name__ == "__main__":
    main()
