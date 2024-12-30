import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from lcb_runner.evaluation.compute_scores import compute_scores as lcb_compute_scores
from lcb_runner.evaluation.compute_scores import get_parser as lcb_cs_parser
from lcb_runner.runner.custom_evaluator import main as lcb_eval
from pydantic import BaseModel


def self_consistency(gen, scores):
    gen_ith = [g.strip() for g in gen]
    candidates = list(set(gen_ith))
    votes = {c: 0 for c in candidates}
    for j, c in enumerate(gen_ith):
        votes[c] += scores[j]

    max_votes = max(votes.values())
    winners = [c for c, v in votes.items() if v == max_votes]

    return winners


class Evaluator(BaseModel):
    path: str
    rerun: bool = False

    verbose: bool = False
    debug: bool = False

    _data: list = None
    _reformulated_path: str = None
    _sc_reformulated_path: str = None
    _reformulated_eval_all_path: str = None
    _sc_reformulated_eval_all_path: str = None

    # pydantic config
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        with open(self.path, "r") as f:
            self._data = json.load(f)

        self._reformulated_path = self.path[:-5] + "-reformulated.json"
        self._sc_reformulated_path = self.path[:-5] + "-sc-reformulated.json"
        temp_path = self._reformulated_path[:-5] + "_testoutputprediction_output.json"
        self._reformulated_eval_all_path = temp_path.replace(".json", "_eval_all.json")
        temp_path = (
            self._sc_reformulated_path[:-5] + "_testoutputprediction_output.json"
        )
        self._sc_reformulated_eval_all_path = temp_path.replace(
            ".json", "_eval_all.json"
        )

    def run(
        self,
    ):
        self.reformulate()
        self.eval()
        self.compute_scores()

    def reformulate(self):
        if (
            Path(self._reformulated_path).exists()
            and Path(self._sc_reformulated_path).exists()
            and self.rerun is False
        ):
            return

        new_data = []
        new_sc_data = []
        for datum in self._data:
            id = datum["id"]
            tc_output = datum["tc_output-pred"]

            question_id = id.split("-")[0]
            for i in range(len(tc_output[0])):
                pred_list = []
                for _tc_output in tc_output:
                    text = _tc_output[i]
                    if "assert" in text:
                        text = text
                    else:
                        try:
                            obj = eval(text)
                            if isinstance(obj, str):
                                if obj == "true":
                                    text = True
                                elif obj == "false":
                                    text = False
                        except:
                            try:
                                text = str(json.loads(text))
                            except:
                                text = f'"{text}"'

                    pred_list.append(text)

                new_data.append(
                    {
                        "question_id": question_id,
                        "test_id": i,
                        "pred_list": pred_list,
                    }
                )
                new_sc_data.append(
                    {
                        "question_id": question_id,
                        "test_id": i,
                        "pred_list": self_consistency(pred_list, [1, 1, 1, 1, 1]),
                    }
                )

        with open(self._reformulated_path, "w") as f:
            json.dump(new_data, f, indent=4)

        with open(self._sc_reformulated_path, "w") as f:
            json.dump(new_sc_data, f, indent=4)

    def eval(self):
        if (
            Path(self._reformulated_eval_all_path).exists()
            and Path(self._sc_reformulated_eval_all_path).exists()
            and self.rerun is False
        ):
            return

        sys.argv = [
            "",
            "--custom_output_file",
            self._reformulated_path,
            "--scenario",
            "testoutputprediction",
        ]
        lcb_eval()

        sys.argv = [
            "",
            "--custom_output_file",
            self._sc_reformulated_path,
            "--scenario",
            "testoutputprediction",
        ]
        lcb_eval()

    def compute_scores(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            sys.argv = [
                "",
                "--scenario",
                "testoutputprediction",
                "--eval_all_file",
                self._reformulated_eval_all_path,
            ]
            lcb_compute_scores(lcb_cs_parser())
        captured = buffer.getvalue()

        with open(self._reformulated_path[:-5] + "_scores.txt", "w") as f:
            f.write(captured)

        buffer = StringIO()
        with redirect_stdout(buffer):
            sys.argv = [
                "",
                "--scenario",
                "testoutputprediction",
                "--eval_all_file",
                self._sc_reformulated_eval_all_path,
            ]
            lcb_compute_scores(lcb_cs_parser())
        captured = buffer.getvalue()

        with open(self._sc_reformulated_path[:-5] + "_scores.txt", "w") as f:
            f.write(captured)

        buffer = StringIO()
        with redirect_stdout(buffer):
            sys.argv = [
                "",
                "--scenario",
                "testoutputprediction",
                "--eval_all_file",
                self._reformulated_eval_all_path,
                "--start_date",
                "2024-01-01",
            ]
            lcb_compute_scores(lcb_cs_parser())
        captured = buffer.getvalue()

        with open(self._reformulated_path[:-5] + "_cutoff_scores.txt", "w") as f:
            f.write(captured)

        buffer = StringIO()
        with redirect_stdout(buffer):
            sys.argv = [
                "",
                "--scenario",
                "testoutputprediction",
                "--eval_all_file",
                self._sc_reformulated_eval_all_path,
                "--start_date",
                "2024-01-01",
            ]
            lcb_compute_scores(lcb_cs_parser())
        captured = buffer.getvalue()

        with open(self._sc_reformulated_path[:-5] + "_cutoff_scores.txt", "w") as f:
            f.write(captured)

    def exit(self):
        pass


if __name__ == "__main__":
    paths = [
        # "results/20241205-gpt4-vanilla-leaderboard/results_merged_1.json",
        # "results/20241205-gpt4-vanilla/results_merged_1_transformed.json",
        # "results/20241205-gpt4-no_trace/results_merged_1.json",
        # "results/20241205-gpt4omini-vanilla/results_merged_1_transformed.json",
        # "results/20241205-gpt4omini-no_trace/results_merged_1.json",
        # "results/20241214-gpt4-tae/results_merged_1.json",
        # "results/20241214-gpt4-no_icl/results_merged_1.json",
        # "results/20241214-gpt4-ours_new/results_merged_1.json",
        # "results/20241214-gpt4omini-no_icl/results_merged_1.json",
        # "results/20241214-gpt4omini-ours_new/results_merged_1.json",
        "results/20241214-gpt4-code_exec/results_merged_1.json",
    ]
    for path in paths:
        Evaluator(path=path, rerun=False).run()

    for path in paths:
        print(path)
        score_paths = [
            path[:-5] + "-reformulated_scores.txt",
            path[:-5] + "-sc-reformulated_scores.txt",
            path[:-5] + "-reformulated_cutoff_scores.txt",
            path[:-5] + "-sc-reformulated_cutoff_scores.txt",
        ]
        for _p in score_paths:
            with open(_p, "r") as f:
                lines = f.readlines()
                indices = [-1 for _ in range(4)]
                for i, line in enumerate(lines):
                    if line.startswith("Pass@1"):
                        indices[0] = i
                    elif line.startswith("Easy Pass@1"):
                        indices[1] = i
                    elif line.startswith("Medium Pass@1"):
                        indices[2] = i
                    elif line.startswith("Hard Pass@1"):
                        indices[3] = i

                def extract_score(line):
                    return float(line.split()[-1]) * 100

                print(f" & \\num{{{extract_score(lines[indices[0]])}}}", end="")
                print(f" & \\num{{{extract_score(lines[indices[1]])}}}", end="")
                print(f" & \\num{{{extract_score(lines[indices[2]])}}}", end="")
                print(f" & \\num{{{extract_score(lines[indices[3]])}}} \\\\")
