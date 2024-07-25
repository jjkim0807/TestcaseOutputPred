import fire

from expand_langchain.evaluator import Evaluator
from expand_langchain.generator import Generator

if __name__ == "__main__":
    fire.Fire(
        {
            "generator": Generator,
            "evaluator": Evaluator,
        }
    )
