import fire
from expand_langchain.generator import Generator
from expand_langchain.loader import Loader

from src.custom_chains import *
from src.evaluator import Evaluator

if __name__ == "__main__":
    fire.Fire(
        {
            "generator": Generator,
            "evaluator": Evaluator,
            "loader": Loader,
        }
    )
