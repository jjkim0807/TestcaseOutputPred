from itertools import combinations
from typing import Optional

from expand_langchain.chain.cot import cot_chain
from expand_langchain.utils.registry import chain_registry
from langchain_core.runnables import RunnableLambda


@chain_registry(name="pairwise_ranking")
def pairwise_ranking_chain(
    key: str,
    examples: Optional[dict] = None,
    **kwargs,
):
    async def _func(data, config={}):
        chain = cot_chain(
            key=key,
            examples=examples,
            **kwargs,
        )

        target_key = kwargs.get("target_key", "target")
        target_list = list(enumerate(data[target_key]))

        raw_result = {}
        for _pair in combinations(target_list, 2):
            for pair in [_pair, _pair[::-1]]:
                pair_key = f"{pair[0][0]}-{pair[1][0]}"
                pair = (pair[0][1], pair[1][1])
                result = await chain.ainvoke(
                    {
                        f"{target_key}_1": pair[0],
                        f"{target_key}_2": pair[1],
                        **{k: v for k, v in data.items() if k not in ["pseudocode"]},
                    },
                    config,
                )

                raw_result[f"{pair_key}_raw"] = result[f"{key}_raw"][0]
                try:
                    raw_result[pair_key] = int(result[key][0])
                except Exception as e:
                    raw_result[pair_key] = result[f"{key}_raw"][0]

        return {key: raw_result}

    return RunnableLambda(_func, name="pairwise_ranking")


@chain_registry(name="pairwise_ranking_parser")
def pairwise_ranking_chain(
    key: str,
    **kwargs,
):
    async def _func(data, config={}):
        chain = cot_chain(
            key="output",
            **kwargs,
        )

        result = {}
        for k, v in data["ranking_pc"].items():
            if "_raw" in k:
                continue
            if isinstance(v, int):
                result[k] = v
                continue

            response = await chain.ainvoke({"text": v}, config)
            try:
                answer = response["output"][0]["answer"]
                if isinstance(answer, int):
                    result[k] = answer
                elif isinstance(answer, str):
                    if answer.isdigit():
                        result[k] = int(answer)
                    elif answer.lower() == "tie":
                        result[k] = "tie"
                    else:
                        result[k] = "error"
                else:
                    result[k] = "error"
            except Exception as e:
                result[k] = "error"

            result[f"{k}_raw"] = v

        return {key: result}

    return RunnableLambda(_func, name="pairwise_ranking_parser")
