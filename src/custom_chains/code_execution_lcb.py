import json
import os
import re
from typing import Optional

from expand_langchain.utils.registry import chain_registry
from langchain_community.utilities.requests import JsonRequestsWrapper
from langchain_core.runnables import RunnableLambda


@chain_registry(name="excode_execution_lcb")
def code_execution_lcb_chain(
    key: str,
    code_key: str,
    tc_input_key: Optional[str] = None,
    **kwargs,
):
    def _func(data, config={}):
        code_list = data[code_key]
        result = [[] for _ in range(len(code_list))]
        exec_code_list = [[] for _ in range(len(code_list))]
        tc_input_list = data.get(tc_input_key)[0]
        for i, code in enumerate(code_list):
            for tc_input in tc_input_list:
                func_name = code.split("(")[0].split(" ")[-1]
                code_head = """\
from collections import *
from heapq import *
from itertools import *
from functools import *
from math import *
from bisect import *
from operator import *
from re import *
from statistics import *
from random import *
from string import *
from datetime import *
from typing import *
from sys import *
from os import *
"""
                tc_input = ",".join(tc_input.split("\n"))
                code_tail = f"""
result = {func_name}({tc_input})
print("```")
import json
import re
print(json.dumps(result, indent=4, ensure_ascii=False))
print("```")
"""
                exec_code = code_head + code + code_tail

                response = send_request(exec_code)
                output = extract_codeblock(response["output"])
                output = output.strip()

                result[i].append(output)
                exec_code_list[i].append(exec_code)

        return {
            key: result,
            f"{key}_exec_code": exec_code_list,
        }

    return RunnableLambda(_func, name=key)


def send_request(code: str):
    return JsonRequestsWrapper().post(
        os.environ["CODEEXEC_ENDPOINT"], data={"code": code}
    )


def extract_codeblock(text: str):
    try:
        return re.findall(r"```(.*?)```", text, re.DOTALL)[0]
    except:
        return text
