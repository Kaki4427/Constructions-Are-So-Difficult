import json
import os
import time

import tiktoken
import openai
from functools import wraps

from hashing import short_hash
from graham import generate_from_openai_chat_completion
import asyncio
from typing import List, Dict

import pandas as pd
import re
import json

OPENAI_ERROR_SLEEP_TIME = 5
API_KEY = "API_KEY"
GPT4_KEY =  "API_KEY"

CACHE_DIR = "cache"  # Directory to store the cached responses
os.makedirs(CACHE_DIR, exist_ok=True)

def count_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def cache_result(func):
    @wraps(func)
    def wrapper(prompts: List[str], system_prompt: str, ignore_cache=False, model="gpt-3.5-turbo"):
        if model=="gpt-3.5-turbo":
            cache_dir = "gpt3_" + CACHE_DIR
        elif model=="gpt-4-0613":
            cache_dir = "gpt4_" + CACHE_DIR
        uncached_prompts = []
        results = []
        uncached_files = []
        if ignore_cache:
            return func(prompts, system_prompt)
        for prompt in prompts:
            cache_name = short_hash(prompt + system_prompt)
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{cache_name}.json")

            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    results.append(f.read())
            else:
                uncached_prompts.append(prompt)
                uncached_files.append(cache_file)

        if len(uncached_prompts) == 0:
            return results

        output = func(uncached_prompts, system_prompt, model=model)
        for prompt, result, cache_file in zip(uncached_prompts, output, uncached_files):
            with open(cache_file, "w") as f:
                f.write(result)

        return wrapper(prompts, system_prompt, model=model)

    return wrapper


@cache_result
def get_model_response(user_prompt_list, system_prompt, ignore_cache=False, model="gpt-3.5-turbo"):
    if model == "gpt-3.5-turbo":
        promise = generate_from_openai_chat_completion(user_prompt_list, "gpt-3.5-turbo-0613", system_prompt, API_KEY)

    elif model == "gpt-4-0613":
        print("4")
        promise = generate_from_openai_chat_completion(user_prompt_list, "gpt-4-0613", system_prompt, GPT4_KEY)
    # await promise and return output with async io
    return asyncio.run(promise)
