import json
from typing import List
import logging
import numpy as np
import replicate

from bookwyrm.models import TextChunk
from .prompts import SYSTEM_PROMPT, QUERY_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def embedding_api(texts):
    resp =  replicate.run(
        "replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305",
        input={"text_batch": json.dumps(texts)},
    )
    flattened_embeds = [o['embedding'] for o in resp]
    return np.array(flattened_embeds)


def generate_prompt(query: str, search_results: List[str]) -> str:
    logger.info("Generating prompt")
    formatted_search_results = "\n\n---\n\n".join(search_results)
    return f"{QUERY_PROMPT}\n\nSearch results:\n```{formatted_search_results}```\n\nQuery: {query}"

def generate_text(prompt: str):
    for event in replicate.stream(
        "meta/meta-llama-3-8b-instruct",
        input={
            "top_k": 0,
            "top_p": 0.9,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.6,
            "system_prompt": SYSTEM_PROMPT,
            "length_penalty": 1,
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "log_performance_metrics": False
        },
    ):
        yield str(event)
