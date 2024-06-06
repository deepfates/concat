from typing import List
import logging

import replicate

from bookwyrm.models import TextChunk

from .prompts import SYSTEM_PROMPT, QUERY_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def generate_prompt(query: str, search_results: List[str]) -> str:
    """
    Generate a prompt for the model based on the query.

    Args:
        query (str): The original text query.

    Returns:
        str: The generated prompt.
    """
    logger.info("Generating prompt")
    formatted_search_results = "\n\n---\n\n".join(search_results)
    prompt = f"{QUERY_PROMPT}\n\nSearch results:\n```{formatted_search_results}```\n\nQuery: {query}"
    return prompt

def generate_text(prompt: str):
    """
    Generate text based on the prompt using the meta/meta-llama-3-8b-instruct model.
    The meta/meta-llama-3-8b-instruct model can stream output as it's running.
    """
    for event in replicate.stream(
        "meta/meta-llama-3-8b-instruct",
        input={
            "top_k": 0,
            "top_p": 0.9,
            "prompt": prompt,
            "max_tokens": 512,
            "min_tokens": 0,
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

def generate_answer(similar_chunks: List[TextChunk], query: str):
    """
    Generate an answer based on the similar documents.

    Args:
        similar_chunks (List[TextChunk]): A list of similar text chunks.
        query (str): The original text query.

    Returns:
        generator: A generator yielding the generated answer text.
    """
    logger.info("Generating answer")
    prompt = generate_prompt(query, [chunk.text for chunk in similar_chunks])
    return generate_text(prompt)
            