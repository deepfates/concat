import json
import logging
from typing import List, Dict, Any
from collections import Counter

import numpy as np
import asyncio

from bookwyrm.models import Bookwyrm, DocumentRecord, TextChunk
from bookwyrm.utils import embedding_api

from .search import nearest_neighbors
from .chat import generate_answer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(location: str = "./wyrm.json") -> Bookwyrm:
    logger.info("Loading model from %s", location)
    with open(location, 'r', encoding='utf-8') as f:
        text: str = json.load(f)['output']
    wyrm = Bookwyrm.from_json(text)
    logger.info("Model loaded successfully")
    return wyrm

async def process_query(query: str) -> np.ndarray:
    logger.info("Processing query: %s", query)
    embed = await embedding_api(query)
    logger.info("Query processed")
    return embed

def get_similar_chunks(model: Bookwyrm, query_embedding: np.ndarray) -> List[TextChunk]:
    logger.info("Finding similar chunks")
    embeddings = model.embeddings
    indices = nearest_neighbors(query_embedding, embeddings)
    similar_chunks = [model.chunks[i] for i in indices]
    logger.info("Found similar chunks, indices: %s", indices)
    logger.info("Found similar chunks, chunks: %s", similar_chunks)
    return similar_chunks

def get_citations(model: Bookwyrm, similar_chunks: List[TextChunk]) -> str:
    logger.info("Generating citations")
    citations: Counter[str] = Counter()
    for chunk in similar_chunks:
        doc = model.documents[chunk.document_index]
        citations[doc.uri] += 1
    sorted_citations = [f"{uri} ({count})" for uri, count in citations.most_common()]
    citations_str = "\n".join(sorted_citations)
    return f"Related files:\n{citations_str}"

async def async_generate_answer(similar_chunks: List[TextChunk], query: str):
    loop = asyncio.get_event_loop()
    for answer_part in await loop.run_in_executor(None, generate_answer, similar_chunks, query):
        yield answer_part

async def get_answer(model: Bookwyrm, query: str):
    logger.info("Starting main process")
    query_embedding = await process_query(query)
    similar_chunks = get_similar_chunks(model, query_embedding)
    citations = get_citations(model, similar_chunks)

    async for answer_part in async_generate_answer(similar_chunks, query):
        yield answer_part

    yield citations

async def main(model: Bookwyrm, query: str):
    async for part in get_answer(model, query):
        yield part

if __name__ == "__main__":
    test_model = load_model()
    TEST_QUERY = "Can I use Replicate models from this library?"

    async def main_wrapper():
        async for part in main(test_model, TEST_QUERY):
            print(part)

    asyncio.run(main_wrapper())
