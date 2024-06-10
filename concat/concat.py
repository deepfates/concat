import json
import logging
from typing import List
from collections import Counter
from dotenv import load_dotenv

import numpy as np

load_dotenv()

from .models import Bookwyrm, TextChunk

from .search import nearest_neighbors
from .chat import generate_prompt, generate_text, embedding_api

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

def process_query(query: str) -> np.ndarray:
    logger.info("Processing query: %s", query)
    embed = embedding_api(query)
    logger.info("Query processed")
    return embed

def get_similar_chunks(model: Bookwyrm, query_embedding: np.ndarray) -> List[TextChunk]:
    logger.info("Finding similar chunks")
    embeddings = model.embeddings
    indices = nearest_neighbors(query_embedding, embeddings)
    similar_chunks = [model.chunks[i] for i in indices]
    logger.info("Found similar chunks")
    return similar_chunks

def get_citations(model: Bookwyrm, similar_chunks: List[TextChunk]) -> str:
    logger.info("Generating citations")
    citations = Counter(chunk.document_index for chunk in similar_chunks)
    sorted_citations = [f"{model.documents[idx].uri} ({count})" for idx, count in citations.most_common()]
    return f"Related files: {', '.join(sorted_citations)}"

def generate_answer(similar_chunks: List[TextChunk], query: str):
    logger.info("Generating answer")
    prompt = generate_prompt(query, [chunk.text for chunk in similar_chunks])
    for text in generate_text(prompt):
        yield text

def get_full_answer(model: Bookwyrm, query: str):
    logger.info("Starting main process")
    query_embedding = process_query(query)
    similar_chunks = get_similar_chunks(model, query_embedding)
    citations = get_citations(model, similar_chunks)

    for answer_part in generate_answer(similar_chunks, query):
        yield answer_part

    yield citations
    
def main():
    model = load_model()
    query = "Can I use Replicate models from this library?"
    parts = []
    for part in get_full_answer(model, query):
        parts.append(part)
        print(part)

if __name__ == "__main__":
   main()
