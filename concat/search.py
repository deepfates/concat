import numpy as np

def nearest_neighbors(query_embedding: np.ndarray, embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Find the indices of the k nearest neighbors of the query embedding in the embeddings.

    Args:
        query_embedding (np.ndarray): The embedding of the query.
        embeddings (np.ndarray): The embeddings of the documents.
        k (int): The number of nearest neighbors to find.

    Returns:
        np.ndarray: The indices of the k nearest neighbors.
    """
    print("Finding nearest neighbors")
    print("Query embedding: ", type(query_embedding))
    print("Embeddings: ", type(embeddings))
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    indices = np.argsort(distances)[:k]
    return indices
