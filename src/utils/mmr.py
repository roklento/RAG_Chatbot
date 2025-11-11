"""
Maximal Marginal Relevance (MMR) for diversifying retrieval results.
"""

from typing import List
import numpy as np
from numpy.linalg import norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score [-1, 1]
    """
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    document_embeddings: List[np.ndarray],
    documents: List[str],
    lambda_param: float = 0.5,
    top_k: int = 10,
) -> List[int]:
    """
    Select documents using Maximal Marginal Relevance.

    MMR balances relevance to the query with diversity among selected documents.

    Formula:
        MMR = argmax[Di ∈ R \ S] [λ * Sim(Di, Q) - (1-λ) * max[Dj ∈ S] Sim(Di, Dj)]

    Where:
        - Di: Candidate document
        - Q: Query
        - S: Already selected documents
        - R: Remaining candidate documents
        - λ: Trade-off parameter (0 = max diversity, 1 = max relevance)

    Args:
        query_embedding: Query embedding vector
        document_embeddings: List of document embedding vectors
        documents: List of document strings (for reference)
        lambda_param: Trade-off between relevance and diversity (0-1)
                     0.0 = maximum diversity
                     0.5 = balanced
                     1.0 = maximum relevance (no diversity)
        top_k: Number of documents to select

    Returns:
        Indices of selected documents in order
    """
    if len(documents) == 0:
        return []

    if len(documents) <= top_k:
        return list(range(len(documents)))

    # Convert to numpy arrays
    query_emb = np.array(query_embedding)
    doc_embs = np.array(document_embeddings)

    # Calculate relevance scores (similarity to query)
    relevance_scores = np.array([
        cosine_similarity(query_emb, doc_emb)
        for doc_emb in doc_embs
    ])

    # Initialize selected and remaining indices
    selected_indices = []
    remaining_indices = list(range(len(documents)))

    # Select first document (highest relevance)
    first_idx = int(np.argmax(relevance_scores))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    # Iteratively select documents
    while len(selected_indices) < top_k and remaining_indices:
        mmr_scores = []

        for idx in remaining_indices:
            # Relevance component
            relevance = relevance_scores[idx]

            # Diversity component (max similarity to already selected docs)
            max_similarity = max([
                cosine_similarity(doc_embs[idx], doc_embs[selected_idx])
                for selected_idx in selected_indices
            ])

            # MMR formula
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append((idx, mmr_score))

        # Select document with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return selected_indices


def apply_mmr_to_documents(
    query_embedding: np.ndarray,
    documents: List[str],
    document_embeddings: List[np.ndarray],
    diversity_score: float = 0.3,
    top_k: int = 10,
) -> List[str]:
    """
    Apply MMR and return reordered documents.

    Args:
        query_embedding: Query embedding
        documents: List of documents
        document_embeddings: Corresponding embeddings
        diversity_score: Diversity weight (0-1). Higher = more diverse.
                        Note: This is (1-λ) in the standard formula
        top_k: Number of documents to return

    Returns:
        Reordered list of documents
    """
    # Convert diversity_score to lambda parameter
    # diversity_score=0.3 means 70% relevance, 30% diversity
    lambda_param = 1.0 - diversity_score

    selected_indices = maximal_marginal_relevance(
        query_embedding=query_embedding,
        document_embeddings=document_embeddings,
        documents=documents,
        lambda_param=lambda_param,
        top_k=top_k,
    )

    return [documents[idx] for idx in selected_indices]
