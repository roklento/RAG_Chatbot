"""
Reciprocal Rank Fusion (RRF) implementation for combining multiple rankings.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ScoredDocument:
    """Document with its fusion score and metadata."""
    content: str
    score: float
    source_ranks: Dict[str, int]  # Which queries/sources ranked this doc and at what position
    metadata: Dict[str, Any] = None


def reciprocal_rank_fusion(
    rankings: List[List[Any]],
    k: int = 60,
    document_key: str = "content",
) -> List[Tuple[Any, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.

    RRF Formula: score(doc) = Σ(1 / (k + rank_i))
    where k is a constant (default 60) and rank_i is the position in ranking i.

    Args:
        rankings: List of ranked lists. Each inner list is a ranking from one source.
                 Elements can be strings or dictionaries with document_key.
        k: RRF constant (default: 60, standard value from literature)
        document_key: Key to extract document content if items are dicts

    Returns:
        List of (document, score) tuples sorted by fused score (descending)

    Example:
        >>> rankings = [
        ...     ["doc1", "doc2", "doc3"],
        ...     ["doc2", "doc1", "doc4"],
        ... ]
        >>> fused = reciprocal_rank_fusion(rankings)
        >>> # doc2 appears at rank 2 in first list: 1/(60+2) = 0.0161
        >>> # doc2 appears at rank 1 in second list: 1/(60+1) = 0.0164
        >>> # Total score for doc2: 0.0161 + 0.0164 = 0.0325
    """
    scores = defaultdict(float)
    source_info = defaultdict(dict)

    for source_idx, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            # Extract document identifier
            if isinstance(item, dict):
                doc_id = item.get(document_key, str(item))
                doc = item
            else:
                doc_id = str(item)
                doc = item

            # Calculate RRF score
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] += rrf_score

            # Track source information
            if doc_id not in source_info:
                source_info[doc_id] = {
                    "document": doc,
                    "ranks": {}
                }
            source_info[doc_id]["ranks"][f"source_{source_idx}"] = rank

    # Sort by score (descending)
    fused_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Return with original documents
    return [(source_info[doc_id]["document"], score) for doc_id, score in fused_results]


def fuse_rankings(
    rankings: List[List[str]],
    k: int = 60,
    return_scored_documents: bool = False,
) -> List[str] | List[ScoredDocument]:
    """
    Fuse multiple text rankings using RRF.

    Args:
        rankings: List of ranked document lists (strings)
        k: RRF constant
        return_scored_documents: If True, return ScoredDocument objects with metadata

    Returns:
        Fused ranking as list of documents (or ScoredDocument objects if requested)
    """
    scores = defaultdict(float)
    source_ranks = defaultdict(dict)

    for source_idx, ranking in enumerate(rankings):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
            source_ranks[doc][f"query_{source_idx}"] = rank

    # Sort by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if return_scored_documents:
        return [
            ScoredDocument(
                content=doc,
                score=score,
                source_ranks=source_ranks[doc],
            )
            for doc, score in sorted_docs
        ]
    else:
        return [doc for doc, _ in sorted_docs]


def fuse_with_weights(
    rankings: List[List[str]],
    weights: List[float],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Weighted RRF fusion for combining rankings with different importance.

    Args:
        rankings: List of ranked document lists
        weights: Weight for each ranking (should sum to 1.0)
        k: RRF constant

    Returns:
        List of (document, weighted_score) tuples
    """
    if len(rankings) != len(weights):
        raise ValueError("Number of rankings must match number of weights")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    scores = defaultdict(float)

    for ranking, weight in zip(rankings, weights):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += weight * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def deduplicate_preserving_order(documents: List[str]) -> List[str]:
    """
    Remove duplicates while preserving order (keeps first occurrence).

    Args:
        documents: List of documents (possibly with duplicates)

    Returns:
        List with duplicates removed
    """
    seen = set()
    result = []
    for doc in documents:
        if doc not in seen:
            seen.add(doc)
            result.append(doc)
    return result
