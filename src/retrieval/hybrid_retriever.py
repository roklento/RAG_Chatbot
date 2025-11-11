"""
Hybrid Search Retriever using Qdrant.
Combines dense (semantic) and sparse (keyword) search with custom weights.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    ScoredPoint,
    QueryVector,
    Prefetch,
    Query,
)

from ..config import Settings
from ..models import QwenEmbedding
from ..utils import reciprocal_rank_fusion


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    content: str
    score: float
    collection: str
    metadata: Optional[Dict[str, Any]] = None


class HybridRetriever:
    """
    Hybrid retriever for Qdrant with collection-specific weights.

    Features:
    - Dense + Sparse search fusion
    - Custom weights per collection (Q&A vs Plain Text)
    - Multi-query retrieval with RRF
    - Collection-aware ranking
    """

    def __init__(
        self,
        settings: Settings,
        qdrant_client: QdrantClient,
        embedding_model: QwenEmbedding,
    ):
        """
        Initialize hybrid retriever.

        Args:
            settings: Application settings
            qdrant_client: Qdrant client instance
            embedding_model: Embedding model for queries
        """
        self.settings = settings
        self.client = qdrant_client
        self.embedding_model = embedding_model

        # Collection configurations
        self.qa_collection = settings.qa_collection_name
        self.text_collection = settings.text_collection_name

        # Hybrid search weights
        self.collection_weights = {
            self.qa_collection: {
                "dense": settings.qa_dense_weight,
                "sparse": settings.qa_sparse_weight,
            },
            self.text_collection: {
                "dense": settings.text_dense_weight,
                "sparse": settings.text_sparse_weight,
            },
        }

    def _hybrid_search_collection(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search on a single collection.

        Note: This is a simplified implementation. For full hybrid search with
        sparse vectors, you would need to configure Qdrant collections with
        sparse vector support and use BM25/SPLADE embeddings.

        For now, we simulate hybrid search by:
        1. Using dense search (primary)
        2. Applying collection-specific weights as score modifiers

        Args:
            collection_name: Collection to search
            query: Query string
            limit: Number of results

        Returns:
            List of retrieval results
        """
        # Get query embedding
        query_vector = self.embedding_model.embed_query(query)

        # Perform dense search
        # In production with full hybrid support, you would:
        # 1. Generate sparse embeddings (BM25/SPLADE)
        # 2. Use Qdrant's query fusion API
        # 3. Combine results server-side

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        )

        # Get collection weights
        weights = self.collection_weights[collection_name]
        dense_weight = weights["dense"]

        # Convert to RetrievalResult objects
        # Apply weight to scores (simulating hybrid fusion)
        retrieval_results = []
        for result in results:
            weighted_score = result.score * dense_weight

            content = result.payload.get("content", "")
            metadata = {k: v for k, v in result.payload.items() if k != "content"}

            retrieval_results.append(
                RetrievalResult(
                    content=content,
                    score=weighted_score,
                    collection=collection_name,
                    metadata=metadata,
                )
            )

        return retrieval_results

    def search_both_collections(
        self,
        query: str,
        top_k_per_collection: int = 15,
    ) -> List[RetrievalResult]:
        """
        Search both Q&A and Plain Text collections.

        Args:
            query: Query string
            top_k_per_collection: Results to retrieve from each collection

        Returns:
            Combined and sorted results from both collections
        """
        # Search Q&A collection
        qa_results = self._hybrid_search_collection(
            collection_name=self.qa_collection,
            query=query,
            limit=top_k_per_collection,
        )

        # Search Plain Text collection
        text_results = self._hybrid_search_collection(
            collection_name=self.text_collection,
            query=query,
            limit=top_k_per_collection,
        )

        # Combine and sort by score
        all_results = qa_results + text_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results

    def multi_query_search(
        self,
        queries: List[str],
        top_k_per_query: int = 15,
        rrf_k: int = 60,
    ) -> List[RetrievalResult]:
        """
        Search with multiple query variants and fuse with RRF.

        Args:
            queries: List of query variants
            top_k_per_query: Results per query per collection
            rrf_k: RRF constant

        Returns:
            Fused and ranked results
        """
        all_rankings = []
        doc_to_result = {}  # Map content to RetrievalResult

        # Search for each query variant
        for query in queries:
            results = self.search_both_collections(
                query=query,
                top_k_per_collection=top_k_per_query,
            )

            # Create ranking (list of document contents)
            ranking = [r.content for r in results]
            all_rankings.append(ranking)

            # Store results
            for result in results:
                if result.content not in doc_to_result:
                    doc_to_result[result.content] = result

        # Apply RRF fusion
        fused = reciprocal_rank_fusion(all_rankings, k=rrf_k)

        # Convert back to RetrievalResult objects
        fused_results = []
        for doc_content, rrf_score in fused:
            if doc_content in doc_to_result:
                result = doc_to_result[doc_content]
                # Update score with RRF score
                result.score = rrf_score
                fused_results.append(result)

        return fused_results

    def retrieve(
        self,
        queries: List[str],
        top_k: int = 30,
    ) -> List[RetrievalResult]:
        """
        Main retrieval method with multi-query support.

        Args:
            queries: List of query variants (from query processor)
            top_k: Total number of results to return

        Returns:
            Top K fused results
        """
        results = self.multi_query_search(
            queries=queries,
            top_k_per_query=self.settings.top_k_per_query,
            rrf_k=self.settings.rrf_k,
        )

        return results[:top_k]


class HybridRetrieverWithSparseSupport(HybridRetriever):
    """
    Extended hybrid retriever with true sparse vector support.

    This requires:
    1. Qdrant collections configured with sparse vectors
    2. BM25 or SPLADE sparse embedding generation
    3. Qdrant's Query API for server-side fusion

    This is a placeholder for future implementation when
    sparse vectors are fully configured.
    """

    def __init__(
        self,
        settings: Settings,
        qdrant_client: QdrantClient,
        embedding_model: QwenEmbedding,
        sparse_embedding_model: Optional[Any] = None,
    ):
        """
        Initialize with sparse embedding support.

        Args:
            settings: Application settings
            qdrant_client: Qdrant client
            embedding_model: Dense embedding model
            sparse_embedding_model: Sparse embedding model (BM25/SPLADE)
        """
        super().__init__(settings, qdrant_client, embedding_model)
        self.sparse_model = sparse_embedding_model

    def _hybrid_search_collection(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """
        True hybrid search with sparse vectors.

        This would use Qdrant's Query API to combine dense and sparse
        searches server-side with RRF fusion.

        Requires Qdrant v1.10.0+ and collections configured with sparse vectors.
        """
        # TODO: Implement when sparse vectors are configured
        # For now, fall back to parent implementation
        return super()._hybrid_search_collection(collection_name, query, limit)
