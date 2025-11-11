"""
Advanced RAG Retriever - Main orchestrator.
Combines query processing, hybrid search, reranking, and MMR.
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from qdrant_client import QdrantClient

from ..config import Settings
from ..models import QwenEmbedding, QwenReranker, QueryProcessor
from .hybrid_retriever import HybridRetriever, RetrievalResult
from ..utils import maximal_marginal_relevance


@dataclass
class FinalResult:
    """Final retrieval result with all metadata."""
    content: str
    relevance_score: float  # Reranker score
    retrieval_score: float  # Original retrieval score
    rank: int  # Final rank after all processing
    collection: str
    metadata: Optional[dict] = None


class AdvancedRetriever:
    """
    Advanced RAG Retriever with complete pipeline.

    Pipeline:
    1. Query Processing (correction + diversification)
    2. Hybrid Multi-Query Search (dense + sparse, RRF fusion)
    3. Reranking (cross-encoder scoring)
    4. MMR Diversity Filtering
    5. Final Result Selection
    """

    def __init__(
        self,
        settings: Settings,
        query_processor: QueryProcessor,
        embedding_model: QwenEmbedding,
        reranker: QwenReranker,
        qdrant_client: QdrantClient,
    ):
        """
        Initialize the advanced retriever.

        Args:
            settings: Application settings
            query_processor: Query processor instance
            embedding_model: Embedding model instance
            reranker: Reranker model instance
            qdrant_client: Qdrant client instance
        """
        self.settings = settings
        self.query_processor = query_processor
        self.embedding_model = embedding_model
        self.reranker = reranker

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            settings=settings,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        apply_mmr: bool = True,
        return_metadata: bool = True,
        verbose: bool = False,
    ) -> List[FinalResult]:
        """
        Complete retrieval pipeline.

        Args:
            query: User query string
            top_k: Number of final results (uses settings default if None)
            apply_mmr: Whether to apply MMR diversity filtering
            return_metadata: Include metadata in results
            verbose: Print pipeline progress

        Returns:
            List of final results
        """
        top_k = top_k or self.settings.final_top_k

        # Step 1: Query Processing
        if verbose:
            print(f"\n{'='*60}")
            print(f"Original Query: {query}")
            print(f"{'='*60}\n")

        processed = self.query_processor.process(query)

        if verbose:
            print(f"Corrected Query: {processed.corrected_query}")
            print(f"\nQuery Variants ({len(processed.query_variants)}):")
            for i, variant in enumerate(processed.query_variants, 1):
                print(f"  {i}. {variant}")
            print()

        # Step 2: Hybrid Multi-Query Retrieval
        if verbose:
            print(f"Retrieving candidates from both collections...")

        candidates_before_rerank = self.settings.candidates_before_rerank
        retrieval_results = self.hybrid_retriever.retrieve(
            queries=processed.all_queries,
            top_k=candidates_before_rerank,
        )

        if verbose:
            print(f"✓ Retrieved {len(retrieval_results)} candidates\n")

        if not retrieval_results:
            return []

        # Step 3: Reranking
        if verbose:
            print(f"Reranking candidates...")

        # Extract documents for reranking
        documents = [r.content for r in retrieval_results]

        # Rerank using corrected query
        reranked_docs = self.reranker.rerank(
            query=processed.corrected_query,
            documents=documents,
            top_k=None,  # Rank all
        )

        if verbose:
            print(f"✓ Reranked {len(reranked_docs)} documents\n")

        # Filter by threshold
        threshold = self.settings.reranker_threshold
        filtered_docs = [doc for doc in reranked_docs if doc.score >= threshold]

        if verbose:
            print(f"Filtered by threshold (>= {threshold}): {len(filtered_docs)} remain\n")

        if not filtered_docs:
            # If no docs pass threshold, take top K from reranked
            filtered_docs = reranked_docs[:top_k]

        # Step 4: Apply MMR for diversity (optional)
        if apply_mmr and len(filtered_docs) > top_k:
            if verbose:
                print(f"Applying MMR diversity filtering...")

            # Get embeddings for documents
            doc_contents = [doc.content for doc in filtered_docs]
            doc_embeddings = self.embedding_model.embed_documents(doc_contents)

            # Get query embedding
            query_embedding = self.embedding_model.embed_query(processed.corrected_query)

            # Apply MMR
            selected_indices = maximal_marginal_relevance(
                query_embedding=query_embedding,
                document_embeddings=list(doc_embeddings),
                documents=doc_contents,
                lambda_param=1.0 - self.settings.mmr_diversity_score,
                top_k=top_k,
            )

            # Reorder documents
            final_docs = [filtered_docs[idx] for idx in selected_indices]

            if verbose:
                print(f"✓ Selected {len(final_docs)} diverse documents\n")
        else:
            final_docs = filtered_docs[:top_k]

        # Step 5: Create final results
        final_results = []
        for rank, reranked_doc in enumerate(final_docs, 1):
            # Find original retrieval result
            original_result = next(
                (r for r in retrieval_results if r.content == reranked_doc.content),
                None,
            )

            final_results.append(
                FinalResult(
                    content=reranked_doc.content,
                    relevance_score=reranked_doc.score,
                    retrieval_score=original_result.score if original_result else 0.0,
                    rank=rank,
                    collection=original_result.collection if original_result else "unknown",
                    metadata=original_result.metadata if (original_result and return_metadata) else None,
                )
            )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Final Results: {len(final_results)}")
            print(f"{'='*60}\n")
            for result in final_results:
                print(f"Rank {result.rank} | Score: {result.relevance_score:.4f} | Collection: {result.collection}")
                print(f"  {result.content[:100]}...")
                print()

        return final_results

    def retrieve_simple(self, query: str, top_k: int = 5) -> List[str]:
        """
        Simplified retrieval interface returning only document contents.

        Args:
            query: User query
            top_k: Number of results

        Returns:
            List of document strings
        """
        results = self.retrieve(query=query, top_k=top_k, verbose=False)
        return [r.content for r in results]

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Retrieval interface returning (document, score) tuples.

        Args:
            query: User query
            top_k: Number of results

        Returns:
            List of (content, score) tuples
        """
        results = self.retrieve(query=query, top_k=top_k, verbose=False)
        return [(r.content, r.relevance_score) for r in results]


def create_advanced_retriever(settings: Settings) -> AdvancedRetriever:
    """
    Factory function to create a fully initialized AdvancedRetriever.

    Args:
        settings: Application settings

    Returns:
        Initialized AdvancedRetriever instance
    """
    # Initialize models
    print("Loading models...")

    print("  - Embedding model...")
    embedding_model = QwenEmbedding(
        model_path=settings.embedding_model_path,
        device=settings.device,
        max_length=settings.embedding_max_length,
    )

    print("  - Reranker model...")
    reranker = QwenReranker(
        model_path=settings.reranker_model_path,
        device=settings.device,
        max_length=settings.reranker_max_length,
    )

    print("  - Query processor (LLM)...")
    query_processor = QueryProcessor(
        model_path=settings.llm_model_path,
        device=settings.device,
        max_new_tokens=settings.llm_max_new_tokens,
        temperature=settings.llm_temperature,
        num_variants=settings.query_variants_count,
    )

    # Initialize Qdrant client
    print("  - Qdrant client...")
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
    )

    print("✓ All models loaded\n")

    # Create retriever
    retriever = AdvancedRetriever(
        settings=settings,
        query_processor=query_processor,
        embedding_model=embedding_model,
        reranker=reranker,
        qdrant_client=qdrant_client,
    )

    return retriever
