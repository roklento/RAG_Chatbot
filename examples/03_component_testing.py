"""
Example 3: Test individual components separately.

This script demonstrates testing:
1. Query processor (correction + diversification)
2. Embedding model
3. Reranker model
4. Hybrid retriever
"""

import sys
sys.path.append('..')

from src.config import get_settings
from src.models import QwenEmbedding, QwenReranker, QueryProcessor
from src.retrieval import HybridRetriever
from qdrant_client import QdrantClient


def test_query_processor():
    """Test query processing component."""
    print("\n" + "="*80)
    print("TESTING QUERY PROCESSOR")
    print("="*80 + "\n")

    settings = get_settings()
    processor = QueryProcessor(
        model_path=settings.llm_model_path,
        device=settings.device,
        num_variants=3,
    )

    # Test query with typo
    query = "What is masheen lerning and how dose it work?"
    print(f"Original Query: {query}\n")

    result = processor.process(query)

    print(f"Corrected Query: {result.corrected_query}\n")
    print(f"Query Variants:")
    for i, variant in enumerate(result.query_variants, 1):
        print(f"  {i}. {variant}")

    print("\n✓ Query processor test complete!")


def test_embedding_model():
    """Test embedding model component."""
    print("\n" + "="*80)
    print("TESTING EMBEDDING MODEL")
    print("="*80 + "\n")

    settings = get_settings()
    model = QwenEmbedding(
        model_path=settings.embedding_model_path,
        device=settings.device,
    )

    # Test embedding
    texts = [
        "Machine learning is awesome",
        "AI is transforming the world",
        "The weather is nice today",
    ]

    print("Texts to embed:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    embeddings = model.embed_documents(texts)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {model.embedding_dimension}")

    # Calculate similarity
    import numpy as np
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    print("\nSimilarity matrix:")
    print("      ", "  ".join([f"Text{i}" for i in range(1, 4)]))
    for i, emb1 in enumerate(embeddings):
        sims = [cosine_sim(emb1, emb2) for emb2 in embeddings]
        print(f"Text{i+1}: " + "  ".join([f"{sim:.3f}" for sim in sims]))

    print("\n✓ Embedding model test complete!")


def test_reranker():
    """Test reranker component."""
    print("\n" + "="*80)
    print("TESTING RERANKER")
    print("="*80 + "\n")

    settings = get_settings()
    reranker = QwenReranker(
        model_path=settings.reranker_model_path,
        device=settings.device,
    )

    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "The weather forecast predicts rain tomorrow.",
        "Deep learning uses neural networks with multiple layers.",
        "Cooking pasta requires boiling water and salt.",
        "Supervised learning uses labeled training data.",
    ]

    print(f"Query: {query}\n")
    print("Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")

    print("\nReranking...")
    results = reranker.rerank(query, documents, top_k=5)

    print("\nReranked Results:")
    for result in results:
        print(f"  Score: {result.score:.4f} | {result.content}")

    print("\n✓ Reranker test complete!")


def test_hybrid_retriever():
    """Test hybrid retriever component."""
    print("\n" + "="*80)
    print("TESTING HYBRID RETRIEVER")
    print("="*80 + "\n")

    settings = get_settings()

    # Initialize components
    embedding_model = QwenEmbedding(
        model_path=settings.embedding_model_path,
        device=settings.device,
    )

    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    retriever = HybridRetriever(
        settings=settings,
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
    )

    # Test multi-query search
    queries = [
        "What is machine learning?",
        "Explain ML concepts",
        "How does machine learning work?",
    ]

    print("Query Variants:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    print("\nSearching...")
    results = retriever.multi_query_search(queries, top_k_per_query=5)

    print(f"\nRetrieved {len(results)} results:\n")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. Score: {result.score:.4f} | Collection: {result.collection}")
        print(f"   {result.content[:100]}...")

    print("\n✓ Hybrid retriever test complete!")


if __name__ == "__main__":
    # Test components individually
    # Uncomment the ones you want to test

    # test_query_processor()
    # test_embedding_model()
    # test_reranker()
    # test_hybrid_retriever()

    # Or run all tests
    print("Select test to run:")
    print("1. Query Processor")
    print("2. Embedding Model")
    print("3. Reranker")
    print("4. Hybrid Retriever")
    print("5. All tests")

    choice = input("\nEnter choice (1-5): ")

    if choice == "1":
        test_query_processor()
    elif choice == "2":
        test_embedding_model()
    elif choice == "3":
        test_reranker()
    elif choice == "4":
        test_hybrid_retriever()
    elif choice == "5":
        test_query_processor()
        test_embedding_model()
        test_reranker()
        test_hybrid_retriever()
    else:
        print("Invalid choice!")
