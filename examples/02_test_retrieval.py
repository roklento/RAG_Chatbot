"""
Example 2: Test the complete retrieval pipeline.

This script demonstrates:
1. Loading the advanced retriever
2. Processing queries
3. Retrieving relevant documents
4. Viewing the complete pipeline output
"""

import sys
sys.path.append('..')

from src.config import get_settings
from src.retrieval.advanced_retriever import create_advanced_retriever


def main():
    # Load settings
    settings = get_settings()
    print("Loading Advanced RAG Retriever...\n")

    # Create retriever (this loads all models)
    retriever = create_advanced_retriever(settings)

    # Test queries
    test_queries = [
        "What is masheen lerning?",  # Intentional typo to test correction
        "Explain transformers architecture",
        "How does attention mechanism work?",
        "What is the difference between BERT and GPT?",
    ]

    print("\n" + "="*80)
    print("TESTING RETRIEVAL PIPELINE")
    print("="*80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Query {i}/{len(test_queries)}")
        print(f"{'='*80}")

        # Retrieve with verbose output
        results = retriever.retrieve(
            query=query,
            top_k=5,
            apply_mmr=True,
            verbose=True,
        )

        print("\nFinal Results Detail:")
        print("-" * 80)
        for result in results:
            print(f"\n[Rank {result.rank}] Score: {result.relevance_score:.4f}")
            print(f"Collection: {result.collection}")
            print(f"Content: {result.content}")
            if result.metadata:
                print(f"Metadata: {result.metadata}")

        print("\n" + "="*80 + "\n")
        input("Press Enter to continue to next query...")

    print("\n✓ All tests complete!")


def test_simple_interface():
    """Test the simplified retrieval interface."""
    print("\n" + "="*80)
    print("TESTING SIMPLE INTERFACE")
    print("="*80 + "\n")

    settings = get_settings()
    retriever = create_advanced_retriever(settings)

    query = "What is deep learning?"
    print(f"Query: {query}\n")

    # Simple retrieval
    results = retriever.retrieve_simple(query, top_k=3)

    print("Results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc}")

    print("\n✓ Simple interface test complete!")


if __name__ == "__main__":
    # Run main test
    main()

    # Uncomment to test simple interface
    # test_simple_interface()
