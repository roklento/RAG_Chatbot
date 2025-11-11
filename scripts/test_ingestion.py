"""
Test Data Ingestion - Verifies that data was ingested correctly.

This script:
1. Connects to Qdrant
2. Checks collections exist
3. Verifies point counts
4. Samples data from each collection
5. Tests basic search functionality
"""

import sys
sys.path.append('..')

from qdrant_client import QdrantClient
from src.config import get_settings
from src.models import QwenEmbedding


def test_connection():
    """Test Qdrant connection."""
    print("\n" + "="*80)
    print("TEST 1: QDRANT CONNECTION")
    print("="*80 + "\n")

    settings = get_settings()

    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        print(f"Connecting to {settings.qdrant_host}:{settings.qdrant_port}...")
        collections = client.get_collections().collections
        print(f"✓ Connected successfully")
        print(f"✓ Found {len(collections)} collections")
        return client
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None


def test_collections(client):
    """Test that required collections exist."""
    print("\n" + "="*80)
    print("TEST 2: COLLECTION EXISTENCE")
    print("="*80 + "\n")

    required_collections = ["qa_pairs", "plain_text"]

    for collection_name in required_collections:
        try:
            info = client.get_collection(collection_name)
            print(f"✓ Collection '{collection_name}' exists")
            print(f"  - Points: {info.points_count}")
            print(f"  - Vectors: {info.vectors_count}")
            print()
        except Exception as e:
            print(f"❌ Collection '{collection_name}' not found: {e}")
            print()


def test_data_samples(client):
    """Sample and display data from collections."""
    print("\n" + "="*80)
    print("TEST 3: DATA SAMPLES")
    print("="*80 + "\n")

    # Sample Q&A pairs
    print("Q&A Pairs Sample:")
    print("-" * 80)
    try:
        points = client.scroll(
            collection_name="qa_pairs",
            limit=2,
        )[0]

        for i, point in enumerate(points, 1):
            print(f"\n{i}. Question: {point.payload['question']}")
            print(f"   Answer: {point.payload['answer'][:150]}...")

        print("\n✓ Q&A pairs data looks good")
    except Exception as e:
        print(f"❌ Error sampling Q&A pairs: {e}")

    print("\n" + "-" * 80)

    # Sample plain text
    print("\nPlain Text Chunks Sample:")
    print("-" * 80)
    try:
        points = client.scroll(
            collection_name="plain_text",
            limit=2,
        )[0]

        for i, point in enumerate(points, 1):
            print(f"\n{i}. Title: {point.payload['source_title']}")
            print(f"   Chunk {point.payload['chunk_index'] + 1}/{point.payload['total_chunks']}")
            print(f"   Content: {point.payload['content'][:150]}...")

        print("\n✓ Plain text data looks good")
    except Exception as e:
        print(f"❌ Error sampling plain text: {e}")


def test_search(client):
    """Test basic search functionality."""
    print("\n" + "="*80)
    print("TEST 4: BASIC SEARCH")
    print("="*80 + "\n")

    # Load embedding model
    print("Loading embedding model...")
    settings = get_settings()

    try:
        embedding_model = QwenEmbedding(
            model_path=settings.embedding_model_path,
            device=settings.device,
        )
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        print("   Skipping search test")
        return

    # Test query
    test_query = "Makine öğrenmesi nedir?"

    print(f"Test Query: {test_query}\n")

    # Search Q&A pairs
    print("Searching Q&A pairs...")
    try:
        query_vector = embedding_model.embed_text(test_query)

        results = client.search(
            collection_name="qa_pairs",
            query_vector=("dense", query_vector),
            limit=3,
        )

        print(f"✓ Found {len(results)} results\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.4f}")
            print(f"   Question: {result.payload['question']}")
            print(f"   Answer: {result.payload['answer'][:100]}...\n")

    except Exception as e:
        print(f"❌ Search failed: {e}")

    # Search plain text
    print("-" * 80)
    print("\nSearching plain text...")
    try:
        results = client.search(
            collection_name="plain_text",
            query_vector=("dense", query_vector),
            limit=3,
        )

        print(f"✓ Found {len(results)} results\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.4f}")
            print(f"   Title: {result.payload['source_title']}")
            print(f"   Content: {result.payload['content'][:100]}...\n")

    except Exception as e:
        print(f"❌ Search failed: {e}")


def test_vector_types(client):
    """Test that both dense and sparse vectors exist."""
    print("\n" + "="*80)
    print("TEST 5: VECTOR TYPES")
    print("="*80 + "\n")

    for collection_name in ["qa_pairs", "plain_text"]:
        print(f"Checking {collection_name}...")
        try:
            info = client.get_collection(collection_name)

            # Check dense vectors
            if "dense" in str(info.config):
                print(f"  ✓ Dense vectors configured")

            # Check sparse vectors
            if "sparse" in str(info.config):
                print(f"  ✓ Sparse vectors configured")

            print()
        except Exception as e:
            print(f"  ❌ Error: {e}\n")


def main():
    """Run all tests."""
    print("="*80)
    print("DATA INGESTION VERIFICATION TESTS")
    print("="*80)

    # Test 1: Connection
    client = test_connection()
    if not client:
        print("\n❌ Cannot proceed without Qdrant connection")
        print("   Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        return

    # Test 2: Collections
    test_collections(client)

    # Test 3: Data samples
    test_data_samples(client)

    # Test 4: Search
    test_search(client)

    # Test 5: Vector types
    test_vector_types(client)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")

    print("If all tests passed, your data ingestion was successful!")
    print("\nNext steps:")
    print("  1. Test full retrieval: python ../examples/03_test_retrieval.py")
    print("  2. Start chatbot: python ../examples/08_complete_rag_demo.py --interactive")


if __name__ == "__main__":
    main()
