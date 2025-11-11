"""
Setup Qdrant Collections - Creates required collections for the RAG system.

This script:
1. Connects to local Qdrant instance
2. Creates two collections: qa_pairs and plain_text
3. Configures for hybrid search (dense + sparse)
4. Sets up proper indexes
"""

import sys
sys.path.append('..')

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
)
from src.config import get_settings


def setup_collections():
    """Setup Qdrant collections for the RAG system."""
    print("="*80)
    print("QDRANT COLLECTION SETUP")
    print("="*80 + "\n")

    # Load settings
    settings = get_settings()

    # Connect to Qdrant
    print(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}...")
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    print("✓ Connected\n")

    # Collection configurations
    collections_config = {
        "qa_pairs": {
            "description": "Q&A pairs collection - optimized for exact matching",
            "sparse_weight": 0.7,  # 70% sparse, 30% dense
            "dense_weight": 0.3,
        },
        "plain_text": {
            "description": "Plain text chunks collection - optimized for semantic search",
            "sparse_weight": 0.3,  # 30% sparse, 70% dense
            "dense_weight": 0.7,
        },
    }

    # Get embedding dimension from settings
    embedding_dim = settings.embedding_dimension  # 1024 for Qwen3-Embedding-4B

    # Create collections
    for collection_name, config in collections_config.items():
        print(f"Setting up collection: {collection_name}")
        print(f"  Description: {config['description']}")
        print(f"  Weights: {config['sparse_weight']*100:.0f}% sparse, {config['dense_weight']*100:.0f}% dense")

        try:
            # Check if collection exists
            try:
                client.get_collection(collection_name)
                print(f"  Collection already exists")

                # Ask if user wants to recreate
                response = input(f"  Do you want to recreate '{collection_name}'? (yes/no): ").strip().lower()
                if response in ['yes', 'y']:
                    client.delete_collection(collection_name)
                    print(f"  ✓ Deleted existing collection")
                else:
                    print(f"  Skipping {collection_name}")
                    print()
                    continue
            except Exception:
                pass  # Collection doesn't exist, will create

            # Create collection with hybrid search support
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    # Dense vectors (from embedding model)
                    "dense": VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    # Sparse vectors (BM25-style)
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,  # Keep in memory for speed
                        ),
                    ),
                },
            )

            print(f"  ✓ Created collection")
            print(f"  - Dense vector dimension: {embedding_dim}")
            print(f"  - Sparse vector support: enabled")
            print()

        except Exception as e:
            print(f"  ❌ Error creating collection: {e}")
            print()

    # List all collections
    print("="*80)
    print("COLLECTIONS SUMMARY")
    print("="*80 + "\n")

    collections = client.get_collections().collections

    for collection in collections:
        print(f"Collection: {collection.name}")
        info = client.get_collection(collection.name)
        print(f"  Points: {info.points_count}")
        print(f"  Vectors: {info.vectors_count}")
        print()

    print("="*80)
    print("SETUP COMPLETE")
    print("="*80)
    print("\nCollections are ready for data ingestion!")
    print("Next step: Run the data ingestion script")
    print("  python scripts/ingest_data.py")


if __name__ == "__main__":
    setup_collections()
