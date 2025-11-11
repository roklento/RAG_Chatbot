"""
Data Ingestion Script - Loads, chunks, embeds and inserts data into Qdrant.

This script:
1. Loads Q&A pairs and plain text documents
2. Chunks plain text into manageable pieces
3. Generates embeddings using Qwen model
4. Generates sparse vectors (BM25)
5. Inserts into appropriate Qdrant collections
"""

import sys
sys.path.append('..')

import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from fastembed import SparseTextEmbedding

from src.config import get_settings
from src.models import QwenEmbedding
from src.data import create_chunker


def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_sparse_vectors(texts: List[str]) -> List[SparseVector]:
    """
    Generate sparse vectors using BM25-style embedding.

    Args:
        texts: List of texts to embed

    Returns:
        List of SparseVector objects
    """
    print("  Generating sparse vectors (BM25)...")

    # Initialize sparse embedding model
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    sparse_vectors = []

    for text in tqdm(texts, desc="  Sparse embedding"):
        # Generate sparse embedding
        embeddings = list(sparse_model.embed([text]))

        if embeddings:
            embedding = embeddings[0]
            # Convert to SparseVector format
            sparse_vectors.append(SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            ))
        else:
            # Empty sparse vector as fallback
            sparse_vectors.append(SparseVector(indices=[], values=[]))

    return sparse_vectors


def ingest_qa_pairs(
    client: QdrantClient,
    embedding_model: QwenEmbedding,
    qa_data: List[Dict],
):
    """
    Ingest Q&A pairs into Qdrant.

    Args:
        client: Qdrant client
        embedding_model: Embedding model
        qa_data: List of Q&A dicts with 'question' and 'answer'
    """
    print("\n" + "="*80)
    print("INGESTING Q&A PAIRS")
    print("="*80 + "\n")

    collection_name = "qa_pairs"

    print(f"Total Q&A pairs: {len(qa_data)}")

    # Prepare texts for embedding (combine question and answer)
    texts = []
    for qa in qa_data:
        combined = f"Soru: {qa['question']}\nCevap: {qa['answer']}"
        texts.append(combined)

    # Generate dense embeddings
    print("\n  Generating dense embeddings...")
    dense_embeddings = []

    for text in tqdm(texts, desc="  Dense embedding"):
        embedding = embedding_model.embed_text(text)
        dense_embeddings.append(embedding)

    # Generate sparse vectors
    sparse_vectors = generate_sparse_vectors(texts)

    # Create points for Qdrant
    print("\n  Creating Qdrant points...")
    points = []

    for idx, (qa, text, dense_emb, sparse_vec) in enumerate(zip(
        qa_data, texts, dense_embeddings, sparse_vectors
    )):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_emb,
                "sparse": sparse_vec,
            },
            payload={
                "question": qa["question"],
                "answer": qa["answer"],
                "combined_text": text,
                "source_type": "qa_pair",
                "index": idx,
            },
        )
        points.append(point)

    # Insert into Qdrant
    print(f"\n  Inserting {len(points)} points into '{collection_name}'...")

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )

    print(f"  ✓ Inserted {len(points)} Q&A pairs")

    # Verify
    collection_info = client.get_collection(collection_name)
    print(f"  Collection now has {collection_info.points_count} total points")


def ingest_plain_text(
    client: QdrantClient,
    embedding_model: QwenEmbedding,
    text_data: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    """
    Ingest plain text documents into Qdrant (with chunking).

    Args:
        client: Qdrant client
        embedding_model: Embedding model
        text_data: List of document dicts with 'title' and 'content'
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
    """
    print("\n" + "="*80)
    print("INGESTING PLAIN TEXT DOCUMENTS")
    print("="*80 + "\n")

    collection_name = "plain_text"

    print(f"Total documents: {len(text_data)}")

    # Create chunker
    chunker = create_chunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=100,
    )

    # Chunk all documents
    print("\n  Chunking documents...")
    chunks = chunker.chunk_documents(
        documents=text_data,
        title_key="title",
        content_key="content",
        strategy="sentence_aware",
    )

    print(f"  ✓ Created {len(chunks)} chunks")

    # Show chunk statistics
    stats = chunker.get_chunk_summary(chunks)
    print(f"\n  Chunk Statistics:")
    print(f"    Total chunks: {stats['total_chunks']}")
    print(f"    Avg length: {stats['avg_length']:.0f} characters")
    print(f"    Min length: {stats['min_length']} characters")
    print(f"    Max length: {stats['max_length']} characters")
    print(f"    Unique sources: {stats['unique_sources']}")

    # Prepare texts
    texts = [chunk.content for chunk in chunks]

    # Generate dense embeddings
    print("\n  Generating dense embeddings...")
    dense_embeddings = []

    for text in tqdm(texts, desc="  Dense embedding"):
        embedding = embedding_model.embed_text(text)
        dense_embeddings.append(embedding)

    # Generate sparse vectors
    sparse_vectors = generate_sparse_vectors(texts)

    # Create points for Qdrant
    print("\n  Creating Qdrant points...")
    points = []

    for chunk, dense_emb, sparse_vec in zip(chunks, dense_embeddings, sparse_vectors):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_emb,
                "sparse": sparse_vec,
            },
            payload={
                "content": chunk.content,
                "source_title": chunk.source_title,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "source_type": "plain_text",
                "metadata": chunk.metadata or {},
            },
        )
        points.append(point)

    # Insert into Qdrant
    print(f"\n  Inserting {len(points)} points into '{collection_name}'...")

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )

    print(f"  ✓ Inserted {len(points)} chunks")

    # Verify
    collection_info = client.get_collection(collection_name)
    print(f"  Collection now has {collection_info.points_count} total points")


def main():
    """Main ingestion pipeline."""
    print("="*80)
    print("DATA INGESTION PIPELINE")
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

    # Initialize embedding model
    print("Loading embedding model...")
    print(f"  Model: {settings.embedding_model_path}")
    print(f"  Device: {settings.device}")

    embedding_model = QwenEmbedding(
        model_path=settings.embedding_model_path,
        device=settings.device,
    )
    print(f"✓ Model loaded ({embedding_model.embedding_dimension} dimensions)\n")

    # Define data paths
    data_dir = Path("../data")
    qa_pairs_file = data_dir / "rise_online_qa_pairs.json"
    plain_text_file = data_dir / "rise_online_plain_text.json"

    # Check if files exist
    if not qa_pairs_file.exists():
        print(f"❌ Q&A pairs file not found: {qa_pairs_file}")
        return

    if not plain_text_file.exists():
        print(f"❌ Plain text file not found: {plain_text_file}")
        return

    # Load data
    print("Loading datasets...")
    qa_data = load_json_data(str(qa_pairs_file))
    text_data = load_json_data(str(plain_text_file))
    print(f"  ✓ Loaded {len(qa_data)} Q&A pairs")
    print(f"  ✓ Loaded {len(text_data)} plain text documents")

    # Ingest Q&A pairs
    ingest_qa_pairs(
        client=client,
        embedding_model=embedding_model,
        qa_data=qa_data,
    )

    # Ingest plain text
    ingest_plain_text(
        client=client,
        embedding_model=embedding_model,
        text_data=text_data,
        chunk_size=512,
        chunk_overlap=50,
    )

    # Final summary
    print("\n" + "="*80)
    print("INGESTION COMPLETE")
    print("="*80 + "\n")

    print("Collection Summary:")
    for collection_name in ["qa_pairs", "plain_text"]:
        info = client.get_collection(collection_name)
        print(f"  {collection_name}: {info.points_count} points")

    print("\n✓ Data ingestion successful!")
    print("\nYour RAG chatbot is now ready to use!")
    print("Next step: Test the chatbot")
    print("  python examples/08_complete_rag_demo.py --interactive")


if __name__ == "__main__":
    main()
