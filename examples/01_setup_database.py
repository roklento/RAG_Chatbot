"""
Example 1: Setup Qdrant database and ingest sample data.

This script demonstrates:
1. Creating Qdrant collections
2. Adding Q&A pairs
3. Adding plain text documents
"""

import sys
sys.path.append('..')

from src.config import get_settings
from src.models import QwenEmbedding
from src.retrieval import QdrantManager


def main():
    # Load settings
    settings = get_settings()
    print("Settings loaded")
    print(f"  Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"  Q&A Collection: {settings.qa_collection_name}")
    print(f"  Text Collection: {settings.text_collection_name}\n")

    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = QwenEmbedding(
        model_path=settings.embedding_model_path,
        device=settings.device,
        max_length=settings.embedding_max_length,
    )
    print(f"✓ Embedding dimension: {embedding_model.embedding_dimension}\n")

    # Initialize Qdrant manager
    print("Initializing Qdrant manager...")
    manager = QdrantManager(settings=settings, embedding_model=embedding_model)
    print("✓ Connected to Qdrant\n")

    # Create collections
    print("Creating collections...")
    manager.create_collections(recreate=True)
    print()

    # Sample Q&A pairs
    qa_pairs = [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "question": "What is the difference between supervised and unsupervised learning?",
            "answer": "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data."
        },
        {
            "question": "What is a neural network?",
            "answer": "A neural network is a computing system inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers."
        },
        {
            "question": "What is deep learning?",
            "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to learn complex patterns."
        },
        {
            "question": "What is natural language processing?",
            "answer": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language."
        },
    ]

    # Add Q&A pairs
    print("Adding Q&A pairs...")
    count = manager.add_qa_pairs(qa_pairs)
    print(f"✓ Added {count} Q&A pairs\n")

    # Sample plain text documents
    plain_texts = [
        "Transformers are a type of neural network architecture introduced in the 'Attention is All You Need' paper. They use self-attention mechanisms to process sequential data and have become the foundation for modern NLP models like BERT and GPT.",

        "The attention mechanism allows models to focus on relevant parts of the input when making predictions. It computes attention weights that determine how much focus to place on different parts of the input sequence.",

        "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model that learns bidirectional context by training on masked language modeling and next sentence prediction tasks.",

        "GPT (Generative Pre-trained Transformer) is an autoregressive language model that generates text by predicting the next token given previous tokens. It is trained on large amounts of text data.",

        "Fine-tuning is the process of taking a pre-trained model and training it on a specific task with task-specific data. This allows leveraging knowledge learned during pre-training for downstream tasks.",

        "Vector embeddings are numerical representations of text that capture semantic meaning. Similar texts have similar embeddings in the vector space, enabling semantic search and similarity comparisons.",

        "RAG (Retrieval-Augmented Generation) combines retrieval systems with generative models. It retrieves relevant documents from a knowledge base and uses them to generate more accurate and informed responses.",
    ]

    # Add plain texts
    print("Adding plain text documents...")
    count = manager.add_plain_text(plain_texts)
    print(f"✓ Added {count} plain text documents\n")

    # Show collection info
    print("Collection Information:")
    for collection_name in [settings.qa_collection_name, settings.text_collection_name]:
        info = manager.get_collection_info(collection_name)
        print(f"\n{collection_name}:")
        print(f"  Points: {info['points_count']}")
        print(f"  Vectors: {info['vectors_count']}")
        print(f"  Status: {info['status']}")

    print("\n✓ Database setup complete!")


if __name__ == "__main__":
    main()
