"""
Colab Demo Script - Complete RAG chatbot demo for Google Colab.

This is the main script to run in Google Colab.
Handles all setup and provides interactive chatbot interface.
"""

import os
import sys
import asyncio
from pathlib import Path

# Ensure we're in the right directory
if not os.path.exists('src'):
    print("❌ Error: 'src' directory not found!")
    print("   Make sure you're running this script from the RAG_Chatbot directory.")
    sys.exit(1)

print("="*80)
print("RISE ONLINE RAG CHATBOT - COLAB DEMO")
print("Powered by Gemma-3n-E4B-it")
print("="*80)
print()

# Import components
from qdrant_client import QdrantClient
from src.config import get_settings
from src.models import QwenEmbedding, QwenReranker, GemmaQueryProcessor
from src.retrieval import create_advanced_retriever
from src.generation import (
    GemmaStreamingGenerator,
    create_colab_rag_pipeline,
)


async def initialize_rag_system():
    """Initialize complete RAG system for Colab."""
    print("🚀 Initializing RAG System...")
    print()

    # 1. Load settings
    print("1️⃣  Loading configuration...")
    settings = get_settings()
    print(f"   ✓ Settings loaded")
    print()

    # 2. Initialize Qdrant (in-memory for Colab)
    print("2️⃣  Initializing Qdrant (in-memory)...")
    qdrant_client = QdrantClient(":memory:")
    print("   ✓ Qdrant initialized")
    print()

    # 3. Load embedding model
    print("3️⃣  Loading embedding model...")
    print(f"   Model: {settings.embedding_model_path}")
    embedding_model = QwenEmbedding(
        model_path=settings.embedding_model_path,
        device=settings.device,
    )
    print(f"   ✓ Embedding model loaded ({embedding_model.embedding_dimension}D)")
    print()

    # 4. Load reranker model
    print("4️⃣  Loading reranker model...")
    print(f"   Model: {settings.reranker_model_path}")
    reranker_model = QwenReranker(
        model_path=settings.reranker_model_path,
        device=settings.device,
    )
    print(f"   ✓ Reranker loaded")
    print()

    # 5. Load Gemma query processor
    print("5️⃣  Loading Gemma query processor...")
    print(f"   Model: {settings.generation_model_path}")
    query_processor = GemmaQueryProcessor(
        model_path=settings.generation_model_path,
        device=settings.device,
    )
    print("   ✓ Query processor loaded")
    print()

    # 6. Ingest data into Qdrant
    print("6️⃣  Ingesting dataset into Qdrant...")
    print("   This may take a few minutes...")

    # Import ingestion functions
    sys.path.append('scripts')
    from scripts.ingest_data import (
        load_json_data,
        ingest_qa_pairs,
        ingest_plain_text,
    )

    # Load datasets
    qa_data = load_json_data('data/rise_online_qa_pairs.json')
    text_data = load_json_data('data/rise_online_plain_text.json')

    # Ingest
    ingest_qa_pairs(qdrant_client, embedding_model, qa_data)
    ingest_plain_text(qdrant_client, embedding_model, text_data)

    print("   ✓ Data ingested")
    print()

    # 7. Create advanced retriever
    print("7️⃣  Setting up retriever...")
    retriever = create_advanced_retriever(
        settings=settings,
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        reranker=reranker_model,
        query_processor=query_processor,
    )
    print("   ✓ Retriever ready")
    print()

    # 8. Load Gemma generator
    print("8️⃣  Loading Gemma generator...")
    print(f"   Model: {settings.generation_model_path}")
    generator = GemmaStreamingGenerator(
        settings=settings,
        model_path=settings.generation_model_path,
        device=settings.device,
    )
    print("   ✓ Generator loaded")
    print()

    # 9. Create RAG pipeline
    print("9️⃣  Creating RAG pipeline...")
    pipeline = create_colab_rag_pipeline(
        settings=settings,
        retriever=retriever,
        generator=generator,
    )
    print()

    print("="*80)
    print("✅ INITIALIZATION COMPLETE!")
    print("="*80)
    print()

    return pipeline


async def interactive_chat(pipeline):
    """Run interactive chat loop."""
    print("💬 Interactive Chat Mode")
    print("-"*80)
    print("Commands:")
    print("  'quit' or 'exit' - Exit chat")
    print("  'reset' - Reset conversation history")
    print("  'stats' - Show conversation statistics")
    print("-"*80)
    print()

    session_id = "colab_session"

    while True:
        try:
            # Get user input
            query = input("\n👤 You: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break

            if query.lower() == 'reset':
                pipeline.reset_conversation(session_id)
                continue

            if query.lower() == 'stats':
                stats = pipeline.get_conversation_stats(session_id)
                print(f"\n📊 Conversation Statistics:")
                print(f"   Messages: {stats['message_count']}")
                print(f"   Tokens: {stats['total_tokens']:,} / {stats['max_tokens']:,}")
                print(f"   Usage: {stats['token_usage_percent']:.1f}%")
                continue

            # Process query with streaming
            print("\n🤖 Assistant: ", end="", flush=True)

            async for result in pipeline.query_stream(
                query,
                session_id=session_id,
                verbose=False,
            ):
                if isinstance(result, str):
                    print(result, end="", flush=True)

            print()  # New line after response

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def demo_queries(pipeline):
    """Run demo with pre-defined queries."""
    print("🎬 Running Demo Queries...")
    print("="*80)
    print()

    demo_queries = [
        "Rise Online nedir?",
        "Hangi karakter sınıfları var?",
        "Seviye nasıl atlanır?",
    ]

    session_id = "demo_session"

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*80}")
        print(f"Demo Query {i}/{len(demo_queries)}")
        print(f"{'='*80}")
        print(f"\n👤 User: {query}\n")
        print("🤖 Assistant: ", end="", flush=True)

        async for result in pipeline.query_stream(
            query,
            session_id=session_id,
            verbose=False,
        ):
            if isinstance(result, str):
                print(result, end="", flush=True)

        print("\n")

        # Wait a bit between queries
        await asyncio.sleep(1)

    print("="*80)
    print("✅ Demo complete!")
    print("="*80)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Colab RAG Chatbot Demo")
    parser.add_argument(
        '--mode',
        choices=['interactive', 'demo'],
        default='demo',
        help='Mode: interactive chat or demo queries'
    )

    args = parser.parse_args()

    # Initialize system
    pipeline = await initialize_rag_system()

    # Run selected mode
    if args.mode == 'interactive':
        await interactive_chat(pipeline)
    else:
        await demo_queries(pipeline)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
