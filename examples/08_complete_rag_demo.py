"""
Example 8: Complete RAG Pipeline Demonstration.

This script demonstrates the FULL RAG system:
1. Query processing (correction + diversification)
2. Hybrid retrieval (dual collections)
3. Reranking
4. MMR diversity filtering
5. Context augmentation
6. Conversation history
7. Streaming generation
8. Response post-processing

This is the complete end-to-end RAG chatbot!
"""

import sys
import asyncio
sys.path.append('..')

from src.config import get_settings
from src.models import QwenEmbedding
from src.retrieval import create_advanced_retriever
from src.generation import (
    VLLMStreamingGenerator,
    create_rag_pipeline,
)
from qdrant_client import QdrantClient


async def initialize_pipeline():
    """Initialize complete RAG pipeline."""
    print("Initializing RAG pipeline...\n")

    settings = get_settings()

    # 1. Initialize Qdrant
    print("1. Connecting to Qdrant...")
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    print("   ✓ Connected")

    # 2. Initialize embedding model
    print("2. Loading embedding model...")
    embedding_model = QwenEmbedding(
        model_path=settings.embedding_model_path,
        device=settings.device,
    )
    print(f"   ✓ Loaded ({embedding_model.embedding_dimension} dimensions)")

    # 3. Initialize advanced retriever
    print("3. Setting up advanced retriever...")
    retriever = create_advanced_retriever(
        settings=settings,
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
    )
    print("   ✓ Retriever ready (query processing + hybrid search + reranking + MMR)")

    # 4. Initialize vLLM generator
    print("4. Connecting to vLLM server...")
    generator = VLLMStreamingGenerator(settings=settings)

    is_healthy = await generator.health_check()
    if not is_healthy:
        raise RuntimeError(
            f"vLLM server not available at {generator.server_url}\n"
            f"Please start the server: bash examples/04_setup_vllm.sh"
        )
    print("   ✓ vLLM server connected")

    # 5. Create RAG pipeline
    print("5. Creating RAG pipeline...")
    pipeline = create_rag_pipeline(
        settings=settings,
        retriever=retriever,
        generator=generator,
    )
    print("   ✓ Pipeline ready")

    print("\n" + "="*80)
    print("RAG PIPELINE INITIALIZED SUCCESSFULLY")
    print("="*80 + "\n")

    return pipeline


async def single_query_demo(pipeline):
    """Demonstrate single query processing."""
    print("\n" + "="*80)
    print("SINGLE QUERY DEMO")
    print("="*80 + "\n")

    query = "Makine öğrenmesi nedir?"

    print(f"Query: {query}\n")
    print("Processing with full RAG pipeline...\n")
    print("-"*80)

    # Process query with streaming
    print("🤖 Response:\n")

    final_response = None
    async for result in pipeline.query_stream(query, session_id="demo_1", verbose=True):
        if isinstance(result, str):
            # Stream tokens
            print(result, end="", flush=True)
        else:
            # Final metadata
            final_response = result

    print("\n" + "-"*80)

    # Show metadata
    print("\n📊 Response Metadata:")
    print(f"   Confidence: {final_response.processed_response.confidence_score:.2f}")
    print(f"   Citations used: {final_response.processed_response.citations_used}")
    print(f"   Sources coverage: {final_response.processed_response.sources_coverage:.1%}")
    print(f"   Retrieval time: {final_response.retrieval_time_ms:.2f}ms")
    print(f"   Generation time: {final_response.generation_time_ms:.2f}ms")
    print(f"   Total time: {final_response.total_time_ms:.2f}ms")
    print(f"   Conversation tokens: {final_response.conversation_tokens:,}")

    if final_response.processed_response.processing_notes:
        print(f"\n   Processing notes:")
        for note in final_response.processed_response.processing_notes:
            print(f"     - {note}")

    print("\n✓ Single query demo complete!")


async def multi_turn_conversation_demo(pipeline):
    """Demonstrate multi-turn conversation."""
    print("\n" + "="*80)
    print("MULTI-TURN CONVERSATION DEMO")
    print("="*80 + "\n")

    session_id = "demo_conversation"

    queries = [
        "Makine öğrenmesi nedir?",
        "Denetimli ve denetimsiz öğrenme arasındaki fark nedir?",
        "Bana bir örnek verebilir misin?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Turn {i}/{len(queries)}")
        print(f"{'='*80}\n")

        print(f"👤 User: {query}\n")
        print("🤖 Assistant: ", end="", flush=True)

        # Process with streaming
        async for result in pipeline.query_stream(query, session_id=session_id, top_k=5):
            if isinstance(result, str):
                print(result, end="", flush=True)
            else:
                final_response = result

        print("\n")

        # Show brief stats
        print(f"   [Confidence: {final_response.processed_response.confidence_score:.2f}, "
              f"Citations: {final_response.processed_response.citations_used}, "
              f"Time: {final_response.total_time_ms:.0f}ms]")

        # Small delay for readability
        await asyncio.sleep(1)

    # Show conversation stats
    print(f"\n{'='*80}")
    print("CONVERSATION STATISTICS")
    print(f"{'='*80}\n")

    stats = pipeline.get_conversation_stats(session_id)
    print(f"Session: {stats['session_id']}")
    print(f"Messages: {stats['message_count']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Token usage: {stats['token_usage_percent']:.1f}%")
    print(f"Should reset: {stats['should_reset']}")

    print("\n✓ Multi-turn conversation demo complete!")


async def interactive_mode(pipeline):
    """Run interactive RAG chatbot."""
    print("\n" + "="*80)
    print("INTERACTIVE RAG CHATBOT")
    print("="*80)
    print("\nType your questions in Turkish or English.")
    print("Commands: 'quit' to exit, 'reset' to clear history, 'stats' for statistics\n")

    session_id = "interactive"

    while True:
        try:
            query = input("\n👤 You: ").strip()

            if not query:
                continue

            if query.lower() == 'quit':
                print("\n👋 Goodbye!")
                break

            if query.lower() == 'reset':
                pipeline.reset_conversation(session_id)
                print("✓ Conversation history reset")
                continue

            if query.lower() == 'stats':
                stats = pipeline.get_conversation_stats(session_id)
                print(f"\n📊 Statistics:")
                print(f"   Messages: {stats['message_count']}")
                print(f"   Tokens: {stats['total_tokens']:,} / {stats['max_tokens']:,}")
                print(f"   Usage: {stats['token_usage_percent']:.1f}%")
                continue

            # Process query
            print("\n🤖 Assistant: ", end="", flush=True)

            async for result in pipeline.query_stream(query, session_id=session_id):
                if isinstance(result, str):
                    print(result, end="", flush=True)

            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def health_check_demo(pipeline):
    """Demonstrate health checking."""
    print("\n" + "="*80)
    print("HEALTH CHECK")
    print("="*80 + "\n")

    health = await pipeline.health_check()

    print("Component Health:")
    print(f"  Retriever: {'✓' if health['retriever'] else '❌'}")
    print(f"  Generator: {'✓' if health['generator'] else '❌'}")
    print(f"  Pipeline: {'✓' if health['pipeline'] else '❌'}")

    if health['pipeline']:
        print("\n✓ All systems operational!")
    else:
        print("\n❌ Some components are not healthy")


async def main():
    """Main demo entry point."""
    import sys

    try:
        # Initialize pipeline
        pipeline = await initialize_pipeline()

        # Check command line arguments
        if len(sys.argv) > 1:
            mode = sys.argv[1]

            if mode == '--interactive':
                await interactive_mode(pipeline)
                return
            elif mode == '--health':
                await health_check_demo(pipeline)
                return

        # Run demos
        await health_check_demo(pipeline)
        await single_query_demo(pipeline)
        await multi_turn_conversation_demo(pipeline)

        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED")
        print("="*80 + "\n")

        print("To run interactive mode: python 08_complete_rag_demo.py --interactive")
        print("To check health only: python 08_complete_rag_demo.py --health")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
