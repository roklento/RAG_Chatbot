"""
Example 6: Test streaming functionality with context.

This script demonstrates:
1. Conversation memory with streaming
2. Context augmentation
3. Prompt building
4. Post-processing
"""

import sys
import asyncio
sys.path.append('..')

from src.config import get_settings
from src.generation import (
    VLLMStreamingGenerator,
    ConversationMemoryManager,
    ContextAugmenter,
    PromptBuilder,
    ResponsePostProcessor,
    AugmentedContext,
)


async def test_conversation_memory():
    """Test conversation memory management."""
    print("\n" + "="*80)
    print("CONVERSATION MEMORY TEST")
    print("="*80 + "\n")

    settings = get_settings()
    memory = ConversationMemoryManager(settings)

    # Add some messages
    session_id = "test_session"

    memory.add_message(session_id, "user", "Merhaba, makine öğrenmesi nedir?")
    memory.add_message(session_id, "assistant", "Makine öğrenmesi, bilgisayarların verilerden öğrenmesini sağlayan bir yapay zeka dalıdır.")
    memory.add_message(session_id, "user", "Derin öğrenme ile arasındaki fark nedir?")

    # Get history
    history = memory.get_history(session_id)

    print(f"Session: {session_id}")
    print(f"Messages: {len(history)}")
    print(f"Total tokens: {memory.get_total_tokens(session_id):,}")
    print(f"\nConversation History:")
    print("-"*80)

    for msg in history:
        role_label = "👤 User" if msg.role == "user" else "🤖 Assistant"
        print(f"\n{role_label} ({msg.tokens} tokens):")
        print(msg.content)

    print("\n✓ Conversation memory test complete!")


def test_context_augmentation():
    """Test context augmentation."""
    print("\n" + "="*80)
    print("CONTEXT AUGMENTATION TEST")
    print("="*80 + "\n")

    settings = get_settings()
    augmenter = ContextAugmenter(settings)

    # Simulate retrieved results
    from src.retrieval.hybrid_retriever import SearchResult

    mock_results = [
        SearchResult(
            content="Makine öğrenmesi, sistemlerin verilerden öğrenmesini sağlayan yapay zekanın bir alt kümesidir.",
            collection="qa_pairs",
            score=0.95,
            metadata={},
        ),
        SearchResult(
            content="Derin öğrenme, çok katmanlı yapay sinir ağları kullanarak öğrenme yapan bir makine öğrenmesi yaklaşımıdır.",
            collection="plain_text",
            score=0.88,
            metadata={},
        ),
        SearchResult(
            content="Denetimli öğrenme, etiketlenmiş eğitim verileri kullanarak model eğitir.",
            collection="qa_pairs",
            score=0.82,
            metadata={},
        ),
    ]

    # Augment contexts
    augmented = augmenter.augment_contexts(mock_results, max_contexts=3)

    print(f"Augmented {len(augmented)} contexts:\n")

    for ctx in augmented:
        print(f"[{ctx.citation_id}] (Score: {ctx.relevance_score:.2f}, Collection: {ctx.source_collection})")
        print(f"    {ctx.content[:100]}...")
        print(f"    Tokens: {ctx.token_count}")
        print()

    # Format for prompt
    formatted = augmenter.format_for_prompt(augmented)
    print("Formatted for prompt:")
    print("-"*80)
    print(formatted)
    print("-"*80)

    print("\n✓ Context augmentation test complete!")


def test_prompt_building():
    """Test prompt building."""
    print("\n" + "="*80)
    print("PROMPT BUILDING TEST")
    print("="*80 + "\n")

    settings = get_settings()
    builder = PromptBuilder(settings)

    # Create mock augmented contexts
    contexts = [
        AugmentedContext(
            citation_id=1,
            content="Makine öğrenmesi, yapay zekanın bir alt dalıdır.",
            source_collection="qa_pairs",
            relevance_score=0.95,
            token_count=15,
        ),
        AugmentedContext(
            citation_id=2,
            content="Derin öğrenme, sinir ağları kullanır.",
            source_collection="plain_text",
            relevance_score=0.88,
            token_count=12,
        ),
    ]

    # Mock conversation history
    from src.generation import ConversationMessage

    history = [
        ConversationMessage(role="user", content="Merhaba", tokens=5),
        ConversationMessage(role="assistant", content="Merhaba! Size nasıl yardımcı olabilirim?", tokens=10),
    ]

    # Build prompt
    query = "Makine öğrenmesi nedir?"
    prompt = builder.build_prompt(query, contexts, history)

    print("Built prompt:")
    print("-"*80)
    print(prompt)
    print("-"*80)
    print(f"\nPrompt length: ~{len(prompt.split())} words")

    print("\n✓ Prompt building test complete!")


async def test_streaming_with_post_processing():
    """Test streaming generation with post-processing."""
    print("\n" + "="*80)
    print("STREAMING + POST-PROCESSING TEST")
    print("="*80 + "\n")

    settings = get_settings()
    generator = VLLMStreamingGenerator(settings)
    post_processor = ResponsePostProcessor(settings)

    # Check health
    is_healthy = await generator.health_check()
    if not is_healthy:
        print("❌ vLLM server not available")
        return

    # Create prompt with contexts
    prompt = """Bağlam bilgileri:

[1] Makine öğrenmesi, sistemlerin verilerden öğrenmesini sağlayan yapay zekanın bir alt kümesidir.

[2] Derin öğrenme, çok katmanlı sinir ağları kullanarak öğrenme yapan bir makine öğrenmesi yaklaşımıdır.

Soru: Makine öğrenmesi nedir?

YALNIZCA verilen bağlam bilgilerini kullanarak yanıtla. Her iddiayı [1], [2] gibi atıflarla destekle.

Yanıt:"""

    print("Streaming response:")
    print("-"*80 + "\n")

    # Stream generation
    tokens = []
    async for token in generator.generate_stream(prompt):
        tokens.append(token)
        print(token, end="", flush=True)

    generated_text = "".join(tokens)

    print("\n\n" + "-"*80)

    # Post-process
    processed = post_processor.process(
        generated_text=generated_text,
        available_citations=[1, 2],
        context_details=[
            {'citation_id': 1, 'source_collection': 'qa_pairs', 'relevance_score': 0.95},
            {'citation_id': 2, 'source_collection': 'plain_text', 'relevance_score': 0.88},
        ],
    )

    print("\nPost-processing results:")
    print(f"  Citations used: {processed.citations_used}")
    print(f"  Confidence score: {processed.confidence_score:.2f}")
    print(f"  Sources coverage: {processed.sources_coverage:.1%}")
    print(f"  Tokens generated: {processed.tokens_generated}")
    print(f"  Formatting issues: {processed.has_formatting_issues}")

    if processed.processing_notes:
        print(f"  Notes:")
        for note in processed.processing_notes:
            print(f"    - {note}")

    print("\n✓ Streaming + post-processing test complete!")


async def main():
    """Run all streaming tests."""
    print("="*80)
    print("STREAMING COMPONENT TESTING")
    print("="*80)

    # Test 1: Conversation memory
    test_conversation_memory()

    # Test 2: Context augmentation
    test_context_augmentation()

    # Test 3: Prompt building
    test_prompt_building()

    # Test 4: Streaming with post-processing
    await test_streaming_with_post_processing()

    print("\n" + "="*80)
    print("ALL STREAMING TESTS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
