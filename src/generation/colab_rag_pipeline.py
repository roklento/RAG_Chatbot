"""
Colab RAG Pipeline - Optimized for Google Colab environment.

Uses Gemma-3n-E4B-it instead of vLLM.
Simplified and memory-efficient.
"""

from typing import AsyncGenerator, Optional, List, Dict
from dataclasses import dataclass
import asyncio
import time

from ..config import Settings
from ..retrieval.advanced_retriever import AdvancedRetriever
from .conversation_memory import ConversationMemoryManager, ConversationMessage
from .context_augmenter import ContextAugmenter, AugmentedContext
from .prompt_builder import PromptBuilder
from .gemma_generator import GemmaStreamingGenerator, GemmaGenerationConfig
from .post_processor import ResponsePostProcessor, ProcessedResponse


@dataclass
class ColabRAGResponse:
    """Complete RAG response for Colab."""

    processed_response: ProcessedResponse
    """Final processed response."""

    retrieved_contexts: List[Dict]
    """Retrieved contexts from database."""

    augmented_contexts: List[AugmentedContext]
    """Augmented contexts used in generation."""

    conversation_tokens: int
    """Total tokens in conversation history."""

    retrieval_time_ms: float
    """Time spent on retrieval (milliseconds)."""

    generation_time_ms: float
    """Time spent on generation (milliseconds)."""

    total_time_ms: float
    """Total pipeline time (milliseconds)."""


class ColabRAGPipeline:
    """
    RAG pipeline optimized for Google Colab.

    Differences from production pipeline:
    - Uses Gemma-3n instead of vLLM
    - Memory-optimized settings
    - Simpler error handling
    - Colab-friendly progress output
    """

    def __init__(
        self,
        settings: Settings,
        retriever: AdvancedRetriever,
        generator: GemmaStreamingGenerator,
    ):
        """
        Initialize Colab RAG pipeline.

        Args:
            settings: Application settings
            retriever: Advanced retriever instance
            generator: Gemma streaming generator instance
        """
        self.settings = settings
        self.retriever = retriever
        self.generator = generator

        # Initialize components
        self.memory_manager = ConversationMemoryManager(settings)
        self.context_augmenter = ContextAugmenter(settings)
        self.prompt_builder = PromptBuilder(settings)
        self.post_processor = ResponsePostProcessor(settings)

        print("✓ Colab RAG Pipeline initialized")

    async def query_stream(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = 5,  # Reduced for Colab
        generation_config: Optional[GemmaGenerationConfig] = None,
        verbose: bool = True,
    ) -> AsyncGenerator[str, ColabRAGResponse]:
        """
        Process query with streaming response.

        Args:
            query: User query
            session_id: Session identifier for conversation tracking
            top_k: Number of contexts to retrieve
            generation_config: Optional generation configuration
            verbose: Print debug information

        Yields:
            Response tokens as they're generated

        Returns:
            Complete ColabRAGResponse with metadata (returned as final yield)
        """
        start_time = time.time()

        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing Query: {query}")
            print(f"{'='*80}\n")

        # 1. Check conversation history
        if self.memory_manager.should_reset(session_id):
            if verbose:
                print(f"⚠️  Session {session_id} approaching token limit")

        history = self.memory_manager.get_history(session_id)

        # 2. Retrieve relevant contexts
        retrieval_start = time.time()

        if verbose:
            print("🔍 Retrieving contexts...")

        retrieved_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            verbose=False,  # Suppress retriever's verbose output
        )

        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        if verbose:
            print(f"   ✓ Retrieved {len(retrieved_results)} contexts ({retrieval_time_ms:.0f}ms)")

        # 3. Augment contexts with citations
        max_contexts = min(top_k, getattr(self.settings, 'max_contexts_for_generation', 5))
        augmented_contexts = self.context_augmenter.augment_contexts(
            retrieved_results,
            max_contexts=max_contexts,
        )

        if verbose:
            print(f"   ✓ Using {len(augmented_contexts)} augmented contexts")

        # 4. Build comprehensive prompt
        prompt = self.prompt_builder.build_prompt(
            query=query,
            augmented_contexts=augmented_contexts,
            conversation_history=history,
        )

        if verbose:
            prompt_tokens = self.memory_manager.count_tokens(prompt)
            print(f"   ✓ Built prompt ({prompt_tokens:,} tokens)")

        # 5. Generate response with streaming
        generation_start = time.time()

        if verbose:
            print(f"\n🤖 Generating response...")
            print("-" * 80)

        generated_tokens = []

        async for token in self.generator.generate_stream(
            prompt=prompt,
            config=generation_config,
        ):
            generated_tokens.append(token)
            yield token  # Stream to caller

        generation_time_ms = (time.time() - generation_start) * 1000

        # Combine all tokens
        generated_text = "".join(generated_tokens)

        if verbose:
            print()  # New line after streaming
            print("-" * 80)
            print(f"   ✓ Generated {len(generated_tokens)} tokens ({generation_time_ms:.0f}ms)")

        # 6. Post-process response
        if verbose:
            print(f"\n📊 Post-processing...")

        available_citations = [ctx.citation_id for ctx in augmented_contexts]
        context_details = [
            {
                'citation_id': ctx.citation_id,
                'source_collection': ctx.source_collection,
                'relevance_score': ctx.relevance_score,
            }
            for ctx in augmented_contexts
        ]

        processed_response = self.post_processor.process(
            generated_text=generated_text,
            available_citations=available_citations,
            context_details=context_details,
        )

        if verbose:
            print(f"   ✓ Confidence: {processed_response.confidence_score:.2f}")
            print(f"   ✓ Citations used: {processed_response.citations_used}")
            if processed_response.processing_notes:
                for note in processed_response.processing_notes:
                    print(f"   ⚠️  {note}")

        # 7. Update conversation history
        self.memory_manager.add_message(
            session_id=session_id,
            role="user",
            content=query,
        )

        self.memory_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=processed_response.text,
        )

        # 8. Build final response object
        total_time_ms = (time.time() - start_time) * 1000
        conversation_tokens = self.memory_manager.get_total_tokens(session_id)

        if verbose:
            print(f"\n⏱️  Total time: {total_time_ms:.0f}ms")
            print(f"💬 Conversation tokens: {conversation_tokens:,}")
            print(f"{'='*80}\n")

        rag_response = ColabRAGResponse(
            processed_response=processed_response,
            retrieved_contexts=[
                {
                    'content': r.content,
                    'collection': r.collection,
                    'score': r.score,
                }
                for r in retrieved_results
            ],
            augmented_contexts=augmented_contexts,
            conversation_tokens=conversation_tokens,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
        )

        # Return final response metadata (as last yield)
        return rag_response

    async def query(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = 5,
        generation_config: Optional[GemmaGenerationConfig] = None,
        verbose: bool = True,
    ) -> ColabRAGResponse:
        """
        Process query without streaming (collect full response).

        Args:
            query: User query
            session_id: Session identifier
            top_k: Number of contexts to retrieve
            generation_config: Optional generation configuration
            verbose: Print debug information

        Returns:
            Complete ColabRAGResponse
        """
        # Collect all tokens
        async for result in self.query_stream(
            query=query,
            session_id=session_id,
            top_k=top_k,
            generation_config=generation_config,
            verbose=verbose,
        ):
            if isinstance(result, ColabRAGResponse):
                return result

        # Should never reach here
        raise RuntimeError("query_stream did not return ColabRAGResponse")

    def reset_conversation(self, session_id: str = "default"):
        """
        Reset conversation history for a session.

        Args:
            session_id: Session to reset
        """
        self.memory_manager.reset_session(session_id)
        print(f"✓ Session '{session_id}' reset")

    def get_conversation_stats(self, session_id: str = "default") -> Dict:
        """
        Get statistics about conversation.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with conversation stats
        """
        total_tokens = self.memory_manager.get_total_tokens(session_id)
        message_count = len(self.memory_manager.get_history(session_id))
        should_reset = self.memory_manager.should_reset(session_id)

        return {
            'session_id': session_id,
            'total_tokens': total_tokens,
            'message_count': message_count,
            'max_tokens': self.settings.max_conversation_tokens,
            'token_usage_percent': (total_tokens / self.settings.max_conversation_tokens) * 100,
            'should_reset': should_reset,
        }

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all pipeline components.

        Returns:
            Dictionary with health status of each component
        """
        health = {
            'retriever': False,
            'generator': False,
            'pipeline': True,
        }

        # Check retriever
        try:
            health['retriever'] = self.retriever is not None
        except Exception:
            health['retriever'] = False

        # Check generator
        try:
            health['generator'] = self.generator.health_check()
        except Exception:
            health['generator'] = False

        # Overall pipeline health
        health['pipeline'] = health['retriever'] and health['generator']

        return health


def create_colab_rag_pipeline(
    settings: Settings,
    retriever: AdvancedRetriever,
    generator: GemmaStreamingGenerator,
) -> ColabRAGPipeline:
    """
    Factory function to create Colab RAG pipeline.

    Args:
        settings: Application settings
        retriever: Advanced retriever instance
        generator: Gemma generator instance

    Returns:
        ColabRAGPipeline instance
    """
    return ColabRAGPipeline(
        settings=settings,
        retriever=retriever,
        generator=generator,
    )
