"""
Streaming RAG Pipeline - Complete orchestration of retrieval and generation.

This is the main entry point that ties together:
- Conversation history management
- Advanced retrieval
- Context augmentation
- Prompt building
- Streaming generation
- Response post-processing
"""

from typing import AsyncGenerator, Optional, List, Dict
from dataclasses import dataclass
import asyncio

from ..config import Settings
from ..retrieval.advanced_retriever import AdvancedRetriever
from .conversation_memory import ConversationMemoryManager, ConversationMessage
from .context_augmenter import ContextAugmenter, AugmentedContext
from .prompt_builder import PromptBuilder
from .vllm_generator import VLLMStreamingGenerator, GenerationConfig
from .post_processor import ResponsePostProcessor, ProcessedResponse


@dataclass
class RAGResponse:
    """Complete RAG response with all metadata."""

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


class StreamingRAGPipeline:
    """
    Complete RAG pipeline with streaming support.

    This orchestrates the entire RAG flow:
    1. Manage conversation history
    2. Retrieve relevant contexts
    3. Augment contexts with citations
    4. Build comprehensive prompt
    5. Generate streaming response
    6. Post-process and validate
    7. Update conversation history

    Features:
    - Token-by-token streaming
    - Conversation memory with token tracking
    - Automatic history truncation
    - Citation tracking
    - Confidence scoring
    """

    def __init__(
        self,
        settings: Settings,
        retriever: AdvancedRetriever,
        generator: VLLMStreamingGenerator,
    ):
        """
        Initialize RAG pipeline.

        Args:
            settings: Application settings
            retriever: Advanced retriever instance
            generator: vLLM streaming generator instance
        """
        self.settings = settings
        self.retriever = retriever
        self.generator = generator

        # Initialize components
        self.memory_manager = ConversationMemoryManager(settings)
        self.context_augmenter = ContextAugmenter(settings)
        self.prompt_builder = PromptBuilder(settings)
        self.post_processor = ResponsePostProcessor(settings)

    async def query_stream(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = 7,
        generation_config: Optional[GenerationConfig] = None,
        verbose: bool = False,
    ) -> AsyncGenerator[str, RAGResponse]:
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
            Complete RAGResponse with metadata (returned as final yield)

        Example:
            ```python
            pipeline = StreamingRAGPipeline(settings, retriever, generator)

            # Collect tokens and final response
            tokens = []
            async for token in pipeline.query_stream("Makine öğrenmesi nedir?"):
                if isinstance(token, str):
                    tokens.append(token)
                    print(token, end="", flush=True)
                else:
                    final_response = token  # This is RAGResponse
            ```
        """
        import time

        start_time = time.time()

        # 1. Check conversation history and token limit
        if self.memory_manager.should_reset(session_id):
            if verbose:
                print(f"⚠️  Session {session_id} approaching token limit - consider resetting")

        history = self.memory_manager.get_history(session_id)

        # 2. Retrieve relevant contexts
        retrieval_start = time.time()

        retrieved_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            verbose=verbose,
        )

        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        if verbose:
            print(f"\n✓ Retrieved {len(retrieved_results)} contexts in {retrieval_time_ms:.2f}ms")

        # 3. Augment contexts with citations
        augmented_contexts = self.context_augmenter.augment_contexts(
            retrieved_results,
            max_contexts=self.settings.max_contexts_for_generation,
        )

        if verbose:
            print(f"✓ Using {len(augmented_contexts)} augmented contexts")

        # 4. Build comprehensive prompt
        prompt = self.prompt_builder.build_prompt(
            query=query,
            augmented_contexts=augmented_contexts,
            conversation_history=history,
        )

        if verbose:
            prompt_tokens = self.memory_manager.count_tokens(prompt)
            print(f"✓ Built prompt with {prompt_tokens:,} tokens")

        # 5. Generate response with streaming
        generation_start = time.time()

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
            print(f"\n✓ Generated {len(generated_tokens)} tokens in {generation_time_ms:.2f}ms")

        # 6. Post-process response
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
            print(f"✓ Post-processed response (confidence: {processed_response.confidence_score:.2f})")
            if processed_response.processing_notes:
                for note in processed_response.processing_notes:
                    print(f"  - {note}")

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

        rag_response = RAGResponse(
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
        top_k: int = 7,
        generation_config: Optional[GenerationConfig] = None,
        verbose: bool = False,
    ) -> RAGResponse:
        """
        Process query without streaming (collect full response).

        Args:
            query: User query
            session_id: Session identifier
            top_k: Number of contexts to retrieve
            generation_config: Optional generation configuration
            verbose: Print debug information

        Returns:
            Complete RAGResponse
        """
        # Collect all tokens
        async for result in self.query_stream(
            query=query,
            session_id=session_id,
            top_k=top_k,
            generation_config=generation_config,
            verbose=verbose,
        ):
            if isinstance(result, RAGResponse):
                return result

        # Should never reach here
        raise RuntimeError("query_stream did not return RAGResponse")

    def reset_conversation(self, session_id: str = "default"):
        """
        Reset conversation history for a session.

        Args:
            session_id: Session to reset
        """
        self.memory_manager.reset_session(session_id)

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

    async def health_check(self) -> Dict[str, bool]:
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

        # Check retriever (via Qdrant)
        try:
            # Simple check - if retriever was initialized, it's healthy
            health['retriever'] = self.retriever is not None
        except Exception:
            health['retriever'] = False

        # Check generator (vLLM server)
        try:
            health['generator'] = await self.generator.health_check()
        except Exception:
            health['generator'] = False

        # Overall pipeline health
        health['pipeline'] = health['retriever'] and health['generator']

        return health


def create_rag_pipeline(
    settings: Settings,
    retriever: AdvancedRetriever,
    generator: VLLMStreamingGenerator,
) -> StreamingRAGPipeline:
    """
    Factory function to create RAG pipeline.

    Args:
        settings: Application settings
        retriever: Advanced retriever instance
        generator: vLLM generator instance

    Returns:
        StreamingRAGPipeline instance
    """
    return StreamingRAGPipeline(
        settings=settings,
        retriever=retriever,
        generator=generator,
    )
