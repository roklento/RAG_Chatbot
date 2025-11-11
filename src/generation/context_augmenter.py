"""
Context Augmenter - Prepares retrieved contexts for generation.
Handles citation tracking, formatting, and token management.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import tiktoken

from ..config import Settings
from ..retrieval.advanced_retriever import FinalResult


@dataclass
class AugmentedContext:
    """
    Augmented context with citation tracking.

    Attributes:
        citation_id: Unique citation number [1], [2], etc.
        content: Document content
        source_collection: Origin collection (qa_pairs or plain_text)
        relevance_score: Reranker score
        retrieval_score: Original retrieval score
        metadata: Additional metadata
        token_count: Number of tokens in content
    """
    citation_id: int
    content: str
    source_collection: str
    relevance_score: float
    retrieval_score: float
    metadata: Optional[Dict] = None
    token_count: int = 0


class ContextAugmenter:
    """
    Prepares retrieved contexts for generation.

    Features:
    - Citation ID assignment
    - Context formatting
    - Source tracking
    - Token counting and management
    - Context truncation if needed
    """

    def __init__(
        self,
        settings: Settings,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize context augmenter.

        Args:
            settings: Application settings
            encoding_name: Tiktoken encoding
        """
        self.settings = settings
        self.max_contexts = settings.max_contexts_for_generation
        self.max_context_tokens = settings.max_context_tokens

        # Initialize tokenizer
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        return len(self.encoder.encode(text))

    def augment_contexts(
        self,
        retrieval_results: List[FinalResult],
        max_contexts: Optional[int] = None,
    ) -> List[AugmentedContext]:
        """
        Augment retrieved contexts with citations and metadata.

        Args:
            retrieval_results: Results from retrieval pipeline
            max_contexts: Maximum contexts to include

        Returns:
            List of augmented contexts
        """
        max_contexts = max_contexts or self.max_contexts

        # Limit to top N contexts
        results = retrieval_results[:max_contexts]

        # Create augmented contexts
        augmented = []
        total_tokens = 0

        for idx, result in enumerate(results, start=1):
            token_count = self.count_tokens(result.content)

            # Check if adding this context exceeds token limit
            if total_tokens + token_count > self.max_context_tokens:
                # Try to truncate content to fit
                truncated = self._truncate_content(
                    result.content,
                    self.max_context_tokens - total_tokens
                )
                if truncated:
                    token_count = self.count_tokens(truncated)
                    content = truncated
                else:
                    # Can't fit, stop here
                    break
            else:
                content = result.content

            augmented_ctx = AugmentedContext(
                citation_id=idx,
                content=content,
                source_collection=result.collection,
                relevance_score=result.relevance_score,
                retrieval_score=result.retrieval_score,
                metadata=result.metadata,
                token_count=token_count,
            )

            augmented.append(augmented_ctx)
            total_tokens += token_count

        return augmented

    def format_for_prompt(
        self,
        augmented_contexts: List[AugmentedContext],
        include_metadata: bool = False,
    ) -> str:
        """
        Format augmented contexts for prompt insertion.

        Args:
            augmented_contexts: Augmented contexts
            include_metadata: Whether to include source metadata

        Returns:
            Formatted context string

        Example output:
            [1] Makine öğrenmesi, sistemlerin deneyimlerden öğrenmesini sağlar...
            [2] Denetimli öğrenme etiketlenmiş veri kullanır...
        """
        if not augmented_contexts:
            return "Bağlam bilgisi bulunamadı."

        formatted_parts = []

        for ctx in augmented_contexts:
            # Format citation
            citation = f"[{ctx.citation_id}]"

            # Add content
            if include_metadata and ctx.metadata:
                # Include source info for transparency
                source_type = "Soru-Cevap" if "qa_pair" in ctx.source_collection else "Döküman"
                formatted_parts.append(
                    f"{citation} ({source_type}) {ctx.content}"
                )
            else:
                formatted_parts.append(f"{citation} {ctx.content}")

        return "\n\n".join(formatted_parts)

    def get_source_mapping(
        self,
        augmented_contexts: List[AugmentedContext],
    ) -> Dict[int, Dict]:
        """
        Get mapping of citation IDs to source information.

        Args:
            augmented_contexts: Augmented contexts

        Returns:
            Dictionary mapping citation_id -> source info
        """
        mapping = {}

        for ctx in augmented_contexts:
            mapping[ctx.citation_id] = {
                "content": ctx.content,
                "collection": ctx.source_collection,
                "relevance_score": ctx.relevance_score,
                "retrieval_score": ctx.retrieval_score,
                "metadata": ctx.metadata,
            }

        return mapping

    def get_token_stats(
        self,
        augmented_contexts: List[AugmentedContext],
    ) -> Dict:
        """
        Get token statistics for contexts.

        Args:
            augmented_contexts: Augmented contexts

        Returns:
            Statistics dictionary
        """
        total_tokens = sum(ctx.token_count for ctx in augmented_contexts)
        avg_tokens = total_tokens / len(augmented_contexts) if augmented_contexts else 0

        return {
            "total_contexts": len(augmented_contexts),
            "total_tokens": total_tokens,
            "average_tokens_per_context": avg_tokens,
            "max_allowed_tokens": self.max_context_tokens,
            "usage_percentage": (total_tokens / self.max_context_tokens) * 100,
        }

    def _truncate_content(self, content: str, max_tokens: int) -> Optional[str]:
        """
        Truncate content to fit within token limit.

        Args:
            content: Content to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated content or None if can't fit meaningfully
        """
        if max_tokens < 50:  # Too small to be meaningful
            return None

        # Encode and truncate
        tokens = self.encoder.encode(content)

        if len(tokens) <= max_tokens:
            return content

        # Truncate tokens
        truncated_tokens = tokens[:max_tokens - 3]  # Leave room for "..."

        # Decode back
        truncated_text = self.encoder.decode(truncated_tokens)

        # Add ellipsis
        return truncated_text + "..."

    def extract_citations_from_response(self, response: str) -> List[int]:
        """
        Extract citation numbers from generated response.

        Args:
            response: Generated response text

        Returns:
            List of citation IDs used
        """
        import re

        # Find all [N] patterns
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, response)

        # Convert to integers and remove duplicates
        citations = list(set(int(m) for m in matches))
        citations.sort()

        return citations


def create_context_augmenter(settings: Settings) -> ContextAugmenter:
    """
    Factory function to create context augmenter.

    Args:
        settings: Application settings

    Returns:
        Initialized ContextAugmenter
    """
    return ContextAugmenter(settings=settings)
