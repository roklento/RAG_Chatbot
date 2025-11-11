"""
Response Post-Processor - Validates and formats generated responses.

Handles:
- Citation extraction and validation
- Response formatting (ensure pure text)
- Source usage tracking
- Confidence scoring
- Final output structuring
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import re
from ..config import Settings


@dataclass
class ProcessedResponse:
    """Final processed response with metadata."""

    text: str
    """Final response text (cleaned and validated)."""

    citations_used: List[int]
    """List of citation IDs actually used in response."""

    citation_details: List[Dict]
    """Details about each citation used."""

    confidence_score: float
    """Confidence score (0-1) based on citation usage and quality."""

    tokens_generated: int
    """Number of tokens in generated response."""

    has_formatting_issues: bool
    """True if markdown/HTML was detected and removed."""

    processing_notes: List[str] = field(default_factory=list)
    """Warnings or notes from processing."""

    sources_coverage: float = 0.0
    """Proportion of provided sources that were cited (0-1)."""


class ResponsePostProcessor:
    """
    Post-processes generated responses for quality and correctness.

    Features:
    - Extract citations from response text
    - Validate citations match provided contexts
    - Remove markdown/HTML formatting
    - Calculate confidence scores
    - Track source usage
    """

    # Patterns for detecting formatting issues
    MARKDOWN_PATTERNS = [
        r'#{1,6}\s',  # Headers
        r'\*\*[^*]+\*\*',  # Bold
        r'\*[^*]+\*',  # Italic
        r'`[^`]+`',  # Code inline
        r'```[^`]*```',  # Code blocks
        r'\[[^\]]+\]\([^)]+\)',  # Links
        r'^\s*[-*+]\s',  # Lists
        r'^\s*\d+\.\s',  # Numbered lists
        r'\|[^|]+\|',  # Tables
    ]

    # Patterns for uncertainty/admission
    UNCERTAINTY_PHRASES_TR = [
        r'bilmiyorum',
        r'emin değilim',
        r'kesin değil',
        r'bilgi yok',
        r'bulamadım',
        r'verilen bilgiler?.{0,20}yok',
        r'sağlanan.{0,20}yeterli değil',
    ]

    def __init__(self, settings: Settings):
        """
        Initialize post-processor.

        Args:
            settings: Application settings
        """
        self.settings = settings

        # Compile regex patterns for efficiency
        self._markdown_regex = re.compile(
            '|'.join(self.MARKDOWN_PATTERNS),
            re.MULTILINE
        )
        self._citation_regex = re.compile(r'\[(\d+)\]')
        self._uncertainty_regex = re.compile(
            '|'.join(self.UNCERTAINTY_PHRASES_TR),
            re.IGNORECASE
        )

    def process(
        self,
        generated_text: str,
        available_citations: List[int],
        context_details: Optional[List[Dict]] = None,
    ) -> ProcessedResponse:
        """
        Process and validate generated response.

        Args:
            generated_text: Raw generated text from LLM
            available_citations: List of citation IDs that were provided
            context_details: Optional details about each context

        Returns:
            ProcessedResponse with validated and cleaned text
        """
        processing_notes = []

        # 1. Extract citations from response
        citations_used = self._extract_citations(generated_text)

        # 2. Validate citations
        invalid_citations = [c for c in citations_used if c not in available_citations]
        if invalid_citations:
            processing_notes.append(
                f"Invalid citations detected: {invalid_citations}"
            )
            # Remove invalid citations from text
            for invalid_id in invalid_citations:
                generated_text = generated_text.replace(f'[{invalid_id}]', '')
            citations_used = [c for c in citations_used if c in available_citations]

        # 3. Check for formatting issues
        has_formatting_issues = bool(self._markdown_regex.search(generated_text))
        if has_formatting_issues:
            processing_notes.append("Markdown/HTML formatting detected and removed")
            generated_text = self._remove_formatting(generated_text)

        # 4. Calculate confidence score
        confidence_score = self._calculate_confidence(
            generated_text,
            citations_used,
            available_citations
        )

        # 5. Calculate sources coverage
        sources_coverage = (
            len(citations_used) / len(available_citations)
            if available_citations else 0.0
        )

        # 6. Build citation details
        citation_details = []
        if context_details:
            for citation_id in citations_used:
                # Find matching context
                matching_context = next(
                    (ctx for ctx in context_details if ctx.get('citation_id') == citation_id),
                    None
                )
                if matching_context:
                    citation_details.append({
                        'citation_id': citation_id,
                        'collection': matching_context.get('source_collection', 'unknown'),
                        'relevance_score': matching_context.get('relevance_score', 0.0),
                    })

        # 7. Clean up text
        cleaned_text = self._clean_text(generated_text)

        # 8. Count tokens (approximate - by whitespace)
        tokens_generated = len(cleaned_text.split())

        return ProcessedResponse(
            text=cleaned_text,
            citations_used=sorted(citations_used),
            citation_details=citation_details,
            confidence_score=confidence_score,
            tokens_generated=tokens_generated,
            has_formatting_issues=has_formatting_issues,
            processing_notes=processing_notes,
            sources_coverage=sources_coverage,
        )

    def _extract_citations(self, text: str) -> List[int]:
        """
        Extract citation IDs from text.

        Args:
            text: Response text

        Returns:
            List of citation IDs (sorted, unique)
        """
        matches = self._citation_regex.findall(text)
        citation_ids = [int(m) for m in matches]
        return sorted(set(citation_ids))

    def _remove_formatting(self, text: str) -> str:
        """
        Remove markdown and HTML formatting.

        Args:
            text: Text with potential formatting

        Returns:
            Plain text
        """
        # Remove markdown bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Remove code formatting
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'```[^`]*```', '', text)

        # Remove headers
        text = re.sub(r'#{1,6}\s+(.+)', r'\1', text)

        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

        return text

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)

        # Remove multiple newlines (max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _calculate_confidence(
        self,
        text: str,
        citations_used: List[int],
        available_citations: List[int],
    ) -> float:
        """
        Calculate confidence score for response.

        Factors:
        - Number of citations used (more = higher confidence)
        - Proportion of available sources cited
        - Presence of uncertainty phrases (lowers confidence)
        - Response length (very short = lower confidence)

        Args:
            text: Response text
            citations_used: Citations actually used
            available_citations: Citations that were available

        Returns:
            Confidence score (0-1)
        """
        score = 0.0

        # Factor 1: Citation usage (0-0.5 points)
        if available_citations:
            citation_ratio = len(citations_used) / len(available_citations)
            score += citation_ratio * 0.5

        # Factor 2: Absolute citation count (0-0.3 points)
        if len(citations_used) >= 3:
            score += 0.3
        elif len(citations_used) == 2:
            score += 0.2
        elif len(citations_used) == 1:
            score += 0.1

        # Factor 3: Response length (0-0.1 points)
        word_count = len(text.split())
        if word_count >= 50:
            score += 0.1
        elif word_count >= 20:
            score += 0.05

        # Factor 4: Uncertainty detection (-0.2 points)
        if self._uncertainty_regex.search(text):
            score -= 0.2

        # Factor 5: No citations at all (-0.3 points)
        if not citations_used and available_citations:
            score -= 0.3

        # Clamp to 0-1 range
        return max(0.0, min(1.0, score))

    def validate_response_quality(
        self,
        response: ProcessedResponse,
        min_confidence: float = 0.3,
    ) -> tuple[bool, List[str]]:
        """
        Validate response quality.

        Args:
            response: Processed response
            min_confidence: Minimum acceptable confidence score

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []

        # Check confidence
        if response.confidence_score < min_confidence:
            issues.append(
                f"Low confidence score: {response.confidence_score:.2f} < {min_confidence}"
            )

        # Check if response is too short
        if response.tokens_generated < 10:
            issues.append("Response too short")

        # Check if no citations when sources available
        if not response.citations_used and response.citation_details:
            issues.append("No citations used despite available sources")

        # Check for formatting issues
        if response.has_formatting_issues:
            issues.append("Formatting issues detected")

        is_valid = len(issues) == 0
        return is_valid, issues

    def format_for_display(self, response: ProcessedResponse) -> str:
        """
        Format response for display to user.

        Args:
            response: Processed response

        Returns:
            Formatted text ready for display
        """
        return response.text

    def get_metrics(self, response: ProcessedResponse) -> Dict:
        """
        Extract metrics from processed response.

        Args:
            response: Processed response

        Returns:
            Dictionary of metrics
        """
        return {
            'confidence_score': response.confidence_score,
            'citations_used_count': len(response.citations_used),
            'sources_coverage': response.sources_coverage,
            'tokens_generated': response.tokens_generated,
            'has_formatting_issues': response.has_formatting_issues,
            'processing_notes_count': len(response.processing_notes),
        }
