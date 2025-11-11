"""
Text Chunking Module - Splits long texts into meaningful chunks.

Supports:
- Character-based chunking with overlap
- Sentence-aware chunking (Turkish language)
- Paragraph-based chunking
"""

from typing import List, Dict, Optional
import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""

    content: str
    """Chunk content."""

    chunk_index: int
    """Index of this chunk in the document."""

    total_chunks: int
    """Total number of chunks in the document."""

    source_title: str
    """Title of source document."""

    char_start: int
    """Character start position in original text."""

    char_end: int
    """Character end position in original text."""

    metadata: Dict = None
    """Additional metadata."""


class TextChunker:
    """
    Text chunker with multiple strategies.

    Supports character-based and sentence-aware chunking.
    Optimized for Turkish language.
    """

    # Turkish sentence endings
    SENTENCE_ENDINGS = r'[.!?]\s+'

    # Paragraph separator
    PARAGRAPH_SEP = r'\n\n+'

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target size of each chunk (characters)
            chunk_overlap: Number of overlapping characters between chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(
        self,
        text: str,
        title: str = "",
        strategy: str = "sentence_aware",
        metadata: Optional[Dict] = None,
    ) -> List[TextChunk]:
        """
        Chunk text using specified strategy.

        Args:
            text: Text to chunk
            title: Title of the document
            strategy: Chunking strategy ("character", "sentence_aware", "paragraph")
            metadata: Optional metadata to attach to chunks

        Returns:
            List of TextChunk objects
        """
        if strategy == "character":
            return self._chunk_by_characters(text, title, metadata)
        elif strategy == "sentence_aware":
            return self._chunk_sentence_aware(text, title, metadata)
        elif strategy == "paragraph":
            return self._chunk_by_paragraphs(text, title, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _chunk_by_characters(
        self,
        text: str,
        title: str,
        metadata: Optional[Dict],
    ) -> List[TextChunk]:
        """Simple character-based chunking with overlap."""
        chunks = []
        text_len = len(text)

        start = 0
        chunk_index = 0

        while start < text_len:
            # Calculate end position
            end = min(start + self.chunk_size, text_len)

            # Extract chunk
            chunk_text = text[start:end].strip()

            # Only keep if above minimum size
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    source_title=title,
                    char_start=start,
                    char_end=end,
                    metadata=metadata or {},
                ))
                chunk_index += 1

            # Move start position (with overlap)
            start += self.chunk_size - self.chunk_overlap

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_sentence_aware(
        self,
        text: str,
        title: str,
        metadata: Optional[Dict],
    ) -> List[TextChunk]:
        """
        Chunk text while respecting sentence boundaries.

        This ensures chunks don't break in the middle of sentences.
        """
        # Split into sentences
        sentences = re.split(self.SENTENCE_ENDINGS, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0
        char_start = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)

                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        content=chunk_text,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        source_title=title,
                        char_start=char_start,
                        char_end=char_start + len(chunk_text),
                        metadata=metadata or {},
                    ))
                    chunk_index += 1

                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                char_start += len(chunk_text) - overlap_length
                current_chunk = overlap_sentences
                current_length = overlap_length

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    source_title=title,
                    char_start=char_start,
                    char_end=char_start + len(chunk_text),
                    metadata=metadata or {},
                ))

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_by_paragraphs(
        self,
        text: str,
        title: str,
        metadata: Optional[Dict],
    ) -> List[TextChunk]:
        """
        Chunk text by paragraphs.

        Each paragraph becomes a chunk (or multiple if too long).
        """
        # Split into paragraphs
        paragraphs = re.split(self.PARAGRAPH_SEP, text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        char_pos = 0
        chunk_index = 0

        for paragraph in paragraphs:
            para_length = len(paragraph)

            # If paragraph is within chunk size, keep it as one chunk
            if para_length <= self.chunk_size:
                if para_length >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        content=paragraph,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        source_title=title,
                        char_start=char_pos,
                        char_end=char_pos + para_length,
                        metadata=metadata or {},
                    ))
                    chunk_index += 1

                char_pos += para_length
            else:
                # Paragraph too long - split with sentence-aware chunking
                para_chunks = self._chunk_sentence_aware(
                    paragraph,
                    title,
                    metadata,
                )

                for chunk in para_chunks:
                    chunk.chunk_index = chunk_index
                    chunk.char_start += char_pos
                    chunk.char_end += char_pos
                    chunks.append(chunk)
                    chunk_index += 1

                char_pos += para_length

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def chunk_documents(
        self,
        documents: List[Dict],
        title_key: str = "title",
        content_key: str = "content",
        strategy: str = "sentence_aware",
    ) -> List[TextChunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dicts with title and content
            title_key: Key for document title
            content_key: Key for document content
            strategy: Chunking strategy to use

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            title = doc.get(title_key, "Untitled")
            content = doc.get(content_key, "")

            if not content:
                continue

            # Create metadata from document (excluding content)
            metadata = {k: v for k, v in doc.items() if k != content_key}

            # Chunk document
            chunks = self.chunk_text(
                text=content,
                title=title,
                strategy=strategy,
                metadata=metadata,
            )

            all_chunks.extend(chunks)

        return all_chunks

    def get_chunk_summary(self, chunks: List[TextChunk]) -> Dict:
        """
        Get summary statistics about chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
                'total_characters': 0,
            }

        lengths = [len(chunk.content) for chunk in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_characters': sum(lengths),
            'unique_sources': len(set(chunk.source_title for chunk in chunks)),
        }


def create_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
) -> TextChunker:
    """
    Factory function to create text chunker.

    Args:
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size to keep

    Returns:
        TextChunker instance
    """
    return TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )
