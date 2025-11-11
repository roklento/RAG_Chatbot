"""
Qwen3-Embedding-4B wrapper for text embeddings.
Uses sentence-transformers for easy integration.
"""

from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from ..config import Settings


class QwenEmbedding:
    """
    Wrapper for Qwen3-Embedding-4B model using sentence-transformers.

    Features:
    - Supports Matryoshka Representation Learning (flexible dimensions)
    - L2 normalized embeddings
    - Batch processing
    - Multi-lingual support (100+ languages)
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Embedding-4B",
        device: str = "auto",
        max_length: int = 8192,
        batch_size: int = 32,
    ):
        """
        Initialize the Qwen embedding model.

        Args:
            model_path: HuggingFace model ID or local path
            device: Device to use ('cuda', 'cpu', or 'auto')
            max_length: Maximum sequence length
            batch_size: Batch size for encoding
        """
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        # Load model using sentence-transformers
        # For better performance, we can enable flash_attention_2
        model_kwargs = {
            "device_map": device if device != "auto" else "auto",
        }

        # Try to use flash attention if available
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass  # Flash attention not available, use default

        self.model = SentenceTransformer(
            model_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
        )

        # Set max sequence length
        self.model.max_seq_length = max_length

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2 normalization
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query.

        Args:
            text: Query text string

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0]

    def embed_batch(
        self,
        texts: List[str],
        is_query: bool = False
    ) -> np.ndarray:
        """
        Embed a batch of texts (unified interface).

        Args:
            texts: List of text strings
            is_query: Whether these are queries (unused for Qwen, kept for compatibility)

        Returns:
            Numpy array of embeddings
        """
        return self.embed_documents(texts)

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()

    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Convenience method to embed text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            Embeddings as numpy array
        """
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_documents(texts)


def create_embedding_model(settings: Settings) -> QwenEmbedding:
    """
    Factory function to create embedding model from settings.

    Args:
        settings: Application settings

    Returns:
        Initialized QwenEmbedding model
    """
    return QwenEmbedding(
        model_path=settings.embedding_model_path,
        device=settings.device,
        max_length=settings.embedding_max_length,
    )
