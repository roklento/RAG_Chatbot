"""
Qwen3-Reranker-4B wrapper for document reranking.
Cross-encoder model for scoring query-document relevance.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from tqdm import tqdm
from ..config import Settings


@dataclass
class RankedDocument:
    """Container for a document with its reranking score."""
    content: str
    score: float
    original_rank: int
    metadata: Optional[dict] = None


class QwenReranker:
    """
    Wrapper for Qwen3-Reranker-4B cross-encoder model.

    Features:
    - Cross-encoder architecture for accurate relevance scoring
    - Support for custom instructions to improve task-specific performance
    - Batch processing for efficiency
    - Score normalization
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Reranker-4B",
        device: str = "auto",
        max_length: int = 8192,
        batch_size: int = 8,
        default_instruction: Optional[str] = None,
    ):
        """
        Initialize the Qwen reranker model.

        Args:
            model_path: HuggingFace model ID or local path
            device: Device to use ('cuda', 'cpu', or 'auto')
            max_length: Maximum sequence length
            batch_size: Batch size for reranking
            default_instruction: Default instruction for reranking task
        """
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size

        # Default instruction for chatbot context
        self.default_instruction = default_instruction or (
            "Kullanıcı sorusuna en uygun ve yardımcı olan bilgileri belirle"
        )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "auto" else "auto",
        )
        self.model.eval()  # Set to evaluation mode

        # Get special tokens for formatting
        self._setup_special_tokens()

    def _setup_special_tokens(self):
        """Setup prefix and suffix tokens for reranker input format."""
        # Based on Qwen3-Reranker documentation
        # The model expects format: <prefix_tokens><formatted_text><suffix_tokens>
        # We'll use the tokenizer to get these
        self.prefix_tokens = self.tokenizer.encode("", add_special_tokens=True)[:-1]
        self.suffix_tokens = self.tokenizer.encode("", add_special_tokens=True)[-1:]

    def format_instruction(
        self,
        query: str,
        document: str,
        instruction: Optional[str] = None
    ) -> str:
        """
        Format query-document pair with instruction.

        Args:
            query: User query
            document: Document to rank
            instruction: Custom instruction (uses default if None)

        Returns:
            Formatted string ready for tokenization
        """
        instruction = instruction or self.default_instruction
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    def _process_inputs(self, formatted_pairs: List[str]) -> dict:
        """
        Process and tokenize formatted query-document pairs.

        Args:
            formatted_pairs: List of formatted strings

        Returns:
            Tokenized inputs ready for model
        """
        # Tokenize without special tokens first
        inputs = self.tokenizer(
            formatted_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        # Add prefix and suffix tokens
        for i in range(len(inputs["input_ids"])):
            inputs["input_ids"][i] = (
                self.prefix_tokens + inputs["input_ids"][i] + self.suffix_tokens
            )

        # Pad the inputs
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        # Move to model device
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    @torch.no_grad()
    def _compute_scores(self, inputs: dict) -> torch.Tensor:
        """
        Compute relevance scores from model logits.

        Args:
            inputs: Tokenized inputs

        Returns:
            Relevance scores as tensor
        """
        outputs = self.model(**inputs)
        # Extract scores from last token logits
        scores = outputs.logits[:, -1, :]
        # Take the max logit as relevance score (based on Qwen3-Reranker pattern)
        scores = scores.max(dim=-1).values
        return scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> List[RankedDocument]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: User query
            documents: List of document strings to rerank
            instruction: Custom instruction for this reranking task
            top_k: Return only top K results (None = all)
            return_scores: Whether to include scores in results

        Returns:
            List of RankedDocument objects sorted by relevance
        """
        if not documents:
            return []

        # Format all query-document pairs
        formatted_pairs = [
            self.format_instruction(query, doc, instruction)
            for doc in documents
        ]

        all_scores = []

        # Process in batches
        for i in tqdm(
            range(0, len(formatted_pairs), self.batch_size),
            desc="Reranking",
            disable=len(formatted_pairs) < 50,
        ):
            batch = formatted_pairs[i:i + self.batch_size]
            inputs = self._process_inputs(batch)
            scores = self._compute_scores(inputs)
            all_scores.extend(scores.cpu().tolist())

        # Create ranked documents
        ranked_docs = [
            RankedDocument(
                content=doc,
                score=float(score),
                original_rank=idx,
            )
            for idx, (doc, score) in enumerate(zip(documents, all_scores))
        ]

        # Sort by score (descending)
        ranked_docs.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k filtering
        if top_k is not None:
            ranked_docs = ranked_docs[:top_k]

        return ranked_docs

    def rerank_with_threshold(
        self,
        query: str,
        documents: List[str],
        threshold: float = 0.5,
        instruction: Optional[str] = None,
    ) -> List[RankedDocument]:
        """
        Rerank documents and filter by score threshold.

        Args:
            query: User query
            documents: List of documents to rerank
            threshold: Minimum score threshold
            instruction: Custom instruction

        Returns:
            Filtered and sorted list of RankedDocument objects
        """
        ranked_docs = self.rerank(query, documents, instruction)
        return [doc for doc in ranked_docs if doc.score >= threshold]

    def score_pairs(
        self,
        query_doc_pairs: List[Tuple[str, str]],
        instruction: Optional[str] = None,
    ) -> List[float]:
        """
        Score multiple query-document pairs.

        Args:
            query_doc_pairs: List of (query, document) tuples
            instruction: Custom instruction

        Returns:
            List of relevance scores
        """
        formatted_pairs = [
            self.format_instruction(q, d, instruction)
            for q, d in query_doc_pairs
        ]

        all_scores = []
        for i in range(0, len(formatted_pairs), self.batch_size):
            batch = formatted_pairs[i:i + self.batch_size]
            inputs = self._process_inputs(batch)
            scores = self._compute_scores(inputs)
            all_scores.extend(scores.cpu().tolist())

        return all_scores


def create_reranker_model(settings: Settings) -> QwenReranker:
    """
    Factory function to create reranker model from settings.

    Args:
        settings: Application settings

    Returns:
        Initialized QwenReranker model
    """
    return QwenReranker(
        model_path=settings.reranker_model_path,
        device=settings.device,
        max_length=settings.reranker_max_length,
    )
