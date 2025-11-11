"""
Configuration management using Pydantic Settings.
Loads configuration from environment variables and .env file.
"""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant server port")
    qdrant_api_key: str | None = Field(default=None, description="Qdrant API key")

    # Collection Names
    qa_collection_name: str = Field(default="qa_pairs", description="Q&A collection name")
    text_collection_name: str = Field(default="plain_text", description="Plain text collection name")

    # Model Paths
    llm_model_path: str = Field(
        default="Qwen/Qwen3-Next-80B-A3B-Instruct",
        description="LLM model path or HuggingFace ID"
    )
    embedding_model_path: str = Field(
        default="Qwen/Qwen3-Embedding-4B",
        description="Embedding model path or HuggingFace ID"
    )
    reranker_model_path: str = Field(
        default="Qwen/Qwen3-Reranker-4B",
        description="Reranker model path or HuggingFace ID"
    )

    # Model Configuration
    llm_max_new_tokens: int = Field(default=16384, description="Max tokens for LLM generation")
    llm_temperature: float = Field(default=0.7, description="LLM temperature")
    embedding_max_length: int = Field(default=8192, description="Max embedding sequence length")
    reranker_max_length: int = Field(default=8192, description="Max reranker sequence length")

    # Retrieval Configuration
    query_variants_count: int = Field(default=3, description="Number of query variants to generate")
    top_k_per_query: int = Field(default=15, description="Top K results per query variant")
    candidates_before_rerank: int = Field(default=30, description="Candidates before reranking")
    final_top_k: int = Field(default=7, description="Final top K results after reranking")
    reranker_threshold: float = Field(default=0.5, description="Minimum reranker score threshold")
    mmr_diversity_score: float = Field(default=0.3, description="MMR diversity weight (0-1)")

    # Hybrid Search Weights
    qa_dense_weight: float = Field(default=0.3, description="Q&A collection dense weight")
    qa_sparse_weight: float = Field(default=0.7, description="Q&A collection sparse weight")
    text_dense_weight: float = Field(default=0.7, description="Text collection dense weight")
    text_sparse_weight: float = Field(default=0.3, description="Text collection sparse weight")

    # RRF Configuration
    rrf_k: int = Field(default=60, description="RRF constant (default: 60)")

    # Device Configuration
    device: Literal["cuda", "cpu", "auto"] = Field(default="auto", description="Device for models")

    # vLLM Configuration
    vllm_server_url: str = Field(default="http://localhost:8000", description="vLLM server URL")
    vllm_model_name: str = Field(default="qwen3-next", description="vLLM model name")
    vllm_tensor_parallel_size: int = Field(default=4, description="Tensor parallel size for vLLM")
    vllm_max_model_len: int = Field(default=262144, description="Max model length (256K)")

    # Generation Settings
    generation_temperature: float = Field(default=0.7, description="Generation temperature")
    generation_top_p: float = Field(default=0.9, description="Top-p sampling")
    generation_max_tokens: int = Field(default=1024, description="Max tokens to generate")
    generation_min_tokens: int = Field(default=50, description="Min tokens to generate")
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")

    # Conversation History Settings
    max_conversation_tokens: int = Field(default=200000, description="Max conversation tokens")
    conversation_warning_threshold: float = Field(default=0.8, description="Warning threshold (0-1)")
    auto_summarize_old_messages: bool = Field(default=False, description="Auto summarize old messages")
    messages_to_keep_full: int = Field(default=20, description="Messages to keep at full detail")

    # Context Settings for Generation
    max_contexts_for_generation: int = Field(default=7, description="Max contexts for generation")
    max_context_tokens: int = Field(default=3000, description="Max tokens for all contexts")
    include_citations: bool = Field(default=True, description="Include citations in response")

    # Response Settings
    min_confidence_threshold: float = Field(default=0.6, description="Min confidence threshold")
    fallback_response: str = Field(
        default="Üzgünüm, bu konuda yeterli bilgim yok.",
        description="Fallback response when no context"
    )

    def get_qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
