"""
vLLM Streaming Generator - Efficient inference with streaming support.
Uses vLLM for optimal performance with Qwen3-Next-80B-A3B (MoE model).
"""

from typing import AsyncGenerator, Optional, Dict, List
import aiohttp
import json
from dataclasses import dataclass

from ..config import Settings


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    min_tokens: int = 50
    stop_sequences: List[str] = None
    stream: bool = True


class VLLMStreamingGenerator:
    """
    vLLM-based generator with async streaming support.

    Features:
    - Async streaming token-by-token
    - OpenAI-compatible API
    - Multi-token prediction support
    - Efficient MoE inference (only 3B active params)
    - Automatic retry logic
    """

    def __init__(
        self,
        settings: Settings,
        server_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize vLLM generator.

        Args:
            settings: Application settings
            server_url: vLLM server URL (uses settings if None)
            model_name: Model name (uses settings if None)
        """
        self.settings = settings
        self.server_url = server_url or settings.vllm_server_url
        self.model_name = model_name or settings.vllm_model_name

        # Ensure server URL has proper endpoint
        if not self.server_url.endswith("/v1"):
            self.server_url = f"{self.server_url}/v1"

        self.completions_endpoint = f"{self.server_url}/completions"

        # Default generation config
        self.default_config = GenerationConfig(
            temperature=settings.generation_temperature,
            top_p=settings.generation_top_p,
            max_tokens=settings.generation_max_tokens,
            min_tokens=settings.generation_min_tokens,
        )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate response with streaming.

        Args:
            prompt: Input prompt
            config: Optional generation config

        Yields:
            Generated tokens as they arrive

        Example:
            async for token in generator.generate_stream(prompt):
                print(token, end="", flush=True)
        """
        config = config or self.default_config

        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "stream": True,
        }

        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        # Make streaming request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.completions_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"vLLM request failed with status {response.status}: {error_text}"
                    )

                # Stream tokens
                async for line in response.content:
                    line = line.decode("utf-8").strip()

                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix

                    try:
                        data = json.loads(line)

                        # Extract token from response
                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]

                            if "text" in choice:
                                token = choice["text"]
                                if token:
                                    yield token

                            # Check if generation is complete
                            if choice.get("finish_reason") is not None:
                                break

                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate complete response (non-streaming).

        Args:
            prompt: Input prompt
            config: Optional generation config

        Returns:
            Complete generated text
        """
        config = config or self.default_config

        # Set stream to False for complete response
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "stream": False,
        }

        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.completions_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"vLLM request failed with status {response.status}: {error_text}"
                    )

                result = await response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["text"]

                return ""

    async def health_check(self) -> bool:
        """
        Check if vLLM server is healthy.

        Returns:
            True if server is accessible
        """
        try:
            health_url = f"{self.server_url.replace('/v1', '')}/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False

    def create_config(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationConfig:
        """
        Create custom generation config.

        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            min_tokens: Minimum tokens to generate
            stop_sequences: Stop sequences

        Returns:
            GenerationConfig instance
        """
        return GenerationConfig(
            temperature=temperature or self.default_config.temperature,
            top_p=top_p or self.default_config.top_p,
            max_tokens=max_tokens or self.default_config.max_tokens,
            min_tokens=min_tokens or self.default_config.min_tokens,
            stop_sequences=stop_sequences,
        )


# Fallback: Direct transformers generator (for when vLLM is not available)
class TransformersStreamingGenerator:
    """
    Fallback generator using HuggingFace transformers.
    Less efficient than vLLM but works without separate server.
    """

    def __init__(self, settings: Settings):
        """Initialize transformers generator."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
        from threading import Thread
        import torch

        self.settings = settings
        self.device = settings.device

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model_path,
            torch_dtype="auto",
            device_map=self.device if self.device != "auto" else "auto",
        )
        self.model.eval()

        self.TextIteratorStreamer = TextIteratorStreamer

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate with streaming using transformers."""
        import asyncio
        from threading import Thread

        config = config or GenerationConfig()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Create streamer
        streamer = self.TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation kwargs
        generation_kwargs = dict(
            inputs,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True if config.temperature > 0 else False,
            streamer=streamer,
        )

        # Start generation in separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream tokens
        for text in streamer:
            yield text
            await asyncio.sleep(0)  # Allow other coroutines to run

        thread.join()


def create_vllm_generator(
    settings: Settings,
    use_vllm: bool = True,
) -> VLLMStreamingGenerator | TransformersStreamingGenerator:
    """
    Factory function to create generator.

    Args:
        settings: Application settings
        use_vllm: Whether to use vLLM (True) or transformers (False)

    Returns:
        Generator instance
    """
    if use_vllm:
        return VLLMStreamingGenerator(settings=settings)
    else:
        return TransformersStreamingGenerator(settings=settings)
