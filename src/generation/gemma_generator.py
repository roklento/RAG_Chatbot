"""
Gemma-3n Generator - Text generation using Google Gemma-3n-E4B-it model.

Optimized for Google Colab environment.
Supports streaming and non-streaming generation.
"""

from typing import AsyncGenerator, Optional, List, Dict
from dataclasses import dataclass
import asyncio
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

from ..config import Settings


@dataclass
class GemmaGenerationConfig:
    """Generation configuration for Gemma model."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    min_tokens: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1


class GemmaStreamingGenerator:
    """
    Gemma-3n-E4B-it model generator with streaming support.

    Features:
    - Text-to-text generation (multimodal model used for text only)
    - Streaming token-by-token generation
    - Turkish language support
    - Memory-efficient loading
    - Colab-optimized
    """

    def __init__(
        self,
        settings: Settings,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """
        Initialize Gemma generator.

        Args:
            settings: Application settings
            model_path: Path to Gemma model (overrides settings)
            device: Device to use (overrides settings)
            dtype: Data type for model (bfloat16, float16, float32)
        """
        self.settings = settings
        self.model_path = model_path or settings.generation_model_path or "google/gemma-3n-E4B-it"
        self.device = device or settings.generation_device or "cuda"

        # Determine dtype
        dtype_str = dtype or getattr(settings, 'generation_dtype', 'bfloat16')
        self.dtype = self._get_torch_dtype(dtype_str)

        self.model = None
        self.processor = None

        # Load model
        self._load_model()

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
            'auto': torch.bfloat16,  # Default to bfloat16
        }
        return dtype_map.get(dtype_str.lower(), torch.bfloat16)

    def _load_model(self):
        """Load Gemma model and processor."""
        print(f"Loading Gemma model from: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")

        try:
            # Load processor (handles tokenization and chat template)
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            # Load model
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto",  # Automatic device placement
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).eval()

            print(f"✓ Model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def create_config(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 512,
        **kwargs,
    ) -> GemmaGenerationConfig:
        """Create generation configuration."""
        return GemmaGenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        """
        Build Gemma chat messages format.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            List of message dicts in Gemma format
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        # Add user prompt
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        })

        return messages

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GemmaGenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming (token-by-token).

        Args:
            prompt: Input prompt
            config: Generation configuration
            system_prompt: Optional system prompt

        Yields:
            Generated tokens as they arrive
        """
        if config is None:
            config = self.create_config()

        # Build messages
        messages = self._build_messages(prompt, system_prompt)

        # Prepare inputs using processor
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": config.max_tokens,
            "min_new_tokens": config.min_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "do_sample": config.do_sample,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        # Streaming generation
        with torch.inference_mode():
            input_length = inputs["input_ids"].shape[1]

            # Generate token by token
            for _ in range(config.max_tokens):
                # Generate next token
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    max_new_tokens=1,  # Generate one token at a time
                )

                # Decode new token
                new_token_id = outputs[0, input_length:]
                new_token = self.processor.decode(new_token_id, skip_special_tokens=True)

                # Check for EOS
                if outputs[0, -1].item() == self.processor.tokenizer.eos_token_id:
                    break

                # Yield token
                if new_token:
                    yield new_token
                    await asyncio.sleep(0)  # Allow other tasks to run

                # Update inputs for next iteration
                inputs = {"input_ids": outputs}
                input_length = outputs.shape[1]

    async def generate(
        self,
        prompt: str,
        config: Optional[GemmaGenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text without streaming (complete response).

        Args:
            prompt: Input prompt
            config: Generation configuration
            system_prompt: Optional system prompt

        Returns:
            Complete generated text
        """
        if config is None:
            config = self.create_config()

        # Build messages
        messages = self._build_messages(prompt, system_prompt)

        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": config.max_tokens,
            "min_new_tokens": config.min_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "do_sample": config.do_sample,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the generated part (not the input)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_length:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def health_check(self) -> bool:
        """
        Check if model is loaded and ready.

        Returns:
            True if model is ready
        """
        return self.model is not None and self.processor is not None

    def get_model_info(self) -> Dict:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'dtype': str(self.dtype),
            'model_loaded': self.model is not None,
            'processor_loaded': self.processor is not None,
        }


def create_gemma_generator(
    settings: Settings,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> GemmaStreamingGenerator:
    """
    Factory function to create Gemma generator.

    Args:
        settings: Application settings
        model_path: Optional model path override
        device: Optional device override

    Returns:
        GemmaStreamingGenerator instance
    """
    return GemmaStreamingGenerator(
        settings=settings,
        model_path=model_path,
        device=device,
    )
