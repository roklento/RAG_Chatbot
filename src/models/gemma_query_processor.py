"""
Gemma Query Processor - Query correction and diversification using Gemma-3n.

Replaces Qwen-based query processing for Colab environment.
"""

from typing import List, Optional
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration


class GemmaQueryProcessor:
    """
    Query processor using Gemma-3n-E4B-it model.

    Features:
    - Query correction (spelling and grammar)
    - Query diversification (generate alternative queries)
    - Turkish language support
    """

    def __init__(
        self,
        model_path: str = "google/gemma-3n-E4B-it",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        """
        Initialize Gemma query processor.

        Args:
            model_path: Path to Gemma model
            device: Device to use
            dtype: Data type (bfloat16, float16, float32)
        """
        self.model_path = model_path
        self.device = device

        # Convert dtype string to torch dtype
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        self.dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

        self.model = None
        self.processor = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load Gemma model and processor."""
        print(f"Loading Gemma model for query processing: {self.model_path}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Load model
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).eval()

        print(f"✓ Query processor model loaded")

    def _generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Build messages
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]

        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,  # Low temperature for consistent corrections
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_length:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        return generated_text.strip()

    def correct_query(self, query: str) -> str:
        """
        Correct spelling and grammar in query.

        Args:
            query: Original query

        Returns:
            Corrected query
        """
        # Prompt for correction (Turkish)
        prompt = f"""Aşağıdaki arama sorgusundaki yazım ve dilbilgisi hatalarını düzelt. Sadece düzeltilmiş sorguyu döndür, başka açıklama yapma.

Orijinal sorgu: {query}

Düzeltilmiş sorgu:"""

        try:
            corrected = self._generate_text(prompt, max_tokens=100)

            # Clean up response (remove any extra explanations)
            # Take first line only
            corrected = corrected.split('\n')[0].strip()

            # If response is empty or too different, return original
            if not corrected or len(corrected) > len(query) * 2:
                return query

            return corrected

        except Exception as e:
            print(f"⚠️  Query correction failed: {e}")
            return query  # Return original on error

    def diversify_query(
        self,
        query: str,
        num_variations: int = 2,
    ) -> List[str]:
        """
        Generate alternative query variations.

        Args:
            query: Original query
            num_variations: Number of variations to generate

        Returns:
            List of query variations (including original)
        """
        # Prompt for diversification (Turkish)
        prompt = f"""Aşağıdaki arama sorgusu için {num_variations} farklı alternatif sorgu üret. Her alternatif aynı anlamı taşımalı ama farklı kelimeler kullanmalı.

Orijinal sorgu: {query}

Her alternatifi yeni satırda yaz. Sadece alternatifleri yaz, numaralandırma veya açıklama yapma.

Alternatifler:"""

        try:
            response = self._generate_text(prompt, max_tokens=150)

            # Parse variations (split by newline)
            variations = [
                line.strip()
                for line in response.split('\n')
                if line.strip() and len(line.strip()) > 5
            ]

            # Filter out invalid variations
            valid_variations = []
            for var in variations[:num_variations]:
                # Remove numbering if present (1., 2., etc.)
                var = var.lstrip('0123456789.-) ')
                # Remove quotes
                var = var.strip('"\'')

                if var and var.lower() != query.lower():
                    valid_variations.append(var)

            # Combine with original
            all_queries = [query] + valid_variations

            return all_queries[:num_variations + 1]  # Original + N variations

        except Exception as e:
            print(f"⚠️  Query diversification failed: {e}")
            return [query]  # Return original only on error

    def process_query(
        self,
        query: str,
        enable_correction: bool = True,
        enable_diversification: bool = True,
        num_variations: int = 2,
    ) -> List[str]:
        """
        Process query with correction and diversification.

        Args:
            query: Original query
            enable_correction: Enable spelling/grammar correction
            enable_diversification: Enable query diversification
            num_variations: Number of variations to generate

        Returns:
            List of processed queries (corrected + variations)
        """
        # Step 1: Correct query
        if enable_correction:
            corrected_query = self.correct_query(query)
        else:
            corrected_query = query

        # Step 2: Diversify query
        if enable_diversification:
            queries = self.diversify_query(corrected_query, num_variations)
        else:
            queries = [corrected_query]

        return queries


def create_gemma_query_processor(
    model_path: str = "google/gemma-3n-E4B-it",
    device: str = "cuda",
) -> GemmaQueryProcessor:
    """
    Factory function to create Gemma query processor.

    Args:
        model_path: Path to Gemma model
        device: Device to use

    Returns:
        GemmaQueryProcessor instance
    """
    return GemmaQueryProcessor(
        model_path=model_path,
        device=device,
    )
