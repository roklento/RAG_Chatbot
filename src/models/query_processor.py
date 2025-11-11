"""
Query Processing using Qwen3-Next-80B-A3B-Instruct.
Handles query correction and diversification for improved retrieval.
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from ..config import Settings


@dataclass
class ProcessedQuery:
    """Container for processed query results."""
    original_query: str
    corrected_query: str
    query_variants: List[str]
    all_queries: List[str]  # corrected + variants


class QueryProcessor:
    """
    Query processor using Qwen3-Next-80B for correction and diversification.

    Features:
    - Spelling and grammar correction
    - Semantic query variation generation
    - Context-aware query understanding
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Next-80B-A3B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        num_variants: int = 3,
    ):
        """
        Initialize the query processor.

        Args:
            model_path: HuggingFace model ID or local path
            device: Device to use ('cuda', 'cpu', or 'auto')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_variants: Number of query variants to generate
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_variants = num_variants

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device if device != "auto" else "auto",
        )
        self.model.eval()

    def _create_correction_prompt(self, query: str) -> str:
        """
        Create prompt for query correction.

        Args:
            query: Original user query

        Returns:
            Formatted prompt for correction
        """
        return f"""You are a helpful assistant that corrects spelling and grammar errors in search queries.

Task: Correct any spelling or grammar mistakes in the following query, but keep the meaning and intent exactly the same. If the query is already correct, return it as-is.

Rules:
- Fix only obvious spelling and grammar errors
- Maintain the original intent and meaning
- Do not add extra information
- Return ONLY the corrected query, nothing else

Query: {query}

Corrected query:"""

    def _create_diversification_prompt(self, query: str, num_variants: int) -> str:
        """
        Create prompt for query diversification.

        Args:
            query: Corrected query
            num_variants: Number of variants to generate

        Returns:
            Formatted prompt for diversification
        """
        return f"""You are a helpful assistant that generates alternative versions of search queries to improve retrieval.

Task: Generate {num_variants} different versions of the following query that maintain the same meaning but use different wording and perspectives.

Rules:
- Each variant should ask for the same information in a different way
- Use synonyms, different phrase structures, or alternative formulations
- Keep the same intent and meaning
- Make the variants diverse from each other
- Return ONLY the variants, one per line, numbered 1-{num_variants}

Original query: {query}

Generate {num_variants} alternative versions:"""

    def _generate_response(self, prompt: str) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                top_p=0.9,
            )

        # Decode only the new tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return response.strip()

    def correct_query(self, query: str) -> str:
        """
        Correct spelling and grammar in the query.

        Args:
            query: Original query

        Returns:
            Corrected query
        """
        prompt = self._create_correction_prompt(query)
        corrected = self._generate_response(prompt)

        # Clean up response (remove any extra text)
        corrected = corrected.split('\n')[0].strip()

        # Fallback to original if correction seems wrong
        if len(corrected) == 0 or len(corrected) > len(query) * 3:
            return query

        return corrected

    def generate_variants(self, query: str, num_variants: int = None) -> List[str]:
        """
        Generate query variants.

        Args:
            query: Corrected query
            num_variants: Number of variants (uses default if None)

        Returns:
            List of query variants
        """
        num_variants = num_variants or self.num_variants
        prompt = self._create_diversification_prompt(query, num_variants)
        response = self._generate_response(prompt)

        # Parse variants from response
        variants = []
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Remove numbering (1., 2., etc.) if present
            if line[0].isdigit() and ('. ' in line or ') ' in line):
                line = line.split('. ', 1)[-1].split(') ', 1)[-1]

            # Remove quotes if present
            line = line.strip('"').strip("'")

            if line and line != query:
                variants.append(line)

        # Ensure we have exactly num_variants (or close to it)
        variants = variants[:num_variants]

        # If we didn't get enough variants, add the original query
        while len(variants) < num_variants:
            variants.append(query)

        return variants

    def process(self, query: str, num_variants: int = None) -> ProcessedQuery:
        """
        Process query: correct and generate variants.

        Args:
            query: Original user query
            num_variants: Number of variants to generate

        Returns:
            ProcessedQuery object with all results
        """
        # Step 1: Correct the query
        corrected = self.correct_query(query)

        # Step 2: Generate variants
        variants = self.generate_variants(corrected, num_variants)

        # Combine all queries
        all_queries = [corrected] + variants

        return ProcessedQuery(
            original_query=query,
            corrected_query=corrected,
            query_variants=variants,
            all_queries=all_queries,
        )

    def __call__(self, query: str) -> ProcessedQuery:
        """Convenience method for processing."""
        return self.process(query)


def create_query_processor(settings: Settings) -> QueryProcessor:
    """
    Factory function to create query processor from settings.

    Args:
        settings: Application settings

    Returns:
        Initialized QueryProcessor
    """
    return QueryProcessor(
        model_path=settings.llm_model_path,
        device=settings.device,
        max_new_tokens=settings.llm_max_new_tokens,
        temperature=settings.llm_temperature,
        num_variants=settings.query_variants_count,
    )
