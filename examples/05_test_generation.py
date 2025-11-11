"""
Example 5: Test generation component (without retrieval).

This script demonstrates:
1. Setting up vLLM generator
2. Testing streaming generation
3. Testing non-streaming generation
4. Health checks
"""

import sys
import asyncio
sys.path.append('..')

from src.config import get_settings
from src.generation import VLLMStreamingGenerator, GenerationConfig


async def test_health_check():
    """Test if vLLM server is healthy."""
    print("\n" + "="*80)
    print("HEALTH CHECK")
    print("="*80 + "\n")

    settings = get_settings()
    generator = VLLMStreamingGenerator(settings=settings)

    is_healthy = await generator.health_check()

    if is_healthy:
        print("✓ vLLM server is healthy and ready")
    else:
        print("❌ vLLM server is not accessible")
        print(f"   Check if server is running at: {generator.server_url}")
        print(f"   Run: bash examples/04_setup_vllm.sh")

    return is_healthy


async def test_streaming_generation():
    """Test streaming generation."""
    print("\n" + "="*80)
    print("STREAMING GENERATION TEST")
    print("="*80 + "\n")

    settings = get_settings()
    generator = VLLMStreamingGenerator(settings=settings)

    # Simple test prompt
    prompt = """Aşağıdaki soruyu Türkçe olarak yanıtla:

Soru: Makine öğrenmesi nedir?

Yanıt:"""

    print("Prompt:")
    print(prompt)
    print("\n" + "-"*80)
    print("Streaming Response:")
    print("-"*80 + "\n")

    # Create generation config
    config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        max_tokens=200,
        stream=True,
    )

    # Stream tokens
    tokens = []
    async for token in generator.generate_stream(prompt, config):
        tokens.append(token)
        print(token, end="", flush=True)

    print("\n\n" + "-"*80)
    print(f"Generated {len(tokens)} tokens")
    print("✓ Streaming generation test complete!")


async def test_non_streaming_generation():
    """Test non-streaming (complete) generation."""
    print("\n" + "="*80)
    print("NON-STREAMING GENERATION TEST")
    print("="*80 + "\n")

    settings = get_settings()
    generator = VLLMStreamingGenerator(settings=settings)

    prompt = """Aşağıdaki soruyu kısaca yanıtla:

Soru: Yapay zeka ve makine öğrenmesi arasındaki fark nedir?

Yanıt:"""

    print("Generating complete response...\n")

    # Generate complete response
    config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        max_tokens=150,
    )

    response = await generator.generate(prompt, config)

    print("Response:")
    print("-"*80)
    print(response)
    print("-"*80)

    print("\n✓ Non-streaming generation test complete!")


async def test_custom_config():
    """Test generation with custom config."""
    print("\n" + "="*80)
    print("CUSTOM CONFIGURATION TEST")
    print("="*80 + "\n")

    settings = get_settings()
    generator = VLLMStreamingGenerator(settings=settings)

    # Test with different temperatures
    prompt = "Kısa bir cümle ile: Günümüzde yapay zeka"

    temperatures = [0.3, 0.7, 1.0]

    for temp in temperatures:
        config = generator.create_config(
            temperature=temp,
            max_tokens=50,
        )

        print(f"\nTemperature: {temp}")
        print("-"*80)

        response = await generator.generate(prompt, config)
        print(response)

    print("\n✓ Custom configuration test complete!")


async def main():
    """Run all generation tests."""
    print("="*80)
    print("GENERATION COMPONENT TESTING")
    print("="*80)

    # Test 1: Health check
    is_healthy = await test_health_check()

    if not is_healthy:
        print("\n❌ vLLM server not available. Please start the server first.")
        print("   Run: bash examples/04_setup_vllm.sh")
        return

    # Test 2: Streaming generation
    await test_streaming_generation()

    # Test 3: Non-streaming generation
    await test_non_streaming_generation()

    # Test 4: Custom configuration
    await test_custom_config()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
