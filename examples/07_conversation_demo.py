"""
Example 7: Multi-turn conversation demonstration.

This script demonstrates:
1. Multi-turn conversation flow
2. Conversation history tracking
3. Token management
4. Session management
"""

import sys
import asyncio
sys.path.append('..')

from src.config import get_settings
from src.generation import (
    VLLMStreamingGenerator,
    ConversationMemoryManager,
    PromptBuilder,
)


class SimpleConversationDemo:
    """Simple conversation demo without retrieval."""

    def __init__(self, settings):
        self.settings = settings
        self.generator = VLLMStreamingGenerator(settings)
        self.memory = ConversationMemoryManager(settings)
        self.prompt_builder = PromptBuilder(settings)

    async def chat(self, query: str, session_id: str = "demo"):
        """Process a single query in conversation."""

        # Get conversation history
        history = self.memory.get_history(session_id)

        # Build prompt (without contexts for this demo)
        prompt = self.prompt_builder.build_prompt(
            query=query,
            augmented_contexts=[],  # No retrieval in this demo
            conversation_history=history,
        )

        # Stream response
        print("\n🤖 Assistant: ", end="", flush=True)

        tokens = []
        async for token in self.generator.generate_stream(prompt):
            tokens.append(token)
            print(token, end="", flush=True)

        print()  # New line after response

        generated_text = "".join(tokens)

        # Update conversation history
        self.memory.add_message(session_id, "user", query)
        self.memory.add_message(session_id, "assistant", generated_text)

        # Show stats
        total_tokens = self.memory.get_total_tokens(session_id)
        message_count = len(self.memory.get_history(session_id))
        usage_percent = (total_tokens / self.settings.max_conversation_tokens) * 100

        print(f"\n📊 Stats: {message_count} messages, {total_tokens:,} tokens ({usage_percent:.1f}% of limit)")

        # Check if approaching limit
        if self.memory.should_reset(session_id):
            print("⚠️  Warning: Approaching token limit (80%). Consider resetting conversation.")

    async def run_demo(self):
        """Run interactive conversation demo."""

        # Check health
        is_healthy = await self.generator.health_check()
        if not is_healthy:
            print("❌ vLLM server not available. Please start it first:")
            print("   bash examples/04_setup_vllm.sh")
            return

        print("="*80)
        print("MULTI-TURN CONVERSATION DEMO")
        print("="*80)
        print("\nThis demo shows conversation history tracking without retrieval.")
        print("Type 'quit' to exit, 'reset' to clear history, 'stats' for statistics.\n")

        session_id = "interactive_demo"

        while True:
            try:
                # Get user input
                query = input("\n👤 You: ").strip()

                if not query:
                    continue

                if query.lower() == 'quit':
                    print("\n👋 Goodbye!")
                    break

                if query.lower() == 'reset':
                    self.memory.reset_session(session_id)
                    print("✓ Conversation history reset")
                    continue

                if query.lower() == 'stats':
                    total_tokens = self.memory.get_total_tokens(session_id)
                    messages = self.memory.get_history(session_id)
                    print(f"\n📊 Conversation Statistics:")
                    print(f"   Messages: {len(messages)}")
                    print(f"   Total tokens: {total_tokens:,}")
                    print(f"   Max tokens: {self.settings.max_conversation_tokens:,}")
                    print(f"   Usage: {(total_tokens / self.settings.max_conversation_tokens) * 100:.1f}%")
                    continue

                # Process query
                await self.chat(query, session_id)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


async def run_scripted_conversation():
    """Run a pre-scripted conversation demo."""
    print("="*80)
    print("SCRIPTED MULTI-TURN CONVERSATION")
    print("="*80 + "\n")

    settings = get_settings()
    demo = SimpleConversationDemo(settings)

    # Check health
    is_healthy = await demo.generator.health_check()
    if not is_healthy:
        print("❌ vLLM server not available")
        return

    session_id = "scripted_demo"

    # Pre-scripted conversation
    conversation = [
        "Merhaba! Makine öğrenmesi hakkında konuşmak istiyorum.",
        "Makine öğrenmesi nedir?",
        "Denetimli öğrenme ile denetimsiz öğrenme arasındaki fark nedir?",
        "Bir örnek verebilir misin?",
        "Teşekkürler, çok yardımcı oldun!",
    ]

    for i, query in enumerate(conversation, 1):
        print(f"\n{'='*80}")
        print(f"Turn {i}/{len(conversation)}")
        print(f"{'='*80}")

        print(f"\n👤 User: {query}")

        await demo.chat(query, session_id)

        # Small delay for readability
        await asyncio.sleep(1)

    print(f"\n\n{'='*80}")
    print("SCRIPTED CONVERSATION COMPLETE")
    print(f"{'='*80}\n")

    # Show final stats
    total_tokens = demo.memory.get_total_tokens(session_id)
    messages = demo.memory.get_history(session_id)

    print(f"Final Statistics:")
    print(f"  Total turns: {len(conversation)}")
    print(f"  Total messages: {len(messages)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens per message: {total_tokens / len(messages):.1f}")


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        # Run interactive demo
        settings = get_settings()
        demo = SimpleConversationDemo(settings)
        await demo.run_demo()
    else:
        # Run scripted demo
        await run_scripted_conversation()
        print("\nTo run interactive mode, use: python 07_conversation_demo.py --interactive")


if __name__ == "__main__":
    asyncio.run(main())
