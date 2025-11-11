"""
Prompt Builder - Constructs prompts for generation.
Includes advanced Turkish system prompt adapted for RAG chatbot.
"""

from typing import List, Optional
from .context_augmenter import AugmentedContext
from .conversation_memory import ConversationMessage
from ..config import Settings


# Advanced Turkish System Prompt for RAG Chatbot
SYSTEM_PROMPT_TURKISH = """ROLÜN: Yapay zeka destekli yardımcı asistan.

HİTAP: Kullanıcıya doğal, samimi ama profesyonel bir şekilde hitap et.

GENEL KURALLAR:
• Varsayılan dil **Türkçe**. Kullanıcı açıkça başka dil istemedikçe Türkçe yanıtla.
• Sohbet (küçük konuşma, selamlaşma, teşekkür) sorularında belgelere dayanman gerekmez; doğal, kısa ve samimi yanıt ver.
• Bilgi/rehberlik sorularında **YALNIZCA SAĞLANAN BAĞLAM BİLGİLERİ**ne dayan. Kesinlikle uydurma yapma.
• Emin olmadığın veya bağlamlarda bulunmayan bilgiler için açıkça "Bu konuda yeterli bilgim yok" de.
• Her bilgiyi köşeli parantez atıflarla destekle: [1], [2], [3].
• Birden fazla kaynak aynı bilgiyi destekliyorsa, hepsini belirt: [1][2].
• Bağlamlarda çelişki varsa, bunu kullanıcıya açıkça belirt ve her iki görüşü de atıflarıyla sun.
• İç düşünme süreçlerini paylaşma; direkt olarak net, anlaşılır yanıt ver.
• Yanıt formatı **saf metin** olmalı: Markdown başlıkları, kod blokları, tablolar, HTML kullanma.
• Eğer kullanıcı takip sorusu soruyor ve önceki sohbete atıfta bulunuyorsa, geçmiş mesajları dikkate al.
• Yanıtların uzunluğunu sorunun karmaşıklığına göre ayarla: Basit sorulara kısa, karmaşık sorulara detaylı yanıt ver.

YANITLAMA PRENSİPLERİ:
1. **Doğruluk**: Sadece verilen bağlamlardaki bilgileri kullan
2. **Şeffaflık**: Bilmediğinde itiraf et, tahmin yapma
3. **Atıf**: Her iddiayı kaynak numarasıyla destekle
4. **Netlik**: Kısa, öz ve anlaşılır ol
5. **Bağlam Farkındalığı**: Önceki konuşmaları ve sağlanan bağlamları birlikte değerlendir
6. **Yardımseverlik**: Kullanıcının sorusuna gerçekten yardımcı olmaya çalış"""


class PromptBuilder:
    """
    Constructs prompts for generation with conversation history and contexts.

    Features:
    - System prompt injection
    - Conversation history formatting
    - Context integration with citations
    - Flexible prompt templates
    """

    def __init__(
        self,
        settings: Settings,
        system_prompt: str = SYSTEM_PROMPT_TURKISH,
    ):
        """
        Initialize prompt builder.

        Args:
            settings: Application settings
            system_prompt: System prompt template
        """
        self.settings = settings
        self.system_prompt = system_prompt

    def build(
        self,
        user_query: str,
        contexts: List[AugmentedContext],
        conversation_history: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build complete prompt for generation.

        Args:
            user_query: Current user query
            contexts: Augmented contexts with citations
            conversation_history: Formatted conversation history
            custom_system_prompt: Optional custom system prompt

        Returns:
            Complete prompt string
        """
        system_prompt = custom_system_prompt or self.system_prompt

        # Format contexts
        from .context_augmenter import ContextAugmenter
        augmenter = ContextAugmenter(self.settings)
        formatted_contexts = augmenter.format_for_prompt(contexts)

        # Build prompt sections
        prompt_parts = [system_prompt]

        # Add contexts section
        prompt_parts.append("\n\nBAĞLAM BİLGİLERİ:")
        prompt_parts.append(formatted_contexts)

        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("\n\nÖNCEKİ SOHBET:")
            prompt_parts.append(conversation_history)

        # Add current query
        prompt_parts.append("\n\nKULLANICI SORUSU:")
        prompt_parts.append(user_query)

        # Add generation instruction
        prompt_parts.append("\n\nYANITINIZ (Türkçe, saf metin, atıflarla):")

        return "\n".join(prompt_parts)

    def build_simple(
        self,
        user_query: str,
        contexts: List[AugmentedContext],
    ) -> str:
        """
        Build simple prompt without conversation history.

        Args:
            user_query: Current user query
            contexts: Augmented contexts

        Returns:
            Simple prompt string
        """
        return self.build(
            user_query=user_query,
            contexts=contexts,
            conversation_history=None,
        )

    def build_chat(
        self,
        user_query: str,
        contexts: List[AugmentedContext],
        messages: List[ConversationMessage],
    ) -> str:
        """
        Build prompt with conversation messages.

        Args:
            user_query: Current user query
            contexts: Augmented contexts
            messages: Conversation messages

        Returns:
            Complete prompt string
        """
        # Format messages
        history_parts = []
        for msg in messages:
            if msg.role == "user":
                history_parts.append(f"Kullanıcı: {msg.content}")
            elif msg.role == "assistant":
                history_parts.append(f"Asistan: {msg.content}")

        conversation_history = "\n\n".join(history_parts) if history_parts else None

        return self.build(
            user_query=user_query,
            contexts=contexts,
            conversation_history=conversation_history,
        )

    def build_fallback(self, user_query: str) -> str:
        """
        Build fallback prompt when no contexts available.

        Args:
            user_query: User query

        Returns:
            Fallback prompt
        """
        fallback_system = """Sen yardımsever bir asistansın. Kullanıcıya nazik bir şekilde
şu anda bu konuda yeterli bilgiye sahip olmadığını belirt ve farklı bir soru sormalarını öner."""

        return f"""{fallback_system}

KULLANICI SORUSU: {user_query}

YANITINIZ:"""

    def estimate_prompt_tokens(
        self,
        user_query: str,
        contexts: List[AugmentedContext],
        conversation_history: Optional[str] = None,
    ) -> int:
        """
        Estimate total tokens in prompt.

        Args:
            user_query: User query
            contexts: Augmented contexts
            conversation_history: Optional conversation history

        Returns:
            Estimated token count
        """
        import tiktoken

        try:
            encoder = tiktoken.get_encoding("cl100k_base")
        except:
            # Rough estimate: 1 token ≈ 4 characters for Turkish
            prompt = self.build(user_query, contexts, conversation_history)
            return len(prompt) // 4

        prompt = self.build(user_query, contexts, conversation_history)
        return len(encoder.encode(prompt))


def create_prompt_builder(
    settings: Settings,
    custom_system_prompt: Optional[str] = None,
) -> PromptBuilder:
    """
    Factory function to create prompt builder.

    Args:
        settings: Application settings
        custom_system_prompt: Optional custom system prompt

    Returns:
        Initialized PromptBuilder
    """
    system_prompt = custom_system_prompt or SYSTEM_PROMPT_TURKISH
    return PromptBuilder(settings=settings, system_prompt=system_prompt)
