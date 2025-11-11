"""
Conversation Memory Manager with token tracking.
Manages conversation history and ensures we stay within context window limits.
"""

from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
import tiktoken
from ..config import Settings


@dataclass
class ConversationMessage:
    """Single message in conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0


@dataclass
class ConversationSession:
    """Conversation session with messages and metadata."""
    session_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    total_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


class ConversationMemoryManager:
    """
    Manages conversation history with intelligent token tracking.

    Features:
    - Accurate token counting using tiktoken
    - Auto-truncation when approaching limit
    - Session management
    - Warning system
    - Optional message summarization
    """

    def __init__(
        self,
        settings: Settings,
        encoding_name: str = "cl100k_base",  # Used by Qwen models
    ):
        """
        Initialize conversation memory manager.

        Args:
            settings: Application settings
            encoding_name: Tiktoken encoding (cl100k_base for Qwen)
        """
        self.settings = settings
        self.max_tokens = settings.max_conversation_tokens
        self.warning_threshold = settings.conversation_warning_threshold
        self.messages_to_keep_full = settings.messages_to_keep_full

        # Initialize tiktoken encoder
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to a known encoding
            self.encoder = tiktoken.get_encoding("cl100k_base")

        # Session storage
        self.sessions: Dict[str, ConversationSession] = {}

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encoder.encode(text))

    def create_session(self, session_id: str) -> ConversationSession:
        """
        Create a new conversation session.

        Args:
            session_id: Unique session identifier

        Returns:
            New conversation session
        """
        session = ConversationSession(session_id=session_id)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get existing session or create new one.

        Args:
            session_id: Session identifier

        Returns:
            Conversation session
        """
        if session_id not in self.sessions:
            return self.create_session(session_id)
        return self.sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
    ) -> ConversationMessage:
        """
        Add message to conversation history.

        Args:
            session_id: Session identifier
            role: Message role
            content: Message content

        Returns:
            Created message
        """
        session = self.get_session(session_id)

        # Count tokens
        token_count = self.count_tokens(content)

        # Create message
        message = ConversationMessage(
            role=role,
            content=content,
            token_count=token_count,
        )

        # Add to session
        session.messages.append(message)
        session.total_tokens += token_count
        session.last_activity = datetime.now()

        # Check if we need to truncate
        if session.total_tokens > self.max_tokens:
            self._truncate_session(session)

        return message

    def get_history(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        include_system: bool = False,
    ) -> List[ConversationMessage]:
        """
        Get conversation history within token limit.

        Args:
            session_id: Session identifier
            max_tokens: Optional token limit (uses default if None)
            include_system: Whether to include system messages

        Returns:
            List of messages within token limit
        """
        session = self.get_session(session_id)

        if not session or not session.messages:
            return []

        # Filter messages
        messages = session.messages
        if not include_system:
            messages = [m for m in messages if m.role != "system"]

        # If no limit, return all
        if max_tokens is None:
            return messages

        # Return most recent messages within token limit
        result = []
        total = 0

        for message in reversed(messages):
            if total + message.token_count <= max_tokens:
                result.insert(0, message)
                total += message.token_count
            else:
                break

        return result

    def get_formatted_history(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Get formatted conversation history as string.

        Args:
            session_id: Session identifier
            max_tokens: Optional token limit

        Returns:
            Formatted conversation string
        """
        messages = self.get_history(session_id, max_tokens)

        if not messages:
            return ""

        formatted = []
        for msg in messages:
            if msg.role == "user":
                formatted.append(f"Kullanıcı: {msg.content}")
            elif msg.role == "assistant":
                formatted.append(f"Asistan: {msg.content}")

        return "\n\n".join(formatted)

    def should_reset(self, session_id: str) -> bool:
        """
        Check if session should be reset due to token limit.

        Args:
            session_id: Session identifier

        Returns:
            True if should reset
        """
        session = self.get_session(session_id)

        if not session:
            return False

        threshold_tokens = int(self.max_tokens * self.warning_threshold)
        return session.total_tokens >= threshold_tokens

    def get_token_usage(self, session_id: str) -> Dict[str, any]:
        """
        Get token usage statistics for session.

        Args:
            session_id: Session identifier

        Returns:
            Usage statistics
        """
        session = self.get_session(session_id)

        if not session:
            return {
                "total_tokens": 0,
                "max_tokens": self.max_tokens,
                "usage_percentage": 0.0,
                "messages_count": 0,
                "should_warn": False,
            }

        usage_pct = (session.total_tokens / self.max_tokens) * 100
        should_warn = session.total_tokens >= (self.max_tokens * self.warning_threshold)

        return {
            "total_tokens": session.total_tokens,
            "max_tokens": self.max_tokens,
            "usage_percentage": usage_pct,
            "messages_count": len(session.messages),
            "should_warn": should_warn,
            "tokens_remaining": self.max_tokens - session.total_tokens,
        }

    def reset_session(self, session_id: str):
        """
        Reset conversation session (clear history).

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
        self.create_session(session_id)

    def _truncate_session(self, session: ConversationSession):
        """
        Truncate old messages to stay within token limit.

        Args:
            session: Session to truncate
        """
        # Keep most recent N messages at full detail
        messages_to_keep = self.messages_to_keep_full

        if len(session.messages) <= messages_to_keep:
            return

        # Calculate how many messages to remove
        messages_to_remove = len(session.messages) - messages_to_keep

        # Remove oldest messages
        removed_messages = session.messages[:messages_to_remove]
        removed_tokens = sum(m.token_count for m in removed_messages)

        session.messages = session.messages[messages_to_remove:]
        session.total_tokens -= removed_tokens

    def clear_old_sessions(self, max_age_hours: int = 24):
        """
        Clear inactive sessions older than specified hours.

        Args:
            max_age_hours: Maximum age in hours
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        sessions_to_remove = []

        for session_id, session in self.sessions.items():
            if session.last_activity < cutoff:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.sessions[session_id]


def create_conversation_manager(settings: Settings) -> ConversationMemoryManager:
    """
    Factory function to create conversation memory manager.

    Args:
        settings: Application settings

    Returns:
        Initialized ConversationMemoryManager
    """
    return ConversationMemoryManager(settings=settings)
