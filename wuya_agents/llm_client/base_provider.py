"""
Base provider abstraction for WuYa LLM Client.

Defines the abstract interface that all LLM providers must implement,
along with the unified response data class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional


@dataclass
class LLMResponse:
    """
    Unified response format from any LLM provider.

    Normalizes the differences between OpenAI, Anthropic, and other
    providers into a single consistent structure.
    """

    content: str
    """The generated text content."""

    model: str = ""
    """Model identifier used for generation."""

    input_tokens: int = 0
    """Number of tokens in the prompt (as reported by the provider)."""

    output_tokens: int = 0
    """Number of tokens in the completion (as reported by the provider)."""

    finish_reason: str = ""
    """Reason for completion: 'stop', 'max_tokens', 'content_filter', etc."""

    provider: str = ""
    """Provider name (e.g., 'openai', 'anthropic')."""

    request_id: str = ""
    """Provider-specific request ID for debugging."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional provider-specific metadata."""

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to a plain dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "provider": self.provider,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ") + ("..." if len(self.content) > 80 else "")
        return (
            f"LLMResponse(model={self.model!r}, "
            f"tokens={self.total_tokens}, "
            f"finish_reason={self.finish_reason!r}, "
            f"content={preview!r})"
        )


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers (OpenAI, Anthropic, etc.) must implement the
    ``generate`` and ``generate_stream`` methods. The provider
    is responsible for:
        - Making HTTP requests to the provider's API
        - Handling provider-specific authentication and headers
        - Mapping provider-specific error codes to LLMError subclasses
        - Parsing responses into the unified LLMResponse format
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The user prompt text.
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in the response.
            system_prompt: Optional system prompt for behavior control.

        Returns:
            LLMResponse with the generated content and metadata.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a completion as an async stream of text chunks.

        Yields incremental text deltas as they arrive from the provider.
        Useful for real-time display of responses.

        Args:
            prompt: The user prompt text.
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in the response.
            system_prompt: Optional system prompt for behavior control.

        Yields:
            str: Incremental text chunks.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def close(self) -> None:
        """
        Clean up resources (e.g., close HTTP sessions).

        Should be called when the provider is no longer needed.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier string (e.g., 'openai', 'anthropic')."""
        ...
