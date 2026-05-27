"""
WuYa LLM Client Module

Unified LLM client infrastructure supporting OpenAI and Anthropic providers.
Provides configuration management, retry strategies, token counting, rate limiting,
provider abstraction, and a unified client interface.
"""

from .config import LLMConfig, LLMProvider
from .retry import (
    LLMError,
    RateLimitError,
    TimeoutError,
    AuthenticationError,
    BadRequestError,
    ServiceUnavailableError,
    ConnectionError,
    ContentFilterError,
    retry_with_backoff,
    classify_http_error,
)
from .token_counter import TokenCounter, TokenUsage
from .rate_limiter import RateLimiter
from .base_provider import BaseLLMProvider, LLMResponse
from .client import LLMClient, create_llm_client

__all__ = [
    # Config
    "LLMConfig",
    "LLMProvider",
    # Retry
    "LLMError",
    "RateLimitError",
    "TimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "ServiceUnavailableError",
    "ConnectionError",
    "ContentFilterError",
    "retry_with_backoff",
    "classify_http_error",
    # Token counting
    "TokenCounter",
    "TokenUsage",
    # Rate limiting
    "RateLimiter",
    # Provider abstraction
    "BaseLLMProvider",
    "LLMResponse",
    # Unified client
    "LLMClient",
    "create_llm_client",
]
