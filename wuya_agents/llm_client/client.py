"""
Unified LLM Client for WuYa Multi-Agent System.

Implements the LLMClient Protocol from base.py, providing a single
interface that delegates to the appropriate provider (OpenAI/Anthropic).
Integrates token counting, rate limiting, and retry strategies.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from .base_provider import BaseLLMProvider, LLMResponse
from .config import LLMConfig, LLMProvider
from .rate_limiter import RateLimiter
from .retry import LLMError
from .token_counter import TokenCounter, TokenUsage

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client implementing the LLMClient Protocol.

    Delegates to the appropriate provider based on configuration,
    and integrates token counting and rate limiting transparently.

    This class satisfies the ``LLMClient`` Protocol defined in
    ``wuya_agents.base``:

        async def generate(prompt, temperature=0.1, max_tokens=2000,
                           system_prompt=None) -> str

    Example::

        # From environment variables
        client = await LLMClient.from_env()

        # Simple generation
        text = await client.generate("Evaluate this paper's methodology.")

        # With metadata
        response = await client.generate_with_metadata(
            "Evaluate this paper's methodology.",
            system_prompt="You are an expert reviewer.",
        )
        print(response.content, response.total_tokens)

        # Usage stats
        print(client.get_usage_stats().summary())

        # Cleanup
        await client.close()

    Example (factory function)::

        client = create_llm_client(
            provider="openai",
            model="gpt-4o",
            api_key="sk-...",
        )
        text = await client.generate("Hello!")
        await client.close()
    """

    def __init__(
        self,
        config: LLMConfig,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            config: LLM configuration specifying provider, model, keys, etc.
        """
        self._config = config
        self._provider: Optional[BaseLLMProvider] = None
        self._token_counter = TokenCounter()
        self._rate_limiter = RateLimiter.from_config(config.rate_limit)
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the client (create provider session).

        Can be called explicitly, or will be called lazily on first use.
        """
        if self._initialized:
            return

        # Validate configuration
        issues = self._config.validate()
        if issues:
            raise ValueError(
                f"Invalid LLM configuration:\n  - " + "\n  - ".join(issues)
            )

        # Create the appropriate provider
        if self._config.provider == LLMProvider.OPENAI:
            from .openai_provider import OpenAIProvider

            self._provider = OpenAIProvider.from_config(self._config)
        elif self._config.provider == LLMProvider.ANTHROPIC:
            from .anthropic_provider import AnthropicProvider

            self._provider = AnthropicProvider.from_config(self._config)
        else:
            raise ValueError(f"Unsupported provider: {self._config.provider}")

        self._initialized = True
        logger.info(
            "LLM client initialized: provider=%s, model=%s",
            self._config.provider.value,
            self._config.model,
        )

    def _ensure_initialized(self) -> BaseLLMProvider:
        """Ensure the client is initialized and return the provider."""
        if not self._initialized or self._provider is None:
            raise RuntimeError(
                "LLM client not initialized. Call await client.initialize() first, "
                "or use await LLMClient.from_env()."
            )
        return self._provider

    # =========================================================================
    # Core API (implements LLMClient Protocol)
    # =========================================================================

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from the LLM.

        This method satisfies the ``LLMClient`` Protocol defined in base.py.

        Args:
            prompt: The user prompt text.
            temperature: Sampling temperature (0.0 - 2.0). Default: 0.1.
            max_tokens: Maximum tokens in the response. Default: 2000.
            system_prompt: Optional system prompt for behavior control.

        Returns:
            The generated text as a string.
        """
        response = await self.generate_with_metadata(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        return response.content

    async def generate_with_metadata(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate text with full metadata (tokens, model, etc.).

        Integrates rate limiting and token counting transparently.

        Args:
            prompt: The user prompt text.
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in the response.
            system_prompt: Optional system prompt for behavior control.

        Returns:
            LLMResponse with content, token counts, and metadata.
        """
        provider = self._ensure_initialized()

        # Estimate tokens for rate limiting
        estimated_tokens = self._token_counter.estimate(prompt)
        if system_prompt:
            estimated_tokens += self._token_counter.estimate(system_prompt)

        # Apply rate limiting
        await self._rate_limiter.acquire(token_count=estimated_tokens)

        try:
            response = await provider.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )

            # Record actual token usage
            self._token_counter.record_usage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                model=response.model,
            )

            # Update rate limiter with actual tokens
            self._rate_limiter.record_tokens(response.total_tokens)

            return response

        except LLMError:
            raise
        except Exception as exc:
            logger.error("Unexpected error during generation: %s", exc)
            raise

    async def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.

        Yields incremental text chunks as they arrive.

        Args:
            prompt: The user prompt text.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            system_prompt: Optional system prompt.

        Yields:
            str: Incremental text chunks.
        """
        provider = self._ensure_initialized()

        estimated_tokens = self._token_counter.estimate(prompt)
        if system_prompt:
            estimated_tokens += self._token_counter.estimate(system_prompt)

        await self._rate_limiter.acquire(token_count=estimated_tokens)

        total_output = 0
        try:
            async for chunk in provider.generate_stream(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            ):
                total_output += self._token_counter.estimate(chunk)
                yield chunk

            # Record estimated usage (streaming doesn't give exact counts)
            self._token_counter.record_usage(
                input_tokens=estimated_tokens,
                output_tokens=total_output,
                model=self._config.model,
            )
            self._rate_limiter.record_tokens(estimated_tokens + total_output)

        except LLMError:
            raise
        except Exception as exc:
            logger.error("Unexpected error during streaming: %s", exc)
            raise

    # =========================================================================
    # Usage & Management
    # =========================================================================

    def get_usage_stats(self) -> TokenUsage:
        """
        Get cumulative token usage statistics.

        Returns:
            TokenUsage instance with input/output token counts,
            request counts, and cost estimates.
        """
        return self._token_counter.get_usage()

    @property
    def config(self) -> LLMConfig:
        """Return the current configuration."""
        return self._config

    @property
    def provider(self) -> Optional[BaseLLMProvider]:
        """Return the underlying provider instance."""
        return self._provider

    async def close(self) -> None:
        """Close the client and release resources."""
        if self._provider is not None:
            await self._provider.close()
            self._provider = None
        self._initialized = False
        logger.info("LLM client closed.")

    async def __aenter__(self) -> LLMClient:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return (
            f"LLMClient(provider={self._config.provider.value}, "
            f"model={self._config.model}, {status})"
        )

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    async def from_env(cls) -> LLMClient:
        """
        Create and initialize an LLM client from environment variables.

        Environment variables:
            WUYA_LLM_PROVIDER       - "openai" or "anthropic" (default: openai)
            OPENAI_API_KEY          - OpenAI API key
            ANTHROPIC_API_KEY       - Anthropic API key
            WUYA_OPENAI_MODEL       - OpenAI model (default: gpt-4o)
            WUYA_ANTHROPIC_MODEL    - Anthropic model (default: claude-3-5-sonnet-20241022)
            WUYA_LLM_TEMPERATURE    - Temperature (default: 0.1)
            WUYA_LLM_MAX_TOKENS     - Max tokens (default: 2000)
            ... and other WUYA_LLM_* variables (see config.py)

        Returns:
            Initialized LLMClient instance ready to use.

        Raises:
            ValueError: If required configuration is missing or invalid.
        """
        config = LLMConfig.from_env()
        client = cls(config=config)
        await client.initialize()
        return client

    @classmethod
    async def from_config(cls, config: LLMConfig) -> LLMClient:
        """
        Create and initialize an LLM client from an explicit config.

        Args:
            config: LLMConfig instance.

        Returns:
            Initialized LLMClient instance.
        """
        client = cls(config=config)
        await client.initialize()
        return client


# =============================================================================
# Factory Function
# =============================================================================


async def create_llm_client(
    provider: str = "openai",
    model: str = "",
    api_key: str = "",
    *,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    request_timeout: float = 120.0,
    max_retries: int = 3,
    rpm: int = 100,
    tpm: int = 100_000,
) -> LLMClient:
    """
    Quick factory function to create an LLM client.

    This is a convenience wrapper around LLMConfig + LLMClient.

    Args:
        provider: Provider name ("openai" or "anthropic").
        model: Model identifier. Uses provider default if empty.
        api_key: API key for the provider.
        temperature: Default sampling temperature.
        max_tokens: Default max tokens.
        request_timeout: HTTP timeout in seconds.
        max_retries: Max retry attempts.
        rpm: Rate limit: requests per minute.
        tpm: Rate limit: tokens per minute.

    Returns:
        Initialized LLMClient instance.

    Example::

        client = await create_llm_client(
            provider="openai",
            model="gpt-4o",
            api_key="sk-...",
        )
        text = await client.generate("Hello!")
        await client.close()
    """
    from .config import RateLimitConfig, RetryConfig

    provider_enum = LLMProvider(provider)

    config = LLMConfig(
        provider=provider_enum,
        openai_api_key=api_key if provider_enum == LLMProvider.OPENAI else None,
        anthropic_api_key=api_key if provider_enum == LLMProvider.ANTHROPIC else None,
        openai_model=model or "gpt-4o",
        anthropic_model=model or "claude-3-5-sonnet-20241022",
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        retry=RetryConfig(max_retries=max_retries),
        rate_limit=RateLimitConfig(rpm=rpm, tpm=tpm),
    )

    client = LLMClient(config=config)
    await client.initialize()
    return client
