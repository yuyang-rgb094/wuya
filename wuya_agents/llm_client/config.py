"""
Configuration management for WuYa LLM Client.

Supports loading configuration from environment variables and .env files.
Uses dataclass for type-safe configuration with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class RetryConfig:
    """Retry policy configuration."""

    max_retries: int = 3
    """Maximum number of retry attempts."""

    base_delay: float = 1.0
    """Base delay in seconds for exponential backoff."""

    max_delay: float = 60.0
    """Maximum delay cap in seconds."""

    backoff_factor: float = 2.0
    """Multiplier for exponential backoff (delay = base_delay * factor^n)."""

    jitter: bool = True
    """Whether to add random jitter to prevent thundering herd."""

    @classmethod
    def from_env(cls) -> RetryConfig:
        """Create RetryConfig from environment variables."""
        return cls(
            max_retries=int(os.getenv("WUYA_LLM_MAX_RETRIES", "3")),
            base_delay=float(os.getenv("WUYA_LLM_RETRY_BASE_DELAY", "1.0")),
            max_delay=float(os.getenv("WUYA_LLM_RETRY_MAX_DELAY", "60.0")),
            backoff_factor=float(os.getenv("WUYA_LLM_RETRY_BACKOFF_FACTOR", "2.0")),
            jitter=os.getenv("WUYA_LLM_RETRY_JITTER", "true").lower() == "true",
        )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    rpm: int = 100
    """Requests per minute limit."""

    tpm: int = 100_000
    """Tokens per minute limit."""

    @classmethod
    def from_env(cls) -> RateLimitConfig:
        """Create RateLimitConfig from environment variables."""
        return cls(
            rpm=int(os.getenv("WUYA_LLM_RATE_LIMIT_RPM", "100")),
            tpm=int(os.getenv("WUYA_LLM_RATE_LIMIT_TPM", "100000")),
        )


@dataclass
class LLMConfig:
    """
    Unified LLM configuration.

    Manages provider selection, API keys, model parameters,
    retry policy, and rate limits. Supports loading from
    environment variables with sensible defaults.
    """

    # --- Provider ---
    provider: LLMProvider = LLMProvider.OPENAI
    """LLM provider to use."""

    # --- API Keys ---
    openai_api_key: Optional[str] = None
    """OpenAI API key. Loaded from OPENAI_API_KEY env var."""

    anthropic_api_key: Optional[str] = None
    """Anthropic API key. Loaded from ANTHROPIC_API_KEY env var."""

    # --- Model Selection ---
    openai_model: str = "gpt-4o"
    """OpenAI model identifier. Default: gpt-4o."""

    anthropic_model: str = "claude-3-5-sonnet-20241022"
    """Anthropic model identifier. Default: claude-3-5-sonnet-20241022."""

    # --- Generation Parameters ---
    temperature: float = 0.1
    """Sampling temperature. Lower = more deterministic. Default: 0.1."""

    max_tokens: int = 2000
    """Maximum tokens in the completion response. Default: 2000."""

    # --- Timeouts ---
    request_timeout: float = 120.0
    """HTTP request timeout in seconds. Default: 120."""

    # --- Retry & Rate Limiting ---
    retry: RetryConfig = field(default_factory=RetryConfig)
    """Retry policy configuration."""

    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    """Rate limiting configuration."""

    # --- Advanced ---
    api_base_url: Optional[str] = None
    """Custom API base URL (for proxies or self-hosted endpoints)."""

    @property
    def model(self) -> str:
        """Return the model name for the current provider."""
        if self.provider == LLMProvider.OPENAI:
            return self.openai_model
        return self.anthropic_model

    @property
    def api_key(self) -> Optional[str]:
        """Return the API key for the current provider."""
        if self.provider == LLMProvider.OPENAI:
            return self.openai_api_key
        return self.anthropic_api_key

    @classmethod
    def from_env(cls) -> LLMConfig:
        """
        Create LLMConfig from environment variables.

        Environment variables:
            WUYA_LLM_PROVIDER       - Provider: "openai" or "anthropic" (default: openai)
            OPENAI_API_KEY          - OpenAI API key
            ANTHROPIC_API_KEY       - Anthropic API key
            WUYA_OPENAI_MODEL       - OpenAI model (default: gpt-4o)
            WUYA_ANTHROPIC_MODEL    - Anthropic model (default: claude-3-5-sonnet-20241022)
            WUYA_LLM_TEMPERATURE    - Temperature (default: 0.1)
            WUYA_LLM_MAX_TOKENS     - Max tokens (default: 2000)
            WUYA_LLM_REQUEST_TIMEOUT - Request timeout in seconds (default: 120)
            WUYA_LLM_API_BASE_URL   - Custom API base URL

        Also loads RetryConfig and RateLimitConfig from their respective env vars.
        """
        provider_str = os.getenv("WUYA_LLM_PROVIDER", "openai").lower()

        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            raise ValueError(
                f"Unknown provider '{provider_str}'. "
                f"Must be one of: {[p.value for p in LLMProvider]}"
            )

        config = cls(
            provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_model=os.getenv("WUYA_OPENAI_MODEL", "gpt-4o"),
            anthropic_model=os.getenv(
                "WUYA_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
            ),
            temperature=float(os.getenv("WUYA_LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("WUYA_LLM_MAX_TOKENS", "2000")),
            request_timeout=float(os.getenv("WUYA_LLM_REQUEST_TIMEOUT", "120")),
            retry=RetryConfig.from_env(),
            rate_limit=RateLimitConfig.from_env(),
            api_base_url=os.getenv("WUYA_LLM_API_BASE_URL"),
        )

        return config

    @classmethod
    def load_dot_env(cls, env_path: str | Path = ".env") -> LLMConfig:
        """
        Load configuration from a .env file, then override with actual env vars.

        This is a lightweight alternative to python-dotenv. It reads key=value
        pairs from the given file and sets them as environment variables before
        calling from_env(). Actual environment variables take precedence.

        Args:
            env_path: Path to the .env file. Defaults to ".env".

        Returns:
            LLMConfig instance.
        """
        env_file = Path(env_path)
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue
                    # Parse KEY=VALUE, ignoring lines without =
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value

        return cls.from_env()

    def validate(self) -> list[str]:
        """
        Validate the configuration and return a list of issues.

        Returns:
            List of validation error messages. Empty list means valid.
        """
        issues: list[str] = []

        if self.provider == LLMProvider.OPENAI and not self.openai_api_key:
            issues.append(
                "OpenAI API key is required when provider is OPENAI. "
                "Set OPENAI_API_KEY environment variable."
            )
        elif self.provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            issues.append(
                "Anthropic API key is required when provider is ANTHROPIC. "
                "Set ANTHROPIC_API_KEY environment variable."
            )

        if not (0.0 <= self.temperature <= 2.0):
            issues.append(
                f"Temperature must be between 0.0 and 2.0, got {self.temperature}."
            )

        if self.max_tokens < 1:
            issues.append(f"max_tokens must be >= 1, got {self.max_tokens}.")

        if self.request_timeout < 1:
            issues.append(
                f"request_timeout must be >= 1 second, got {self.request_timeout}."
            )

        if self.retry.max_retries < 0:
            issues.append(
                f"max_retries must be >= 0, got {self.retry.max_retries}."
            )

        if self.rate_limit.rpm < 1:
            issues.append(f"RPM must be >= 1, got {self.rate_limit.rpm}.")

        if self.rate_limit.tpm < 1:
            issues.append(f"TPM must be >= 1, got {self.rate_limit.tpm}.")

        return issues

    def __repr__(self) -> str:
        """String representation with API key masked."""
        masked_openai = (
            f"{self.openai_api_key[:8]}...{self.openai_api_key[-4:]}"
            if self.openai_api_key and len(self.openai_api_key) > 12
            else "***"
        )
        masked_anthropic = (
            f"{self.anthropic_api_key[:8]}...{self.anthropic_api_key[-4:]}"
            if self.anthropic_api_key and len(self.anthropic_api_key) > 12
            else "***"
        )
        return (
            f"LLMConfig(provider={self.provider.value}, "
            f"model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, "
            f"openai_key={masked_openai}, "
            f"anthropic_key={masked_anthropic})"
        )
