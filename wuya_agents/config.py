"""
Global configuration management for WuYa Agents.

Uses pydantic-settings for type-safe configuration with .env file support.
All configuration can be overridden via environment variables.

Usage:
    from wuya_agents.config import get_settings, Settings

    settings = get_settings()
    print(settings.llm.provider)
    print(settings.rag.vector_store_type)
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LLMProviderType(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class VectorStoreType(str, Enum):
    """Supported vector store backends."""
    IN_MEMORY = "in_memory"
    CHROMADB = "chromadb"


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMSettings(BaseSettings):
    """LLM client configuration."""

    model_config = SettingsConfigDict(
        env_prefix="WUYA_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: LLMProviderType = LLMProviderType.OPENAI
    """LLM provider to use."""

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    """OpenAI API key."""

    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    """Anthropic API key."""

    openai_model: str = "gpt-4o"
    """OpenAI model identifier."""

    anthropic_model: str = "claude-3-5-sonnet-20241022"
    """Anthropic model identifier."""

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    """Sampling temperature. Lower = more deterministic."""

    max_tokens: int = Field(default=2000, ge=1)
    """Maximum tokens in the completion response."""

    request_timeout: float = Field(default=120.0, ge=1.0)
    """HTTP request timeout in seconds."""

    max_retries: int = Field(default=3, ge=0)
    """Maximum number of retry attempts."""

    base_delay: float = Field(default=1.0, ge=0.1)
    """Base delay in seconds for exponential backoff."""

    max_delay: float = Field(default=60.0, ge=1.0)
    """Maximum delay cap in seconds."""

    rate_limit_rpm: int = Field(default=100, ge=1)
    """Requests per minute limit."""

    rate_limit_tpm: int = Field(default=100_000, ge=1)
    """Tokens per minute limit."""

    api_base_url: Optional[str] = None
    """Custom API base URL (for proxies or self-hosted endpoints)."""

    @property
    def model(self) -> str:
        """Return the model name for the current provider."""
        if self.provider == LLMProviderType.OPENAI:
            return self.openai_model
        return self.anthropic_model

    @property
    def api_key(self) -> Optional[str]:
        """Return the API key for the current provider."""
        if self.provider == LLMProviderType.OPENAI:
            return self.openai_api_key
        return self.anthropic_api_key


class RAGSettings(BaseSettings):
    """RAG (Retrieval-Augmented Generation) configuration."""

    model_config = SettingsConfigDict(
        env_prefix="WUYA_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = True
    """Whether RAG retrieval is enabled."""

    vector_store_type: VectorStoreType = VectorStoreType.IN_MEMORY
    """Vector store backend."""

    chromadb_host: str = "localhost"
    """ChromaDB server host."""

    chromadb_port: int = Field(default=8000, ge=1, le=65535)
    """ChromaDB server port."""

    chromadb_collection: str = "wuya_papers"
    """ChromaDB collection name."""

    embedding_provider: str = "mock"
    """Embedding provider: 'mock', 'openai'."""

    openai_embedding_model: str = "text-embedding-3-small"
    """OpenAI embedding model."""

    top_k: int = Field(default=5, ge=1)
    """Number of documents to retrieve."""

    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    """Minimum similarity score for retrieval results."""

    cache_enabled: bool = True
    """Whether to cache embedding results."""


class EvaluationSettings(BaseSettings):
    """Evaluation pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="WUYA_EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cudos_threshold: float = Field(default=3.0, ge=0.0, le=5.0)
    """CUDOS gate threshold. Papers below this are rejected."""

    dea_enabled: bool = True
    """Whether DEA analysis is enabled."""

    dea_min_references: int = Field(default=50, ge=1)
    """Minimum reference papers required for DEA analysis."""

    frontier_enabled: bool = True
    """Whether frontier discovery is enabled."""

    parallel_evaluation: bool = True
    """Whether to run sub-agents in parallel."""

    max_concurrent_agents: int = Field(default=4, ge=1, le=10)
    """Maximum number of concurrent sub-agent evaluations."""


class ServerSettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="WUYA_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    """Server bind host."""

    port: int = Field(default=8000, ge=1, le=65535)
    """Server bind port."""

    workers: int = Field(default=1, ge=1)
    """Number of worker processes."""

    cors_origins: list[str] = Field(default=["*"])
    """Allowed CORS origins."""

    api_prefix: str = "/api/v1"
    """API URL prefix."""


class Settings(BaseSettings):
    """
    Top-level application settings.

    Loads configuration from environment variables and .env file.
    Environment variables take precedence over .env file values.

    Environment precedence:
        1. Explicit environment variables
        2. .env file
        3. Defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="WUYA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "WuYa Agents"
    """Application name."""

    environment: Environment = Environment.DEVELOPMENT
    """Application environment."""

    debug: bool = False
    """Debug mode (enables verbose logging)."""

    log_level: LogLevel = LogLevel.INFO
    """Logging level."""

    config_path: Optional[str] = None
    """Path to additional configuration file."""

    # --- Sub-configurations ---
    llm: LLMSettings = Field(default_factory=LLMSettings)
    """LLM client settings."""

    rag: RAGSettings = Field(default_factory=RAGSettings)
    """RAG settings."""

    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    """Evaluation pipeline settings."""

    server: ServerSettings = Field(default_factory=ServerSettings)
    """API server settings."""

    @field_validator("debug")
    @classmethod
    def debug_sets_log_level(cls, v: bool, info) -> dict:
        """When debug=True, override log_level to DEBUG unless explicitly set."""
        return v

    def get_logging_config(self) -> dict:
        """Return a logging configuration dict suitable for dictConfig."""
        level = LogLevel.DEBUG if self.debug else self.log_level
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "detailed": {
                    "format": (
                        "%(asctime)s [%(levelname)s] %(name)s "
                        "%(filename)s:%(lineno)d: %(message)s"
                    ),
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "detailed" if self.debug else "standard",
                    "level": level.value,
                },
            },
            "root": {
                "handlers": ["console"],
                "level": level.value,
            },
            "loggers": {
                "wuya_agents": {
                    "handlers": ["console"],
                    "level": level.value,
                    "propagate": False,
                },
                # Silence noisy third-party loggers
                "aiohttp": {
                    "handlers": ["console"],
                    "level": logging.WARNING,
                    "propagate": False,
                },
                "chromadb": {
                    "handlers": ["console"],
                    "level": logging.WARNING,
                    "propagate": False,
                },
            },
        }

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    def is_testing(self) -> bool:
        """Check if running in test environment."""
        return self.environment == Environment.TESTING


@lru_cache()
def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Get cached application settings.

    Settings are loaded once and cached for the lifetime of the process.
    Pass config_path to load from a specific .env file.

    Args:
        config_path: Optional path to a .env file.

    Returns:
        Cached Settings instance.
    """
    if config_path:
        return Settings(_env_file=config_path)
    return Settings()


def reset_settings() -> None:
    """Reset the cached settings. Useful for testing."""
    get_settings.cache_clear()
