"""
Retry strategies for WuYa LLM Client.

Provides fine-grained exception types and an exponential backoff retry
decorator with jitter to handle transient API failures gracefully.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Any, Callable, Optional, Set, Type, TypeVar

from .config import RetryConfig

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Exception Hierarchy
# =============================================================================


class LLMError(Exception):
    """Base exception for all LLM client errors."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        model: str = "",
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.request_id = request_id
        detail = f" [{provider}" if provider else " ["
        detail += f" / {model}" if model else ""
        detail += "]"
        if status_code is not None:
            detail += f" (HTTP {status_code})"
        if request_id:
            detail += f" request_id={request_id}"
        super().__init__(f"{message}{detail}")


class RateLimitError(LLMError):
    """Raised when API rate limit is exceeded (HTTP 429).

    This error is retryable — the client should back off and retry.
    """

    def __init__(self, message: str = "Rate limit exceeded", **kwargs: Any):
        kwargs.setdefault("status_code", 429)
        super().__init__(message, **kwargs)


class TimeoutError(LLMError):
    """Raised when an API request times out (HTTP 408 or connection timeout).

    This error is retryable.
    """

    def __init__(self, message: str = "Request timed out", **kwargs: Any):
        kwargs.setdefault("status_code", 408)
        super().__init__(message, **kwargs)


class ServiceUnavailableError(LLMError):
    """Raised when the API service is unavailable (HTTP 503/529).

    This error is retryable.
    """

    def __init__(self, message: str = "Service unavailable", **kwargs: Any):
        kwargs.setdefault("status_code", 503)
        super().__init__(message, **kwargs)


class ConnectionError(LLMError):
    """Raised when a network connection error occurs.

    This error is retryable.
    """

    def __init__(self, message: str = "Connection error", **kwargs: Any):
        super().__init__(message, **kwargs)


class AuthenticationError(LLMError):
    """Raised when authentication fails (HTTP 401).

    This error is NOT retryable — check your API key.
    """

    def __init__(self, message: str = "Authentication failed", **kwargs: Any):
        kwargs.setdefault("status_code", 401)
        super().__init__(message, **kwargs)


class BadRequestError(LLMError):
    """Raised when the request is malformed (HTTP 400).

    This error is NOT retryable — fix the request parameters.
    """

    def __init__(self, message: str = "Bad request", **kwargs: Any):
        kwargs.setdefault("status_code", 400)
        super().__init__(message, **kwargs)


class ContentFilterError(LLMError):
    """Raised when content is filtered by the provider's safety system.

    This error is NOT retryable — modify the input content.
    """

    def __init__(self, message: str = "Content filtered by safety system", **kwargs: Any):
        super().__init__(message, **kwargs)


# Exceptions that should trigger a retry
RETRYABLE_ERRORS: Set[Type[LLMError]] = {
    RateLimitError,
    TimeoutError,
    ServiceUnavailableError,
    ConnectionError,
}

# HTTP status codes that map to retryable errors
RETRYABLE_STATUS_CODES: Set[int] = {408, 429, 500, 502, 503, 529}

# HTTP status codes that map to non-retryable errors
NON_RETRYABLE_STATUS_CODES: Set[int] = {400, 401, 403, 404, 422}


def classify_http_error(
    status_code: int,
    body: str = "",
    provider: str = "",
    model: str = "",
    request_id: Optional[str] = None,
) -> LLMError:
    """
    Classify an HTTP status code into the appropriate LLMError subclass.

    Args:
        status_code: HTTP status code from the API response.
        body: Response body for additional context.
        provider: Provider name (e.g., "openai", "anthropic").
        model: Model name for error context.
        request_id: Request ID for debugging.

    Returns:
        Appropriate LLMError subclass instance.

    Raises:
        ValueError: If the status code is not recognized.
    """
    kwargs = {
        "provider": provider,
        "model": model,
        "status_code": status_code,
        "request_id": request_id,
    }

    if status_code == 400:
        return BadRequestError(f"Bad request: {body}", **kwargs)
    elif status_code == 401:
        return AuthenticationError(f"Authentication failed: {body}", **kwargs)
    elif status_code == 408:
        return TimeoutError(**kwargs)
    elif status_code == 429:
        return RateLimitError(body or "Rate limit exceeded", **kwargs)
    elif status_code == 503:
        return ServiceUnavailableError(f"Service unavailable: {body}", **kwargs)
    elif status_code == 529:
        # Anthropic-specific: overloaded
        return ServiceUnavailableError("Anthropic overloaded", **kwargs)
    elif status_code in (500, 502):
        return ServiceUnavailableError(
            f"Server error ({status_code}): {body}", **kwargs
        )
    else:
        return LLMError(
            f"Unexpected HTTP {status_code}: {body}",
            **kwargs,
        )


# =============================================================================
# Retry Decorator
# =============================================================================


def _calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate the delay before the next retry attempt.

    Uses exponential backoff: delay = base_delay * backoff_factor^attempt
    Capped at max_delay. Adds random jitter if enabled.

    Args:
        attempt: The retry attempt number (0-indexed).
        config: Retry configuration.

    Returns:
        Delay in seconds.
    """
    delay = config.base_delay * (config.backoff_factor ** attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add uniform jitter: [0.5 * delay, 1.5 * delay]
        delay = delay * (0.5 + random.random())

    return delay


def retry_with_backoff(
    retry_config: Optional[RetryConfig] = None,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable[[F], F]:
    """
    Decorator for retrying async functions with exponential backoff and jitter.

    Args:
        retry_config: Retry configuration. Uses defaults if None.
        retryable_exceptions: Set of exception types that should trigger a retry.
            Defaults to all RETRYABLE_ERRORS from this module.
        on_retry: Optional callback invoked before each retry.
            Receives (attempt_number, exception, delay_seconds).

    Returns:
        Decorated function.

    Example::

        @retry_with_backoff(retry_config=RetryConfig(max_retries=5))
        async def call_api(prompt: str) -> str:
            ...
    """
    config = retry_config or RetryConfig()
    exc_set = retryable_exceptions or set(RETRYABLE_ERRORS)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(exc_set) as exc:
                    last_exception = exc

                    if attempt >= config.max_retries:
                        logger.error(
                            "Max retries (%d) exceeded for %s: %s",
                            config.max_retries,
                            func.__qualname__,
                            exc,
                        )
                        raise

                    delay = _calculate_delay(attempt, config)
                    logger.warning(
                        "Retry %d/%d for %s in %.2fs: %s",
                        attempt + 1,
                        config.max_retries,
                        func.__qualname__,
                        delay,
                        exc,
                    )

                    if on_retry is not None:
                        on_retry(attempt + 1, exc, delay)

                    await asyncio.sleep(delay)
                except Exception:
                    # Non-retryable exceptions are re-raised immediately
                    raise

            # Should not reach here, but just in case
            if last_exception is not None:
                raise last_exception

            raise RuntimeError("Unexpected state in retry logic")

        return wrapper  # type: ignore[return-value]

    return decorator
