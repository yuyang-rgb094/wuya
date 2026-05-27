"""
Anthropic provider implementation for WuYa LLM Client.

Calls the Anthropic Messages API using aiohttp.
Supports system prompts (top-level field), streaming (SSE),
and Anthropic-specific error handling (e.g., 529 Overloaded).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp

from .base_provider import BaseLLMProvider, LLMResponse
from .config import LLMConfig
from .retry import (
    ConnectionError as LLMConnectionError,
    LLMError,
    TimeoutError as LLMTimeoutError,
    classify_http_error,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)

# Anthropic API constants
DEFAULT_API_BASE = "https://api.anthropic.com/v1"
MESSAGES_PATH = "/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Messages API provider.

    Uses aiohttp for async HTTP requests. Supports both standard
    and streaming responses. Handles Anthropic-specific features:
        - System prompt as a top-level field (not in messages)
        - Anthropic-Version header
        - 529 Overloaded error code

    Example::

        provider = AnthropicProvider(
            api_key="sk-ant-...",
            model="claude-3-5-sonnet-20241022",
        )
        response = await provider.generate("Hello, world!")
        print(response.content)
        await provider.close()
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        *,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = 120.0,
        max_retries: int = 3,
        anthropic_version: str = ANTHROPIC_VERSION,
    ) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model: Model identifier (e.g., 'claude-3-5-sonnet-20241022').
            api_base: API base URL. Override for proxies.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum retry attempts for transient errors.
            anthropic_version: Anthropic API version header value.
        """
        self._api_key = api_key
        self._model = model
        self._api_base = api_base.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries
        self._anthropic_version = anthropic_version
        self._session: Optional[aiohttp.ClientSession] = None

    @classmethod
    def from_config(cls, config: LLMConfig) -> AnthropicProvider:
        """Create an AnthropicProvider from an LLMConfig."""
        return cls(
            api_key=config.anthropic_api_key or "",
            model=config.anthropic_model,
            api_base=config.api_base_url or DEFAULT_API_BASE,
            timeout=config.request_timeout,
            max_retries=config.retry.max_retries,
        )

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self._session is None or self._session.closed:
            headers = {
                "x-api-key": self._api_key,
                "anthropic-version": self._anthropic_version,
                "Content-Type": "application/json",
            }
            self._session = aiohttp.ClientSession(
                base_url=self._api_base,
                headers=headers,
                timeout=self._timeout,
            )
        return self._session

    def _build_messages(
        self,
        prompt: str,
    ) -> list[Dict[str, str]]:
        """
        Build the messages array for the Anthropic API.

        Note: Anthropic uses a top-level 'system' field for system prompts,
        not a message with role 'system'. System prompt is passed separately.
        """
        return [{"role": "user", "content": prompt}]

    def _build_body(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Build the request body."""
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if system_prompt:
            body["system"] = system_prompt
        return body

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Parse the Anthropic API response into LLMResponse."""
        # Anthropic returns content as a list of content blocks
        content_blocks = data.get("content", [])
        text_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        content = "\n".join(text_parts)

        usage = data.get("usage", {})
        stop_reason = data.get("stop_reason", "")

        # Map Anthropic stop reasons to common format
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "max_tokens",
            "stop_sequence": "stop",
        }
        finish_reason = finish_reason_map.get(stop_reason, stop_reason)

        return LLMResponse(
            content=content,
            model=data.get("model", self._model),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            finish_reason=finish_reason,
            provider="anthropic",
            request_id=data.get("id", ""),
            metadata={
                "stop_reason": stop_reason,
                "type": data.get("type", ""),
            },
        )

    def _extract_request_id(self, response: aiohttp.ClientResponse) -> str:
        """Extract the request ID from response headers."""
        return response.headers.get("anthropic-request-id", "") or response.headers.get(
            "x-request-id", ""
        )

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a completion from Anthropic."""
        messages = self._build_messages(prompt)
        body = self._build_body(messages, temperature, max_tokens, system_prompt, stream=False)

        @retry_with_backoff()
        async def _do_request() -> LLMResponse:
            session = self._get_session()
            try:
                async with session.post(
                    MESSAGES_PATH,
                    json=body,
                ) as resp:
                    request_id = self._extract_request_id(resp)
                    body_text = await resp.text()

                    if resp.status != 200:
                        raise classify_http_error(
                            resp.status,
                            body=body_text[:500],
                            provider="anthropic",
                            model=self._model,
                            request_id=request_id,
                        )

                    data = json.loads(body_text)
                    return self._parse_response(data)

            except aiohttp.ClientError as exc:
                if isinstance(exc, aiohttp.ClientConnectorError):
                    raise LLMConnectionError(
                        f"Connection failed: {exc}",
                        provider="anthropic",
                        model=self._model,
                    ) from exc
                if isinstance(exc, asyncio.TimeoutError):
                    raise LLMTimeoutError(
                        f"Request timed out: {exc}",
                        provider="anthropic",
                        model=self._model,
                    ) from exc
                raise LLMConnectionError(
                    f"HTTP client error: {exc}",
                    provider="anthropic",
                    model=self._model,
                ) from exc

        return await _do_request()

    async def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion from Anthropic.

        Yields incremental text deltas as SSE events arrive.
        Anthropic streaming events use 'content_block_delta' for text chunks.
        """
        messages = self._build_messages(prompt)
        body = self._build_body(messages, temperature, max_tokens, system_prompt, stream=True)

        session = self._get_session()
        try:
            async with session.post(
                MESSAGES_PATH,
                json=body,
            ) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    raise classify_http_error(
                        resp.status,
                        body=body_text[:500],
                        provider="anthropic",
                        model=self._model,
                        request_id=self._extract_request_id(resp),
                    )

                # Parse SSE stream
                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()

                    if not decoded:
                        continue

                    if decoded.startswith("data: "):
                        json_str = decoded[6:]
                        try:
                            chunk = json.loads(json_str)
                            event_type = chunk.get("type", "")

                            if event_type == "content_block_delta":
                                delta = chunk.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        yield text

                            elif event_type == "message_stop":
                                break

                        except (json.JSONDecodeError, KeyError, IndexError):
                            logger.debug("Skipping malformed SSE chunk: %s", json_str)
                            continue

        except aiohttp.ClientError as exc:
            if isinstance(exc, aiohttp.ClientConnectorError):
                raise LLMConnectionError(
                    f"Connection failed: {exc}",
                    provider="anthropic",
                    model=self._model,
                ) from exc
            raise LLMConnectionError(
                f"HTTP client error: {exc}",
                provider="anthropic",
                model=self._model,
            ) from exc

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self._model!r})"
