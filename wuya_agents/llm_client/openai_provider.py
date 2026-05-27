"""
OpenAI provider implementation for WuYa LLM Client.

Calls the OpenAI Chat Completions API using aiohttp.
Supports system prompts, streaming (SSE), and error mapping.
"""

from __future__ import annotations

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

# Default API endpoint
DEFAULT_API_BASE = "https://api.openai.com/v1"
CHAT_COMPLETIONS_PATH = "/chat/completions"


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI Chat Completions API provider.

    Uses aiohttp for async HTTP requests. Supports both standard
    and streaming responses.

    Example::

        provider = OpenAIProvider(
            api_key="sk-...",
            model="gpt-4o",
        )
        response = await provider.generate("Hello, world!")
        print(response.content)
        await provider.close()
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        *,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model identifier (e.g., 'gpt-4o', 'gpt-4o-mini').
            api_base: API base URL. Override for proxies or Azure endpoints.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum retry attempts for transient errors.
        """
        self._api_key = api_key
        self._model = model
        self._api_base = api_base.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    @classmethod
    def from_config(cls, config: LLMConfig) -> OpenAIProvider:
        """Create an OpenAIProvider from an LLMConfig."""
        return cls(
            api_key=config.openai_api_key or "",
            model=config.openai_model,
            api_base=config.api_base_url or DEFAULT_API_BASE,
            timeout=config.request_timeout,
            max_retries=config.retry.max_retries,
        )

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
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
        system_prompt: Optional[str] = None,
    ) -> list[Dict[str, str]]:
        """Build the messages array for the API request."""
        messages: list[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_body(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Build the request body."""
        return {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Parse the OpenAI API response into LLMResponse."""
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=message["content"],
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
            provider="openai",
            request_id=data.get("id", ""),
            metadata={
                "system_fingerprint": data.get("system_fingerprint", ""),
                "created": data.get("created", 0),
            },
        )

    def _extract_request_id(self, response: aiohttp.ClientResponse) -> str:
        """Extract the request ID from response headers."""
        return response.headers.get("x-request-id", "")

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a completion from OpenAI."""
        messages = self._build_messages(prompt, system_prompt)
        body = self._build_body(messages, temperature, max_tokens, stream=False)

        @retry_with_backoff()
        async def _do_request() -> LLMResponse:
            session = self._get_session()
            try:
                async with session.post(
                    CHAT_COMPLETIONS_PATH,
                    json=body,
                ) as resp:
                    request_id = self._extract_request_id(resp)
                    body_text = await resp.text()

                    if resp.status != 200:
                        raise classify_http_error(
                            resp.status,
                            body=body_text[:500],
                            provider="openai",
                            model=self._model,
                            request_id=request_id,
                        )

                    data = json.loads(body_text)
                    return self._parse_response(data)

            except aiohttp.ClientError as exc:
                if isinstance(exc, aiohttp.ClientConnectorError):
                    raise LLMConnectionError(
                        f"Connection failed: {exc}",
                        provider="openai",
                        model=self._model,
                    ) from exc
                if isinstance(exc, asyncio.TimeoutError):
                    raise LLMTimeoutError(
                        f"Request timed out: {exc}",
                        provider="openai",
                        model=self._model,
                    ) from exc
                raise LLMConnectionError(
                    f"HTTP client error: {exc}",
                    provider="openai",
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
        Generate a streaming completion from OpenAI.

        Yields incremental text deltas as SSE events arrive.
        """
        import asyncio

        messages = self._build_messages(prompt, system_prompt)
        body = self._build_body(messages, temperature, max_tokens, stream=True)

        session = self._get_session()
        try:
            async with session.post(
                CHAT_COMPLETIONS_PATH,
                json=body,
            ) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    raise classify_http_error(
                        resp.status,
                        body=body_text[:500],
                        provider="openai",
                        model=self._model,
                        request_id=self._extract_request_id(resp),
                    )

                # Parse SSE stream
                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()

                    if not decoded:
                        continue

                    if decoded == "data: [DONE]":
                        break

                    if decoded.startswith("data: "):
                        json_str = decoded[6:]
                        try:
                            chunk = json.loads(json_str)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            logger.debug("Skipping malformed SSE chunk: %s", json_str)
                            continue

        except aiohttp.ClientError as exc:
            if isinstance(exc, aiohttp.ClientConnectorError):
                raise LLMConnectionError(
                    f"Connection failed: {exc}",
                    provider="openai",
                    model=self._model,
                ) from exc
            raise LLMConnectionError(
                f"HTTP client error: {exc}",
                provider="openai",
                model=self._model,
            ) from exc

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self._model!r})"


# Need asyncio for TimeoutError check in generate()
import asyncio
