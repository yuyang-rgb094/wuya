"""
Token counting for WuYa LLM Client.

Provides heuristic token estimation without external dependencies (tiktoken).
Tracks cumulative token usage and estimates API costs.

Heuristic rules (approximate):
    - Chinese characters: ~1.5 tokens each
    - English words:       ~1.3 tokens each
    - Other characters:    ~0.5 tokens each

These estimates are intentionally conservative. For production use
with precise counting, consider integrating the tiktoken library.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


# =============================================================================
# Cost per 1K tokens (USD) — approximate pricing as of 2024
# =============================================================================

# fmt: off
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {
        "input": 0.0025,    # $2.50 / 1M input tokens
        "output": 0.01,     # $10.00 / 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.00015,
        "output": 0.0006,
    },
    "gpt-4-turbo": {
        "input": 0.01,
        "output": 0.03,
    },
    "gpt-4": {
        "input": 0.03,
        "output": 0.06,
    },
    "gpt-3.5-turbo": {
        "input": 0.0005,
        "output": 0.0015,
    },
    # Anthropic models
    "claude-3-5-sonnet-20241022": {
        "input": 0.003,     # $3 / 1M input tokens
        "output": 0.015,    # $15 / 1M output tokens
    },
    "claude-3-opus-20240229": {
        "input": 0.015,
        "output": 0.075,
    },
    "claude-3-haiku-20240307": {
        "input": 0.00025,
        "output": 0.00125,
    },
    "claude-3-5-haiku-20241022": {
        "input": 0.001,
        "output": 0.005,
    },
}
# fmt: on

# Default pricing for unknown models (conservative estimate)
_DEFAULT_PRICING = {"input": 0.01, "output": 0.03}


# =============================================================================
# Heuristic Token Estimation
# =============================================================================

# Precompiled regex patterns for token estimation
_CJK_PATTERN = re.compile(
    r"[\u4e00-\u9fff"     # CJK Unified Ideographs
    r"\u3400-\u4dbf"      # CJK Extension A
    r"\u3000-\u303f"      # CJK Symbols and Punctuation
    r"\uff00-\uffef]"      # Halfwidth and Fullwidth Forms
)
_WORD_PATTERN = re.compile(r"[a-zA-Z]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string using heuristic rules.

    Rules:
        - Each CJK character counts as ~1.5 tokens
        - Each English word counts as ~1.3 tokens
        - Whitespace is ignored
        - Other characters (punctuation, numbers, etc.) count as ~0.5 tokens

    This is a rough approximation. For exact counts, use tiktoken.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count (rounded up to nearest integer).
    """
    if not text:
        return 0

    total = 0.0

    # Count CJK characters
    cjk_chars = _CJK_PATTERN.findall(text)
    total += len(cjk_chars) * 1.5

    # Remove CJK characters and whitespace, then count English words
    text_without_cjk = _CJK_PATTERN.sub("", text)
    text_clean = _WHITESPACE_PATTERN.sub("", text_without_cjk)

    english_words = _WORD_PATTERN.findall(text_clean)
    total += len(english_words) * 1.3

    # Remaining characters (punctuation, numbers, symbols, etc.)
    remaining = _WORD_PATTERN.sub("", text_clean)
    total += len(remaining) * 0.5

    return max(1, int(total + 0.5))  # Round up, minimum 1


# =============================================================================
# Token Usage Tracking
# =============================================================================


@dataclass
class TokenUsage:
    """
    Cumulative token usage statistics.

    Tracks total tokens consumed across all API calls, broken down
    by input (prompt) and output (completion) tokens.
    """

    total_input_tokens: int = 0
    """Cumulative input (prompt) tokens."""

    total_output_tokens: int = 0
    """Cumulative output (completion) tokens."""

    total_requests: int = 0
    """Total number of API requests made."""

    start_time: float = field(default_factory=time.time)
    """Timestamp when tracking started."""

    _per_model_input: Dict[str, int] = field(default_factory=dict)
    _per_model_output: Dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
    ) -> None:
        """
        Record a single API call's token usage.

        Args:
            input_tokens: Number of tokens in the prompt.
            output_tokens: Number of tokens in the completion.
            model: Model name for per-model tracking.
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

        if model:
            self._per_model_input[model] = (
                self._per_model_input.get(model, 0) + input_tokens
            )
            self._per_model_output[model] = (
                self._per_model_output.get(model, 0) + output_tokens
            )

    def estimate_cost(self, model: str = "") -> float:
        """
        Estimate the total cost in USD based on recorded usage.

        Args:
            model: Model name to look up pricing. If empty, uses default pricing.

        Returns:
            Estimated cost in USD.
        """
        pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
        input_cost = (self.total_input_tokens / 1000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def estimate_cost_per_model(self) -> Dict[str, float]:
        """
        Estimate cost broken down by model.

        Returns:
            Dict mapping model name to estimated cost in USD.
        """
        costs: Dict[str, float] = {}
        all_models = set(self._per_model_input.keys()) | set(
            self._per_model_output.keys()
        )
        for model in all_models:
            pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
            inp = self._per_model_input.get(model, 0)
            out = self._per_model_output.get(model, 0)
            costs[model] = (inp / 1000) * pricing["input"] + (out / 1000) * pricing["output"]
        return costs

    def reset(self) -> None:
        """Reset all usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.start_time = time.time()
        self._per_model_input.clear()
        self._per_model_output.clear()

    def summary(self) -> str:
        """Return a human-readable summary of token usage."""
        elapsed = time.time() - self.start_time
        minutes = elapsed / 60 if elapsed > 0 else 1
        tpm = self.total_tokens / minutes

        lines = [
            f"Token Usage Summary",
            f"  Total requests:  {self.total_requests}",
            f"  Input tokens:    {self.total_input_tokens:,}",
            f"  Output tokens:   {self.total_output_tokens:,}",
            f"  Total tokens:    {self.total_tokens:,}",
            f"  Avg tokens/req:  {self.total_tokens / max(1, self.total_requests):,.0f}",
            f"  Throughput:      {tpm:,.0f} tokens/min",
        ]
        return "\n".join(lines)


# =============================================================================
# Token Counter
# =============================================================================


class TokenCounter:
    """
    Token estimation and usage tracking utility.

    Provides both static estimation methods and instance-level
    cumulative usage tracking.

    Example::

        counter = TokenCounter()

        # Estimate tokens for a prompt
        prompt_tokens = counter.estimate("Hello, world!")
        print(f"Estimated {prompt_tokens} tokens")

        # Record actual usage from an API response
        counter.record_usage(input_tokens=150, output_tokens=80, model="gpt-4o")

        # Get usage statistics
        print(counter.usage.summary())
        print(f"Estimated cost: ${counter.usage.estimate_cost('gpt-4o'):.4f}")
    """

    def __init__(self) -> None:
        """Initialize a new TokenCounter with empty usage tracking."""
        self.usage = TokenUsage()

    @staticmethod
    def estimate(text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        return estimate_tokens(text)

    @staticmethod
    def estimate_messages(
        messages: list[dict[str, str]],
    ) -> int:
        """
        Estimate tokens for a list of chat messages.

        Each message is a dict with "role" and "content" keys.
        Adds ~4 tokens per message for role/formatting overhead.

        Args:
            messages: List of message dicts with "role" and "content".

        Returns:
            Estimated total token count.
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            total += estimate_tokens(content)
            total += estimate_tokens(role)
            total += 4  # Overhead per message (role, delimiters, etc.)
        total += 3  # Conversation overhead (priming, assistant prefix)
        return total

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
    ) -> None:
        """
        Record token usage from an API response.

        Args:
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens generated.
            model: Model name for cost estimation.
        """
        self.usage.record(input_tokens, output_tokens, model)

    def get_usage(self) -> TokenUsage:
        """
        Get the current usage statistics.

        Returns:
            TokenUsage instance with cumulative statistics.
        """
        return self.usage

    def reset(self) -> None:
        """Reset all usage counters."""
        self.usage.reset()
