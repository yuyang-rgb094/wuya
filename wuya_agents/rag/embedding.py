"""
RAG Embedding Provider Module

Provides text embedding capabilities with caching support.
Supports OpenAI embeddings and mock embeddings for testing.

Key Design:
    - Abstract base class for provider extensibility
    - Embedding cache to avoid duplicate API calls
    - Batch embedding support for efficiency
    - Cost tracking and rate limiting hooks

Author: WuYa Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import hashlib
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from an embedding operation."""
    text: str
    vector: List[float]
    model: str
    dimensions: int
    cached: bool = False
    api_call_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "vector": self.vector,
            "model": self.model,
            "dimensions": self.dimensions,
            "cached": self.cached,
        }


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All embedding providers must implement:
    1. embed(text) -> List[float]
    2. embed_batch(texts) -> List[List[float]]
    3. get_dimensions() -> int
    """

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Embed a single text into a vector."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts into vectors."""
        ...

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the embedding model."""
        ...


class EmbeddingCache:
    """
    In-memory cache for embedding results.

    Uses text hash as key to avoid duplicate API calls for identical texts.
    Tracks cache hit rate and total API call savings.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of entries to store.
        """
        self._cache: Dict[str, List[float]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate a stable hash for text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding if available."""
        key = self._hash_text(text)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, vector: List[float]) -> None:
        """Store embedding in cache with eviction policy."""
        if len(self._cache) >= self._max_size:
            # Simple FIFO eviction: remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[self._hash_text(text)] = vector

    def get_batch(self, texts: List[str]) -> tuple[List[Optional[List[float]]], List[int]]:
        """
        Retrieve cached embeddings for a batch of texts.

        Returns:
            Tuple of (results, miss_indices) where results contains None for misses
            and miss_indices indicates which texts need to be embedded.
        """
        results = []
        miss_indices = []
        for i, text in enumerate(texts):
            cached = self.get(text)
            results.append(cached)
            if cached is None:
                miss_indices.append(i)
        return results, miss_indices

    def put_batch(self, texts: List[str], vectors: List[List[float]]) -> None:
        """Store multiple embeddings in cache."""
        for text, vector in zip(texts, vectors):
            self.put(text, vector)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI Embedding Provider using text-embedding-3-small.

    Features:
        - Uses OpenAI API for high-quality embeddings
        - Embedding cache to reduce API costs
        - Batch embedding support (up to 2048 texts per request)
        - API call counting for cost tracking
        - Rate limiting awareness

    Example::

        provider = OpenAIEmbeddingProvider(api_key="sk-...")
        vector = await provider.embed("Hello world")
        print(f"Dimensions: {provider.get_dimensions()}")

    Example (with cache)::

        provider = OpenAIEmbeddingProvider(
            api_key="sk-...",
            cache=EmbeddingCache(max_size=5000)
        )
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSIONS = 1536
    MAX_BATCH_SIZE = 2048

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        cache: Optional[EmbeddingCache] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize OpenAI Embedding Provider.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Embedding model name. Defaults to text-embedding-3-small.
            dimensions: Output dimensions. Defaults to 1536 for text-embedding-3-small.
            cache: Optional embedding cache.
            max_retries: Maximum retry attempts for API calls.
            retry_delay: Delay between retries in seconds.
        """
        self._api_key = api_key
        self._model = model or self.DEFAULT_MODEL
        self._dimensions = dimensions or self.DEFAULT_DIMENSIONS
        self._cache = cache or EmbeddingCache()
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._api_call_count = 0
        self._total_tokens = 0
        self._client = None

    def _get_client(self):
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIEmbeddingProvider. "
                    "Install it with: pip install openai"
                )
        return self._client

    async def embed(self, text: str) -> List[float]:
        """
        Embed a single text into a vector.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        # Check cache first
        cached = self._cache.get(text)
        if cached is not None:
            logger.debug(f"Embedding cache hit for text (len={len(text)})")
            return cached

        # Call API
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts into vectors.

        Automatically splits large batches into chunks of MAX_BATCH_SIZE.
        Uses cache for previously embedded texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Check cache for all texts
        cached_results, miss_indices = self._cache.get_batch(texts)

        if not miss_indices:
            logger.debug(f"All {len(texts)} embeddings served from cache")
            return cached_results  # type: ignore

        # Embed only missing texts
        miss_texts = [texts[i] for i in miss_indices]
        logger.info(
            f"Embedding {len(miss_texts)} texts "
            f"({len(texts) - len(miss_indices)} from cache)"
        )

        # Split into batches
        all_new_vectors = []
        for chunk_start in range(0, len(miss_texts), self.MAX_BATCH_SIZE):
            chunk = miss_texts[chunk_start:chunk_start + self.MAX_BATCH_SIZE]
            vectors = await self._call_api(chunk)
            all_new_vectors.extend(vectors)

        # Update cache
        self._cache.put_batch(miss_texts, all_new_vectors)

        # Merge results
        results = list(cached_results)
        for i, idx in enumerate(miss_indices):
            results[idx] = all_new_vectors[i]

        return results  # type: ignore

    async def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embedding API with retry logic."""
        import asyncio

        client = self._get_client()

        for attempt in range(self._max_retries):
            try:
                response = await client.embeddings.create(
                    input=texts,
                    model=self._model,
                    dimensions=self._dimensions,
                )

                self._api_call_count += 1
                self._total_tokens += response.usage.total_tokens

                vectors = [item.embedding for item in response.data]
                logger.info(
                    f"OpenAI embedding API call: {len(texts)} texts, "
                    f"{response.usage.total_tokens} tokens"
                )
                return vectors

            except Exception as e:
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"OpenAI embedding API error (attempt {attempt + 1}): {e}. "
                        f"Retrying in {self._retry_delay}s..."
                    )
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                else:
                    logger.error(f"OpenAI embedding API failed after {self._max_retries} attempts: {e}")
                    raise RuntimeError(f"OpenAI embedding API failed: {e}") from e

        raise RuntimeError("Unreachable code in _call_api")

    def get_dimensions(self) -> int:
        """Return embedding dimensionality."""
        return self._dimensions

    def get_model_name(self) -> str:
        """Return the model name."""
        return self._model

    @property
    def api_call_count(self) -> int:
        """Total number of API calls made."""
        return self._api_call_count

    @property
    def total_tokens(self) -> int:
        """Total tokens processed."""
        return self._total_tokens

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Combined cache and API statistics."""
        return {
            "model": self._model,
            "dimensions": self._dimensions,
            "api_call_count": self._api_call_count,
            "total_tokens": self._total_tokens,
            "cache": self._cache.stats,
        }


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock Embedding Provider for testing and development.

    Generates deterministic pseudo-random vectors based on text hash.
    No API calls needed - suitable for unit tests and local development.

    Example::

        provider = MockEmbeddingProvider(dimensions=128)
        vector = await provider.embed("test text")
        # Same text always produces same vector
        assert vector == await provider.embed("test text")
    """

    def __init__(
        self,
        dimensions: int = 128,
        cache: Optional[EmbeddingCache] = None,
        seed: int = 42,
    ):
        """
        Initialize Mock Embedding Provider.

        Args:
            dimensions: Dimensionality of mock vectors.
            cache: Optional embedding cache.
            seed: Random seed for reproducibility.
        """
        self._dimensions = dimensions
        self._cache = cache or EmbeddingCache()
        self._seed = seed

    def _generate_vector(self, text: str) -> List[float]:
        """Generate a deterministic vector from text hash."""
        import hashlib
        import struct
        import math

        # Generate enough hash bytes for all dimensions
        full_hash = hashlib.sha256(
            (text + str(self._seed)).encode("utf-8")
        ).digest()

        # Extend hash by re-hashing if needed for large dimensions
        extended_hash = bytearray(full_hash)
        while len(extended_hash) < self._dimensions * 4:
            extended_hash.extend(hashlib.sha256(
                extended_hash[-32:] + text.encode("utf-8")
            ).digest())

        # Generate deterministic float values from hash bytes
        raw_bytes = bytes(extended_hash[:self._dimensions * 4])
        values = struct.unpack(f'<{self._dimensions}f', raw_bytes)

        # Convert to clean floats, replacing NaN/Inf with 0
        vector = []
        for v in values:
            if math.isnan(v) or math.isinf(v):
                vector.append(0.0)
            else:
                # Normalize to [-1, 1] range
                vector.append(v / (2**31))

        # Normalize to unit vector for cosine similarity
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [v / magnitude for v in vector]

        return vector

    async def embed(self, text: str) -> List[float]:
        """Embed a single text (mock, no API call)."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        vector = self._generate_vector(text)
        self._cache.put(text, vector)
        return vector

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (mock, no API call)."""
        return [await self.embed(text) for text in texts]

    def get_dimensions(self) -> int:
        """Return mock dimensionality."""
        return self._dimensions

    def get_model_name(self) -> str:
        """Return mock model name."""
        return f"mock-embedding-{self._dimensions}d"


# =============================================================================
# Factory Functions
# =============================================================================

def create_embedding_provider(
    provider_type: str = "mock",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    cache_size: int = 10000,
) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider.

    Args:
        provider_type: Type of provider ("openai" or "mock").
        api_key: API key for OpenAI provider.
        model: Model name for OpenAI provider.
        dimensions: Embedding dimensions.
        cache_size: Maximum cache entries.

    Returns:
        Configured EmbeddingProvider instance.

    Raises:
        ValueError: If provider_type is not recognized.
    """
    cache = EmbeddingCache(max_size=cache_size)

    if provider_type == "openai":
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            cache=cache,
        )
    elif provider_type == "mock":
        return MockEmbeddingProvider(
            dimensions=dimensions or 128,
            cache=cache,
        )
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Supported: 'openai', 'mock'"
        )
