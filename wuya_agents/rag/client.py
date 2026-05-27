"""
RAG Client Module

Implements the RAGClient Protocol from base.py, combining
EmbeddingProvider and VectorStore for semantic retrieval.

Key Design:
    - Implements RAGClient Protocol (retrieve method)
    - Composes EmbeddingProvider + VectorStore
    - Supports metadata filtering (discipline, year, source, etc.)
    - Optional hybrid retrieval (semantic + keyword matching)
    - RetrievalResult dataclass for structured output

Author: WuYa Team
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import re
import math

from .embedding import EmbeddingProvider, MockEmbeddingProvider, create_embedding_provider
from .vector_store import VectorStore, Document, SearchResult, InMemoryVectorStore, create_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    A single retrieval result from RAG.

    Attributes:
        content: Retrieved text content.
        score: Relevance score (0.0 to 1.0).
        source: Source identifier (document ID).
        metadata: Document metadata.
        citation: Formatted citation string.
        rank: Rank in results (0-based).
    """
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    citation: str = ""
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": round(self.score, 4),
            "source": self.source,
            "metadata": self.metadata,
            "citation": self.citation,
            "rank": self.rank,
        }


class HybridRetriever:
    """
    Hybrid retriever combining semantic similarity with keyword matching.

    Uses a weighted combination of:
    - Semantic score (from vector similarity)
    - Keyword score (from BM25-like term frequency matching)

    The hybrid score is: alpha * semantic_score + (1 - alpha) * keyword_score
    """

    def __init__(self, alpha: float = 0.7):
        """
        Initialize hybrid retriever.

        Args:
            alpha: Weight for semantic score (0.0 to 1.0).
                   Keyword score weight is (1 - alpha).
        """
        self.alpha = alpha

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r'\w+', text.lower())

    def _keyword_score(self, query: str, content: str) -> float:
        """
        Compute keyword relevance score (BM25-like).

        Uses term frequency with inverse document frequency approximation.
        """
        query_tokens = set(self._tokenize(query))
        content_tokens = self._tokenize(content)

        if not query_tokens or not content_tokens:
            return 0.0

        content_len = len(content_tokens)
        matches = sum(1 for t in query_tokens if t in content_tokens)
        coverage = matches / len(query_tokens)

        # Term frequency factor
        tf = matches / max(content_len, 1)

        # Simple scoring: coverage * normalized TF
        score = coverage * min(tf * 100, 1.0)
        return min(score, 1.0)

    def compute_hybrid_score(
        self,
        semantic_score: float,
        query: str,
        content: str,
    ) -> float:
        """
        Compute hybrid score combining semantic and keyword scores.

        Args:
            semantic_score: Score from vector similarity (0.0 to 1.0).
            query: Original query text.
            content: Document content.

        Returns:
            Combined hybrid score (0.0 to 1.0).
        """
        keyword_score = self._keyword_score(query, content)
        return self.alpha * semantic_score + (1 - self.alpha) * keyword_score


class RAGClientImpl:
    """
    RAG Client implementation combining EmbeddingProvider and VectorStore.

    Implements the RAGClient Protocol from base.py:
        async def retrieve(query, context, top_k) -> List[Dict[str, Any]]

    Features:
        - Semantic retrieval via embedding similarity
        - Metadata filtering (discipline, year, source, etc.)
        - Optional hybrid retrieval (semantic + keyword)
        - Document management (add, delete, update)
        - Citation formatting

    Example::

        # Create with mock providers for development
        rag = RAGClientImpl(
            embedding_provider=MockEmbeddingProvider(dimensions=128),
            vector_store=InMemoryVectorStore(),
        )

        # Add documents
        await rag.add_texts([
            ("doc1", "Machine learning for drug discovery", {"discipline": "cs"}),
            ("doc2", "Deep learning in medical imaging", {"discipline": "cs"}),
        ])

        # Retrieve
        results = await rag.retrieve("deep learning applications", top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content'][:80]}")

    Example (with filters)::

        results = await rag.retrieve(
            "methodology best practices",
            top_k=5,
            filters={"discipline": "physics", "year": 2024}
        )
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        hybrid_alpha: float = 0.7,
        enable_hybrid: bool = False,
        default_top_k: int = 5,
    ):
        """
        Initialize RAG Client.

        Args:
            embedding_provider: Provider for text embeddings.
                Defaults to MockEmbeddingProvider(dimensions=128).
            vector_store: Vector store for document storage.
                Defaults to InMemoryVectorStore.
            hybrid_alpha: Weight for semantic score in hybrid retrieval.
            enable_hybrid: Whether to enable hybrid (semantic + keyword) retrieval.
            default_top_k: Default number of results to return.
        """
        self._embedding_provider = embedding_provider or MockEmbeddingProvider(dimensions=128)
        self._vector_store = vector_store or InMemoryVectorStore()
        self._hybrid_retriever = HybridRetriever(alpha=hybrid_alpha) if enable_hybrid else None
        self._enable_hybrid = enable_hybrid
        self._default_top_k = default_top_k
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the RAG client (called once at startup)."""
        if self._initialized:
            return
        self._initialized = True
        logger.info(
            f"RAG Client initialized: "
            f"provider={self._embedding_provider.get_model_name()}, "
            f"store={type(self._vector_store).__name__}, "
            f"hybrid={self._enable_hybrid}"
        )

    async def retrieve(
        self,
        query: str,
        context: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant passages based on query.

        This method implements the RAGClient Protocol from base.py.

        Args:
            query: Search query text.
            context: Optional additional context for the query.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of retrieval result dicts with keys:
            - content: Retrieved text
            - score: Relevance score
            - source: Document ID
            - metadata: Document metadata
            - citation: Formatted citation
        """
        if not self._initialized:
            await self.initialize()

        effective_top_k = top_k or self._default_top_k

        # Enhance query with context if provided
        effective_query = query
        if context:
            effective_query = f"{query} {context}"

        # Embed query
        query_vector = await self._embedding_provider.embed(effective_query)

        # Search vector store
        search_results = await self._vector_store.search(
            query_vector=query_vector,
            top_k=effective_top_k,
            filters=filters,
        )

        # Convert to retrieval results
        retrieval_results = []
        for sr in search_results:
            result = RetrievalResult(
                content=sr.document.content,
                score=sr.score,
                source=sr.document.id,
                metadata=sr.document.metadata,
                citation=self._format_citation(sr.document),
                rank=sr.rank,
            )

            # Apply hybrid scoring if enabled
            if self._enable_hybrid and self._hybrid_retriever:
                hybrid_score = self._hybrid_retriever.compute_hybrid_score(
                    sr.score, effective_query, sr.document.content
                )
                result.score = hybrid_score

            retrieval_results.append(result)

        # Re-sort by score after hybrid adjustment
        if self._enable_hybrid:
            retrieval_results.sort(key=lambda r: r.score, reverse=True)
            for rank, r in enumerate(retrieval_results):
                r.rank = rank

        logger.info(
            f"Retrieved {len(retrieval_results)} results for query: "
            f"'{query[:50]}...' (top_k={effective_top_k})"
        )

        return [r.to_dict() for r in retrieval_results]

    async def add_texts(
        self,
        texts: List[tuple[str, str, Dict[str, Any]]],
    ) -> None:
        """
        Add texts to the RAG system.

        Args:
            texts: List of (id, content, metadata) tuples.
        """
        if not texts:
            return

        # Separate texts and metadata
        doc_ids = [t[0] for t in texts]
        contents = [t[1] for t in texts]
        metadatas = [t[2] if len(t) > 2 else {} for t in texts]

        # Batch embed
        embeddings = await self._embedding_provider.embed_batch(contents)

        # Create documents and add to store
        documents = [
            Document(id=doc_id, content=content, embedding=embedding, metadata=metadata)
            for doc_id, content, embedding, metadata in zip(doc_ids, contents, embeddings, metadatas)
        ]

        await self._vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to RAG store")

    async def add_documents(self, documents: List[Document]) -> None:
        """
        Add pre-embedded documents to the RAG system.

        Args:
            documents: List of Document objects with embeddings.
        """
        await self._vector_store.add_documents(documents)

    async def delete(self, doc_ids: List[str]) -> None:
        """Delete documents from the RAG system."""
        await self._vector_store.delete(doc_ids)

    async def update(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update a document in the RAG system.

        If content is updated, the embedding is automatically re-computed.
        """
        embedding = None
        if content is not None:
            embedding = await self._embedding_provider.embed(content)

        return await self._vector_store.update(
            doc_id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
        )

    def count(self) -> int:
        """Return total number of documents in the store."""
        return self._vector_store.count()

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Access the underlying embedding provider."""
        return self._embedding_provider

    @property
    def vector_store(self) -> VectorStore:
        """Access the underlying vector store."""
        return self._vector_store

    @staticmethod
    def _format_citation(doc: Document) -> str:
        """Format a citation string from document metadata."""
        meta = doc.metadata
        authors = meta.get("authors", "")
        title = meta.get("title", doc.content[:80])
        year = meta.get("year", "")
        source = meta.get("source", "")

        parts = []
        if authors:
            parts.append(authors)
        if title:
            parts.append(f'"{title}"')
        if year:
            parts.append(f"({year})")
        if source:
            parts.append(source)

        return ". ".join(parts) if parts else doc.id

    @property
    def stats(self) -> Dict[str, Any]:
        """RAG client statistics."""
        stats = {
            "initialized": self._initialized,
            "document_count": self.count(),
            "embedding_model": self._embedding_provider.get_model_name(),
            "embedding_dimensions": self._embedding_provider.get_dimensions(),
            "hybrid_enabled": self._enable_hybrid,
        }

        # Add provider-specific stats if available
        if hasattr(self._embedding_provider, "cache_stats"):
            stats["embedding_cache"] = self._embedding_provider.cache_stats
        if hasattr(self._embedding_provider, "api_call_count"):
            stats["api_call_count"] = self._embedding_provider.api_call_count

        return stats


# =============================================================================
# Factory Functions
# =============================================================================

def create_rag_client(
    provider_type: str = "mock",
    store_type: str = "memory",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    enable_hybrid: bool = False,
    hybrid_alpha: float = 0.7,
    cache_size: int = 10000,
) -> RAGClientImpl:
    """
    Factory function to create a fully configured RAG client.

    Args:
        provider_type: Embedding provider type ("openai" or "mock").
        store_type: Vector store type ("memory" or "chromadb").
        api_key: API key for OpenAI provider.
        model: Model name for OpenAI provider.
        dimensions: Embedding dimensions.
        enable_hybrid: Whether to enable hybrid retrieval.
        hybrid_alpha: Weight for semantic score in hybrid retrieval.
        cache_size: Maximum embedding cache entries.

    Returns:
        Configured RAGClientImpl instance.

    Example::

        # Development
        rag = create_rag_client(provider_type="mock", store_type="memory")

        # Production
        rag = create_rag_client(
            provider_type="openai",
            store_type="chromadb",
            api_key="sk-...",
            enable_hybrid=True,
        )
    """
    embedding_provider = create_embedding_provider(
        provider_type=provider_type,
        api_key=api_key,
        model=model,
        dimensions=dimensions,
        cache_size=cache_size,
    )

    vector_store = create_vector_store(store_type=store_type)

    return RAGClientImpl(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        hybrid_alpha=hybrid_alpha,
        enable_hybrid=enable_hybrid,
    )
