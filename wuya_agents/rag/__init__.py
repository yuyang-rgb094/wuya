"""
WuYa RAG (Retrieval-Augmented Generation) Module

Provides semantic retrieval capabilities for the WuYa multi-agent system.

Components:
    - embedding: Text embedding providers (OpenAI, Mock)
    - vector_store: Vector storage and similarity search
    - client: RAG client combining embedding + vector store

Author: WuYa Team
"""

from .embedding import (
    EmbeddingProvider,
    EmbeddingCache,
    EmbeddingResult,
    OpenAIEmbeddingProvider,
    MockEmbeddingProvider,
    create_embedding_provider,
)

from .vector_store import (
    VectorStore,
    Document,
    SearchResult,
    InMemoryVectorStore,
    ChromaDBVectorStore,
    create_vector_store,
)

from .client import (
    RAGClientImpl,
    RetrievalResult,
    HybridRetriever,
    create_rag_client,
)

__all__ = [
    # Embedding
    "EmbeddingProvider",
    "EmbeddingCache",
    "EmbeddingResult",
    "OpenAIEmbeddingProvider",
    "MockEmbeddingProvider",
    "create_embedding_provider",
    # Vector Store
    "VectorStore",
    "Document",
    "SearchResult",
    "InMemoryVectorStore",
    "ChromaDBVectorStore",
    "create_vector_store",
    # Client
    "RAGClientImpl",
    "RetrievalResult",
    "HybridRetriever",
    "create_rag_client",
]
