"""
RAG Vector Store Module

Provides vector storage and similarity search capabilities.
Supports in-memory store for development/testing and
ChromaDB interface for production use.

Key Design:
    - Abstract base class for store extensibility
    - In-memory store with cosine similarity for dev/test
    - Metadata filtering support (discipline, year, source, etc.)
    - CRUD operations: add, search, delete, update
    - SearchResult dataclass for structured results

Author: WuYa Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import math
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    A document to be stored in the vector store.

    Attributes:
        id: Unique document identifier.
        content: Text content of the document.
        embedding: Vector embedding of the content.
        metadata: Optional metadata dict (discipline, year, source, etc.).
    """
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """
    A result from vector similarity search.

    Attributes:
        document: The matched document.
        score: Similarity score (0.0 to 1.0, higher is more similar).
        rank: Rank in search results (0-based).
    """
    document: Document
    score: float
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.document.id,
            "content": self.document.content,
            "metadata": self.document.metadata,
            "score": round(self.score, 4),
            "rank": self.rank,
        }


class VectorStore(ABC):
    """
    Abstract base class for vector stores.

    All vector stores must implement:
    1. add_documents(docs) -> None
    2. search(query_vector, top_k, filters) -> List[SearchResult]
    3. delete(doc_ids) -> None
    4. update(doc_id, content, embedding, metadata) -> None
    5. get(doc_id) -> Optional[Document]
    """

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the store."""
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of search results sorted by similarity score (descending).
        """
        ...

    @abstractmethod
    async def delete(self, doc_ids: List[str]) -> None:
        """Delete documents by IDs."""
        ...

    @abstractmethod
    async def update(
        self,
        doc_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a document. Returns True if document was found and updated."""
        ...

    @abstractmethod
    async def get(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of documents in the store."""
        ...


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store with cosine similarity search.

    Suitable for development, testing, and small-scale deployments.
    Supports metadata filtering with exact match and list membership.

    Example::

        store = InMemoryVectorStore()
        await store.add_documents([
            Document(id="1", content="...", embedding=[0.1, 0.2, ...]),
            Document(id="2", content="...", embedding=[0.3, 0.4, ...]),
        ])
        results = await store.search(query_vector, top_k=3)
        for r in results:
            print(f"  {r.document.id}: score={r.score:.4f}")

    Example (with metadata filtering)::

        results = await store.search(
            query_vector,
            top_k=5,
            filters={"discipline": "computer_science", "year": 2024}
        )
    """

    def __init__(self):
        """Initialize in-memory vector store."""
        self._documents: Dict[str, Document] = {}
        self._vectors: Dict[str, List[float]] = {}

    async def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the store.

        If a document with the same ID exists, it will be overwritten.
        """
        for doc in documents:
            self._documents[doc.id] = doc
            self._vectors[doc.id] = doc.embedding
        logger.info(f"Added {len(documents)} documents (total: {len(self._documents)})")

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters.
                Supports exact match: {"key": "value"}
                Supports list membership: {"key": ["value1", "value2"]}

        Returns:
            List of SearchResult sorted by score (descending).
        """
        if not self._documents:
            return []

        # Apply metadata filters first
        candidate_ids = self._apply_filters(filters)

        if not candidate_ids:
            return []

        # Compute cosine similarity for all candidates
        scored = []
        query_norm = self._magnitude(query_vector)

        for doc_id in candidate_ids:
            doc_vector = self._vectors.get(doc_id)
            if doc_vector is None:
                continue

            similarity = self._cosine_similarity(query_vector, doc_vector, query_norm)
            scored.append((doc_id, similarity))

        # Sort by similarity (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for rank, (doc_id, score) in enumerate(scored[:top_k]):
            doc = self._documents[doc_id]
            results.append(SearchResult(
                document=doc,
                score=score,
                rank=rank,
            ))

        logger.debug(
            f"Search returned {len(results)} results "
            f"(from {len(candidate_ids)} candidates)"
        )
        return results

    async def delete(self, doc_ids: List[str]) -> None:
        """Delete documents by IDs."""
        deleted = 0
        for doc_id in doc_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                del self._vectors[doc_id]
                deleted += 1
        logger.info(f"Deleted {deleted} documents (total: {len(self._documents)})")

    async def update(
        self,
        doc_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a document. Returns True if found and updated."""
        doc = self._documents.get(doc_id)
        if doc is None:
            return False

        if content is not None:
            doc.content = content
        if embedding is not None:
            doc.embedding = embedding
            self._vectors[doc_id] = embedding
        if metadata is not None:
            doc.metadata.update(metadata)

        logger.info(f"Updated document {doc_id}")
        return True

    async def get(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        return self._documents.get(doc_id)

    def count(self) -> int:
        """Return total number of documents."""
        return len(self._documents)

    def _apply_filters(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Apply metadata filters to get candidate document IDs.

        Filter types supported:
        - Exact match: {"discipline": "cs"}
        - List membership: {"year": [2023, 2024]}
        """
        if not filters:
            return list(self._documents.keys())

        candidate_ids = list(self._documents.keys())

        for key, value in filters.items():
            filtered = []
            for doc_id in candidate_ids:
                doc = self._documents[doc_id]
                doc_value = doc.metadata.get(key)

                if isinstance(value, list):
                    # List membership filter
                    if doc_value in value:
                        filtered.append(doc_id)
                else:
                    # Exact match filter
                    if doc_value == value:
                        filtered.append(doc_id)

            candidate_ids = filtered

        return candidate_ids

    @staticmethod
    def _magnitude(vector: List[float]) -> float:
        """Compute vector magnitude (L2 norm)."""
        return math.sqrt(sum(v * v for v in vector))

    @staticmethod
    def _cosine_similarity(
        a: List[float],
        b: List[float],
        a_norm: Optional[float] = None,
    ) -> float:
        """Compute cosine similarity between two vectors."""
        if a_norm is None:
            a_norm = math.sqrt(sum(v * v for v in a))
        b_norm = math.sqrt(sum(v * v for v in b))

        if a_norm == 0 or b_norm == 0:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        return dot / (a_norm * b_norm)

    def clear(self) -> None:
        """Remove all documents from the store."""
        self._documents.clear()
        self._vectors.clear()


class ChromaDBVectorStore(VectorStore):
    """
    ChromaDB-based vector store for production use.

    Provides persistent storage and efficient similarity search.
    Requires chromadb package to be installed.

    Example::

        store = ChromaDBVectorStore(
            collection_name="wuya_papers",
            persist_directory="./chroma_db"
        )
        await store.add_documents(documents)
        results = await store.search(query_vector, top_k=10)

    Note:
        This is a placeholder implementation. The actual ChromaDB
        integration should be completed when deploying to production.
    """

    def __init__(
        self,
        collection_name: str = "wuya_papers",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory for persistent storage.
        """
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._collection = None
        self._client = None

    def _ensure_initialized(self):
        """Lazy-initialize ChromaDB client and collection."""
        if self._collection is not None:
            return

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb package is required for ChromaDBVectorStore. "
                "Install it with: pip install chromadb"
            )

        if self._persist_directory:
            self._client = chromadb.PersistentClient(path=self._persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to ChromaDB collection."""
        self._ensure_initialized()

        if not documents:
            return

        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # ChromaDB add supports up to ~41000 items per call
        self._collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(documents)} documents to ChromaDB")

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search ChromaDB for similar documents."""
        self._ensure_initialized()

        query_params: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
        }

        if filters:
            # Convert our filter format to ChromaDB where filter format
            chroma_filters = self._convert_filters(filters)
            if chroma_filters:
                query_params["where"] = chroma_filters

        results = self._collection.query(**query_params)

        # Parse results
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for rank, (doc_id, content, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                doc = Document(
                    id=doc_id,
                    content=content,
                    embedding=[],  # ChromaDB doesn't return stored embeddings
                    metadata=metadata or {},
                )
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                score = 1.0 - distance / 2.0
                search_results.append(SearchResult(
                    document=doc,
                    score=max(0.0, min(1.0, score)),
                    rank=rank,
                ))

        return search_results

    async def delete(self, doc_ids: List[str]) -> None:
        """Delete documents from ChromaDB."""
        self._ensure_initialized()
        if doc_ids:
            self._collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from ChromaDB")

    async def update(
        self,
        doc_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a document in ChromaDB."""
        self._ensure_initialized()

        # ChromaDB update requires all fields
        existing = self._collection.get(ids=[doc_id])
        if not existing or not existing["ids"]:
            return False

        update_data: Dict[str, Any] = {"ids": [doc_id]}

        if content is not None:
            update_data["documents"] = [content]
        elif existing["documents"]:
            update_data["documents"] = existing["documents"]

        if embedding is not None:
            update_data["embeddings"] = [embedding]
        elif existing["embeddings"]:
            update_data["embeddings"] = existing["embeddings"]

        if metadata is not None:
            # Merge with existing metadata
            existing_meta = existing["metadatas"][0] if existing["metadatas"] else {}
            merged = {**existing_meta, **metadata}
            update_data["metadatas"] = [merged]
        elif existing["metadatas"]:
            update_data["metadatas"] = existing["metadatas"]

        self._collection.update(**update_data)
        return True

    async def get(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document from ChromaDB."""
        self._ensure_initialized()

        result = self._collection.get(ids=[doc_id])
        if not result or not result["ids"]:
            return None

        return Document(
            id=result["ids"][0],
            content=result["documents"][0] if result["documents"] else "",
            embedding=result["embeddings"][0] if result["embeddings"] else [],
            metadata=result["metadatas"][0] if result["metadatas"] else {},
        )

    def count(self) -> int:
        """Return total number of documents."""
        self._ensure_initialized()
        return self._collection.count()

    @staticmethod
    def _convert_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert our filter format to ChromaDB where filter format.

        Our format:
            {"discipline": "cs", "year": [2023, 2024]}

        ChromaDB format:
            {"$and": [{"discipline": "cs"}, {"year": {"$in": [2023, 2024]}}]}
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: value})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# =============================================================================
# Factory Functions
# =============================================================================

def create_vector_store(
    store_type: str = "memory",
    collection_name: str = "wuya_papers",
    persist_directory: Optional[str] = None,
) -> VectorStore:
    """
    Factory function to create a vector store.

    Args:
        store_type: Type of store ("memory" or "chromadb").
        collection_name: Collection name for ChromaDB.
        persist_directory: Persistence directory for ChromaDB.

    Returns:
        Configured VectorStore instance.

    Raises:
        ValueError: If store_type is not recognized.
    """
    if store_type == "memory":
        return InMemoryVectorStore()
    elif store_type == "chromadb":
        return ChromaDBVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    else:
        raise ValueError(
            f"Unknown store type: {store_type}. "
            f"Supported: 'memory', 'chromadb'"
        )
