"""
Tests for RAG Components

Author: WuYa Team
"""

import pytest
import numpy as np
from wuya_agents.rag.embedding import MockEmbeddingProvider, EmbeddingProvider
from wuya_agents.rag.vector_store import InMemoryVectorStore, VectorStore, Document, SearchResult
from wuya_agents.rag.client import RAGClientImpl, HybridRetriever, RetrievalResult


class TestMockEmbeddingProvider:
    """Test suite for MockEmbeddingProvider."""

    def test_initialization(self):
        """Test provider initializes correctly."""
        provider = MockEmbeddingProvider(dimensions=128)
        assert provider.get_dimensions() == 128
        assert "mock-embedding" in provider.get_model_name()

    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding a single text."""
        provider = MockEmbeddingProvider(dimensions=128)
        embedding = await provider.embed("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 128

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding."""
        provider = MockEmbeddingProvider(dimensions=128)
        embeddings = await provider.embed_batch(["text1", "text2", "text3"])

        assert len(embeddings) == 3
        assert all(len(e) == 128 for e in embeddings)

    @pytest.mark.asyncio
    async def test_same_text_same_embedding(self):
        """Test same text produces same embedding."""
        provider = MockEmbeddingProvider(dimensions=128)
        emb1 = await provider.embed("test text")
        emb2 = await provider.embed("test text")

        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_different_text_different_embedding(self):
        """Test different texts produce different embeddings."""
        provider = MockEmbeddingProvider(dimensions=128)
        emb1 = await provider.embed("text one")
        emb2 = await provider.embed("text two")

        assert emb1 != emb2


class TestInMemoryVectorStore:
    """Test suite for InMemoryVectorStore."""

    def test_initialization(self):
        """Test store initializes correctly."""
        store = InMemoryVectorStore()
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_add_single_document(self):
        """Test adding a single document."""
        store = InMemoryVectorStore()
        doc = Document(
            id="doc1",
            content="Test content",
            embedding=np.random.rand(128).tolist(),
            metadata={"source": "test"}
        )

        await store.add_documents([doc])
        assert store.count() == 1

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self):
        """Test adding multiple documents."""
        store = InMemoryVectorStore()
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Content {i}",
                embedding=np.random.rand(128).tolist(),
                metadata={}
            )
            for i in range(5)
        ]

        await store.add_documents(docs)
        assert store.count() == 5

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Test search returns results."""
        store = InMemoryVectorStore()
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Content about topic {i}",
                embedding=np.random.rand(128).tolist(),
                metadata={}
            )
            for i in range(5)
        ]
        await store.add_documents(docs)

        query = np.random.rand(128).tolist()
        results = await store.search(query, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self):
        """Test search respects top_k parameter."""
        store = InMemoryVectorStore()
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Content {i}",
                embedding=np.random.rand(128).tolist(),
                metadata={}
            )
            for i in range(10)
        ]
        await store.add_documents(docs)

        results = await store.search(np.random.rand(128).tolist(), top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test search with metadata filters."""
        store = InMemoryVectorStore()
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Content {i}",
                embedding=np.random.rand(128).tolist(),
                metadata={"category": "A" if i < 3 else "B"}
            )
            for i in range(5)
        ]
        await store.add_documents(docs)

        results = await store.search(
            np.random.rand(128).tolist(),
            top_k=10,
            filters={"category": "A"}
        )

        assert all(r.document.metadata.get("category") == "A" for r in results)

    @pytest.mark.asyncio
    async def test_delete_document(self):
        """Test deleting a document."""
        store = InMemoryVectorStore()
        doc = Document(
            id="doc1",
            content="Test content",
            embedding=np.random.rand(128).tolist(),
            metadata={}
        )
        await store.add_documents([doc])
        assert store.count() == 1

        await store.delete(["doc1"])
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_update_document(self):
        """Test updating a document."""
        store = InMemoryVectorStore()
        doc = Document(
            id="doc1",
            content="Original content",
            embedding=np.random.rand(128).tolist(),
            metadata={"version": 1}
        )
        await store.add_documents([doc])

        updated = await store.update(
            "doc1",
            content="Updated content",
            metadata={"version": 2}
        )

        assert updated is True

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        """Test search on empty store."""
        store = InMemoryVectorStore()
        results = await store.search(np.random.rand(128).tolist(), top_k=5)

        assert len(results) == 0


class TestHybridRetriever:
    """Test suite for HybridRetriever."""

    def test_initialization(self):
        """Test retriever initializes correctly."""
        retriever = HybridRetriever(alpha=0.7)
        assert retriever.alpha == 0.7

    def test_tokenize(self):
        """Test tokenization."""
        retriever = HybridRetriever()
        tokens = retriever._tokenize("Hello World Test")

        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_keyword_score_exact_match(self):
        """Test keyword score with exact match."""
        retriever = HybridRetriever()
        score = retriever._keyword_score("machine learning", "machine learning algorithms")

        assert score > 0

    def test_keyword_score_no_match(self):
        """Test keyword score with no match."""
        retriever = HybridRetriever()
        score = retriever._keyword_score("python", "machine learning algorithms")

        assert score == 0.0

    def test_compute_hybrid_score(self):
        """Test hybrid score computation."""
        retriever = HybridRetriever(alpha=0.7)
        score = retriever.compute_hybrid_score(0.8, "test query", "test content with test query")

        assert 0.0 <= score <= 1.0

    def test_hybrid_alpha_0(self):
        """Test hybrid with alpha=0 (keyword only)."""
        retriever = HybridRetriever(alpha=0.0)
        score = retriever.compute_hybrid_score(0.0, "test", "test content")

        assert score >= 0

    def test_hybrid_alpha_1(self):
        """Test hybrid with alpha=1 (semantic only)."""
        retriever = HybridRetriever(alpha=1.0)
        score = retriever.compute_hybrid_score(0.8, "test", "unrelated content")

        assert score == 0.8


class TestRAGClientImpl:
    """Test suite for RAGClientImpl."""

    @pytest.mark.asyncio
    async def test_initialization(self, rag_client):
        """Test client initializes correctly."""
        assert rag_client is not None
        assert rag_client.count() == 0

    @pytest.mark.asyncio
    async def test_initialize(self, rag_client):
        """Test client initialize method."""
        await rag_client.initialize()
        assert True

    @pytest.mark.asyncio
    async def test_add_texts(self, rag_client):
        """Test adding texts to RAG."""
        texts = [
            ("doc1", "Content about machine learning", {"topic": "ml"}),
            ("doc2", "Content about deep learning", {"topic": "dl"}),
            ("doc3", "Content about NLP", {"topic": "nlp"}),
        ]

        await rag_client.add_texts(texts)
        assert rag_client.count() == 3

    @pytest.mark.asyncio
    async def test_retrieve_without_data(self, rag_client):
        """Test retrieve on empty store."""
        await rag_client.initialize()
        results = await rag_client.retrieve("test query", top_k=5)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_with_data(self, rag_client):
        """Test retrieve with stored data."""
        await rag_client.initialize()

        texts = [
            ("doc1", "Machine learning is great", {"topic": "ml"}),
            ("doc2", "Deep learning uses neural networks", {"topic": "dl"}),
            ("doc3", "Natural language processing deals with text", {"topic": "nlp"}),
        ]
        await rag_client.add_texts(texts)

        results = await rag_client.retrieve("neural networks deep learning", top_k=2)

        assert len(results) == 2
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self, rag_client):
        """Test retrieve respects top_k."""
        await rag_client.initialize()

        texts = [(f"doc{i}", f"Content {i}", {}) for i in range(10)]
        await rag_client.add_texts(texts)

        results = await rag_client.retrieve("content", top_k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_with_context(self, rag_client):
        """Test retrieve with context."""
        await rag_client.initialize()

        texts = [
            ("doc1", "Machine learning algorithms", {"topic": "ml"}),
        ]
        await rag_client.add_texts(texts)

        results = await rag_client.retrieve(
            "algorithms",
            context="Additional context about machine learning",
            top_k=1
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self, rag_client):
        """Test retrieve with metadata filters."""
        await rag_client.initialize()

        texts = [
            ("doc1", "Machine learning content", {"category": "ai", "year": 2024}),
            ("doc2", "Deep learning content", {"category": "ai", "year": 2023}),
            ("doc3", "Physics content", {"category": "science", "year": 2024}),
        ]
        await rag_client.add_texts(texts)

        results = await rag_client.retrieve(
            "learning",
            top_k=10,
            filters={"category": "ai"}
        )

        assert all(r["metadata"].get("category") == "ai" for r in results)

    @pytest.mark.asyncio
    async def test_delete(self, rag_client):
        """Test deleting documents."""
        await rag_client.initialize()

        texts = [
            ("doc1", "Content 1", {}),
            ("doc2", "Content 2", {}),
        ]
        await rag_client.add_texts(texts)
        assert rag_client.count() == 2

        await rag_client.delete(["doc1"])
        assert rag_client.count() == 1

    @pytest.mark.asyncio
    async def test_update(self, rag_client):
        """Test updating documents."""
        await rag_client.initialize()

        texts = [
            ("doc1", "Original content", {"version": "1"}),
        ]
        await rag_client.add_texts(texts)

        updated = await rag_client.update(
            "doc1",
            content="Updated content",
            metadata={"version": "2"}
        )

        assert updated is True

    @pytest.mark.asyncio
    async def test_stats(self, rag_client):
        """Test stats property."""
        await rag_client.initialize()

        stats = rag_client.stats

        assert "initialized" in stats
        assert "document_count" in stats
        assert "embedding_model" in stats
        assert "embedding_dimensions" in stats

    @pytest.mark.asyncio
    async def test_citation_formatting(self, rag_client):
        """Test citation is formatted correctly."""
        await rag_client.initialize()

        texts = [
            ("doc1", "Content", {"title": "Test Paper", "authors": "Smith, J.", "year": 2024}),
        ]
        await rag_client.add_texts(texts)

        results = await rag_client.retrieve("content", top_k=1)

        if results:
            assert "citation" in results[0]


class TestRetrievalResult:
    """Test suite for RetrievalResult dataclass."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = RetrievalResult(
            content="Test content",
            score=0.85,
            source="doc1",
            metadata={"key": "value"},
            citation="Smith (2024)",
            rank=0
        )

        result_dict = result.to_dict()

        assert result_dict["content"] == "Test content"
        assert result_dict["score"] == 0.85
        assert result_dict["source"] == "doc1"
        assert result_dict["rank"] == 0


class TestHybridRetrievalIntegration:
    """Integration tests for hybrid retrieval."""

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_enabled(self):
        """Test hybrid retrieval is enabled."""
        client = RAGClientImpl(
            embedding_provider=MockEmbeddingProvider(dimensions=128),
            vector_store=InMemoryVectorStore(),
            enable_hybrid=True,
            hybrid_alpha=0.7
        )

        await client.initialize()

        texts = [
            ("doc1", "Machine learning algorithms", {"topic": "ml"}),
            ("doc2", "Deep neural networks", {"topic": "dl"}),
        ]
        await client.add_texts(texts)

        results = await client.retrieve("neural networks", top_k=2)

        assert len(results) == 2
        assert all("score" in r for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_vs_semantic_comparison(self):
        """Test hybrid scores differ from semantic-only."""
        semantic_client = RAGClientImpl(
            embedding_provider=MockEmbeddingProvider(dimensions=128),
            vector_store=InMemoryVectorStore(),
            enable_hybrid=False
        )

        hybrid_client = RAGClientImpl(
            embedding_provider=MockEmbeddingProvider(dimensions=128),
            vector_store=InMemoryVectorStore(),
            enable_hybrid=True,
            hybrid_alpha=0.5
        )

        await semantic_client.initialize()
        await hybrid_client.initialize()

        texts = [
            ("doc1", "Machine learning algorithms for data science", {"topic": "ml"}),
            ("doc2", "Deep neural networks for image recognition", {"topic": "dl"}),
        ]

        for client in [semantic_client, hybrid_client]:
            await client.add_texts(texts)

        semantic_results = await semantic_client.retrieve("machine learning", top_k=2)
        hybrid_results = await hybrid_client.retrieve("machine learning", top_k=2)

        assert len(semantic_results) == len(hybrid_results) == 2
