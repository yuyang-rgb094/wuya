"""
RAG Usage Example

Demonstrates the complete RAG retrieval flow:
1. Create RAG client with mock providers
2. Add documents to the vector store
3. Perform semantic search with and without filters
4. Demonstrate hybrid retrieval
5. Show cache statistics

Run:
    python examples/rag_usage.py

Author: WuYa Team
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Sample academic papers for demonstration
SAMPLE_PAPERS = [
    {
        "id": "paper_001",
        "content": (
            "Deep learning has revolutionized natural language processing. "
            "Transformer architectures, introduced in 'Attention is All You Need', "
            "have become the foundation for modern NLP systems including BERT, GPT, and T5. "
            "These models achieve state-of-the-art results on numerous benchmarks."
        ),
        "metadata": {
            "title": "Survey of Deep Learning in NLP",
            "authors": "Zhang, Li, Wang",
            "discipline": "computer_science",
            "year": 2024,
            "source": "arxiv",
        },
    },
    {
        "id": "paper_002",
        "content": (
            "Randomized controlled trials remain the gold standard for clinical evidence. "
            "This meta-analysis examines 150 RCTs published between 2018-2023, "
            "finding that proper blinding and adequate sample size are the most "
            "critical factors for reducing bias in medical research."
        ),
        "metadata": {
            "title": "Meta-analysis of RCT Quality in Medical Research",
            "authors": "Johnson, Smith, Brown",
            "discipline": "medicine",
            "year": 2024,
            "source": "pubmed",
        },
    },
    {
        "id": "paper_003",
        "content": (
            "CRISPR-Cas9 gene editing has opened new avenues for cancer therapy. "
            "This review covers recent advances in using CRISPR for targeted "
            "immunotherapy, including CAR-T cell engineering and tumor suppressor "
            "gene restoration. Clinical trials show promising response rates."
        ),
        "metadata": {
            "title": "CRISPR Applications in Cancer Immunotherapy",
            "authors": "Chen, Park, Kim",
            "discipline": "biology",
            "year": 2023,
            "source": "nature",
        },
    },
    {
        "id": "paper_004",
        "content": (
            "Methodological rigor in social science research requires careful "
            "consideration of confounding variables, selection bias, and "
            "measurement validity. This paper proposes a framework for "
            "improving reproducibility in quantitative social science studies."
        ),
        "metadata": {
            "title": "Improving Methodological Rigor in Social Sciences",
            "authors": "Anderson, Taylor, Martinez",
            "discipline": "social_science",
            "year": 2024,
            "source": "springer",
        },
    },
    {
        "id": "paper_005",
        "content": (
            "Transformer models for computer vision, such as Vision Transformer (ViT), "
            "have achieved competitive performance with convolutional neural networks. "
            "This paper introduces a hybrid architecture combining CNN local features "
            "with Transformer global attention for image classification."
        ),
        "metadata": {
            "title": "Hybrid CNN-Transformer for Image Classification",
            "authors": "Liu, Wang, Zhang",
            "discipline": "computer_science",
            "year": 2023,
            "source": "cvpr",
        },
    },
    {
        "id": "paper_006",
        "content": (
            "Bayesian statistical methods provide a principled framework for "
            "incorporating prior knowledge into data analysis. This tutorial "
            "covers Markov Chain Monte Carlo sampling, variational inference, "
            "and their applications in clinical trial design."
        ),
        "metadata": {
            "title": "Bayesian Methods for Clinical Trials",
            "authors": "Gelman, Carlin, Stern",
            "discipline": "statistics",
            "year": 2024,
            "source": "arxiv",
        },
    },
    {
        "id": "paper_007",
        "content": (
            "Reinforcement learning from human feedback (RLHF) has become "
            "a key technique for aligning large language models with human "
            "preferences. This paper analyzes the theoretical foundations "
            "of RLHF and proposes improvements to the reward modeling process."
        ),
        "metadata": {
            "title": "Theoretical Analysis of RLHF for LLM Alignment",
            "authors": "Christiano, Leike, Amodei",
            "discipline": "computer_science",
            "year": 2024,
            "source": "neurips",
        },
    },
    {
        "id": "paper_008",
        "content": (
            "Evidence-based medicine requires systematic review of available "
            "literature. The PRISMA guidelines provide a standardized framework "
            "for conducting and reporting systematic reviews and meta-analyses "
            "across medical disciplines."
        ),
        "metadata": {
            "title": "PRISMA Guidelines for Systematic Reviews",
            "authors": "Moher, Liberati, Tetzlaff",
            "discipline": "medicine",
            "year": 2023,
            "source": "lancet",
        },
    },
]


async def demo_basic_retrieval():
    """Demo: Basic semantic retrieval."""
    print("=" * 60)
    print("Demo 1: Basic Semantic Retrieval")
    print("=" * 60)

    from wuya_agents.rag import (
        MockEmbeddingProvider,
        InMemoryVectorStore,
        RAGClientImpl,
    )

    # Create RAG client with mock providers
    rag = RAGClientImpl(
        embedding_provider=MockEmbeddingProvider(dimensions=128),
        vector_store=InMemoryVectorStore(),
    )

    # Add sample papers
    texts = [
        (p["id"], p["content"], p["metadata"])
        for p in SAMPLE_PAPERS
    ]
    await rag.add_texts(texts)
    print(f"\nAdded {rag.count()} documents to RAG store")

    # Search for relevant papers
    query = "deep learning transformer architecture"
    print(f"\nQuery: '{query}'")
    print("-" * 40)

    results = await rag.retrieve(query, top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['source']}: {r['content'][:60]}...")

    # Another query
    query2 = "clinical trial methodology evidence"
    print(f"\nQuery: '{query2}'")
    print("-" * 40)

    results2 = await rag.retrieve(query2, top_k=3)
    for r in results2:
        print(f"  [{r['score']:.4f}] {r['source']}: {r['content'][:60]}...")

    return rag


async def demo_filtered_retrieval():
    """Demo: Retrieval with metadata filters."""
    print("\n" + "=" * 60)
    print("Demo 2: Metadata-Filtered Retrieval")
    print("=" * 60)

    from wuya_agents.rag import (
        MockEmbeddingProvider,
        InMemoryVectorStore,
        RAGClientImpl,
    )

    rag = RAGClientImpl(
        embedding_provider=MockEmbeddingProvider(dimensions=128),
        vector_store=InMemoryVectorStore(),
    )

    texts = [(p["id"], p["content"], p["metadata"]) for p in SAMPLE_PAPERS]
    await rag.add_texts(texts)

    # Filter by discipline
    query = "research methodology"
    print(f"\nQuery: '{query}'")
    print(f"Filter: discipline=computer_science")
    print("-" * 40)

    results = await rag.retrieve(
        query,
        top_k=5,
        filters={"discipline": "computer_science"},
    )
    for r in results:
        print(f"  [{r['score']:.4f}] {r['source']}: {r['content'][:60]}...")

    # Filter by year
    print(f"\nQuery: '{query}'")
    print(f"Filter: year=2023")
    print("-" * 40)

    results = await rag.retrieve(
        query,
        top_k=5,
        filters={"year": 2023},
    )
    for r in results:
        print(f"  [{r['score']:.4f}] {r['source']}: {r['content'][:60]}...")


async def demo_hybrid_retrieval():
    """Demo: Hybrid retrieval (semantic + keyword)."""
    print("\n" + "=" * 60)
    print("Demo 3: Hybrid Retrieval (Semantic + Keyword)")
    print("=" * 60)

    from wuya_agents.rag import (
        MockEmbeddingProvider,
        InMemoryVectorStore,
        RAGClientImpl,
    )

    # Create with hybrid enabled
    rag = RAGClientImpl(
        embedding_provider=MockEmbeddingProvider(dimensions=128),
        vector_store=InMemoryVectorStore(),
        enable_hybrid=True,
        hybrid_alpha=0.7,  # 70% semantic, 30% keyword
    )

    texts = [(p["id"], p["content"], p["metadata"]) for p in SAMPLE_PAPERS]
    await rag.add_texts(texts)

    query = "reinforcement learning RLHF"
    print(f"\nQuery: '{query}' (hybrid mode)")
    print("-" * 40)

    results = await rag.retrieve(query, top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['source']}: {r['content'][:80]}...")


async def demo_cache_and_stats():
    """Demo: Embedding cache and statistics."""
    print("\n" + "=" * 60)
    print("Demo 4: Cache and Statistics")
    print("=" * 60)

    from wuya_agents.rag import (
        MockEmbeddingProvider,
        InMemoryVectorStore,
        RAGClientImpl,
    )

    rag = RAGClientImpl(
        embedding_provider=MockEmbeddingProvider(dimensions=128),
        vector_store=InMemoryVectorStore(),
    )

    texts = [(p["id"], p["content"], p["metadata"]) for p in SAMPLE_PAPERS]
    await rag.add_texts(texts)

    # Same query twice - second should hit cache
    query = "machine learning"
    print(f"\nFirst query: '{query}'")
    await rag.retrieve(query, top_k=3)

    print(f"Second query: '{query}' (should use cache)")
    await rag.retrieve(query, top_k=3)

    # Show stats
    stats = rag.stats
    print(f"\nRAG Client Statistics:")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  Embedding dimensions: {stats['embedding_dimensions']}")
    if "embedding_cache" in stats:
        cache = stats["embedding_cache"]
        print(f"  Cache hits: {cache['hits']}")
        print(f"  Cache misses: {cache['misses']}")
        print(f"  Cache hit rate: {cache['hit_rate']:.2%}")


async def demo_factory():
    """Demo: Using factory functions."""
    print("\n" + "=" * 60)
    print("Demo 5: Factory Functions")
    print("=" * 60)

    from wuya_agents.rag import create_rag_client

    # Create with factory (mock mode)
    rag = create_rag_client(
        provider_type="mock",
        store_type="memory",
        dimensions=256,
        enable_hybrid=False,
    )

    texts = [(p["id"], p["content"], p["metadata"]) for p in SAMPLE_PAPERS]
    await rag.add_texts(texts)

    results = await rag.retrieve("transformer attention mechanism", top_k=3)
    print(f"\nFactory-created RAG client (mock, 256d):")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['source']}: {r['content'][:60]}...")

    # Show what production config would look like
    print("\nProduction configuration example:")
    print("  rag = create_rag_client(")
    print("      provider_type='openai',")
    print("      store_type='chromadb',")
    print("      api_key='sk-...',")
    print("      enable_hybrid=True,")
    print("  )")


async def main():
    """Run all RAG demos."""
    print("\n" + "=" * 60)
    print("WuYa RAG System - Usage Examples")
    print("=" * 60)

    await demo_basic_retrieval()
    await demo_filtered_retrieval()
    await demo_hybrid_retrieval()
    await demo_cache_and_stats()
    await demo_factory()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
