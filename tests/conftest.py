"""
Pytest configuration and shared fixtures for WuYa tests.

This module provides:
- Mock LLM client
- Mock RAG client
- Sample paper fixtures
- Common test utilities

Author: WuYa Team
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

# Import project modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wuya_agents.base import (
    AgentStatus,
    EvaluationDimension,
    ParsedPaper,
    SubDimensionScore,
    DimensionResult,
    ScoreVector,
    CUDOSResult,
    RAGClient,
    LLMClient,
    BaseSubAgent,
    CUDOSSubAgent as BaseCUDOSSubAgent,
)
from wuya_agents.subagents import (
    CUDOSSubAgent,
    InnovationSubAgent,
    MethodSubAgent,
    EvidenceSubAgent,
    ApplicationSubAgent,
    FrontierDiscoverySubAgent,
)
from wuya_agents.router import TwoPhaseRouter
from wuya_agents.rag.client import RAGClientImpl
from wuya_agents.rag.embedding import MockEmbeddingProvider
from wuya_agents.rag.vector_store import InMemoryVectorStore
from wuya_agents.dea_subagent import (
    DEASubAgent,
    DEAEngine,
    ScoreVector as DEAScoreVector,
    DEAResult,
    DEAStatus,
)


# =============================================================================
# Mock LLM Client
# =============================================================================

class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        response_generator: Optional[callable] = None,
        should_fail: bool = False,
        failure_exception: Exception = None,
    ):
        """
        Initialize mock LLM client.

        Args:
            responses: Dict mapping prompt patterns to responses.
            response_generator: Function that generates responses.
            should_fail: If True, raise exception on generate().
            failure_exception: Exception to raise on failure.
        """
        self.responses = responses or {}
        self.response_generator = response_generator
        self.should_fail = should_fail
        self.failure_exception = failure_exception or RuntimeError("LLM failure")
        self.call_history = []
        self.api_call_count = 0

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Mock LLM generate method."""
        self.call_history.append({
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
        })
        self.api_call_count += 1

        if self.should_fail:
            raise self.failure_exception

        if self.response_generator:
            return self.response_generator(prompt, system_prompt)

        # Return response based on prompt pattern
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response

        # Default response
        return self._default_response(prompt)

    def _default_response(self, prompt: str) -> str:
        """Generate default response based on prompt content."""
        if "CUDOS" in prompt or "communalism" in prompt.lower():
            return '''{
    "communalism": {"score": 4.0, "issues": [], "justification": "Good open sharing"},
    "universalism": {"score": 4.5, "issues": [], "justification": "Objective evaluation"},
    "disinterestedness": {"score": 4.0, "issues": [], "justification": "No conflicts"},
    "organized_skepticism": {"score": 4.5, "issues": [], "justification": "Acknowledged limitations"}
}'''
        elif "innovation" in prompt.lower() or "novelty" in prompt.lower():
            return '''{
    "novelty": {"score": 4.0, "justification": "Novel approach"},
    "significance": {"score": 4.5, "justification": "Significant contribution"},
    "advancement": {"score": 4.0, "justification": "Advances the field"}
}'''
        elif "method" in prompt.lower() or "methodology" in prompt.lower():
            return '''{
    "rigor": {"score": 4.0, "justification": "Rigorous methods"},
    "validity": {"score": 4.5, "justification": "High validity"},
    "reproducibility": {"score": 4.0, "justification": "Reproducible design"}
}'''
        elif "evidence" in prompt.lower():
            return '''{
    "strength": {"score": 4.0, "justification": "Strong evidence"},
    "consistency": {"score": 4.5, "justification": "Consistent results"},
    "sufficiency": {"score": 4.0, "justification": "Sufficient data"}
}'''
        elif "application" in prompt.lower():
            return '''{
    "relevance": {"score": 4.0, "justification": "Highly relevant"},
    "feasibility": {"score": 4.5, "justification": "Feasible implementation"},
    "impact": {"score": 4.0, "justification": "High impact potential"}
}'''
        else:
            return '{"score": 4.0, "issues": [], "justification": "Default evaluation"}'


# =============================================================================
# Mock RAG Client
# =============================================================================

class MockRAGClient:
    """Mock RAG client for testing."""

    def __init__(
        self,
        retrieval_results: Optional[List[Dict[str, Any]]] = None,
        should_fail: bool = False,
    ):
        """
        Initialize mock RAG client.

        Args:
            retrieval_results: Pre-configured retrieval results.
            should_fail: If True, raise exception on retrieve().
        """
        self.retrieval_results = retrieval_results or []
        self.should_fail = should_fail
        self.query_history = []
        self.retrieval_count = 0

    async def retrieve(
        self,
        query: str,
        context: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Mock RAG retrieve method."""
        self.query_history.append({
            "query": query,
            "context": context,
            "top_k": top_k,
            "filters": filters,
        })
        self.retrieval_count += 1

        if self.should_fail:
            raise RuntimeError("RAG retrieval failure")

        return self.retrieval_results[:top_k]

    async def initialize(self) -> None:
        """Mock initialize."""
        pass

    async def add_texts(self, texts: List[tuple]) -> None:
        """Mock add_texts."""
        pass


# =============================================================================
# Sample Paper Fixtures
# =============================================================================

@pytest.fixture
def sample_paper() -> ParsedPaper:
    """Create a sample paper for testing."""
    return ParsedPaper(
        paper_id="test-paper-001",
        title="Deep Learning for Scientific Discovery: A Novel Approach",
        abstract="This paper presents a novel deep learning approach for scientific discovery. We demonstrate significant improvements over existing methods.",
        content="""Introduction

Deep learning has revolutionized many fields of science. In this paper, we present a novel approach
that combines neural networks with domain-specific knowledge.

Methodology

We propose a new architecture that incorporates:
1. Domain-specific constraints
2. Physics-informed loss functions
3. Interpretable feature extraction

Experiments

We evaluate our method on multiple benchmark datasets:
- Dataset A: 98.5% accuracy (baseline: 95.2%)
- Dataset B: 94.3% accuracy (baseline: 91.8%)
- Dataset C: 91.7% accuracy (baseline: 88.4%)

Results demonstrate significant improvements across all benchmarks.

Conclusion

Our approach provides a principled way to incorporate domain knowledge
into deep learning models, achieving state-of-the-art results.""",
        authors=["Alice Smith", "Bob Johnson"],
        keywords=["deep learning", "scientific discovery", "neural networks"],
        discipline="computer science",
        citations=["paper-1", "paper-2", "paper-3"],
        references=["ref-1", "ref-2", "ref-3"],
        figures=["fig1.png", "fig2.png"],
        tables=["table1.csv"],
    )


@pytest.fixture
def sample_paper_low_quality() -> ParsedPaper:
    """Create a low-quality paper for testing edge cases."""
    return ParsedPaper(
        paper_id="test-paper-low-quality",
        title="Some Thoughts on AI",
        abstract="This paper discusses AI.",
        content="AI is interesting. We did some experiments. Results were okay.",
        authors=["Unknown Author"],
        keywords=["AI"],
        discipline="computer science",
        citations=[],
        references=[],
    )


@pytest.fixture
def sample_paper_ethical_concerns() -> ParsedPaper:
    """Create a paper with potential ethical concerns for testing CUDOS veto."""
    return ParsedPaper(
        paper_id="test-paper-ethical",
        title="Conflict of Interest in Academic Research",
        abstract="This paper examines conflicts of interest.",
        content="The authors have undisclosed conflicts of interest. Data may have been fabricated.",
        authors=["Dr. X"],
        keywords=["ethics", "conflict of interest"],
        discipline="general",
        citations=[],
        references=[],
    )


@pytest.fixture
def sample_reference_papers() -> List[Dict[str, Any]]:
    """Create sample reference papers for DEA analysis."""
    import numpy as np
    np.random.seed(42)

    papers = []
    for i in range(100):
        papers.append({
            "paper_id": f"ref_{i}",
            "scores": {
                "innovation": float(np.random.uniform(2.5, 5.0)),
                "method": float(np.random.uniform(3.0, 5.0)),
                "evidence": float(np.random.uniform(2.5, 5.0)),
                "application": float(np.random.uniform(2.0, 5.0)),
                "cudos": float(np.random.uniform(4.0, 5.0)),
            },
            "citations": int(np.random.randint(10, 500)),
        })
    return papers


@pytest.fixture
def sample_dea_score_vector() -> DEAScoreVector:
    """Create a sample score vector for DEA testing."""
    return DEAScoreVector(
        innovation=4.5,
        method=4.2,
        evidence=3.5,
        application=4.0,
        cudos=4.8,
    )


# =============================================================================
# Sub-agent Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Create a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def mock_rag_client() -> MockRAGClient:
    """Create a mock RAG client."""
    return MockRAGClient()


@pytest.fixture
def cudos_agent(mock_llm_client, mock_rag_client) -> CUDOSSubAgent:
    """Create a CUDOS sub-agent with mock clients."""
    return CUDOSSubAgent(
        llm_client=mock_llm_client,
        rag_client=mock_rag_client,
        veto_threshold=2.0,
    )


@pytest.fixture
def innovation_agent(mock_llm_client, mock_rag_client) -> InnovationSubAgent:
    """Create an Innovation sub-agent with mock clients."""
    return InnovationSubAgent(
        llm_client=mock_llm_client,
        rag_client=mock_rag_client,
    )


@pytest.fixture
def method_agent(mock_llm_client, mock_rag_client) -> MethodSubAgent:
    """Create a Method sub-agent with mock clients."""
    return MethodSubAgent(
        llm_client=mock_llm_client,
        rag_client=mock_rag_client,
    )


@pytest.fixture
def evidence_agent(mock_llm_client, mock_rag_client) -> EvidenceSubAgent:
    """Create an Evidence sub-agent with mock clients."""
    return EvidenceSubAgent(
        llm_client=mock_llm_client,
        rag_client=mock_rag_client,
    )


@pytest.fixture
def application_agent(mock_llm_client, mock_rag_client) -> ApplicationSubAgent:
    """Create an Application sub-agent with mock clients."""
    return ApplicationSubAgent(
        llm_client=mock_llm_client,
        rag_client=mock_rag_client,
    )


@pytest.fixture
def frontier_agent(mock_llm_client, mock_rag_client) -> FrontierDiscoverySubAgent:
    """Create a Frontier Discovery sub-agent with mock clients."""
    return FrontierDiscoverySubAgent(
        llm_client=mock_llm_client,
        rag_client=mock_rag_client,
    )


@pytest.fixture
def dea_engine() -> DEAEngine:
    """Create a DEA engine for testing."""
    return DEAEngine(min_reference_papers=50)


@pytest.fixture
def dea_agent(dea_engine, mock_llm_client, mock_rag_client) -> DEASubAgent:
    """Create a DEA sub-agent with mock clients."""
    return DEASubAgent(
        dea_engine=dea_engine,
        llm_client=mock_llm_client,
        rag_client=mock_rag_client,
    )


@pytest.fixture
def rag_client() -> RAGClientImpl:
    """Create a RAG client for testing."""
    return RAGClientImpl(
        embedding_provider=MockEmbeddingProvider(dimensions=128),
        vector_store=InMemoryVectorStore(),
    )


@pytest.fixture
def two_phase_router(
    cudos_agent,
    innovation_agent,
    method_agent,
    evidence_agent,
    application_agent,
    dea_agent,
    mock_rag_client,
) -> TwoPhaseRouter:
    """Create a TwoPhaseRouter for testing."""
    return TwoPhaseRouter(
        cudos_agent=cudos_agent,
        innovation_agent=innovation_agent,
        method_agent=method_agent,
        evidence_agent=evidence_agent,
        application_agent=application_agent,
        dea_agent=dea_agent,
        rag_client=mock_rag_client,
        enable_parallel=True,
    )


# =============================================================================
# Event Loop Fixture for Async Tests
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Test Utilities
# =============================================================================

def assert_dimension_result_valid(result: DimensionResult) -> None:
    """Assert that a dimension result has valid structure and values."""
    assert result.dimension is not None
    assert 0.0 <= result.overall_score <= 5.0
    assert result.narrative is not None
    assert isinstance(result.sub_dimensions, list)
    assert result.status in [AgentStatus.SUCCESS, AgentStatus.FAILED, AgentStatus.RAG_TRIGGERED]


def assert_cudos_result_valid(result: CUDOSResult, allow_veto: bool = True) -> None:
    """Assert that a CUDOS result has valid structure."""
    assert isinstance(result.gate_pass, bool)
    assert isinstance(result.dimensions, dict)
    assert isinstance(result.veto_reason, (str, type(None)))
    assert isinstance(result.rag_citations, list)

    if result.gate_pass or allow_veto:
        # If passed, all scores should be above threshold
        if result.dimensions:
            scores = [d.get("score", 0) for d in result.dimensions.values()]
            assert all(s >= 2.0 for s in scores), "Passed CUDOS should have all scores >= 2.0"


def create_dimension_result(
    dimension: EvaluationDimension,
    score: float = 4.0,
    narrative: str = "Test evaluation",
) -> DimensionResult:
    """Create a dimension result for testing."""
    return DimensionResult(
        dimension=dimension,
        overall_score=score,
        sub_dimensions=[
            SubDimensionScore(name="test_dim", score=score, justification="Test"),
        ],
        narrative=narrative,
        status=AgentStatus.SUCCESS,
        processing_time_ms=100,
    )
