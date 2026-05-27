"""
Tests for Evidence Sub-agent

Author: WuYa Team
"""

import pytest
from wuya_agents.subagents import EvidenceSubAgent
from wuya_agents.base import AgentStatus, EvaluationDimension


class TestEvidenceSubAgent:
    """Test suite for EvidenceSubAgent."""

    # ========================================================================
    # Basic Functionality Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_evidence_agent_initialization(self, evidence_agent):
        """Test Evidence agent initializes correctly."""
        assert evidence_agent is not None
        assert evidence_agent.dimension == EvaluationDimension.EVIDENCE
        await evidence_agent.initialize()
        assert evidence_agent._initialized is True

    @pytest.mark.asyncio
    async def test_evidence_evaluate_returns_dimension_result(self, evidence_agent, sample_paper):
        """Test that evaluate returns a DimensionResult."""
        result = await evidence_agent.evaluate(sample_paper)

        assert result is not None
        assert result.dimension == EvaluationDimension.EVIDENCE
        assert isinstance(result.overall_score, float)
        assert isinstance(result.narrative, str)

    @pytest.mark.asyncio
    async def test_evidence_three_dimensions(self, evidence_agent, sample_paper):
        """Test that all three evidence dimensions are evaluated."""
        result = await evidence_agent.evaluate(sample_paper)

        expected_dims = {"strength", "consistency", "sufficiency"}
        actual_dims = {sub_dim.name for sub_dim in result.sub_dimensions}
        assert actual_dims == expected_dims

    @pytest.mark.asyncio
    async def test_evidence_scores_in_valid_range(self, evidence_agent, sample_paper):
        """Test that all scores are in valid range."""
        result = await evidence_agent.evaluate(sample_paper)

        assert 0.0 <= result.overall_score <= 5.0
        for sub_dim in result.sub_dimensions:
            assert 0.0 <= sub_dim.score <= 5.0

    # ========================================================================
    # RAG Trigger Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_evidence_triggers_rag_on_low_sufficiency(self, mock_llm_client, mock_rag_client, sample_paper):
        """Test that RAG is triggered on low sufficiency score."""
        agent = EvidenceSubAgent(llm_client=mock_llm_client, rag_client=mock_rag_client, rag_threshold=3.0)
        agent._preliminary_scores = {"strength": 3.0, "consistency": 3.0, "sufficiency": 2.0}

        should_trigger = agent.should_trigger_rag(sample_paper)

        assert should_trigger is True

    @pytest.mark.asyncio
    async def test_evidence_no_rag_without_rag_client(self, mock_llm_client, sample_paper):
        """Test that RAG is not triggered without rag_client."""
        agent = EvidenceSubAgent(llm_client=mock_llm_client, rag_client=None)
        agent._preliminary_scores = {"strength": 3.0, "consistency": 3.0, "sufficiency": 2.0}

        should_trigger = agent.should_trigger_rag(sample_paper)

        assert should_trigger is False

    @pytest.mark.asyncio
    async def test_evidence_rag_trigger_on_evidence_gaps(self, mock_llm_client, mock_rag_client):
        """Test RAG trigger on evidence gap indicators."""
        from wuya_agents.base import ParsedPaper

        agent = EvidenceSubAgent(llm_client=mock_llm_client, rag_client=mock_rag_client)

        paper = ParsedPaper(
            paper_id="gap",
            title="Research Paper",
            abstract="Insufficient evidence was found. More research needed.",
            content="The findings are preliminary. Further study required. Evidence is lacking.",
            authors=["Author"],
            keywords=["evidence"],
            discipline="cs"
        )

        should_trigger = agent.should_trigger_rag(paper)
        assert should_trigger is True

    # ========================================================================
    # Mock Mode Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_evidence_works_without_llm(self, sample_paper, mock_rag_client):
        """Test Evidence works in mock mode."""
        agent = EvidenceSubAgent(llm_client=None, rag_client=mock_rag_client)

        result = await agent.evaluate(sample_paper)

        assert result is not None
        assert result.overall_score > 0
        assert len(result.sub_dimensions) == 3

    # ========================================================================
    # System Prompt Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_system_prompt_contains_dimensions(self, evidence_agent):
        """Test system prompt contains evidence dimensions."""
        prompt = evidence_agent.get_system_prompt()

        assert "strength" in prompt.lower()
        assert "consistency" in prompt.lower()
        assert "sufficiency" in prompt.lower()

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_evidence_handles_llm_failure(self, sample_paper):
        """Test Evidence handles LLM failures gracefully."""
        from tests.conftest import MockLLMClient

        mock_llm = MockLLMClient(should_fail=True)
        agent = EvidenceSubAgent(llm_client=mock_llm)

        result = await agent.evaluate(sample_paper)

        assert result is not None
        assert isinstance(result.overall_score, float)

    # ========================================================================
    # Edge Cases
    # ========================================================================

    @pytest.mark.asyncio
    async def test_evidence_empty_paper(self, evidence_agent):
        """Test Evidence handles empty paper content."""
        from wuya_agents.base import ParsedPaper

        empty_paper = ParsedPaper(
            paper_id="empty",
            title="Empty",
            abstract="",
            content="",
            authors=["Author"],
            keywords=[],
            discipline="cs"
        )

        result = await evidence_agent.evaluate(empty_paper)

        assert result is not None
        assert isinstance(result.overall_score, float)

    @pytest.mark.asyncio
    async def test_evidence_custom_rag_threshold(self):
        """Test Evidence with custom RAG threshold."""
        from tests.conftest import MockLLMClient, MockRAGClient

        mock_llm = MockLLMClient()
        mock_rag = MockRAGClient()

        agent = EvidenceSubAgent(
            llm_client=mock_llm,
            rag_client=mock_rag,
            rag_threshold=4.0
        )

        assert agent.rag_threshold == 4.0

    # ========================================================================
    # Narrative Generation Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_evidence_narrative_generated(self, evidence_agent, sample_paper):
        """Test that narrative is generated."""
        result = await evidence_agent.evaluate(sample_paper)

        assert result.narrative is not None
        assert len(result.narrative) > 0

    @pytest.mark.asyncio
    async def test_evidence_processing_time_recorded(self, evidence_agent, sample_paper):
        """Test that processing time is recorded."""
        result = await evidence_agent.evaluate(sample_paper)

        assert result.processing_time_ms >= 0

    # ========================================================================
    # RAG Query Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_evidence_rag_query_construction(self, evidence_agent, sample_paper):
        """Test RAG query is properly constructed."""
        query = evidence_agent._construct_rag_query(sample_paper)

        assert len(query) > 0
        assert "empirical" in query.lower() or "evidence" in query.lower()
