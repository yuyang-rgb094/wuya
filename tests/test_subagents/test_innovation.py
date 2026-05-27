"""
Tests for Innovation Sub-agent

Author: WuYa Team
"""

import pytest
from wuya_agents.subagents import InnovationSubAgent
from wuya_agents.base import AgentStatus, EvaluationDimension


class TestInnovationSubAgent:
    """Test suite for InnovationSubAgent."""

    # ========================================================================
    # Basic Functionality Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_innovation_agent_initialization(self, innovation_agent):
        """Test Innovation agent initializes correctly."""
        assert innovation_agent is not None
        assert innovation_agent.dimension == EvaluationDimension.INNOVATION
        await innovation_agent.initialize()
        assert innovation_agent._initialized is True

    @pytest.mark.asyncio
    async def test_innovation_evaluate_returns_dimension_result(self, innovation_agent, sample_paper):
        """Test that evaluate returns a DimensionResult."""
        result = await innovation_agent.evaluate(sample_paper)

        assert result is not None
        assert result.dimension == EvaluationDimension.INNOVATION
        assert isinstance(result.overall_score, float)
        assert isinstance(result.narrative, str)
        assert isinstance(result.sub_dimensions, list)

    @pytest.mark.asyncio
    async def test_innovation_scores_in_valid_range(self, innovation_agent, sample_paper):
        """Test that overall and sub-dimension scores are in valid range."""
        result = await innovation_agent.evaluate(sample_paper)

        assert 0.0 <= result.overall_score <= 5.0
        for sub_dim in result.sub_dimensions:
            assert 0.0 <= sub_dim.score <= 5.0

    @pytest.mark.asyncio
    async def test_innovation_three_dimensions(self, innovation_agent, sample_paper):
        """Test that all three innovation dimensions are evaluated."""
        result = await innovation_agent.evaluate(sample_paper)

        expected_dims = {"novelty", "significance", "advancement"}
        actual_dims = {sub_dim.name for sub_dim in result.sub_dimensions}
        assert actual_dims == expected_dims

    @pytest.mark.asyncio
    async def test_innovation_narrative_generated(self, innovation_agent, sample_paper):
        """Test that narrative is generated for the evaluation."""
        result = await innovation_agent.evaluate(sample_paper)

        assert result.narrative is not None
        assert len(result.narrative) > 0
        assert "Innovation" in result.narrative or "innovation" in result.narrative.lower()

    # ========================================================================
    # RAG Trigger Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_innovation_triggers_rag_when_client_available(self, mock_llm_client, mock_rag_client, sample_paper):
        """Test that RAG is triggered when rag_client is available."""
        agent = InnovationSubAgent(llm_client=mock_llm_client, rag_client=mock_rag_client)
        should_trigger = agent.should_trigger_rag(sample_paper)
        assert should_trigger is True

    @pytest.mark.asyncio
    async def test_innovation_no_rag_when_no_client(self, mock_llm_client, sample_paper):
        """Test that RAG is not triggered when rag_client is None."""
        agent = InnovationSubAgent(llm_client=mock_llm_client, rag_client=None)
        should_trigger = agent.should_trigger_rag(sample_paper)
        assert should_trigger is False

    @pytest.mark.asyncio
    async def test_rag_query_constructed_from_paper(self, innovation_agent, sample_paper):
        """Test RAG query is constructed from paper content."""
        query = innovation_agent._construct_rag_query(sample_paper)

        assert len(query) > 0
        # Query should include paper keywords or discipline
        assert any(kw in query.lower() for kw in sample_paper.keywords) or \
               sample_paper.discipline in query.lower()

    # ========================================================================
    # Mock Mode Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_innovation_works_without_llm(self, sample_paper, mock_rag_client):
        """Test Innovation works in mock mode (no LLM)."""
        agent = InnovationSubAgent(llm_client=None, rag_client=mock_rag_client)
        result = await agent.evaluate(sample_paper)

        assert result is not None
        assert result.overall_score > 0
        assert len(result.sub_dimensions) == 3

    @pytest.mark.asyncio
    async def test_innovation_mock_returns_consistent_scores(self, sample_paper, mock_rag_client):
        """Test that mock mode returns consistent scores."""
        agent = InnovationSubAgent(llm_client=None, rag_client=mock_rag_client)

        result1 = await agent.evaluate(sample_paper)
        result2 = await agent.evaluate(sample_paper)

        # Mock should return identical results
        assert result1.overall_score == result2.overall_score

    # ========================================================================
    # System Prompt Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_system_prompt_contains_dimensions(self, innovation_agent):
        """Test system prompt contains innovation dimensions."""
        prompt = innovation_agent.get_system_prompt()

        assert "novelty" in prompt.lower()
        assert "significance" in prompt.lower()
        assert "advancement" in prompt.lower()

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_innovation_handles_llm_failure(self, sample_paper):
        """Test Innovation handles LLM failures gracefully."""
        from tests.conftest import MockLLMClient

        mock_llm = MockLLMClient(should_fail=True)
        agent = InnovationSubAgent(llm_client=mock_llm)
        result = await agent.evaluate(sample_paper)

        # Should return a result (possibly with failed status)
        assert result is not None
        assert isinstance(result.overall_score, float)

    # ========================================================================
    # Edge Cases
    # ========================================================================

    @pytest.mark.asyncio
    async def test_innovation_with_empty_keywords(self, innovation_agent):
        """Test Innovation handles paper with no keywords."""
        from wuya_agents.base import ParsedPaper

        paper = ParsedPaper(
            paper_id="no_keywords",
            title="Paper Without Keywords",
            abstract="This paper has no keywords",
            content="Content",
            authors=["Author"],
            keywords=[],
            discipline="cs"
        )

        result = await innovation_agent.evaluate(paper)

        assert result is not None
        # Should still work, using discipline for RAG query
        assert result.overall_score > 0

    @pytest.mark.asyncio
    async def test_innovation_status_success(self, innovation_agent, sample_paper):
        """Test that successful evaluation has SUCCESS status."""
        result = await innovation_agent.evaluate(sample_paper)

        assert result.status == AgentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_innovation_processing_time_recorded(self, innovation_agent, sample_paper):
        """Test that processing time is recorded."""
        result = await innovation_agent.evaluate(sample_paper)

        assert result.processing_time_ms >= 0

    # ========================================================================
    # Sub-dimension Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_innovation_sub_dimensions_have_justifications(self, innovation_agent, sample_paper):
        """Test that sub-dimensions have justifications."""
        result = await innovation_agent.evaluate(sample_paper)

        for sub_dim in result.sub_dimensions:
            assert hasattr(sub_dim, 'justification')
            assert isinstance(sub_dim.justification, str)

    @pytest.mark.asyncio
    async def test_innovation_sub_dimensions_weighted(self, innovation_agent, sample_paper):
        """Test that sub-dimensions have weights."""
        result = await innovation_agent.evaluate(sample_paper)

        for sub_dim in result.sub_dimensions:
            assert hasattr(sub_dim, 'weight')
            assert sub_dim.weight > 0

    # ========================================================================
    # Context Handling Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_innovation_handles_context(self, innovation_agent, sample_paper):
        """Test Innovation handles context parameter."""
        context = {"previous_dimension": "cudos", "score": 4.5}

        result = await innovation_agent.evaluate(sample_paper, context=context)

        assert result is not None
        assert result.overall_score > 0
