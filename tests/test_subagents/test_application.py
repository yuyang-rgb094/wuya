"""
Tests for Application Sub-agent

Author: WuYa Team
"""

import pytest
from wuya_agents.subagents import ApplicationSubAgent
from wuya_agents.base import AgentStatus, EvaluationDimension


class TestApplicationSubAgent:
    """Test suite for ApplicationSubAgent."""

    # ========================================================================
    # Basic Functionality Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_application_agent_initialization(self, application_agent):
        """Test Application agent initializes correctly."""
        assert application_agent is not None
        assert application_agent.dimension == EvaluationDimension.APPLICATION
        await application_agent.initialize()
        assert application_agent._initialized is True

    @pytest.mark.asyncio
    async def test_application_evaluate_returns_dimension_result(self, application_agent, sample_paper):
        """Test that evaluate returns a DimensionResult."""
        result = await application_agent.evaluate(sample_paper)

        assert result is not None
        assert result.dimension == EvaluationDimension.APPLICATION
        assert isinstance(result.overall_score, float)
        assert isinstance(result.narrative, str)

    @pytest.mark.asyncio
    async def test_application_three_dimensions(self, application_agent, sample_paper):
        """Test that all three application dimensions are evaluated."""
        result = await application_agent.evaluate(sample_paper)

        expected_dims = {"relevance", "feasibility", "impact"}
        actual_dims = {sub_dim.name for sub_dim in result.sub_dimensions}
        assert actual_dims == expected_dims

    @pytest.mark.asyncio
    async def test_application_scores_in_valid_range(self, application_agent, sample_paper):
        """Test that all scores are in valid range."""
        result = await application_agent.evaluate(sample_paper)

        assert 0.0 <= result.overall_score <= 5.0
        for sub_dim in result.sub_dimensions:
            assert 0.0 <= sub_dim.score <= 5.0

    # ========================================================================
    # RAG Trigger Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_application_triggers_rag_when_client_available(self, mock_llm_client, mock_rag_client, sample_paper):
        """Test that RAG is triggered when rag_client is available."""
        agent = ApplicationSubAgent(llm_client=mock_llm_client, rag_client=mock_rag_client)

        should_trigger = agent.should_trigger_rag(sample_paper)
        assert should_trigger is True

    @pytest.mark.asyncio
    async def test_application_no_rag_without_client(self, mock_llm_client, sample_paper):
        """Test that RAG is not triggered without rag_client."""
        agent = ApplicationSubAgent(llm_client=mock_llm_client, rag_client=None)

        should_trigger = agent.should_trigger_rag(sample_paper)
        assert should_trigger is False

    # ========================================================================
    # Application Scenarios Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_application_scenarios_retrieved(self, application_agent, sample_paper):
        """Test that application scenarios are generated."""
        result = await application_agent.evaluate(sample_paper)

        # After evaluation, scenarios should be available
        scenarios = application_agent.get_application_scenarios()
        assert scenarios is not None
        assert isinstance(scenarios, list)

    @pytest.mark.asyncio
    async def test_application_top_scenario_calculation(self, application_agent, sample_paper):
        """Test that top scenario is calculated correctly."""
        await application_agent.evaluate(sample_paper)

        top = application_agent.get_top_scenario()
        # May be None if no scenarios were generated
        assert top is None or hasattr(top, 'feasibility')
        assert top is None or hasattr(top, 'impact')

    # ========================================================================
    # Mock Mode Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_application_works_without_llm(self, sample_paper, mock_rag_client):
        """Test Application works in mock mode."""
        agent = ApplicationSubAgent(llm_client=None, rag_client=mock_rag_client)

        result = await agent.evaluate(sample_paper)

        assert result is not None
        assert result.overall_score > 0
        assert len(result.sub_dimensions) == 3

    # ========================================================================
    # System Prompt Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_system_prompt_contains_dimensions(self, application_agent):
        """Test system prompt contains application dimensions."""
        prompt = application_agent.get_system_prompt()

        assert "relevance" in prompt.lower()
        assert "feasibility" in prompt.lower()
        assert "impact" in prompt.lower()

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_application_handles_llm_failure(self, sample_paper):
        """Test Application handles LLM failures gracefully."""
        from tests.conftest import MockLLMClient

        mock_llm = MockLLMClient(should_fail=True)
        agent = ApplicationSubAgent(llm_client=mock_llm)

        result = await agent.evaluate(sample_paper)

        assert result is not None
        assert isinstance(result.overall_score, float)

    # ========================================================================
    # Edge Cases
    # ========================================================================

    @pytest.mark.asyncio
    async def test_application_empty_paper(self, application_agent):
        """Test Application handles empty paper content."""
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

        result = await application_agent.evaluate(empty_paper)

        assert result is not None
        assert isinstance(result.overall_score, float)

    @pytest.mark.asyncio
    async def test_application_custom_rag_threshold(self):
        """Test Application with custom RAG threshold."""
        from tests.conftest import MockLLMClient, MockRAGClient

        mock_llm = MockLLMClient()
        mock_rag = MockRAGClient()

        agent = ApplicationSubAgent(
            llm_client=mock_llm,
            rag_client=mock_rag,
            rag_threshold=4.0
        )

        assert agent.rag_threshold == 4.0

    # ========================================================================
    # Narrative Generation Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_application_narrative_generated(self, application_agent, sample_paper):
        """Test that narrative is generated."""
        result = await application_agent.evaluate(sample_paper)

        assert result.narrative is not None
        assert len(result.narrative) > 0

    @pytest.mark.asyncio
    async def test_application_processing_time_recorded(self, application_agent, sample_paper):
        """Test that processing time is recorded."""
        result = await application_agent.evaluate(sample_paper)

        assert result.processing_time_ms >= 0

    # ========================================================================
    # RAG Query Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_application_rag_query_construction(self, application_agent, sample_paper):
        """Test RAG query is properly constructed."""
        query = application_agent._construct_rag_query(sample_paper)

        assert len(query) > 0
        # Query should include application-related terms
        assert any(term in query.lower() for term in ["application", "implementation", "practical"])
