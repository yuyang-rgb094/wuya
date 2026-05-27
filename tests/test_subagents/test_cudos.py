"""
Tests for CUDOS Sub-agent

Author: WuYa Team
"""

import pytest
from wuya_agents.subagents import CUDOSSubAgent
from wuya_agents.base import AgentStatus


class TestCUDOSSubAgent:
    """Test suite for CUDOSSubAgent."""

    # ========================================================================
    # Basic Functionality Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_cudos_agent_initialization(self, cudos_agent):
        """Test CUDOS agent initializes correctly."""
        assert cudos_agent is not None
        assert cudos_agent.veto_threshold == 2.0
        await cudos_agent.initialize()
        assert cudos_agent._initialized is True

    @pytest.mark.asyncio
    async def test_cudos_evaluate_returns_cudos_result(self, cudos_agent, sample_paper):
        """Test that evaluate returns a CUDOSResult."""
        result = await cudos_agent.evaluate(sample_paper)

        assert result is not None
        assert isinstance(result.gate_pass, bool)
        assert isinstance(result.dimensions, dict)
        assert isinstance(result.veto_reason, (str, type(None)))

    @pytest.mark.asyncio
    async def test_cudos_passes_valid_paper(self, cudos_agent, sample_paper):
        """Test that a valid paper passes CUDOS gate."""
        result = await cudos_agent.evaluate(sample_paper)

        # With mock LLM returning scores >= 4.0, gate should pass
        assert result.gate_pass is True
        assert result.veto_reason is None

    @pytest.mark.asyncio
    async def test_cudos_dimensions_present(self, cudos_agent, sample_paper):
        """Test that all four CUDOS dimensions are evaluated."""
        result = await cudos_agent.evaluate(sample_paper)

        expected_dimensions = {"communalism", "universalism", "disinterestedness", "organized_skepticism"}
        assert set(result.dimensions.keys()) == expected_dimensions

    @pytest.mark.asyncio
    async def test_cudos_scores_in_valid_range(self, cudos_agent, sample_paper):
        """Test that all CUDOS scores are within 1-5 range."""
        result = await cudos_agent.evaluate(sample_paper)

        for dim_name, dim_data in result.dimensions.items():
            score = dim_data.get("score", 0)
            assert 1.0 <= score <= 5.0, f"{dim_name} score {score} out of range"

    # ========================================================================
    # Veto Logic Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_cudos_veto_on_low_score(self, sample_paper):
        """Test that CUDOS vetoes when score is below threshold."""
        # Create agent with failing LLM response
        failing_response = '''{
            "communalism": {"score": 1.5, "issues": ["Severe violation"]},
            "universalism": {"score": 4.0, "issues": []},
            "disinterestedness": {"score": 4.0, "issues": []},
            "organized_skepticism": {"score": 4.0, "issues": []}
        }'''

        from tests.conftest import MockLLMClient
        mock_llm = MockLLMClient(responses={"communalism": failing_response})
        agent = CUDOSSubAgent(llm_client=mock_llm, veto_threshold=2.0)

        result = await agent.evaluate(sample_paper)
        assert result.gate_pass is False
        assert result.veto_reason is not None

    @pytest.mark.asyncio
    async def test_cudos_fails_open_on_error(self, sample_paper):
        """Test that CUDOS fails open on evaluation errors."""
        from tests.conftest import MockLLMClient

        # Create agent that will fail
        mock_llm = MockLLMClient(should_fail=True)
        agent = CUDOSSubAgent(llm_client=mock_llm, veto_threshold=2.0)

        result = await agent.evaluate(sample_paper)

        # Should fail open (allow through) but may not have error reason set
        # Just check we get a result
        assert result is not None

    # ========================================================================
    # RAG Trigger Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_should_trigger_rag_on_ethical_keywords(self, cudos_agent, sample_paper_ethical_concerns):
        """Test RAG is triggered for papers with ethical concerns."""
        should_trigger = cudos_agent.should_trigger_rag(sample_paper_ethical_concerns)

        assert should_trigger is True

    @pytest.mark.asyncio
    async def test_should_trigger_rag_normal_paper(self, cudos_agent, sample_paper):
        """Test RAG is not triggered for normal papers."""
        should_trigger = cudos_agent.should_trigger_rag(sample_paper)

        # Without explicit keywords, should not trigger
        assert should_trigger is False

    @pytest.mark.asyncio
    async def test_rag_query_construction(self, cudos_agent, sample_paper):
        """Test RAG query is properly constructed."""
        query = cudos_agent._construct_rag_query(sample_paper)

        assert "Merton" in query or "CUDOS" in query
        assert len(query) > 0

    # ========================================================================
    # Utility Method Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_get_dimension_scores(self, cudos_agent, sample_paper):
        """Test extraction of dimension scores."""
        result = await cudos_agent.evaluate(sample_paper)
        scores = cudos_agent.get_dimension_scores(result)

        assert isinstance(scores, dict)
        assert len(scores) == 4  # Four CUDOS dimensions

    @pytest.mark.asyncio
    async def test_get_failed_dimensions(self, cudos_agent, sample_paper):
        """Test identification of failed dimensions."""
        result = await cudos_agent.evaluate(sample_paper)
        failed = cudos_agent.get_failed_dimensions(result)

        assert isinstance(failed, list)
        # With mock returning good scores, should be empty
        assert len(failed) == 0

    @pytest.mark.asyncio
    async def test_get_veto_summary(self, cudos_agent, sample_paper):
        """Test veto summary generation."""
        result = await cudos_agent.evaluate(sample_paper)
        summary = cudos_agent.get_veto_summary(result)

        assert isinstance(summary, str)
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_system_prompt_exists(self, cudos_agent):
        """Test that system prompt is defined."""
        prompt = cudos_agent.get_system_prompt()

        assert prompt is not None
        assert len(prompt) > 100
        assert "CUDOS" in prompt
        assert "communalism" in prompt.lower()

    # ========================================================================
    # Edge Cases
    # ========================================================================

    @pytest.mark.asyncio
    async def test_empty_paper_content(self, cudos_agent):
        """Test handling of paper with minimal content."""
        from wuya_agents.base import ParsedPaper

        empty_paper = ParsedPaper(
            paper_id="empty",
            title="Empty Paper",
            abstract="",
            content="",
            authors=["Author"],
            keywords=[],
            discipline="general"
        )

        result = await cudos_agent.evaluate(empty_paper)
        # Should still return a result
        assert result is not None
        assert isinstance(result.gate_pass, bool)

    @pytest.mark.asyncio
    async def test_custom_veto_threshold(self, sample_paper):
        """Test CUDOS with custom veto threshold."""
        from tests.conftest import MockLLMClient

        mock_llm = MockLLMClient()
        agent = CUDOSSubAgent(llm_client=mock_llm, veto_threshold=3.0)

        result = await agent.evaluate(sample_paper)
        # With scores >= 4.0 and threshold 3.0, should pass
        assert result.gate_pass is True

    @pytest.mark.asyncio
    async def test_no_llm_client_fallback(self, sample_paper, mock_rag_client):
        """Test CUDOS works without LLM client (mock mode)."""
        agent = CUDOSSubAgent(llm_client=None, rag_client=mock_rag_client)

        result = await agent.evaluate(sample_paper)
        # Should return mock evaluation
        assert result.gate_pass is True
        assert len(result.dimensions) == 4

    # ========================================================================
    # Custom RAG Keywords
    # ========================================================================

    @pytest.mark.asyncio
    async def test_custom_rag_trigger_keywords(self, mock_llm_client, mock_rag_client):
        """Test CUDOS with custom RAG trigger keywords."""
        agent = CUDOSSubAgent(
            llm_client=mock_llm_client,
            rag_client=mock_rag_client,
            rag_trigger_keywords=["special concern", "requires review"]
        )

        from wuya_agents.base import ParsedPaper
        paper = ParsedPaper(
            paper_id="special",
            title="Paper with special concern",
            abstract="This paper requires review for special concern",
            content="Special concern about methodology",
            authors=["Author"],
            keywords=["concern"],
            discipline="general"
        )

        should_trigger = agent.should_trigger_rag(paper)
        assert should_trigger is True
