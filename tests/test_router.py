"""
Tests for TwoPhaseRouter

Author: WuYa Team
"""

import pytest
from wuya_agents.router import TwoPhaseRouter, EvaluationReport
from wuya_agents.base import EvaluationDimension, AgentStatus


class TestTwoPhaseRouter:
    """Test suite for TwoPhaseRouter."""

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_router_initialization(self, two_phase_router):
        """Test router initializes correctly."""
        assert two_phase_router is not None
        assert two_phase_router.enable_parallel is True

    def test_router_agents_registered(self, two_phase_router):
        """Test all agents are registered."""
        assert two_phase_router.cudos_agent is not None
        assert two_phase_router.innovation_agent is not None
        assert two_phase_router.method_agent is not None
        assert two_phase_router.evidence_agent is not None
        assert two_phase_router.application_agent is not None

    # ========================================================================
    # Phase 1: CUDOS Gatekeeping Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_phase1_cudos_gate_pass(self, two_phase_router, sample_paper):
        """Test Phase 1 CUDOS gate passes for valid paper."""
        result = await two_phase_router._phase1_cudos_gatekeeping(sample_paper)

        assert result is not None
        assert result.gate_pass is True

    @pytest.mark.asyncio
    async def test_phase1_cudos_gate_veto(self, sample_paper):
        """Test Phase 1 CUDOS gate vetoes problematic paper."""
        from tests.conftest import MockLLMClient
        from wuya_agents.subagents import CUDOSSubAgent
        from wuya_agents.subagents import InnovationSubAgent, MethodSubAgent, EvidenceSubAgent, ApplicationSubAgent
        from wuya_agents.dea_subagent import DEASubAgent, DEAEngine
        from tests.conftest import MockRAGClient

        # Create failing CUDOS response
        failing_response = '''{
            "communalism": {"score": 1.0, "issues": ["Severe violation"]},
            "universalism": {"score": 4.0, "issues": []},
            "disinterestedness": {"score": 4.0, "issues": []},
            "organized_skepticism": {"score": 4.0, "issues": []}
        }'''

        mock_llm = MockLLMClient(responses={"communalism": failing_response})
        mock_rag = MockRAGClient()

        cudos = CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag, veto_threshold=2.0)
        innovation = InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        method = MethodSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        evidence = EvidenceSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        application = ApplicationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        dea = DEASubAgent(dea_engine=DEAEngine())

        router = TwoPhaseRouter(
            cudos_agent=cudos,
            innovation_agent=innovation,
            method_agent=method,
            evidence_agent=evidence,
            application_agent=application,
            dea_agent=dea,
            rag_client=mock_rag,
        )

        result = await router._phase1_cudos_gatekeeping(sample_paper)

        assert result.gate_pass is False
        assert result.veto_reason is not None

    # ========================================================================
    # Phase 2: Parallel Evaluation Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_phase2_parallel_evaluation(self, two_phase_router, sample_paper):
        """Test Phase 2 parallel evaluation."""
        dimension_results, errors = await two_phase_router._phase2_parallel_evaluation(sample_paper)

        assert len(dimension_results) == 4
        assert EvaluationDimension.INNOVATION in dimension_results
        assert EvaluationDimension.METHOD in dimension_results
        assert EvaluationDimension.EVIDENCE in dimension_results
        assert EvaluationDimension.APPLICATION in dimension_results

    @pytest.mark.asyncio
    async def test_phase2_returns_valid_results(self, two_phase_router, sample_paper):
        """Test Phase 2 returns valid dimension results."""
        dimension_results, errors = await two_phase_router._phase2_parallel_evaluation(sample_paper)

        for dim, result in dimension_results.items():
            assert result.dimension == dim
            assert 0.0 <= result.overall_score <= 5.0

    @pytest.mark.asyncio
    async def test_phase2_handles_errors(self, sample_paper):
        """Test Phase 2 handles sub-agent failures gracefully."""
        from tests.conftest import MockLLMClient, MockRAGClient
        from wuya_agents.subagents import CUDOSSubAgent, InnovationSubAgent, MethodSubAgent, EvidenceSubAgent, ApplicationSubAgent
        from wuya_agents.dea_subagent import DEASubAgent, DEAEngine

        # Create failing LLM client
        mock_llm = MockLLMClient(should_fail=True)
        mock_rag = MockRAGClient()

        cudos = CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        innovation = InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        method = MethodSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        evidence = EvidenceSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        application = ApplicationSubAgent(llm_client=mock_llm, rag_client=mock_rag)

        router = TwoPhaseRouter(
            cudos_agent=cudos,
            innovation_agent=innovation,
            method_agent=method,
            evidence_agent=evidence,
            application_agent=application,
            rag_client=mock_rag,
        )

        dimension_results, errors = await router._phase2_parallel_evaluation(sample_paper)

        # Should still return results (with or without errors depending on implementation)
        assert len(dimension_results) == 4
        # Errors may be empty or contain failures
        assert isinstance(errors, list)

    # ========================================================================
    # Score Vector Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_build_score_vector(self, two_phase_router, sample_paper):
        """Test score vector building."""
        dimension_results, _ = await two_phase_router._phase2_parallel_evaluation(sample_paper)
        cudos_result = await two_phase_router._phase1_cudos_gatekeeping(sample_paper)

        score_vector = two_phase_router._build_score_vector(dimension_results, cudos_result)

        assert score_vector is not None
        assert hasattr(score_vector, 'innovation')
        assert hasattr(score_vector, 'method')
        assert hasattr(score_vector, 'evidence')
        assert hasattr(score_vector, 'application')
        assert hasattr(score_vector, 'cudos')

    # ========================================================================
    # Full Workflow Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_full_workflow_completes(self, two_phase_router, sample_paper):
        """Test full evaluation workflow completes."""
        report = await two_phase_router.route(sample_paper)

        assert isinstance(report, EvaluationReport)
        assert report.paper_id == sample_paper.paper_id

    @pytest.mark.asyncio
    async def test_full_workflow_passes_cudos(self, two_phase_router, sample_paper):
        """Test workflow passes CUDOS for valid paper."""
        report = await two_phase_router.route(sample_paper)

        assert report.cudos_passed is True
        assert report.status == "completed"

    @pytest.mark.asyncio
    async def test_full_workflow_dimension_scores(self, two_phase_router, sample_paper):
        """Test workflow produces dimension scores."""
        report = await two_phase_router.route(sample_paper)

        assert isinstance(report.dimension_scores, dict)
        assert "innovation" in report.dimension_scores
        assert "method" in report.dimension_scores
        assert "evidence" in report.dimension_scores
        assert "application" in report.dimension_scores

    @pytest.mark.asyncio
    async def test_full_workflow_tier_estimate(self, two_phase_router, sample_paper):
        """Test workflow produces tier estimate."""
        report = await two_phase_router.route(sample_paper)

        assert report.tier_estimate is not None
        assert len(report.tier_estimate) > 0

    @pytest.mark.asyncio
    async def test_full_workflow_overall_score(self, two_phase_router, sample_paper):
        """Test workflow calculates overall score."""
        report = await two_phase_router.route(sample_paper)

        assert isinstance(report.overall_score, float)
        assert 0.0 <= report.overall_score <= 5.0

    # ========================================================================
    # Veto Workflow Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_veto_workflow_returns_vetoed_report(self, sample_paper):
        """Test that vetoed papers return vetoed report."""
        from tests.conftest import MockLLMClient, MockRAGClient
        from wuya_agents.subagents import CUDOSSubAgent, InnovationSubAgent, MethodSubAgent, EvidenceSubAgent, ApplicationSubAgent
        from wuya_agents.dea_subagent import DEASubAgent, DEAEngine

        # Create failing CUDOS response
        failing_response = '''{
            "communalism": {"score": 1.0, "issues": ["Severe violation"]},
            "universalism": {"score": 4.0, "issues": []},
            "disinterestedness": {"score": 4.0, "issues": []},
            "organized_skepticism": {"score": 4.0, "issues": []}
        }'''

        mock_llm = MockLLMClient(responses={"communalism": failing_response})
        mock_rag = MockRAGClient()

        cudos = CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag, veto_threshold=2.0)
        innovation = InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        method = MethodSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        evidence = EvidenceSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        application = ApplicationSubAgent(llm_client=mock_llm, rag_client=mock_rag)

        router = TwoPhaseRouter(
            cudos_agent=cudos,
            innovation_agent=innovation,
            method_agent=method,
            evidence_agent=evidence,
            application_agent=application,
            rag_client=mock_rag,
        )

        report = await router.route(sample_paper)

        assert report.status == "vetoed"
        assert report.cudos_passed is False
        assert report.veto_reason is not None
        assert report.tier_estimate == "N/A"

    # ========================================================================
    # DEA Integration Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_dea_analysis_with_sufficient_references(
        self, two_phase_router, sample_paper, sample_reference_papers
    ):
        """Test DEA analysis with sufficient reference papers."""
        report = await two_phase_router.route(
            sample_paper,
            target_journal="Nature Machine Intelligence",
            reference_papers=sample_reference_papers
        )

        # Should have DEA summary
        assert report.dea_summary is not None
        assert hasattr(report.dea_summary, 'efficiency_score')

    @pytest.mark.asyncio
    async def test_dea_skipped_without_references(self, two_phase_router, sample_paper):
        """Test DEA is skipped when reference papers not provided."""
        report = await two_phase_router.route(
            sample_paper,
            target_journal="Nature",
            reference_papers=None
        )

        # DEA should be skipped
        assert report.dea_summary is None

    @pytest.mark.asyncio
    async def test_dea_skipped_without_journal(self, two_phase_router, sample_paper, sample_reference_papers):
        """Test DEA is skipped when target journal not provided."""
        report = await two_phase_router.route(
            sample_paper,
            target_journal=None,
            reference_papers=sample_reference_papers
        )

        # DEA should be skipped
        assert report.dea_summary is None

    @pytest.mark.asyncio
    async def test_dea_skipped_without_agent(self, sample_paper, sample_reference_papers):
        """Test DEA is skipped when no DEA agent configured."""
        from tests.conftest import MockLLMClient, MockRAGClient
        from wuya_agents.subagents import CUDOSSubAgent, InnovationSubAgent, MethodSubAgent, EvidenceSubAgent, ApplicationSubAgent

        mock_llm = MockLLMClient()
        mock_rag = MockRAGClient()

        cudos = CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        innovation = InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        method = MethodSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        evidence = EvidenceSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        application = ApplicationSubAgent(llm_client=mock_llm, rag_client=mock_rag)

        router = TwoPhaseRouter(
            cudos_agent=cudos,
            innovation_agent=innovation,
            method_agent=method,
            evidence_agent=evidence,
            application_agent=application,
            dea_agent=None,  # No DEA agent
            rag_client=mock_rag,
        )

        report = await router.route(
            sample_paper,
            target_journal="Nature",
            reference_papers=sample_reference_papers
        )

        # DEA should be skipped
        assert report.dea_summary is None

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_error_handling_subagent_failure(self, sample_paper):
        """Test error handling when sub-agent fails."""
        from tests.conftest import MockLLMClient, MockRAGClient
        from wuya_agents.subagents import CUDOSSubAgent, InnovationSubAgent, MethodSubAgent, EvidenceSubAgent, ApplicationSubAgent
        from wuya_agents.dea_subagent import DEASubAgent, DEAEngine

        # One agent will fail
        failing_llm = MockLLMClient(should_fail=True)
        good_llm = MockLLMClient()
        mock_rag = MockRAGClient()

        cudos = CUDOSSubAgent(llm_client=good_llm, rag_client=mock_rag)
        innovation = InnovationSubAgent(llm_client=failing_llm, rag_client=mock_rag)  # Will fail
        method = MethodSubAgent(llm_client=good_llm, rag_client=mock_rag)
        evidence = EvidenceSubAgent(llm_client=good_llm, rag_client=mock_rag)
        application = ApplicationSubAgent(llm_client=good_llm, rag_client=mock_rag)

        router = TwoPhaseRouter(
            cudos_agent=cudos,
            innovation_agent=innovation,
            method_agent=method,
            evidence_agent=evidence,
            application_agent=application,
            rag_client=mock_rag,
        )

        report = await router.route(sample_paper)

        # Should still complete (with or without errors depending on implementation)
        assert report.status == "completed"
        assert isinstance(report.errors, list)

    # ========================================================================
    # Report Generation Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_report_to_dict(self, two_phase_router, sample_paper):
        """Test report serialization to dict."""
        report = await two_phase_router.route(sample_paper)

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert report_dict["paper_id"] == sample_paper.paper_id
        assert report_dict["status"] in ["completed", "vetoed"]

    @pytest.mark.asyncio
    async def test_report_processing_time_recorded(self, two_phase_router, sample_paper):
        """Test processing time is recorded in report."""
        report = await two_phase_router.route(sample_paper)

        assert report.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_report_timestamp_present(self, two_phase_router, sample_paper):
        """Test timestamp is present in report."""
        report = await two_phase_router.route(sample_paper)

        assert report.timestamp is not None
        assert len(report.timestamp) > 0

    # ========================================================================
    # Recommendation Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_journal_matches_generated(self, two_phase_router, sample_paper):
        """Test journal matches are generated."""
        report = await two_phase_router.route(
            sample_paper,
            target_journal="Nature"
        )

        assert isinstance(report.journal_matches, list)

    @pytest.mark.asyncio
    async def test_improvement_suggestions_generated(self, two_phase_router, sample_paper):
        """Test improvement suggestions are generated."""
        report = await two_phase_router.route(sample_paper)

        assert isinstance(report.improvement_suggestions, list)

    # ========================================================================
    # Dimension Details Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_dimension_details_generated(self, two_phase_router, sample_paper):
        """Test dimension details are generated."""
        report = await two_phase_router.route(sample_paper)

        assert isinstance(report.dimension_details, dict)
        assert len(report.dimension_details) == 4

    # ========================================================================
    # Edge Cases
    # ========================================================================

    @pytest.mark.asyncio
    async def test_sequential_execution_mode(self, sample_paper, sample_reference_papers):
        """Test router with sequential (non-parallel) execution."""
        from tests.conftest import MockLLMClient, MockRAGClient
        from wuya_agents.subagents import CUDOSSubAgent, InnovationSubAgent, MethodSubAgent, EvidenceSubAgent, ApplicationSubAgent
        from wuya_agents.dea_subagent import DEASubAgent, DEAEngine

        mock_llm = MockLLMClient()
        mock_rag = MockRAGClient()

        cudos = CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        innovation = InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        method = MethodSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        evidence = EvidenceSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        application = ApplicationSubAgent(llm_client=mock_llm, rag_client=mock_rag)

        router = TwoPhaseRouter(
            cudos_agent=cudos,
            innovation_agent=innovation,
            method_agent=method,
            evidence_agent=evidence,
            application_agent=application,
            rag_client=mock_rag,
            enable_parallel=False,  # Sequential execution
        )

        report = await router.route(sample_paper)

        assert report.status == "completed"
        assert report.cudos_passed is True

    @pytest.mark.asyncio
    async def test_router_with_minimum_dea_papers(self, sample_paper):
        """Test DEA with exactly minimum reference papers."""
        from tests.conftest import MockLLMClient, MockRAGClient
        from wuya_agents.subagents import CUDOSSubAgent, InnovationSubAgent, MethodSubAgent, EvidenceSubAgent, ApplicationSubAgent
        from wuya_agents.dea_subagent import DEASubAgent, DEAEngine

        mock_llm = MockLLMClient()
        mock_rag = MockRAGClient()

        # Exactly 50 papers (minimum)
        reference_papers = [
            {
                "paper_id": f"ref_{i}",
                "scores": {
                    "innovation": 4.0,
                    "method": 3.5,
                    "evidence": 4.0,
                    "application": 3.5,
                    "cudos": 4.5
                },
                "citations": 50
            }
            for i in range(50)
        ]

        cudos = CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        innovation = InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        method = MethodSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        evidence = EvidenceSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        application = ApplicationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        dea = DEASubAgent(dea_engine=DEAEngine(min_reference_papers=50))

        router = TwoPhaseRouter(
            cudos_agent=cudos,
            innovation_agent=innovation,
            method_agent=method,
            evidence_agent=evidence,
            application_agent=application,
            dea_agent=dea,
            rag_client=mock_rag,
        )

        report = await router.route(
            sample_paper,
            target_journal="Nature",
            reference_papers=reference_papers
        )

        # Should have DEA results (at minimum threshold)
        assert report.dea_summary is not None
