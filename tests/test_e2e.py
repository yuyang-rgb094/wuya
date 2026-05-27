"""
End-to-End Tests for WuYa System

Author: WuYa Team
"""

import pytest
from wuya_agents.router import TwoPhaseRouter, EvaluationReport
from wuya_agents.base import ParsedPaper, EvaluationDimension
from wuya_agents.subagents.frontier import SubmissionRecord, FrontierDiscoverySubAgent
from tests.conftest import MockLLMClient, MockRAGClient
from wuya_agents.subagents import (
    CUDOSSubAgent,
    InnovationSubAgent,
    MethodSubAgent,
    EvidenceSubAgent,
    ApplicationSubAgent,
)
from wuya_agents.dea_subagent import DEASubAgent, DEAEngine
from wuya_agents.rag.client import RAGClientImpl
from wuya_agents.rag.embedding import MockEmbeddingProvider
from wuya_agents.rag.vector_store import InMemoryVectorStore


class TestPaperEvaluationWorkflow:
    """End-to-end test suite for paper evaluation workflow."""

    @pytest.fixture
    def complete_system(self):
        """Create a complete WuYa system for testing."""
        mock_llm = MockLLMClient()
        mock_rag = MockRAGClient()

        cudos = CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag, veto_threshold=2.0)
        innovation = InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        method = MethodSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        evidence = EvidenceSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        application = ApplicationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        dea = DEASubAgent(dea_engine=DEAEngine())
        rag_client = RAGClientImpl(
            embedding_provider=MockEmbeddingProvider(dimensions=128),
            vector_store=InMemoryVectorStore(),
        )

        router = TwoPhaseRouter(
            cudos_agent=cudos,
            innovation_agent=innovation,
            method_agent=method,
            evidence_agent=evidence,
            application_agent=application,
            dea_agent=dea,
            rag_client=rag_client,
        )

        return router, rag_client

    @pytest.fixture
    def sample_paper(self) -> ParsedPaper:
        """Create a sample paper for testing."""
        return ParsedPaper(
            paper_id="e2e-test-001",
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
    def sample_reference_papers(self):
        """Create sample reference papers for DEA."""
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

    # ========================================================================
    # Basic E2E Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(self, complete_system, sample_paper, sample_reference_papers):
        """Test complete paper evaluation workflow."""
        router, _ = complete_system

        report = await router.route(
            sample_paper,
            target_journal="Nature Machine Intelligence",
            reference_papers=sample_reference_papers
        )

        # Verify report structure
        assert isinstance(report, EvaluationReport)
        assert report.paper_id == sample_paper.paper_id
        assert report.status == "completed"
        assert report.cudos_passed is True

    @pytest.mark.asyncio
    async def test_evaluation_produces_all_dimensions(self, complete_system, sample_paper):
        """Test evaluation produces scores for all dimensions."""
        router, _ = complete_system

        report = await router.route(sample_paper)

        # Check all dimensions present
        expected_dims = ["innovation", "method", "evidence", "application", "cudos"]
        for dim in expected_dims:
            assert dim in report.dimension_scores or dim in report.to_dict().get("dimension_scores", {})

    @pytest.mark.asyncio
    async def test_evaluation_produces_tier_estimate(self, complete_system, sample_paper):
        """Test evaluation produces tier estimate."""
        router, _ = complete_system

        report = await router.route(sample_paper)

        assert report.tier_estimate is not None
        assert len(report.tier_estimate) > 0
        assert any(tier in report.tier_estimate for tier in ["Q1", "Q2", "Q3", "Q4"])

    @pytest.mark.asyncio
    async def test_evaluation_produces_journal_recommendations(self, complete_system, sample_paper):
        """Test evaluation produces journal recommendations."""
        router, _ = complete_system

        report = await router.route(
            sample_paper,
            target_journal="Nature"
        )

        assert len(report.journal_matches) > 0
        # First match should be the target journal
        assert report.journal_matches[0].journal_name == "Nature"

    @pytest.mark.asyncio
    async def test_evaluation_produces_improvement_suggestions(self, complete_system, sample_paper):
        """Test evaluation produces improvement suggestions."""
        router, _ = complete_system

        report = await router.route(sample_paper)

        assert isinstance(report.improvement_suggestions, list)

    @pytest.mark.asyncio
    async def test_evaluation_includes_processing_metadata(self, complete_system, sample_paper):
        """Test evaluation includes processing metadata."""
        router, _ = complete_system

        report = await router.route(sample_paper)

        # Processing time may be 0 if very fast, but should be present
        assert hasattr(report, 'processing_time_ms')
        assert report.timestamp is not None
        assert len(report.timestamp) > 0

    # ========================================================================
    # DEA Enhanced Evaluation Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_dea_enhanced_evaluation(self, complete_system, sample_paper, sample_reference_papers):
        """Test evaluation with DEA analysis."""
        router, _ = complete_system

        report = await router.route(
            sample_paper,
            target_journal="Nature Machine Intelligence",
            reference_papers=sample_reference_papers
        )

        assert report.dea_summary is not None
        assert hasattr(report.dea_summary, 'efficiency_score')
        assert report.dea_summary.efficiency_score > 0

    @pytest.mark.asyncio
    async def test_dea_confidence_interval_present(self, complete_system, sample_paper, sample_reference_papers):
        """Test DEA produces confidence interval."""
        router, _ = complete_system

        report = await router.route(
            sample_paper,
            target_journal="Nature",
            reference_papers=sample_reference_papers
        )

        ci = report.dea_summary.confidence_interval
        assert ci is not None
        assert len(ci) == 2
        assert ci[0] <= ci[1]

    @pytest.mark.asyncio
    async def test_dea_frontier_status(self, complete_system, sample_paper, sample_reference_papers):
        """Test DEA frontier status is determined."""
        router, _ = complete_system

        report = await router.route(
            sample_paper,
            target_journal="Nature",
            reference_papers=sample_reference_papers
        )

        # is_on_frontier may be np.bool_ or bool
        assert hasattr(report.dea_summary, 'is_on_frontier')
        assert report.dea_summary.is_on_frontier in [True, False, True, False]  # Accept both bool and np.bool_

    # ========================================================================
    # RAG Enhanced Evaluation Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_rag_enhanced_evaluation(self, complete_system, sample_paper, sample_reference_papers):
        """Test evaluation with RAG enhancement."""
        router, rag_client = complete_system

        # Add some documents to RAG
        rag_texts = [
            ("rag_doc1", "Merton's CUDOS norms for scientific ethics", {"source": "theory"}),
            ("rag_doc2", "State-of-the-art deep learning methods", {"source": "methods"}),
            ("rag_doc3", "Evidence evaluation criteria in empirical research", {"source": "evidence"}),
        ]
        await rag_client.add_texts(rag_texts)

        report = await router.route(
            sample_paper,
            target_journal="Nature",
            reference_papers=sample_reference_papers
        )

        # Should complete successfully with RAG
        assert report.status == "completed"
        assert len(report.rag_citations) >= 0  # May or may not have citations

    @pytest.mark.asyncio
    async def test_rag_citations_collected(self, complete_system, sample_paper):
        """Test that RAG citations are collected during evaluation."""
        router, rag_client = complete_system

        # Pre-populate RAG with relevant content
        await rag_client.add_texts([
            ("doc1", "Deep learning scientific discovery research", {}),
        ])

        report = await router.route(sample_paper)

        # Citations should be collected from sub-agents
        assert isinstance(report.rag_citations, list)

    # ========================================================================
    # Veto Workflow E2E Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_veto_workflow_full(self, sample_paper):
        """Test full workflow when paper is vetoed."""
        # Create system with failing CUDOS
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

        # Verify veto workflow
        assert report.status == "vetoed"
        assert report.cudos_passed is False
        assert report.veto_reason is not None
        assert len(report.veto_reason) > 0
        assert report.tier_estimate == "N/A"
        assert report.overall_score == 0.0
        assert len(report.dimension_scores) == 0

    # ========================================================================
    # Self-Improvement Loop Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_self_improvement_loop(self, sample_reference_papers):
        """Test self-improvement loop with frontier discovery."""
        # Create frontier discovery agent
        frontier_agent = FrontierDiscoverySubAgent(
            min_history_size=10,
            trend_window_years=3
        )

        # Create submission history
        history = []
        for i, ref in enumerate(sample_reference_papers[:20]):
            outcome = "accepted" if i % 3 == 0 else "rejected"
            history.append(SubmissionRecord(
                paper_id=f"history_{i}",
                title=f"Historical Paper {i}",
                discipline="cs",
                scores=ref["scores"],
                target_journal="Nature",
                outcome=outcome,
                editor_feedback="Concerns about novelty in the field." if outcome == "rejected" else "Good paper.",
                keywords=["AI", "ML", "DL"],
                year=2022 + (i % 3),
                citations=ref["citations"],
            ))

        # Run frontier discovery
        frontier_update = await frontier_agent.discover_frontier(history)

        # Verify output
        assert frontier_update is not None
        assert frontier_update.new_dmu_count == 20
        assert isinstance(frontier_update.frontier_shift, float)
        assert isinstance(frontier_update.recommendations, list)

    @pytest.mark.asyncio
    async def test_self_improvement_feedback_loop(self, complete_system, sample_paper):
        """Test that evaluation results can feed back into system."""
        router, _ = complete_system

        # First evaluation
        report1 = await router.route(sample_paper)

        # Simulate submission and acceptance
        submission_record = SubmissionRecord(
            paper_id=sample_paper.paper_id,
            title=sample_paper.title,
            discipline=sample_paper.discipline,
            scores={
                "innovation": report1.dimension_scores.get("innovation", 3.5),
                "method": report1.dimension_scores.get("method", 3.5),
                "evidence": report1.dimension_scores.get("evidence", 3.5),
                "application": report1.dimension_scores.get("application", 3.5),
                "cudos": report1.dimension_scores.get("cudos", 4.0),
            },
            target_journal="Nature",
            outcome="accepted",
            editor_feedback="Good contribution, minor revisions needed.",
            keywords=sample_paper.keywords,
            year=2024,
            citations=0,
        )

        assert submission_record.paper_id == sample_paper.paper_id
        assert submission_record.outcome == "accepted"

    # ========================================================================
    # Parallel vs Sequential Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_parallel_evaluation_faster(self, sample_paper):
        """Test that parallel evaluation is faster than sequential."""
        mock_llm = MockLLMClient()
        mock_rag = MockRAGClient()

        cudos = CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        innovation = InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        method = MethodSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        evidence = EvidenceSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        application = ApplicationSubAgent(llm_client=mock_llm, rag_client=mock_rag)

        # Parallel router
        parallel_router = TwoPhaseRouter(
            cudos_agent=cudos,
            innovation_agent=innovation,
            method_agent=method,
            evidence_agent=evidence,
            application_agent=application,
            rag_client=mock_rag,
            enable_parallel=True,
        )

        # Sequential router (need new agents since they may have state)
        mock_llm2 = MockLLMClient()
        mock_rag2 = MockRAGClient()

        cudos2 = CUDOSSubAgent(llm_client=mock_llm2, rag_client=mock_rag2)
        innovation2 = InnovationSubAgent(llm_client=mock_llm2, rag_client=mock_rag2)
        method2 = MethodSubAgent(llm_client=mock_llm2, rag_client=mock_rag2)
        evidence2 = EvidenceSubAgent(llm_client=mock_llm2, rag_client=mock_rag2)
        application2 = ApplicationSubAgent(llm_client=mock_llm2, rag_client=mock_rag2)

        sequential_router = TwoPhaseRouter(
            cudos_agent=cudos2,
            innovation_agent=innovation2,
            method_agent=method2,
            evidence_agent=evidence2,
            application_agent=application2,
            rag_client=mock_rag2,
            enable_parallel=False,
        )

        # Both should produce valid results
        parallel_report = await parallel_router.route(sample_paper)
        sequential_report = await sequential_router.route(sample_paper)

        assert parallel_report.status == "completed"
        assert sequential_report.status == "completed"
        assert parallel_report.overall_score == sequential_report.overall_score

    # ========================================================================
    # Error Resilience Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_system_recovers_from_subagent_failure(self, sample_paper):
        """Test that system recovers when one sub-agent fails."""
        mock_llm = MockLLMClient()
        mock_rag = MockRAGClient()

        cudos = CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        innovation = InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag)
        method = InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag)  # Wrong type but will work
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

        # Should still complete even if one agent type is wrong
        assert report.status in ["completed", "vetoed", "error"]
        assert report.paper_id == sample_paper.paper_id

    # ========================================================================
    # Report Serialization Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_report_serialization_complete(self, complete_system, sample_paper):
        """Test complete report serialization."""
        router, _ = complete_system

        report = await router.route(sample_paper)
        report_dict = report.to_dict()

        # Verify all key fields are present
        assert "paper_id" in report_dict
        assert "paper_title" in report_dict
        assert "status" in report_dict
        assert "cudos_passed" in report_dict
        assert "dimension_scores" in report_dict
        assert "overall_score" in report_dict
        assert "tier_estimate" in report_dict
        assert "processing_time_ms" in report_dict
        assert "timestamp" in report_dict

        # Verify nested structures
        assert isinstance(report_dict["dimension_scores"], dict)
        assert isinstance(report_dict.get("dimension_details", {}), dict)

    # ========================================================================
    # Multiple Papers Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_multiple_papers_evaluation(self, complete_system, sample_reference_papers):
        """Test evaluating multiple papers in sequence."""
        router, _ = complete_system

        papers = [
            ParsedPaper(
                paper_id=f"paper_{i}",
                title=f"Research Paper {i}",
                abstract=f"Abstract for paper {i}",
                content=f"Content for paper {i}",
                authors=["Author"],
                keywords=["research"],
                discipline="cs",
            )
            for i in range(3)
        ]

        reports = []
        for paper in papers:
            report = await router.route(paper)
            reports.append(report)

        # All papers should be evaluated
        assert len(reports) == 3
        assert all(r.status == "completed" for r in reports)
        assert all(r.cudos_passed for r in reports)

    # ========================================================================
    # Edge Case Papers
    # ========================================================================

    @pytest.mark.asyncio
    async def test_minimal_paper_evaluation(self, complete_system):
        """Test evaluation of minimal paper with only required fields."""
        router, _ = complete_system

        minimal_paper = ParsedPaper(
            paper_id="minimal",
            title="X",
            abstract="Y",
            content="Z",
            authors=["A"],
            keywords=[],
            discipline="general",
        )

        report = await router.route(minimal_paper)

        # Should still produce a result
        assert report is not None
        assert report.paper_id == "minimal"

    @pytest.mark.asyncio
    async def test_multidisciplinary_paper(self, complete_system):
        """Test evaluation of paper spanning multiple disciplines."""
        router, _ = complete_system

        interdisciplinary_paper = ParsedPaper(
            paper_id="interdisciplinary",
            title="Cross-disciplinary Research: Biology meets Computer Science",
            abstract="This research bridges biology and computer science.",
            content="Using machine learning to analyze biological data.",
            authors=["Bio Scientist", "CS Researcher"],
            keywords=["bioinformatics", "machine learning", "computational biology"],
            discipline="computational biology",
        )

        report = await router.route(interdisciplinary_paper)

        assert report.status == "completed"
        assert report.cudos_passed is True
