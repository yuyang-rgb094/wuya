"""
Tests for DEA Engine

Author: WuYa Team
"""

import pytest
import numpy as np
from wuya_agents.dea_subagent import (
    DEAEngine,
    DEASubAgent,
    ScoreVector,
    DEAResult,
    DEAStatus,
)


class TestDEAEngine:
    """Test suite for DEAEngine."""

    def test_initialization(self, dea_engine):
        """Test engine initializes correctly."""
        assert dea_engine is not None
        assert dea_engine.min_reference_papers == 50

    def test_custom_min_papers(self):
        """Test engine with custom minimum papers."""
        engine = DEAEngine(min_reference_papers=30)
        assert engine.min_reference_papers == 30

    # ========================================================================
    # Score Transformation Tests
    # ========================================================================

    def test_transform_scores(self, dea_engine, sample_dea_score_vector, sample_reference_papers):
        """Test score transformation to DEA format."""
        ref_scores = [
            ScoreVector(**p["scores"])
            for p in sample_reference_papers[:10]
        ]

        target_in, target_out, ref_in, ref_out = dea_engine.transform_scores(
            sample_dea_score_vector, ref_scores
        )

        assert len(target_in) == 2  # Method flaws, Evidence gap
        assert len(target_out) == 2  # Innovation, Application
        assert ref_in.shape[0] == 10  # 10 reference papers
        assert ref_in.shape[1] == 2  # 2 inputs
        assert ref_out.shape[0] == 10
        assert ref_out.shape[1] == 2

    def test_transform_scores_ranges(self, dea_engine, sample_dea_score_vector, sample_reference_papers):
        """Test transformed scores are in valid ranges."""
        ref_scores = [
            ScoreVector(**p["scores"])
            for p in sample_reference_papers[:10]
        ]

        target_in, target_out, ref_in, ref_out = dea_engine.transform_scores(
            sample_dea_score_vector, ref_scores
        )

        # Inputs should be positive (6 - score)
        assert np.all(target_in >= 0)
        assert np.all(ref_in >= 0)

        # Outputs should be in 1-5 range
        assert np.all(target_out >= 1)
        assert np.all(target_out <= 5)

    # ========================================================================
    # Super-Efficiency Calculation Tests
    # ========================================================================

    def test_solve_super_efficiency_basic(self, dea_engine):
        """Test basic super-efficiency calculation."""
        target_in = np.array([1.5, 1.8])  # Low method flaws, low evidence gap
        target_out = np.array([4.5, 4.0])  # High innovation, high application

        ref_in = np.array([
            [2.0, 2.0],
            [1.8, 2.2],
            [2.1, 1.9],
        ])
        ref_out = np.array([
            [3.5, 3.5],
            [4.0, 3.8],
            [3.8, 4.2],
        ])

        theta, lambdas = dea_engine.solve_super_efficiency(
            target_in, target_out, ref_in, ref_out
        )

        assert isinstance(theta, (float, np.floating))
        assert theta > 0
        assert len(lambdas) == 3
        assert all(l >= 0 for l in lambdas)

    def test_solve_super_efficiency_returns_theta_greater_than_1(self, dea_engine):
        """Test that excellent DMU returns theta > 1."""
        # Target is better than all references
        target_in = np.array([0.5, 0.5])  # Very low flaws
        target_out = np.array([5.0, 5.0])  # Maximum outputs

        ref_in = np.array([
            [2.0, 2.0],
            [2.5, 2.5],
        ])
        ref_out = np.array([
            [3.0, 3.0],
            [3.5, 3.5],
        ])

        theta, _ = dea_engine.solve_super_efficiency(
            target_in, target_out, ref_in, ref_out
        )

        assert theta > 1.0

    def test_solve_super_efficiency_returns_theta_less_than_1(self, dea_engine):
        """Test that poor DMU returns theta < 1."""
        # Target is worse than references
        target_in = np.array([4.0, 4.0])  # High flaws
        target_out = np.array([1.0, 1.0])  # Low outputs

        ref_in = np.array([
            [2.0, 2.0],
            [2.5, 2.5],
        ])
        ref_out = np.array([
            [3.0, 3.0],
            [3.5, 3.5],
        ])

        theta, _ = dea_engine.solve_super_efficiency(
            target_in, target_out, ref_in, ref_out
        )

        assert theta < 1.0

    # ========================================================================
    # Bootstrap Confidence Interval Tests
    # ========================================================================

    def test_bootstrap_ci_basic(self, dea_engine):
        """Test bootstrap confidence interval calculation."""
        target_in = np.array([1.5, 1.8])
        target_out = np.array([4.0, 3.5])

        ref_in = np.array([
            [2.0, 2.0],
            [1.8, 2.2],
            [2.1, 1.9],
            [2.3, 2.0],
            [1.9, 2.1],
        ])
        ref_out = np.array([
            [3.5, 3.5],
            [4.0, 3.8],
            [3.8, 4.2],
            [3.6, 3.9],
            [4.1, 3.7],
        ])

        (ci_lower, ci_upper), std = dea_engine.bootstrap_confidence_interval(
            target_in, target_out, ref_in, ref_out,
            n_iterations=50  # Fewer for faster test
        )

        assert ci_lower <= ci_upper
        assert std >= 0

    def test_bootstrap_ci_with_reproducible_seed(self, dea_engine):
        """Test bootstrap with seed for reproducibility."""
        target_in = np.array([1.5, 1.8])
        target_out = np.array([4.0, 3.5])

        ref_in = np.array([
            [2.0, 2.0],
            [1.8, 2.2],
            [2.1, 1.9],
        ])
        ref_out = np.array([
            [3.5, 3.5],
            [4.0, 3.8],
            [3.8, 4.2],
        ])

        # Run twice with same seed
        np.random.seed(42)
        result1 = dea_engine.bootstrap_confidence_interval(
            target_in, target_out, ref_in, ref_out, n_iterations=30
        )

        np.random.seed(42)
        result2 = dea_engine.bootstrap_confidence_interval(
            target_in, target_out, ref_in, ref_out, n_iterations=30
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(result1[0], result2[0])

    # ========================================================================
    # Analysis Tests
    # ========================================================================

    def test_analyze_insufficient_data(self, dea_engine, sample_dea_score_vector):
        """Test analyze with insufficient reference papers."""
        # Only 5 papers, minimum is 50
        ref_scores = [
            ScoreVector(innovation=4.0, method=3.5, evidence=4.0, application=3.5, cudos=4.5)
            for _ in range(5)
        ]

        result = dea_engine.analyze(sample_dea_score_vector, ref_scores)

        assert result.status == DEAStatus.INSUFFICIENT_DATA
        assert result.efficiency_score == 0.0

    def test_analyze_success(self, dea_engine, sample_dea_score_vector, sample_reference_papers):
        """Test successful analysis."""
        ref_scores = [
            ScoreVector(**p["scores"])
            for p in sample_reference_papers
        ]

        result = dea_engine.analyze(sample_dea_score_vector, ref_scores)

        assert result.status in [DEAStatus.SUCCESS, DEAStatus.FRONTIER_UNREACHABLE]
        assert result.efficiency_score > 0
        assert result.reference_set_size == len(ref_scores)

    def test_analyze_frontier_detection(self, dea_engine, sample_reference_papers):
        """Test frontier detection logic."""
        # Create a very strong paper (should be on frontier)
        strong_score = ScoreVector(
            innovation=5.0,
            method=5.0,
            evidence=5.0,
            application=5.0,
            cudos=5.0
        )

        ref_scores = [
            ScoreVector(**p["scores"])
            for p in sample_reference_papers
        ]

        result = dea_engine.analyze(strong_score, ref_scores)

        # Strong paper should be on or near frontier
        assert result.is_on_frontier or result.efficiency_score >= 1.0

    # ========================================================================
    # Edge Cases
    # ========================================================================

    def test_analyze_all_identical_scores(self, dea_engine, sample_reference_papers):
        """Test analysis with identical scores."""
        target = ScoreVector(
            innovation=4.0,
            method=4.0,
            evidence=4.0,
            application=4.0,
            cudos=4.0
        )

        ref_scores = [
            ScoreVector(**p["scores"])
            for p in sample_reference_papers
        ]

        result = dea_engine.analyze(target, ref_scores)

        assert result.status in [DEAStatus.SUCCESS, DEAStatus.FRONTIER_UNREACHABLE]

    def test_analyze_extreme_scores(self, dea_engine, sample_reference_papers):
        """Test analysis with extreme (max/min) scores."""
        extreme_score = ScoreVector(
            innovation=1.0,  # Minimum
            method=5.0,
            evidence=1.0,
            application=5.0,
            cudos=4.0
        )

        ref_scores = [
            ScoreVector(**p["scores"])
            for p in sample_reference_papers
        ]

        result = dea_engine.analyze(extreme_score, ref_scores)

        # Score should be valid (either < 1, = 1, or > 1 depending on implementation)
        assert result.efficiency_score > 0

    def test_transform_with_zero_inputs(self, dea_engine):
        """Test transformation with edge case scores."""
        score = ScoreVector(
            innovation=5.0,
            method=5.0,  # Max score -> 6-5=1 (not zero)
            evidence=5.0,
            application=5.0,
            cudos=5.0
        )

        ref_scores = [
            ScoreVector(innovation=4.0, method=3.5, evidence=4.0, application=3.5, cudos=4.5)
            for _ in range(10)
        ]

        target_in, _, _, _ = dea_engine.transform_scores(score, ref_scores)

        # Should not produce zero inputs
        assert all(target_in > 0)


class TestDEASubAgent:
    """Test suite for DEASubAgent."""

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_dea_agent_initialization(self, dea_agent):
        """Test DEA agent initializes correctly."""
        assert dea_agent is not None
        assert dea_agent.dea_engine is not None

    # ========================================================================
    # Evaluation Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_evaluate_insufficient_data(self, dea_agent, sample_paper, sample_reference_papers, sample_dea_score_vector):
        """Test evaluation with insufficient reference papers."""
        # Only 10 papers, minimum is 50
        small_refs = sample_reference_papers[:10]

        result = await dea_agent.evaluate(
            paper_id=sample_paper.paper_id,
            score_vector=sample_dea_score_vector,
            target_journal="Nature",
            reference_papers=small_refs
        )

        assert "dea_result" in result
        assert result["dea_result"]["status"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_evaluate_success(self, dea_agent, sample_paper, sample_reference_papers, sample_dea_score_vector):
        """Test successful evaluation."""
        result = await dea_agent.evaluate(
            paper_id=sample_paper.paper_id,
            score_vector=sample_dea_score_vector,
            target_journal="Nature",
            reference_papers=sample_reference_papers
        )

        assert "dea_result" in result
        assert "efficiency_score" in result["dea_result"]
        assert "confidence_interval" in result["dea_result"]

    @pytest.mark.asyncio
    async def test_evaluate_with_path_a_estimate(self, dea_agent, sample_paper, sample_reference_papers, sample_dea_score_vector):
        """Test evaluation with Path A estimate for cross-validation."""
        result = await dea_agent.evaluate(
            paper_id=sample_paper.paper_id,
            score_vector=sample_dea_score_vector,
            target_journal="Nature",
            reference_papers=sample_reference_papers,
            path_a_estimate="Q1-A"
        )

        assert "consistency_with_path_a" in result

    # ========================================================================
    # Interpretation Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_interpretation_high_efficiency(self, dea_agent):
        """Test interpretation for high efficiency score."""
        dea_result = DEAResult(
            efficiency_score=1.15,
            confidence_interval=(1.05, 1.25),
            status=DEAStatus.SUCCESS,
            reference_set_size=100,
            is_on_frontier=True,
            bootstrap_std=0.05
        )

        score_vector = ScoreVector(
            innovation=4.5, method=4.0, evidence=4.0, application=4.5, cudos=4.8
        )

        interpretation = await dea_agent._interpret_result(
            dea_result, score_vector, "Nature", "Q1-A"
        )

        assert "explanation" in interpretation
        assert "above" in interpretation["explanation"].lower() or "frontier" in interpretation["explanation"].lower()

    @pytest.mark.asyncio
    async def test_interpretation_low_efficiency(self, dea_agent):
        """Test interpretation for low efficiency score."""
        dea_result = DEAResult(
            efficiency_score=0.75,
            confidence_interval=(0.65, 0.85),
            status=DEAStatus.SUCCESS,
            reference_set_size=100,
            is_on_frontier=False,
            bootstrap_std=0.08
        )

        score_vector = ScoreVector(
            innovation=3.0, method=3.0, evidence=3.0, application=3.0, cudos=4.0
        )

        interpretation = await dea_agent._interpret_result(
            dea_result, score_vector, "Nature", "Q2"
        )

        assert "explanation" in interpretation
        assert "below" in interpretation["explanation"].lower() or "frontier" in interpretation["explanation"].lower()

    @pytest.mark.asyncio
    async def test_interpretation_insufficient_data(self, dea_agent):
        """Test interpretation for insufficient data."""
        dea_result = DEAResult(
            efficiency_score=0.0,
            confidence_interval=(0.0, 0.0),
            status=DEAStatus.INSUFFICIENT_DATA,
            reference_set_size=10,
            is_on_frontier=False,
            bootstrap_std=0.0
        )

        score_vector = ScoreVector(
            innovation=4.0, method=4.0, evidence=4.0, application=4.0, cudos=4.0
        )

        interpretation = await dea_agent._interpret_result(
            dea_result, score_vector, "Nature", None
        )

        assert "explanation" in interpretation
        assert interpretation["rag_triggered"] is True

    # ========================================================================
    # Consistency Check Tests
    # ========================================================================

    def test_check_consistency_match(self, dea_agent):
        """Test consistency check when Path A and DEA agree."""
        dea_result = DEAResult(
            efficiency_score=1.1,
            confidence_interval=(1.0, 1.2),
            status=DEAStatus.SUCCESS,
            reference_set_size=100,
            is_on_frontier=True,
            bootstrap_std=0.05
        )

        result = dea_agent._check_consistency(dea_result, "Q1-A")

        assert "consistent" in result
        assert "dea_tier" in result

    def test_check_consistency_mismatch(self, dea_agent):
        """Test consistency check when Path A and DEA disagree."""
        dea_result = DEAResult(
            efficiency_score=0.7,  # Below frontier
            confidence_interval=(0.6, 0.8),
            status=DEAStatus.SUCCESS,
            reference_set_size=100,
            is_on_frontier=False,
            bootstrap_std=0.1
        )

        result = dea_agent._check_consistency(dea_result, "Q1-A")

        assert result["consistent"] is False
        assert "discrepancy" in result

    # ========================================================================
    # Recommendation Tests
    # ========================================================================

    def test_generate_recommendation_strong(self, dea_agent):
        """Test recommendation for strong paper."""
        dea_result = DEAResult(
            efficiency_score=1.15,
            confidence_interval=(1.05, 1.25),
            status=DEAStatus.SUCCESS,
            reference_set_size=100,
            is_on_frontier=True,
            bootstrap_std=0.05
        )

        recommendation = dea_agent._generate_recommendation(dea_result, None)

        assert "strong" in recommendation.lower() or "exceed" in recommendation.lower()

    def test_generate_recommendation_weak(self, dea_agent):
        """Test recommendation for weak paper."""
        dea_result = DEAResult(
            efficiency_score=0.7,
            confidence_interval=(0.6, 0.8),
            status=DEAStatus.SUCCESS,
            reference_set_size=100,
            is_on_frontier=False,
            bootstrap_std=0.1
        )

        recommendation = dea_agent._generate_recommendation(dea_result, None)

        assert "not" in recommendation.lower() or "lower" in recommendation.lower()

    def test_generate_recommendation_inconsistent(self, dea_agent):
        """Test recommendation when Path A and DEA inconsistent."""
        dea_result = DEAResult(
            efficiency_score=1.0,
            confidence_interval=(0.9, 1.1),
            status=DEAStatus.SUCCESS,
            reference_set_size=100,
            is_on_frontier=True,
            bootstrap_std=0.05
        )

        consistency = {
            "consistent": False,
            "discrepancy": "Path A: Q1-A, DEA: B"
        }

        recommendation = dea_agent._generate_recommendation(dea_result, consistency)

        assert "diverge" in recommendation.lower() or "review" in recommendation.lower()


# Helper fixture
@pytest.fixture
def sample_dea_score_vector():
    """Create sample score vector for DEA testing."""
    return ScoreVector(
        innovation=4.5,
        method=4.2,
        evidence=3.5,
        application=4.0,
        cudos=4.8,
    )
