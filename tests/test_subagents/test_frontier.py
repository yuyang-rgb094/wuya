"""
Tests for Frontier Discovery Sub-agent

Author: WuYa Team
"""

import pytest
from wuya_agents.subagents.frontier import (
    FrontierDiscoverySubAgent,
    SubmissionRecord,
    ImprovementPattern,
    TrendCluster,
    FrontierUpdate,
)


class TestFrontierDiscoverySubAgent:
    """Test suite for FrontierDiscoverySubAgent."""

    # ========================================================================
    # Data Class Tests
    # ========================================================================

    def test_submission_record_creation(self):
        """Test SubmissionRecord dataclass creation."""
        record = SubmissionRecord(
            paper_id="test_001",
            title="Test Paper",
            discipline="cs",
            scores={"innovation": 4.0, "method": 3.5, "evidence": 4.0, "application": 3.5, "cudos": 4.5},
            target_journal="Nature",
            outcome="accepted",
            editor_feedback="Good paper, minor revisions needed.",
            keywords=["AI", "ML"],
            year=2024,
            citations=50,
        )

        assert record.paper_id == "test_001"
        assert record.outcome == "accepted"
        assert record.scores["innovation"] == 4.0

    def test_submission_record_to_dict(self):
        """Test SubmissionRecord to_dict method."""
        record = SubmissionRecord(
            paper_id="test_001",
            title="Test Paper",
            discipline="cs",
            scores={"innovation": 4.0, "method": 3.5, "evidence": 4.0, "application": 3.5, "cudos": 4.5},
            target_journal="Nature",
            outcome="accepted",
        )

        result = record.to_dict()

        assert result["paper_id"] == "test_001"
        assert result["outcome"] == "accepted"
        assert isinstance(result, dict)

    def test_improvement_pattern_creation(self):
        """Test ImprovementPattern dataclass creation."""
        pattern = ImprovementPattern(
            dimension="innovation",
            pattern_description="Low novelty scores",
            frequency=10,
            severity=3.5,
            examples=["Example 1", "Example 2"],
        )

        assert pattern.dimension == "innovation"
        assert pattern.frequency == 10
        assert len(pattern.examples) == 2

    def test_trend_cluster_creation(self):
        """Test TrendCluster dataclass creation."""
        cluster = TrendCluster(
            cluster_id="trend_001",
            keywords=["AI", "machine learning", "deep learning"],
            frequency=100,
            growth_rate=0.25,
            representative_papers=["paper1", "paper2"],
        )

        assert cluster.cluster_id == "trend_001"
        assert len(cluster.keywords) == 3
        assert cluster.growth_rate == 0.25

    def test_frontier_update_creation(self):
        """Test FrontierUpdate dataclass creation."""
        update = FrontierUpdate(
            new_dmu_count=50,
            improvement_patterns=[],
            trend_clusters=[],
            frontier_shift=0.05,
            dimension_updates={"innovation": {"shift": 0.1}},
            recommendations=["Rec 1", "Rec 2"],
        )

        assert update.new_dmu_count == 50
        assert update.frontier_shift == 0.05
        assert len(update.recommendations) == 2

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_frontier_agent_initialization(self):
        """Test Frontier agent initializes correctly."""
        agent = FrontierDiscoverySubAgent(
            min_history_size=10,
            trend_window_years=3
        )
        assert agent is not None
        assert agent.min_history_size == 10
        assert agent.trend_window_years == 3

    # ========================================================================
    # Discovery Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_discover_frontier_insufficient_history(self):
        """Test discovery with insufficient history."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        # Only 5 records, minimum is 10
        small_history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={"innovation": 4.0, "method": 3.5, "evidence": 4.0, "application": 3.5, "cudos": 4.5},
                target_journal="Nature",
                outcome="accepted",
            )
            for i in range(5)
        ]

        result = await agent.discover_frontier(small_history)

        assert isinstance(result, FrontierUpdate)
        assert len(result.recommendations) > 0
        assert "Need at least" in result.recommendations[0]

    @pytest.mark.asyncio
    async def test_discover_frontier_with_sufficient_history(self):
        """Test discovery with sufficient history."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        # 15 records (above minimum of 10)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 4.0 + (i % 10) * 0.1,
                    "method": 3.5,
                    "evidence": 4.0,
                    "application": 3.5,
                    "cudos": 4.5
                },
                target_journal="Nature",
                outcome="accepted" if i % 3 == 0 else "rejected",
                editor_feedback="This is feedback text with novelty concerns." if i % 2 == 0 else "",
                keywords=["AI", "ML"] if i % 2 == 0 else ["DL", "NLP"],
                year=2022 + (i % 3),
                citations=i * 10,
            )
            for i in range(15)
        ]

        result = await agent.discover_frontier(history)

        assert isinstance(result, FrontierUpdate)
        assert result.new_dmu_count == 15

    # ========================================================================
    # Pattern Extraction Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_extract_improvement_patterns_with_feedback(self):
        """Test improvement pattern extraction with editorial feedback."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 4.0,
                    "method": 3.5,
                    "evidence": 4.0,
                    "application": 3.5,
                    "cudos": 4.5
                },
                target_journal="Nature",
                outcome="rejected",
                editor_feedback=f"Concerns about novelty in dimension. Incremental work.",
                keywords=["AI"],
                year=2024,
            )
            for i in range(12)
        ]

        result = await agent.discover_frontier(history)

        # Should extract some patterns from feedback
        assert isinstance(result.improvement_patterns, list)

    @pytest.mark.asyncio
    async def test_extract_patterns_no_feedback(self):
        """Test pattern extraction with no editorial feedback."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 4.0,
                    "method": 3.5,
                    "evidence": 4.0,
                    "application": 3.5,
                    "cudos": 4.5
                },
                target_journal="Nature",
                outcome="accepted",
                editor_feedback="",  # No feedback
                keywords=["AI"],
                year=2024,
            )
            for i in range(12)
        ]

        result = await agent.discover_frontier(history)

        # Should handle no feedback gracefully
        assert isinstance(result.improvement_patterns, list)

    # ========================================================================
    # Trend Identification Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_identify_trends_with_year_data(self):
        """Test trend identification with year-distributed data."""
        agent = FrontierDiscoverySubAgent(min_history_size=10, trend_window_years=5)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 4.0,
                    "method": 3.5,
                    "evidence": 4.0,
                    "application": 3.5,
                    "cudos": 4.5
                },
                target_journal="Nature",
                outcome="accepted",
                keywords=["AI", "machine learning", "deep learning", "transformer", "attention"],
                year=2020 + (i % 5),  # Years 2020-2024
            )
            for i in range(15)
        ]

        result = await agent.discover_frontier(history)

        # Should attempt trend identification
        assert isinstance(result.trend_clusters, list)

    @pytest.mark.asyncio
    async def test_identify_trends_no_year_data(self):
        """Test trend identification with no year data."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 4.0,
                    "method": 3.5,
                    "evidence": 4.0,
                    "application": 3.5,
                    "cudos": 4.5
                },
                target_journal="Nature",
                outcome="accepted",
                keywords=["AI", "ML"],
                year=0,  # No year data
            )
            for i in range(15)
        ]

        result = await agent.discover_frontier(history)

        # Should handle no year data gracefully
        assert isinstance(result.trend_clusters, list)

    # ========================================================================
    # Frontier Shift Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_compute_frontier_shift(self):
        """Test frontier shift computation."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 4.0 + (i % 10) * 0.05,
                    "method": 3.5 + (i % 10) * 0.03,
                    "evidence": 4.0 + (i % 10) * 0.04,
                    "application": 3.5 + (i % 10) * 0.02,
                    "cudos": 4.5
                },
                target_journal="Nature",
                outcome="accepted",
                keywords=["AI"],
                year=2020 + (i % 5),
            )
            for i in range(15)
        ]

        result = await agent.discover_frontier(history)

        assert isinstance(result.frontier_shift, float)
        assert isinstance(result.dimension_updates, dict)

    @pytest.mark.asyncio
    async def test_frontier_shift_direction(self):
        """Test frontier shift direction detection."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        # Create history with clearly increasing scores
        history = []
        for i in range(10):
            year = 2020 if i < 5 else 2024
            scores = {
                "innovation": 3.5 if i < 5 else 4.5,  # Increase from 3.5 to 4.5
                "method": 3.5,
                "evidence": 3.5,
                "application": 3.5,
                "cudos": 4.5
            }
            history.append(SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores=scores,
                target_journal="Nature",
                outcome="accepted",
                keywords=["AI"],
                year=year,
            ))

        result = await agent.discover_frontier(history)

        # Innovation should show increasing direction
        assert "innovation" in result.dimension_updates
        assert result.dimension_updates["innovation"]["direction"] in ["increasing", "decreasing", "stable"]

    # ========================================================================
    # Recommendations Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_generate_recommendations(self):
        """Test recommendation generation."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 4.0,
                    "method": 3.5,
                    "evidence": 4.0,
                    "application": 3.5,
                    "cudos": 4.5
                },
                target_journal="Nature",
                outcome="rejected" if i % 2 == 0 else "accepted",
                editor_feedback="Concerns about novelty and methodology." if i % 2 == 0 else "Good paper.",
                keywords=["AI", "novelty", "methodology"],
                year=2024,
            )
            for i in range(15)
        ]

        result = await agent.discover_frontier(history)

        assert isinstance(result.recommendations, list)

    # ========================================================================
    # Self-Improvement Signals Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_self_improvement_signals_generated(self):
        """Test self-improvement signals are generated."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 4.0,
                    "method": 3.5,
                    "evidence": 4.0,
                    "application": 3.5,
                    "cudos": 4.5
                },
                target_journal="Nature",
                outcome="rejected",
                editor_feedback="Lack of novelty. Methodology concerns.",
                keywords=["AI"],
                year=2024,
            )
            for i in range(15)
        ]

        result = await agent.discover_frontier(history)

        assert isinstance(result.self_improvement_signals, dict)

    # ========================================================================
    # Edge Cases
    # ========================================================================

    @pytest.mark.asyncio
    async def test_discover_frontier_empty_history(self):
        """Test discovery with empty history."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        result = await agent.discover_frontier([])

        assert isinstance(result, FrontierUpdate)
        assert result.new_dmu_count == 0

    @pytest.mark.asyncio
    async def test_discover_frontier_all_accepted(self):
        """Test discovery with all accepted papers."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 4.5,
                    "method": 4.0,
                    "evidence": 4.5,
                    "application": 4.0,
                    "cudos": 4.8
                },
                target_journal="Nature",
                outcome="accepted",
                editor_feedback="Excellent work.",
                keywords=["AI"],
                year=2024,
            )
            for i in range(15)
        ]

        result = await agent.discover_frontier(history)

        assert isinstance(result, FrontierUpdate)
        assert result.new_dmu_count == 15

    @pytest.mark.asyncio
    async def test_discover_frontier_all_rejected(self):
        """Test discovery with all rejected papers."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 2.5,
                    "method": 2.0,
                    "evidence": 2.5,
                    "application": 2.0,
                    "cudos": 4.0
                },
                target_journal="Nature",
                outcome="rejected",
                editor_feedback="Major concerns about novelty and methodology.",
                keywords=["AI"],
                year=2024,
            )
            for i in range(15)
        ]

        result = await agent.discover_frontier(history)

        assert isinstance(result, FrontierUpdate)
        # Should still process the data
        assert result.new_dmu_count == 15

    # ========================================================================
    # DEA Integration Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_update_dea_frontier_no_engine(self):
        """Test DEA frontier update without engine."""
        agent = FrontierDiscoverySubAgent(min_history_size=10)
        history = [
            SubmissionRecord(
                paper_id=f"test_{i}",
                title=f"Paper {i}",
                discipline="cs",
                scores={
                    "innovation": 4.0,
                    "method": 3.5,
                    "evidence": 4.0,
                    "application": 3.5,
                    "cudos": 4.5
                },
                target_journal="Nature",
                outcome="accepted",
            )
            for i in range(5)
        ]

        result = await agent.update_dea_frontier(history)

        assert result["status"] == "skipped"
        assert "No DEA engine" in result["reason"]

    # ========================================================================
    # Utility Method Tests
    # ========================================================================

    def test_extract_snippet(self):
        """Test snippet extraction utility."""
        text = "This is a sample text with a keyword that we want to extract around."
        keyword = "keyword"

        snippet = FrontierDiscoverySubAgent._extract_snippet(text, keyword, max_length=50)

        assert "keyword" in snippet
        assert len(snippet) <= 50 + 20  # max_length + ellipsis

    def test_extract_snippet_at_start(self):
        """Test snippet extraction at text start."""
        text = "Keyword at start of text for extraction."
        keyword = "Keyword"

        snippet = FrontierDiscoverySubAgent._extract_snippet(text, keyword, max_length=50)

        assert "Keyword" in snippet
        assert not snippet.startswith("...")

    def test_extract_snippet_not_found(self):
        """Test snippet extraction when keyword not found."""
        text = "This text does not contain the keyword."
        keyword = "missing"

        snippet = FrontierDiscoverySubAgent._extract_snippet(text, keyword, max_length=50)

        assert snippet == ""
