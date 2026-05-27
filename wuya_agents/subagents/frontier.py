"""
Frontier Discovery Sub-agent

Implements DEA frontier discovery mechanism for the WuYa multi-agent system.

Responsibilities:
    1. Extract improvement patterns from editorial feedback
    2. Update DEA efficiency frontier with new DMUs (papers)
    3. Identify emerging research trends via keyword clustering
    4. Output frontier boundary update suggestions
    5. Provide self-improvement loop interface (results feed back to evaluation)

Architecture:
    Router -> FrontierDiscoverySubAgent -> DEA Engine (frontier update)
                                       -> RAG Client (historical data retrieval)
                                       -> Trend Analyzer (keyword clustering)

Author: WuYa Team
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SubmissionRecord:
    """
    Historical submission record.

    Attributes:
        paper_id: Unique paper identifier.
        title: Paper title.
        discipline: Academic discipline.
        scores: Five-dimensional score vector.
        target_journal: Target journal name.
        outcome: Submission outcome (accepted, rejected, revision, etc.).
        editor_feedback: Editor/reviewer feedback text.
        keywords: Paper keywords.
        year: Submission year.
        citations: Citation count (if available).
    """
    paper_id: str
    title: str
    discipline: str
    scores: Dict[str, float]  # {innovation, method, evidence, application, cudos}
    target_journal: str
    outcome: str  # "accepted", "rejected", "revision", "desk_reject"
    editor_feedback: str = ""
    keywords: List[str] = field(default_factory=list)
    year: int = 0
    citations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "discipline": self.discipline,
            "scores": self.scores,
            "target_journal": self.target_journal,
            "outcome": self.outcome,
            "editor_feedback": self.editor_feedback,
            "keywords": self.keywords,
            "year": self.year,
            "citations": self.citations,
        }


@dataclass
class ImprovementPattern:
    """
    An improvement pattern extracted from editorial feedback.

    Attributes:
        dimension: Which evaluation dimension this pattern relates to.
        pattern_description: Human-readable description of the pattern.
        frequency: How often this pattern appears in feedback.
        severity: Average severity (1-5, 5 = most critical).
        examples: Example feedback excerpts.
    """
    dimension: str
    pattern_description: str
    frequency: int
    severity: float
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "pattern_description": self.pattern_description,
            "frequency": self.frequency,
            "severity": round(self.severity, 2),
            "examples": self.examples[:3],  # Limit examples
        }


@dataclass
class TrendCluster:
    """
    A cluster of related research keywords representing an emerging trend.

    Attributes:
        cluster_id: Unique cluster identifier.
        keywords: Keywords in this cluster.
        frequency: Total frequency across all papers.
        growth_rate: Growth rate (comparing recent vs older papers).
        representative_papers: Paper IDs that best represent this trend.
    """
    cluster_id: str
    keywords: List[str]
    frequency: int
    growth_rate: float
    representative_papers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "keywords": self.keywords,
            "frequency": self.frequency,
            "growth_rate": round(self.growth_rate, 4),
            "representative_papers": self.representative_papers[:5],
        }


@dataclass
class FrontierUpdate:
    """
    Result of frontier discovery analysis.

    Attributes:
        timestamp: When this update was generated.
        new_dmu_count: Number of new DMUs (papers) added to frontier.
        improvement_patterns: Improvement patterns extracted from feedback.
        trend_clusters: Emerging research trend clusters.
        frontier_shift: How much the frontier has shifted (positive = higher bar).
        dimension_updates: Per-dimension frontier boundary updates.
        recommendations: Actionable recommendations based on analysis.
        self_improvement_signals: Signals for the evaluation system to self-improve.
    """
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    new_dmu_count: int = 0
    improvement_patterns: List[ImprovementPattern] = field(default_factory=list)
    trend_clusters: List[TrendCluster] = field(default_factory=list)
    frontier_shift: float = 0.0
    dimension_updates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    self_improvement_signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "new_dmu_count": self.new_dmu_count,
            "improvement_patterns": [p.to_dict() for p in self.improvement_patterns],
            "trend_clusters": [t.to_dict() for t in self.trend_clusters],
            "frontier_shift": round(self.frontier_shift, 4),
            "dimension_updates": self.dimension_updates,
            "recommendations": self.recommendations,
            "self_improvement_signals": self.self_improvement_signals,
        }


# =============================================================================
# Frontier Discovery Sub-agent
# =============================================================================

class FrontierDiscoverySubAgent:
    """
    Frontier Discovery Sub-agent for DEA frontier analysis.

    Analyzes historical submission data to:
    1. Extract improvement patterns from editorial feedback
    2. Update DEA efficiency frontier with new DMUs
    3. Identify emerging research trends via keyword clustering
    4. Generate frontier boundary update suggestions
    5. Provide self-improvement loop signals

    Example::

        agent = FrontierDiscoverySubAgent()
        update = await agent.discover_frontier(submission_history)
        print(f"Frontier shift: {update.frontier_shift:.4f}")
        for pattern in update.improvement_patterns:
            print(f"  Pattern: {pattern.pattern_description}")

    Example (with DEA engine)::

        from wuya_agents.dea_subagent import DEAEngine
        agent = FrontierDiscoverySubAgent(dea_engine=DEAEngine())
        update = await agent.discover_frontier(history)
    """

    # Dimension-specific feedback keywords
    DIMENSION_FEEDBACK_KEYWORDS = {
        "innovation": [
            "novelty", "originality", "incremental", "already known",
            "not new", "lacks innovation", "derivative", "contribution",
            "creative", "breakthrough", "groundbreaking",
        ],
        "method": [
            "methodology", "rigor", "statistical", "sample size",
            "reproducibility", "bias", "confounding", "control",
            "validation", "robustness", "experimental design",
        ],
        "evidence": [
            "evidence", "data", "results", "findings", "support",
            "insufficient", "contradict", "correlation", "causation",
            "significance", "effect size", "confidence",
        ],
        "application": [
            "practical", "application", "impact", "relevance",
            "implementation", "real-world", "deployment", "scalability",
            "use case", "industry", "clinical",
        ],
        "cudos": [
            "ethics", "conflict of interest", "data availability",
            "transparency", "replication", "peer review", "plagiarism",
            "attribution", "credit", "open science",
        ],
    }

    def __init__(
        self,
        dea_engine=None,
        rag_client=None,
        min_history_size: int = 10,
        trend_window_years: int = 3,
    ):
        """
        Initialize Frontier Discovery Sub-agent.

        Args:
            dea_engine: Optional DEA engine for frontier computation.
            rag_client: Optional RAG client for historical data retrieval.
            min_history_size: Minimum records needed for analysis.
            trend_window_years: Years to consider for trend analysis.
        """
        self.dea_engine = dea_engine
        self.rag_client = rag_client
        self.min_history_size = min_history_size
        self.trend_window_years = trend_window_years
        self._history: List[SubmissionRecord] = []

    async def discover_frontier(
        self,
        history: List[SubmissionRecord],
    ) -> FrontierUpdate:
        """
        Main entry point: analyze historical data and discover frontier updates.

        Args:
            history: List of historical submission records.

        Returns:
            FrontierUpdate with analysis results.
        """
        logger.info(f"Starting frontier discovery with {len(history)} records")

        if len(history) < self.min_history_size:
            logger.warning(
                f"Insufficient history: {len(history)} < {self.min_history_size}"
            )
            return FrontierUpdate(
                recommendations=[
                    f"Need at least {self.min_history_size} submission records "
                    f"for meaningful frontier analysis (have {len(history)})."
                ],
            )

        self._history = history

        # Step 1: Extract improvement patterns from editorial feedback
        improvement_patterns = self._extract_improvement_patterns(history)

        # Step 2: Identify emerging research trends
        trend_clusters = self._identify_trends(history)

        # Step 3: Compute frontier shift
        frontier_shift, dimension_updates = self._compute_frontier_shift(history)

        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(
            improvement_patterns, trend_clusters, frontier_shift
        )

        # Step 5: Generate self-improvement signals
        self_improvement_signals = self._generate_self_improvement_signals(
            improvement_patterns, dimension_updates
        )

        update = FrontierUpdate(
            new_dmu_count=len(history),
            improvement_patterns=improvement_patterns,
            trend_clusters=trend_clusters,
            frontier_shift=frontier_shift,
            dimension_updates=dimension_updates,
            recommendations=recommendations,
            self_improvement_signals=self_improvement_signals,
        )

        logger.info(
            f"Frontier discovery complete: shift={frontier_shift:.4f}, "
            f"patterns={len(improvement_patterns)}, "
            f"trends={len(trend_clusters)}"
        )

        return update

    # =========================================================================
    # Step 1: Improvement Pattern Extraction
    # =========================================================================

    def _extract_improvement_patterns(
        self,
        history: List[SubmissionRecord],
    ) -> List[ImprovementPattern]:
        """
        Extract improvement patterns from editorial feedback.

        Analyzes feedback text to identify recurring issues by dimension.
        """
        patterns = []

        # Filter records with feedback
        records_with_feedback = [
            r for r in history
            if r.editor_feedback and len(r.editor_feedback.strip()) > 10
        ]

        if not records_with_feedback:
            logger.info("No editorial feedback available for pattern extraction")
            return patterns

        for dimension, keywords in self.DIMENSION_FEEDBACK_KEYWORDS.items():
            dimension_patterns = self._extract_dimension_patterns(
                dimension, keywords, records_with_feedback
            )
            patterns.extend(dimension_patterns)

        # Sort by frequency (descending)
        patterns.sort(key=lambda p: p.frequency, reverse=True)

        return patterns[:20]  # Top 20 patterns

    def _extract_dimension_patterns(
        self,
        dimension: str,
        keywords: List[str],
        records: List[SubmissionRecord],
    ) -> List[ImprovementPattern]:
        """Extract patterns for a specific dimension."""
        # Count keyword occurrences in feedback
        keyword_counts: Counter = Counter()
        keyword_examples: Dict[str, List[str]] = {}

        for record in records:
            feedback_lower = record.editor_feedback.lower()
            for keyword in keywords:
                if keyword in feedback_lower:
                    keyword_counts[keyword] += 1
                    if keyword not in keyword_examples:
                        keyword_examples[keyword] = []
                    # Store a snippet around the keyword
                    snippet = self._extract_snippet(
                        record.editor_feedback, keyword, max_length=150
                    )
                    if snippet and len(keyword_examples[keyword]) < 3:
                        keyword_examples[keyword].append(snippet)

        # Group related keywords into patterns
        patterns = []
        for keyword, count in keyword_counts.most_common(5):
            if count >= 2:  # Only include patterns that appear at least twice
                # Estimate severity based on outcome correlation
                severity = self._estimate_pattern_severity(
                    dimension, keyword, records
                )

                patterns.append(ImprovementPattern(
                    dimension=dimension,
                    pattern_description=f"Frequent mention of '{keyword}' in {dimension} feedback",
                    frequency=count,
                    severity=severity,
                    examples=keyword_examples.get(keyword, []),
                ))

        return patterns

    @staticmethod
    def _extract_snippet(text: str, keyword: str, max_length: int = 150) -> str:
        """Extract a text snippet around a keyword occurrence."""
        lower_text = text.lower()
        idx = lower_text.find(keyword.lower())
        if idx == -1:
            return ""

        start = max(0, idx - max_length // 2)
        end = min(len(text), idx + len(keyword) + max_length // 2)
        snippet = text[start:end].strip()

        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    def _estimate_pattern_severity(
        self,
        dimension: str,
        keyword: str,
        records: List[SubmissionRecord],
    ) -> float:
        """
        Estimate severity of a pattern based on correlation with rejection.

        Higher severity = more strongly associated with negative outcomes.
        """
        keyword_lower = keyword.lower()
        mentioned_in_rejected = 0
        mentioned_in_accepted = 0

        for record in records:
            if keyword_lower in record.editor_feedback.lower():
                if record.outcome in ("rejected", "desk_reject"):
                    mentioned_in_rejected += 1
                elif record.outcome == "accepted":
                    mentioned_in_accepted += 1

        total_mentions = mentioned_in_rejected + mentioned_in_accepted
        if total_mentions == 0:
            return 3.0  # Neutral severity

        rejection_rate = mentioned_in_rejected / total_mentions
        # Map rejection rate to severity (0-5)
        severity = 1.0 + rejection_rate * 4.0
        return round(severity, 2)

    # =========================================================================
    # Step 2: Trend Identification
    # =========================================================================

    def _identify_trends(
        self,
        history: List[SubmissionRecord],
    ) -> List[TrendCluster]:
        """
        Identify emerging research trends via keyword clustering.

        Compares keyword frequency between recent and older papers
        to detect growing trends.
        """
        if not history:
            return []

        # Determine year split
        years = [r.year for r in history if r.year > 0]
        if not years:
            return []

        max_year = max(years)
        split_year = max_year - self.trend_window_years

        recent_papers = [r for r in history if r.year > split_year]
        older_papers = [r for r in history if 0 < r.year <= split_year]

        if not recent_papers or not older_papers:
            logger.info("Insufficient year-distributed data for trend analysis")
            return []

        # Count keywords in each period
        recent_keywords = self._count_keywords(recent_papers)
        older_keywords = self._count_keywords(older_papers)

        # Compute growth rates
        growth_rates = {}
        for keyword, recent_count in recent_keywords.items():
            older_count = older_keywords.get(keyword, 0)
            if older_count > 0:
                # Growth rate: (recent - older) / older
                rate = (recent_count - older_count) / older_count
            else:
                # New keyword: assign high growth rate
                rate = 1.0 if recent_count >= 2 else 0.0
            growth_rates[keyword] = rate

        # Cluster related keywords (simple co-occurrence clustering)
        clusters = self._cluster_keywords(
            growth_rates, recent_papers, min_growth=0.3
        )

        return clusters

    @staticmethod
    def _count_keywords(records: List[SubmissionRecord]) -> Counter:
        """Count keyword frequencies across records."""
        counter = Counter()
        for record in records:
            for kw in record.keywords:
                counter[kw.lower()] += 1
        return counter

    def _cluster_keywords(
        self,
        growth_rates: Dict[str, float],
        records: List[SubmissionRecord],
        min_growth: float = 0.3,
    ) -> List[TrendCluster]:
        """
        Cluster keywords by co-occurrence in papers.

        Simple approach: group keywords that frequently appear together.
        """
        # Filter for growing keywords
        growing_keywords = {
            kw: rate for kw, rate in growth_rates.items()
            if rate >= min_growth
        }

        if not growing_keywords:
            return []

        # Build co-occurrence matrix
        co_occurrence: Dict[str, Counter] = {}
        for record in records:
            paper_kws = set(kw.lower() for kw in record.keywords)
            growing_in_paper = paper_kws & set(growing_keywords.keys())

            for kw1 in growing_in_paper:
                if kw1 not in co_occurrence:
                    co_occurrence[kw1] = Counter()
                for kw2 in growing_in_paper:
                    if kw1 != kw2:
                        co_occurrence[kw1][kw2] += 1

        # Simple greedy clustering
        clusters = []
        assigned = set()

        # Sort by total growth score (frequency * growth_rate)
        sorted_keywords = sorted(
            growing_keywords.keys(),
            key=lambda kw: growth_rates[kw] * sum(
                1 for r in records if kw in [k.lower() for k in r.keywords]
            ),
            reverse=True,
        )

        cluster_id = 0
        for kw in sorted_keywords:
            if kw in assigned:
                continue

            # Start a new cluster
            cluster_keywords = [kw]
            assigned.add(kw)

            # Find related keywords
            related = co_occurrence.get(kw, {})
            for related_kw, count in related.most_common(5):
                if related_kw not in assigned and count >= 2:
                    cluster_keywords.append(related_kw)
                    assigned.add(related_kw)

            if len(cluster_keywords) >= 1:
                # Calculate cluster statistics
                total_freq = sum(
                    sum(1 for r in records if ck in [k.lower() for k in r.keywords])
                    for ck in cluster_keywords
                )
                avg_growth = sum(growing_keywords[k] for k in cluster_keywords) / len(cluster_keywords)

                # Find representative papers
                representative_papers = self._find_representative_papers(
                    cluster_keywords, records
                )

                clusters.append(TrendCluster(
                    cluster_id=f"trend_{cluster_id:03d}",
                    keywords=cluster_keywords,
                    frequency=total_freq,
                    growth_rate=avg_growth,
                    representative_papers=representative_papers,
                ))
                cluster_id += 1

        # Sort by growth rate
        clusters.sort(key=lambda c: c.growth_rate, reverse=True)

        return clusters[:10]  # Top 10 trends

    @staticmethod
    def _find_representative_papers(
        keywords: List[str],
        records: List[SubmissionRecord],
    ) -> List[str]:
        """Find papers that best represent a keyword cluster."""
        scores = []
        for record in records:
            paper_kws = set(kw.lower() for kw in record.keywords)
            overlap = len(paper_kws & set(keywords))
            if overlap > 0:
                scores.append((record.paper_id, overlap))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:5]]

    # =========================================================================
    # Step 3: Frontier Shift Computation
    # =========================================================================

    def _compute_frontier_shift(
        self,
        history: List[SubmissionRecord],
    ) -> Tuple[float, Dict[str, Dict[str, Any]]]:
        """
        Compute how much the DEA efficiency frontier has shifted.

        Compares accepted papers' score profiles across time periods
        to detect shifts in quality standards.
        """
        accepted = [r for r in history if r.outcome == "accepted"]
        if len(accepted) < 5:
            return 0.0, {}

        # Split by year
        years = [r.year for r in accepted if r.year > 0]
        if not years:
            return 0.0, {}

        max_year = max(years)
        mid_year = (min(years) + max_year) // 2

        older_accepted = [r for r in accepted if r.year <= mid_year]
        recent_accepted = [r for r in accepted if r.year > mid_year]

        if not older_accepted or not recent_accepted:
            return 0.0, {}

        # Compute average scores for each period
        dimensions = ["innovation", "method", "evidence", "application", "cudos"]
        dimension_updates = {}

        total_shift = 0.0
        for dim in dimensions:
            older_avg = self._avg_dimension(older_accepted, dim)
            recent_avg = self._avg_dimension(recent_accepted, dim)

            shift = recent_avg - older_avg
            total_shift += shift

            dimension_updates[dim] = {
                "older_average": round(older_avg, 3),
                "recent_average": round(recent_avg, 3),
                "shift": round(shift, 3),
                "direction": "increasing" if shift > 0.05 else (
                    "decreasing" if shift < -0.05 else "stable"
                ),
                "older_count": len(older_accepted),
                "recent_count": len(recent_accepted),
            }

        # Average shift across dimensions
        avg_shift = total_shift / len(dimensions)

        return round(avg_shift, 4), dimension_updates

    @staticmethod
    def _avg_dimension(records: List[SubmissionRecord], dimension: str) -> float:
        """Compute average score for a dimension across records."""
        scores = [r.scores.get(dimension, 0.0) for r in records]
        return sum(scores) / len(scores) if scores else 0.0

    # =========================================================================
    # Step 4: Recommendations
    # =========================================================================

    def _generate_recommendations(
        self,
        patterns: List[ImprovementPattern],
        trends: List[TrendCluster],
        frontier_shift: float,
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Recommendations from improvement patterns
        high_severity_patterns = [p for p in patterns if p.severity >= 3.5]
        if high_severity_patterns:
            top_pattern = high_severity_patterns[0]
            recommendations.append(
                f"Critical improvement area: {top_pattern.dimension} - "
                f"'{top_pattern.pattern_description}' "
                f"(severity={top_pattern.severity:.1f}, frequency={top_pattern.frequency})"
            )

        # Recommendations from trends
        if trends:
            top_trend = trends[0]
            recommendations.append(
                f"Emerging trend detected: {', '.join(top_trend.keywords[:3])} "
                f"(growth_rate={top_trend.growth_rate:.1%})"
            )

        # Recommendations from frontier shift
        if frontier_shift > 0.1:
            recommendations.append(
                f"Quality bar is rising: frontier shifted +{frontier_shift:.3f}. "
                f"Consider updating evaluation criteria to reflect higher standards."
            )
        elif frontier_shift < -0.1:
            recommendations.append(
                f"Quality bar appears to be declining: frontier shifted {frontier_shift:.3f}. "
                f"Review evaluation criteria for potential recalibration."
            )

        # General recommendations
        if len(patterns) > 10:
            recommendations.append(
                f"High number of improvement patterns ({len(patterns)}). "
                f"Consider focused editorial guidelines to address recurring issues."
            )

        return recommendations

    # =========================================================================
    # Step 5: Self-Improvement Signals
    # =========================================================================

    def _generate_self_improvement_signals(
        self,
        patterns: List[ImprovementPattern],
        dimension_updates: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate signals for the evaluation system to self-improve.

        These signals can be fed back into the evaluation framework
        to adjust scoring criteria, RAG retrieval priorities, etc.
        """
        signals = {
            "dimension_weight_adjustments": {},
            "rag_priority_updates": [],
            "evaluation_criteria_updates": [],
        }

        # Adjust dimension weights based on pattern frequency
        dimension_pattern_counts: Counter = Counter()
        for pattern in patterns:
            dimension_pattern_counts[pattern.dimension] += pattern.frequency

        total_patterns = sum(dimension_pattern_counts.values())
        if total_patterns > 0:
            for dim, count in dimension_pattern_counts.items():
                # Dimensions with more improvement patterns may need
                # higher evaluation weight to catch issues earlier
                current_weight = 1.0
                suggested_weight = 1.0 + (count / total_patterns) * 0.5
                signals["dimension_weight_adjustments"][dim] = {
                    "current_weight": current_weight,
                    "suggested_weight": round(suggested_weight, 2),
                    "reason": f"{count} improvement patterns in {dim}",
                }

        # Update RAG retrieval priorities based on frontier shifts
        for dim, update in dimension_updates.items():
            if update.get("direction") == "increasing":
                signals["rag_priority_updates"].append(
                    f"Prioritize {dim} literature retrieval - "
                    f"standards are rising (shift={update['shift']:+.3f})"
                )

        # Suggest evaluation criteria updates
        for pattern in patterns[:5]:
            if pattern.severity >= 4.0:
                signals["evaluation_criteria_updates"].append(
                    f"Add evaluation checkpoint for {pattern.dimension}: "
                    f"{pattern.pattern_description}"
                )

        return signals

    # =========================================================================
    # DEA Integration
    # =========================================================================

    async def update_dea_frontier(
        self,
        history: List[SubmissionRecord],
    ) -> Dict[str, Any]:
        """
        Update DEA engine's reference set with new data.

        Converts submission records to DEA ScoreVectors and
        triggers frontier recomputation.

        Args:
            history: Submission records to add as DMUs.

        Returns:
            Status dict with update results.
        """
        if not self.dea_engine:
            return {
                "status": "skipped",
                "reason": "No DEA engine configured",
            }

        from ..dea_subagent import ScoreVector as DEAScoreVector

        # Convert to DEA format
        new_dmus = []
        for record in history:
            scores = record.scores
            dmu = DEAScoreVector(
                innovation=scores.get("innovation", 3.0),
                method=scores.get("method", 3.0),
                evidence=scores.get("evidence", 3.0),
                application=scores.get("application", 3.0),
                cudos=scores.get("cudos", 4.0),
            )
            new_dmus.append(dmu)

        logger.info(f"Prepared {len(new_dmus)} DMUs for DEA frontier update")

        return {
            "status": "prepared",
            "new_dmu_count": len(new_dmus),
            "note": "DMUs prepared for next DEA analysis cycle",
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_frontier_discovery_agent(
    dea_engine=None,
    rag_client=None,
    min_history_size: int = 10,
    trend_window_years: int = 3,
) -> FrontierDiscoverySubAgent:
    """
    Factory function to create a Frontier Discovery Sub-agent.

    Args:
        dea_engine: Optional DEA engine for frontier computation.
        rag_client: Optional RAG client for historical data retrieval.
        min_history_size: Minimum records needed for analysis.
        trend_window_years: Years to consider for trend analysis.

    Returns:
        Configured FrontierDiscoverySubAgent instance.
    """
    return FrontierDiscoverySubAgent(
        dea_engine=dea_engine,
        rag_client=rag_client,
        min_history_size=min_history_size,
        trend_window_years=trend_window_years,
    )
