"""
Result Aggregator Module for WuYa System

This module implements the ResultAggregator that combines results from
all evaluation sub-agents and DEA analysis into a comprehensive evaluation report.

Features:
- Aggregates dimension scores from all sub-agents
- Integrates DEA efficiency analysis
- Generates improvement suggestions
- Provides journal matching recommendations
- Produces structured evaluation reports

Author: WuYa Team
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    AgentStatus,
    CUDOSResult,
    DimensionResult,
    EvaluationDimension,
    ParsedPaper,
    ScoreVector,
)
from .dea_subagent import DEAResult

logger = logging.getLogger(__name__)


# =============================================================================
# Aggregation Result Data Classes
# =============================================================================

@dataclass
class AggregatedDimension:
    """Aggregated result for a single dimension."""
    name: str
    score: float
    weight: float
    status: str
    narrative: str
    sub_scores: Dict[str, float]
    processing_time_ms: int


@dataclass
class AggregatedDEA:
    """Aggregated DEA analysis results."""
    efficiency_score: float
    confidence_interval: Tuple[float, float]
    is_on_frontier: bool
    status: str
    explanation: str
    reference_set_size: int


@dataclass
class JournalRecommendation:
    """Journal recommendation with reasoning."""
    journal_name: str
    match_score: float
    tier_estimate: str
    confidence: str
    reasoning: str
    alternative_journals: List[str]


@dataclass
class ImprovementItem:
    """Specific improvement suggestion."""
    dimension: str
    sub_dimension: Optional[str]
    priority: str  # critical, high, medium, low
    current_score: float
    target_score: float
    suggestion: str
    actionable_steps: List[str]


@dataclass
class EvaluationSummary:
    """High-level evaluation summary."""
    overall_score: float
    tier_estimate: str
    confidence_level: str
    key_strengths: List[str]
    key_weaknesses: List[str]
    recommendation: str


@dataclass
class EvaluationReport:
    """
    Complete evaluation report aggregating all sub-agent results.

    This is the final output of the ResultAggregator, containing:
    - Aggregated dimension scores
    - DEA efficiency analysis
    - Journal recommendations
    - Improvement suggestions
    - Evaluation summary
    """
    # Paper Info
    paper_id: str
    paper_title: str
    discipline: str

    # CUDOS Gatekeeping
    cudos_passed: bool
    cudos_details: Dict[str, Any]
    veto_reason: Optional[str]

    # Dimension Results
    dimensions: Dict[str, AggregatedDimension]
    score_vector: ScoreVector

    # DEA Analysis
    dea_analysis: Optional[AggregatedDEA]

    # Recommendations
    summary: EvaluationSummary
    journal_recommendations: List[JournalRecommendation]
    improvement_suggestions: List[ImprovementItem]

    # Metadata
    processing_time_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "discipline": self.discipline,
            "cudos_passed": self.cudos_passed,
            "cudos_details": self.cudos_details,
            "veto_reason": self.veto_reason,
            "dimensions": {
                name: {
                    "name": dim.name,
                    "score": dim.score,
                    "weight": dim.weight,
                    "status": dim.status,
                    "narrative": dim.narrative,
                    "sub_scores": dim.sub_scores,
                    "processing_time_ms": dim.processing_time_ms,
                }
                for name, dim in self.dimensions.items()
            },
            "score_vector": self.score_vector.to_dict() if self.score_vector else None,
            "dea_analysis": {
                "efficiency_score": self.dea_analysis.efficiency_score,
                "confidence_interval": self.dea_analysis.confidence_interval,
                "is_on_frontier": self.dea_analysis.is_on_frontier,
                "status": self.dea_analysis.status,
                "explanation": self.dea_analysis.explanation,
                "reference_set_size": self.dea_analysis.reference_set_size,
            } if self.dea_analysis else None,
            "summary": {
                "overall_score": self.summary.overall_score,
                "tier_estimate": self.summary.tier_estimate,
                "confidence_level": self.summary.confidence_level,
                "key_strengths": self.summary.key_strengths,
                "key_weaknesses": self.summary.key_weaknesses,
                "recommendation": self.summary.recommendation,
            },
            "journal_recommendations": [
                {
                    "journal_name": jr.journal_name,
                    "match_score": jr.match_score,
                    "tier_estimate": jr.tier_estimate,
                    "confidence": jr.confidence,
                    "reasoning": jr.reasoning,
                    "alternative_journals": jr.alternative_journals,
                }
                for jr in self.journal_recommendations
            ],
            "improvement_suggestions": [
                {
                    "dimension": item.dimension,
                    "sub_dimension": item.sub_dimension,
                    "priority": item.priority,
                    "current_score": item.current_score,
                    "target_score": item.target_score,
                    "suggestion": item.suggestion,
                    "actionable_steps": item.actionable_steps,
                }
                for item in self.improvement_suggestions
            ],
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    def to_markdown(self) -> str:
        """Generate markdown formatted report."""
        lines = []

        # Header
        lines.append(f"# Evaluation Report: {self.paper_title}")
        lines.append(f"\n**Paper ID:** {self.paper_id}")
        lines.append(f"**Discipline:** {self.discipline}")
        lines.append(f"**Evaluation Date:** {self.timestamp}")

        # CUDOS
        lines.append("\n## CUDOS Gatekeeping")
        if self.cudos_passed:
            lines.append("✅ **PASSED** - Paper meets CUDOS normative standards")
        else:
            lines.append(f"❌ **VETOED** - {self.veto_reason}")
            return "\n".join(lines)

        # Summary
        lines.append("\n## Executive Summary")
        lines.append(f"**Overall Score:** {self.summary.overall_score:.2f}/5.0")
        lines.append(f"**Tier Estimate:** {self.summary.tier_estimate}")
        lines.append(f"**Confidence:** {self.summary.confidence_level}")
        lines.append(f"\n**Recommendation:** {self.summary.recommendation}")

        if self.summary.key_strengths:
            lines.append("\n### Key Strengths")
            for strength in self.summary.key_strengths:
                lines.append(f"- {strength}")

        if self.summary.key_weaknesses:
            lines.append("\n### Key Weaknesses")
            for weakness in self.summary.key_weaknesses:
                lines.append(f"- {weakness}")

        # Dimensions
        lines.append("\n## Dimension Scores")
        for name, dim in self.dimensions.items():
            lines.append(f"\n### {name.title()}: {dim.score:.2f}/5.0")
            lines.append(f"Status: {dim.status}")
            lines.append(f"\n{dim.narrative}")

        # DEA
        if self.dea_analysis:
            lines.append("\n## DEA Efficiency Analysis")
            lines.append(f"**Efficiency Score:** {self.dea_analysis.efficiency_score:.3f}")
            lines.append(f"**95% Confidence Interval:** ({self.dea_analysis.confidence_interval[0]:.3f}, {self.dea_analysis.confidence_interval[1]:.3f})")
            lines.append(f"**On Frontier:** {'Yes' if self.dea_analysis.is_on_frontier else 'No'}")
            lines.append(f"\n{self.dea_analysis.explanation}")

        # Journal Recommendations
        if self.journal_recommendations:
            lines.append("\n## Journal Recommendations")
            for jr in self.journal_recommendations:
                lines.append(f"\n### {jr.journal_name}")
                lines.append(f"- Match Score: {jr.match_score:.2f}")
                lines.append(f"- Tier Estimate: {jr.tier_estimate}")
                lines.append(f"- Confidence: {jr.confidence}")
                lines.append(f"- Reasoning: {jr.reasoning}")

        # Improvement Suggestions
        if self.improvement_suggestions:
            lines.append("\n## Improvement Suggestions")
            for item in self.improvement_suggestions:
                lines.append(f"\n### [{item.priority.upper()}] {item.dimension}")
                if item.sub_dimension:
                    lines.append(f"Sub-dimension: {item.sub_dimension}")
                lines.append(f"Current Score: {item.current_score:.2f} → Target: {item.target_score:.2f}")
                lines.append(f"\n{item.suggestion}")
                if item.actionable_steps:
                    lines.append("\nActionable Steps:")
                    for step in item.actionable_steps:
                        lines.append(f"- {step}")

        # Footer
        lines.append(f"\n---")
        lines.append(f"*Report generated by WuYa Evaluation System v{self.version}*")
        lines.append(f"*Processing time: {self.processing_time_ms}ms*")

        return "\n".join(lines)


# =============================================================================
# Result Aggregator Implementation
# =============================================================================

class ResultAggregator:
    """
    Aggregates results from all evaluation sub-agents into a comprehensive report.

    The ResultAggregator combines:
    - CUDOS gatekeeping results
    - Four dimension evaluation results (Innovation, Method, Evidence, Application)
    - DEA efficiency analysis

    And produces:
    - Structured EvaluationReport
    - Journal recommendations
    - Improvement suggestions
    - Executive summary

    Example::

        # Aggregate results
        aggregator = ResultAggregator()
        report = await aggregator.aggregate(
            paper=paper,
            cudos_result=cudos_result,
            dimension_results=dimension_results,
            dea_result=dea_result,
            target_journal="Nature ML",
        )

        # Access report data
        print(f"Overall score: {report.summary.overall_score}")
        print(f"Tier estimate: {report.summary.tier_estimate}")

        # Export as markdown
        markdown = report.to_markdown()
        with open("report.md", "w") as f:
            f.write(markdown)

    Example (with veto)::

        report = await aggregator.aggregate(
            paper=paper,
            cudos_result=cudos_result,  # gate_pass=False
        )

        if not report.cudos_passed:
            print(f"Paper vetoed: {report.veto_reason}")
    """

    # Dimension weights for overall score calculation
    DEFAULT_DIMENSION_WEIGHTS = {
        "innovation": 1.0,
        "method": 1.0,
        "evidence": 1.0,
        "application": 1.0,
    }

    # Tier thresholds
    TIER_THRESHOLDS = [
        (4.5, "Q1-A"),
        (4.0, "Q1-B"),
        (3.5, "Q2-A"),
        (3.0, "Q2-B"),
        (2.5, "Q3"),
        (0.0, "Q4"),
    ]

    # Journal suggestions by tier
    TIER_JOURNALS = {
        "Q1-A": ["Nature", "Science", "Cell", "Nature Reviews"],
        "Q1-B": ["Nature Communications", "PNAS", "PLOS Biology", "eLife"],
        "Q2-A": ["IEEE Transactions", "ACM Computing Surveys", "Oxford Academic"],
        "Q2-B": ["Journal of Machine Learning Research", "NeurIPS", "ICML"],
        "Q3": ["Specialized domain journals", "Regional journals"],
        "Q4": ["Emerging journals", "Conference proceedings"],
    }

    def __init__(
        self,
        dimension_weights: Optional[Dict[str, float]] = None,
        min_confidence_threshold: float = 0.7,
    ):
        """
        Initialize ResultAggregator.

        Args:
            dimension_weights: Weights for each dimension in overall score.
            min_confidence_threshold: Minimum confidence for high confidence rating.
        """
        self.dimension_weights = dimension_weights or self.DEFAULT_DIMENSION_WEIGHTS.copy()
        self.min_confidence_threshold = min_confidence_threshold

    async def aggregate(
        self,
        paper: ParsedPaper,
        cudos_result: CUDOSResult,
        dimension_results: Optional[Dict[EvaluationDimension, DimensionResult]] = None,
        dea_result: Optional[DEAResult] = None,
        target_journal: Optional[str] = None,
        processing_time_ms: int = 0,
    ) -> EvaluationReport:
        """
        Aggregate all evaluation results into a comprehensive report.

        Args:
            paper: The parsed paper being evaluated.
            cudos_result: CUDOS gatekeeping result.
            dimension_results: Results from four evaluation sub-agents.
            dea_result: DEA analysis result (optional).
            target_journal: Target journal for recommendations (optional).
            processing_time_ms: Total processing time in milliseconds.

        Returns:
            Complete EvaluationReport.
        """
        logger.info(f"Aggregating results for paper: {paper.paper_id}")

        # Handle CUDOS veto
        if not cudos_result.gate_pass:
            return self._create_veto_report(paper, cudos_result, processing_time_ms)

        # Build score vector from dimension results
        score_vector = self._build_score_vector(
            dimension_results or {}, cudos_result
        )

        # Aggregate dimensions
        aggregated_dims = self._aggregate_dimensions(dimension_results or {})

        # Aggregate DEA results
        aggregated_dea = self._aggregate_dea(dea_result)

        # Calculate overall score
        overall_score = self._calculate_overall_score(score_vector)

        # Estimate tier
        tier_estimate = self._estimate_tier(overall_score, aggregated_dea)

        # Determine confidence
        confidence = self._determine_confidence(
            dimension_results or {}, aggregated_dea
        )

        # Generate summary
        summary = self._generate_summary(
            overall_score, tier_estimate, confidence,
            aggregated_dims, aggregated_dea
        )

        # Generate journal recommendations
        journal_recs = self._generate_journal_recommendations(
            score_vector, tier_estimate, target_journal
        )

        # Generate improvement suggestions
        improvements = self._generate_improvement_suggestions(
            aggregated_dims, score_vector
        )

        return EvaluationReport(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            discipline=paper.discipline,
            cudos_passed=True,
            cudos_details=cudos_result.to_dict(),
            veto_reason=None,
            dimensions=aggregated_dims,
            score_vector=score_vector,
            dea_analysis=aggregated_dea,
            summary=summary,
            journal_recommendations=journal_recs,
            improvement_suggestions=improvements,
            processing_time_ms=processing_time_ms,
        )

    def _create_veto_report(
        self,
        paper: ParsedPaper,
        cudos_result: CUDOSResult,
        processing_time_ms: int,
    ) -> EvaluationReport:
        """Create a report for vetoed papers."""
        return EvaluationReport(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            discipline=paper.discipline,
            cudos_passed=False,
            cudos_details=cudos_result.to_dict(),
            veto_reason=cudos_result.veto_reason,
            dimensions={},
            score_vector=ScoreVector(0, 0, 0, 0, 0),
            dea_analysis=None,
            summary=EvaluationSummary(
                overall_score=0.0,
                tier_estimate="N/A",
                confidence_level="low",
                key_strengths=[],
                key_weaknesses=["Failed CUDOS gatekeeping"],
                recommendation="Paper does not meet CUDOS normative standards",
            ),
            journal_recommendations=[],
            improvement_suggestions=[],
            processing_time_ms=processing_time_ms,
        )

    def _build_score_vector(
        self,
        dimension_results: Dict[EvaluationDimension, DimensionResult],
        cudos_result: CUDOSResult,
    ) -> ScoreVector:
        """Build ScoreVector from dimension results."""
        def get_score(dim: EvaluationDimension) -> float:
            result = dimension_results.get(dim)
            return result.overall_score if result else 0.0

        # Calculate CUDOS score
        cudos_scores = cudos_result.dimensions
        if cudos_scores:
            cudos = sum(d.get("score", 4.0) for d in cudos_scores.values()) / len(cudos_scores)
        else:
            cudos = 4.0

        def get_details(dim: EvaluationDimension) -> Dict[str, float]:
            result = dimension_results.get(dim)
            if result:
                return {s.name: s.score for s in result.sub_dimensions}
            return {}

        return ScoreVector(
            innovation=get_score(EvaluationDimension.INNOVATION),
            method=get_score(EvaluationDimension.METHOD),
            evidence=get_score(EvaluationDimension.EVIDENCE),
            application=get_score(EvaluationDimension.APPLICATION),
            cudos=cudos,
            innovation_details=get_details(EvaluationDimension.INNOVATION),
            method_details=get_details(EvaluationDimension.METHOD),
            evidence_details=get_details(EvaluationDimension.EVIDENCE),
            application_details=get_details(EvaluationDimension.APPLICATION),
        )

    def _aggregate_dimensions(
        self,
        dimension_results: Dict[EvaluationDimension, DimensionResult],
    ) -> Dict[str, AggregatedDimension]:
        """Aggregate dimension results into structured format."""
        aggregated = {}

        for dim, result in dimension_results.items():
            dim_name = dim.value
            weight = self.dimension_weights.get(dim_name, 1.0)

            sub_scores = {
                sub.name: sub.score
                for sub in result.sub_dimensions
            }

            aggregated[dim_name] = AggregatedDimension(
                name=dim_name,
                score=result.overall_score,
                weight=weight,
                status=result.status.value,
                narrative=result.narrative,
                sub_scores=sub_scores,
                processing_time_ms=result.processing_time_ms,
            )

        return aggregated

    def _aggregate_dea(self, dea_result: Optional[DEAResult]) -> Optional[AggregatedDEA]:
        """Aggregate DEA results."""
        if not dea_result:
            return None

        return AggregatedDEA(
            efficiency_score=dea_result.efficiency_score,
            confidence_interval=dea_result.confidence_interval,
            is_on_frontier=dea_result.is_on_frontier,
            status=dea_result.status.value,
            explanation=dea_result.explanation,
            reference_set_size=dea_result.reference_set_size,
        )

    def _calculate_overall_score(self, score_vector: ScoreVector) -> float:
        """Calculate weighted overall score."""
        scores = [
            (score_vector.innovation, self.dimension_weights.get("innovation", 1.0)),
            (score_vector.method, self.dimension_weights.get("method", 1.0)),
            (score_vector.evidence, self.dimension_weights.get("evidence", 1.0)),
            (score_vector.application, self.dimension_weights.get("application", 1.0)),
        ]

        total_weight = sum(w for _, w in scores)
        weighted_sum = sum(s * w for s, w in scores)

        return round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0

    def _estimate_tier(
        self,
        overall_score: float,
        dea_analysis: Optional[AggregatedDEA],
    ) -> str:
        """Estimate journal tier based on scores."""
        # Base tier on overall score
        for threshold, tier in self.TIER_THRESHOLDS:
            if overall_score >= threshold:
                base_tier = tier
                break
        else:
            base_tier = "Q4"

        # Adjust based on DEA
        if dea_analysis and dea_analysis.status == "success":
            if dea_analysis.efficiency_score > 1.0 and base_tier == "Q1-B":
                return "Q1-A"
            elif dea_analysis.efficiency_score < 0.8 and base_tier == "Q1-A":
                return "Q1-B"

        return base_tier

    def _determine_confidence(
        self,
        dimension_results: Dict[EvaluationDimension, DimensionResult],
        dea_analysis: Optional[AggregatedDEA],
    ) -> str:
        """Determine confidence level."""
        # Check for failures
        failed_count = sum(
            1 for r in dimension_results.values()
            if r.status != AgentStatus.SUCCESS
        )

        if failed_count > 1:
            return "low"

        if dea_analysis:
            ci_width = (
                dea_analysis.confidence_interval[1] -
                dea_analysis.confidence_interval[0]
            )
            if ci_width > 0.3:
                return "medium"

        return "high"

    def _generate_summary(
        self,
        overall_score: float,
        tier_estimate: str,
        confidence: str,
        dimensions: Dict[str, AggregatedDimension],
        dea_analysis: Optional[AggregatedDEA],
    ) -> EvaluationSummary:
        """Generate executive summary."""
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        for name, dim in dimensions.items():
            if dim.score >= 4.5:
                strengths.append(f"Exceptional {name} ({dim.score:.1f}/5.0)")
            elif dim.score >= 4.0:
                strengths.append(f"Strong {name} ({dim.score:.1f}/5.0)")
            elif dim.score < 3.0:
                weaknesses.append(f"Weak {name} ({dim.score:.1f}/5.0)")

        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_score, tier_estimate, dea_analysis
        )

        return EvaluationSummary(
            overall_score=overall_score,
            tier_estimate=tier_estimate,
            confidence_level=confidence,
            key_strengths=strengths,
            key_weaknesses=weaknesses,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        overall_score: float,
        tier_estimate: str,
        dea_analysis: Optional[AggregatedDEA],
    ) -> str:
        """Generate submission recommendation."""
        if overall_score >= 4.5:
            base_rec = f"Strong candidate for {tier_estimate} journals"
        elif overall_score >= 4.0:
            base_rec = f"Good fit for {tier_estimate} journals"
        elif overall_score >= 3.5:
            base_rec = f"Competitive for {tier_estimate} journals with minor revisions"
        elif overall_score >= 3.0:
            base_rec = f"Marginal fit for {tier_estimate} journals, significant improvements recommended"
        else:
            base_rec = "Consider lower-tier journals or major revisions"

        # Add DEA context
        if dea_analysis:
            if dea_analysis.is_on_frontier:
                base_rec += ". Paper is on the efficiency frontier for target journal."
            elif dea_analysis.efficiency_score < 0.8:
                base_rec += ". Paper is below the efficiency frontier."

        return base_rec

    def _generate_journal_recommendations(
        self,
        score_vector: ScoreVector,
        tier_estimate: str,
        target_journal: Optional[str],
    ) -> List[JournalRecommendation]:
        """Generate journal recommendations."""
        recommendations = []

        # Target journal recommendation
        if target_journal:
            fit_score = self._calculate_journal_fit(score_vector, tier_estimate)
            confidence = "high" if fit_score > 0.8 else "medium" if fit_score > 0.6 else "low"

            alternatives = [
                j for j in self.TIER_JOURNALS.get(tier_estimate, [])
                if j != target_journal
            ][:3]

            recommendations.append(JournalRecommendation(
                journal_name=target_journal,
                match_score=fit_score,
                tier_estimate=tier_estimate,
                confidence=confidence,
                reasoning=f"Based on {tier_estimate} tier estimate and score profile",
                alternative_journals=alternatives,
            ))

        # Suggest journals from tier
        tier_journals = self.TIER_JOURNALS.get(tier_estimate, [])
        for journal in tier_journals[:2]:
            if journal != target_journal:
                recommendations.append(JournalRecommendation(
                    journal_name=journal,
                    match_score=0.8,
                    tier_estimate=tier_estimate,
                    confidence="medium",
                    reasoning=f"Recommended {tier_estimate} journal",
                    alternative_journals=[],
                ))

        return recommendations

    def _calculate_journal_fit(
        self,
        score_vector: ScoreVector,
        tier_estimate: str,
    ) -> float:
        """Calculate fit score for target journal."""
        avg_score = (
            score_vector.innovation +
            score_vector.method +
            score_vector.evidence +
            score_vector.application
        ) / 4

        tier_expectations = {
            "Q1-A": 4.5,
            "Q1-B": 4.0,
            "Q2-A": 3.5,
            "Q2-B": 3.0,
            "Q3": 2.5,
            "Q4": 2.0,
        }

        expected = tier_expectations.get(tier_estimate, 3.0)
        fit = 1.0 - abs(avg_score - expected) / 5.0
        return max(0.0, min(1.0, fit))

    def _generate_improvement_suggestions(
        self,
        dimensions: Dict[str, AggregatedDimension],
        score_vector: ScoreVector,
    ) -> List[ImprovementItem]:
        """Generate improvement suggestions."""
        suggestions = []

        # Check main dimensions
        dim_scores = {
            "innovation": score_vector.innovation,
            "method": score_vector.method,
            "evidence": score_vector.evidence,
            "application": score_vector.application,
        }

        for dim_name, score in dim_scores.items():
            if score < 3.0:
                suggestions.append(ImprovementItem(
                    dimension=dim_name,
                    sub_dimension=None,
                    priority="critical",
                    current_score=score,
                    target_score=3.5,
                    suggestion=f"Major improvements needed in {dim_name}",
                    actionable_steps=[
                        f"Review {dim_name} best practices in field",
                        f"Seek expert feedback on {dim_name}",
                    ],
                ))
            elif score < 4.0:
                suggestions.append(ImprovementItem(
                    dimension=dim_name,
                    sub_dimension=None,
                    priority="high",
                    current_score=score,
                    target_score=4.0,
                    suggestion=f"Moderate improvements could strengthen {dim_name}",
                    actionable_steps=[
                        f"Enhance {dim_name} presentation",
                    ],
                ))

        # Check sub-dimensions
        for dim_name, dim in dimensions.items():
            for sub_name, sub_score in dim.sub_scores.items():
                if sub_score < 3.0:
                    suggestions.append(ImprovementItem(
                        dimension=dim_name,
                        sub_dimension=sub_name,
                        priority="high",
                        current_score=sub_score,
                        target_score=3.5,
                        suggestion=f"Address weakness in {sub_name}",
                        actionable_steps=[f"Improve {sub_name} aspects"],
                    ))

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda x: priority_order.get(x.priority, 4))

        return suggestions[:10]  # Limit to top 10


# =============================================================================
# Factory Function
# =============================================================================

def create_result_aggregator(
    dimension_weights: Optional[Dict[str, float]] = None,
    min_confidence_threshold: float = 0.7,
) -> ResultAggregator:
    """
    Factory function to create a ResultAggregator.

    Args:
        dimension_weights: Weights for each dimension.
        min_confidence_threshold: Minimum confidence for high rating.

    Returns:
        Configured ResultAggregator instance.
    """
    return ResultAggregator(
        dimension_weights=dimension_weights,
        min_confidence_threshold=min_confidence_threshold,
    )
