"""
Two-Phase Router Module for WuYa System

This module implements the TwoPhaseRouter that orchestrates the complete
evaluation workflow:

Phase 1: CUDOS Gatekeeping - Ethical/normative screening with veto power
Phase 2: Parallel Evaluation - Four evaluation sub-agents run concurrently
Phase 3: DEA Analysis - Efficiency analysis with super-efficiency scores
Phase 4: Result Aggregation - Final report generation

Key Design Points:
- Two-stage routing: CUDOS veto exits early without Phase 2
- Parallel execution: Phase 2 sub-agents use asyncio for concurrency
- Error handling: Each sub-agent failure is logged but doesn't block others
- RAG triggering: Low scores in Method/Evidence trigger retrieval

Author: WuYa Team
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import uuid

from .base import (
    AgentStatus,
    BaseSubAgent,
    CUDOSResult,
    CUDOSSubAgent,
    DimensionResult,
    EvaluationDimension,
    LLMClient,
    ParsedPaper,
    RAGClient,
    ScoreVector,
)
from .dea_subagent import DEASubAgent, ScoreVector as DEAScoreVector

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Report Data Classes
# =============================================================================

@dataclass
class DimensionSummary:
    """Summary of a single dimension evaluation."""
    dimension: str
    score: float
    status: str
    narrative: str
    processing_time_ms: int


@dataclass
class DEASummary:
    """Summary of DEA analysis results."""
    efficiency_score: float
    confidence_interval: Tuple[float, float]
    is_on_frontier: bool
    status: str
    explanation: str


@dataclass
class JournalMatch:
    """Journal matching recommendation."""
    journal_name: str
    match_score: float
    tier_estimate: str
    reasoning: str


@dataclass
class ImprovementSuggestion:
    """Suggestion for paper improvement."""
    dimension: str
    priority: str  # high, medium, low
    suggestion: str
    actionable: bool


@dataclass
class EvaluationReport:
    """
    Complete evaluation report for a paper.

    This is the final output of the TwoPhaseRouter, containing all
    evaluation results, recommendations, and improvement suggestions.
    """
    paper_id: str
    paper_title: str
    status: str  # completed, vetoed, error

    # Phase 1: CUDOS
    cudos_passed: bool
    cudos_details: Dict[str, Any]
    veto_reason: Optional[str]

    # Phase 2: Dimension Scores
    dimension_scores: Dict[str, float]
    score_vector: ScoreVector
    dimension_details: Dict[str, DimensionSummary]

    # Phase 3: DEA Analysis
    dea_summary: Optional[DEASummary]

    # Phase 4: Recommendations
    overall_score: float
    tier_estimate: str
    confidence: str
    journal_matches: List[JournalMatch]
    improvement_suggestions: List[ImprovementSuggestion]

    # Metadata
    processing_time_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    rag_citations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "status": self.status,
            "cudos_passed": self.cudos_passed,
            "cudos_details": self.cudos_details,
            "veto_reason": self.veto_reason,
            "dimension_scores": self.dimension_scores,
            "score_vector": self.score_vector.to_dict() if self.score_vector else None,
            "dimension_details": {
                k: {
                    "dimension": v.dimension,
                    "score": v.score,
                    "status": v.status,
                    "narrative": v.narrative,
                    "processing_time_ms": v.processing_time_ms,
                }
                for k, v in self.dimension_details.items()
            },
            "dea_summary": {
                "efficiency_score": self.dea_summary.efficiency_score,
                "confidence_interval": self.dea_summary.confidence_interval,
                "is_on_frontier": self.dea_summary.is_on_frontier,
                "status": self.dea_summary.status,
                "explanation": self.dea_summary.explanation,
            } if self.dea_summary else None,
            "overall_score": self.overall_score,
            "tier_estimate": self.tier_estimate,
            "confidence": self.confidence,
            "journal_matches": [
                {
                    "journal_name": j.journal_name,
                    "match_score": j.match_score,
                    "tier_estimate": j.tier_estimate,
                    "reasoning": j.reasoning,
                }
                for j in self.journal_matches
            ],
            "improvement_suggestions": [
                {
                    "dimension": s.dimension,
                    "priority": s.priority,
                    "suggestion": s.suggestion,
                    "actionable": s.actionable,
                }
                for s in self.improvement_suggestions
            ],
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "rag_citations": self.rag_citations,
            "errors": self.errors,
        }


# =============================================================================
# Two-Phase Router Implementation
# =============================================================================

class TwoPhaseRouter:
    """
    Two-Phase Router for orchestrating the complete evaluation workflow.

    The router implements a four-phase evaluation process:

    **Phase 1 - CUDOS Gatekeeping:**
    - Calls CUDOSSubAgent for ethical/normative review
    - If gate_pass=False, returns veto result immediately
    - If ethical issues detected, triggers RAG retrieval

    **Phase 2 - Parallel Evaluation:**
    - Concurrently calls 4 evaluation sub-agents:
      - InnovationSubAgent
      - MethodSubAgent
      - EvidenceSubAgent
      - ApplicationSubAgent
    - Collects dimension scores and narratives

    **Phase 3 - DEA Analysis:**
    - Calls DEASubAgent for efficiency analysis
    - Calculates super-efficiency scores
    - Generates Bootstrap confidence intervals

    **Phase 4 - Result Aggregation:**
    - Integrates all dimension results
    - Generates final evaluation report
    - Provides journal matching and improvement suggestions

    Example::

        # Initialize router
        router = TwoPhaseRouter(
            cudos_agent=cudos_agent,
            innovation_agent=innovation_agent,
            method_agent=method_agent,
            evidence_agent=evidence_agent,
            application_agent=application_agent,
            dea_agent=dea_agent,
        )

        # Route paper for evaluation
        report = await router.route(paper, target_journal="Nature ML")

        if report.status == "vetoed":
            print(f"Paper vetoed: {report.veto_reason}")
        else:
            print(f"Overall score: {report.overall_score}")
            print(f"Tier estimate: {report.tier_estimate}")

    Example (with error handling)::

        try:
            report = await router.route(paper)
        except RouterError as e:
            logger.error(f"Routing failed: {e}")
    """

    def __init__(
        self,
        cudos_agent: CUDOSSubAgent,
        innovation_agent: BaseSubAgent,
        method_agent: BaseSubAgent,
        evidence_agent: BaseSubAgent,
        application_agent: BaseSubAgent,
        dea_agent: Optional[DEASubAgent] = None,
        llm_client: Optional[LLMClient] = None,
        rag_client: Optional[RAGClient] = None,
        enable_parallel: bool = True,
        dea_min_reference_papers: int = 50,
    ):
        """
        Initialize TwoPhaseRouter.

        Args:
            cudos_agent: CUDOS Sub-agent for gatekeeping.
            innovation_agent: Innovation Sub-agent.
            method_agent: Method Sub-agent.
            evidence_agent: Evidence Sub-agent.
            application_agent: Application Sub-agent.
            dea_agent: Optional DEA Sub-agent for efficiency analysis.
            llm_client: Optional LLM client for result interpretation.
            rag_client: Optional RAG client for retrieval.
            enable_parallel: Whether to run Phase 2 sub-agents in parallel.
            dea_min_reference_papers: Minimum papers needed for DEA analysis.
        """
        self.cudos_agent = cudos_agent
        self.innovation_agent = innovation_agent
        self.method_agent = method_agent
        self.evidence_agent = evidence_agent
        self.application_agent = application_agent
        self.dea_agent = dea_agent
        self.llm_client = llm_client
        self.rag_client = rag_client
        self.enable_parallel = enable_parallel
        self.dea_min_reference_papers = dea_min_reference_papers

        # Map dimensions to agents
        self._evaluation_agents = {
            EvaluationDimension.INNOVATION: innovation_agent,
            EvaluationDimension.METHOD: method_agent,
            EvaluationDimension.EVIDENCE: evidence_agent,
            EvaluationDimension.APPLICATION: application_agent,
        }

        logger.info("TwoPhaseRouter initialized")

    async def route(
        self,
        paper: ParsedPaper,
        target_journal: Optional[str] = None,
        reference_papers: Optional[List[Dict[str, Any]]] = None,
        user_intent: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """
        Execute the complete two-phase routing workflow.

        Args:
            paper: Parsed paper to evaluate.
            target_journal: Optional target journal for DEA analysis.
            reference_papers: Optional list of reference papers for DEA.
            user_intent: Optional user intent dict with preferences.

        Returns:
            Complete EvaluationReport.
        """
        start_time = datetime.now()
        errors = []
        rag_citations = []

        logger.info(f"Starting evaluation for paper: {paper.paper_id}")

        # ======================================================================
        # Phase 1: CUDOS Gatekeeping
        # ======================================================================
        logger.info("Phase 1: CUDOS Gatekeeping")
        cudos_result = await self._phase1_cudos_gatekeeping(paper)

        if not cudos_result.gate_pass:
            logger.warning(f"Paper vetoed by CUDOS: {cudos_result.veto_reason}")
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

            return EvaluationReport(
                paper_id=paper.paper_id,
                paper_title=paper.title,
                status="vetoed",
                cudos_passed=False,
                cudos_details=cudos_result.to_dict(),
                veto_reason=cudos_result.veto_reason,
                dimension_scores={},
                score_vector=ScoreVector(0, 0, 0, 0, 0),
                dimension_details={},
                dea_summary=None,
                overall_score=0.0,
                tier_estimate="N/A",
                confidence="low",
                journal_matches=[],
                improvement_suggestions=[],
                processing_time_ms=processing_time,
                rag_citations=cudos_result.rag_citations,
            )

        logger.info("CUDOS gate passed")
        rag_citations.extend(cudos_result.rag_citations)

        # ======================================================================
        # Phase 2: Parallel Evaluation
        # ======================================================================
        logger.info("Phase 2: Parallel Evaluation")
        dimension_results, phase2_errors = await self._phase2_parallel_evaluation(paper)
        errors.extend(phase2_errors)

        # Collect RAG citations from dimension results
        for result in dimension_results.values():
            rag_citations.extend(result.rag_citations)

        # Build score vector
        score_vector = self._build_score_vector(dimension_results, cudos_result)

        # Create dimension summaries
        dimension_details = self._create_dimension_summaries(dimension_results)
        dimension_scores = {
            dim.value: result.overall_score
            for dim, result in dimension_results.items()
        }

        # ======================================================================
        # Phase 3: DEA Analysis
        # ======================================================================
        logger.info("Phase 3: DEA Analysis")
        dea_summary = None
        if self.dea_agent and target_journal and reference_papers:
            dea_summary = await self._phase3_dea_analysis(
                paper, score_vector, target_journal, reference_papers
            )
        else:
            logger.info("Skipping DEA analysis (no agent, journal, or references)")

        # ======================================================================
        # Phase 4: Result Aggregation
        # ======================================================================
        logger.info("Phase 4: Result Aggregation")
        report = await self._phase4_aggregate_results(
            paper=paper,
            cudos_result=cudos_result,
            dimension_results=dimension_results,
            score_vector=score_vector,
            dea_summary=dea_summary,
            target_journal=target_journal,
            start_time=start_time,
            errors=errors,
            rag_citations=rag_citations,
        )

        logger.info(f"Evaluation completed for paper: {paper.paper_id}")
        return report

    async def _phase1_cudos_gatekeeping(self, paper: ParsedPaper) -> CUDOSResult:
        """
        Phase 1: CUDOS gatekeeping.

        Evaluates paper against Merton's CUDOS norms.
        If gate_pass=False, evaluation stops here.
        """
        try:
            cudos_result = await self.cudos_agent.evaluate(paper)
            return cudos_result
        except Exception as e:
            logger.error(f"CUDOS evaluation failed: {e}")
            # Fail open - allow through on error
            return CUDOSResult(
                gate_pass=True,
                dimensions={},
                veto_reason=f"CUDOS evaluation error: {str(e)}",
            )

    async def _phase2_parallel_evaluation(
        self,
        paper: ParsedPaper,
    ) -> Tuple[Dict[EvaluationDimension, DimensionResult], List[str]]:
        """
        Phase 2: Parallel evaluation by four sub-agents.

        Runs Innovation, Method, Evidence, and Application sub-agents
        concurrently using asyncio.
        """
        errors = []
        dimension_results = {}

        if self.enable_parallel:
            # Run all evaluations in parallel
            tasks = []
            for dimension, agent in self._evaluation_agents.items():
                task = self._evaluate_with_error_handling(agent, paper, dimension)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, (dimension, _) in enumerate(self._evaluation_agents.items()):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"{dimension.value} evaluation failed: {result}")
                    errors.append(f"{dimension.value}: {str(result)}")
                    # Create failed result
                    dimension_results[dimension] = DimensionResult(
                        dimension=dimension,
                        overall_score=0.0,
                        sub_dimensions=[],
                        narrative=f"Evaluation failed: {str(result)}",
                        status=AgentStatus.FAILED,
                    )
                else:
                    dimension_results[dimension] = result
        else:
            # Sequential evaluation
            for dimension, agent in self._evaluation_agents.items():
                try:
                    result = await agent.evaluate(paper)
                    dimension_results[dimension] = result
                except Exception as e:
                    logger.error(f"{dimension.value} evaluation failed: {e}")
                    errors.append(f"{dimension.value}: {str(e)}")
                    dimension_results[dimension] = DimensionResult(
                        dimension=dimension,
                        overall_score=0.0,
                        sub_dimensions=[],
                        narrative=f"Evaluation failed: {str(e)}",
                        status=AgentStatus.FAILED,
                    )

        return dimension_results, errors

    async def _evaluate_with_error_handling(
        self,
        agent: BaseSubAgent,
        paper: ParsedPaper,
        dimension: EvaluationDimension,
    ) -> DimensionResult:
        """Evaluate with error handling for parallel execution."""
        try:
            return await agent.evaluate(paper)
        except Exception as e:
            logger.error(f"{dimension.value} evaluation error: {e}")
            raise

    def _build_score_vector(
        self,
        dimension_results: Dict[EvaluationDimension, DimensionResult],
        cudos_result: CUDOSResult,
    ) -> ScoreVector:
        """Build ScoreVector from dimension results."""
        # Get scores from dimension results
        innovation = dimension_results.get(
            EvaluationDimension.INNOVATION, DimensionResult(
                dimension=EvaluationDimension.INNOVATION,
                overall_score=0.0,
                sub_dimensions=[],
                narrative="",
            )
        ).overall_score

        method = dimension_results.get(
            EvaluationDimension.METHOD, DimensionResult(
                dimension=EvaluationDimension.METHOD,
                overall_score=0.0,
                sub_dimensions=[],
                narrative="",
            )
        ).overall_score

        evidence = dimension_results.get(
            EvaluationDimension.EVIDENCE, DimensionResult(
                dimension=EvaluationDimension.EVIDENCE,
                overall_score=0.0,
                sub_dimensions=[],
                narrative="",
            )
        ).overall_score

        application = dimension_results.get(
            EvaluationDimension.APPLICATION, DimensionResult(
                dimension=EvaluationDimension.APPLICATION,
                overall_score=0.0,
                sub_dimensions=[],
                narrative="",
            )
        ).overall_score

        # Calculate CUDOS score from dimensions
        cudos_scores = cudos_result.dimensions
        if cudos_scores:
            cudos = sum(d.get("score", 4.0) for d in cudos_scores.values()) / len(cudos_scores)
        else:
            cudos = 4.0  # Default if no CUDOS evaluation

        return ScoreVector(
            innovation=innovation,
            method=method,
            evidence=evidence,
            application=application,
            cudos=cudos,
            innovation_details={
                s.name: s.score
                for s in dimension_results.get(EvaluationDimension.INNOVATION, DimensionResult(
                    dimension=EvaluationDimension.INNOVATION,
                    overall_score=0.0,
                    sub_dimensions=[],
                    narrative="",
                )).sub_dimensions
            },
            method_details={
                s.name: s.score
                for s in dimension_results.get(EvaluationDimension.METHOD, DimensionResult(
                    dimension=EvaluationDimension.METHOD,
                    overall_score=0.0,
                    sub_dimensions=[],
                    narrative="",
                )).sub_dimensions
            },
            evidence_details={
                s.name: s.score
                for s in dimension_results.get(EvaluationDimension.EVIDENCE, DimensionResult(
                    dimension=EvaluationDimension.EVIDENCE,
                    overall_score=0.0,
                    sub_dimensions=[],
                    narrative="",
                )).sub_dimensions
            },
            application_details={
                s.name: s.score
                for s in dimension_results.get(EvaluationDimension.APPLICATION, DimensionResult(
                    dimension=EvaluationDimension.APPLICATION,
                    overall_score=0.0,
                    sub_dimensions=[],
                    narrative="",
                )).sub_dimensions
            },
        )

    def _create_dimension_summaries(
        self,
        dimension_results: Dict[EvaluationDimension, DimensionResult],
    ) -> Dict[str, DimensionSummary]:
        """Create dimension summaries from results."""
        summaries = {}
        for dim, result in dimension_results.items():
            summaries[dim.value] = DimensionSummary(
                dimension=dim.value,
                score=result.overall_score,
                status=result.status.value,
                narrative=result.narrative,
                processing_time_ms=result.processing_time_ms,
            )
        return summaries

    async def _phase3_dea_analysis(
        self,
        paper: ParsedPaper,
        score_vector: ScoreVector,
        target_journal: str,
        reference_papers: List[Dict[str, Any]],
    ) -> Optional[DEASummary]:
        """
        Phase 3: DEA analysis.

        Calculates super-efficiency scores and confidence intervals.
        """
        if not self.dea_agent:
            return None

        # Check if we have enough reference papers
        if len(reference_papers) < self.dea_min_reference_papers:
            logger.warning(
                f"Insufficient reference papers for DEA: {len(reference_papers)} < "
                f"{self.dea_min_reference_papers}"
            )
            return None

        try:
            # Convert ScoreVector to DEA format
            dea_score_vector = DEAScoreVector(
                innovation=score_vector.innovation,
                method=score_vector.method,
                evidence=score_vector.evidence,
                application=score_vector.application,
                cudos=score_vector.cudos,
            )

            # Call DEA sub-agent
            dea_result = await self.dea_agent.evaluate(
                paper_id=paper.paper_id,
                score_vector=dea_score_vector,
                target_journal=target_journal,
                reference_papers=reference_papers,
            )

            # Extract DEA result
            dea_data = dea_result.get("dea_result", {})

            return DEASummary(
                efficiency_score=dea_data.get("efficiency_score", 0.0),
                confidence_interval=dea_data.get("confidence_interval", (0.0, 0.0)),
                is_on_frontier=dea_data.get("is_on_frontier", False),
                status=dea_data.get("status", "unknown"),
                explanation=dea_data.get("explanation", ""),
            )

        except Exception as e:
            logger.error(f"DEA analysis failed: {e}")
            return None

    async def _phase4_aggregate_results(
        self,
        paper: ParsedPaper,
        cudos_result: CUDOSResult,
        dimension_results: Dict[EvaluationDimension, DimensionResult],
        score_vector: ScoreVector,
        dea_summary: Optional[DEASummary],
        target_journal: Optional[str],
        start_time: datetime,
        errors: List[str],
        rag_citations: List[str],
    ) -> EvaluationReport:
        """
        Phase 4: Result aggregation.

        Generates final evaluation report with recommendations.
        """
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Calculate overall score (weighted average of dimensions)
        dimension_scores = [
            score_vector.innovation,
            score_vector.method,
            score_vector.evidence,
            score_vector.application,
        ]
        overall_score = sum(dimension_scores) / len(dimension_scores)

        # Determine tier estimate
        tier_estimate = self._estimate_tier(score_vector, dea_summary)

        # Determine confidence
        confidence = self._determine_confidence(dimension_results, dea_summary)

        # Generate journal matches
        journal_matches = self._generate_journal_matches(
            score_vector, tier_estimate, target_journal
        )

        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            dimension_results, score_vector
        )

        return EvaluationReport(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            status="completed",
            cudos_passed=True,
            cudos_details=cudos_result.to_dict(),
            veto_reason=None,
            dimension_scores={
                "innovation": score_vector.innovation,
                "method": score_vector.method,
                "evidence": score_vector.evidence,
                "application": score_vector.application,
                "cudos": score_vector.cudos,
            },
            score_vector=score_vector,
            dimension_details=self._create_dimension_summaries(dimension_results),
            dea_summary=dea_summary,
            overall_score=round(overall_score, 2),
            tier_estimate=tier_estimate,
            confidence=confidence,
            journal_matches=journal_matches,
            improvement_suggestions=improvement_suggestions,
            processing_time_ms=processing_time,
            rag_citations=rag_citations,
            errors=errors,
        )

    def _estimate_tier(
        self,
        score_vector: ScoreVector,
        dea_summary: Optional[DEASummary],
    ) -> str:
        """Estimate journal tier based on scores."""
        # Base tier on average score
        avg_score = (
            score_vector.innovation +
            score_vector.method +
            score_vector.evidence +
            score_vector.application
        ) / 4

        if avg_score >= 4.5:
            base_tier = "Q1-A"
        elif avg_score >= 4.0:
            base_tier = "Q1-B"
        elif avg_score >= 3.5:
            base_tier = "Q2-A"
        elif avg_score >= 3.0:
            base_tier = "Q2-B"
        elif avg_score >= 2.5:
            base_tier = "Q3"
        else:
            base_tier = "Q4"

        # Adjust based on DEA if available
        if dea_summary and dea_summary.status == "success":
            if dea_summary.efficiency_score > 1.0:
                # On or above frontier - can aim higher
                if base_tier == "Q1-B":
                    return "Q1-A"
            elif dea_summary.efficiency_score < 0.8:
                # Below frontier - be conservative
                if base_tier == "Q1-A":
                    return "Q1-B"

        return base_tier

    def _determine_confidence(
        self,
        dimension_results: Dict[EvaluationDimension, DimensionResult],
        dea_summary: Optional[DEASummary],
    ) -> str:
        """Determine confidence level in the evaluation."""
        # Check for failures
        failed_dims = [
            dim for dim, result in dimension_results.items()
            if result.status != AgentStatus.SUCCESS
        ]

        if len(failed_dims) > 1:
            return "low"

        if dea_summary:
            ci_width = dea_summary.confidence_interval[1] - dea_summary.confidence_interval[0]
            if ci_width > 0.3:
                return "medium"

        return "high"

    def _generate_journal_matches(
        self,
        score_vector: ScoreVector,
        tier_estimate: str,
        target_journal: Optional[str],
    ) -> List[JournalMatch]:
        """Generate journal matching recommendations."""
        matches = []

        # If target journal specified, evaluate fit
        if target_journal:
            fit_score = self._calculate_journal_fit(score_vector, tier_estimate)
            matches.append(JournalMatch(
                journal_name=target_journal,
                match_score=fit_score,
                tier_estimate=tier_estimate,
                reasoning=f"Based on {tier_estimate} tier estimate and score profile",
            ))

        # Suggest alternative journals based on tier
        tier_journals = {
            "Q1-A": ["Nature", "Science", "Cell"],
            "Q1-B": ["Nature Communications", "PNAS", "PLOS Biology"],
            "Q2-A": ["IEEE Transactions", "ACM Computing Surveys"],
            "Q2-B": ["Journal of Machine Learning Research", "NeurIPS Proceedings"],
            "Q3": ["Specialized domain journals"],
            "Q4": ["Regional or emerging journals"],
        }

        for tier, journals in tier_journals.items():
            if tier != tier_estimate:
                continue
            for journal in journals[:2]:  # Limit suggestions
                if journal != target_journal:
                    matches.append(JournalMatch(
                        journal_name=journal,
                        match_score=0.8,
                        tier_estimate=tier,
                        reasoning=f"Alternative {tier} journal",
                    ))

        return matches

    def _calculate_journal_fit(
        self,
        score_vector: ScoreVector,
        tier_estimate: str,
    ) -> float:
        """Calculate fit score for target journal."""
        # Simple heuristic based on scores
        avg_score = (
            score_vector.innovation +
            score_vector.method +
            score_vector.evidence +
            score_vector.application
        ) / 4

        # Map tier to expected score range
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
        dimension_results: Dict[EvaluationDimension, DimensionResult],
        score_vector: ScoreVector,
    ) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions based on scores."""
        suggestions = []

        # Check each dimension for low scores
        dimension_scores = {
            "innovation": (score_vector.innovation, EvaluationDimension.INNOVATION),
            "method": (score_vector.method, EvaluationDimension.METHOD),
            "evidence": (score_vector.evidence, EvaluationDimension.EVIDENCE),
            "application": (score_vector.application, EvaluationDimension.APPLICATION),
        }

        for dim_name, (score, dim_enum) in dimension_scores.items():
            if score < 3.0:
                suggestions.append(ImprovementSuggestion(
                    dimension=dim_name,
                    priority="high",
                    suggestion=f"Significant improvement needed in {dim_name}",
                    actionable=True,
                ))
            elif score < 4.0:
                suggestions.append(ImprovementSuggestion(
                    dimension=dim_name,
                    priority="medium",
                    suggestion=f"Moderate improvements could strengthen {dim_name}",
                    actionable=True,
                ))

        # Add specific suggestions from sub-dimensions
        for dim, result in dimension_results.items():
            for sub in result.sub_dimensions:
                if sub.score < 3.0:
                    suggestions.append(ImprovementSuggestion(
                        dimension=f"{dim.value}.{sub.name}",
                        priority="high",
                        suggestion=sub.justification or f"Improve {sub.name}",
                        actionable=True,
                    ))

        return suggestions[:10]  # Limit to top 10 suggestions


# =============================================================================
# Factory Function
# =============================================================================

def create_two_phase_router(
    cudos_agent: CUDOSSubAgent,
    innovation_agent: BaseSubAgent,
    method_agent: BaseSubAgent,
    evidence_agent: BaseSubAgent,
    application_agent: BaseSubAgent,
    dea_agent: Optional[DEASubAgent] = None,
    llm_client: Optional[LLMClient] = None,
    rag_client: Optional[RAGClient] = None,
    enable_parallel: bool = True,
) -> TwoPhaseRouter:
    """
    Factory function to create a TwoPhaseRouter.

    Args:
        cudos_agent: CUDOS Sub-agent for gatekeeping.
        innovation_agent: Innovation Sub-agent.
        method_agent: Method Sub-agent.
        evidence_agent: Evidence Sub-agent.
        application_agent: Application Sub-agent.
        dea_agent: Optional DEA Sub-agent.
        llm_client: Optional LLM client.
        rag_client: Optional RAG client.
        enable_parallel: Whether to enable parallel Phase 2 execution.

    Returns:
        Configured TwoPhaseRouter instance.
    """
    return TwoPhaseRouter(
        cudos_agent=cudos_agent,
        innovation_agent=innovation_agent,
        method_agent=method_agent,
        evidence_agent=evidence_agent,
        application_agent=application_agent,
        dea_agent=dea_agent,
        llm_client=llm_client,
        rag_client=rag_client,
        enable_parallel=enable_parallel,
    )
