"""
Evidence Sub-agent Implementation

Evaluates academic papers on evidence quality dimensions:
- Strength: How strong is the evidence supporting the claims?
- Consistency: How consistent is the evidence internally and with prior work?
- Sufficiency: Is there sufficient evidence to support the conclusions?

This sub-agent evaluates the quality and sufficiency of evidence,
with RAG integration to retrieve supplementary empirical studies.

Author: WuYa Team
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import (
    AgentStatus,
    BaseSubAgent,
    DimensionResult,
    EvaluationDimension,
    LLMClient,
    ParsedPaper,
    RAGClient,
    SubDimensionScore,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Evidence Dimension Definitions
# =============================================================================

@dataclass
class EvidenceDimensionScore:
    """Score for a single evidence dimension."""
    name: str
    score: float  # 0-5 scale
    rationale: str = ""
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "rationale": self.rationale,
            "citations": self.citations,
        }


# Evidence dimension names
STRENGTH = "strength"
CONSISTENCY = "consistency"
SUFFICIENCY = "sufficiency"

EVIDENCE_DIMENSIONS = [STRENGTH, CONSISTENCY, SUFFICIENCY]

# Dimension descriptions for LLM prompts
DIMENSION_DESCRIPTIONS = {
    STRENGTH: (
        "Strength: How strong is the evidence supporting the claims? "
        "Evaluate: Is the evidence direct or indirect? "
        "How robust are the findings? "
        "Are there multiple lines of evidence?"
    ),
    CONSISTENCY: (
        "Consistency: How consistent is the evidence? "
        "Evaluate: Is evidence internally consistent? "
        "Does it align with prior research? "
        "Are there contradictions or anomalies?"
    ),
    SUFFICIENCY: (
        "Sufficiency: Is there sufficient evidence for the conclusions? "
        "Evaluate: Is the evidence adequate for the claims made? "
        "Are there gaps in the evidence? "
        "Would more evidence strengthen the conclusions?"
    ),
}


# =============================================================================
# Evidence Sub-agent Implementation
# =============================================================================

class EvidenceSubAgent(BaseSubAgent):
    """
    Evidence Sub-agent for evaluating strength, consistency, and sufficiency.

    Evaluates papers on their evidence quality. Has RAG integration
    to retrieve supplementary empirical studies when sufficiency is low.

    Key Design:
        - Three dimensions: strength, consistency, sufficiency
        - Each dimension outputs {score: 0-5, rationale: str, citations: List[str]}
        - RAG triggered when sufficiency < 3.0 to retrieve supplementary studies
        - Uses LLMClient for evaluation

    Example::

        # Create with real LLM client
        llm_client = await LLMClient.from_env()
        evidence = EvidenceSubAgent(llm_client=llm_client, rag_client=rag_client)

        # Evaluate a paper
        result = await evidence.evaluate(paper)
        print(f"Evidence score: {result.overall_score}")

    Example (RAG trigger)::

        # If sufficiency < 3.0, RAG is triggered to retrieve supplementary empirical studies
        # This helps identify what additional evidence is needed
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_client: Optional[RAGClient] = None,
        timeout_seconds: float = 60.0,
        rag_threshold: float = 3.0,  # Trigger RAG if sufficiency < 3.0
    ):
        """
        Initialize Evidence Sub-agent.

        Args:
            llm_client: LLM client for evaluation. Can be None for mock testing.
            rag_client: RAG client for retrieving supplementary studies.
            timeout_seconds: Timeout for evaluation.
            rag_threshold: Score below this triggers RAG retrieval.
        """
        super().__init__(
            dimension=EvaluationDimension.EVIDENCE,
            llm_client=llm_client,
            rag_client=rag_client,
            timeout_seconds=timeout_seconds,
            rag_threshold=rag_threshold,
        )
        # Store preliminary scores for RAG trigger decision
        self._preliminary_scores: Dict[str, float] = {}

    # =========================================================================
    # Core Evaluation Logic
    # =========================================================================

    async def _evaluate_impl(
        self,
        paper: ParsedPaper,
        context: Optional[Dict[str, Any]],
        rag_citations: List[str]
    ) -> DimensionResult:
        """
        Implementation-specific evaluation logic.

        Args:
            paper: Parsed paper content.
            context: Optional context from other agents.
            rag_citations: Citations from RAG retrieval.

        Returns:
            DimensionResult with evidence scores and narrative.
        """
        if self.llm_client is None:
            # Mock mode: return default scores
            logger.warning("No LLM client provided, using mock evaluation")
            return self._get_mock_result(paper, rag_citations)

        try:
            # Build evaluation prompt
            prompt = self._build_evaluation_prompt(paper, rag_citations)

            # Call LLM
            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.get_system_prompt(),
                temperature=0.1,
                max_tokens=2000,
            )

            # Parse response
            dimensions = self._parse_llm_response(response)

            # Validate and normalize scores
            dimensions = self._validate_dimensions(dimensions)

            # Store preliminary scores for RAG trigger decision
            self._preliminary_scores = {
                dim: data["score"] for dim, data in dimensions.items()
            }

            # Build sub-dimension scores
            sub_dimensions = [
                SubDimensionScore(
                    name=dim,
                    score=data["score"],
                    justification=data.get("rationale", ""),
                )
                for dim, data in dimensions.items()
            ]

            # Calculate overall score (weighted average)
            # Give more weight to sufficiency
            weights = {STRENGTH: 1.0, CONSISTENCY: 1.0, SUFFICIENCY: 1.5}
            total_weight = sum(weights.values())
            weighted_sum = sum(
                data["score"] * weights.get(dim, 1.0)
                for dim, data in dimensions.items()
            )
            overall_score = weighted_sum / total_weight

            # Generate narrative
            narrative = self._generate_narrative(dimensions, overall_score)

            # Determine status based on RAG trigger
            status = AgentStatus.SUCCESS
            if rag_citations and dimensions.get(SUFFICIENCY, {}).get("score", 5.0) < self.rag_threshold:
                status = AgentStatus.RAG_TRIGGERED

            return DimensionResult(
                dimension=EvaluationDimension.EVIDENCE,
                overall_score=round(overall_score, 1),
                sub_dimensions=sub_dimensions,
                narrative=narrative,
                rag_citations=rag_citations,
                status=status,
            )

        except Exception as e:
            logger.error(f"Evidence evaluation failed: {e}")
            return DimensionResult(
                dimension=EvaluationDimension.EVIDENCE,
                overall_score=0.0,
                sub_dimensions=[],
                narrative=f"Evidence evaluation failed: {str(e)}",
                status=AgentStatus.FAILED,
            )

    def _build_evaluation_prompt(
        self,
        paper: ParsedPaper,
        rag_citations: List[str]
    ) -> str:
        """Build the evaluation prompt for the LLM."""
        sections = []

        # Paper metadata
        sections.append("# Paper to Evaluate\n")
        sections.append(f"Title: {paper.title}")
        sections.append(f"Authors: {', '.join(paper.authors)}")
        sections.append(f"Discipline: {paper.discipline}")
        sections.append(f"\n## Abstract\n{paper.abstract}")

        # Include content (truncated if too long)
        content = paper.content[:3000] if len(paper.content) > 3000 else paper.content
        sections.append(f"\n## Content (excerpt)\n{content}")

        # RAG citations if available (supplementary studies)
        if rag_citations:
            sections.append("\n## Supplementary Studies (from RAG)")
            for citation in rag_citations[:5]:
                sections.append(f"- {citation}")

        # Evaluation instructions
        sections.append("\n# Evaluation Task")
        sections.append(
            "Evaluate this paper's evidence quality. "
            "For each dimension, provide a score (0-5) and rationale.\n"
        )
        sections.append("Output JSON format:")
        sections.append("""{
    "strength": {
        "score": <0-5>,
        "rationale": "explanation of strength assessment"
    },
    "consistency": {
        "score": <0-5>,
        "rationale": "explanation of consistency assessment"
    },
    "sufficiency": {
        "score": <0-5>,
        "rationale": "explanation of sufficiency assessment"
    }
}""")

        # Dimension descriptions
        sections.append("\n# Dimension Definitions")
        for dim, desc in DIMENSION_DESCRIPTIONS.items():
            sections.append(f"- {desc}")

        # Scoring guide
        sections.append("\n# Scoring Guide")
        sections.append("""
- 5: Compelling evidence, no gaps
- 4: Strong evidence, minor gaps
- 3: Adequate evidence, some concerns
- 2: Weak evidence, significant gaps
- 1: Insufficient evidence, major concerns
- 0: No evidence or severely flawed
""")

        return "\n".join(sections)

    def _parse_llm_response(self, response: str) -> Dict[str, Dict[str, Any]]:
        """Parse LLM response into structured dimension scores."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")

        # Fallback: try to extract scores from text
        dimensions = {}
        for dim in EVIDENCE_DIMENSIONS:
            score_match = re.search(
                rf'{dim}["\']?\s*[:=]\s*["\']?(\d+\.?\d*)',
                response,
                re.IGNORECASE
            )
            if score_match:
                dimensions[dim] = {
                    "score": float(score_match.group(1)),
                    "rationale": "Extracted from response",
                }

        return dimensions if dimensions else self._get_mock_dimensions()

    def _validate_dimensions(
        self,
        dimensions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Validate and normalize dimension scores."""
        validated = {}

        for dim in EVIDENCE_DIMENSIONS:
            if dim in dimensions:
                score = dimensions[dim].get("score", 2.5)
                # Clamp to valid range
                score = max(0.0, min(5.0, float(score)))
                validated[dim] = {
                    "score": round(score, 1),
                    "rationale": dimensions[dim].get("rationale", ""),
                }
            else:
                # Missing dimension: use neutral score
                validated[dim] = {
                    "score": 2.5,
                    "rationale": f"Dimension {dim} not evaluated, using default",
                }

        return validated

    def _generate_narrative(
        self,
        dimensions: Dict[str, Dict[str, Any]],
        overall_score: float
    ) -> str:
        """Generate a narrative summary of the evaluation."""
        parts = [f"## Evidence Evaluation Summary\n"]
        parts.append(f"Overall Evidence Score: {overall_score:.1f}/5.0\n")

        for dim in EVIDENCE_DIMENSIONS:
            data = dimensions.get(dim, {})
            score = data.get("score", 0.0)
            rationale = data.get("rationale", "No rationale provided")
            parts.append(f"**{dim.title()}** ({score:.1f}/5.0): {rationale}")

        # Add warning if sufficiency is low
        sufficiency_score = dimensions.get(SUFFICIENCY, {}).get("score", 5.0)
        if sufficiency_score < 3.0:
            parts.append(f"\n⚠️ **Warning**: Low sufficiency score ({sufficiency_score:.1f}). "
                        "Additional evidence may be needed to support conclusions.")

        return "\n".join(parts)

    def _get_mock_dimensions(self) -> Dict[str, Dict[str, Any]]:
        """Return mock dimensions for testing."""
        return {
            STRENGTH: {"score": 3.5, "rationale": "Mock evaluation"},
            CONSISTENCY: {"score": 3.5, "rationale": "Mock evaluation"},
            SUFFICIENCY: {"score": 3.5, "rationale": "Mock evaluation"},
        }

    def _get_mock_result(
        self,
        paper: ParsedPaper,
        rag_citations: List[str]
    ) -> DimensionResult:
        """Return mock result for testing."""
        dimensions = self._get_mock_dimensions()
        sub_dimensions = [
            SubDimensionScore(
                name=dim,
                score=data["score"],
                justification=data["rationale"],
            )
            for dim, data in dimensions.items()
        ]
        overall_score = sum(s.score for s in sub_dimensions) / len(sub_dimensions)

        return DimensionResult(
            dimension=EvaluationDimension.EVIDENCE,
            overall_score=round(overall_score, 1),
            sub_dimensions=sub_dimensions,
            narrative="Mock evidence evaluation",
            rag_citations=rag_citations,
            status=AgentStatus.SUCCESS,
        )

    # =========================================================================
    # RAG Trigger Logic
    # =========================================================================

    def should_trigger_rag(
        self,
        paper: ParsedPaper,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if RAG should be triggered.

        RAG is triggered when:
        1. Sufficiency score < rag_threshold (default 3.0)
        2. Or if preliminary scores not available, check for evidence gaps

        Args:
            paper: Parsed paper content.
            context: Optional context from previous evaluations.

        Returns:
            True if RAG should be triggered.
        """
        if self.rag_client is None:
            return False

        # Check preliminary scores if available
        if self._preliminary_scores:
            sufficiency_score = self._preliminary_scores.get(SUFFICIENCY, 5.0)
            if sufficiency_score < self.rag_threshold:
                logger.info(f"RAG trigger: sufficiency score {sufficiency_score} < threshold {self.rag_threshold}")
                return True

        # Check for evidence gap indicators in paper content
        evidence_gaps = [
            "insufficient evidence",
            "more research needed",
            "limited data",
            "small sample",
            "preliminary findings",
            "further study required",
            "evidence is lacking",
        ]

        text_to_check = f"{paper.abstract} {paper.content}".lower()
        for gap in evidence_gaps:
            if gap in text_to_check:
                logger.info(f"RAG trigger: found evidence gap indicator '{gap}'")
                return True

        return False

    def _construct_rag_query(self, paper: ParsedPaper) -> str:
        """
        Construct query for RAG retrieval.

        Queries for supplementary empirical studies.
        """
        queries = []

        # Query by title keywords for related empirical work
        title_words = paper.title.split()[:8]
        queries.append(f"empirical study: {' '.join(title_words)}")

        # Query by discipline for relevant evidence
        if paper.discipline:
            queries.append(f"{paper.discipline} empirical evidence")

        # Query for specific evidence types
        queries.append("empirical study experimental evidence")
        queries.append("quantitative qualitative evidence")

        # Query by keywords
        if paper.keywords:
            queries.append(f"research on: {', '.join(paper.keywords[:3])}")

        return " | ".join(queries)

    # =========================================================================
    # System Prompt
    # =========================================================================

    def get_system_prompt(self) -> str:
        """Return the system prompt for evidence evaluation."""
        return """You are the Evidence Sub-agent for academic paper evaluation.

Your task is to evaluate papers on their evidence quality:

1. STRENGTH: How strong is the evidence supporting the claims?
   - Look for: direct evidence, robust findings, multiple lines of evidence
   - Red flags: indirect evidence, weak findings, single line of evidence

2. CONSISTENCY: How consistent is the evidence?
   - Look for: internal consistency, alignment with prior work
   - Red flags: contradictions, anomalies, conflicts with prior research

3. SUFFICIENCY: Is there sufficient evidence for the conclusions?
   - Look for: adequate evidence for claims, comprehensive coverage
   - Red flags: evidence gaps, overreaching conclusions, missing evidence

SCORING GUIDE (0-5 scale):
- 5: Compelling evidence, no gaps
- 4: Strong evidence, minor gaps
- 3: Adequate evidence, some concerns
- 2: Weak evidence, significant gaps
- 1: Insufficient evidence, major concerns
- 0: No evidence or severely flawed

IMPORTANT: If sufficiency < 3.0, RAG will be triggered to retrieve supplementary empirical studies.

Output your evaluation as valid JSON with scores and rationales for each dimension."""


# =============================================================================
# Factory Function
# =============================================================================

def create_evidence_subagent(
    llm_client: Optional[LLMClient] = None,
    rag_client: Optional[RAGClient] = None,
    rag_threshold: float = 3.0,
) -> EvidenceSubAgent:
    """
    Factory function to create an Evidence Sub-agent.

    Args:
        llm_client: LLM client for evaluation.
        rag_client: RAG client for supplementary study retrieval.
        rag_threshold: Score below this triggers RAG.

    Returns:
        Configured EvidenceSubAgent instance.
    """
    return EvidenceSubAgent(
        llm_client=llm_client,
        rag_client=rag_client,
        rag_threshold=rag_threshold,
    )
