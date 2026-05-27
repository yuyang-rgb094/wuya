"""
Innovation Sub-agent Implementation

Evaluates academic papers on innovation dimensions:
- Novelty: How new/original is the contribution?
- Significance: How important is the contribution to the field?
- Advancement: How much does this advance the state of the art?

This sub-agent evaluates the creative and innovative aspects of research,
identifying breakthrough contributions and novel approaches.

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
# Innovation Dimension Definitions
# =============================================================================

@dataclass
class InnovationDimensionScore:
    """Score for a single innovation dimension."""
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


# Innovation dimension names
NOVELTY = "novelty"
SIGNIFICANCE = "significance"
ADVANCEMENT = "advancement"

INNOVATION_DIMENSIONS = [NOVELTY, SIGNIFICANCE, ADVANCEMENT]

# Dimension descriptions for LLM prompts
DIMENSION_DESCRIPTIONS = {
    NOVELTY: (
        "Novelty: How new and original is the contribution? "
        "Evaluate: Does this introduce new concepts, methods, or findings? "
        "Is this a significant departure from existing work? "
        "Are the ideas fresh and not derivative?"
    ),
    SIGNIFICANCE: (
        "Significance: How important is the contribution to the field? "
        "Evaluate: Does this address an important problem? "
        "Will this influence future research directions? "
        "Is the potential impact high?"
    ),
    ADVANCEMENT: (
        "Advancement: How much does this advance the state of the art? "
        "Evaluate: Does this push the boundaries of knowledge? "
        "Does it improve upon existing methods significantly? "
        "Is it a breakthrough or incremental improvement?"
    ),
}


# =============================================================================
# Innovation Sub-agent Implementation
# =============================================================================

class InnovationSubAgent(BaseSubAgent):
    """
    Innovation Sub-agent for evaluating novelty, significance, and advancement.

    Evaluates papers on their innovative contributions and potential impact
    on the field. Identifies breakthrough research and novel approaches.

    Key Design:
        - Three dimensions: novelty, significance, advancement
        - Each dimension outputs {score: 0-5, rationale: str, citations: List[str]}
        - Uses LLMClient for evaluation
        - Can trigger RAG to compare with existing literature

    Example::

        # Create with real LLM client
        llm_client = await LLMClient.from_env()
        innovation = InnovationSubAgent(llm_client=llm_client)

        # Evaluate a paper
        result = await innovation.evaluate(paper)
        print(f"Innovation score: {result.overall_score}")

    Example (with mock for testing)::

        class MockLLMClient:
            async def generate(self, prompt, **kwargs):
                return json.dumps({
                    "novelty": {"score": 4.0, "rationale": "Novel approach"},
                    "significance": {"score": 4.5, "rationale": "Important problem"},
                    "advancement": {"score": 3.5, "rationale": "Moderate advancement"},
                })

        innovation = InnovationSubAgent(llm_client=MockLLMClient())
        result = await innovation.evaluate(paper)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_client: Optional[RAGClient] = None,
        timeout_seconds: float = 60.0,
        rag_threshold: float = 3.0,  # Trigger RAG if any dimension < 3.0
    ):
        """
        Initialize Innovation Sub-agent.

        Args:
            llm_client: LLM client for evaluation. Can be None for mock testing.
            rag_client: RAG client for retrieving related literature.
            timeout_seconds: Timeout for evaluation.
            rag_threshold: Score below this triggers RAG retrieval.
        """
        super().__init__(
            dimension=EvaluationDimension.INNOVATION,
            llm_client=llm_client,
            rag_client=rag_client,
            timeout_seconds=timeout_seconds,
            rag_threshold=rag_threshold,
        )

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
            DimensionResult with innovation scores and narrative.
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
            overall_score = sum(s.score for s in sub_dimensions) / len(sub_dimensions)

            # Generate narrative
            narrative = self._generate_narrative(dimensions, overall_score)

            return DimensionResult(
                dimension=EvaluationDimension.INNOVATION,
                overall_score=round(overall_score, 1),
                sub_dimensions=sub_dimensions,
                narrative=narrative,
                rag_citations=rag_citations,
                status=AgentStatus.SUCCESS,
            )

        except Exception as e:
            logger.error(f"Innovation evaluation failed: {e}")
            return DimensionResult(
                dimension=EvaluationDimension.INNOVATION,
                overall_score=0.0,
                sub_dimensions=[],
                narrative=f"Innovation evaluation failed: {str(e)}",
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

        # RAG citations if available
        if rag_citations:
            sections.append("\n## Related Literature (from RAG)")
            for citation in rag_citations[:5]:
                sections.append(f"- {citation}")

        # Evaluation instructions
        sections.append("\n# Evaluation Task")
        sections.append(
            "Evaluate this paper's innovation. "
            "For each dimension, provide a score (0-5) and rationale.\n"
        )
        sections.append("Output JSON format:")
        sections.append("""{
    "novelty": {
        "score": <0-5>,
        "rationale": "explanation of novelty assessment"
    },
    "significance": {
        "score": <0-5>,
        "rationale": "explanation of significance assessment"
    },
    "advancement": {
        "score": <0-5>,
        "rationale": "explanation of advancement assessment"
    }
}""")

        # Dimension descriptions
        sections.append("\n# Dimension Definitions")
        for dim, desc in DIMENSION_DESCRIPTIONS.items():
            sections.append(f"- {desc}")

        # Scoring guide
        sections.append("\n# Scoring Guide")
        sections.append("""
- 5: Exceptional/groundbreaking contribution
- 4: Strong contribution with clear impact
- 3: Moderate contribution, some novelty
- 2: Limited contribution, incremental work
- 1: Minimal contribution, mostly derivative
- 0: No innovation, entirely derivative or trivial
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
        for dim in INNOVATION_DIMENSIONS:
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

        for dim in INNOVATION_DIMENSIONS:
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
        parts = [f"## Innovation Evaluation Summary\n"]
        parts.append(f"Overall Innovation Score: {overall_score:.1f}/5.0\n")

        for dim in INNOVATION_DIMENSIONS:
            data = dimensions.get(dim, {})
            score = data.get("score", 0.0)
            rationale = data.get("rationale", "No rationale provided")
            parts.append(f"**{dim.title()}** ({score:.1f}/5.0): {rationale}")

        return "\n".join(parts)

    def _get_mock_dimensions(self) -> Dict[str, Dict[str, Any]]:
        """Return mock dimensions for testing."""
        return {
            NOVELTY: {"score": 3.5, "rationale": "Mock evaluation"},
            SIGNIFICANCE: {"score": 3.5, "rationale": "Mock evaluation"},
            ADVANCEMENT: {"score": 3.5, "rationale": "Mock evaluation"},
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
            dimension=EvaluationDimension.INNOVATION,
            overall_score=round(overall_score, 1),
            sub_dimensions=sub_dimensions,
            narrative="Mock innovation evaluation",
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

        RAG is triggered to compare with existing literature when
        evaluating novelty and advancement.

        Args:
            paper: Parsed paper content.
            context: Optional context from previous evaluations.

        Returns:
            True if RAG should be triggered.
        """
        # Always trigger RAG for innovation evaluation to compare with literature
        # This helps assess novelty by finding related work
        if self.rag_client is not None:
            return True
        return False

    def _construct_rag_query(self, paper: ParsedPaper) -> str:
        """
        Construct query for RAG retrieval.

        Queries for related work to compare novelty and advancement.
        """
        queries = []

        # Query by title keywords
        title_words = paper.title.split()[:10]
        queries.append(f"related work: {' '.join(title_words)}")

        # Query by abstract keywords
        if paper.keywords:
            queries.append(f"research on: {', '.join(paper.keywords[:5])}")

        # Query by discipline
        if paper.discipline:
            queries.append(f"{paper.discipline} state of the art")

        return " | ".join(queries)

    # =========================================================================
    # System Prompt
    # =========================================================================

    def get_system_prompt(self) -> str:
        """Return the system prompt for innovation evaluation."""
        return """You are the Innovation Sub-agent for academic paper evaluation.

Your task is to evaluate papers on their innovative contributions:

1. NOVELTY: How new and original is the contribution?
   - Look for: new concepts, methods, findings, approaches
   - Red flags: derivative work, minor variations, known ideas

2. SIGNIFICANCE: How important is the contribution to the field?
   - Look for: important problems, wide applicability, influence potential
   - Red flags: trivial problems, narrow scope, limited impact

3. ADVANCEMENT: How much does this advance the state of the art?
   - Look for: breakthroughs, significant improvements, new capabilities
   - Red flags: incremental improvements, marginal gains

SCORING GUIDE (0-5 scale):
- 5: Exceptional/groundbreaking contribution
- 4: Strong contribution with clear impact
- 3: Moderate contribution, some novelty
- 2: Limited contribution, incremental work
- 1: Minimal contribution, mostly derivative
- 0: No innovation, entirely derivative or trivial

Output your evaluation as valid JSON with scores and rationales for each dimension."""


# =============================================================================
# Factory Function
# =============================================================================

def create_innovation_subagent(
    llm_client: Optional[LLMClient] = None,
    rag_client: Optional[RAGClient] = None,
    rag_threshold: float = 3.0,
) -> InnovationSubAgent:
    """
    Factory function to create an Innovation Sub-agent.

    Args:
        llm_client: LLM client for evaluation.
        rag_client: RAG client for literature retrieval.
        rag_threshold: Score below this triggers RAG.

    Returns:
        Configured InnovationSubAgent instance.
    """
    return InnovationSubAgent(
        llm_client=llm_client,
        rag_client=rag_client,
        rag_threshold=rag_threshold,
    )
