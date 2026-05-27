"""
Method Sub-agent Implementation

Evaluates academic papers on methodology dimensions:
- Rigor: How rigorous is the methodological approach?
- Validity: How valid are the methods and conclusions?
- Reproducibility: Can the study be reproduced by others?

This sub-agent evaluates the methodological quality of research,
with RAG integration to retrieve methodological best practices.

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
# Method Dimension Definitions
# =============================================================================

@dataclass
class MethodDimensionScore:
    """Score for a single method dimension."""
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


# Method dimension names
RIGOR = "rigor"
VALIDITY = "validity"
REPRODUCIBILITY = "reproducibility"

METHOD_DIMENSIONS = [RIGOR, VALIDITY, REPRODUCIBILITY]

# Dimension descriptions for LLM prompts
DIMENSION_DESCRIPTIONS = {
    RIGOR: (
        "Rigor: How rigorous is the methodological approach? "
        "Evaluate: Are methods clearly described and justified? "
        "Is the design appropriate for the research questions? "
        "Are statistical methods correctly applied?"
    ),
    VALIDITY: (
        "Validity: How valid are the methods and conclusions? "
        "Evaluate: Do the methods address the research questions? "
        "Are there threats to internal/external validity? "
        "Are conclusions supported by the evidence?"
    ),
    REPRODUCIBILITY: (
        "Reproducibility: Can the study be reproduced by others? "
        "Evaluate: Are methods described in sufficient detail? "
        "Is data/code available? "
        "Could another researcher replicate this work?"
    ),
}


# =============================================================================
# Method Sub-agent Implementation
# =============================================================================

class MethodSubAgent(BaseSubAgent):
    """
    Method Sub-agent for evaluating rigor, validity, and reproducibility.

    Evaluates papers on their methodological quality. Has RAG integration
    to retrieve methodological best practices when rigor is low.

    Key Design:
        - Three dimensions: rigor, validity, reproducibility
        - Each dimension outputs {score: 0-5, rationale: str, citations: List[str]}
        - RAG triggered when rigor < 3.0 to retrieve methodological literature
        - Uses LLMClient for evaluation

    Example::

        # Create with real LLM client
        llm_client = await LLMClient.from_env()
        method = MethodSubAgent(llm_client=llm_client, rag_client=rag_client)

        # Evaluate a paper
        result = await method.evaluate(paper)
        print(f"Method score: {result.overall_score}")

    Example (RAG trigger)::

        # If rigor < 3.0, RAG is triggered to retrieve methodological best practices
        # This helps identify what's missing and suggests improvements
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_client: Optional[RAGClient] = None,
        timeout_seconds: float = 60.0,
        rag_threshold: float = 3.0,  # Trigger RAG if rigor < 3.0
    ):
        """
        Initialize Method Sub-agent.

        Args:
            llm_client: LLM client for evaluation. Can be None for mock testing.
            rag_client: RAG client for retrieving methodological literature.
            timeout_seconds: Timeout for evaluation.
            rag_threshold: Score below this triggers RAG retrieval.
        """
        super().__init__(
            dimension=EvaluationDimension.METHOD,
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
            DimensionResult with method scores and narrative.
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
            # Give more weight to rigor
            weights = {RIGOR: 1.5, VALIDITY: 1.0, REPRODUCIBILITY: 1.0}
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
            if rag_citations and dimensions.get(RIGOR, {}).get("score", 5.0) < self.rag_threshold:
                status = AgentStatus.RAG_TRIGGERED

            return DimensionResult(
                dimension=EvaluationDimension.METHOD,
                overall_score=round(overall_score, 1),
                sub_dimensions=sub_dimensions,
                narrative=narrative,
                rag_citations=rag_citations,
                status=status,
            )

        except Exception as e:
            logger.error(f"Method evaluation failed: {e}")
            return DimensionResult(
                dimension=EvaluationDimension.METHOD,
                overall_score=0.0,
                sub_dimensions=[],
                narrative=f"Method evaluation failed: {str(e)}",
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

        # RAG citations if available (methodological best practices)
        if rag_citations:
            sections.append("\n## Methodological References (from RAG)")
            for citation in rag_citations[:5]:
                sections.append(f"- {citation}")

        # Evaluation instructions
        sections.append("\n# Evaluation Task")
        sections.append(
            "Evaluate this paper's methodology. "
            "For each dimension, provide a score (0-5) and rationale.\n"
        )
        sections.append("Output JSON format:")
        sections.append("""{
    "rigor": {
        "score": <0-5>,
        "rationale": "explanation of rigor assessment"
    },
    "validity": {
        "score": <0-5>,
        "rationale": "explanation of validity assessment"
    },
    "reproducibility": {
        "score": <0-5>,
        "rationale": "explanation of reproducibility assessment"
    }
}""")

        # Dimension descriptions
        sections.append("\n# Dimension Definitions")
        for dim, desc in DIMENSION_DESCRIPTIONS.items():
            sections.append(f"- {desc}")

        # Scoring guide
        sections.append("\n# Scoring Guide")
        sections.append("""
- 5: Exemplary methodology, no concerns
- 4: Strong methodology, minor issues
- 3: Adequate methodology, some concerns
- 2: Weak methodology, significant issues
- 1: Poor methodology, major concerns
- 0: Severely flawed methodology
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
        for dim in METHOD_DIMENSIONS:
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

        for dim in METHOD_DIMENSIONS:
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
        parts = [f"## Method Evaluation Summary\n"]
        parts.append(f"Overall Method Score: {overall_score:.1f}/5.0\n")

        for dim in METHOD_DIMENSIONS:
            data = dimensions.get(dim, {})
            score = data.get("score", 0.0)
            rationale = data.get("rationale", "No rationale provided")
            parts.append(f"**{dim.title()}** ({score:.1f}/5.0): {rationale}")

        # Add warning if rigor is low
        rigor_score = dimensions.get(RIGOR, {}).get("score", 5.0)
        if rigor_score < 3.0:
            parts.append(f"\n⚠️ **Warning**: Low rigor score ({rigor_score:.1f}). "
                        "Consider reviewing methodological approach.")

        return "\n".join(parts)

    def _get_mock_dimensions(self) -> Dict[str, Dict[str, Any]]:
        """Return mock dimensions for testing."""
        return {
            RIGOR: {"score": 3.5, "rationale": "Mock evaluation"},
            VALIDITY: {"score": 3.5, "rationale": "Mock evaluation"},
            REPRODUCIBILITY: {"score": 3.5, "rationale": "Mock evaluation"},
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
            dimension=EvaluationDimension.METHOD,
            overall_score=round(overall_score, 1),
            sub_dimensions=sub_dimensions,
            narrative="Mock method evaluation",
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
        1. Rigor score < rag_threshold (default 3.0)
        2. Or if preliminary scores not available, check for methodological red flags

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
            rigor_score = self._preliminary_scores.get(RIGOR, 5.0)
            if rigor_score < self.rag_threshold:
                logger.info(f"RAG trigger: rigor score {rigor_score} < threshold {self.rag_threshold}")
                return True

        # Check for methodological red flags in paper content
        red_flags = [
            "methodology unclear",
            "methods not described",
            "no statistical analysis",
            "sample size not justified",
            "validity concerns",
            "reproducibility issues",
        ]

        text_to_check = f"{paper.abstract} {paper.content}".lower()
        for flag in red_flags:
            if flag in text_to_check:
                logger.info(f"RAG trigger: found red flag '{flag}'")
                return True

        return False

    def _construct_rag_query(self, paper: ParsedPaper) -> str:
        """
        Construct query for RAG retrieval.

        Queries for methodological best practices and guidelines.
        """
        queries = []

        # Query by discipline for methodological standards
        if paper.discipline:
            queries.append(f"{paper.discipline} research methodology best practices")

        # Query for specific methodological concerns
        queries.append("research methodology rigor validity reproducibility")
        queries.append("statistical methods best practices")

        # Query by keywords
        if paper.keywords:
            method_keywords = [kw for kw in paper.keywords if any(
                term in kw.lower() for term in ["method", "design", "analysis", "statistical"]
            )]
            if method_keywords:
                queries.append(f"methodology: {', '.join(method_keywords[:3])}")

        return " | ".join(queries)

    # =========================================================================
    # System Prompt
    # =========================================================================

    def get_system_prompt(self) -> str:
        """Return the system prompt for method evaluation."""
        return """You are the Method Sub-agent for academic paper evaluation.

Your task is to evaluate papers on their methodological quality:

1. RIGOR: How rigorous is the methodological approach?
   - Look for: clear methods, appropriate design, correct statistics
   - Red flags: vague methods, inappropriate design, statistical errors

2. VALIDITY: How valid are the methods and conclusions?
   - Look for: methods address research questions, conclusions supported
   - Red flags: validity threats, overgeneralization, unsupported claims

3. REPRODUCIBILITY: Can the study be reproduced by others?
   - Look for: detailed methods, available data/code, clear procedures
   - Red flags: missing details, unavailable data, ambiguous procedures

SCORING GUIDE (0-5 scale):
- 5: Exemplary methodology, no concerns
- 4: Strong methodology, minor issues
- 3: Adequate methodology, some concerns
- 2: Weak methodology, significant issues
- 1: Poor methodology, major concerns
- 0: Severely flawed methodology

IMPORTANT: If rigor < 3.0, RAG will be triggered to retrieve methodological best practices.

Output your evaluation as valid JSON with scores and rationales for each dimension."""


# =============================================================================
# Factory Function
# =============================================================================

def create_method_subagent(
    llm_client: Optional[LLMClient] = None,
    rag_client: Optional[RAGClient] = None,
    rag_threshold: float = 3.0,
) -> MethodSubAgent:
    """
    Factory function to create a Method Sub-agent.

    Args:
        llm_client: LLM client for evaluation.
        rag_client: RAG client for methodological literature retrieval.
        rag_threshold: Score below this triggers RAG.

    Returns:
        Configured MethodSubAgent instance.
    """
    return MethodSubAgent(
        llm_client=llm_client,
        rag_client=rag_client,
        rag_threshold=rag_threshold,
    )
