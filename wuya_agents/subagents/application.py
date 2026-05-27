"""
Application Sub-agent Implementation

Evaluates academic papers on application relevance dimensions:
- Relevance: How relevant is this work to practical applications?
- Feasibility: How feasible is it to apply this work in practice?
- Impact: What is the potential impact of applying this work?

This sub-agent evaluates the practical relevance and applicability
of research, identifying potential application scenarios.

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
# Application Dimension Definitions
# =============================================================================

@dataclass
class ApplicationDimensionScore:
    """Score for a single application dimension."""
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


@dataclass
class ApplicationScenario:
    """A potential application scenario for the research."""
    domain: str
    description: str
    feasibility: float  # 0-1
    impact: float  # 0-1
    barriers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "description": self.description,
            "feasibility": self.feasibility,
            "impact": self.impact,
            "barriers": self.barriers,
        }


# Application dimension names
RELEVANCE = "relevance"
FEASIBILITY = "feasibility"
IMPACT = "impact"

APPLICATION_DIMENSIONS = [RELEVANCE, FEASIBILITY, IMPACT]

# Dimension descriptions for LLM prompts
DIMENSION_DESCRIPTIONS = {
    RELEVANCE: (
        "Relevance: How relevant is this work to practical applications? "
        "Evaluate: Does this address real-world problems? "
        "Is there demand for this research? "
        "Does it connect to practical needs?"
    ),
    FEASIBILITY: (
        "Feasibility: How feasible is it to apply this work in practice? "
        "Evaluate: Can the results be implemented? "
        "Are there technical barriers? "
        "Is the cost/benefit favorable?"
    ),
    IMPACT: (
        "Impact: What is the potential impact of applying this work? "
        "Evaluate: How significant would the impact be? "
        "Who would benefit? "
        "Is the impact broad or narrow?"
    ),
}


# =============================================================================
# Application Sub-agent Implementation
# =============================================================================

class ApplicationSubAgent(BaseSubAgent):
    """
    Application Sub-agent for evaluating relevance, feasibility, and impact.

    Evaluates papers on their practical relevance and applicability.
    Identifies potential application scenarios and assesses barriers.

    Key Design:
        - Three dimensions: relevance, feasibility, impact
        - Each dimension outputs {score: 0-5, rationale: str, citations: List[str]}
        - Outputs application scenario suggestions
        - Uses LLMClient for evaluation

    Example::

        # Create with real LLM client
        llm_client = await LLMClient.from_env()
        application = ApplicationSubAgent(llm_client=llm_client)

        # Evaluate a paper
        result = await application.evaluate(paper)
        print(f"Application score: {result.overall_score}")

    Example (with application scenarios)::

        # The evaluation includes suggested application scenarios
        # with feasibility and impact assessments
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_client: Optional[RAGClient] = None,
        timeout_seconds: float = 60.0,
        rag_threshold: float = 3.0,
    ):
        """
        Initialize Application Sub-agent.

        Args:
            llm_client: LLM client for evaluation. Can be None for mock testing.
            rag_client: RAG client for retrieving application case studies.
            timeout_seconds: Timeout for evaluation.
            rag_threshold: Score below this triggers RAG retrieval.
        """
        super().__init__(
            dimension=EvaluationDimension.APPLICATION,
            llm_client=llm_client,
            rag_client=rag_client,
            timeout_seconds=timeout_seconds,
            rag_threshold=rag_threshold,
        )
        # Store application scenarios
        self._application_scenarios: List[ApplicationScenario] = []

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
            DimensionResult with application scores and narrative.
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
                max_tokens=2500,
            )

            # Parse response
            dimensions, scenarios = self._parse_llm_response(response)

            # Validate and normalize scores
            dimensions = self._validate_dimensions(dimensions)

            # Store application scenarios
            self._application_scenarios = scenarios

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
            # Give more weight to impact
            weights = {RELEVANCE: 1.0, FEASIBILITY: 1.0, IMPACT: 1.5}
            total_weight = sum(weights.values())
            weighted_sum = sum(
                data["score"] * weights.get(dim, 1.0)
                for dim, data in dimensions.items()
            )
            overall_score = weighted_sum / total_weight

            # Generate narrative
            narrative = self._generate_narrative(dimensions, overall_score, scenarios)

            return DimensionResult(
                dimension=EvaluationDimension.APPLICATION,
                overall_score=round(overall_score, 1),
                sub_dimensions=sub_dimensions,
                narrative=narrative,
                rag_citations=rag_citations,
                status=AgentStatus.SUCCESS,
            )

        except Exception as e:
            logger.error(f"Application evaluation failed: {e}")
            return DimensionResult(
                dimension=EvaluationDimension.APPLICATION,
                overall_score=0.0,
                sub_dimensions=[],
                narrative=f"Application evaluation failed: {str(e)}",
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

        # RAG citations if available (application case studies)
        if rag_citations:
            sections.append("\n## Application Case Studies (from RAG)")
            for citation in rag_citations[:5]:
                sections.append(f"- {citation}")

        # Evaluation instructions
        sections.append("\n# Evaluation Task")
        sections.append(
            "Evaluate this paper's practical application potential. "
            "For each dimension, provide a score (0-5) and rationale. "
            "Also suggest 2-3 potential application scenarios.\n"
        )
        sections.append("Output JSON format:")
        sections.append("""{
    "dimensions": {
        "relevance": {
            "score": <0-5>,
            "rationale": "explanation of relevance assessment"
        },
        "feasibility": {
            "score": <0-5>,
            "rationale": "explanation of feasibility assessment"
        },
        "impact": {
            "score": <0-5>,
            "rationale": "explanation of impact assessment"
        }
    },
    "scenarios": [
        {
            "domain": "application domain",
            "description": "how it could be applied",
            "feasibility": <0-1>,
            "impact": <0-1>,
            "barriers": ["barrier1", "barrier2"]
        }
    ]
}""")

        # Dimension descriptions
        sections.append("\n# Dimension Definitions")
        for dim, desc in DIMENSION_DESCRIPTIONS.items():
            sections.append(f"- {desc}")

        # Scoring guide
        sections.append("\n# Scoring Guide")
        sections.append("""
- 5: Highly applicable, immediate impact potential
- 4: Strong application potential, clear path to implementation
- 3: Moderate application potential, some barriers
- 2: Limited application potential, significant barriers
- 1: Minimal application potential, major barriers
- 0: No practical application potential
""")

        return "\n".join(sections)

    def _parse_llm_response(
        self, response: str
    ) -> tuple[Dict[str, Dict[str, Any]], List[ApplicationScenario]]:
        """Parse LLM response into structured dimension scores and scenarios."""
        dimensions = self._get_mock_dimensions()
        scenarios = []

        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())

                # Extract dimensions
                if "dimensions" in data:
                    dimensions = data["dimensions"]
                else:
                    # Try flat structure
                    for dim in APPLICATION_DIMENSIONS:
                        if dim in data:
                            dimensions[dim] = data[dim]

                # Extract scenarios
                if "scenarios" in data:
                    for s in data["scenarios"]:
                        scenarios.append(ApplicationScenario(
                            domain=s.get("domain", "Unknown"),
                            description=s.get("description", ""),
                            feasibility=float(s.get("feasibility", 0.5)),
                            impact=float(s.get("impact", 0.5)),
                            barriers=s.get("barriers", []),
                        ))

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")

        # Fallback: try to extract scores from text
        if not dimensions or all(dim not in dimensions for dim in APPLICATION_DIMENSIONS):
            for dim in APPLICATION_DIMENSIONS:
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

        return dimensions, scenarios

    def _validate_dimensions(
        self,
        dimensions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Validate and normalize dimension scores."""
        validated = {}

        for dim in APPLICATION_DIMENSIONS:
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
        overall_score: float,
        scenarios: List[ApplicationScenario]
    ) -> str:
        """Generate a narrative summary of the evaluation."""
        parts = [f"## Application Evaluation Summary\n"]
        parts.append(f"Overall Application Score: {overall_score:.1f}/5.0\n")

        for dim in APPLICATION_DIMENSIONS:
            data = dimensions.get(dim, {})
            score = data.get("score", 0.0)
            rationale = data.get("rationale", "No rationale provided")
            parts.append(f"**{dim.title()}** ({score:.1f}/5.0): {rationale}")

        # Add application scenarios
        if scenarios:
            parts.append("\n### Potential Application Scenarios")
            for i, scenario in enumerate(scenarios, 1):
                parts.append(f"\n{i}. **{scenario.domain}**")
                parts.append(f"   - Description: {scenario.description}")
                parts.append(f"   - Feasibility: {scenario.feasibility:.0%}")
                parts.append(f"   - Impact: {scenario.impact:.0%}")
                if scenario.barriers:
                    parts.append(f"   - Barriers: {', '.join(scenario.barriers)}")

        return "\n".join(parts)

    def _get_mock_dimensions(self) -> Dict[str, Dict[str, Any]]:
        """Return mock dimensions for testing."""
        return {
            RELEVANCE: {"score": 3.5, "rationale": "Mock evaluation"},
            FEASIBILITY: {"score": 3.5, "rationale": "Mock evaluation"},
            IMPACT: {"score": 3.5, "rationale": "Mock evaluation"},
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

        # Create mock scenarios
        self._application_scenarios = [
            ApplicationScenario(
                domain=paper.discipline or "Research",
                description="Potential application in the field",
                feasibility=0.6,
                impact=0.5,
                barriers=["Further validation needed"],
            )
        ]

        return DimensionResult(
            dimension=EvaluationDimension.APPLICATION,
            overall_score=round(overall_score, 1),
            sub_dimensions=sub_dimensions,
            narrative="Mock application evaluation",
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

        RAG is triggered to retrieve application case studies
        when evaluating practical relevance.

        Args:
            paper: Parsed paper content.
            context: Optional context from previous evaluations.

        Returns:
            True if RAG should be triggered.
        """
        # Trigger RAG if client is available to get application case studies
        if self.rag_client is not None:
            return True
        return False

    def _construct_rag_query(self, paper: ParsedPaper) -> str:
        """
        Construct query for RAG retrieval.

        Queries for application case studies and implementation examples.
        """
        queries = []

        # Query by title keywords for applications
        title_words = paper.title.split()[:8]
        queries.append(f"application case study: {' '.join(title_words)}")

        # Query by discipline for practical applications
        if paper.discipline:
            queries.append(f"{paper.discipline} practical applications")
            queries.append(f"{paper.discipline} implementation examples")

        # Query for technology transfer
        queries.append("technology transfer industry application")

        return " | ".join(queries)

    # =========================================================================
    # System Prompt
    # =========================================================================

    def get_system_prompt(self) -> str:
        """Return the system prompt for application evaluation."""
        return """You are the Application Sub-agent for academic paper evaluation.

Your task is to evaluate papers on their practical application potential:

1. RELEVANCE: How relevant is this work to practical applications?
   - Look for: real-world problems, practical needs, demand
   - Red flags: purely theoretical, no practical connection

2. FEASIBILITY: How feasible is it to apply this work in practice?
   - Look for: implementable results, clear path, favorable cost/benefit
   - Red flags: technical barriers, high cost, complexity

3. IMPACT: What is the potential impact of applying this work?
   - Look for: significant impact, broad benefit, transformative potential
   - Red flags: narrow impact, marginal benefit

SCORING GUIDE (0-5 scale):
- 5: Highly applicable, immediate impact potential
- 4: Strong application potential, clear path to implementation
- 3: Moderate application potential, some barriers
- 2: Limited application potential, significant barriers
- 1: Minimal application potential, major barriers
- 0: No practical application potential

Also suggest 2-3 potential application scenarios with:
- Domain of application
- Description of how it could be applied
- Feasibility (0-1)
- Impact (0-1)
- Key barriers

Output your evaluation as valid JSON with dimensions and scenarios."""

    # =========================================================================
    # Additional Utility Methods
    # =========================================================================

    def get_application_scenarios(self) -> List[ApplicationScenario]:
        """Get the application scenarios from the last evaluation."""
        return self._application_scenarios

    def get_top_scenario(self) -> Optional[ApplicationScenario]:
        """Get the top application scenario by impact * feasibility."""
        if not self._application_scenarios:
            return None
        return max(
            self._application_scenarios,
            key=lambda s: s.feasibility * s.impact
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_application_subagent(
    llm_client: Optional[LLMClient] = None,
    rag_client: Optional[RAGClient] = None,
    rag_threshold: float = 3.0,
) -> ApplicationSubAgent:
    """
    Factory function to create an Application Sub-agent.

    Args:
        llm_client: LLM client for evaluation.
        rag_client: RAG client for case study retrieval.
        rag_threshold: Score below this triggers RAG.

    Returns:
        Configured ApplicationSubAgent instance.
    """
    return ApplicationSubAgent(
        llm_client=llm_client,
        rag_client=rag_client,
        rag_threshold=rag_threshold,
    )
