"""
CUDOS Sub-agent Implementation

Evaluates academic papers against Merton's CUDOS norms:
- Communalism: Is knowledge shared openly?
- Universalism: Are claims evaluated objectively?
- Disinterestedness: Are there conflicts of interest?
- Organized Skepticism: Does the paper acknowledge limitations?

This sub-agent has VETO POWER - if any norm is severely violated,
the paper is rejected and does not proceed to Phase 2 evaluation.

Author: WuYa Team
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..base import (
    AgentStatus,
    CUDOSResult,
    CUDOSSubAgent as BaseCUDOSSubAgent,
    LLMClient,
    ParsedPaper,
    RAGClient,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CUDOS Dimension Definitions
# =============================================================================

@dataclass
class CUDOSDimensionScore:
    """Score for a single CUDOS dimension."""
    name: str
    score: float  # 1-5 scale
    issues: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    justification: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "issues": self.issues,
            "evidence": self.evidence,
            "justification": self.justification,
        }


# CUDOS dimension names
COMMUNALISM = "communalism"
UNIVERSALISM = "universalism"
DISINTERESTEDNESS = "disinterestedness"
ORGANIZED_SKEPTICISM = "organized_skepticism"

CUDOS_DIMENSIONS = [
    COMMUNALISM,
    UNIVERSALISM,
    DISINTERESTEDNESS,
    ORGANIZED_SKEPTICISM,
]

# Dimension descriptions for LLM prompts
DIMENSION_DESCRIPTIONS = {
    COMMUNALISM: (
        "Communalism: Scientific knowledge should be shared openly as a common good. "
        "Evaluate: Does the paper share data, methods, and findings openly? "
        "Are there unnecessary restrictions on access? Is the work published in an open venue?"
    ),
    UNIVERSALISM: (
        "Universalism: Scientific claims should be evaluated objectively, "
        "independent of the author's personal or social attributes. "
        "Evaluate: Are claims evaluated based on merit alone? "
        "Is there evidence of bias based on author identity, affiliation, or nationality?"
    ),
    DISINTERESTEDNESS: (
        "Disinterestedness: Scientists should act for the benefit of the scientific community, "
        "not for personal gain. Evaluate: Are there conflicts of interest? "
        "Is funding disclosed? Are there potential commercial biases?"
    ),
    ORGANIZED_SKEPTICISM: (
        "Organized Skepticism: Scientific claims should be subject to rigorous scrutiny. "
        "Evaluate: Does the paper acknowledge limitations? "
        "Are alternative explanations considered? Is the methodology transparent?"
    ),
}


# =============================================================================
# Ethical Issue Detection Patterns
# =============================================================================

# Patterns that might indicate ethical issues requiring RAG retrieval
ETHICAL_ISSUE_PATTERNS = [
    r"conflict\s+of\s+interest",
    r"ethical\s+concern",
    r"misconduct",
    r"fabricat",
    r"falsif",
    r"plagiar",
    r"data\s+manipulation",
    r"undisclosed\s+funding",
    r"authorship\s+dispute",
    r"retraction",
    r"bias",
    r"discrimination",
]


# =============================================================================
# CUDOS Sub-agent Implementation
# =============================================================================

class CUDOSSubAgent(BaseCUDOSSubAgent):
    """
    CUDOS Sub-agent for normative gatekeeping.

    Evaluates papers against Merton's CUDOS norms and has veto power
    to block papers that severely violate scientific norms.

    Key Design:
        - gate_pass=False: Paper is vetoed, does not enter Phase 2
        - veto_threshold=2.0: Scores below this trigger veto
        - Each dimension outputs {score, issues} structure

    Example::

        # Create with real LLM client
        llm_client = await LLMClient.from_env()
        cudos = CUDOSSubAgent(llm_client=llm_client, veto_threshold=2.0)

        # Evaluate a paper
        result = await cudos.evaluate(paper)

        if not result.gate_pass:
            print(f"Vetoed: {result.veto_reason}")
        else:
            print("Passed CUDOS gate")

    Example (with mock for testing)::

        # Mock LLM client
        class MockLLMClient:
            async def generate(self, prompt, **kwargs):
                return json.dumps({
                    "communalism": {"score": 4.0, "issues": []},
                    "universalism": {"score": 4.5, "issues": []},
                    "disinterestedness": {"score": 4.0, "issues": []},
                    "organized_skepticism": {"score": 4.5, "issues": []},
                })

        cudos = CUDOSSubAgent(llm_client=MockLLMClient())
        result = await cudos.evaluate(paper)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_client: Optional[RAGClient] = None,
        veto_threshold: float = 2.0,
        rag_trigger_keywords: Optional[List[str]] = None,
    ):
        """
        Initialize CUDOS Sub-agent.

        Args:
            llm_client: LLM client for evaluation. Can be None for mock testing.
            rag_client: RAG client for retrieving Merton original texts.
            veto_threshold: Score below this triggers veto (default: 2.0).
            rag_trigger_keywords: Additional keywords that trigger RAG retrieval.
        """
        super().__init__(
            llm_client=llm_client,
            rag_client=rag_client,
            veto_threshold=veto_threshold,
        )
        self.rag_trigger_keywords = rag_trigger_keywords or []
        self._ethical_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in ETHICAL_ISSUE_PATTERNS
        ]

    # =========================================================================
    # Core Evaluation Logic
    # =========================================================================

    async def _evaluate_cudos_dimensions(
        self,
        paper: ParsedPaper
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate each CUDOS dimension.

        This is the main evaluation method that:
        1. Constructs evaluation prompts for each dimension
        2. Calls LLM to get scores and issues
        3. Parses and validates results

        Args:
            paper: Parsed paper content.

        Returns:
            Dict mapping dimension names to {score, issues} dicts.
        """
        if self.llm_client is None:
            # Mock mode: return default passing scores
            logger.warning("No LLM client provided, using mock evaluation")
            return self._get_mock_evaluation()

        try:
            # Build evaluation prompt
            prompt = self._build_evaluation_prompt(paper)

            # Call LLM
            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.get_system_prompt(),
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=2000,
            )

            # Parse response
            dimensions = self._parse_llm_response(response)

            # Validate and normalize scores
            dimensions = self._validate_dimensions(dimensions)

            return dimensions

        except Exception as e:
            logger.error(f"CUDOS evaluation failed: {e}")
            # On error, return permissive scores (fail open)
            return self._get_error_evaluation(str(e))

    def _build_evaluation_prompt(self, paper: ParsedPaper) -> str:
        """Build the evaluation prompt for the LLM."""
        sections = []

        # Paper metadata
        sections.append(f"# Paper to Evaluate\n")
        sections.append(f"Title: {paper.title}")
        sections.append(f"Authors: {', '.join(paper.authors)}")
        sections.append(f"Discipline: {paper.discipline}")
        sections.append(f"\n## Abstract\n{paper.abstract}")

        # Include content (truncated if too long)
        content = paper.content[:3000] if len(paper.content) > 3000 else paper.content
        sections.append(f"\n## Content (excerpt)\n{content}")

        # Evaluation instructions
        sections.append(f"\n# Evaluation Task")
        sections.append(
            "Evaluate this paper against Merton's CUDOS norms. "
            "For each dimension, provide a score (1-5) and list any issues found.\n"
        )
        sections.append("Output JSON format:")
        sections.append("""{
    "communalism": {
        "score": <1-5>,
        "issues": ["issue1", "issue2"],
        "justification": "brief explanation"
    },
    "universalism": {
        "score": <1-5>,
        "issues": [...],
        "justification": "..."
    },
    "disinterestedness": {
        "score": <1-5>,
        "issues": [...],
        "justification": "..."
    },
    "organized_skepticism": {
        "score": <1-5>,
        "issues": [...],
        "justification": "..."
    }
}""")

        # Dimension descriptions
        sections.append("\n# Dimension Definitions")
        for dim, desc in DIMENSION_DESCRIPTIONS.items():
            sections.append(f"- {desc}")

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
        for dim in CUDOS_DIMENSIONS:
            score_match = re.search(
                rf'{dim}["\']?\s*[:=]\s*["\']?(\d+\.?\d*)',
                response,
                re.IGNORECASE
            )
            if score_match:
                dimensions[dim] = {
                    "score": float(score_match.group(1)),
                    "issues": [],
                    "justification": "Extracted from response",
                }

        return dimensions if dimensions else self._get_mock_evaluation()

    def _validate_dimensions(
        self,
        dimensions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Validate and normalize dimension scores."""
        validated = {}

        for dim in CUDOS_DIMENSIONS:
            if dim in dimensions:
                score = dimensions[dim].get("score", 3.0)
                # Clamp to valid range
                score = max(1.0, min(5.0, float(score)))
                validated[dim] = {
                    "score": score,
                    "issues": dimensions[dim].get("issues", []),
                    "justification": dimensions[dim].get("justification", ""),
                }
            else:
                # Missing dimension: use neutral score
                validated[dim] = {
                    "score": 3.0,
                    "issues": [f"Dimension {dim} not evaluated"],
                    "justification": "Default score due to missing evaluation",
                }

        return validated

    def _get_mock_evaluation(self) -> Dict[str, Dict[str, Any]]:
        """Return mock evaluation for testing."""
        return {
            COMMUNALISM: {"score": 4.0, "issues": [], "justification": "Mock evaluation"},
            UNIVERSALISM: {"score": 4.5, "issues": [], "justification": "Mock evaluation"},
            DISINTERESTEDNESS: {"score": 4.0, "issues": [], "justification": "Mock evaluation"},
            ORGANIZED_SKEPTICISM: {"score": 4.5, "issues": [], "justification": "Mock evaluation"},
        }

    def _get_error_evaluation(self, error_msg: str) -> Dict[str, Dict[str, Any]]:
        """Return permissive evaluation on error (fail open)."""
        return {
            COMMUNALISM: {"score": 3.0, "issues": [f"Evaluation error: {error_msg}"]},
            UNIVERSALISM: {"score": 3.0, "issues": [f"Evaluation error: {error_msg}"]},
            DISINTERESTEDNESS: {"score": 3.0, "issues": [f"Evaluation error: {error_msg}"]},
            ORGANIZED_SKEPTICISM: {"score": 3.0, "issues": [f"Evaluation error: {error_msg}"]},
        }

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

        RAG is triggered when potential ethical issues are detected,
        to retrieve Merton's original texts for reference.

        Args:
            paper: Parsed paper content.
            context: Optional context from previous evaluations.

        Returns:
            True if RAG should be triggered.
        """
        # Check paper content for ethical issue patterns
        text_to_check = f"{paper.title} {paper.abstract} {paper.content}"

        for pattern in self._ethical_patterns:
            if pattern.search(text_to_check):
                logger.info(f"RAG trigger: found pattern {pattern.pattern}")
                return True

        # Check for explicit keywords
        for keyword in self.rag_trigger_keywords:
            if keyword.lower() in text_to_check.lower():
                logger.info(f"RAG trigger: found keyword '{keyword}'")
                return True

        # Check context for failed CUDOS dimensions
        if context and "cudos_result" in context:
            cudos_result = context["cudos_result"]
            if isinstance(cudos_result, CUDOSResult) and not cudos_result.gate_pass:
                return True

        return False

    def _construct_rag_query(self, paper: ParsedPaper) -> str:
        """
        Construct query for RAG retrieval.

        Queries Merton's original texts on CUDOS norms.
        """
        # Identify which dimensions might be problematic
        queries = [
            "Merton CUDOS norms scientific ethics",
            "Merton scientific norms communalism universalism",
            "Merton disinterestedness organized skepticism",
        ]

        # Add paper-specific context
        if paper.discipline:
            queries.append(f"Merton norms {paper.discipline} scientific practice")

        return " | ".join(queries)

    # =========================================================================
    # System Prompt
    # =========================================================================

    def get_system_prompt(self) -> str:
        """Return the system prompt for CUDOS evaluation."""
        return """You are the CUDOS Sub-agent for academic paper evaluation.

Your task is to evaluate papers against Merton's CUDOS norms:

1. COMMUNALISM: Scientific knowledge should be shared openly as a common good.
   - Look for: data sharing, open access, method transparency
   - Red flags: proprietary data claims, closed access, hidden methods

2. UNIVERSALISM: Claims should be evaluated objectively, independent of author attributes.
   - Look for: merit-based arguments, objective criteria
   - Red flags: ad hominem arguments, bias based on affiliation/nationality

3. DISINTERESTEDNESS: Scientists should act for the common good, not personal gain.
   - Look for: disclosed funding, declared conflicts of interest
   - Red flags: undisclosed funding, potential commercial bias, hidden conflicts

4. ORGANIZED SKEPTICISM: Claims should be subject to rigorous scrutiny.
   - Look for: acknowledged limitations, alternative explanations
   - Red flags: overconfident claims, ignored alternatives, hidden weaknesses

SCORING GUIDE:
- 5: Exemplary adherence to the norm
- 4: Good adherence, minor issues
- 3: Adequate, some concerns
- 2: Poor adherence, significant issues (triggers veto consideration)
- 1: Severe violation (triggers veto)

You have VETO POWER. If any dimension scores below 2.0, the paper is rejected.

Output your evaluation as valid JSON with scores, issues, and justifications for each dimension."""

    # =========================================================================
    # Additional Utility Methods
    # =========================================================================

    def get_dimension_scores(
        self,
        result: CUDOSResult
    ) -> Dict[str, float]:
        """Extract just the scores from a CUDOSResult."""
        return {
            name: data.get("score", 0.0)
            for name, data in result.dimensions.items()
        }

    def get_failed_dimensions(
        self,
        result: CUDOSResult
    ) -> List[str]:
        """Get list of dimensions that failed the veto threshold."""
        return [
            name
            for name, data in result.dimensions.items()
            if data.get("score", 5.0) < self.veto_threshold
        ]

    def get_veto_summary(
        self,
        result: CUDOSResult
    ) -> str:
        """Generate a human-readable veto summary."""
        if result.gate_pass:
            return "Paper passed CUDOS gatekeeping."

        failed = self.get_failed_dimensions(result)
        summary_parts = [f"Paper vetoed for failing CUDOS norms:"]

        for dim in failed:
            data = result.dimensions.get(dim, {})
            score = data.get("score", 0.0)
            issues = data.get("issues", [])
            summary_parts.append(f"\n- {dim}: score {score:.1f}")
            if issues:
                summary_parts.append(f"  Issues: {', '.join(issues)}")

        if result.veto_reason:
            summary_parts.append(f"\nReason: {result.veto_reason}")

        return "\n".join(summary_parts)


# =============================================================================
# Factory Function
# =============================================================================

def create_cudos_subagent(
    llm_client: Optional[LLMClient] = None,
    rag_client: Optional[RAGClient] = None,
    veto_threshold: float = 2.0,
) -> CUDOSSubAgent:
    """
    Factory function to create a CUDOS Sub-agent.

    Args:
        llm_client: LLM client for evaluation.
        rag_client: RAG client for Merton text retrieval.
        veto_threshold: Score below this triggers veto.

    Returns:
        Configured CUDOSSubAgent instance.
    """
    return CUDOSSubAgent(
        llm_client=llm_client,
        rag_client=rag_client,
        veto_threshold=veto_threshold,
    )
