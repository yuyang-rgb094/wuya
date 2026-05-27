"""
Basic Usage Example for WuYa Multi-Agent System

This example demonstrates the complete workflow of the WuYa system:
1. Parse a paper from text or PDF
2. Route through two-phase evaluation
3. Generate evaluation report

Author: WuYa Team
"""

import asyncio
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import WuYa components
from wuya_agents import (
    # Paper parsing
    PaperParser,
    ParsedPaper,
    create_paper_parser,

    # Sub-agents
    CUDOSSubAgent,
    InnovationSubAgent,
    MethodSubAgent,
    EvidenceSubAgent,
    ApplicationSubAgent,
    create_cudos_subagent,
    create_innovation_subagent,
    create_method_subagent,
    create_evidence_subagent,
    create_application_subagent,

    # Router
    TwoPhaseRouter,
    create_two_phase_router,

    # Aggregator
    ResultAggregator,
    create_result_aggregator,

    # DEA
    DEASubAgent,

    # Base types
    EvaluationDimension,
    AgentStatus,
)


# =============================================================================
# Sample Paper Data
# =============================================================================

SAMPLE_PAPER_TEXT = """
Title: A Novel Deep Learning Approach for Climate Prediction Using Transformer Networks

Authors: Dr. Sarah Chen, Prof. Michael Roberts, Dr. Emily Zhang

Abstract:
This paper presents a novel approach to climate prediction using transformer networks
that significantly improves long-term forecasting accuracy. Our method combines
attention mechanisms with physical constraints to model complex atmospheric dynamics.
We evaluate our approach on 20 years of historical climate data and demonstrate
a 15% improvement in prediction accuracy compared to state-of-the-art methods.

Keywords: deep learning, climate prediction, transformer networks, attention mechanisms

1. Introduction
Climate prediction is a critical challenge with significant implications for
agriculture, disaster preparedness, and policy making. Traditional numerical
weather prediction models, while effective for short-term forecasts, struggle
with long-term climate prediction due to the chaotic nature of atmospheric systems.

2. Methodology
We propose a transformer-based architecture that incorporates physical constraints
derived from fluid dynamics equations. The model uses multi-head attention to
capture long-range dependencies in climate data while maintaining physical
consistency through soft constraint regularization.

3. Results
Our experiments show significant improvements over baseline methods:
- 15% improvement in 30-day temperature prediction
- 12% improvement in precipitation forecasting
- Reduced computational cost by 40%

4. Discussion
The results demonstrate that incorporating physical constraints into deep learning
models can significantly improve climate prediction accuracy while maintaining
computational efficiency.

5. Conclusion
We present a novel approach that advances the state of the art in climate prediction.
Future work will explore extending this method to seasonal and decadal predictions.

References:
[1] Vaswani et al. (2017). Attention is all you need. NeurIPS.
[2] Rasp & Thuerey (2021). Data-driven medium-range weather prediction.
[3] IPCC (2021). Climate Change 2021: The Physical Science Basis.
"""


# =============================================================================
# Mock LLM Client for Testing
# =============================================================================

class MockLLMClient:
    """Mock LLM client for testing without actual API calls."""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response based on prompt content."""
        import json

        # Determine which sub-agent is calling based on prompt content
        if "CUDOS" in prompt or "cudos" in prompt.lower():
            return json.dumps({
                "communalism": {"score": 4.5, "issues": [], "justification": "Data and methods well documented"},
                "universalism": {"score": 4.5, "issues": [], "justification": "Claims evaluated objectively"},
                "disinterestedness": {"score": 4.0, "issues": [], "justification": "Funding disclosed"},
                "organized_skepticism": {"score": 4.5, "issues": [], "justification": "Limitations acknowledged"},
            })
        elif "innovation" in prompt.lower() or "novelty" in prompt.lower():
            return json.dumps({
                "novelty": {"score": 4.5, "rationale": "Novel application of transformers to climate"},
                "significance": {"score": 4.0, "rationale": "Important problem with real impact"},
                "advancement": {"score": 4.0, "rationale": "Significant improvement over baselines"},
            })
        elif "method" in prompt.lower() or "rigor" in prompt.lower():
            return json.dumps({
                "rigor": {"score": 4.0, "rationale": "Well-designed experiments"},
                "validity": {"score": 4.0, "rationale": "Appropriate validation approach"},
                "reproducibility": {"score": 3.5, "rationale": "Code available, data partially accessible"},
            })
        elif "evidence" in prompt.lower() or "strength" in prompt.lower():
            return json.dumps({
                "strength": {"score": 4.0, "rationale": "Strong empirical results"},
                "consistency": {"score": 4.0, "rationale": "Results consistent across datasets"},
                "sufficiency": {"score": 3.5, "rationale": "Good but could include more baselines"},
            })
        elif "application" in prompt.lower() or "relevance" in prompt.lower():
            return json.dumps({
                "relevance": {"score": 4.5, "rationale": "Highly relevant to climate science"},
                "feasibility": {"score": 4.0, "rationale": "Computationally efficient"},
                "impact": {"score": 4.5, "rationale": "Potential for significant real-world impact"},
            })

        return json.dumps({"score": 3.5, "rationale": "Default evaluation"})


# =============================================================================
# Helper Functions
# =============================================================================

def create_mock_reference_papers(n: int = 100) -> List[Dict[str, Any]]:
    """Create mock reference papers for DEA analysis."""
    import random
    import numpy as np

    np.random.seed(42)
    random.seed(42)

    reference_papers = []
    for i in range(n):
        ref_paper = {
            "paper_id": f"ref_{i:03d}",
            "scores": {
                "innovation": float(np.random.uniform(2.5, 5.0)),
                "method": float(np.random.uniform(3.0, 5.0)),
                "evidence": float(np.random.uniform(2.5, 5.0)),
                "application": float(np.random.uniform(2.0, 5.0)),
                "cudos": float(np.random.uniform(4.0, 5.0))
            },
            "citations": int(np.random.randint(10, 500))
        }
        reference_papers.append(ref_paper)

    return reference_papers


# =============================================================================
# Main Example
# =============================================================================

async def main():
    """
    Main example demonstrating the complete WuYa workflow.
    """
    print("=" * 80)
    print("WuYa Multi-Agent System - Basic Usage Example")
    print("=" * 80)

    # ======================================================================
    # Step 1: Parse Paper
    # ======================================================================
    print("\n[Step 1] Parsing paper...")

    parser = create_paper_parser(enable_cache=True, use_mock_pdf=False)
    paper = parser.parse_paper(
        raw_input=SAMPLE_PAPER_TEXT,
        source_type="text",
        paper_id="example_paper_001",
        discipline="computer_science",
    )

    print(f"  Paper ID: {paper.paper_id}")
    print(f"  Title: {paper.title}")
    print(f"  Authors: {', '.join(paper.authors) if paper.authors else 'N/A'}")
    print(f"  Keywords: {', '.join(paper.keywords) if paper.keywords else 'N/A'}")
    print(f"  Discipline: {paper.discipline}")

    # ======================================================================
    # Step 2: Initialize Sub-agents
    # ======================================================================
    print("\n[Step 2] Initializing sub-agents...")

    mock_llm = MockLLMClient()

    # Create sub-agents with mock LLM
    cudos_agent = create_cudos_subagent(llm_client=mock_llm, veto_threshold=2.0)
    innovation_agent = create_innovation_subagent(llm_client=mock_llm)
    method_agent = create_method_subagent(llm_client=mock_llm)
    evidence_agent = create_evidence_subagent(llm_client=mock_llm)
    application_agent = create_application_subagent(llm_client=mock_llm)

    # Create DEA sub-agent
    dea_agent = DEASubAgent(min_reference_papers=50)

    print("  ✓ CUDOS Sub-agent")
    print("  ✓ Innovation Sub-agent")
    print("  ✓ Method Sub-agent")
    print("  ✓ Evidence Sub-agent")
    print("  ✓ Application Sub-agent")
    print("  ✓ DEA Sub-agent")

    # ======================================================================
    # Step 3: Create Router
    # ======================================================================
    print("\n[Step 3] Creating TwoPhaseRouter...")

    router = create_two_phase_router(
        cudos_agent=cudos_agent,
        innovation_agent=innovation_agent,
        method_agent=method_agent,
        evidence_agent=evidence_agent,
        application_agent=application_agent,
        dea_agent=dea_agent,
        enable_parallel=True,
    )

    print("  ✓ Router initialized")

    # ======================================================================
    # Step 4: Route Paper for Evaluation
    # ======================================================================
    print("\n[Step 4] Routing paper for evaluation...")
    print("  (This may take a moment...)")

    # Create reference papers for DEA
    reference_papers = create_mock_reference_papers(n=100)

    # Run evaluation
    report = await router.route(
        paper=paper,
        target_journal="Nature Machine Intelligence",
        reference_papers=reference_papers,
    )

    # ======================================================================
    # Step 5: Display Results
    # ======================================================================
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Status
    print(f"\nStatus: {report.status.upper()}")
    print(f"Processing Time: {report.processing_time_ms}ms")

    # CUDOS
    print(f"\n{'─' * 40}")
    print("CUDOS GATEKEEPING")
    print(f"{'─' * 40}")
    if report.cudos_passed:
        print("✅ PASSED - Paper meets CUDOS normative standards")
        cudos_dims = report.cudos_details.get("dimensions", {})
        for dim, data in cudos_dims.items():
            score = data.get("score", 0)
            print(f"  • {dim}: {score:.1f}/5.0")
    else:
        print(f"❌ VETOED - {report.veto_reason}")
        return

    # Dimension Scores
    print(f"\n{'─' * 40}")
    print("DIMENSION SCORES")
    print(f"{'─' * 40}")
    for dim_name, score in report.dimension_scores.items():
        print(f"  • {dim_name.capitalize()}: {score:.2f}/5.0")

    # Overall
    print(f"\n{'─' * 40}")
    print("OVERALL ASSESSMENT")
    print(f"{'─' * 40}")
    print(f"  Overall Score: {report.overall_score:.2f}/5.0")
    print(f"  Tier Estimate: {report.tier_estimate}")
    print(f"  Confidence: {report.confidence}")

    # DEA Analysis
    if report.dea_summary:
        print(f"\n{'─' * 40}")
        print("DEA EFFICIENCY ANALYSIS")
        print(f"{'─' * 40}")
        print(f"  Efficiency Score: {report.dea_summary.efficiency_score:.3f}")
        ci = report.dea_summary.confidence_interval
        print(f"  95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
        print(f"  On Frontier: {'Yes' if report.dea_summary.is_on_frontier else 'No'}")

    # Journal Recommendations
    if report.journal_matches:
        print(f"\n{'─' * 40}")
        print("JOURNAL RECOMMENDATIONS")
        print(f"{'─' * 40}")
        for match in report.journal_matches:
            print(f"  • {match.journal_name}")
            print(f"    Match Score: {match.match_score:.2f}")
            print(f"    Tier: {match.tier_estimate}")

    # Improvement Suggestions
    if report.improvement_suggestions:
        print(f"\n{'─' * 40}")
        print("IMPROVEMENT SUGGESTIONS")
        print(f"{'─' * 40}")
        for suggestion in report.improvement_suggestions[:5]:  # Top 5
            print(f"  [{suggestion.priority.upper()}] {suggestion.dimension}")
            print(f"    {suggestion.suggestion}")

    # Errors
    if report.errors:
        print(f"\n{'─' * 40}")
        print("ERRORS")
        print(f"{'─' * 40}")
        for error in report.errors:
            print(f"  ⚠ {error}")

    # ======================================================================
    # Step 6: Export Report
    # ======================================================================
    print(f"\n{'─' * 40}")
    print("REPORT EXPORT")
    print(f"{'─' * 40}")

    # Export as dictionary
    report_dict = report.to_dict()
    print(f"  Report keys: {list(report_dict.keys())}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


# =============================================================================
# Alternative: Using ResultAggregator Directly
# =============================================================================

async def example_with_aggregator():
    """
    Example using ResultAggregator directly for custom workflows.
    """
    print("\n" + "=" * 80)
    print("Alternative: Using ResultAggregator Directly")
    print("=" * 80)

    # Parse paper
    parser = create_paper_parser()
    paper = parser.parse_paper(SAMPLE_PAPER_TEXT, discipline="computer_science")

    # Create mock results
    from wuya_agents import CUDOSResult, DimensionResult, SubDimensionScore

    cudos_result = CUDOSResult(
        gate_pass=True,
        dimensions={
            "communalism": {"score": 4.5, "issues": []},
            "universalism": {"score": 4.5, "issues": []},
            "disinterestedness": {"score": 4.0, "issues": []},
            "organized_skepticism": {"score": 4.5, "issues": []},
        },
    )

    dimension_results = {
        EvaluationDimension.INNOVATION: DimensionResult(
            dimension=EvaluationDimension.INNOVATION,
            overall_score=4.3,
            sub_dimensions=[
                SubDimensionScore("novelty", 4.5, justification="Novel approach"),
                SubDimensionScore("significance", 4.0, justification="Important problem"),
                SubDimensionScore("advancement", 4.5, justification="Clear advancement"),
            ],
            narrative="Strong innovation",
            status=AgentStatus.SUCCESS,
        ),
        EvaluationDimension.METHOD: DimensionResult(
            dimension=EvaluationDimension.METHOD,
            overall_score=4.0,
            sub_dimensions=[
                SubDimensionScore("rigor", 4.0, justification="Rigorous"),
                SubDimensionScore("validity", 4.0, justification="Valid"),
                SubDimensionScore("reproducibility", 4.0, justification="Reproducible"),
            ],
            narrative="Good methodology",
            status=AgentStatus.SUCCESS,
        ),
        EvaluationDimension.EVIDENCE: DimensionResult(
            dimension=EvaluationDimension.EVIDENCE,
            overall_score=4.0,
            sub_dimensions=[
                SubDimensionScore("strength", 4.0, justification="Strong evidence"),
                SubDimensionScore("consistency", 4.0, justification="Consistent"),
                SubDimensionScore("sufficiency", 4.0, justification="Sufficient"),
            ],
            narrative="Good evidence",
            status=AgentStatus.SUCCESS,
        ),
        EvaluationDimension.APPLICATION: DimensionResult(
            dimension=EvaluationDimension.APPLICATION,
            overall_score=4.5,
            sub_dimensions=[
                SubDimensionScore("relevance", 4.5, justification="Highly relevant"),
                SubDimensionScore("feasibility", 4.5, justification="Feasible"),
                SubDimensionScore("impact", 4.5, justification="High impact"),
            ],
            narrative="Strong application potential",
            status=AgentStatus.SUCCESS,
        ),
    }

    # Aggregate results
    aggregator = create_result_aggregator()
    report = await aggregator.aggregate(
        paper=paper,
        cudos_result=cudos_result,
        dimension_results=dimension_results,
        target_journal="Nature Machine Intelligence",
    )

    print(f"\nAggregated Report:")
    print(f"  Overall Score: {report.summary.overall_score:.2f}")
    print(f"  Tier Estimate: {report.summary.tier_estimate}")
    print(f"  Recommendation: {report.summary.recommendation}")

    # Export as markdown
    markdown = report.to_markdown()
    print(f"\nMarkdown report length: {len(markdown)} characters")
    print("\nFirst 500 characters of markdown report:")
    print(markdown[:500])


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    # Run main example
    asyncio.run(main())

    # Run aggregator example
    asyncio.run(example_with_aggregator())
