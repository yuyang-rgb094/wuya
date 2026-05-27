"""
Frontier Discovery Example

Demonstrates the DEA frontier discovery mechanism:
1. Create sample submission history
2. Run frontier discovery analysis
3. Display improvement patterns, trends, and frontier shifts
4. Show self-improvement signals

Run:
    python examples/frontier_discovery.py

Author: WuYa Team
"""

import asyncio
import logging
import random

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_history(n: int = 50, seed: int = 42) -> list:
    """Generate synthetic submission history for demonstration."""
    random.seed(seed)

    disciplines = ["computer_science", "medicine", "biology", "physics", "social_science"]
    journals = ["Nature", "Science", "PNAS", "IEEE Trans", "PLOS ONE", "Lancet", "Cell"]
    outcomes = ["accepted", "rejected", "revision", "desk_reject"]
    outcome_weights = [0.25, 0.35, 0.30, 0.10]  # Realistic distribution

    feedback_templates = {
        "innovation": [
            "The novelty of this work is limited; similar approaches have been published.",
            "The contribution is incremental rather than groundbreaking.",
            "While the methodology is sound, the innovation is insufficient for this venue.",
            "The authors present a creative approach to an important problem.",
        ],
        "method": [
            "The sample size is too small to draw reliable conclusions.",
            "Statistical methods are appropriate but the experimental design has limitations.",
            "The methodology is rigorous and well-described.",
            "Reproducibility concerns: key parameters are not specified.",
        ],
        "evidence": [
            "The evidence supporting the main claims is insufficient.",
            "Results are consistent with prior work but lack strong causal evidence.",
            "Compelling evidence from multiple experiments supports the conclusions.",
            "Effect sizes are small and confidence intervals are wide.",
        ],
        "application": [
            "Practical implications are not well explored.",
            "The proposed method has clear real-world applications.",
            "Scalability and deployment considerations are missing.",
            "Industry relevance is high but clinical validation is needed.",
        ],
    }

    history = []
    for i in range(n):
        year = random.choice([2021, 2022, 2023, 2024])
        outcome = random.choices(outcomes, weights=outcome_weights, k=1)[0]

        # Generate scores correlated with outcome
        if outcome == "accepted":
            base = random.uniform(3.8, 5.0)
        elif outcome == "revision":
            base = random.uniform(3.0, 4.2)
        elif outcome == "rejected":
            base = random.uniform(2.0, 3.5)
        else:  # desk_reject
            base = random.uniform(1.0, 2.5)

        scores = {
            "innovation": round(min(5.0, max(0.0, base + random.uniform(-0.5, 0.5))), 2),
            "method": round(min(5.0, max(0.0, base + random.uniform(-0.5, 0.5))), 2),
            "evidence": round(min(5.0, max(0.0, base + random.uniform(-0.5, 0.5))), 2),
            "application": round(min(5.0, max(0.0, base + random.uniform(-0.5, 0.5))), 2),
            "cudos": round(min(5.0, max(3.0, base + random.uniform(-0.3, 0.5))), 2),
        }

        # Generate feedback based on low-scoring dimensions
        feedback_parts = []
        for dim, score in scores.items():
            if score < 3.5 and dim in feedback_templates:
                feedback_parts.append(random.choice(feedback_templates[dim]))
        feedback = " ".join(feedback_parts) if feedback_parts else "No major concerns."

        # Generate keywords
        topic_keywords = {
            "computer_science": ["deep learning", "transformer", "neural network", "NLP", "RL"],
            "medicine": ["clinical trial", "RCT", "meta-analysis", "treatment", "diagnosis"],
            "biology": ["CRISPR", "gene editing", "protein", "cell", "genomics"],
            "physics": ["quantum", "particle", "simulation", "modeling", "experiment"],
            "social_science": ["survey", "behavioral", "policy", "economics", "education"],
        }

        discipline = random.choice(disciplines)
        keywords = random.sample(
            topic_keywords.get(discipline, ["research"]),
            k=random.randint(2, 4)
        )

        # Some keywords trend in recent years
        if year >= 2023:
            trending = {
                "computer_science": ["LLM", "RLHF", "diffusion model"],
                "medicine": ["AI diagnostics", "digital health"],
                "biology": ["single-cell", "spatial transcriptomics"],
            }
            if discipline in trending:
                keywords.extend(random.sample(trending[discipline], k=min(2, len(trending[discipline]))))

        history.append({
            "paper_id": f"paper_{i:04d}",
            "title": f"Research Paper {i} in {discipline.replace('_', ' ').title()}",
            "discipline": discipline,
            "scores": scores,
            "target_journal": random.choice(journals),
            "outcome": outcome,
            "editor_feedback": feedback,
            "keywords": keywords,
            "year": year,
            "citations": random.randint(0, 200) if outcome == "accepted" else random.randint(0, 20),
        })

    return history


async def demo_frontier_discovery():
    """Main demo: Run frontier discovery analysis."""
    print("=" * 60)
    print("Frontier Discovery Analysis Demo")
    print("=" * 60)

    from wuya_agents.subagents.frontier import (
        FrontierDiscoverySubAgent,
        SubmissionRecord,
    )

    # Generate sample history
    raw_history = generate_sample_history(n=60, seed=42)
    history = [SubmissionRecord(**h) for h in raw_history]

    print(f"\nGenerated {len(history)} submission records")
    print(f"  Disciplines: {set(r.discipline for r in history)}")
    print(f"  Years: {sorted(set(r.year for r in history))}")
    print(f"  Outcomes: {dict(sorted({o: sum(1 for r in history if r.outcome == o) for o in set(r.outcome for r in history)}.items()))}")

    # Create agent and run analysis
    agent = FrontierDiscoverySubAgent(
        min_history_size=10,
        trend_window_years=2,
    )

    update = await agent.discover_frontier(history)

    # Display results
    print(f"\n{'=' * 60}")
    print(f"Frontier Discovery Results")
    print(f"{'=' * 60}")
    print(f"  New DMUs analyzed: {update.new_dmu_count}")
    print(f"  Frontier shift: {update.frontier_shift:+.4f}")

    # Improvement patterns
    print(f"\n--- Improvement Patterns (top 5) ---")
    for pattern in update.improvement_patterns[:5]:
        print(f"  [{pattern.severity:.1f}/5.0] {pattern.dimension}: "
              f"{pattern.pattern_description} (freq={pattern.frequency})")

    # Trend clusters
    print(f"\n--- Emerging Trends ---")
    for trend in update.trend_clusters[:5]:
        print(f"  [{trend.growth_rate:+.1%}] {', '.join(trend.keywords[:3])} "
              f"(freq={trend.frequency})")

    # Dimension updates
    print(f"\n--- Dimension Frontier Updates ---")
    for dim, data in update.dimension_updates.items():
        direction_icon = {"increasing": "↑", "decreasing": "↓", "stable": "→"}
        icon = direction_icon.get(data["direction"], "?")
        print(f"  {dim:12s}: {icon} {data['shift']:+.3f} "
              f"(older={data['older_average']:.2f}, recent={data['recent_average']:.2f})")

    # Recommendations
    print(f"\n--- Recommendations ---")
    for rec in update.recommendations:
        print(f"  • {rec}")

    # Self-improvement signals
    print(f"\n--- Self-Improvement Signals ---")
    signals = update.self_improvement_signals
    if signals.get("dimension_weight_adjustments"):
        print("  Dimension weight adjustments:")
        for dim, adj in signals["dimension_weight_adjustments"].items():
            print(f"    {dim}: {adj['current_weight']:.2f} → {adj['suggested_weight']:.2f} "
                  f"({adj['reason']})")

    if signals.get("rag_priority_updates"):
        print("  RAG priority updates:")
        for update_msg in signals["rag_priority_updates"]:
            print(f"    • {update_msg}")

    if signals.get("evaluation_criteria_updates"):
        print("  Evaluation criteria updates:")
        for crit in signals["evaluation_criteria_updates"]:
            print(f"    • {crit}")

    return update


async def demo_minimal_history():
    """Demo: Handling insufficient history."""
    print("\n" + "=" * 60)
    print("Demo: Insufficient History Handling")
    print("=" * 60)

    from wuya_agents.subagents.frontier import (
        FrontierDiscoverySubAgent,
        SubmissionRecord,
    )

    agent = FrontierDiscoverySubAgent(min_history_size=10)

    # Only 3 records
    small_history = [
        SubmissionRecord(
            paper_id="p1", title="Paper 1", discipline="cs",
            scores={"innovation": 4.0, "method": 3.5, "evidence": 3.0, "application": 4.0, "cudos": 4.5},
            target_journal="Nature", outcome="revision",
            editor_feedback="Methodology needs improvement.", keywords=["AI"], year=2024,
        ),
        SubmissionRecord(
            paper_id="p2", title="Paper 2", discipline="cs",
            scores={"innovation": 3.5, "method": 4.0, "evidence": 3.5, "application": 3.0, "cudos": 4.0},
            target_journal="Science", outcome="rejected",
            editor_feedback="Insufficient evidence for claims.", keywords=["ML"], year=2024,
        ),
        SubmissionRecord(
            paper_id="p3", title="Paper 3", discipline="bio",
            scores={"innovation": 4.5, "method": 4.0, "evidence": 4.0, "application": 3.5, "cudos": 4.5},
            target_journal="Cell", outcome="accepted",
            editor_feedback="Good work.", keywords=["CRISPR"], year=2023,
        ),
    ]

    update = await agent.discover_frontier(small_history)
    print(f"\nWith only {len(small_history)} records:")
    print(f"  Recommendations: {update.recommendations}")


async def demo_dea_integration():
    """Demo: DEA frontier update integration."""
    print("\n" + "=" * 60)
    print("Demo: DEA Frontier Integration")
    print("=" * 60)

    from wuya_agents.subagents.frontier import (
        FrontierDiscoverySubAgent,
        SubmissionRecord,
    )
    from wuya_agents.dea_subagent import DEAEngine

    # Create with DEA engine
    dea_engine = DEAEngine(min_reference_papers=10)
    agent = FrontierDiscoverySubAgent(
        dea_engine=dea_engine,
        min_history_size=10,
    )

    # Generate history
    raw_history = generate_sample_history(n=30, seed=42)
    history = [SubmissionRecord(**h) for h in raw_history]

    # Update DEA frontier
    result = await agent.update_dea_frontier(history)
    print(f"\nDEA frontier update status: {result['status']}")
    print(f"  New DMUs: {result['new_dmu_count']}")
    print(f"  Note: {result['note']}")


async def main():
    """Run all frontier discovery demos."""
    print("\n" + "=" * 60)
    print("WuYa Frontier Discovery - Examples")
    print("=" * 60)

    await demo_frontier_discovery()
    await demo_minimal_history()
    await demo_dea_integration()

    print("\n" + "=" * 60)
    print("All frontier discovery demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
