# WuYa API Documentation

## Overview

WuYa is a multi-agent system for academic paper evaluation and journal recommendation. This document provides detailed API documentation for all public interfaces.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Sub-agents](#sub-agents)
3. [RAG Components](#rag-components)
4. [DEA Engine](#dea-engine)
5. [Router](#router)
6. [Data Models](#data-models)

---

## Core Classes

### ParsedPaper

Represents a parsed academic paper with all metadata.

```python
from wuya_agents.base import ParsedPaper

paper = ParsedPaper(
    paper_id="paper_001",
    title="Your Paper Title",
    abstract="Paper abstract...",
    content="Full paper content...",
    authors=["Author 1", "Author 2"],
    keywords=["keyword1", "keyword2"],
    discipline="computer science",
    citations=["ref1", "ref2"],
    references=["ref1", "ref2"],
    figures=["fig1.png"],
    tables=["table1.csv"],
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `paper_id` | str | Unique identifier |
| `title` | str | Paper title |
| `abstract` | str | Paper abstract |
| `content` | str | Full paper content |
| `authors` | List[str] | List of author names |
| `keywords` | List[str] | Paper keywords |
| `discipline` | str | Academic discipline |
| `citations` | List[str] | Referenced paper IDs |
| `references` | List[str] | Reference IDs |

---

### AgentStatus

Enumeration for agent execution status.

```python
from wuya_agents.base import AgentStatus

status = AgentStatus.SUCCESS
```

**Values:**
- `SUCCESS`: Agent completed successfully
- `FAILED`: Agent failed
- `RAG_TRIGGERED`: Agent triggered RAG retrieval

---

### EvaluationDimension

Enumeration for evaluation dimensions.

```python
from wuya_agents.base import EvaluationDimension

dim = EvaluationDimension.INNOVATION
```

**Values:**
- `INNOVATION`: Novelty and significance
- `METHOD`: Methodological rigor
- `EVIDENCE`: Evidence quality
- `APPLICATION`: Practical applicability

---

### CUDOSResult

Result of CUDOS gate evaluation.

```python
from wuya_agents.base import CUDOSResult

result = CUDOSResult(
    gate_pass=True,
    dimensions={...},
    veto_reason=None,
    rag_citations=[...],
    processing_time_ms=150,
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `gate_pass` | bool | Whether paper passes CUDOS |
| `dimensions` | Dict | CUDOS dimension scores |
| `veto_reason` | Optional[str] | Reason for veto (if any) |
| `rag_citations` | List[Dict] | RAG-retrieved citations |

---

### DimensionResult

Result of single dimension evaluation.

```python
from wuya_agents.base import DimensionResult, SubDimensionScore, AgentStatus

result = DimensionResult(
    dimension=EvaluationDimension.INNOVATION,
    overall_score=4.5,
    sub_dimensions=[
        SubDimensionScore(
            name="novelty",
            score=4.5,
            justification="Highly novel approach",
            weight=0.4
        ),
    ],
    narrative="The paper demonstrates...",
    status=AgentStatus.SUCCESS,
    processing_time_ms=100,
)
```

---

### EvaluationReport

Final evaluation report from the router.

```python
from wuya_agents.router import EvaluationReport

report = EvaluationReport(
    paper_id="paper_001",
    paper_title="Your Paper",
    status="completed",
    cudos_passed=True,
    dimension_scores={"innovation": 4.5, ...},
    overall_score=4.2,
    tier_estimate="Q1-A",
    dea_summary=None,
    rag_citations=[...],
    journal_matches=[...],
    improvement_suggestions=[...],
    processing_time_ms=500,
    timestamp="2024-01-01T12:00:00Z",
)
```

---

## Sub-agents

### CUDOSSubAgent

Evaluates papers against Merton's CUDOS scientific norms.

```python
from wuya_agents.subagents import CUDOSSubAgent
from wuya_agents.base import ParsedPaper, AgentStatus

agent = CUDOSSubAgent(
    llm_client=llm_client,        # Optional
    rag_client=rag_client,        # Optional
    veto_threshold=2.0,           # Default: 2.0
    rag_trigger_keywords=[...],   # Custom keywords
)
await agent.initialize()

# Evaluate paper
result = await agent.evaluate(paper)
```

**Methods:**
- `evaluate(paper, context=None)`: Evaluate paper against CUDOS
- `should_trigger_rag(paper)`: Check if RAG should be triggered
- `get_dimension_scores(result)`: Extract scores
- `get_failed_dimensions(result)`: Get dimensions below threshold
- `get_veto_summary(result)`: Generate veto summary

---

### InnovationSubAgent

Evaluates paper novelty and significance.

```python
from wuya_agents.subagents import InnovationSubAgent

agent = InnovationSubAgent(
    llm_client=llm_client,
    rag_client=rag_client,
)
await agent.initialize()

result = await agent.evaluate(paper)
```

**Sub-dimensions:** novelty, significance, advancement

---

### MethodSubAgent

Evaluates methodological rigor.

```python
from wuya_agents.subagents import MethodSubAgent

agent = MethodSubAgent(
    llm_client=llm_client,
    rag_client=rag_client,
    rag_threshold=3.0,  # Score below which RAG is triggered
)
await agent.initialize()

result = await agent.evaluate(paper)
```

**Sub-dimensions:** rigor, validity, reproducibility

---

### EvidenceSubAgent

Evaluates evidence quality.

```python
from wuya_agents.subagents import EvidenceSubAgent

agent = EvidenceSubAgent(
    llm_client=llm_client,
    rag_client=rag_client,
    rag_threshold=3.0,
)
await agent.initialize()

result = await agent.evaluate(paper)
```

**Sub-dimensions:** strength, consistency, sufficiency

---

### ApplicationSubAgent

Evaluates practical applicability.

```python
from wuya_agents.subagents import ApplicationSubAgent

agent = ApplicationSubAgent(
    llm_client=llm_client,
    rag_client=rag_client,
)
await agent.initialize()

result = await agent.evaluate(paper)

# Get application scenarios
scenarios = agent.get_application_scenarios()
top = agent.get_top_scenario()
```

**Sub-dimensions:** relevance, feasibility, impact

---

### FrontierDiscoverySubAgent

Discovers evaluation frontier and improvement patterns.

```python
from wuya_agents.subagents.frontier import FrontierDiscoverySubAgent, SubmissionRecord

agent = FrontierDiscoverySubAgent(
    min_history_size=10,       # Minimum submission history
    trend_window_years=3,     # Years to consider for trends
)
await agent.initialize()

# Discover frontier from submission history
history = [
    SubmissionRecord(
        paper_id="...",
        title="...",
        discipline="cs",
        scores={"innovation": 4.0, ...},
        target_journal="Nature",
        outcome="accepted",
        editor_feedback="...",
        keywords=[...],
        year=2024,
        citations=50,
    ),
    ...
]

result = await agent.discover_frontier(history)
```

**Output:** `FrontierUpdate` containing:
- `new_dmu_count`: Number of papers processed
- `improvement_patterns`: Detected patterns
- `trend_clusters`: Identified trends
- `frontier_shift`: Year-over-year change
- `dimension_updates`: Per-dimension analysis
- `recommendations`: Suggestions

---

## RAG Components

### RAGClientImpl

Retrieval-Augmented Generation client.

```python
from wuya_agents.rag.client import RAGClientImpl
from wuya_agents.rag.embedding import MockEmbeddingProvider
from wuya_agents.rag.vector_store import InMemoryVectorStore

client = RAGClientImpl(
    embedding_provider=MockEmbeddingProvider(dimensions=128),
    vector_store=InMemoryVectorStore(),
    enable_hybrid=True,     # Enable hybrid search
    hybrid_alpha=0.7,       # Weight for semantic (1-alpha for keyword)
)

await client.initialize()

# Add documents
await client.add_texts([
    ("doc1", "Content about machine learning", {"topic": "ml"}),
    ("doc2", "Deep learning content", {"topic": "dl"}),
])

# Retrieve
results = await client.retrieve(
    query="neural networks",
    context="Additional context",
    top_k=5,
    filters={"topic": "ml"},
)

# Results format:
# [
#     {
#         "content": "...",
#         "score": 0.85,
#         "source": "doc1",
#         "metadata": {...},
#         "citation": "Author (2024)",
#     },
#     ...
# ]

# Delete documents
await client.delete(["doc1"])

# Update documents
await client.update("doc1", content="New content")
```

---

### MockEmbeddingProvider

Mock embedding provider for testing.

```python
from wuya_agents.rag.embedding import MockEmbeddingProvider

provider = MockEmbeddingProvider(dimensions=128)
embedding = await provider.embed("test text")
embeddings = await provider.embed_batch(["text1", "text2"])
```

---

### InMemoryVectorStore

In-memory vector storage.

```python
from wuya_agents.rag.vector_store import InMemoryVectorStore, Document
import numpy as np

store = InMemoryVectorStore()

doc = Document(
    id="doc1",
    content="Content text",
    embedding=np.random.rand(128).tolist(),
    metadata={"source": "test"},
)

await store.add_documents([doc])
results = await store.search(query_embedding, top_k=5)
await store.delete(["doc1"])
```

---

## DEA Engine

### DEAEngine

Data Envelopment Analysis engine.

```python
from wuya_agents.dea_subagent import DEAEngine, ScoreVector

engine = DEAEngine(
    min_reference_papers=50,    # Minimum refs for valid DEA
    bootstrap_iterations=100,  # Bootstrap samples
    confidence_level=0.95,     # CI confidence level
)

# Analyze a paper
score_vector = ScoreVector(
    innovation=4.5,
    method=4.2,
    evidence=3.5,
    application=4.0,
    cudos=4.8,
)

ref_scores = [ScoreVector(**paper["scores"]) for paper in reference_papers]
result = engine.analyze(score_vector, ref_scores)
```

**Result Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `efficiency_score` | float | Super-efficiency score |
| `confidence_interval` | Tuple | (lower, upper) bounds |
| `status` | DEAStatus | SUCCESS or INSUFFICIENT_DATA |
| `is_on_frontier` | bool | Whether paper is on frontier |
| `reference_set_size` | int | Number of reference papers |
| `bootstrap_std` | float | Bootstrap standard deviation |

---

### DEASubAgent

DEA sub-agent for paper evaluation.

```python
from wuya_agents.dea_subagent import DEASubAgent, DEAEngine

agent = DEASubAgent(
    dea_engine=DEAEngine(),
    llm_client=llm_client,
    rag_client=rag_client,
)

result = await agent.evaluate(
    paper_id="paper_001",
    score_vector=ScoreVector(...),
    target_journal="Nature",
    reference_papers=[...],
    path_a_estimate="Q1-A",  # Optional
)
```

---

## Router

### TwoPhaseRouter

Main orchestration router for paper evaluation.

```python
from wuya_agents.router import TwoPhaseRouter
from wuya_agents.subagents import (
    CUDOSSubAgent, InnovationSubAgent, MethodSubAgent,
    EvidenceSubAgent, ApplicationSubAgent
)
from wuya_agents.dea_subagent import DEASubAgent, DEAEngine

router = TwoPhaseRouter(
    cudos_agent=CUDOSSubAgent(...),
    innovation_agent=InnovationSubAgent(...),
    method_agent=MethodSubAgent(...),
    evidence_agent=EvidenceSubAgent(...),
    application_agent=ApplicationSubAgent(...),
    dea_agent=DEASubAgent(DEAEngine()),  # Optional
    rag_client=rag_client,               # Optional
    enable_parallel=True,                # Parallel sub-agent execution
)

# Run evaluation
report = await router.route(
    paper,
    target_journal="Nature",           # Optional
    reference_papers=[...],             # Optional (for DEA)
)

print(f"Overall: {report.overall_score}")
print(f"Tier: {report.tier_estimate}")
print(f"Status: {report.status}")
```

**Workflow:**
1. Phase 1: CUDOS gatekeeping
2. Phase 2: Parallel evaluation (if CUDOS passes)
3. Optional: DEA analysis (if references provided)

---

## Data Models

### ScoreVector

Score vector for DEA analysis.

```python
from wuya_agents.dea_subagent import ScoreVector

vector = ScoreVector(
    innovation=4.5,
    method=4.2,
    evidence=3.5,
    application=4.0,
    cudos=4.8,
)
```

---

### SubmissionRecord

Submission history record.

```python
from wuya_agents.subagents.frontier import SubmissionRecord

record = SubmissionRecord(
    paper_id="...",
    title="...",
    discipline="cs",
    scores={"innovation": 4.0, ...},
    target_journal="Nature",
    outcome="accepted",  # or "rejected"
    editor_feedback="...",
    keywords=[...],
    year=2024,
    citations=50,
)
```

---

### JournalMatch

Journal matching result.

```python
from wuya_agents.router import JournalMatch

match = JournalMatch(
    journal_name="Nature",
    match_score=0.85,
    tier="Q1",
    impact_factor=64.0,
    relevance=0.90,
    reach=0.75,
)
```

---

## Error Handling

All async methods may raise exceptions. Handle errors gracefully:

```python
try:
    result = await agent.evaluate(paper)
except Exception as e:
    print(f"Evaluation failed: {e}")
    # Fallback behavior
```

---

## Testing

See [Testing Guide](./README.md#testing) for detailed testing documentation.

### Example Test

```python
import pytest
from wuya_agents.subagents import CUDOSSubAgent
from wuya_agents.base import ParsedPaper

@pytest.mark.asyncio
async def test_cudos_passes_valid_paper():
    agent = CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag)
    
    paper = ParsedPaper(
        paper_id="test",
        title="Test",
        abstract="...",
        content="...",
        authors=["Author"],
        keywords=["test"],
        discipline="cs",
    )
    
    result = await agent.evaluate(paper)
    assert result.gate_pass is True
```

---

## License

MIT License - See [LICENSE](../LICENSE)
