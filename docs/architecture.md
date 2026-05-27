# WuYa Architecture Documentation

## Overview

WuYa is a theory-driven multi-agent system that evaluates academic papers using classical philosophical principles combined with modern AI techniques.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        WuYa System                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │   Input     │  ParsedPaper                                    │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    TwoPhaseRouter                          │ │
│  │  ┌─────────────────────────────────────────────────────┐    │ │
│  │  │              Phase 1: CUDOS Gate                   │    │ │
│  │  │  ┌─────────────────────────────────────────────┐    │    │ │
│  │  │  │         CUDOSSubAgent                      │    │    │ │
│  │  │  │   ┌──────────────────────────────────┐    │    │    │ │
│  │  │  │   │ Communalism | Universalism      │    │    │    │ │
│  │  │  │   │ Disinterestedness               │    │    │    │ │
│  │  │  │   │ Organized Skepticism            │    │    │    │ │
│  │  │  │   └──────────────────────────────────┘    │    │    │ │
│  │  │  └─────────────────────────────────────────────┘    │    │ │
│  │  └─────────────────────────────────────────────────────┘    │ │
│  │                            │                               │ │
│  │         ┌─────────────────┴─────────────────┐              │ │
│  │         ▼                                   ▼              │ │
│  │  ┌─────────────┐                   ┌─────────────────┐    │ │
│  │  │  VETOED     │                   │  Phase 2:       │    │ │
│  │  │  Return     │                   │  Parallel Eval  │    │ │
│  │  │  Report     │                   │                 │    │ │
│  │  └─────────────┘                   │  ┌───────────┐  │    │ │
│  │                                    │  │Innovation │  │    │ │
│  │                                    │  │  Agent   │  │    │ │
│  │                                    │  └───────────┘  │    │ │
│  │                                    │  ┌───────────┐  │    │ │
│  │                                    │  │  Method  │  │    │ │
│  │                                    │  │  Agent   │  │    │ │
│  │                                    │  └───────────┘  │    │ │
│  │                                    │  ┌───────────┐  │    │ │
│  │                                    │  │ Evidence │  │    │ │
│  │                                    │  │  Agent   │  │    │ │
│  │                                    │  └───────────┘  │    │ │
│  │                                    │  ┌───────────┐  │    │ │
│  │                                    │  │Application│  │    │ │
│  │                                    │  │  Agent   │  │    │ │
│  │                                    │  └───────────┘  │    │ │
│  │                                    └─────────────────┘    │ │
│  │                                          │                │ │
│  │                    ┌─────────────────────┴───────────┐    │ │
│  │                    ▼                                 ▼    │ │
│  │          ┌─────────────────┐              ┌────────────────┐│ │
│  │          │  DEA SubAgent  │              │    RAG         ││ │
│  │          │  (Optional)    │              │    Client      ││ │
│  │          └─────────────────┘              └────────────────┘│ │
│  └─────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   EvaluationReport                         │ │
│  │  - Dimension Scores  - DEA Summary  - Journal Matches     │ │
│  │  - Tier Estimate    - RAG Citations - Suggestions         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### TwoPhaseRouter

The main orchestrator that coordinates the evaluation workflow.

**Responsibilities:**
1. Execute Phase 1 (CUDOS gatekeeping)
2. If passed, execute Phase 2 (parallel evaluation)
3. Optionally trigger DEA analysis
4. Aggregate results into EvaluationReport

**Key Features:**
- Parallel sub-agent execution (configurable)
- RAG integration for enhanced evaluation
- Error handling and recovery
- Comprehensive reporting

### CUDOSSubAgent

Implements Merton's CUDOS scientific norms for ethical evaluation.

**Dimensions:**
1. **Communalism**: Open sharing of knowledge
2. **Universalism**: Universal evaluation standards
3. **Disinterestedness**: Absence of personal interest
4. **Organized Skepticism**: Critical scrutiny

**Key Features:**
- Veto mechanism for papers with ethical concerns
- RAG-triggered evaluation for suspicious papers
- Fail-open error handling

### Evaluation Sub-agents

Four parallel agents evaluating different aspects:

#### InnovationSubAgent
- Novelty of contribution
- Significance to the field
- Advancement beyond current state

#### MethodSubAgent
- Methodological rigor
- Validity of approach
- Reproducibility

#### EvidenceSubAgent
- Strength of evidence
- Consistency with existing knowledge
- Sufficiency for claims

#### ApplicationSubAgent
- Relevance to practical problems
- Feasibility of implementation
- Potential impact

### RAG Components

**Hybrid Retrieval Strategy:**
- Semantic search (embedding-based)
- Keyword matching
- Configurable weighting (alpha parameter)

**Key Components:**
1. `EmbeddingProvider`: Generates text embeddings
2. `VectorStore`: Stores and检索s vectors
3. `RAGClient`: Orchestrates retrieval

### DEA Engine

Data Envelopment Analysis for quantitative efficiency measurement.

**Features:**
1. Super-efficiency calculation
2. Bootstrap confidence intervals
3. Frontier identification
4. Self-improvement signal generation

### FrontierDiscoverySubAgent

Self-learning component that improves evaluation over time.

**Responsibilities:**
1. Collect submission history
2. Identify improvement patterns
3. Detect emerging trends
4. Generate self-improvement signals

## Data Flow

### Standard Evaluation Flow

```
1. Input: ParsedPaper
       │
2. CUDOS Gate ─────────────────┐
   │                             │
   │ Pass?                       │ Fail?
   ▼                             ▼
3. Parallel Evaluation      4. Return VetoedReport
   │
4. DEA Analysis (optional) ─────────────┐
   │                                   │
5. Result Aggregation                   │
   │                                   │
6. Output: EvaluationReport ◄──────────┘
```

### RAG-Enhanced Flow

```
Sub-agent Evaluation
       │
       ▼
Check RAG Trigger?
       │
       ├── Yes ──► Query RAG Client ──► Retrieve Citations
       │                                    │
       │◄───────────────────────────────────┘
       │
       ▼
Include Citations in Result
```

## Self-Improvement Loop

```
┌────────────────────────────────────────────────────────────┐
│                    Self-Improvement Loop                   │
├────────────────────────────────────────────────────────────┤
│                                                             │
│   Submission History ──► FrontierDiscoveryAgent             │
│         │                        │                         │
│         │                        ▼                         │
│         │              ┌──────────────────┐                │
│         │              │ Pattern Analysis │                │
│         │              │ Trend Detection  │                │
│         │              │ Frontier Mapping │                │
│         │              └────────┬─────────┘                │
│         │                       │                          │
│         │                       ▼                          │
│         │              ┌──────────────────┐                │
│         │              │   DEA Engine     │                │
│         │              │   Update         │                │
│         │              └────────┬─────────┘                │
│         │                       │                          │
│         │                       ▼                          │
│         │              ┌──────────────────┐                │
│         │              │ Self-Improvement │                │
│         │              │ Signals          │                │
│         │              └────────┬─────────┘                │
│         │                       │                          │
│         │◄──────────────────────┘                          │
│         │                                                    │
│         ▼                                                    │
│   Update Evaluation Parameters ◄────────────────────────────┘
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## Design Patterns

### 1. Strategy Pattern

Different sub-agents implement the same interface for interchangeable evaluation strategies.

### 2. Chain of Responsibility

CUDOS gate either passes or fails; failed papers don't proceed to evaluation.

### 3. Observer Pattern

Sub-agents observe paper content and trigger RAG when needed.

### 4. Factory Pattern

Agent creation is centralized in the router for consistent configuration.

### 5. Template Method

DEA analysis follows a fixed template: transform → solve → bootstrap → interpret.

## Error Handling

### Error Recovery Strategies

1. **Fail-Open for CUDOS**: Ethical concerns don't block evaluation
2. **Graceful Degradation**: Missing components use defaults
3. **Retry Logic**: Transient failures are retried
4. **Error Aggregation**: All errors are collected in the report

### Error Reporting

```python
report = await router.route(paper)
if report.status == "completed":
    if report.errors:
        for error in report.errors:
            print(f"Warning: {error}")
```

## Performance Considerations

### Parallel Execution

- Sub-agents execute in parallel when `enable_parallel=True`
- RAG queries can be batched
- DEA bootstrap iterations are parallelizable

### Caching

- Embeddings are cached by content hash
- Vector store supports efficient retrieval
- LLM responses can be cached (if configured)

### Resource Limits

- Default timeout: 60 seconds per evaluation
- Max RAG results: 10 per query
- Min reference papers for DEA: 50

## Extension Points

### Adding New Sub-agents

1. Inherit from `BaseSubAgent`
2. Implement `evaluate()` method
3. Register in `TwoPhaseRouter`
4. Add tests in `tests/test_subagents/`

### Adding New RAG Providers

1. Implement `EmbeddingProvider` interface
2. Implement `VectorStore` interface
3. Configure in `RAGClient`

### Custom DEA Models

1. Extend `DEAEngine` class
2. Override `transform_scores()` and `solve()`
3. Configure in `DEASubAgent`

## Security Considerations

1. **Input Validation**: All paper fields are validated
2. **RAG Source Verification**: Citations include source metadata
3. **Error Message Sanitization**: No internal details exposed
4. **Rate Limiting**: LLM calls are rate-limited

## Monitoring

### Metrics Collected

- Evaluation latency per sub-agent
- RAG retrieval accuracy
- DEA efficiency scores distribution
- CUDOS veto rate

### Logging

- DEBUG: Sub-agent reasoning traces
- INFO: Evaluation workflow steps
- WARNING: RAG misses, DEA insufficient data
- ERROR: Agent failures

## Testing Strategy

See [Testing Documentation](./README.md#testing) for comprehensive testing guidelines.

### Test Levels

1. **Unit Tests**: Individual sub-agents
2. **Integration Tests**: Component interactions
3. **E2E Tests**: Complete workflows
4. **Performance Tests**: Latency and throughput

---

## License

MIT License - See [LICENSE](../LICENSE)
