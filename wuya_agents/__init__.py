"""
WuYa Multi-Agent System

A theory-driven multi-agent system for academic paper evaluation and journal recommendation.

Modules:
    - base: Base classes and interfaces for all sub-agents
    - subagents: Evaluation sub-agents (CUDOS, Innovation, Method, Evidence, Application, Frontier)
    - dea_subagent: DEA (Data Envelopment Analysis) Sub-agent for paper localization
    - rag: RAG (Retrieval-Augmented Generation) components
    - llm_client: Unified LLM client supporting OpenAI and Anthropic providers
    - parser: Paper parsing module
    - router: Two-phase routing orchestration
    - aggregator: Result aggregation and report generation

Author: WuYa Team
"""

__version__ = "0.1.0"
__author__ = "WuYa Team"

# Base classes
from .base import (
    BaseSubAgent,
    CUDOSSubAgent as BaseCUDOSSubAgent,
    CUDOSResult,
    DimensionResult,
    EvaluationDimension,
    AgentStatus,
    ParsedPaper,
    ScoreVector,
    SubDimensionScore,
    Router,
    RAGClient,
    LLMClient as LLMClientProtocol,
)

# Sub-agents
from .subagents import (
    # CUDOS
    CUDOSSubAgent,
    CUDOSDimensionScore,
    CUDOS_DIMENSIONS,
    COMMUNALISM,
    UNIVERSALISM,
    DISINTERESTEDNESS,
    ORGANIZED_SKEPTICISM,
    CUDOS_DIMENSION_DESCRIPTIONS,
    create_cudos_subagent,
    # Innovation
    InnovationSubAgent,
    InnovationDimensionScore,
    INNOVATION_DIMENSIONS,
    NOVELTY,
    SIGNIFICANCE,
    ADVANCEMENT,
    INNOVATION_DIMENSION_DESCRIPTIONS,
    create_innovation_subagent,
    # Method
    MethodSubAgent,
    MethodDimensionScore,
    METHOD_DIMENSIONS,
    RIGOR,
    VALIDITY,
    REPRODUCIBILITY,
    METHOD_DIMENSION_DESCRIPTIONS,
    create_method_subagent,
    # Evidence
    EvidenceSubAgent,
    EvidenceDimensionScore,
    EVIDENCE_DIMENSIONS,
    STRENGTH,
    CONSISTENCY,
    SUFFICIENCY,
    EVIDENCE_DIMENSION_DESCRIPTIONS,
    create_evidence_subagent,
    # Application
    ApplicationSubAgent,
    ApplicationDimensionScore,
    ApplicationScenario,
    APPLICATION_DIMENSIONS,
    RELEVANCE,
    FEASIBILITY,
    IMPACT,
    APPLICATION_DIMENSION_DESCRIPTIONS,
    create_application_subagent,
    # Frontier Discovery
    FrontierDiscoverySubAgent,
    SubmissionRecord,
    ImprovementPattern,
    TrendCluster,
    FrontierUpdate,
    create_frontier_discovery_agent,
)

# DEA Sub-agent
from .dea_subagent import (
    DEAEngine,
    DEASubAgent,
    DEASubAgentService,
    DEAResult,
    DEAStatus,
)

# LLM Client
from .llm_client import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    create_llm_client,
)

# Paper Parser
from .parser import (
    PaperParser,
    PDFExtractor,
    PyPDF2Extractor,
    MockPDFExtractor,
    create_paper_parser,
)

# Router
from .router import (
    TwoPhaseRouter,
    EvaluationReport,
    DimensionSummary,
    DEASummary,
    JournalMatch,
    ImprovementSuggestion,
    create_two_phase_router,
)

# Aggregator
from .aggregator import (
    ResultAggregator,
    AggregatedDimension,
    AggregatedDEA,
    JournalRecommendation,
    ImprovementItem,
    EvaluationSummary,
    EvaluationReport as AggregatedEvaluationReport,
    create_result_aggregator,
)

# RAG Components
from .rag import (
    EmbeddingProvider,
    EmbeddingCache,
    EmbeddingResult,
    OpenAIEmbeddingProvider,
    MockEmbeddingProvider,
    create_embedding_provider,
    VectorStore,
    Document,
    SearchResult,
    InMemoryVectorStore,
    ChromaDBVectorStore,
    create_vector_store,
    RAGClientImpl,
    RetrievalResult,
    HybridRetriever,
    create_rag_client,
)

__all__ = [
    # Base classes
    "BaseSubAgent",
    "BaseCUDOSSubAgent",
    "CUDOSResult",
    "DimensionResult",
    "EvaluationDimension",
    "AgentStatus",
    "ParsedPaper",
    "ScoreVector",
    "SubDimensionScore",
    "Router",
    "RAGClient",
    "LLMClientProtocol",
    # CUDOS Sub-agent
    "CUDOSSubAgent",
    "CUDOSDimensionScore",
    "CUDOS_DIMENSIONS",
    "COMMUNALISM",
    "UNIVERSALISM",
    "DISINTERESTEDNESS",
    "ORGANIZED_SKEPTICISM",
    "CUDOS_DIMENSION_DESCRIPTIONS",
    "create_cudos_subagent",
    # Innovation Sub-agent
    "InnovationSubAgent",
    "InnovationDimensionScore",
    "INNOVATION_DIMENSIONS",
    "NOVELTY",
    "SIGNIFICANCE",
    "ADVANCEMENT",
    "INNOVATION_DIMENSION_DESCRIPTIONS",
    "create_innovation_subagent",
    # Method Sub-agent
    "MethodSubAgent",
    "MethodDimensionScore",
    "METHOD_DIMENSIONS",
    "RIGOR",
    "VALIDITY",
    "REPRODUCIBILITY",
    "METHOD_DIMENSION_DESCRIPTIONS",
    "create_method_subagent",
    # Evidence Sub-agent
    "EvidenceSubAgent",
    "EvidenceDimensionScore",
    "EVIDENCE_DIMENSIONS",
    "STRENGTH",
    "CONSISTENCY",
    "SUFFICIENCY",
    "EVIDENCE_DIMENSION_DESCRIPTIONS",
    "create_evidence_subagent",
    # Application Sub-agent
    "ApplicationSubAgent",
    "ApplicationDimensionScore",
    "ApplicationScenario",
    "APPLICATION_DIMENSIONS",
    "RELEVANCE",
    "FEASIBILITY",
    "IMPACT",
    "APPLICATION_DIMENSION_DESCRIPTIONS",
    "create_application_subagent",
    # Frontier Discovery Sub-agent
    "FrontierDiscoverySubAgent",
    "SubmissionRecord",
    "ImprovementPattern",
    "TrendCluster",
    "FrontierUpdate",
    "create_frontier_discovery_agent",
    # DEA Sub-agent
    "DEAEngine",
    "DEASubAgent",
    "DEASubAgentService",
    "DEAResult",
    "DEAStatus",
    # LLM Client
    "LLMClient",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "create_llm_client",
    # Paper Parser
    "PaperParser",
    "PDFExtractor",
    "PyPDF2Extractor",
    "MockPDFExtractor",
    "create_paper_parser",
    # Router
    "TwoPhaseRouter",
    "EvaluationReport",
    "DimensionSummary",
    "DEASummary",
    "JournalMatch",
    "ImprovementSuggestion",
    "create_two_phase_router",
    # Aggregator
    "ResultAggregator",
    "AggregatedDimension",
    "AggregatedDEA",
    "JournalRecommendation",
    "ImprovementItem",
    "EvaluationSummary",
    "AggregatedEvaluationReport",
    "create_result_aggregator",
    # RAG Components
    "EmbeddingProvider",
    "EmbeddingCache",
    "EmbeddingResult",
    "OpenAIEmbeddingProvider",
    "MockEmbeddingProvider",
    "create_embedding_provider",
    "VectorStore",
    "Document",
    "SearchResult",
    "InMemoryVectorStore",
    "ChromaDBVectorStore",
    "create_vector_store",
    "RAGClientImpl",
    "RetrievalResult",
    "HybridRetriever",
    "create_rag_client",
]
