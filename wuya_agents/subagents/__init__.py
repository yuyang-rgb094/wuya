"""
WuYa Sub-agents Module

This package contains the evaluation sub-agents for the WuYa multi-agent system:

- CUDOS Sub-agent: Normative gatekeeping with veto power
- Innovation Sub-agent: Novelty and creativity evaluation
- Method Sub-agent: Methodology rigor evaluation
- Evidence Sub-agent: Evidence quality evaluation
- Application Sub-agent: Practical relevance evaluation
- Frontier Discovery Sub-agent: DEA frontier discovery and trend analysis

Author: WuYa Team
"""

# CUDOS Sub-agent
from .cudos import (
    CUDOSSubAgent,
    CUDOSDimensionScore,
    CUDOS_DIMENSIONS,
    COMMUNALISM,
    UNIVERSALISM,
    DISINTERESTEDNESS,
    ORGANIZED_SKEPTICISM,
    DIMENSION_DESCRIPTIONS as CUDOS_DIMENSION_DESCRIPTIONS,
    create_cudos_subagent,
)

# Innovation Sub-agent
from .innovation import (
    InnovationSubAgent,
    InnovationDimensionScore,
    INNOVATION_DIMENSIONS,
    NOVELTY,
    SIGNIFICANCE,
    ADVANCEMENT,
    DIMENSION_DESCRIPTIONS as INNOVATION_DIMENSION_DESCRIPTIONS,
    create_innovation_subagent,
)

# Method Sub-agent
from .method import (
    MethodSubAgent,
    MethodDimensionScore,
    METHOD_DIMENSIONS,
    RIGOR,
    VALIDITY,
    REPRODUCIBILITY,
    DIMENSION_DESCRIPTIONS as METHOD_DIMENSION_DESCRIPTIONS,
    create_method_subagent,
)

# Evidence Sub-agent
from .evidence import (
    EvidenceSubAgent,
    EvidenceDimensionScore,
    EVIDENCE_DIMENSIONS,
    STRENGTH,
    CONSISTENCY,
    SUFFICIENCY,
    DIMENSION_DESCRIPTIONS as EVIDENCE_DIMENSION_DESCRIPTIONS,
    create_evidence_subagent,
)

# Application Sub-agent
from .application import (
    ApplicationSubAgent,
    ApplicationDimensionScore,
    ApplicationScenario,
    APPLICATION_DIMENSIONS,
    RELEVANCE,
    FEASIBILITY,
    IMPACT,
    DIMENSION_DESCRIPTIONS as APPLICATION_DIMENSION_DESCRIPTIONS,
    create_application_subagent,
)

# Frontier Discovery Sub-agent
from .frontier import (
    FrontierDiscoverySubAgent,
    SubmissionRecord,
    ImprovementPattern,
    TrendCluster,
    FrontierUpdate,
    create_frontier_discovery_agent,
)

__all__ = [
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
]
