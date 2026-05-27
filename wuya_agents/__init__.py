"""
WuYa Multi-Agent System

A theory-driven multi-agent system for academic paper evaluation and journal recommendation.

Modules:
    - dea_subagent: DEA (Data Envelopment Analysis) Sub-agent for paper localization

Author: WuYa Team
"""

__version__ = "0.1.0"
__author__ = "WuYa Team"

from .dea_subagent import (
    DEAEngine,
    DEASubAgent,
    DEASubAgentService,
    ScoreVector,
    DEAResult,
    DEAStatus,
)

__all__ = [
    "DEAEngine",
    "DEASubAgent",
    "DEASubAgentService",
    "ScoreVector",
    "DEAResult",
    "DEAStatus",
]
