"""
Base classes and interfaces for WuYa Sub-agents

This module defines the common interfaces and base classes that all sub-agents
must implement to ensure consistent integration with the Router.

Author: WuYa Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
from enum import Enum
import asyncio
from datetime import datetime
import uuid


class EvaluationDimension(Enum):
    """Five evaluation dimensions"""
    INNOVATION = "innovation"
    METHOD = "method"
    EVIDENCE = "evidence"
    APPLICATION = "application"
    CUDOS = "cudos"


class AgentStatus(Enum):
    """Sub-agent execution status"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RAG_TRIGGERED = "rag_triggered"


@dataclass
class ParsedPaper:
    """Structured paper content after parsing"""
    paper_id: str
    title: str
    abstract: str
    content: str  # Full text or extracted content
    authors: List[str]
    keywords: List[str]
    discipline: str
    citations: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "keywords": self.keywords,
            "discipline": self.discipline,
            "citations_count": len(self.citations),
            "references_count": len(self.references),
        }


@dataclass
class SubDimensionScore:
    """Score for a sub-dimension"""
    name: str
    score: float  # 1-5
    weight: float = 1.0
    justification: str = ""
    
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class DimensionResult:
    """Result from a single evaluation dimension"""
    dimension: EvaluationDimension
    overall_score: float  # 1-5
    sub_dimensions: List[SubDimensionScore]
    narrative: str
    rag_citations: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.SUCCESS
    processing_time_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "overall_score": self.overall_score,
            "sub_dimensions": [
                {"name": s.name, "score": s.score, "justification": s.justification}
                for s in self.sub_dimensions
            ],
            "narrative": self.narrative,
            "rag_citations": self.rag_citations,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }


@dataclass
class ScoreVector:
    """Five-dimensional score vector from all evaluation sub-agents"""
    innovation: float
    method: float
    evidence: float
    application: float
    cudos: float
    
    # Detailed sub-scores
    innovation_details: Dict[str, float] = field(default_factory=dict)
    method_details: Dict[str, float] = field(default_factory=dict)
    evidence_details: Dict[str, float] = field(default_factory=dict)
    application_details: Dict[str, float] = field(default_factory=dict)
    cudos_details: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "innovation": self.innovation,
            "method": self.method,
            "evidence": self.evidence,
            "application": self.application,
            "cudos": self.cudos,
            "details": {
                "innovation": self.innovation_details,
                "method": self.method_details,
                "evidence": self.evidence_details,
                "application": self.application_details,
                "cudos": self.cudos_details,
            }
        }
    
    @classmethod
    def from_dimension_results(
        cls,
        results: Dict[EvaluationDimension, DimensionResult]
    ) -> "ScoreVector":
        """Build ScoreVector from individual dimension results"""
        return cls(
            innovation=results[EvaluationDimension.INNOVATION].overall_score,
            method=results[EvaluationDimension.METHOD].overall_score,
            evidence=results[EvaluationDimension.EVIDENCE].overall_score,
            application=results[EvaluationDimension.APPLICATION].overall_score,
            cudos=results[EvaluationDimension.CUDOS].overall_score,
            innovation_details={
                s.name: s.score 
                for s in results[EvaluationDimension.INNOVATION].sub_dimensions
            },
            method_details={
                s.name: s.score 
                for s in results[EvaluationDimension.METHOD].sub_dimensions
            },
            evidence_details={
                s.name: s.score 
                for s in results[EvaluationDimension.EVIDENCE].sub_dimensions
            },
            application_details={
                s.name: s.score 
                for s in results[EvaluationDimension.APPLICATION].sub_dimensions
            },
            cudos_details={
                s.name: s.score 
                for s in results[EvaluationDimension.CUDOS].sub_dimensions
            },
        )


class RAGClient(Protocol):
    """Protocol for RAG client - can be implemented by various backends"""
    
    async def retrieve(
        self, 
        query: str, 
        context: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant passages based on query"""
        ...


class LLMClient(Protocol):
    """Protocol for LLM client - can be implemented by various providers"""
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text from LLM"""
        ...


class BaseSubAgent(ABC):
    """
    Abstract base class for all evaluation sub-agents.
    
    All sub-agents must implement:
    1. evaluate() - main evaluation logic
    2. get_system_prompt() - system prompt for the agent
    3. should_trigger_rag() - determine when to trigger RAG
    
    The base class provides:
    1. Common initialization
    2. RAG integration hooks
    3. Result formatting
    4. Error handling
    """
    
    def __init__(
        self,
        dimension: EvaluationDimension,
        llm_client: Optional[LLMClient] = None,
        rag_client: Optional[RAGClient] = None,
        timeout_seconds: float = 60.0,
        rag_threshold: float = 0.7  # Confidence threshold for RAG trigger
    ):
        self.dimension = dimension
        self.llm_client = llm_client
        self.rag_client = rag_client
        self.timeout_seconds = timeout_seconds
        self.rag_threshold = rag_threshold
        self._initialized = False
    
    async def initialize(self):
        """Initialize the sub-agent - called once at startup"""
        self._initialized = True
    
    async def evaluate(
        self,
        paper: ParsedPaper,
        context: Optional[Dict[str, Any]] = None
    ) -> DimensionResult:
        """
        Main evaluation entry point.
        
        Args:
            paper: Parsed paper content
            context: Optional context from other agents or previous evaluations
            
        Returns:
            DimensionResult with scores and narrative
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            # Check if RAG should be triggered
            rag_citations = []
            if self.should_trigger_rag(paper, context):
                rag_citations = await self._trigger_rag(paper)
            
            # Perform evaluation
            result = await self._evaluate_impl(paper, context, rag_citations)
            
            # Calculate processing time
            processing_time = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            result.processing_time_ms = processing_time
            
            return result
            
        except asyncio.TimeoutError:
            return DimensionResult(
                dimension=self.dimension,
                overall_score=0.0,
                sub_dimensions=[],
                narrative=f"{self.dimension.value} evaluation timed out",
                status=AgentStatus.TIMEOUT
            )
        except Exception as e:
            return DimensionResult(
                dimension=self.dimension,
                overall_score=0.0,
                sub_dimensions=[],
                narrative=f"{self.dimension.value} evaluation failed: {str(e)}",
                status=AgentStatus.FAILED
            )
    
    @abstractmethod
    async def _evaluate_impl(
        self,
        paper: ParsedPaper,
        context: Optional[Dict[str, Any]],
        rag_citations: List[str]
    ) -> DimensionResult:
        """
        Implementation-specific evaluation logic.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this sub-agent"""
        pass
    
    @abstractmethod
    def should_trigger_rag(
        self,
        paper: ParsedPaper,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Determine if RAG should be triggered for this evaluation.
        Can be based on confidence, paper characteristics, etc.
        """
        pass
    
    async def _trigger_rag(self, paper: ParsedPaper) -> List[str]:
        """Trigger RAG retrieval and return citations"""
        if not self.rag_client:
            return []
        
        query = self._construct_rag_query(paper)
        results = await self.rag_client.retrieve(query)
        
        return [r.get("citation", "") for r in results]
    
    @abstractmethod
    def _construct_rag_query(self, paper: ParsedPaper) -> str:
        """Construct query for RAG retrieval"""
        pass


class CUDOSResult:
    """Special result type for CUDOS gatekeeping"""
    
    def __init__(
        self,
        gate_pass: bool,
        dimensions: Dict[str, Dict[str, Any]],
        veto_reason: Optional[str] = None,
        rag_citations: List[str] = None
    ):
        self.gate_pass = gate_pass
        self.dimensions = dimensions
        self.veto_reason = veto_reason
        self.rag_citations = rag_citations or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_pass": self.gate_pass,
            "dimensions": self.dimensions,
            "veto_reason": self.veto_reason,
            "rag_citations": self.rag_citations,
        }


class CUDOSSubAgent(BaseSubAgent):
    """
    CUDOS Sub-agent for normative gatekeeping.
    
    Special behavior: Has veto power - can block paper from further evaluation.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_client: Optional[RAGClient] = None,
        veto_threshold: float = 2.0  # Score below this triggers veto
    ):
        super().__init__(
            dimension=EvaluationDimension.CUDOS,
            llm_client=llm_client,
            rag_client=rag_client
        )
        self.veto_threshold = veto_threshold
    
    async def evaluate(
        self,
        paper: ParsedPaper,
        context: Optional[Dict[str, Any]] = None
    ) -> CUDOSResult:
        """
        CUDOS evaluation - returns CUDOSResult instead of DimensionResult.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Evaluate CUDOS dimensions
            dimensions = await self._evaluate_cudos_dimensions(paper)
            
            # Determine if veto
            gate_pass = all(
                d["score"] >= self.veto_threshold 
                for d in dimensions.values()
            )
            
            veto_reason = None
            if not gate_pass:
                failed_dims = [
                    name for name, d in dimensions.items() 
                    if d["score"] < self.veto_threshold
                ]
                veto_reason = f"Failed CUDOS dimensions: {', '.join(failed_dims)}"
            
            # Trigger RAG if needed
            rag_citations = []
            if not gate_pass and self.rag_client:
                rag_citations = await self._trigger_rag(paper)
            
            return CUDOSResult(
                gate_pass=gate_pass,
                dimensions=dimensions,
                veto_reason=veto_reason,
                rag_citations=rag_citations
            )
            
        except Exception as e:
            # On error, fail open (allow through) but log
            return CUDOSResult(
                gate_pass=True,
                dimensions={},
                veto_reason=f"CUDOS evaluation error: {str(e)}"
            )
    
    async def _evaluate_cudos_dimensions(
        self,
        paper: ParsedPaper
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate each CUDOS dimension"""
        # TODO: Implement actual evaluation logic
        # For now, return placeholder
        return {
            "communalism": {"score": 4.0, "issues": []},
            "universalism": {"score": 4.5, "issues": []},
            "disinterestedness": {"score": 4.0, "issues": []},
            "organized_skepticism": {"score": 4.5, "issues": []},
        }
    
    async def _evaluate_impl(self, *args, **kwargs) -> DimensionResult:
        """Not used - CUDOS uses custom evaluate()"""
        raise NotImplementedError("CUDOS uses evaluate() instead")
    
    def get_system_prompt(self) -> str:
        return """You are the CUDOS Sub-agent for academic paper evaluation.
        
Your task is to evaluate the paper against Merton's CUDOS norms:
- Communalism: Is knowledge shared openly?
- Universalism: Are claims evaluated objectively?
- Disinterestedness: Are there conflicts of interest?
- Organized Skepticism: Does the paper acknowledge limitations?

You have VETO power - if any norm is severely violated, the paper is rejected.
"""
    
    def should_trigger_rag(self, paper: ParsedPaper, context: Optional[Dict]) -> bool:
        """Trigger RAG when potential ethical issues are detected"""
        # TODO: Implement actual detection logic
        return False
    
    def _construct_rag_query(self, paper: ParsedPaper) -> str:
        return "Merton CUDOS norms scientific ethics"


# ============================================================================
# Router Interface
# ============================================================================

class Router:
    """
    Router orchestrates the two-phase evaluation workflow.
    
    Phase 1: CUDOS gatekeeping
    Phase 2: Parallel evaluation by four sub-agents
    """
    
    def __init__(
        self,
        cudos_agent: CUDOSSubAgent,
        innovation_agent: BaseSubAgent,
        method_agent: BaseSubAgent,
        evidence_agent: BaseSubAgent,
        application_agent: BaseSubAgent,
        dea_service=None,  # Optional DEA service for Path B
        llm_client: Optional[LLMClient] = None,
    ):
        self.cudos_agent = cudos_agent
        self.innovation_agent = innovation_agent
        self.method_agent = method_agent
        self.evidence_agent = evidence_agent
        self.application_agent = application_agent
        self.dea_service = dea_service
        self.llm_client = llm_client
        
        self._agents = {
            EvaluationDimension.INNOVATION: innovation_agent,
            EvaluationDimension.METHOD: method_agent,
            EvaluationDimension.EVIDENCE: evidence_agent,
            EvaluationDimension.APPLICATION: application_agent,
        }
    
    async def process_paper(
        self,
        paper: ParsedPaper,
        user_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main entry point for paper processing.
        
        Args:
            paper: Parsed paper content
            user_intent: {
                "type": "recommendation" | "review",
                "target_journal": Optional[str],
                "author_preferences": Optional[Dict]
            }
            
        Returns:
            Complete evaluation result
        """
        # Phase 1: CUDOS Gatekeeping
        cudos_result = await self.cudos_agent.evaluate(paper)
        
        if not cudos_result.gate_pass:
            return {
                "paper_id": paper.paper_id,
                "status": "vetoed",
                "cudos_result": cudos_result.to_dict(),
                "recommendation": None,
                "review_report": None,
            }
        
        # Phase 2: Parallel Evaluation
        evaluation_tasks = [
            agent.evaluate(paper)
            for agent in self._agents.values()
        ]
        
        dimension_results_list = await asyncio.gather(*evaluation_tasks)
        
        dimension_results = {
            result.dimension: result
            for result in dimension_results_list
        }
        
        # Build score vector
        score_vector = ScoreVector.from_dimension_results(dimension_results)
        
        # Path A: LLM as Mapper
        path_a_result = await self._path_a_mapper(score_vector, paper.discipline)
        
        # Path B: DEA Analysis (if target journal specified)
        path_b_result = None
        if user_intent.get("target_journal") and self.dea_service:
            path_b_result = await self._path_b_dea(
                score_vector, 
                user_intent["target_journal"]
            )
        
        # Cross-validation
        consistency = None
        if path_b_result:
            consistency = self._check_consistency(path_a_result, path_b_result)
        
        # Generate output based on user intent
        if user_intent.get("type") == "recommendation":
            output = self._generate_recommendation(
                paper, score_vector, path_a_result, path_b_result, consistency
            )
        else:
            output = self._generate_review_report(
                paper, dimension_results, score_vector, path_a_result, path_b_result
            )
        
        return {
            "paper_id": paper.paper_id,
            "status": "completed",
            "cudos_result": cudos_result.to_dict(),
            "dimension_results": {
                dim.value: result.to_dict()
                for dim, result in dimension_results.items()
            },
            "score_vector": score_vector.to_dict(),
            "path_a": path_a_result,
            "path_b": path_b_result,
            "consistency": consistency,
            "output": output,
        }
    
    async def _path_a_mapper(
        self,
        score_vector: ScoreVector,
        discipline: str
    ) -> Dict[str, Any]:
        """
        Path A: LLM as Mapper with discipline prior.
        """
        if not self.llm_client:
            return {"estimated_tier": "unknown", "confidence": "low"}
        
        # TODO: Implement actual LLM mapping with prior
        # For now, return placeholder
        avg_score = (
            score_vector.innovation + 
            score_vector.method + 
            score_vector.evidence + 
            score_vector.application
        ) / 4
        
        if avg_score > 4.0:
            tier = "Q1-A"
        elif avg_score > 3.5:
            tier = "Q1-B"
        elif avg_score > 3.0:
            tier = "Q2-A"
        elif avg_score > 2.5:
            tier = "Q2-B"
        else:
            tier = "Q3"
        
        return {
            "estimated_tier": tier,
            "confidence": "medium",
            "reasoning": f"Based on average score {avg_score:.2f}",
        }
    
    async def _path_b_dea(
        self,
        score_vector: ScoreVector,
        target_journal: str
    ) -> Optional[Dict[str, Any]]:
        """
        Path B: DEA efficiency analysis.
        """
        if not self.dea_service:
            return None
        
        # Delegate to DEA service
        # TODO: Implement actual DEA call
        return {
            "efficiency_score": 1.05,
            "confidence_interval": (0.95, 1.15),
            "status": "success",
        }
    
    def _check_consistency(
        self,
        path_a: Dict[str, Any],
        path_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check consistency between Path A and Path B"""
        # TODO: Implement actual consistency check
        return {
            "consistent": True,
            "discrepancy": None,
        }
    
    def _generate_recommendation(
        self,
        paper: ParsedPaper,
        score_vector: ScoreVector,
        path_a: Dict[str, Any],
        path_b: Optional[Dict[str, Any]],
        consistency: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate submission recommendation"""
        return {
            "type": "recommendation",
            "estimated_tier": path_a.get("estimated_tier"),
            "recommended_journals": [],
            "reasoning": path_a.get("reasoning"),
        }
    
    def _generate_review_report(
        self,
        paper: ParsedPaper,
        dimension_results: Dict[EvaluationDimension, DimensionResult],
        score_vector: ScoreVector,
        path_a: Dict[str, Any],
        path_b: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate review report"""
        return {
            "type": "review_report",
            "dimensions": {
                dim.value: result.to_dict()
                for dim, result in dimension_results.items()
            },
            "overall_assessment": path_a.get("reasoning"),
        }
