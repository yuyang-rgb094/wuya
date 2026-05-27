"""
DEA Sub-agent for WuYa System

This module implements the DEA (Data Envelopment Analysis) Sub-agent that:
1. Transforms 5-dimensional score vectors into DEA inputs/outputs
2. Calls the DEA Engine for efficiency calculation
3. Interprets results and generates natural language explanations
4. Handles edge cases (insufficient data, cross-disciplinary papers)

Architecture:
    Router -> DEA Sub-agent -> DEA Engine (Python/scipy)
                          -> LLM (for result interpretation)

Author: WuYa Team
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from scipy.optimize import linprog
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DEAStatus(Enum):
    """DEA calculation status"""
    SUCCESS = "success"
    INSUFFICIENT_DATA = "insufficient_data"
    COMPUTATION_ERROR = "computation_error"
    FRONTIER_UNREACHABLE = "frontier_unreachable"


@dataclass
class ScoreVector:
    """Five-dimensional score vector from evaluation sub-agents"""
    innovation: float  # 1-5
    method: float      # 1-5
    evidence: float    # 1-5
    application: float # 1-5
    cudos: float       # 1-5 (normative score)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "innovation": self.innovation,
            "method": self.method,
            "evidence": self.evidence,
            "application": self.application,
            "cudos": self.cudos
        }


@dataclass
class DEAResult:
    """Result from DEA analysis"""
    efficiency_score: float
    confidence_interval: Tuple[float, float]
    status: DEAStatus
    reference_set_size: int
    is_on_frontier: bool
    bootstrap_std: float
    explanation: str = ""
    rag_triggered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "efficiency_score": self.efficiency_score,
            "confidence_interval": self.confidence_interval,
            "status": self.status.value,
            "reference_set_size": self.reference_set_size,
            "is_on_frontier": self.is_on_frontier,
            "bootstrap_std": self.bootstrap_std,
            "explanation": self.explanation,
            "rag_triggered": self.rag_triggered
        }


class DEAEngine:
    """
    DEA Calculation Engine
    
    Implements super-efficiency DEA model with bootstrap confidence intervals.
    This is a pure Python/numpy/scipy implementation - no LLM involved.
    """
    
    def __init__(self, min_reference_papers: int = 50):
        self.min_reference_papers = min_reference_papers
        
    def transform_scores(
        self,
        target_scores: ScoreVector,
        reference_scores: List[ScoreVector]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform 5-dimensional scores into DEA inputs and outputs.
        
        DEA Model:
        - Inputs (minimize): Method flaws, Evidence gap
        - Outputs (maximize): Innovation, Application
        
        Returns:
            target_inputs, target_outputs, ref_inputs, ref_outputs
        """
        # Target paper transformation
        target_inputs = np.array([
            6 - target_scores.method,      # Method flaws
            6 - target_scores.evidence     # Evidence gap
        ])
        target_outputs = np.array([
            target_scores.innovation,        # Innovation
            target_scores.application        # Application
        ])
        
        # Reference set transformation
        ref_inputs = np.array([
            [6 - s.method, 6 - s.evidence]
            for s in reference_scores
        ])
        ref_outputs = np.array([
            [s.innovation, s.application]
            for s in reference_scores
        ])
        
        return target_inputs, target_outputs, ref_inputs, ref_outputs
    
    def solve_super_efficiency(
        self,
        target_inputs: np.ndarray,
        target_outputs: np.ndarray,
        ref_inputs: np.ndarray,
        ref_outputs: np.ndarray
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Solve super-efficiency DEA model using linear programming.
        
        Model:
            min θ
            s.t. Σλ_i * x_ij <= θ * x_kj  (for all inputs j)
                 Σλ_i * y_ir >= y_kr       (for all outputs r)
                 λ_i >= 0
        
        Args:
            target_inputs: (m,) array - m inputs for target DMU
            target_outputs: (s,) array - s outputs for target DMU
            ref_inputs: (n, m) array - n DMUs, m inputs
            ref_outputs: (n, s) array - n DMUs, s outputs
            
        Returns:
            efficiency_score: θ (> 1 means super-efficient, on frontier)
            lambda_weights: optimal weights for reference DMUs
        """
        n_ref = ref_inputs.shape[0]
        n_inputs = len(target_inputs)
        n_outputs = len(target_outputs)
        
        # Objective: minimize θ
        c = np.zeros(n_ref + 1)
        c[0] = 1.0  # θ is the first variable
        
        # Inequality constraints: A_ub @ x <= b_ub
        # Format: [θ, λ_1, λ_2, ..., λ_n]
        
        constraints = []
        
        # Input constraints: Σλ_i * x_ij - θ * x_kj <= 0
        for j in range(n_inputs):
            row = np.zeros(n_ref + 1)
            row[0] = -target_inputs[j]  # coefficient for θ
            row[1:] = ref_inputs[:, j]   # coefficients for λ
            constraints.append(row)
        
        # Output constraints: -Σλ_i * y_ir <= -y_kr
        for r in range(n_outputs):
            row = np.zeros(n_ref + 1)
            row[0] = 0  # θ doesn't appear in output constraints
            row[1:] = ref_outputs[:, r]  # Note: constraint is -Σλy >= -y_k, which is Σλy <= y_k
            constraints.append(-row)  # Negate to get <= form
        
        A_ub = np.array(constraints)
        
        # Right-hand side
        b_ub = np.zeros(n_inputs + n_outputs)
        b_ub[n_inputs:] = -target_outputs  # output RHS
        
        # Bounds: θ unbounded below (can be negative in theory, but we expect > 0)
        # λ_i >= 0
        bounds = [(None, None)] + [(0, None)] * n_ref
        
        # Solve
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'maxiter': 10000}
        )
        
        if not result.success:
            logger.error(f"DEA optimization failed: {result.message}")
            raise RuntimeError(f"DEA optimization failed: {result.message}")
        
        theta = result.x[0]
        lambdas = result.x[1:]
        
        return theta, lambdas
    
    def bootstrap_confidence_interval(
        self,
        target_inputs: np.ndarray,
        target_outputs: np.ndarray,
        ref_inputs: np.ndarray,
        ref_outputs: np.ndarray,
        n_iterations: int = 200,
        confidence_level: float = 0.95
    ) -> Tuple[Tuple[float, float], float]:
        """
        Compute bootstrap confidence interval for efficiency score.
        
        Args:
            n_iterations: number of bootstrap samples (default: 200)
            confidence_level: confidence level (default: 0.95)
            
        Returns:
            (ci_lower, ci_upper), std_dev
        """
        n_ref = ref_inputs.shape[0]
        bootstrap_scores = []
        
        np.random.seed(42)  # For reproducibility
        
        for _ in range(n_iterations):
            # Resample reference set with replacement
            indices = np.random.choice(n_ref, size=n_ref, replace=True)
            boot_inputs = ref_inputs[indices]
            boot_outputs = ref_outputs[indices]
            
            try:
                theta, _ = self.solve_super_efficiency(
                    target_inputs, target_outputs,
                    boot_inputs, boot_outputs
                )
                bootstrap_scores.append(theta)
            except RuntimeError:
                # Skip failed iterations
                continue
        
        if len(bootstrap_scores) < n_iterations // 2:
            logger.warning("Too many bootstrap iterations failed")
            return (float('nan'), float('nan')), float('nan')
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Compute percentile-based confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
        std_dev = np.std(bootstrap_scores)
        
        return (ci_lower, ci_upper), std_dev
    
    def analyze(
        self,
        target_scores: ScoreVector,
        reference_scores: List[ScoreVector]
    ) -> DEAResult:
        """
        Main entry point for DEA analysis.
        
        Args:
            target_scores: ScoreVector for the paper being evaluated
            reference_scores: List of ScoreVectors from reference papers
            
        Returns:
            DEAResult with efficiency score and interpretation
        """
        # Check sufficient data
        if len(reference_scores) < self.min_reference_papers:
            logger.warning(
                f"Insufficient reference papers: {len(reference_scores)} < "
                f"{self.min_reference_papers}"
            )
            return DEAResult(
                efficiency_score=0.0,
                confidence_interval=(0.0, 0.0),
                status=DEAStatus.INSUFFICIENT_DATA,
                reference_set_size=len(reference_scores),
                is_on_frontier=False,
                bootstrap_std=0.0,
                explanation="Insufficient reference papers for reliable DEA analysis."
            )
        
        try:
            # Transform scores
            target_in, target_out, ref_in, ref_out = self.transform_scores(
                target_scores, reference_scores
            )
            
            # Solve DEA
            theta, lambdas = self.solve_super_efficiency(
                target_in, target_out, ref_in, ref_out
            )
            
            # Bootstrap confidence interval
            ci, std = self.bootstrap_confidence_interval(
                target_in, target_out, ref_in, ref_out
            )
            
            # Determine if on frontier
            # θ > 1 means super-efficient (on frontier)
            # θ ≈ 1 means on frontier
            # θ < 1 means below frontier
            is_on_frontier = theta >= 0.99
            ci_includes_one = ci[0] <= 1.0 <= ci[1]
            
            status = DEAStatus.SUCCESS
            if ci_includes_one:
                status = DEAStatus.FRONTIER_UNREACHABLE
            
            return DEAResult(
                efficiency_score=float(theta),
                confidence_interval=ci,
                status=status,
                reference_set_size=len(reference_scores),
                is_on_frontier=is_on_frontier,
                bootstrap_std=float(std)
            )
            
        except Exception as e:
            logger.error(f"DEA computation error: {e}")
            return DEAResult(
                efficiency_score=0.0,
                confidence_interval=(0.0, 0.0),
                status=DEAStatus.COMPUTATION_ERROR,
                reference_set_size=len(reference_scores),
                is_on_frontier=False,
                bootstrap_std=0.0,
                explanation=f"Computation error: {str(e)}"
            )


class DEASubAgent:
    """
    DEA Sub-agent for WuYa System
    
    Responsibilities:
    1. Data preparation: Transform score vectors to DEA format
    2. Call DEA Engine for numerical computation
    3. Result interpretation: Convert numerical results to natural language
    4. RAG triggering: When results are ambiguous or inconsistent with Path A
    
    The Sub-agent itself does NOT perform DEA calculations - it orchestrates
    the DEA Engine and interprets results.
    """
    
    def __init__(
        self,
        dea_engine: Optional[DEAEngine] = None,
        llm_client=None,  # Optional: for result interpretation
        rag_client=None,  # Optional: for theory retrieval
        min_reference_papers: int = 50
    ):
        self.dea_engine = dea_engine or DEAEngine(min_reference_papers)
        self.llm_client = llm_client
        self.rag_client = rag_client
        self.min_reference_papers = min_reference_papers
        
    async def evaluate(
        self,
        paper_id: str,
        score_vector: ScoreVector,
        target_journal: str,
        reference_papers: List[Dict[str, Any]],
        path_a_estimate: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for DEA evaluation.
        
        Args:
            paper_id: Unique identifier for the paper
            score_vector: 5-dimensional scores from evaluation sub-agents
            target_journal: Target journal ID/name
            reference_papers: List of historical papers from target journal
                Each paper should have: {
                    "paper_id": str,
                    "scores": ScoreVector,
                    "citations": int
                }
            path_a_estimate: Optional tier estimate from LLM-as-Mapper (for cross-validation)
            
        Returns:
            Complete evaluation result with interpretation
        """
        logger.info(f"Starting DEA evaluation for paper {paper_id} -> {target_journal}")
        
        # Step 1: Extract reference score vectors
        reference_scores = [
            ScoreVector(**p["scores"])
            for p in reference_papers
            if "scores" in p
        ]
        
        # Step 2: Call DEA Engine (pure Python computation)
        dea_result = self.dea_engine.analyze(score_vector, reference_scores)
        
        # Step 3: Generate interpretation (can use LLM or template-based)
        interpretation = await self._interpret_result(
            dea_result,
            score_vector,
            target_journal,
            path_a_estimate
        )
        dea_result.explanation = interpretation["explanation"]
        dea_result.rag_triggered = interpretation.get("rag_triggered", False)
        
        # Step 4: Cross-validation with Path A if available
        consistency_check = None
        if path_a_estimate:
            consistency_check = self._check_consistency(
                dea_result, path_a_estimate
            )
        
        return {
            "paper_id": paper_id,
            "target_journal": target_journal,
            "dea_result": dea_result.to_dict(),
            "consistency_with_path_a": consistency_check,
            "recommendation": self._generate_recommendation(dea_result, consistency_check)
        }
    
    async def _interpret_result(
        self,
        dea_result: DEAResult,
        score_vector: ScoreVector,
        target_journal: str,
        path_a_estimate: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate natural language interpretation of DEA results.
        
        Can be implemented as:
        1. Template-based (fast, deterministic)
        2. LLM-based (more nuanced, but slower/expensive)
        
        For production, recommend template-based with LLM fallback for edge cases.
        """
        theta = dea_result.efficiency_score
        ci_lower, ci_upper = dea_result.confidence_interval
        
        # Template-based interpretation
        if dea_result.status == DEAStatus.INSUFFICIENT_DATA:
            explanation = (
                f"Insufficient historical data for {target_journal} "
                f"(only {dea_result.reference_set_size} papers available, "
                f"minimum {self.min_reference_papers} required). "
                f"Falling back to Path A (LLM-as-Mapper) estimate."
            )
            return {"explanation": explanation, "rag_triggered": True}
        
        if dea_result.status == DEAStatus.COMPUTATION_ERROR:
            return {
                "explanation": "DEA computation failed. Please check input data.",
                "rag_triggered": True
            }
        
        # Interpret efficiency score
        if theta > 1.1:
            tier_assessment = "significantly above the frontier"
            recommendation = "strong candidate for this journal"
        elif theta > 1.0:
            tier_assessment = "on the efficiency frontier"
            recommendation = "good fit for this journal"
        elif theta > 0.9:
            tier_assessment = "near the frontier"
            recommendation = "competitive candidate"
        elif theta > 0.8:
            tier_assessment = "below but approaching the frontier"
            recommendation = "marginal fit, consider strengthening"
        else:
            tier_assessment = "substantially below the frontier"
            recommendation = "unlikely to be accepted"
        
        # Check confidence interval
        ci_includes_one = ci_lower <= 1.0 <= ci_upper
        uncertainty_note = ""
        rag_triggered = False
        
        if ci_includes_one:
            uncertainty_note = (
                " However, the 95% confidence interval includes the frontier, "
                "indicating statistical uncertainty in this assessment."
            )
            rag_triggered = True
        
        if dea_result.bootstrap_std > 0.15:
            uncertainty_note += (
                " High variability in bootstrap estimates suggests "
                "the reference set may be heterogeneous."
            )
            rag_triggered = True
        
        explanation = (
            f"DEA efficiency score: {theta:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}). "
            f"The paper is {tier_assessment} of {target_journal}. "
            f"Recommendation: {recommendation}.{uncertainty_note}"
        )
        
        # If significant uncertainty, trigger RAG for theoretical grounding
        if rag_triggered and self.rag_client:
            explanation += await self._retrieve_theoretical_grounding(
                score_vector, theta
            )
        
        return {"explanation": explanation, "rag_triggered": rag_triggered}
    
    async def _retrieve_theoretical_grounding(
        self,
        score_vector: ScoreVector,
        theta: float
    ) -> str:
        """
        Retrieve canonical texts for theoretical grounding when results are ambiguous.
        """
        if not self.rag_client:
            return ""
        
        # Construct query based on score pattern
        if score_vector.innovation > 4.5 and theta < 0.9:
            # High innovation but low efficiency - theory-application gap
            query = "Mokyr Q-knowledge A-knowledge bridging theory application"
        elif score_vector.method > 4.5 and score_vector.evidence < 3.0:
            # Strong method, weak evidence
            query = "Campbell internal validity external validity tradeoff"
        else:
            query = "DEA efficiency frontier research evaluation methodology"
        
        # Call RAG (placeholder - integrate with your RAG system)
        # retrieved_texts = await self.rag_client.retrieve(query)
        
        return f" [RAG: Retrieved theoretical grounding for '{query}']"
    
    def _check_consistency(
        self,
        dea_result: DEAResult,
        path_a_estimate: str
    ) -> Dict[str, Any]:
        """
        Check consistency between Path A (LLM) and Path B (DEA) estimates.
        """
        # Parse tier from path_a_estimate (e.g., "Q1-A" -> tier "A")
        path_a_tier = path_a_estimate.split("-")[-1] if "-" in path_a_estimate else path_a_estimate
        
        # Map DEA score to tier
        theta = dea_result.efficiency_score
        if theta > 1.0:
            dea_tier = "A"
        elif theta > 0.9:
            dea_tier = "B"
        elif theta > 0.8:
            dea_tier = "C"
        else:
            dea_tier = "D"
        
        consistent = (path_a_tier == dea_tier)
        
        return {
            "path_a_tier": path_a_tier,
            "dea_tier": dea_tier,
            "consistent": consistent,
            "discrepancy": None if consistent else f"Path A: {path_a_tier}, DEA: {dea_tier}"
        }
    
    def _generate_recommendation(
        self,
        dea_result: DEAResult,
        consistency_check: Optional[Dict]
    ) -> str:
        """Generate final recommendation based on DEA results."""
        if dea_result.status != DEAStatus.SUCCESS:
            return "Unable to provide DEA-based recommendation."
        
        theta = dea_result.efficiency_score
        
        if consistency_check and not consistency_check["consistent"]:
            return (
                f"Path A and DEA estimates diverge. "
                f"{consistency_check['discrepancy']}. "
                f"Consider deeper analysis or human expert review."
            )
        
        if theta > 1.0:
            return "Strong recommendation: Paper exceeds target journal frontier."
        elif theta > 0.9:
            return "Recommended: Paper is competitive for target journal."
        elif theta > 0.8:
            return "Conditional: Address identified weaknesses before submission."
        else:
            return "Not recommended: Consider lower-tier journals."


# ============================================================================
# Startup and Invocation Logic
# ============================================================================

class DEASubAgentService:
    """
    Service wrapper for DEA Sub-agent startup and lifecycle management.
    
    This can be integrated with:
    - FastAPI for REST API endpoints
    - Celery for async task queues
    - Redis for message passing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.dea_engine = None
        self.sub_agent = None
        self._initialized = False
        
    async def startup(self):
        """
        Initialize the DEA Sub-agent service.
        
        Called once at system startup.
        """
        logger.info("Starting DEA Sub-agent service...")
        
        # Initialize DEA Engine (pure Python - no external dependencies)
        min_papers = self.config.get("min_reference_papers", 50)
        self.dea_engine = DEAEngine(min_reference_papers=min_papers)
        
        # Initialize Sub-agent
        # LLM and RAG clients are optional - can work without them
        self.sub_agent = DEASubAgent(
            dea_engine=self.dea_engine,
            llm_client=self.config.get("llm_client"),
            rag_client=self.config.get("rag_client"),
            min_reference_papers=min_papers
        )
        
        self._initialized = True
        logger.info("DEA Sub-agent service started successfully")
        
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down DEA Sub-agent service...")
        self._initialized = False
        
    async def evaluate(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main API endpoint for DEA evaluation.
        
        Request format:
        {
            "paper_id": "uuid",
            "score_vector": {
                "innovation": 4.5,
                "method": 4.2,
                "evidence": 3.5,
                "application": 4.0,
                "cudos": 4.8
            },
            "target_journal": "Nature ML",
            "reference_papers": [...],
            "path_a_estimate": "Q1-B"  # optional
        }
        """
        if not self._initialized:
            raise RuntimeError("DEA Sub-agent service not initialized")
        
        # Parse request
        score_vector = ScoreVector(**request["score_vector"])
        
        # Call sub-agent
        result = await self.sub_agent.evaluate(
            paper_id=request["paper_id"],
            score_vector=score_vector,
            target_journal=request["target_journal"],
            reference_papers=request["reference_papers"],
            path_a_estimate=request.get("path_a_estimate")
        )
        
        return result


# ============================================================================
# Example Usage and Testing
# ============================================================================

async def main():
    """
    Example usage of DEA Sub-agent.
    
    This demonstrates the complete flow from startup to evaluation.
    """
    
    # Configuration
    config = {
        "min_reference_papers": 50,
        # "llm_client": None,  # Optional: provide LLM client
        # "rag_client": None,  # Optional: provide RAG client
    }
    
    # Initialize service
    service = DEASubAgentService(config)
    await service.startup()
    
    # Create sample data
    target_paper = ScoreVector(
        innovation=4.5,
        method=4.2,
        evidence=3.5,
        application=4.0,
        cudos=4.8
    )
    
    # Generate synthetic reference papers (in production, from database)
    np.random.seed(42)
    reference_papers = []
    for i in range(100):
        ref_paper = {
            "paper_id": f"ref_{i}",
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
    
    # Evaluation request
    request = {
        "paper_id": "test_paper_001",
        "score_vector": target_paper.to_dict(),
        "target_journal": "Nature Machine Intelligence",
        "reference_papers": reference_papers,
        "path_a_estimate": "Q1-A"
    }
    
    # Execute evaluation
    print("=" * 60)
    print("DEA Sub-agent Evaluation")
    print("=" * 60)
    
    result = await service.evaluate(request)
    
    print(f"\nPaper ID: {result['paper_id']}")
    print(f"Target Journal: {result['target_journal']}")
    print(f"\nDEA Result:")
    print(f"  Efficiency Score: {result['dea_result']['efficiency_score']:.3f}")
    print(f"  Confidence Interval: {result['dea_result']['confidence_interval']}")
    print(f"  On Frontier: {result['dea_result']['is_on_frontier']}")
    print(f"  Status: {result['dea_result']['status']}")
    print(f"\nExplanation:")
    print(f"  {result['dea_result']['explanation']}")
    print(f"\nConsistency Check:")
    print(f"  {result['consistency_with_path_a']}")
    print(f"\nRecommendation:")
    print(f"  {result['recommendation']}")
    
    # Shutdown
    await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
