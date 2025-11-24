import logging
from agentics import AG
from agentics.core.llm_connections import get_llm_provider
from typing import List, Optional
from agentic_energy.schemas import ReasoningRequest, ReasoningResponse, SolveRequest, SolveResponse

logger = logging.getLogger(__name__)


class BatteryReasoningAG:
    """Agentics-based reasoning system for battery decisions"""
    
    def __init__(self, llm_provider: str = "gemini"):
        """Initialize the reasoning system with specified LLM provider"""
        self.llm = get_llm_provider(llm_provider)
        self.target = AG(
            atype=ReasoningResponse,
            llm=self.llm,
            verbose_agent=True,
            verbose_transduction=True,
            instructions="""
            You are an expert system explaining battery charging/discharging/idle decisions in an energy arbitrage context.
            Analyze the provided data including:
            1. Battery state (SoC, capacity, efficiency)
            2. Market conditions (prices, demand)
            3. Algorithm decisions (charge/discharge/idle)
            
            Explain why the decision was optimal given the constraints and objectives.
            Consider:
            - Price patterns and arbitrage opportunities
            - Battery constraints (capacity, power limits)
            - Efficiency losses
            - Future price expectations if available
            """
        )
    
    async def explain_decision(self, request: ReasoningRequest) -> ReasoningResponse:
        """Generate an explanation for a specific battery decision"""
        source = AG(
            atype=ReasoningRequest,
            states=[request]
        )
        result=await (self.target << source)
        return result.states[0]
    
    async def explain_sequence(self, 
                             solve_request: SolveRequest, 
                             solve_response: SolveResponse,
                             indices: Optional[List[int]] = None) -> List[ReasoningResponse]:
        """Generate explanations for a sequence of decisions
        
        Args:
            solve_request: Original solve request
            solve_response: Solver response with decisions
            indices: Specific timesteps to explain. If None, explains all decisions.
        """
        if indices is None:
            indices = range(len(solve_response.decision))
            
        requests = [
            ReasoningRequest(
                solve_request=solve_request,
                solve_response=solve_response,
                timestamp_index=i
            ) for i in indices
        ]
        
        source = AG(
            atype=ReasoningRequest,
            states=requests
        )
        
        result_ag = await (self.target << source)
        return result_ag.states

# class ReasoningRequest(BaseModel):
#     """Input schema for the reasoning system"""
#     solve_request: SolveRequest = Field(..., description="Original solve request with battery parameters and inputs")
#     solve_response: SolveResponse = Field(..., description="Solver response containing decisions and results")
#     timestamp_index: int = Field(..., description="Index of the decision to explain")
#     context_window: int = Field(default=6, description="Number of timesteps before/after to consider for context")

# class ReasoningResponse(BaseModel):
#     """Output schema containing the reasoning explanation"""
#     explanation: str = Field(..., description="Detailed natural language explanation of the decision")
#     key_factors: List[str] = Field(default_factory=list, description="Key factors that influenced the decision")
#     confidence: float = Field(..., ge=0, le=1, description="Confidence in the explanation (0-1)")
#     supporting_data: Dict[str, float] = Field(
#         default_factory=dict,
#         description="Relevant numerical data supporting the explanation (e.g., price_delta, soc_margin)"
#     )