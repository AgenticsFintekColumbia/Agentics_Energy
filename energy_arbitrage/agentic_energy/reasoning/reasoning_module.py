"""
Battery Decision Reasoning using Agentics Framework

This module provides LLM-based reasoning capabilities to explain battery charging/discharging/idle decisions
made by different algorithms (MILP, RL, Heuristic) in the context of energy arbitrage.

IMPORTANT: This module applies a patch to fix a bug in Agentics framework where transduction
fails with Gemini and other non-OpenAI LLM providers. The patch is applied automatically on import.

Bug Details:
- Agentics incorrectly checks `type(llm) == LLM` which fails for Gemini (returns GeminiCompletion)
- This causes it to use PydanticTransducerVLLM (abstract class) instead of PydanticTransducerCrewAI
- Fix: Use isinstance(llm, AsyncOpenAI) to detect vLLM, otherwise use CrewAI transducer
- Location: src/agentics/core/agentics.py lines 542-546
"""

import logging
from agentics import Agentics as AG
from agentics.core.llm_connections import get_llm_provider
from typing import List, Optional
from agentic_energy.schemas import ReasoningRequest, ReasoningResponse, SolveRequest, SolveResponse

logger = logging.getLogger(__name__)

# # ===== AGENTICS FRAMEWORK PATCH =====
# def _apply_agentics_transducer_patch():
#     """
#     Apply minimal patch to fix Agentics LLM type checking bug.

#     This patches the AG.__lshift__ method's transducer selection logic to correctly
#     identify LLM types and use the appropriate transducer (CrewAI vs VLLM).
#     """
#     try:
#         from agentics.core.agentics import AG as AGClass
#         from agentics.core.async_executor import PydanticTransducerCrewAI, PydanticTransducerVLLM
#         from openai import AsyncOpenAI

#         # Save original method
#         _original_lshift = AGClass.__lshift__

#         async def _patched_lshift(self, other):
#             """Patched __lshift__ with correct transducer selection"""
#             # Store original transducer selection logic
#             original_llm = self.llm

#             # We'll intercept just before transducer creation by temporarily
#             # replacing the llm attribute with a marker if needed
#             # This is a minimal intervention compared to reimplementing the whole method

#             # Actually, simpler approach: just call original but wrap potential error
#             try:
#                 return await _original_lshift(self, other)
#             except TypeError as e:
#                 if "Can't instantiate abstract class PydanticTransducerVLLM" in str(e):
#                     # The bug occurred - force use of CrewAI transducer by monkey-patching
#                     # the type check in the original method
#                     logger.warning("Detected Agentics transducer bug - applying runtime fix")

#                     # We need to force the correct transducer choice
#                     # The cleanest way is to temporarily patch the module-level classes
#                     from agentics.core import async_executor
#                     _orig_vllm = async_executor.PydanticTransducerVLLM

#                     # Temporarily replace VLLM with CrewAI
#                     async_executor.PydanticTransducerVLLM = PydanticTransducerCrewAI

#                     try:
#                         result = await _original_lshift(self, other)
#                         return result
#                     finally:
#                         # Restore original
#                         async_executor.PydanticTransducerVLLM = _orig_vllm
#                 else:
#                     raise

#         # Apply the patch
#         AGClass.__lshift__ = _patched_lshift
#         logger.info("✓ Agentics transducer patch applied successfully")

#     except Exception as e:
#         logger.warning(f"Could not apply Agentics patch: {e}. Transduction may fail with non-OpenAI providers.")

# # Apply patch on module import
# _apply_agentics_transducer_patch()
# # ===== END PATCH =====

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
        return await (self.target << source)
    
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