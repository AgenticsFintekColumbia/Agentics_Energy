# agentic_energy/reasoning/reasoning_mcp_server.py

import asyncio
import logging
from typing import List, Optional

import numpy as np
from agentics import AG
from agentics.core.llm_connections import get_llm_provider

from agentic_energy.schemas import (
    ReasoningRequest,
    ReasoningResponse,
    SolveRequest,
    SolveResponse,
)
from agentic_energy.reasoning.reasoning_module import BatteryReasoningAG

from mcp.server.fastmcp import FastMCP
logger = logging.getLogger(__name__)
mcp = FastMCP("REASONING")

# Instantiate a single shared reasoning agent
_reasoner = BatteryReasoningAG(llm_provider="gemini")


@mcp.tool()
async def reasoning_explain(reasoningrequest: ReasoningRequest) -> ReasoningResponse:
    """
    Explain a single battery decision (one timestamp).

    reasoningrequest:
        reasoningrequest: ReasoningRequest with solve_request, solve_response, and timestamp_index.

    Returns:
        ReasoningResponse with explanation, key_factors, confidence, etc.
    """
    return await _reasoner.explain_decision(reasoningrequest)


# # If you want a sequence tool you can call from a single MCP invocation,
# # you can define a small wrapper schema here:

from pydantic import BaseModel, Field
class ReasoningSequenceRequest(BaseModel):
    """Wrapper for sequence-level reasoning calls."""
    solve_request: SolveRequest = Field(..., description="Original solve request with battery + inputs.")
    solve_response: SolveResponse = Field(..., description="Solver response containing schedules/decisions.")
    indices: Optional[List[int]] = Field(default=None,description="Which timesteps to explain. If None, explain all.",)

@mcp.tool()
async def reasoning_explain_sequence(reasoningsequencerequest: ReasoningSequenceRequest) -> List[ReasoningResponse]:
    """
    Explain a sequence of battery decisions.

    reasoningsequencerequest:
        reasoningsequencerequest: ReasoningSequenceRequest with solve_request, solve_response, indices.

    Returns:
        List[ReasoningResponse] for the requested timesteps.
    """
    return await _reasoner.explain_sequence(
            solve_request=reasoningsequencerequest.solve_request,
            solve_response=reasoningsequencerequest.solve_response,
            indices=reasoningsequencerequest.indices,
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
