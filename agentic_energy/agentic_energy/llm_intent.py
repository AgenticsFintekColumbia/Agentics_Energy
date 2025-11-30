# agentic_energy_app/llm_intents.py

import json
from textwrap import dedent
from typing import Optional

from pydantic import BaseModel
from crewai import LLM


class ChatIntent(BaseModel):
    intent: str                      # "start_pipeline" | "reasoning" | "generic_qa"
    timestamp_index_asked: Optional[int] = None


def classify_intent(prompt: str, context: str = "") -> ChatIntent:
    """
    Zero-shot classification via LLM to decide which agent to trigger.
    """
    system_prompt = dedent(
        """
        You are an intent router for an energy arbitrage assistant.
        Classify the user's message into one of three intents:

        - "start_pipeline": The user wants to generate or run a schedule, 
          simulate tomorrow, run the optimization, or similar.
        - "reasoning": The user wants an explanation for optimizer decisions,
          e.g., asks "why", "explain", or "interpret" the schedule or actions.
        - "generic_qa": The user just asks conceptual questions about battery 
          arbitrage, markets, or anything not directly triggering the pipeline.

        Additionally, if intent is "reasoning" and the user refers to a 
        specific hour or time index t (e.g., "at hour 5", "at index 10"), 
        set "timestamp_index_asked" to that zero-based integer. If not clear, 
        use null.

        Respond ONLY with JSON matching this schema:
        {
          "intent": "start_pipeline" | "reasoning" | "generic_qa",
          "timestamp_index_asked": <integer or null>
        }
        """
    )

    full_prompt = system_prompt + "\n\nContext:\n" + context + "\n\nUser:\n" + prompt

    try:
        struct_llm = LLM(
            model="gemini/gemini-2.0-flash",
            response_format=ChatIntent,
        )
        response = struct_llm.call(full_prompt)

        if isinstance(response, ChatIntent):
            return response
        if isinstance(response, dict):
            return ChatIntent.model_validate(response)
        if hasattr(response, "model_dump"):
            return ChatIntent.model_validate(response.model_dump())
        if isinstance(response, str):
            data = json.loads(response)
            return ChatIntent.model_validate(data)

        return ChatIntent(intent="generic_qa", timestamp_index_asked=None)

    except Exception:
        lp = prompt.lower()
        if "schedule" in lp or "run" in lp or "tomorrow" in lp:
            return ChatIntent(intent="start_pipeline", timestamp_index_asked=None)
        if "why" in lp or "explain" in lp:
            return ChatIntent(intent="reasoning", timestamp_index_asked=None)
        return ChatIntent(intent="generic_qa", timestamp_index_asked=None)


def answer_generic_qa(prompt: str, context: str = "") -> str:
    """
    Generic chat about arbitrage + data. Does NOT trigger agents, just Q&A.
    """
    system_prompt = dedent(
        """
        You are an expert assistant on battery arbitrage in electricity markets.
        Explain things clearly and concretely. If a daily arbitrage run has been
        computed, you may refer to the schedule and objective cost from context.
        """
    )

    full_prompt = system_prompt + "\n\nContext:\n" + context + "\n\nUser:\n" + prompt

    try:
        llm = LLM(model="gemini/gemini-2.0-flash")
        response = llm.call(full_prompt)
        return str(response)
    except Exception as e:
        return (
            "LLM chat is not fully wired up yet. "
            f"(Error: {e})"
        )
