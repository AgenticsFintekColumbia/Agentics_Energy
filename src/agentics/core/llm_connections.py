import os

from crewai import LLM
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict

# class SolveResponse(BaseModel):
#     status: str 
#     message: Optional[str] = None
#     objective_cost: Optional[float] = Field(..., description="total objective cost i.e. sum of (price_sell times grid_export subtracted from price_buy times grid_import) multiplied by the sample time of operation dt_hours across all timestamps")
#     charge_kw: Optional[List[float]] =Field(None, description="Battery charge schedule in kW")
#     discharge_kw: Optional[List[float]] = Field(None, description="Battery discharge schedule in kW")
#     import_kw: Optional[List[float]] = Field(None, description="Grid import schedule in kW")
#     export_kw: Optional[List[float]] = Field(None, description="Grid export schedule in kW") #le=0 #class validators and also add constraints to ensure that the outputs make sense
#     soc: Optional[List[float]] = Field(None, description="State of Charge (SoC) over time")
#     decision: Optional[List[float]] = Field(None, description="Decision taken at each time step by the battery - charge (+1), discharge (-1), idle (0)")
#     confidence: Optional[List[float]] = Field(None, description="Confidence level of each decision (0 to 1)")

# Turbo Models gpt-oss:20b, deepseek-v3.1:671b

load_dotenv()

verbose = True

def get_llm_provider(provider_name: str = None) -> LLM:
    """
    Retrieve the LLM instance based on the provider name. If no provider name is given,
    the function returns the first available LLM.

    Args:
        provider_name (str): The name of the LLM provider (e.g., 'openai', 'watsonx', 'gemini').

    Returns:
        LLM: The corresponding LLM instance.

    Raises:
        ValueError: If the specified provider is not available.
    """

    if provider_name is None or provider_name == "":
        if len(available_llms) > 0:
            if verbose:
                logger.debug(
                    f"Available LLM providers: {list(available_llms)}. None specified, defaulting to '{list(available_llms)[2]}'"
                )
            return list(available_llms.values())[2]
        else:
            raise ValueError(
                "No LLM is available. Please check your .env configuration."
            )

    else:
        if provider_name in available_llms:
            if verbose:
                logger.debug(f"Using specified LLM provider: {provider_name}")
            return available_llms[provider_name]
        else:
            raise ValueError(
                f"LLM provider '{provider_name}' is not available. Please check your .env configuration."
            )


available_llms = {}

gemini_llm = (
    LLM(model=os.getenv("GEMINI_MODEL_ID", "gemini/gemini-2.0-flash"), temperature=0.7)
    if os.getenv("GEMINI_API_KEY")
    else None
)


ollama_llm = (
    LLM(model=os.getenv("OLLAMA_MODEL_ID"), base_url="http://localhost:11434")
    if os.getenv("OLLAMA_MODEL_ID")
    else None
)

# ollama_llm = (
#     LLM(
#         model=os.getenv("OLLAMA_MODEL_ID"), 
#         base_url="http://localhost:11434",
#         temperature=0.1,
#         max_tokens=1024,
#         stop=["}}", "}\n", "\n}"]
#     )
#     if os.getenv("OLLAMA_MODEL_ID")
#     else None
# )

# ollama_llm = (
#     LLM(
#         model=os.getenv("OLLAMA_MODEL_ID"), 
#         base_url="http://localhost:11434",
#         temperature=0.1,
#         response_format={
#             "type": "json_object",
#             "schema": {
#                 "type": "object",
#                 "properties": {
#                     "status": {"type": "string", "enum": ["success", "failure"]},
#                     "message": {"type": "string"},
#                     "objective_cost": {"type": "number"},
#                     "charge_kw": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "minItems": 24,
#                         "maxItems": 24
#                     },
#                     "discharge_kw": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "minItems": 24,
#                         "maxItems": 24
#                     },
#                     "import_kw": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "minItems": 24,
#                         "maxItems": 24
#                     },
#                     "export_kw": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "minItems": 24,
#                         "maxItems": 24
#                     },
#                     "soc": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "minItems": 25,
#                         "maxItems": 25
#                     },
#                     "decision": {
#                         "type": "array",
#                         "items": {"type": "integer", "enum": [-1, 0, 1]},
#                         "minItems": 24,
#                         "maxItems": 24
#                     }
#                 },
#                 "required": ["status", "message", "objective_cost", "charge_kw", "discharge_kw", "import_kw", "export_kw", "soc", "decision"]
#             }
#         }
#     )
#     if os.getenv("OLLAMA_MODEL_ID")
#     else None
# )


openai_llm = (
    LLM(
        model=os.getenv(
            "OPENAI_MODEL_ID", "openai/gpt-4"
        ),  # call model by provider/model_name
        temperature=1,
        # top_p=0.9,
        stop=["END"],
        api_key=os.getenv("OPENAI_API_KEY"),
        seed=42,
    )
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL_ID")
    else None
)

watsonx_llm = (
    LLM(
        model=os.getenv("MODEL_ID"),
        base_url=os.getenv("WATSONX_URL"),
        project_id=os.getenv("WATSONX_PROJECTID"),
        api_key=os.getenv("WATSONX_APIKEY"),
        temperature=0,
        max_input_tokens=100000,
    )
    if os.getenv("WATSONX_APIKEY")
    and os.getenv("WATSONX_URL")
    and os.getenv("WATSONX_PROJECTID")
    and os.getenv("MODEL_ID")
    else None
)

vllm_llm = (
    AsyncOpenAI(
        api_key="EMPTY",
        base_url=os.getenv("VLLM_URL"),
        default_headers={
            "Content-Type": "application/json",
        },
    )
    if os.getenv("VLLM_URL")
    else None
)

vllm_crewai = (
    LLM(
        model=os.getenv("VLLM_MODEL_ID"),
        api_key="EMPTY",
        base_url=os.getenv("VLLM_URL"),
        max_tokens=1000,
        temperature=0.0,
    )
    if os.getenv("VLLM_URL") and os.getenv("VLLM_MODEL_ID")
    else None
)

i = 0
if watsonx_llm:
    available_llms["watsonx"] = watsonx_llm
    i += 1
if gemini_llm:
    available_llms["gemini"] = gemini_llm
    i += 1
if openai_llm:
    available_llms["openai"] = openai_llm
if ollama_llm:
    available_llms["ollama"] = ollama_llm
