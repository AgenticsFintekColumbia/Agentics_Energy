import os
import sys
import json
from textwrap import dedent
from typing import List, Optional
from urllib import response

from pydantic import BaseModel
import streamlit as st
import pandas as pd

from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

from agentics.core.llm_connections import get_llm_provider
from agentic_energy.schemas import (
    BatteryParams,
    DayInputs,
    SolveRequest,
    SolveResponse,
    PlotRequest,
    PlotResponse,
    ReasoningRequest,
    ReasoningResponse,
)
from crewai import LLM

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_tool(tools, name: str):
    """Return MCP tool object by name."""
    for t in tools:
        if t.name == name:
            return t
    raise RuntimeError(f"Tool {name!r} not found. Available: {[t.name for t in tools]}")


def run_milp_solver(battery: BatteryParams, day: DayInputs) -> SolveResponse:
    """Call the milp_solve MCP tool and return a SolveResponse."""
    milp_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.milp.milp_mcp_server"],
        env=os.environ,
    )

    solve_request = SolveRequest(
        battery=battery,
        day=day,
        solver="MILP",
        solver_opts=None,
    )

    with MCPServerAdapter(milp_params) as milp_tools:
        milp_tool = get_tool(milp_tools, "milp_solve")
        call_fn = getattr(milp_tool, "call", None) or getattr(milp_tool, "run", None) or getattr(milp_tool, "__call__", None)
        if call_fn is None:
            raise RuntimeError("Tool has no callable interface")

        raw = call_fn(solverequest=solve_request.model_dump(exclude_none=True))
        data = json.loads(raw)
        return SolveResponse.model_validate(data)


def run_plot(
    solve_request: SolveRequest,
    solve_response: SolveResponse,
    out_path: str = "./plots/daily_battery_schedule.png",
) -> PlotResponse:
    """Call the plot_price_soc MCP tool and return a PlotResponse."""
    viz_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.visualization.visualization_mcp_server"],
        env=os.environ,
    )

    plot_request = PlotRequest(
        solve_request=solve_request,
        solve_response=solve_response,
        out_path=out_path,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with MCPServerAdapter(viz_params) as viz_tools:
        viz_tool = get_tool(viz_tools, "plot_price_soc")
        viz_call_fn = getattr(viz_tool, "call", None) or getattr(viz_tool, "run", None) or getattr(viz_tool, "__call__", None)
        if viz_call_fn is None:
                raise RuntimeError("Tool has no callable interface")
    
        raw = viz_call_fn(plotrequest = plot_request.model_dump(exclude_none=True))
        data = json.loads(raw)
        return PlotResponse.model_validate(data)

def run_reasoning_tool(
    timestamp_index: int,
) -> str:
    """Call the reasoning MCP tool and return a textual explanation."""

    # 🔧 Adjust this path if your reasoning server lives elsewhere
    reasoning_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.reasoning.reasoning_server"],
        env=os.environ,
    )

    # 🔧 Adapt field names of ReasoningRequest to your actual schema
    req = ReasoningRequest(
        solve_request=st.session_state.last_solve_request,
        solve_response=st.session_state.last_solve_response,
        timestamp_index=timestamp_index
    )

    with MCPServerAdapter(reasoning_params) as tools:
        # 🔧 Adapt tool name to whatever you registered in the reasoning MCP server
        reasoning_tool = get_tool(tools, "reasoning_explain")

        raw = reasoning_tool.run(
            reasoningrequest=req.model_dump(exclude_none=True)
        )

        # Handle different possible return types
        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        elif hasattr(raw, "model_dump"):
            data = raw.model_dump()
        else:
            raise TypeError(f"Unexpected reasoning tool return type: {type(raw)}")

        resp = ReasoningResponse.model_validate(data)

        # 🔧 Adapt this to however you named the main text field
        return resp.explanation  # or resp.answer / resp.text

class Question(BaseModel):
    timestamp_index_asked: int
    
def answer_chat(prompt: str, context: str = "") -> str:
    """Simple LLM-based chat about arbitrage + data.

    NOTE: You may need to adapt this to how get_llm_provider(...) is used
    in your stack (CrewAI-compatible LLM wrapper).
    """
    system_prompt = dedent(
        """
        You are an expert assistant on battery arbitrage in electricity markets.
        Explain things clearly and concretely. If a daily arbitrage run has been
        computed, you may refer to the schedule and objective cost from context.
        """
    )

    try:
        struct_llm = LLM(model="gemini/gemini-2.0-flash", response_format=Question)

        # TODO: Adapt this call to your LLM provider interface if needed.
        # Here we assume `llm` is callable like a simple text completion.
        full_prompt = (
            system_prompt
            + "\n\nContext:\n"
            + context
            + "\n\nUser:\n"
            + prompt
        )
        response = struct_llm.call(prompt)
        response_dict = json.loads(response)
        question = Question.model_validate(response_dict)
        response_text = run_reasoning_tool(question.timestamp_index_asked)
        return str(response_text)
    except Exception as e:
        return (
            "LLM chat is not fully wired up yet. "
            f"Please connect get_llm_provider(...) here. (Error: {e})"
        )


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Battery Arbitrage Agent",
    layout="wide",
)

st.title("🔋 Agentic Battery Arbitrage Assistant")

st.markdown(
    """
This app lets you:

1. Configure **battery parameters**  
2. Upload **day-ahead prices & demand** (CSV)  
3. Chat about **arbitrage and your data**  
4. Run the **MILP optimizer + visualization** and see the schedule & plot  
    """
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages: List[dict] = []

if "last_solve_response" not in st.session_state:
    st.session_state.last_solve_response: Optional[SolveResponse] = None

if "last_solve_request" not in st.session_state:
    st.session_state.last_solve_request: Optional[SolveRequest] = None

if "last_plot" not in st.session_state:
    st.session_state.last_plot: Optional[PlotResponse] = None


# ---------------------------------------------------------------------
# Sidebar: Battery + Data
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Battery Parameters")

    cap = st.number_input("Capacity (MWh)", value=20.0, min_value=0.1)
    soc_init = st.slider("Initial SoC (fraction)", 0.0, 1.0, 0.5, 0.01)
    soc_min = st.slider("Min SoC (fraction)", 0.0, 1.0, 0.1, 0.01)
    soc_max = st.slider("Max SoC (fraction)", 0.0, 1.0, 0.9, 0.01)

    cmax = st.number_input("Max charge power cmax_MW", value=6.0, min_value=0.1)
    dmax = st.number_input("Max discharge power dmax_MW", value=6.0, min_value=0.1)

    eta_c = st.slider("Charge efficiency η_c", 0.5, 1.0, 0.95, 0.01)
    eta_d = st.slider("Discharge efficiency η_d", 0.5, 1.0, 0.95, 0.01)

    soc_target = st.slider("Target SoC at end of day", 0.0, 1.0, 0.5, 0.01)
    dt_hours = st.selectbox("Timestep size (hours)", [0.25, 0.5, 1.0], index=2)

    st.markdown("---")
    st.header("📎 Upload Day Data")

    uploaded_file = st.file_uploader(
        "Upload CSV with price and demand for one day",
        type=["csv"],
        help="Include columns like 'price' / 'prices' and 'demand' / 'demand_MW'.",
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.dataframe(df.head())

        price_col = st.selectbox(
            "Price column",
            df.columns,
            index=0,
        )
        demand_col = st.selectbox(
            "Demand column",
            df.columns,
            index=min(1, len(df.columns) - 1),
        )

        prices = df[price_col].astype(float).tolist()
        demand = df[demand_col].astype(float).tolist()
    else:
        st.info("No CSV uploaded. Using a simple synthetic 24h profile.")
        T = 24
        prices = [0.12] * 6 + [0.15] * 6 + [0.22] * 6 + [0.16] * 6
        demand = [1.0] * T

    allow_export = False

    battery_params = BatteryParams(
        capacity_MWh=cap,
        soc_init=soc_init,
        soc_min=soc_min,
        soc_max=soc_max,
        cmax_MW=cmax,
        dmax_MW=dmax,
        eta_c=eta_c,
        eta_d=eta_d,
        soc_target=soc_target,
    )

    day_inputs = DayInputs(
        prices_buy=prices,
        prices_sell=prices,
        demand_MW=demand,
        allow_export=allow_export,
        dt_hours=dt_hours,
    )

    st.markdown("---")
    run_button = st.button("🚀 Run MILP Optimization")


# ---------------------------------------------------------------------
# Main panel: Chat + Results
# ---------------------------------------------------------------------

col_chat, col_results = st.columns([2, 3])

with col_chat:
    st.subheader("💬 Chat with the Arbitrage Agent")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_prompt = st.chat_input(
        "Ask about arbitrage or type 'run' to optimize with current settings…"
    )

    trigger_run = False

    if user_prompt:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        if "run" in user_prompt.lower():
            trigger_run = True
        else:
            # Use LLM for general Q&A about arbitrage + (optional) last run context
            context_parts = []
            if st.session_state.last_solve_response is not None:
                sr = st.session_state.last_solve_response
                context_parts.append(
                    f"Last objective cost: {sr.objective_cost:.4f}, "
                    f"status: {sr.status}."
                    f"soc schedule: {sr.soc}."
                    f"decisions schedule: {sr.decision}."
                )
            context = "\n".join(context_parts)
            answer = answer_chat(user_prompt, context=context)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()

    if run_button:
        trigger_run = True

with col_results:
    st.subheader("📊 Optimization Results & Plot")

    # If optimization is requested
    if trigger_run:
        with st.spinner("Running MILP optimization via MCP server…"):
            try:
                solve_response = run_milp_solver(battery_params, day_inputs)
                solve_request = SolveRequest(
                    battery=battery_params,
                    day=day_inputs,
                    solver="MILP",
                    solver_opts=None,
                )

                st.session_state.last_solve_response = solve_response
                st.session_state.last_solve_request = solve_request

                # Now run visualization
                plot_response = run_plot(
                    solve_request=solve_request,
                    solve_response=solve_response,
                    out_path="./plots/daily_battery_schedule.png",
                )
                st.session_state.last_plot = plot_response

                summary = (
                    f"Optimization status: **{solve_response.status}**  \n"
                    f"Objective cost: **{solve_response.objective_cost:.4f}**"
                )

                st.markdown(summary)

                if plot_response.image_path and os.path.exists(plot_response.image_path):
                    st.image(plot_response.image_path, caption=plot_response.caption)
                else:
                    st.warning(
                        f"Plot generated, but file not found at `{plot_response.image_path}`."
                    )

                # Also drop a short assistant message into the chat history
                auto_msg = (
                    "I’ve run the MILP optimizer with your current battery & data. "
                    f"Status: {solve_response.status}, "
                    f"objective cost: {solve_response.objective_cost:.4f}. "
                    "See the plot and schedule summary in the results panel."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": auto_msg}
                )

                st.rerun()

            except Exception as e:
                st.error(f"Error while running optimization: {e}")

    # If we already have a result from previous run, show it
    elif st.session_state.last_solve_response is not None:
        sr = st.session_state.last_solve_response
        st.markdown(
            f"Last run – status: **{sr.status}**, "
            f"objective cost: **{sr.objective_cost:.4f}**"
        )
        if st.session_state.last_plot is not None:
            pr = st.session_state.last_plot
            if pr.image_path and os.path.exists(pr.image_path):
                st.image(pr.image_path, caption=pr.caption)
            else:
                st.info("Plot info available but image file not found.")
    else:
        st.info("No optimization run yet. Upload data & click **Run MILP Optimization**, or type `run` in chat.")

