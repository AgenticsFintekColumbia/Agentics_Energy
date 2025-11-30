import os
import sys
import warnings
from textwrap import dedent
from typing import Any, Iterable, List, Optional, Dict
import json

import agentic_energy
import agentics

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

# LLM provider from your agentics stack
from agentics.core.llm_connections import get_llm_provider

# Agentic energy schemas
from agentic_energy.schemas import (
    BatteryParams,
    DayInputs,
    SolveRequest,
    SolveResponse,
    ReasoningRequest,
    ReasoningResponse,
    PlotRequest,
    PlotResponse,
    ForecastRequest,
    ForecastResponse,
    ForecastFeatures,
)

from datetime import datetime, timedelta


# -----------------------------------------------------------------------------
# Environment / path setup
# -----------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv(find_dotenv())
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


# -----------------------------------------------------------------------------
# Helper to find a tool by name (sanity check)
# -----------------------------------------------------------------------------

def get_tool(tools: Iterable[Any], name: str) -> Any:
    """Return the MCP tool object with a given name."""
    for t in tools:
        if t.name == name:
            return t
    raise RuntimeError(f"Tool {name!r} not found. Available: {[t.name for t in tools]}")


# -----------------------------------------------------------------------------
# Pydantic models for task-level outputs (like your WebSearchReport example)
# -----------------------------------------------------------------------------

class OptimizationResult(BaseModel):
    """Task-level output for the optimizer agent."""
    solve_request: Dict[str, Any] = Field(
        ..., description="SolveRequest sent to milp_solve"
    )
    solve_response: SolveResponse = Field(
        ..., description="SolveResponse returned by milp_solve"
    )


class VisualizationResult(BaseModel):
    """Task-level output for the visualization agent."""
    plot: PlotResponse = Field(
        ..., description="Price vs SoC plot (path + caption)"
    )


class ReasoningResult(BaseModel):
    """Task-level output for the reasoning agent."""
    explanations: List[ReasoningResponse] = Field(
        ..., description="List of reasoning explanations for one or more timesteps"
    )


class FinalReport(BaseModel):
    """Final report aggregated from all agents."""
    markdown_report: str = Field(
        ..., description="Full markdown report summarizing instance, MILP, viz, and reasoning."
    )


class ForecastResult(BaseModel):
    """Task-level output for the forecast agent."""
    forecast_response: ForecastResponse = Field(
        ..., description="ForecastResponse returned by forecast_predict"
    )


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def main():
    # 1) Define MCP server commands (must match your existing servers)
    forecast_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.forecast.forecast_mcp_server"],
        env=os.environ,
    )
        
    milp_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.milp.milp_mcp_server"],
        env=os.environ,
    )

    reasoning_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.reasoning.reasoning_server"],
        env=os.environ,
    )

    viz_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.visualization.visualization_mcp_server"],
        env=os.environ,
    )



    # 2) Example daily instance (replace with your actual/forecast data)
    T = 24
    prices = [0.12] * 6 + [0.15] * 6 + [0.22] * 6 + [0.16] * 6
    demand = [1.0] * T

    battery = BatteryParams(
        capacity_MWh=20.0,
        soc_init=0.5,
        soc_min=0.10,
        soc_max=0.90,
        cmax_MW=6.0,
        dmax_MW=6.0,
        eta_c=0.95,
        eta_d=0.95,
        soc_target=0.5,
    )
    battery_json = battery.model_dump()

    day = DayInputs(
        prices_buy=prices,
        demand_MW=demand,
        prices_sell=prices,
        allow_export=False,
        dt_hours=1.0,
    )
    day_json = day.model_dump()

    # 3) Start all MCP servers and expose their tools to CrewAI
    print("🚀 Starting MCP servers (MILP, Reasoning, Viz) …")

    with MCPServerAdapter(milp_params) as milp_tools, \
         MCPServerAdapter(reasoning_params) as reason_tools, \
         MCPServerAdapter(viz_params) as viz_tools,\
         MCPServerAdapter(forecast_params) as forecast_tools:

        print("✅ Connected to servers.")
        print("   Forecast tools :", [t.name for t in forecast_tools])
        print("   MILP tools     :", [t.name for t in milp_tools])
        print("   Reasoning tools:", [t.name for t in reason_tools])
        print("   Viz tools      :", [t.name for t in viz_tools])

        # Sanity checks that expected tools exist
        forecast_tool = get_tool(forecast_tools, "forecast_predict")
        milp_tool = get_tool(milp_tools, "milp_solve")
        reasoning_tool = get_tool(reason_tools, "reasoning_explain")
        viz_tool = get_tool(viz_tools, "plot_price_soc")

        forecast_call_fn = getattr(forecast_tool, "call", None) or getattr(forecast_tool, "run", None) or getattr(forecast_tool, "__call__", None)
        if forecast_call_fn is None:
                raise RuntimeError("Tool has no callable interface")
        milp_call_fn = getattr(milp_tool, "call", None) or getattr(milp_tool, "run", None) or getattr(milp_tool, "__call__", None)
        if milp_call_fn is None:
                raise RuntimeError("Tool has no callable interface")
        reasoning_call_fn = getattr(reasoning_tool, "call", None) or getattr(reasoning_tool, "run", None) or getattr(reasoning_tool, "__call__", None)
        if reasoning_call_fn is None:
                raise RuntimeError("Tool has no callable interface")
        viz_call_fn = getattr(viz_tool, "call", None) or getattr(viz_tool, "run", None) or getattr(viz_tool, "__call__", None)
        if viz_call_fn is None:
                raise RuntimeError("Tool has no callable interface")
        
        

        
        # # Separate tool sets per agent
        # optimizer_tools = list(milp_tool)
        # reasoning_tools_list = list(reasoning_tool)
        # viz_tools_list = list(viz_tool)

        # LLM
        llm = get_llm_provider("gemini")  # or "openai", etc.

        # ---------------------------------------------------------------------
        # Agents
        # ---------------------------------------------------------------------

        optimizer_agent = Agent(
            role="Battery optimizer",
            goal=(
                "Take SolveRequest directly from the input, "
                "and call milp_solve to obtain an optimal daily dispatch schedule as SolveResponse."
            ),
            backstory=(
                "You are an expert in MILP-based battery arbitrage. "
                "You ONLY use the MILP tool: milp_solve, to compute optimal schedules."
            ),
            tools=[milp_tool],
            llm=llm,
            verbose=True,
            reasoning=False,
            memory=True,
        )

        viz_agent = Agent(
            role="Battery visualization agent",
            goal=(
                "Given a SolveRequest and SolveResponse, prepare the schema PlotRequest which takes both of them"
                " and call plot_price_soc to generate a price vs SoC plot and return the plot path and caption from the Schema PlotResponse."
            ),
            backstory="You specialize in plotting prices vs SoC over time. and you only use the mentioned tool : plot_price_soc",
            tools=[viz_tool],
            llm=llm,
            verbose=True,
            reasoning=False,
            memory=True,
        )

        reasoning_agent = Agent(
            role="Battery reasoning agent",
            goal=(
                "Given a SolveRequest and SolveResponse, form the schema ReasoningRequest,"
                " and call reasoning_explain tool EXACTLY ONCE to explain interesting decisions in the form of the Schema ReasoningResponse."
            ),
            backstory="You explain charge/discharge decisions using the reasoning MCP tool : reasoning_explain.",
            tools=[reasoning_tool],
            llm=llm,
            verbose=True,
            reasoning=True,
            memory=True,
        )

        report_agent = Agent(
            role="Battery arbitrage reporter",
            goal="Combine optimization, visualization, and reasoning outputs into a markdown report.",
            backstory="You summarize technical results into clear reports for humans, also include visualizations.",
            tools=[],  # no tools, pure LLM summarization
            llm=llm,
            verbose=True,
            reasoning=False,
            memory=True,
        )

        forecast_agent = Agent(
            role="Energy Forecast Specialist",
            goal=(
                "Generate accurate price and consumption forecasts using ML models "
                "(RF or LSTM) and compare predictions against actual values to compute error metrics."
            ),
            backstory=(
                "You are an expert in energy market forecasting with deep knowledge "
                "of machine learning models (Random Forest and LSTM). You analyze historical patterns, "
                "weather data, and temporal features to predict future prices and consumption. "
                "You ONLY use the forecast_predict tool."
            ),
            tools=[forecast_tool],
            llm=llm,
            verbose=True,
            reasoning=False,
            memory=True,
        )

        # ---------------------------------------------------------------------
        # Tasks
        # ---------------------------------------------------------------------

        # 1) Optimization Task → OptimizationResult
        optimize_task = Task(
            description=dedent(
                f"""
                You are given the following inputs:
                You are solving a daily battery scheduling optimization problem using actual market data and constraint satisfaction.

                    You are provided with actual market data:

                    ACTUAL INPUTS (for ex-post evaluation):
                        - Realized buying prices: {day_json["prices_buy"]}  (array of length T)
                        - Realized selling prices: {day_json["prices_sell"]}  (array of length T)
                        - Realized demand: {day_json["demand_MW"]}  (array of length T)

                    BATTERY PARAMETERS:
                        - capacity_MWh: {battery_json["capacity_MWh"]}
                        - charge/discharge limits: cmax_MW={battery_json["cmax_MW"]}, dmax_MW={battery_json["dmax_MW"]}
                        - efficiencies: eta_c={battery_json["eta_c"]}, eta_d={battery_json["eta_d"]}
                        - SoC bounds: {battery_json["soc_min"]} ≤ SoC ≤ {battery_json["soc_max"]}
                        - initial SoC: soc_init={battery_json["soc_init"]}
                        - target SoC: soc_target={battery_json["soc_target"]}

                    HORIZON:
                        - Number of timesteps: T = {len(day_json["prices_buy"])}
                        - Duration per step: dt_hours = {day_json["dt_hours"]}
                        - Export allowed: {day_json["allow_export"]}


                Your job:

                1. Use the SolveRequest schema provided with:
                     - battery = {battery_json}
                     - day = {day_json}
                     - solver = "MILP"
                     - solver_opts = null

                2. Call the `milp_solve` MCP tool.

                   IMPORTANT: 
                    - The tool schema requires the SolveRequest inside a top-level "solverequest" field. 
                    - Use valid JSON (double quotes, `null` for missing values).
                    - Do NOT change field names or values.
                    - Do NOT write the string "None" anywhere; use `null` instead if needed.

                3. Take the result from milp_solve as a SolveResponse.

                4. Return an OptimizationResult whose fields are:
                     - solve_request:  <the SAME solve_request JSON you were given>
                     - solve_response: <the tool output JSON from milp_solve>
                """
            ),
            expected_output="A structured OptimizationResult instance which contains solve_request and solve_response so that downstream tasks can parse it.",
            agent=optimizer_agent,
            output_pydantic=OptimizationResult,  # like WebSearchReport
            markdown=False,
        )

        # 2) Visualization Task → VisualizationResult
        viz_task = Task(
            description=dedent(
                """
                You are the visualization agent.

                You are given the output of the optimization task in the form of OptimizationResult.

                This is a JSON-serialized OptimizationResult with fields:
                  - solve_request
                  - solve_response

                Your steps:

                1. Parse the previous output to recover and create a PlotRequest schema with fields:
                     - solve_request
                     - solve_response

                2. Call the `plot_price_soc` MCP tool with JSON of the form:
                   {
                     "plotrequest": {
                       "solve_request": solve_request,
                       "solve_response": solve_response,
                       "out_path": "./plots/daily_battery_schedule.png"
                     }
                   }

                3. The tool returns a PlotResponse (with image_path and caption).

                4. Wrap this in a VisualizationResult object:

                   {
                     "plot": <PlotResponse>
                   }

                Return that VisualizationResult.
                """
            ),
            expected_output="A VisualizationResult instance.",
            agent=viz_agent,
            context=[optimize_task],
            output_pydantic=VisualizationResult,
            markdown=False,
        )

        # 3) Reasoning Task → ReasoningResult
        reasoning_task = Task(
            description=dedent(
                """
                You are the reasoning agent.

                You are given the output of the optimization task in the form of OptimizationResult.

                This is a JSON-serialized OptimizationResult with fields:
                  - solve_request
                  - solve_response

                Your steps:

                1. Parse the previous output to recover solve_request and solve_response.

                2. Choose at ONE interesting timestamp index (0-based) to explain,
                   for example a high-price or high-discharge hour.
                
                3. Choose a context window of some interesting timesteps, so that you can provide meaningful explanations.

                4. Call the `reasoning_explain` MCP tool for ONE such index at a time, with JSON of the form ReasoningRequest schema:

                   {
                     "reasoningrequest": {
                       "solve_request": solve_request,
                       "solve_response": solve_response,
                       "timestamp_index": <chosen_index>,
                       "context_window": <chosen_context_window>
                     }
                   }

                5. Collect the resulting ReasoningResponse objects into a list,
                   and return a ReasoningResult:

                   {
                     "explanations": [ <ReasoningResponse>, ... ]
                   }
                """
            ),
            expected_output="A ReasoningResult instance.",
            agent=reasoning_agent,
            context=[optimize_task],
            output_pydantic=ReasoningResult,
            markdown=False,
        )

        # 4) Final Report Task → FinalReport
        report_task = Task(
            description=dedent(
                """
                You are the reporting agent.

                You are given the outputs of three previous tasks in the respective schema formats:

                - OptimizationResult
                - VisualizationResult
                - ReasoningResult
                - ForecastingResult

                These objects have the following shapes:

                OptimizationResult:
                  - solve_request (SolveRequest)
                  - solve_response (SolveResponse)

                VisualizationResult:
                  - plot (PlotResponse with image_path and caption)

                ReasoningResult:
                  - explanations (list[ReasoningResponse])

                Your job:

                1. Use ONLY the information present in these objects
                   (do not invent new numbers or paths).

                2. Produce a markdown report that:
                   - Briefly describes the instance (battery, prices, demand)
                     based on solve_request.
                   - Reports the MILP objective cost from solve_response.objective_cost.
                   - Embeds the plot using:

                        ![Battery schedule](<image_path>)

                     where image_path is plot.image_path.
                   - Summarizes the charge/discharge pattern qualitatively, using
                     fields in solve_response (e.g., soc, charge_MW, discharge_MW).
                   - Includes one or more reasoning explanations from the explanations
                     list, mentioning key_factors and confidence.

                3. Return a FinalReport object with:

                   {
                     "markdown_report": "<your full markdown report here>"
                   }
                """
            ),
            expected_output="A FinalReport instance.",
            agent=report_agent,
            context=[optimize_task, viz_task, reasoning_task],
            output_pydantic=FinalReport,
            markdown=False,
        )

# 4) Forecast Task → ForecastResult
        forecast_task = Task(
            description=dedent(
                """
                You are the forecast agent.

                You are given a ForecastRequest with fields:
                  - target (str): "prices" or "consumption"
                  - model_type (str): "RF" or "LSTM"
                  - features (List[ForecastFeatures]): weather and temporal features
                  - timestamps (List[str]): ISO format timestamps

                Your steps:

                1. Call the `forecast_predict` MCP tool with JSON of the form:
                   {
                     "args": {
                       "target": <target>,
                       "model_type": <model_type>,
                       "features": <list of ForecastFeatures dicts>,
                       "timestamps": <list of timestamp strings>
                     }
                   }

                2. The tool returns a ForecastResponse with:
                   - predictions: list of floats

                3. Return a ForecastResult:
                   {
                     "forecast_response": <ForecastResponse>,
                   }
                """
            ),
            expected_output="A ForecastResult instance with forecast predictions and error metrics.",
            agent=forecast_agent,
            output_pydantic=ForecastResult,
            markdown=False,
        )

        # ---------------------------------------------------------------------
        # Crew and kickoff
        # ---------------------------------------------------------------------

        crew = Crew(
            agents=[optimizer_agent, viz_agent, reasoning_agent, report_agent, forecast_agent],
            tasks=[optimize_task, viz_task, reasoning_task, report_task, forecast_task],
            verbose=True,
        )

        inputs = SolveRequest(
            battery=battery,
            day=day,
            solver="MILP",
            solver_opts=None,
        )

        solve_request_json = json.dumps(inputs.model_dump(exclude_none=True), indent=2)
        print("\n🧠 Launching Crew…\n")
        result = crew.kickoff(inputs={"solve_request": solve_request_json})

        # Final result is from report_task → FinalReport
        try:
            from IPython.display import Markdown, display  # type: ignore

            if hasattr(result, "pydantic") and result.pydantic is not None:
                final: FinalReport = result.pydantic  # type: ignore
                display(Markdown(final.markdown_report))
            else:
                # Fall back to raw text
                display(Markdown(result.raw if hasattr(result, "raw") else str(result)))
        except Exception:
            print("\n=========== FINAL REPORT ===========\n")
            if hasattr(result, "pydantic") and result.pydantic is not None:
                final: FinalReport = result.pydantic  # type: ignore
                print(final.markdown_report)
            else:
                print(result.raw if hasattr(result, "raw") else result)
            print("\n====================================\n")


if __name__ == "__main__":
    main()
