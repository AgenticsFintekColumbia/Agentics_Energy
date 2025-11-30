# agentic_energy_app/app.py

import os
import datetime
from typing import List, Optional
import time
import pandas as pd
import streamlit as st

from agentic_energy.schemas import (
    BatteryParams,
    DayInputs,
    SolveRequest,
    SolveResponse,
    PlotResponse,
)

from agentic_energy.mcp_clients import (
    run_milp_solver,
    run_heuristic,
    run_rl_agent,
    run_llm_agent,
    run_schedule_animation,
    run_explanation_plot,
    run_reasoning_tool,
)
from agentic_energy.data_utils import run_forecast_step
from agentic_energy.llm_intent import ChatIntent, classify_intent, answer_generic_qa

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------- Streamlit page config ----------

st.set_page_config(
    page_title="Battery Arbitrage Agent",
    layout="wide",
)

st.title("🔋 Agentic Battery Arbitrage Assistant")

st.markdown(
    """
This app lets you:

1. Configure **battery parameters**  
2. Pull **actual & forecast prices** directly via `EnergyDataLoader`  
3. Use **visualization MCP** for price forecast, animated schedules, and explanation plots  
4. Use **optimization agents** (MILP / heuristics / RL / LLM) for schedules  
5. Use a **reasoning agent** to explain decisions with plots  
6. Chat generally about **arbitrage and your data**
    """
)

# ---------- Session state ----------

if "messages" not in st.session_state:
    st.session_state.messages: List[dict] = []

if "last_solve_response" not in st.session_state:
    st.session_state.last_solve_response: Optional[SolveResponse] = None

if "last_solve_request" not in st.session_state:
    st.session_state.last_solve_request: Optional[SolveRequest] = None

if "forecast_plot" not in st.session_state:
    st.session_state.forecast_plot: Optional[PlotResponse] = None

if "last_plot" not in st.session_state:
    st.session_state.last_plot: Optional[PlotResponse] = None

if "explanation_plot" not in st.session_state:
    st.session_state.explanation_plot: Optional[PlotResponse] = None

if "day_inputs" not in st.session_state:
    st.session_state.day_inputs: Optional[DayInputs] = None

if "actual_df" not in st.session_state:
    st.session_state.actual_df: Optional[pd.DataFrame] = None

if "forecast_df" not in st.session_state:
    st.session_state.forecast_df: Optional[pd.DataFrame] = None

if "pipeline_stage" not in st.session_state:
    # idle / need_forecast_choice / running_forecast /
    # need_optimizer_choice / running_optimizer / done
    st.session_state.pipeline_stage = "idle"

if "chosen_forecast_model" not in st.session_state:
    st.session_state.chosen_forecast_model = None

if "chosen_optimizer" not in st.session_state:
    st.session_state.chosen_optimizer = None

if "selected_region" not in st.session_state:
    st.session_state.selected_region = "ITALY"

if "selected_date_str" not in st.session_state:
    st.session_state.selected_date_str = "2018-01-01"

# if "run_history" not in st.session_state:
#     # each item: dict with date, region, plots, solve_response, etc.
#     st.session_state.run_history = []


# ---------- Sidebar: battery + data selection ----------

with st.sidebar:
    st.header("⚙️ Battery Parameters")

    cap = st.number_input("Capacity (MWh)", value=20.0, min_value=0.1)
    soc_init = st.slider("Initial SoC (fraction)", 0.0, 1.0, 0.5, 0.01)
    soc_min = st.slider("Min SoC (fraction)", 0.0, 1.0, 0.0, 0.01)
    soc_max = st.slider("Max SoC (fraction)", 0.0, 1.0, 1.0, 0.01)

    cmax = st.number_input("Max charge power cmax_MW", value=5.0, min_value=0.1)
    dmax = st.number_input("Max discharge power dmax_MW", value=5.0, min_value=0.1)

    eta_c = st.slider("Charge efficiency η_c", 0.5, 1.0, 0.95, 0.01)
    eta_d = st.slider("Discharge efficiency η_d", 0.5, 1.0, 0.95, 0.01)

    soc_target = st.slider("Target SoC at end of day", 0.0, 1.0, 0.5, 0.01)

    st.markdown("---")
    st.header("🌍 Data Selection")

    region = st.selectbox("Region", ["ITALY"], index=0)
    st.session_state.selected_region = region

    date_val = st.date_input(
        "Date for schedule",
        value=datetime.date.fromisoformat(st.session_state.selected_date_str),
    )
    st.session_state.selected_date_str = date_val.isoformat()

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


# ---------- Layout: chat + results ----------

col_chat, col_results = st.columns([2, 3])

with col_chat:
    st.subheader("💬 Chat with the Arbitrage Agent")

    # 1) Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    stage = st.session_state.pipeline_stage

    # 2) Pipeline-stage UIs (ABOVE chat input)

    if stage == "need_forecast_choice":
        with st.chat_message("assistant"):
            st.markdown(
                "**Please choose a forecasting model for day-ahead prices:**\n\n"
                "- **Random Forest (RF)**: robust, handles non-linearities well; works "
                "great with moderate data and tabular features (weather, calendar, load).\n"
                "- **LSTM**: sequence model that can capture temporal patterns; can "
                "outperform RF when you have long historical time series and rich temporal structure.\n\n"
                "**Recommendation:**\n"
                "- If you have *structured features but limited data*, start with **RF**.\n"
                "- If you have *long time series and care about dynamics*, try **LSTM**."
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔮 Use RF forecast", key="btn_rf"):
                    st.session_state.chosen_forecast_model = "RF"
                    st.session_state.pipeline_stage = "running_forecast"
                    st.rerun()
            with c2:
                if st.button("📈 Use LSTM forecast", key="btn_lstm"):
                    st.session_state.chosen_forecast_model = "LSTM"
                    st.session_state.pipeline_stage = "running_forecast"
                    st.rerun()

    elif stage == "running_forecast":
        with st.chat_message("assistant"):
            st.markdown(
                f"⏳ Running **{st.session_state.chosen_forecast_model}** forecast for "
                f"**{st.session_state.selected_region}** on "
                f"**{st.session_state.selected_date_str}**…"
            )
        with st.spinner(
            f"Running {st.session_state.chosen_forecast_model} forecasts "
            f"for {st.session_state.selected_region} on {st.session_state.selected_date_str}..."
        ):
            try:
                day_inputs, actual_df, forecast_df, forecast_plot = run_forecast_step(
                    region=st.session_state.selected_region,
                    date_str=st.session_state.selected_date_str,
                    forecast_type=st.session_state.chosen_forecast_model,
                    forecast_plot_path="./plots/price_forecast.png",
                )
                st.session_state.day_inputs = day_inputs
                st.session_state.actual_df = actual_df
                st.session_state.forecast_df = forecast_df
                st.session_state.forecast_plot = forecast_plot

                st.session_state.pipeline_stage = "need_optimizer_choice"

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"✅ Forecast completed using **{st.session_state.chosen_forecast_model}** "
                            "and I’ve generated a price forecast plot.\n\n"
                            # "Now I’m **hitting the Optimization Agent** for the next step.\n"
                            # "Please choose which optimizer you’d like me to use."
                        ),
                    }
                )

            except Exception as e:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Forecasting/dataloader error: {e}"}
                )
                st.session_state.pipeline_stage = "idle"
            st.rerun()

    elif stage == "need_optimizer_choice":
        with st.spinner(f"⚙️ Now, hitting the Optimization Agent for the next step."):
            with st.chat_message("assistant"):
                st.markdown(
                    "**Choose an optimization strategy for the battery schedule:**\n\n"
                    "- **MILP**: exact optimization with all constraints; best when you want a provably optimal baseline.\n"
                    "- **Heuristics (time-based)**: simple rules (e.g., charge at night, discharge at peak); fast and interpretable.\n"
                    "- **Heuristics (quantile-based)**: act only when price is in top/bottom quantiles; good when you care about extreme spreads.\n"
                    "- **RL Agent**: learns a policy from experience; good when environment is complex or changing.\n"
                    "- **Agentics Gemini / Ollama LLM**: language-model-based controller; can embed qualitative rules and explanations,\n"
                    "  but may be less stable than MILP.\n\n"
                    "**Recommendation:**\n"
                    "- Use **MILP** for benchmarking and rigorous evaluation.\n"
                    "- Use **heuristics** when you need something fast, simple, and robust.\n"
                    "- Use **RL/LLM agents** when you want adaptivity or richer, human-like behavior."
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("⚙️ MILP", key="opt_milp"):
                        st.session_state.chosen_optimizer = "MILP"
                        st.session_state.pipeline_stage = "running_optimizer"
                        st.rerun()
                with c2:
                    if st.button("⏱️ Heuristic (time)", key="opt_heur_time"):
                        st.session_state.chosen_optimizer = "heuristic_time"
                        st.session_state.pipeline_stage = "running_optimizer"
                        st.rerun()
                with c3:
                    if st.button("📊 Heuristic (quantile)", key="opt_heur_quant"):
                        st.session_state.chosen_optimizer = "heuristic_quantile"
                        st.session_state.pipeline_stage = "running_optimizer"
                        st.rerun()

                c4, c5, c6 = st.columns(3)
                with c4:
                    if st.button("🤖 RL Agent", key="opt_rl"):
                        st.session_state.chosen_optimizer = "rl"
                        st.session_state.pipeline_stage = "running_optimizer"
                        st.rerun()
                with c5:
                    if st.button("🧠 Agentics Gemini", key="opt_gemini"):
                        st.session_state.chosen_optimizer = "gemini"
                        st.session_state.pipeline_stage = "running_optimizer"
                        st.rerun()
                with c6:
                    if st.button("🖥️ Ollama Agent", key="opt_ollama"):
                        st.session_state.chosen_optimizer = "ollama"
                        st.session_state.pipeline_stage = "running_optimizer"
                        st.rerun()

    elif stage == "running_optimizer":
        with st.chat_message("assistant"):
            st.markdown(
                f"⏳ Running optimizer **{st.session_state.chosen_optimizer}** and "
                "generating the animated schedule…"
            )
        with st.spinner(
            f"Running optimizer: {st.session_state.chosen_optimizer} and generating animated schedule…"
        ):
            # try:
                if st.session_state.day_inputs is None:
                    raise RuntimeError("DayInputs is not set. Did forecasting succeed?")

                day_inputs = st.session_state.day_inputs
                opt = st.session_state.chosen_optimizer

                solve_request = SolveRequest(
                    battery=battery_params,
                    day=day_inputs,
                    solver=st.session_state.chosen_optimizer,
                    solver_opts=None,
                )

                # currently only MILP wired
                if opt == "MILP":
                    solve_response = run_milp_solver(solve_request)
                elif opt == "heuristic_time":
                    solve_response = run_heuristic(solve_request, mode = "time")
                elif opt == "heuristic_quantile":
                    solve_response = run_heuristic(solve_request, mode = "quantile")
                elif opt == "rl":
                    solve_response = run_rl_agent(solve_request, date = st.session_state.selected_date_str)
                elif opt == "gemini":
                    solve_response = run_llm_agent(model_name="gemini", solve_request=solve_request)
                elif opt == "ollama":
                    solve_response = run_llm_agent(model_name="ollama", solve_request=solve_request)
                else:
                    # fall back so app doesn't crash if something is mis-set
                    solve_response = run_milp_solver(solve_request)

                st.session_state.last_solve_response = solve_response
                st.session_state.last_solve_request = solve_request

                anim_plot = run_schedule_animation(
                    solve_request=solve_request,
                    solve_response=solve_response,
                    out_path="./plots/daily_battery_schedule.mp4",
                )
                st.session_state.last_plot = anim_plot

                st.session_state.pipeline_stage = "done"

                # Append this run to history
                # st.session_state.run_history.append(
                #     {
                #         "region": st.session_state.selected_region,
                #         "date": st.session_state.selected_date_str,
                #         "forecast_model": st.session_state.chosen_forecast_model,
                #         "optimizer": st.session_state.chosen_optimizer,
                #         "forecast_plot": st.session_state.forecast_plot,
                #         "anim_plot": anim_plot,
                #         "solve_response": solve_response,
                #         "explanation_plot": st.session_state.explanation_plot,
                #     }
                # )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"✅ Optimization completed with **{st.session_state.chosen_optimizer}**.\n\n"
                            "I’ve generated a schedule and an animation showing how the battery acts throughout the day.\n"
                        ),
                    }
                )
            # except Exception as e:
            #     st.session_state.messages.append(
            #         {"role": "assistant", "content": f"Optimization error: {e}"}
            #     )
            #     st.session_state.pipeline_stage = "idle"
                st.rerun()

    # Examples only in idle state
    if stage == "idle":
        st.markdown(
            """
            **💬 Examples:**  
            • “generate schedules for tomorrow”  
            • “run optimization”  
            • “why is the optimizer doing this?”
            """
        )

    # 3) Chat input at the very bottom
    user_prompt = st.chat_input("Ask something…")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # context for LLM
        context_parts = []
        if st.session_state.last_solve_response is not None:
            sr = st.session_state.last_solve_response
            context_parts.append(
                f"Last objective cost: {sr.objective_cost:.4f}, "
                f"status: {sr.status}. "
                f"SoC schedule: {sr.soc}. "
                f"Decision schedule: {sr.decision}."
            )
        context = "\n".join(context_parts)

        intent: ChatIntent = classify_intent(user_prompt, context=context)

        # ---------- intent routing ----------
        if intent.intent == "start_pipeline":
            # Show a transient spinner instead of a chat bubble
            with st.spinner(
                f"🔍 Hitting the forecasting agent for "
                f"{st.session_state.selected_region} on {st.session_state.selected_date_str}..."
            ):
                time.sleep(3)
                st.session_state.pipeline_stage = "need_forecast_choice"
                st.session_state.chosen_forecast_model = None
                st.session_state.chosen_optimizer = None

            st.rerun()

        elif intent.intent == "reasoning":
            if (
                st.session_state.last_solve_response is not None
                and st.session_state.last_solve_request is not None
            ):
                ts_idx = intent.timestamp_index_asked or 0
                try:
                    explanation = run_reasoning_tool(
                        solve_request=st.session_state.last_solve_request,
                        solve_response=st.session_state.last_solve_response,
                        timestamp_index=ts_idx,
                    )
                    exp_plot = run_explanation_plot(
                        solve_request=st.session_state.last_solve_request,
                        solve_response=st.session_state.last_solve_response,
                        out_path="./plots/arbitrage_explanation.png",
                    )
                    st.session_state.explanation_plot = exp_plot
                    explanation += "\n\n(I’ve also generated an explanation plot in the results panel.)"
                except Exception as e:
                    explanation = f"Reasoning agent error: {e}"
                st.session_state.messages.append(
                    {"role": "assistant", "content": explanation}
                )
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I don’t have a recent optimization run to explain yet. "
                            "Ask me to generate a schedule first."
                        ),
                    }
                )
            st.rerun()

        else:  # generic_qa
            answer = answer_generic_qa(user_prompt, context=context)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()


# ---------- Results column ----------

with col_results:
    st.subheader("📊 Data, Forecast, & Optimization Results")

    # forecast plot
    if st.session_state.forecast_plot is not None:
        prf = st.session_state.forecast_plot
        if prf.image_path and os.path.exists(prf.image_path):
            st.image(prf.image_path, caption=prf.caption)
        else:
            st.info("Forecast plot info available but image file not found.")

    # # quick peek at actual vs forecast
    # if st.session_state.actual_df is not None and st.session_state.forecast_df is not None:
    #     st.markdown(
    #         f"**Actual vs {st.session_state.chosen_forecast_model or 'N/A'} forecast** "
    #         f"for {st.session_state.selected_region} on {st.session_state.selected_date_str}"
    #     )

    # optimization summary + animation
    if st.session_state.last_solve_response is not None:
        sr = st.session_state.last_solve_response
        st.markdown(
            f"**Last run** – status: **{sr.status}**, "
            f"objective cost: **{sr.objective_cost:.4f}**"
        )

        if st.session_state.last_plot is not None:
            pr = st.session_state.last_plot
            if pr.image_path and os.path.exists(pr.image_path):
                if pr.image_path.endswith(".mp4"):
                    st.video(pr.image_path)
                    st.caption(pr.caption)
                elif pr.image_path.endswith(".gif"):
                    st.image(pr.image_path, caption=pr.caption)
                else:
                    st.image(pr.image_path, caption=pr.caption)
            else:
                st.info("Schedule visualization generated but image file not found.")

    # explanation plot
    if st.session_state.explanation_plot is not None:
        exp_plot = st.session_state.explanation_plot
        st.markdown("### 🧠 Arbitrage Explanation Plot")
        if exp_plot.image_path and os.path.exists(exp_plot.image_path):
            st.image(exp_plot.image_path, caption=exp_plot.caption)
        else:
            st.info("Explanation plot info available but image file not found.")

    if st.session_state.last_solve_response is None:
        st.info(
            "No optimization run yet. Ask me to *generate schedules for tomorrow* "
            "to start the agentic pipeline using real Italy data and visualization MCP."
        )
