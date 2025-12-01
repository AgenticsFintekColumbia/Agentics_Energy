# AlphaSearch: Agentic Energy Arbitrage with Battery Storage ⚡

AlphaSearch is an **agentic AI pipeline** for daily battery arbitrage in
wholesale electricity markets.

It combines:

- A **MILP oracle** (via `agentic_energy`) for exact charge/discharge schedules  
- **RL and heuristic controllers** for adaptive operation  
- **CrewAI agents + MCP tools** for orchestration, reasoning, and visualization  
- A **Streamlit UI** for interactive exploration and “chat with your battery” style workflows

---

## Core Idea

You bring:

- Historical or live price data  
- A storage asset (capacity, power limits, efficiencies)  

AlphaSearch:

1. **Forecasts prices** for the next day  
2. **Chooses an optimization agent** (MILP, heuristics, RL, or LLM-based)  
3. **Generates a schedule** of when to charge, discharge, or stay idle  
4. **Explains decisions** with plots and natural-language reasoning

You interact with everything through a simple chat box and a results panel.

---

## Who Is This For?

- Energy & power systems researchers  
- Quant / trading / storage optimization teams  
- People interested in **agentic AI** applied to real physical systems

If you want to understand the repo structure and launch the app, start with the
[Quickstart](quickstart.md).
