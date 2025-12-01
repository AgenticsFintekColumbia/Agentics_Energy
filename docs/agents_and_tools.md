# Agents & Tools

This page summarizes the main agents and MCP tools in AlphaSearch.

![Agentic Energy Architecture](assets/EnArb_SeqFlowAgentic.png)
---

## 1. Forecasting Agents

Typical forecasting options include:

- **Random Forest**
- (Extensible) LSTM, transformer-based models, TimeGPT, etc.

The forecast agent:

1. Picks a model (optionally with a recommendation)
2. Generates next-day price trajectories
3. Annotates low/high price zones for arbitrage opportunities

---

## 2. Optimization Agents

Available controllers include:

- **MILP Oracle**
  - Exact optimization with all constraints  
  - Best for benchmarking and rigorous evaluation  

- **Heuristic (time-based / quantile-based)**
  - Simple rules: charge at night, discharge at peak  
  - Fast and interpretable  

- **RL Agent**
  - Learns a policy from simulated or historical experience  
  - Good for complex, changing environments  

- **LLM-based Controllers (Gemini / Ollama)**
  - Use language-model reasoning with embedded qualitative rules  

---

## 3. Reasoning Agent

The reasoning agent answers questions like:

- “Why did you discharge at hour 10?”
- “What is driving profits here?”
- “How sensitive is this schedule to small forecast errors?”

It uses:

- The last forecast and schedule  
- Confined plots (prices, power, SoC)  
- Domain-specific heuristics

to generate short explanations in natural language.

---

## 4. MCP Tools

AlphaSearch exposes many operations as MCP tools, including:

- `milp_solve` – solve the daily MILP arbitrage problem  
- `plot_schedule` – produce candlestick + SoC plots and animations  
- `forecast_prices` – generate price forecasts  
- Reasoning tools to summarize and explain behavior

These tools can be orchestrated by CrewAI, called from notebooks, or integrated
into other agentic workflows.
