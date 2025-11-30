# AlphaSearch :Agentic Energy Arbitrage with Battery Storage ⚡🤖

This repository implements an **agentic AI pipeline** for daily battery arbitrage using:

- A **MILP oracle** (via `agentic_energy`) for optimal charge/discharge decisions  
- **CrewAI agents** to orchestrate optimization, reasoning, and visualization  
- **MCP tools** that expose the MILP, Heuristics, Ollama, Gemini LLMs and plotting logic as tools callable by LLMs  
- A suite of **Jupyter notebooks** for testing, debugging, and experimentation  

---

## 🧱 High-Level Architecture

![Agentic Energy Architecture](plots/EnArb_SeqFlowAgentic.png)

<!-- The system is organized into **four layers**:

### 1️⃣ Data Layer

- **Storages**: battery specs, historical operations, and other asset data.  
- **Markets**: price time series and related financial signals.  
- **Consumer**: demand/load profiles and consumption patterns.  
- **Sources**:
  - Online datasheets (static URLs) for technical parameters.
  - Local storage (databases, CSV files) for historical data.
- **Config**: central place for capacity, SoC bounds, efficiency, and other model parameters.

### 2️⃣ Forecast Layer

Takes raw data and produces **price and demand trajectories** used by the optimizer.

- Models:
  - **Gaussian noise** and simple baselines.
  - **Random Forest (binned)** regressors for structured forecasts.
  - **LSTM** sequence models for temporal dynamics.
  - **TimeGPT / LLM-based forecaster** as a “Novel LLM as a Forecaster”.
- Outputs:
  - Forecasted buying/selling prices.
  - Forecasted demand/load.
  - Uncertainty that can be fed into robust optimization or RL.

### 3️⃣ Optimizer Layer

Given forecasts and constraints, this layer **computes the battery schedule**:

- **Mixed Integer Linear Programming (MILP)**:  
  - Hard constraints (SoC, power limits, efficiency, export rules).  
  - Objective: minimize cost / maximize arbitrage profit.
- **Reinforcement Learning (RL)**:
  - Policy-based control for longer horizons or different reward shaping.
- **Heuristics**:
  - Simple rules or analytical baselines for comparison or warm-starts.
- **Novel LLM as an Optimizer**:
  - LLM that emulates the MILP solution (via QLoRA fine-tuning) and proposes feasible charge/discharge schedules.

### 4️⃣ Reasoning Layer – Interactive Analyst & Explainer

A top-level **agentic reasoning layer** that:

- Orchestrates calls to:
  - Data and forecast routines.
  - MILP/RL/heuristic optimizers.
  - Visualization tools.
- Explains:
  - Why a particular schedule was chosen.
  - How SoC evolves over time.
  - How profits, imports, and exports break down.
- Powered by CrewAI + MCP tools (milp solver, reasoning tools, viz tools).

--- -->

## 1. 🚀 Deployment Information

- **Project Slug:** `agenticsenergy-streamlit`  
- **Main Entry File:** `app.py`  
- **Default Port:** `8501`  
- **Environment Name (Conda/Mamba):** `agentics`  
- **Base Image Used in Docker:** `mambaorg/micromamba:1.5.8`  


---
## 2. 🔧 Environment Variables Required

These variables must be set for **both local and Docker deployments**.


#### **LLM / Gemini Configuration**

| Variable           | Description                |
|-------------------|----------------------------|
| `GEMINI_API_KEY`  | Your Gemini API Key        |
| `GEMINI_MODEL_ID` | Default: `gemini/gemini-2.0-flash` |

#### **Gurobi Web License Service (WLS)**

| Variable           | Description              |
|-------------------|--------------------------|
| `GRB_WLSACCESSID` | Gurobi WLS Access ID     |
| `GRB_WLSSECRET`   | Gurobi WLS Secret        |
| `GRB_LICENSEID`   | Numeric License ID       |

> **Note:** You do **not** need a `gurobi.lic` file when using WLS.

---

## 3. 🛠 Local Development Setup

### 3.1 Create and Activate Environment  

**Using Conda:**

```bash
conda env create -f environment.yml -n agentics
conda activate agentics
```

or **using micromamba:**

```bash
micromamba create -n agentics -f environment.yml
micromamba activate agentics
```

### 3.2 Install System Tools

For Ubuntu/macOS:
```bash
sudo apt-get update
sudo apt-get install -y git build-essential
```

### 3.3 Install the Project in Editable Mode

```bash
pip install --upgrade pip
pip install -e ./agentics
pip install -e ./agentic_energy
```

### 3.4 Set Environment Variables

```bash
export GEMINI_API_KEY="your_key"
export GEMINI_MODEL_ID="gemini/gemini-2.0-flash"

export GRB_WLSACCESSID="your_wls_id"
export GRB_WLSSECRET="your_wls_secret"
export GRB_LICENSEID="your_license"
```

### 3.5 Run the Streamlit App 🎨⚡ 

```bash
streamlit run app.py --server.port=8501
```

---

## 4. 🐳 Docker Deployment

### 4.1 📦 Build the Image

```bash
docker build -t agenticsenergy-streamlit .
```

### 4.2 ▶️ Run the Container

```bash
docker run --rm \
  -p 8501:8501 \
  -e GRB_WLSACCESSID="your-wls-id" \
  -e GRB_WLSSECRET="your-wls-secret" \
  -e GRB_LICENSEID="your-license-id" \
  -e GEMINI_API_KEY="your-gemini-api-key" \
  -e GEMINI_MODEL_ID="gemini/gemini-2.0-flash" \
  agenticsenergy-streamlit
```

Then open : 
```bash
http://localhost:8501
```


<!-- #### 2. Create and Activate a Virtual Environment

You can use either **conda** or **venv**. Pick one.

### Option A: Using `conda` (recommended)

```bash
conda create -n agentics python=3.11 -y
conda activate agentics
```

### Option B: Using venv
``` bash
python -m venv agentics
source agentics/bin/activate      # on macOS / Linux
# .\agentics\Scripts\activate     # on Windows
```
Once the environment is active, you’ll install the two local packages: agentics and agentic_energy.

---

## 3. Install the `agentics` Package

From the root of this repo, go into the `agentics` folder and install it in editable mode:
``` bash
cd agentics
pip install -e .
```

This makes the `agentics` Python package available in your environment while still pointing to the local source code (so code changes are immediately reflected).

---

## 4. Install the agentic_energy Package

Next, install the core energy optimization package:
``` bash
cd ../agentic_energy
pip install -e .
```
Again, -e installs in editable mode, ideal for active development.

At this point, both:
- `agentics`
- `agentic_energy`
should be importable in Python.

You can quickly verify:
``` bash
python -c "import agentics, agentic_energy; print('OK:', agentics.__name__, agentic_energy.__name__)"
``` --> 

---

## 5. Repository Structure (High-Level)

A typical layout looks like:
``` text
Agentics_for_EnergyArbitrage_Battery/
├── agentics/                 # Agent orchestration, CrewAI/MCP integration
├── agentic_energy/           # MILP models, schemas, data loaders, MCP servers
├── notebooks/                # Jupyter notebooks for experiments & testing
├── battery_agent_crewai.py   # Main entrypoint for the CrewAI battery agent
└── README.md
```

`notebooks/` folder

The `notebooks/` directory contains testable notebooks that let you:
- Run the MILP battery arbitrage logic directly
- Inspect price / SoC trajectories
- Debug & validate the underlying optimization and data-loading
- Prototype new ideas before integrating them into the agentic pipeline

These are a good place to start if you want to understand the core math and optimization with and without the agentic layer on top. 

Following is an example plot which you may see that shows when to charge and discharge your storage based on the price volatility.

![Battery Decisions](plots/Result_ToView.png)


---

## 6. Running the Agentic Battery Orchestrator

Once your environment is set up and both packages are installed:

1. Go back to the project root (if you’re not already there):
``` bash
cd ../   # ensure you're back at Agentics_for_EnergyArbitrage_Battery
```
2. Run the main CrewAI script:
``` bash
python battery_agent_crewai.py
```

What this script does:
1. Starts the MCP servers 
2. Launches the Battery Optimizer agent via CrewAI
3. Orchestrates a full run.

If configured, you’ll see logs such as:

- Available LLM providers (gemini, openai, ollama)
- MCP servers connecting and listing tools
- The CrewAI “Crew Execution Started” banner
- Tool calls like Using Tool: milp_solve and the resulting objective and schedule


## 👨‍💻 Authors & Contributors

- **Millend Roy** 
- **Vlad Pyltsov** 
- **Marcel Ayora I. Mexia** 


## 📄 License

This repository is released under the **MIT License**,  You are free to use, extend, or integrate this work into your own projects, as long as you preserve the original copyright notice.

For full details, see the [`LICENSE`](./LICENSE) file.

## 🤝 Contributing

Contributions are most welcome!

If you want to contribute to **AlphaSearch**, here are the ways you can help:

- Fix bugs, improve documentation, or optimize components  
- Add new forecasting models (RF, LSTM, transformers, TimeGPT, etc.)  
- Extend MILP, RL, or heuristic optimizers  
- Improve the Streamlit UI or plotting  
- Add new MCP tools or agentic reasoning modules

To contribute:

1. Fork the repo  
2. Create a feature branch  
3. Submit a pull request

We will review PRs promptly and collaborate to improve the project together.

## 📚 Citation

If you use AlphaSearch in academic work or publications, please cite: 
```
pending
```
