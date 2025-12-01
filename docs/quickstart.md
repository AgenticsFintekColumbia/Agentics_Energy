# Quickstart

This page shows how to get AlphaSearch running on your machine.

---

## 1. Clone the Repository

```bash
git clone https://github.com/AgenticsFintekColumbia/Agentics_Energy.git
cd Agentics_Energy
```

## 2. Create and Activate Environment  

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

## 3. Install Packages in Editable Mode

```bash
pip install --upgrade pip
pip install -e ./agentics
pip install -e ./agentic_energy
```

## 4. Set Environment Variables

```bash
export GEMINI_API_KEY="your_key"
export GEMINI_MODEL_ID="gemini/gemini-2.0-flash"

export GRB_WLSACCESSID="your_wls_id"
export GRB_WLSSECRET="your_wls_secret"
export GRB_LICENSEID="your_license"
```

## 5. Run the Streamlit App  🎨⚡

```bash
streamlit run app.py --server.port=8501
```

Then open : 
```text
http://localhost:8501
```

You should see the Agentic Battery Arbitrage Assistant:
- Left panel: battery parameters
- Middle panel: chat with the arbitrage agent
- Right panel: data, forecasts, and optimization results

Next: see the [Architecture](architecture.md) page for how the pieces fit together.