# FAQ

### 1.  Is Gurobi required?

Yes. The default MILP oracle uses **Gurobi** via the Web License Service (WLS).
You must set:

- `GRB_WLSACCESSID`
- `GRB_WLSSECRET`
- `GRB_LICENSEID`

If you do not have Gurobi access, you can:

- Use heuristic or RL agents, or  
- Replace the MILP backend with another solver (PRs welcome!).


### 2. Which LLMs are supported?

Currently:

- **Gemini** via API
- **Ollama** for local models (if configured)

The LLM is used for:

- Intent classification
- Reasoning explanations
- Generic Q&A


### 3. Can I plug in my own data?

Yes. The data loader can be extended to new markets and datasets. See the
`agentic_energy` package and notebooks for examples of loading Italy/Germany
price data and mapping it into the daily arbitrage problem.


### 4. Is this production-ready?

This is a **research-grade** framework. It is suitable for:

- Experiments
- Prototyping strategies
- Teaching / demos

If you want to use it in production, please review the code carefully, add your
own logging, monitoring, and risk controls.