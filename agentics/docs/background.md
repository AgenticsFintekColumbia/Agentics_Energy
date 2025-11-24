# Background: Redefining Agentic AI with Semantically Rich Data

🔍 What Is Agentics?

Agentics is a novel framework that redefines how we think about agentic AI. Rather than treating data as static input manipulated by intelligent agents, Agentics inverts this paradigm: the intelligence lives in the data itself. By leveraging the power of GenAI and Pydantic types, Agentics transforms data into semantically rich, self-describing objects. This enables elegant, compact, and algebraically principled workflows.

At the core of Agentics lies the concept of logical transduction—the idea that large language models (LLMs) are best understood as transductors, transforming one structured semantic object into another. Agentics turns this insight into a practical system for building declarative, composable, and intelligent workflows.

## 🧠 Design Philosophy

The central design principle of Agentics is to abstract away agent behavior modeling from the developer. Instead of defining agents, goals, and prompts explicitly, developers simply enrich their data structures with semantics via Pydantic models.

In other words:
	•	You don’t program the agent.
	•	You describe the data.
	•	The semantics guide the AI.

Agentics leverages these semantic hints (like field names and descriptions) to guide the internal agent behavior implicitly and transparently.

The foundational building block of the framework is the Agentics class. This class wraps:
	•	A Pydantic type (called atype)
	•	A list of instances of that type (called states)
	•	And optionally, AI features like LLMs, memory, tools, and instructions

This dual nature allows Agentics to behave both like a Python list of Pydantic objects and like an intelligent, semantically-aware agent system.


The core operator in Agentics is the logical transduction operator: <<. This allows you to transform one Agentics object into another:

Demographics << Resumes

Here, each Demographics object is constructed by interpreting the semantic information from corresponding Resumes, using LLMs (and optionally memory, and tools in more advanced cases) but without requiring any manual prompting or parsing logic.

Agentics orchestrates this automatically by creating task agents for each transformation, leveraging the semantic field descriptions of the target Pydantic model to guide extraction.

💡 How It Works
	1.	Define source and target types using Pydantic, with meaningful descriptions for each field.
	2.	Wrap your data using the Agentics class.
	3.	Apply the transduction: the system fills in target attributes using source semantics.
	4.	Use the result as standard Python objects; no JSON parsing, no unstructured strings.

This enables use cases like:
	•	Extracting structured data from text
	•	Mapping between heterogeneous schemas
	•	Building semantically-aware data pipelines

📐 Why It Matters
	•	No more prompt spaghetti: Instructions are embedded as field-level metadata, not external templates.
	•	Type safety by default: Pydantic constraints ensure each LLM call returns well-structured data.
	•	Composability: Algebraic properties of transduction make it easy to chain and compose workflows.
	•	Interoperability: Supports loading from CSV, JSON, and DBs. Agentics can also infer and reshape types on the fly.

## Transduction Algebra

Let:

\[
T \coloneqq \left( (s_1, T_{s_1}), (s_2, T_{s_2}), \ldots, (s_n, T_{s_n}) \right)
\]

be a **Pydantic type**, where:

- \( s_i \) are **slot names** (strings)
- \( T_{s_i} \), \( T \in \Theta \), are **Pydantic types**

Let:

\[
\{X, Y, Z, T, \ldots\} = \Theta
\]

be the **set of all possible types** (denoted by uppercase letters).

Let:

\[
x \in X
\]

denote a **state** for the type \( X \) (denoted by a lowercase letter).

We use the notation:

\[
x[s_1]
\]

to refer to the value of the slot \( s_1 \) in the instance \( x \) of type \( X \).

---

### Logical Transduction

Let:

\[
y' : Y = y : Y \ll x : X
\]

Here, \( \ll \) denotes the **logical transduction operator**.

This operator executes logical transduction from all **non-empty slots** of the source state \( x \) into the target state \( y \).

- \( x : X \) is the source
- \( y : Y \) is the target
- \( y' : Y \) is the result of transduction
---

### Tools


Let:

\[
\mathbb{W} = \{ x \in X \mid X \in \Theta \}
\]

be the **logical world**, i.e., the set of all *thinkable* states across all types in \( \Theta \).

Let:

\[
\mathbb{R}^t \subset \mathbb{W}
\]

be the **real world at time \( t \in \mathbb{T} \)** — that is, the subset of \( \mathbb{W} \) consisting of states that are actually *observable* at that specific time.


A **tool** is a logical transduction that incorporates knowledge of observable states at a given time.

Formally, a tool is defined as:

\[
\varphi : X, \mathbb{R}^t \rightarrow Y
\]

where:

- \( \varphi \in \Theta \)
- \( t \in \mathbb{T} \)
- \( X, Y \in \Theta \)
- \( \mathbb{R}^t \subset \mathbb{W} \) represents *observations* available at time \( t \)

In other words, a tool is a function that transforms a source state \( x \in X \) into a target state \( y \in Y \), with the help of contextual information from the real world \( \mathbb{R}^t \).



### Agentics (AG): A Meta-Type for State Collections

Let

$$
AG := \{ s_{\text{tools}}: \text{List}[\text{Type}[\theta]],\ s_{\text{atype}}: \text{Type}[\theta],\ s_{\text{states}}: \text{List}[s_{\text{atype}}] \}
$$

`Agentics` is a type that provides a **meta-representation for a list of states of the same type**, where:

* $\theta$ is the set of all possible Pydantic types.
* $s_{\text{tools}}$ is the list of available tools (i.e., logical transductions).
* $s_{\text{atype}}$ is the common Pydantic type shared by all states.
* $s_{\text{states}}$ is the actual list of Pydantic objects (states) of type $s_{\text{atype}}$.

---

## Logical Transduction Operator: $\ll$

The **logical transduction operator** is defined as a function:

$$
\ll : (AG, AG) \rightarrow AG
$$

That is, it maps two `Agentics` instances (source and target) to a new `Agentics` instance by applying logical transduction between their respective states.

Let $x \ll y$ be defined as:

$$
x \ll y := AG \left(
\begin{aligned}
& \text{tools} = x[\text{tools}], \\
& \text{atype} = x[\text{atype}], \\
& \text{states} = \{ x[i] \ll y[i], (x[z],y[z]) \mid y[i] \in y[\text{states}] \land x[z] \neq \emptyset \}
\end{aligned}
\right)
$$

Where:

* Each target state $y_i$ is transduced using the corresponding source state $x_i$.
* If the number of target states $|y| > |x|$, excess $y$ states are appended unchanged.
* The non empty states of x are used as few shot training to inform the transduction.

---

### Output Behavior

* The output `Agentics` instance contains the same number of states as in the **target** $y$.
* For each $y_i$, the result of $x_i \ll y_i$ preserves any filled slots from $x_i$ while completing additional ones through LLM-based inference and tool usage.
* This behavior enables declarative, algebraically sound workflows with minimal user specification, relying on embedded semantics.
