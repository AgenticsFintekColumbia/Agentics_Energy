
# 👋 Welcome to Agentics

**Agentics** is a lightweight, Python-native framework for building structured, agentic workflows over tabular or JSON-based data using Pydantic types and transduction logic. Designed to work seamlessly with large language models (LLMs), Agentics enables users to define input and output schemas as structured types and apply declarative, composable transformations—called transductions—across data collections. It supports asynchronous execution, built-in memory for structured retrieval-augmented generation (RAG), and self-transduction for tasks like data imputation and few-shot learning. With no-code and low-code interfaces, Agentics is ideal for rapidly prototyping intelligent systems that require structured reasoning, flexible memory access, and interpretable outputs.

## 📚 Documentation Overview

This documentation introduces the core concepts behind Agentics and provides everything you need to start building structured, agentic workflows using its data model and transduction framework.

⸻

👉 [Getting Started](getting_started.md): Learn how to install Agentics, set up your environment, and run your first logical transduction.

📘 [Why Agentics?](background.md): Understand the foundational principles and architecture of the Agentics framework.

🚀 [Use Cases](use_cases.md): Explore real-world scenarios where Agentics enhances data intelligence and reasoning capabilities.

🧠  [Agentics](agentics.md): See how Agentics wraps pydantic models into transduction-ready agents for structured execution.

🔁 [Transduction](transduction.md): Discover how the << operator enables logical transduction between types and how to customize its behavior.

🧬  [Memory](memory.md): Use external knowledge from documents to augment transduction through the built-in memory system.

🛠️ [Tools](tools.md): Integrate with external frameworks like LangChain or CrewAI to provide dynamic access to data sources during transduction.
