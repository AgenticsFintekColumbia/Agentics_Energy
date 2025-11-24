# Use Cases

**Agentics** is a versatile framework designed for a wide range of applications involving the manipulation of **tabular data** and, more broadly, **JSON objects of arbitrary structure**. Below are the core use cases that Agentics was originally designed to support. These capabilities are built-in and available natively within the system.

---

## Native Capabilities

### ✅ Information Extraction from Documents

This foundational use case is modeled in Agentics by defining a Pydantic type for the output schema and executing a transduction from a list of input texts extracted from that document for to that type.

#### Advantages of using Agentics

- 🚀 **Asynchronous Execution**: Enables >10× speedup through parallel LLM calls.

- 🧠 **No-Code Interface**: Define output types using simple YAML or an interactive editor—no Python code required.

- 🗂️ **Seamless Ingestion from multiple document types**: Agentics offers built in import and export capabilities to JSON, CSV, TXT and JSONL documents. Additionally, Agentics uses [Docling](https://docling-project.github.io/docling/) to enable ingestion of multiple document formats incl. PDF, DOCX, XLSX, HTML, images, and more .

#### Application scenarios

- Information Extraction from Financial Reports, Medical Records, Invoices, Technical Documentation. 
- Quality Evaluation of ETL workflows output, including Text2SQL
- Automatic population of DBs Tables from texts

---

### ✅ Data Imputation in DB Tables

Agentics handles missing values in structured data by importing it into an `Agentics` object with column-based types. The system then applies **self-transduction** to each column with missing values, using the available (non-missing) data as few-shot examples.

#### Advantages of using Agentics

- ⚡ **Asynchronous Execution**: Efficient batch processing of imputation tasks.

- 🔁 **Native Self-Transduction**: Built-in support for learning from partial data and iteratively filling in missing values.

#### Application Scenarios

- Automated Data Science: Inputation of missing values on table is a generalization of supervised learning from positive examples on a multiclass scenario. 

- Data Curation: Inputation of missing value enable data repair and augmentation in DBs

- Data Enrichment: Dynamic extension of data types enables interactive definition of new dimension. 

---

### ✅ Structured Retrieval-Augmented Generation (RAG)

Agentics includes a built-in memory component to support structured RAG, where both **inputs and outputs** are modeled as Pydantic types.

This approach generalizes RAG in two key ways:

1. 🔣 The input can be any **structured object**, not just a single query.

2. 🧩 The output is a **structured object**, capturing multiple dimensions or aspects of the answer.

**Advantages of using Agentics:**

- 🧱 **Structured Inputs and Outputs**: Fully typed I/O using Pydantic.

- 🧰 **Built-in Memory Server**: No additional setup required. Enable Ingestions of Multiple Data and Document types using Docling. 


- ⚙️ **Async Execution**: Executes RAG operations in parallel for significant performance gains.

#### Application Scenarios

- Document QA: This is implemented natively by Agentics by ingesting the document corpus in a memory collection and transducing the question into an answer. Docling enable ingestion of a large variety of document sources. Low code (1 line) implementation in agentics.

- Text2SQL: it is an excellent case of structured RAG, where the input is a question and additional data about the target source such as the DB schema, and the output is a SQL query which is further executed to return a DataFrame Object. All this is modelled by a single structured RAG operation in Agentics. 



---

### ✅ Structured Data Workflows

Agentics integrates seamlessly with tools like **LangGraph** and can infer attributes of state graphs or structured workflows by modeling each step as a self-transducing unit.

This allows you to:

- Represent states using typed objects

- Apply self-transduction to infer unknown attributes from known ones

- Compose steps algebraically to define low-code, multi-stage logic

#### Advantages of using Agentics

- ✨ **Streamlined Code**: More readable and maintainable than typical Langchain-style graphs.

- 🔗 **Pydantic Compatibility**: Fully aligned with agentic frameworks using Pydantic types.

- 🧠 **Composable Transduction**: Enables advanced, multi-step agentic pipelines.

#### Application Scenario

- **visual IDEs for GenAI Workflows**: Transduction operation among Agentics can be easily modeled by means of flow diagrams in a very intuitive and no code manner, extending the capabilities of frameworks like LangFlow. 

- **NO Code ETL Workflows**: Agentics enable representation of any type of structured data, providing helpful utilities to asynchronously apply transductions and/or ad hoc logics to modify data
