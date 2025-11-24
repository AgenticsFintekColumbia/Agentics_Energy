## Text-to-SQL in Agentics

This example demonstrates how to use Agentics to convert natural-language questions into SQL queries and execute them over a database.  The core of the workflow is the `execute_questions` method.  It accepts an `AG[Text2sqlQuestion]` object and returns the same object with several additional fields populated:

- **`schema` (str)** – a string representation of the target database schema.  If you don’t supply a schema, it is generated automatically by querying the database named `db_id` under the path specified by `SQL_DB_PATH`.
- **`generated_query` (str)** – the SQL statement produced via logical transduction.  This is the query that Agentics generates to answer the input question.
- **`system_output_df` (str)** – a JSON-encoded dump of the DataFrame produced by running `generated_query`.  If the query fails, this field contains a string starting with `"Error: "` followed by the error message returned by the SQL parser.
- **`gt_output_df` (str)** – similar to `system_output_df`, but for the provided ground-truth (GT) query.  If no GT query is provided, this field remains `None`.

### Setup

1. Download the BIRD dev dataset from [this link](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip) to your local machine.  This dataset contains example questions, ground-truth queries and the corresponding SQLite databases used for evaluation.
2. Unzip the downloaded archive and locate the `databases.zip` file inside it.  Extract `databases.zip` into a directory; this directory will hold multiple sub-directories, one per database.
3. Set the environment variable `SQL_DB_PATH` to the root of the extracted databases folder.  Agentics uses this variable to find the database corresponding to the `db_id` field in each `Text2sqlQuestion`.

Once these steps are complete you can call `execute_questions` on a collection of questions.  Agentics will automatically inspect the target database schema (if the `schema` field is missing), generate a SQL query to answer each question, execute it, and compare the result with any provided ground-truth query.  See `text2sql.py` for an end-to-end example of loading questions from a JSONL file, running `execute_questions` and inspecting the outputs.

### Evaluate on Bird Benchmark

In the unzipped dev folder you will find dev set dev.json
You can pass the path of that file as an argument for this function

run_evaluation_benchmark(path="YOUR_LOCAL_dev.json")

### Using training data

 Download the BIRD training dataset from [this link](https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip) to your local machine. 

Unzip the downloaded archive and locate the `databases.zip` file inside it.  Extract `databases.zip` into a directory; this directory will hold multiple sub-directories, one per database. 

copy the content of this folder into your SQL_DB_PATH which contains the dev dataset, so that the agent can access them all

run_evaluation_benchmark(path="YOUR_LOCAL_dev.json", few_shots_path="YOUR_LOCAL_train.json")