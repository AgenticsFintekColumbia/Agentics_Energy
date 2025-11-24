import asyncio
import json
import os

import streamlit as st
from db import DB
from text2sql import Text2sqlQuestion, enrich_all_dbs, execute_questions, load_db
from utils import evaluate_execution_accuracy, get_schema_from_file, load_benchmark

from agentics import AG

if "benchmark_questions" not in st.session_state:
    st.session_state.benchmark_questions = AG(atype=Text2sqlQuestion)

if "benchmark_metadata" not in st.session_state:
    st.session_state.benchmark_metadata = {}


st.header("Agentics Text2SQL")

with st.sidebar:
    use_answer_validation = st.toggle("Answer Validation", value=False)
    use_enrichments = st.toggle("Enrichments", value=False)


def select_benchmark(benchmark_id):
    st.session_state.benchmark_metadata = load_benchmark(benchmark_id)
    if "datasource_url" in st.session_state.benchmark_metadata:
        questions = json.load(
            open(
                os.path.join(os.getenv("SQL_BENCHMARKS_FOLDER"), benchmark_id + ".json")
            )
        )
        st.session_state.benchmark_questions = AG(atype=Text2sqlQuestion)
        for question in questions[: n_questions if n_questions > 0 else None]:
            st.session_state.benchmark_questions.states.append(
                Text2sqlQuestion(
                    question=question["page_content"],
                    sql=question["sql"],
                    benchmark_id=benchmark_id,
                    endpoint_id=str(
                        st.session_state.benchmark_metadata["datasource_url"].split(
                            "/"
                        )[-1]
                    ),
                )
            )

    else:

        st.session_state.benchmark_questions = AG.from_jsonl(
            os.path.join(os.getenv("SQL_BENCHMARKS_FOLDER"), benchmark_id + ".json"),
            jsonl=False,
            atype=Text2sqlQuestion,
            max_rows=n_questions if n_questions > 0 else None,
        )
    final_questions = []
    for question in st.session_state.benchmark_questions:
        question.benchmark_id = benchmark_id
        final_questions.append(question)
    st.session_state.benchmark_questions.states = final_questions
    st.rerun()


tab1, tab2, tab3 = st.tabs(["Benchmarks", "Questions", "DBs"])

with tab1:
    with st.form("Benchmark Selection"):
        st.session_state.benchmark_id = st.selectbox(
            "Choose your Benchmark", options=list(load_benchmark().keys())
        )
        n_questions = st.number_input("Max Questions", value=1000)
        select_benchmark_button = st.form_submit_button("Select Benchmark")
        save_experiment_path = st.text_input("Save Experiments Path", value=None)
        number_of_experiments = st.number_input("Average N experiments", value=5)
        evaluate_benchmark = st.form_submit_button("Evaluate Benchmark")

    if select_benchmark_button:
        select_benchmark(st.session_state.benchmark_id)

    if evaluate_benchmark and st.session_state.benchmark_questions:
        with st.spinner("Wait Benchmark Execution In Progress"):
            new_questions = []
            for question in st.session_state.benchmark_questions:
                question.generated_query = None
                question.answer_assessment = None
                question.system_output_df = None
                new_questions.append(question)
            st.session_state.benchmark_questions.states = new_questions
            st.session_state.benchmark_questions, execution_accuracy = asyncio.run(
                execute_questions(
                    st.session_state.benchmark_questions,
                    answer_validation=use_answer_validation,
                    enrichments=use_enrichments,
                    multiple_runs=number_of_experiments,
                    save_run_path=save_experiment_path,
                )
            )
            ex, text = evaluate_execution_accuracy(st.session_state.benchmark_questions)
            st.markdown(f"Average Execution Accuracy {execution_accuracy}")
            # if save_experiment_path: st.session_state.benchmark_questions.to_jsonl(save_experiment_path)

with tab2:
    with st.form("Select Question"):

        select_question = st.selectbox(
            "Choose a question",
            format_func=lambda x: x.question,
            options=st.session_state.benchmark_questions.states,
        )
        execute_selected = st.form_submit_button("Execute Selected Question")

    with st.form("Ask your own question"):
        db = st.selectbox(
            "Choose Target DB",
            options=list(get_schema_from_file(st.session_state.benchmark_id).keys()),
        )
        user_question = st.text_input("Aks your question")
        execute_user_question = st.form_submit_button("Ask Question")

    if execute_selected:
        st.session_state.test = AG(atype=Text2sqlQuestion, states=[select_question])

    if execute_user_question:
        question = Text2sqlQuestion(
            question=user_question,
            db_id=db,
            benchmark_id=st.session_state.benchmark_id,
        )
        st.session_state.test = AG(atype=Text2sqlQuestion, states=[question])

    if execute_selected or execute_user_question:
        st.write(st.session_state.test[0])
        test = asyncio.run(execute_questions(st.session_state.test))
        col1, col2 = st.columns(2)
        col1.markdown(f"### System\n\n```sql\n{test[0].generated_query}")

        try:
            col1.dataframe(json.loads(test[0].system_output_df))
        except:
            col1.write(test[0].system_output_df)

        col2.markdown(f"### GT\n\n```sql\n{test[0].sql or test[0].query}")

        try:
            col2.dataframe(json.loads(test[0].gt_output_df))
        except:
            col2.write(test[0].gt_output_df)

        with st.popover("See details"):
            st.write(test[0])

with tab3:
    with st.form("DB"):
        enrich_all_db = st.form_submit_button("Enrich all DBs")
        db_id = st.selectbox(
            "Choose Target DB",
            options=list(get_schema_from_file(st.session_state.benchmark_id).keys()),
        )
        # db_path=st.text_input("Chose Output DB folder")
        show_selected_db = st.form_submit_button("Show Selected DB")
        show_enriched_db = st.form_submit_button("Show Enriched DB")

        # enrich_current_db = st.form_submit_button("Enrich current DBs")

    if show_selected_db:
        db = DB(benchmark_id=st.session_state.benchmark_id, db_id=db_id)
        st.write(asyncio.run(db.load_db()))
    if show_enriched_db:
        db = DB(benchmark_id=st.session_state.benchmark_id, db_id=db_id)
        st.write(asyncio.run(db.load_enrichments()))

    if enrich_all_db:
        with st.spinner("Wait Benchmark Execution In Progress"):
            output = asyncio.run(enrich_all_dbs(st.session_state.benchmark_questions))
        # if output[1]:
        #     db = output[0]
        # else:

        #     db=DB(benchmark_id=benchmark_id,endpoint_id=endpoint_id,db_id=db_id, db_path=schema_path)
        #     db = db.get_schema_from_sqllite()
        #     db = asyncio.run(db.generate_db_description())
        #     db =  asyncio.run(db.get_schema_enrichments())
        #     db.ddl = db.db_schema.generate_ddl()
        #     with open(enrichment_path, "w") as f: f.write(db.model_dump_json())

        # st.write(output)
