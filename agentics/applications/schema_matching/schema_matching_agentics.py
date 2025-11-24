
from agentics.core.mapping import AttributeMapping, ATypeMapping
from typing import Any
from pydantic import Field, create_model
import json
from agentics import AG

def read_gt(input_file):
    gt = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                try:
                    gt.append(json.loads(line))

                except json.JSONDecodeError as e:
                    print(f"Skipping invalid line: {e}")
    return gt



def read_task(input_file):
    final_task={}
    pydantic_types_for_tables={}
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                try:
                    jline=  json.loads(line)
                    table_name=jline["relation_name"].lower() 
                    table_description=jline["relation_description"]
                    if table_name not in final_task:
                        final_task[table_name]={"table_description": table_description,
                                                "table_name":table_name,
                                                "columns": {}}
                    final_task[table_name]["columns"][jline["attribute_name"]] = \
                        (Any,Field(None))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid line: {e}")
        for table in final_task.values():
            pydantic_types_for_tables[table["table_name"]] = create_model(table["table_name"], **table["columns"])

    return pydantic_types_for_tables




async def execute_schema_mappings(source, target):
    tasks = {
        "AdCO": ("admissions", "condition_occurrence"), 
        "AdVD": ("admissions", "visit_detail"), 
        "AdVO": ("admissions", "visit_occurrence"), 
        "DiCO": ("diagnoses_icd", "condition_occurrence"), 
        "LaMe": ("labevents", "measurement"), 
        "PaPe": ("patients", "person"), 
        "PrDE": ("prescriptions", "drug_exposure"), 
        "SeVD": ("services", "visit_detail"), 
        "TrVD": ("transfers", "visit_detail")
    }
    tasks_output=[]
    for task_name, table_pair in tasks.items():
        
        source_ag = AG(atype=source[table_pair[0]])
        target_ag = AG(atype=target[table_pair[1]])
        mapping = await target_ag.map_atypes(source_ag)
        for map in mapping.attribute_mappings:
            if map.confidence>0:
                tasks_output.append({
                "task_name": task_name,
                "source_table": table_pair[0],
                "source_attribute": map.source_field,
                "target_table":table_pair[1],
                "target_attribute": map.target_field,
                "confidence":map.confidence
                })
    with open("/tmp/mappings.json","w") as f:
        f.write(json.dumps(tasks_output))
    return tasks_output
    # print(tasks_output.pretty_print())
    # tasks_output.to_jsonl("/tmp/mappings.jsonl")

def evaluate_mappings(gt_mappings: set, system_mappings: set):
    tp = len(gt_mappings & system_mappings)   # true positives
    fp = len(system_mappings - gt_mappings)   # false positives
    fn = len(gt_mappings - system_mappings)   # false negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = tp / len(gt_mappings | system_mappings) if (gt_mappings | system_mappings) else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard
    }


import asyncio
async def main(source_task, target_task, gt):         
    mimic = read_task(source_task)
    omop = read_task(target_task)
    gt = read_gt(gt)


    task_output=json.load(open("/tmp/mappings.json"))
    gt_mappings=set()
    system_mappings=set()
    for mapping in gt:
        gt_key = f'{mapping["mimic"]["table_id"]}:{mapping["mimic"]["column_name"]}:{mapping["omop"]["table_id"]}:{mapping["omop"]["column_name"]}'
        gt_mappings.add(gt_key.lower())
    for mapping in task_output:
        if mapping["confidence"]>0.9:
            sys_key=f'{mapping["source_table"]}:{mapping["source_attribute"]}:{mapping["target_table"]}:{mapping["target_attribute"]}'
            system_mappings.add(sys_key.lower())

    print(evaluate_mappings(gt_mappings,system_mappings))


#tasks_output = asyncio.run(execute_schema_mappings(mimic, omop))
