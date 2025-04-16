from common_utils import Preprocessor
from inject_syntax import DEP
from parse_sql import analyze_sql_to_vector,components_list
import os
import re
import json

def create_dev_set(output_path: str, custom_entities: dict, progress_path: str):
    preprocessor = Preprocessor(db_dir=r"Data\dev_databases")
    dep = DEP()
    database_schema_path = r"Data\dev_databases\dev_tables.json"
    dev_question_path = r"Data\dev_databases\dev.json"

    with open(database_schema_path, "r") as f:
        database_schema = json.load(f)
    with open(dev_question_path, "r") as f:
        dev_questions = json.load(f)

    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            progress = json.load(f)
    else:
        progress = {"processed_databases": [], "processed_questions": {}}

    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            json.dump([], f)

    for table_info in database_schema:
        table_name = table_info["db_id"]
        if table_name in progress["processed_databases"]:
            print(f"Skipping already processed database: {table_name}")
            continue

        print(f"Processing questions for database: {table_name}")
        proper_noun = None
        for custom_entity in custom_entities:
            if custom_entity['db_id'] == table_name:
                proper_noun = custom_entity['stopwords']
                print(f"Processing {table_name} custom entity: {proper_noun}")
                print('*' * 100)
                break

        db = preprocessor.preprocess_database(db=table_info, custom_entities=proper_noun)
        progress["processed_questions"].setdefault(table_name, [])

        for question in dev_questions:
            if question["db_id"] == table_name and question["question_id"] not in progress["processed_questions"][table_name]:
                question_text = question["question"]
                question_toks_unclean = re.split(r'[^\w\s_]', question_text)
                question_toks = [item.strip() for item in question_toks_unclean if item.strip()]
                question_id = question["question_id"]
                question_entry = {
                    'question_id': question_id,
                    'db_id': table_name,
                    'question': question_text,
                    'question_toks': question_toks
                }
                entry_after_question_preprocessing = preprocessor.preprocess_question(
                    entry=question_entry, db=db, custom_entities=proper_noun)
                entry_after_schema_link = preprocessor.schema_linking(
                    entry=entry_after_question_preprocessing, db=db, custom_entities=proper_noun)
                entry_after_injection = dep.inject_syntax(entry=entry_after_schema_link)

                with open(output_path, "r+") as f:
                    data = json.load(f)
                    data.append(entry_after_injection)
                    f.seek(0)
                    json.dump(data, f, indent=4)

                print(f"Processed question: {question_text} for database: {table_name}")

                progress["processed_questions"][table_name].append(question_id)
                with open(progress_path, "w") as f:
                    json.dump(progress, f, indent=4)

        progress["processed_databases"].append(table_name)
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=4)

if __name__ == "__main__":
    custom_entities_path = r"Data\stopwords\stopwords_codeversion.json"
    with open(custom_entities_path, "r") as f:
        custom_entities = json.load(f)

    output_path = r"Data\dev_databases\dev_data.json"
    progress_path = r"Data\dev_databases\progress_schema.json" 

    create_dev_set(output_path, custom_entities, progress_path)