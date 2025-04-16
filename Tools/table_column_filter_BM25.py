import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from databasemanger import DatabaseManager
from typing import Annotated
from pydantic import BaseModel, Field
import ollama
import sqlite3
import time
# from langchain_community.chat_models import ChatOllama
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import SystemMessage,HumanMessage
from collections import defaultdict
from Tools.BM25_test import BM25Fuzzy
import re

def LCS(str1_orig: str, str2_orig: str, overlap_threshold1=0.2, overlap_threshold2=0.1):
    """
    计算最长公共子串(LCS)。
    """
    str1 = str1_orig.lower()
    str2 = str2_orig.lower()
    
    # if ' ' in str2:
    #     keywords = str2.split()
    #     total_match_length = 0
    #     total_keywords_length = sum(len(k) for k in keywords)
        
    #     for keyword in keywords:
    #         m, n = len(str1), len(keyword)
    #         dp = [[0] * (n + 1) for _ in range(m + 1)]
    #         max_length = 0
            
    #         for i in range(1, m + 1):
    #             for j in range(1, n + 1):
    #                 if str1[i - 1] == keyword[j - 1]:
    #                     dp[i][j] = dp[i - 1][j - 1] + 1
    #                     if dp[i][j] > max_length:
    #                         max_length = dp[i][j]
    #                 else:
    #                     dp[i][j] = 0
            
    #         total_match_length += max_length
        
    #     overlap_ratio1 = total_match_length / total_keywords_length if total_keywords_length > 0 else 0
    #     overlap_ratio2 = total_match_length / len(str1) if len(str1) > 0 else 0
        
    #     if overlap_ratio1 >= overlap_threshold1 and overlap_ratio2 >= overlap_threshold2:
    #         return {str1_orig: total_match_length}
    #     return {str1_orig: 0}
    
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
            else:
                dp[i][j] = 0

    overlap_ratio1 = max_length / len(str2) if len(str2) > 0 else 0
    overlap_ratio2 = max_length / len(str1) if len(str1) > 0 else 0

    if overlap_ratio1 >= overlap_threshold1 and overlap_ratio2 >= overlap_threshold2:
        return {str1_orig: (overlap_ratio1,overlap_ratio2)}
    return {str1_orig: (0,0)}

def apply_bm25_matching(items, keyword, threshold=0.5):
    """
    Use BM25Fuzzy algorithm to match items
    """
    if not items or not keyword:
        return []
    
    tokenized_items = [item.lower().split() for item in items]
    
    try:
        bm25_fuzzy = BM25Fuzzy(tokenized_items)
        scores = bm25_fuzzy.score(keyword.lower().split())
        
        item_score_pairs = list(zip(items, scores))
        item_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        matched_items = [(item, score) for item, score in item_score_pairs if score > threshold]
        return matched_items
    except Exception as e:
        print(f"BM25 matching error: {str(e)}")
        return []

class TableColumnFilter(BaseModel):
    keywords: Annotated[list, Field(description="List of keywords to filter tables and columns.")]
    db_name: Annotated[str, Field(description="Name of the database.")]


def filter(input: Annotated[TableColumnFilter, "Input parameters for table column filter."]):
    """
    Select the most relevant tables and columns based on the given list of keywords and database name.
    """
    keywords = input.keywords
    db_name = input.db_name
    db_manager = DatabaseManager(db_name)
    table_names = db_manager.get_table_description_filenames()
    tables_and_columns = db_manager.get_table_columns_dict()
    column_names = []
    primary_keys = db_manager.get_primary_foreign_keys()["primary_keys"]
    print(f"table_names:{table_names}")
    print(f"tables_and_columns:{tables_and_columns}")
    for table_name in table_names:
        column_names.extend(tables_and_columns[table_name])
    
    client = db_manager.client
    sqlite_path = str(db_manager.sqlite_path)
    sqlite_path_info = str(db_manager.sqlite_path).replace(".sqlite", "_info.sqlite")
    sqlite_path_description = str(db_manager.sqlite_path).replace(".sqlite", "_description.sqlite")
    relate_tables_by_table_names = []
    relate_tables_by_column_names = []
    relate_tables_by_entity = []
    relate_tables_by_description = []
    relate_tables_by_description_dict = {}
    relate_tables_by_column_dict = {}
    relate_tables_by_entity_dict = {}
    
    # 1. Table Name Matching
    for keyword in keywords:
        if keyword in table_names:
            relate_tables_by_table_names.append(keyword)
            if keyword in primary_keys:
                relate_tables_by_column_dict[keyword] = primary_keys[keyword]
        else:
            matched_tables = apply_bm25_matching(table_names, keyword, threshold=0.1)
            
            for table, score in matched_tables:
                relate_tables_by_table_names.append(table)
                if table in primary_keys:
                    relate_tables_by_column_dict[table] = primary_keys[table]
    
    relate_tables_by_table_names = list(set(relate_tables_by_table_names))
    
    # 2. Column Name Matching
    for keyword in keywords:
        if keyword in column_names:
            with sqlite3.connect(sqlite_path_info) as connection:
                cursor = connection.cursor()
                cursor.execute(f"SELECT table_name FROM info WHERE columns='{keyword}'")
                table_name_by_column_keyword = cursor.fetchall()    
                for table_name in table_name_by_column_keyword:
                    relate_tables_by_column_names.append(table_name[0])
        else:
            matched_columns = apply_bm25_matching(column_names, keyword, threshold=0.1)
            
            for column, score in matched_columns:
                with sqlite3.connect(sqlite_path_info) as connection:
                    cursor = connection.cursor()
                    cursor.execute(f"SELECT table_name FROM info WHERE columns='{column}'")
                    table_name_by_column_names = cursor.fetchall()
                    for table_name_by_column_name in table_name_by_column_names:
                        if table_name_by_column_name[0]:
                            relate_tables_by_column_names.append(table_name_by_column_name[0])
                        if table_name_by_column_name[0] in relate_tables_by_column_dict:
                            relate_tables_by_column_dict[table_name_by_column_name[0]].append(column)
                        else:
                            relate_tables_by_column_dict[table_name_by_column_name[0]] = [column]

    relate_tables_by_column_names = list(set(relate_tables_by_column_names))
    print(3)
    entity_match_results = defaultdict(lambda: defaultdict(set))
    
    # 3. Entity Matching
    for keyword in keywords:
        print(f"\nProcessing keyword: '{keyword}'")
        start_time = time.time()
        
        table_data = db_manager.get_data_to_embed()
        
        for table_name in table_names:
            for column_name in tables_and_columns[table_name]:
                df = table_data[table_name]
                
                unique_values = df[column_name].dropna().unique().tolist()
                unique_values = [str(val) for val in unique_values]
                
                if len(unique_values) > 0:
                    try:
                        matched_values_with_scores = apply_bm25_matching(unique_values, keyword, threshold=0.1)
                        
                        matched_values = []
                        for value, score in matched_values_with_scores:
                            matched_values.append(value)
                            relate_tables_by_entity.append(table_name)
                            if table_name in relate_tables_by_entity_dict:
                                if column_name not in relate_tables_by_entity_dict[table_name]:
                                    relate_tables_by_entity_dict[table_name].append(column_name)
                            else:
                                relate_tables_by_entity_dict[table_name] = [column_name]
                        
                        if matched_values:
                            entity_match_results[table_name][column_name].update(matched_values[:3])
                    
                    except Exception as e:
                        print(f"Processing error: {str(e)}")
                        continue

            execution_time = time.time() - start_time
            print(f"Keyword '{keyword}' processing completed, time: {execution_time:.2f} seconds")

    print(4)
    relate_tables_by_entity = list(set(relate_tables_by_entity))

    # 4. Description Matching
    for keyword in keywords:
        for table_name in table_names:
            for column in ["original_column_name", "column_name", "column_description", "value_description"]:
                with sqlite3.connect(sqlite_path_description) as connection:
                    cursor = connection.cursor()
                    cursor.execute(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL')
                    descriptions = [row[0] for row in cursor.fetchall() if row[0]]
                
                if descriptions:
                    matched_descriptions = apply_bm25_matching(descriptions, keyword, threshold=0.1)
                    
                    for description, score in matched_descriptions:
                        with sqlite3.connect(sqlite_path_description) as connection:
                            cursor = connection.cursor()
                            cursor.execute(f'SELECT "original_column_name" FROM "{table_name}" WHERE "{column}" = ?', (description,))
                            results = cursor.fetchall()
                            for result in results:   
                                if result and result[0]:  
                                    relate_tables_by_description.append(table_name)
                                    if table_name in relate_tables_by_description_dict:
                                        relate_tables_by_description_dict[table_name].append(result[0])
                                    else:
                                        relate_tables_by_description_dict[table_name] = [result[0]]

    relate_tables_by_description = list(set(relate_tables_by_description))

    # Merge results
    relate_tables = relate_tables_by_table_names + relate_tables_by_column_names + relate_tables_by_entity + relate_tables_by_description
    relate_tables = list(set(relate_tables))
    
    merged_dict = defaultdict(set)
    all_keys = set(relate_tables_by_table_names)
    for d in [relate_tables_by_column_dict, relate_tables_by_entity_dict, relate_tables_by_description_dict]:
        all_keys.update(d.keys())
    
    for key in all_keys:
        for d in [relate_tables_by_column_dict, relate_tables_by_entity_dict, relate_tables_by_description_dict]:
            if key in d:
                merged_dict[key].update(d[key])

    filter_dict = {key:list(value) for key, value in merged_dict.items()}

    entity_results_dict = {}
    for table, columns in entity_match_results.items():
        entity_results_dict[table] = {}
        for col, entities in columns.items():
            if entities: 
                sorted_entities = sorted(
                    [entity.lstrip() for entity in list(entities)],
                    key=lambda x: (
                        list(LCS(x, ' '.join(keywords)).values())[0][0],  
                        list(LCS(x, ' '.join(keywords)).values())[0][1]   
                    ),
                    reverse=True
                )
                if sorted_entities:
                    entity_results_dict[table][col] = sorted_entities

    if not entity_results_dict:
        print("Debug: entity_match_results =", dict(entity_match_results))

    return (filter_dict, entity_results_dict)
    
if __name__ == '__main__':
    input = TableColumnFilter(keywords=['average writing score', 'highest number of test takers', 'total SAT scores', 'greater or equal to 1500', 'city'], db_name="california_schools")
    # input = TableColumnFilter(keywords=['DisplayName','average score','Stephen Turner','posts','AVG(Score)'], db_name="codebase_community")
    result = filter(input)
    print("Filter results:", result[0])
    print("Entity match results:", result[1])
