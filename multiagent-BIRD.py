from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Type, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import dotenv
import os
import sqlite3
import time
import re
from pydantic import BaseModel
from typing import Literal
from collections import defaultdict

from prompt_all import *
from Tools.table_column_filter_BM25 import filter, TableColumnFilter
from Tools.classifier import classifier, ClassifierInput
from databasemanger import DatabaseManager

@dataclass
class AgentContext:
    question: str
    hint: str
    db_name: str
    intermediate_results: Dict[str, Any]

class ReasoningStep(BaseModel):
    title: str
    content: str
    next_action: Literal["continue", "final_answer"]
    final_sql: Optional[str] = None

class FinalAnswer(BaseModel):
    title: str
    content: str

class Node(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def process(self, context: AgentContext) -> Dict[str, Any]:
        pass

class AgentNode(Node):
    def __init__(self, name: str):
        super().__init__(name)
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o',
            temperature=0
        )
        self.system_prompt = ""

def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            format_schema = ReasoningStep if not is_final_answer else FinalAnswer
            chat_model = ChatOpenAI(
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                model='gpt-4o',
                temperature=0.6
            )

            messages[0]["content"] += f"\nOutput must strictly follow this JSON schema:\n{json.dumps(format_schema.model_json_schema(), indent=2)}"
            
            formatted_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    formatted_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    formatted_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_messages.append(AIMessage(content=msg["content"]))
            
            response = chat_model.invoke(formatted_messages)
            
            try:
                content = response.content
                if isinstance(content, str):
                    content = content.replace('```json', '').replace('```', '').strip()
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start != -1 and end != 0:
                        content = content[start:end]
                
                return format_schema.model_validate_json(content)
            except Exception as json_error:
                print(f"JSON parsing error: {str(json_error)}")
                print(f"Raw content: {content}")
                raise
                
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return FinalAnswer(title="Error", content=f"Failed to generate final answer after 3 attempts. Error: {str(e)}")
                else:
                    return ReasoningStep(title="Error", 
                                       content=f"Failed to generate step after 3 attempts. Error: {str(e)}", 
                                       next_action="final_answer")
            time.sleep(1) 


def generate_o1_reasoning(prompt):
    messages = [
        {"role": "system", "content": """You are an expert SQL designer that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. 

When your next_action is "final_answer", you MUST include a "final_sql" field with the complete SQL query.

Respond in JSON format with 'title', 'content', 'next_action' (either 'continue' or 'final_answer') keys, and 'final_sql' when providing the final answer. 

In all your reasoning steps, at least one step must involve revising the important rules. During this step, you need to check each rule individually to ensure that the SQL complies with the rules.

USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE SQL DESIGNS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

IMPORTANT SQL DESIGN GUIDELINES:
1. When finding maximum or minimum values based on specific conditions, use ORDER BY + LIMIT 1 instead of MAX/MIN in subqueries.
2. Pay attention to the data storage format when sorting, comparing sizes, or performing calculations. If the data is in string format, process it using INSTR or SUBSTR before comparison.
3. If your query includes an ORDER BY clause to sort results, only include columns used for sorting in the SELECT clause if specifically requested in the question. Otherwise, omit these columns from the SELECT clause.
4. Ensure you only output information requested in the question. If the question asks for specific columns, ensure the SELECT clause only includes those columns, nothing more.
5. The query should return all information requested in the question - no more, no less.
6. For key phrases mentioned in the question, we have marked the most similar values with "EXAMPLE" in front of the corresponding column names. This is an important hint indicating the correct columns to use for SQL queries.
7. NEVER use || ' ' || for string concatenation. This is strictly prohibited and will result in severe penalties.

Example of a valid JSON response for a continuing step:
```json
{
    "title": "Identifying Key Tables and Columns",
    "content": "To begin solving this SQL problem, we need to carefully examine the given information and identify the crucial tables and columns that will guide our solution process. This involves...",
    "next_action": "continue"
}```

Example of a valid JSON response for a revising step:
```json
{
    "title": "Revising The Importang Rules",
    "content": "I will revise the import rules row by row: 
    1. I will use ORDER BY time replace the MIN(time);
    2. The time is in TEXT format, so I will use (CAST(SUBSTR(event_time, 1, INSTR(event_time, ':') - 1) AS INT) * 60 * 1000) + (CAST(SUBSTR(event_time, INSTR(event_time, ':') + 1, INSTR(event_time, '.') - INSTR(event_time, ':') - 1) AS INT) * 1000) + CAST(SUBSTR(event_time, INSTR(event_time, '.') + 1) AS INT) to calculate the total milliseconds;
    3. The question asks for event name, but I return the time and names at the same time, so I will remove the time column from SELECT clause;
    4. After I remove the time column, the sql give back the right information;
    5. For time EXAMPLE 15:42:176 in TEXT format, I will perform right calculations using INSTR and SUBSTR.
    6. The SQL I generated does not include the use of || , || or CONCAT for string concatenation.
    Now I think I rightly revise all the rules and generate right sql: SELECT event_name FROM events ORDER BY (CAST(SUBSTR(time, 1, INSTR(time, ':') - 1) AS INT) * 60 * 1000) + (CAST(SUBSTR(time, INSTR(time, ':') + 1, INSTR(time, '.') - INSTR(time, ':') - 1) AS INT) * 1000) + CAST(SUBSTR(time, INSTR(time, '.') + 1) AS INT) ASC LIMIT 1;",
    "next_action": "continue"
}```

Example of a valid JSON response for the final step:
```json
{
    "title": "Finalizing the SQL Query",
    "content": "After careful analysis and consideration of all alternatives, I've determined the optimal SQL query for this problem...",
    "next_action": "final_answer",
    "final_sql": "SELECT column1, column2 FROM table1 JOIN table2 ON table1.id = table2.id WHERE condition = 'value';"
}```
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    
    while True:
        print(f"\n===== Chain-of-thought Step {step_count} =====")
        step_data = make_api_call(messages, 300)
        steps.append(step_data)

        print(f"Title: {step_data.title}")
        print(f"Content: {step_data.content[:200]}..." if len(step_data.content) > 200 else f"Content: {step_data.content}")
        print(f"Next action: {step_data.next_action}")
        if step_data.final_sql:
            print(f"Final SQL: {step_data.final_sql}")
        
        messages.append({"role": "assistant", "content": step_data.model_dump_json()})
        
        if step_data.next_action == 'final_answer' and hasattr(step_data, 'final_sql') and step_data.final_sql:
            return f"FINAL SQL: {step_data.final_sql}"
        
        if step_data.next_action == 'final_answer' or step_count > 10:  
            break
        
        step_count += 1
    
    print("\n===== Generate final answer =====")
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above. Make sure to include the final SQL query in the 'final_sql' field."})
    final_data = make_api_call(messages, 200, is_final_answer=True)
    
    print(f"Final Answer Title: {final_data.title}")
    print(f"Final Answer Content: {final_data.content[:200]}..." if len(final_data.content) > 200 else f"Final Answer Content: {final_data.content}")
    
    if hasattr(final_data, 'final_sql') and final_data.final_sql:
        print(f"FINAL SQL: {final_data.final_sql}")
        return f"FINAL SQL: {final_data.final_sql}"

    return final_data.content

class ToolNode(Node):
    def __init__(self, 
                 name: str, 
                 func: Callable, 
                 description: str,
                 input_constructor: Type,
                 result_key: str,
                 required_keys: List[str] = None,
                 param_mapping: Dict[str, str] = None):
        super().__init__(name)
        self.func = func
        self.description = description
        self.input_constructor = input_constructor
        self.result_key = result_key
        self.required_keys = required_keys or []
        self.param_mapping = param_mapping or {}
        
    def process(self, context: AgentContext) -> Dict[str, Any]:
        input_args = {}
        
        if "db_name" in self.param_mapping:
            input_args[self.param_mapping["db_name"]] = context.db_name
        if "question" in self.param_mapping:
            input_args[self.param_mapping["question"]] = context.question
        
        for key in self.required_keys:
            if key in context.intermediate_results:
                param_name = key.split('.')[-1]
                if param_name in self.param_mapping:
                    param_name = self.param_mapping[param_name]
                input_args[param_name] = context.intermediate_results[key]
        
        input_obj = self.input_constructor(**input_args)
        result = self.func(input=input_obj)
        
        return {self.result_key: result}

class KeywordAgent(AgentNode):
    def __init__(self):
        super().__init__("keyword_extractor")
        self.system_prompt = KEYWORD_EXTRACTOR_PROMPT
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o',
            temperature=0
        )
    
    def process(self, context: AgentContext) -> Dict[str, Any]:
        try:
            query = f'''################################################################################################################################################################################################
Task:
Given the following question and hint, identify and list all relevant keywords, keyphrases, and named entities.
REMEMBER: 
1. For a keyword phrase, you MUST give the phrase and split it into multiple keywords at the same time.
2. For datetime and special number, list them as they are.         

Question: {context.question}
Hint: {context.hint}

Return only the list of keywords and keyphrases, formatted as a Python list.
################################################################################################################################################################################################
'''
            
            result = self.chat_model.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query)
            ])
            
            keywords = self._parse_keywords(result.content)
            return {"keywords": keywords}
        except Exception as e:
            raise
            
    def _parse_keywords(self, content: str) -> List[str]:
        try:
            return json.loads(content)
        except:
            content = content.strip('[]')
            keywords = re.findall(r'"([^"]*)"', content)
            if not keywords:
                keywords = [k.strip().strip('"').strip("'") for k in content.split(',')]
            return [k.strip() for k in keywords if k.strip()]

class TableAgent(AgentNode):
    def __init__(self):
        super().__init__("table_selector")
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o',
            temperature=0
        )
        self.system_prompt = TABLE_SELECTOR_PROMPT
    
    def process(self, context: AgentContext) -> Dict[str, Any]:
        try:
            filter_tuple = context.intermediate_results.get("filter_tuple", ())
            entity_results_dict = filter_tuple[1]

            table_column_description = self._get_table_column_descriptions(
                filter_tuple[0], 
                context.db_name,
                entity_results_dict
            )

            query = f'''################################################################################################################################################################################################
QUESTION: {context.question}
HINT: {context.hint}
FILTER TABLE: {json.dumps(filter_tuple[0], ensure_ascii=False, indent=4)}
COLUMN DESCRIPTION: {json.dumps(table_column_description, ensure_ascii=False, indent=4)}

This schema provides a detailed definition of the database's structure, including tables, their columns, and any relevant details about relationships or constraints.  
For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "EXAMPLE" in front of the corresponding column names. This is a critical hint to identify the tables and columns that will be used in the SQL query.

Now follow the instructions and the example to select the relevant tables and columns:
Let's think step by step as the example:  
1. Understand the question and hint.  
2. Examine the available tables and columns.  
3. Prioritize the tables and columns explicitly mentioned in the HINT.  
4. Use column descriptions and EXAMPLE to verify additional relevance, but only after confirming the HINT's tables and columns are selected.  
5. Generate output.

---

### Important Notes:
1. **Priority Selection**:  
   - **You MUST prioritize the tables and columns explicitly mentioned in the HINT**: `{context.hint}`. These tables and columns must be selected directly. Avoid using similar ones unless explicitly permitted, as this will lead to errors.  

2. **EXAMPLE Relevance Check**:  
   - If a column's `EXAMPLE` contains key phrases from the question, you may select it **only after confirming** that all relevant tables and columns from the HINT are included.  

3. **All Relevant Columns**:  
   - Select all possible relevant columns from the tables mentioned in the HINT, not just the most relevant ones. Missing any column explicitly listed or related to the HINT will lead to errors.  

4. **Avoid Similar Tables and Columns**:  
   - Do not infer or select tables or columns that appear similar to those in the HINT but are not explicitly listed. This rule ensures accuracy and avoids logical errors in the SQL query.  

5. **Output Format**:  
   - Ensure that the final output is presented in JSON format as shown below.  

---

### OUTPUT FORMAT:
```json
{{
    "table1": ["column1", "column2", "..."],
    "table2": ["column3", "column4", "..."]
}}
################################################################################################################################################################################################
'''
            
            result = self.chat_model.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query)
            ])
            print(result.content)
            selected_tables = self._parse_json_output(result.content)
            return {"selected_tables": selected_tables}
        except Exception as e:
            print(f"TableAgent error: {str(e)}")
            raise
            
    def _get_table_column_descriptions(self, filter_dict: Dict[str, List[str]], db_name: str, entity_examples: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict]:

        import pandas as pd
        table_column_description = {}
        sqlite_path = f"Data\dev_databases\{db_name}\{db_name}.sqlite"
        base_path = f"Data\dev_databases\{db_name}"
        
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

        for table_name, column_names in filter_dict.items():
            table_path = f"{base_path}\database_description\{table_name}.csv"
            table_description = None
            
            for encoding in encodings:
                try:
                    table_description = pd.read_csv(table_path, encoding=encoding)
                    break 
                except UnicodeDecodeError:
                    continue  
                except Exception as e:
                    print(f"A non-encoded error occurred while reading table {table_name}: {str(e)}")
                    break
                
            if table_description is None:
                print(f"The description file for table {table_name} cannot be read using any encoding")
                table_column_description[table_name] = {}
                continue
            
            try:
                column_description = {}
                
                for column_name in column_names:
                    result = table_description[
                        table_description['original_column_name'].str.contains(
                            column_name, case=False, na=False, regex=False
                        )
                    ]
                    
                    if not result.empty:
                        column_description[column_name] = result.iloc[0].to_dict()
                    else:
                        column_description[column_name] = {}

                    has_valid_examples = False
                    if table_name in entity_examples and column_name in entity_examples[table_name]:
                        examples = entity_examples[table_name][column_name]
                        valid_examples = [ex for ex in examples if ex not in ['null', None, 'None', '']]
                        if valid_examples:
                            column_description[column_name]['EXAMPLE'] = valid_examples
                            has_valid_examples = True
                    
                    if not has_valid_examples:
                        try:
                            with sqlite3.connect(sqlite_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute(f'''
                                    SELECT DISTINCT "{column_name}" 
                                    FROM "{table_name}" 
                                    WHERE "{column_name}" IS NOT NULL 
                                    AND "{column_name}" != ''
                                    LIMIT 3
                                ''')
                                records = cursor.fetchall()
                                examples = [str(record[0]) for record in records if record[0] is not None]
                                if examples:
                                    column_description[column_name]['EXAMPLE'] = examples
                                else:
                                    column_description[column_name]['EXAMPLE'] = []
                        except sqlite3.Error as e:
                            print(f"An error occurred while obtaining an example of column {column_name} in table {table_name}: {str(e)}")
                            column_description[column_name]['EXAMPLE'] = []
                
                table_column_description[table_name] = column_description
            except Exception as e:
                print(f"An error occurred while processing the description information of table {table_name}: {str(e)}")
                table_column_description[table_name] = {}
            
        return table_column_description
            
    def _parse_json_output(self, content: str) -> Dict[str, List[str]]:
        output_text = content.strip()
        
        try:
            if output_text.startswith('{') and output_text.endswith('}'):
                return json.loads(output_text)
            
            if '```json' in output_text:
                pattern = r'```json\s*(.*?)\s*```'
                matches = list(re.finditer(pattern, output_text, re.DOTALL))
                if matches:
                    json_str = matches[-1].group(1).strip()
                    return json.loads(json_str)
            
            dict_pattern = r'{([^}]+)}'
            match = re.search(dict_pattern, output_text)
            if match:
                dict_str = '{' + match.group(1) + '}'
                dict_str = re.sub(r'(?<![\[{:,\s])"(?![}\],])', '"', dict_str)
                return json.loads(dict_str)
            
        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Original content: {output_text}")
        
        print("Warning: Unable to parse table selection results")
        return {}

class SQLDesignerAgent(AgentNode):
    def __init__(self):
        super().__init__("sql_designer")
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o',
            temperature=0.7
        )
    
    def process(self, context: AgentContext) -> Dict[str, Any]:
        try:
            question = context.question
            hint = context.hint
            db_name = context.db_name
            
            selected_tables = context.intermediate_results.get("selected_tables", {})
            filter_tuple = context.intermediate_results.get("filter_tuple", ())
            

            db_manager = DatabaseManager(db_name)
            primary_keys = db_manager.get_primary_foreign_keys()["primary_keys"]
            foreign_keys = db_manager.get_primary_foreign_keys()["foreign_keys"]

            description = self._get_table_column_descriptions(
                selected_tables, 
                db_name,
                filter_tuple[1] if filter_tuple else {}
            )
            
            sql_candidates = []
            for run in range(2):
                print(f"Using o1-like chain-of-thought to generate SQL... (Running {run + 1}/2)")
                o1_query = self._construct_o1_sql_query(
                    question=question,
                    hint=hint,
                    description=description,
                    filter_table_column=selected_tables,
                    sql_skeleton=context.intermediate_results.get("sql_components", []),
                    primary_keys=primary_keys,
                    foreign_keys=foreign_keys
                )
                o1_result = generate_o1_reasoning(o1_query)
                sql_candidates.append(o1_result)
            
            cleaned_sql_candidates = self._clean_sql_results(sql_candidates)
            
            return {"sql_candidates": cleaned_sql_candidates}
            
        except Exception as e:
            print(f"SQLDesignerAgent error: {str(e)}")
            raise
            
    def _get_table_column_descriptions(self, filter_dict: Dict[str, List[str]], db_name: str, entity_examples: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict]:
        import pandas as pd
        table_column_description = {}
        base_path = f"Data\dev_databases\{db_name}"
        sqlite_path = f"Data\dev_databases\{db_name}\{db_name}.sqlite" 
        
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

        for table_name, column_names in filter_dict.items():
            table_path = f"{base_path}\database_description\{table_name}.csv"
            table_description = None
            
            for encoding in encodings:
                try:
                    table_description = pd.read_csv(table_path, encoding=encoding)
                    break  
                except UnicodeDecodeError:
                    continue  
                except Exception as e:
                    print(f"Non-encoded error occurred while reading table {table_name}: {str(e)}")
                    break
                
            if table_description is None:
                print(f"The description file for table {table_name} cannot be read using any encoding")
                table_column_description[table_name] = {}
                continue
            
            try:
                column_description = {}
                
                for column_name in column_names:
                    result = table_description[
                        table_description['original_column_name'].str.contains(
                            column_name, case=False, na=False, regex=False
                        )
                    ]
                    
                    if not result.empty:
                        column_description[column_name] = result.iloc[0].to_dict()
                    else:
                        column_description[column_name] = {}
                    
                    has_valid_examples = False
                    if table_name in entity_examples and column_name in entity_examples[table_name]:
                        examples = entity_examples[table_name][column_name]
                        valid_examples = [ex for ex in examples if ex not in ['null', None, 'None', '']]
                        if valid_examples:
                            column_description[column_name]['EXAMPLE'] = valid_examples
                            has_valid_examples = True
                    
                    if not has_valid_examples:
                        try:
                            with sqlite3.connect(sqlite_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute(f'''
                                    SELECT DISTINCT "{column_name}" 
                                    FROM "{table_name}" 
                                    WHERE "{column_name}" IS NOT NULL 
                                    AND "{column_name}" != ''
                                    LIMIT 3
                                ''')
                                records = cursor.fetchall()
                                examples = [str(record[0]) for record in records if record[0] is not None]
                                if examples:
                                    column_description[column_name]['EXAMPLE'] = examples
                                else:
                                    column_description[column_name]['EXAMPLE'] = []
                        except sqlite3.Error as e:
                            print(f"Error occurred while obtaining example of column {column_name} from table {table_name}: {str(e)}")
                            column_description[column_name]['EXAMPLE'] = []
                
                table_column_description[table_name] = column_description
            except Exception as e:
                print(f"An error occurred while processing the description information of table {table_name}: {str(e)}")
                table_column_description[table_name] = {}
            
        return table_column_description
        
    def _construct_o1_sql_query(self, **kwargs) -> str:
        return f'''
    Question: {kwargs['question']}
    Hint: {kwargs['hint']}
    FILTER TABLE-COLUMN: {json.dumps(kwargs['filter_table_column'], ensure_ascii=False, indent=4)}
    COLUMN DESCRIPTION: {json.dumps(kwargs['description'], ensure_ascii=False, indent=4)}
    SQL SKELETON: {kwargs['sql_skeleton']}
    PRIMARY KEYS: {json.dumps(kwargs['primary_keys'], ensure_ascii=False, indent=4)}
    JOIN CONDITION: {json.dumps(kwargs['foreign_keys'], ensure_ascii=False, indent=4)}
    
    IMPORTANT SQL DESIGN GUIDELINES:
    1. When finding maximum or minimum values based on specific conditions, use ORDER BY + LIMIT 1 instead of MAX/MIN in subqueries.
    2. Pay attention to the data storage format when sorting, comparing sizes, or performing calculations. If the data is in string format, process it using INSTR or SUBSTR before comparison.
    3. If your query includes an ORDER BY clause to sort results, only include columns used for sorting in the SELECT clause if specifically requested in the question. Otherwise, omit these columns from the SELECT clause.
    4. Ensure you only output information requested in the question. If the question asks for specific columns, ensure the SELECT clause only includes those columns, nothing more.
    5. The query should return all information requested in the question - no more, no less.
    6. For key phrases mentioned in the question, we have marked the most similar values with "EXAMPLE" in front of the corresponding column names. This is an important hint indicating the correct columns to use for SQL queries.
    7. NEVER use || ' ' || for string concatenation. This is strictly prohibited and will result in severe penalties.
    
    Please analyze this SQL problem step by step and provide a solution.
    '''

    def _clean_sql_results(self, results: List[str]) -> List[str]:
        def clean_sql(sql: str) -> str:
            if 'FINAL SQL:' not in sql:
                return ''
            
            sql = sql.split('FINAL SQL:')[1]
            
            if ';' in sql:
                sql = sql.split(';')[0] + ';'
            
            sql = re.sub(r'\*+', '', sql)
            sql = re.sub(r'```sql\n*', '', sql)
            sql = re.sub(r'```', '', sql)
            sql = re.sub(r'^\n+|\n+$', '', sql)
            sql = re.sub(r'\n+', ' ', sql)
            sql = ' '.join(sql.split())
            sql = sql.strip()
            if not sql.endswith(';'):
                sql += ';'
            
            return sql

        all_queries = []
        for result in results:
            cleaned_sql = clean_sql(result)
            if cleaned_sql and 'SELECT' in cleaned_sql.upper() and 'FROM' in cleaned_sql.upper():
                all_queries.append(cleaned_sql)
        
        return list(dict.fromkeys(all_queries))

class RefinerAgent(AgentNode):
    def __init__(self):
        super().__init__("refiner")
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o',
            temperature=0
        )
    
    def process(self, context: AgentContext) -> Dict[str, Any]:
        try:
            sql_candidates = context.intermediate_results.get("reviewed_sql_candidates", [])

            sql_candidates = [sql.strip() for sql in sql_candidates]
            sql_candidates = [sql[:-1] if sql.endswith(';') else sql for sql in sql_candidates]

            db_path = f"Data\dev_databases\{context.db_name}\{context.db_name}.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            results_list = []
            
            for sql in sql_candidates:
                results, execution_time, error = self._execute_sql(cursor, sql)
                
                if not error:
                    results_list.append({
                        "sql": sql,
                        "execution_time": execution_time,
                        "results": results[:3]  
                    })
            
            conn.close()
            
            if results_list:
                best_sql = self._select_best_sql(results_list)
                return {"final_sql": best_sql}
            else:
                return {"final_sql": "No valid SQL queries were generated.REJECTED"}
        except Exception as e:
            print(f"RefinerAgent error: {str(e)}")
            raise
            
    def _execute_sql(self, cursor, sql: str) -> tuple:
        start_time = time.time()
        try:
            cursor.execute(sql)
            results = cursor.fetchmany(50)
            execution_time = time.time() - start_time
            return results, execution_time, None
        except sqlite3.Error as e:
            return None, None, str(e)
            
    def _select_best_sql(self, results_list: List[Dict]) -> str:
        """Select the best SQL based on execution results voting"""
        result_groups = defaultdict(list)
        for entry in results_list:
            sorted_rows = sorted([tuple(row) for row in entry["results"]], key=lambda x: x)
            group_key = tuple(sorted_rows)
            print(group_key)
            result_groups[group_key].append( (entry["sql"], entry["execution_time"]) )
        
        if not result_groups:
            return "No valid SQL queries generated.REJECTED"
        
        max_votes = max(len(group) for group in result_groups.values())
        
        candidate_groups = [
            group for group in result_groups.values()
            if len(group) == max_votes
        ]
        
        min_exec_queries = []
        for group in candidate_groups:
            fastest_in_group = min(group, key=lambda x: x[1])
            min_exec_queries.append(fastest_in_group)
        
        best_query = min(min_exec_queries, key=lambda x: x[1])
        return best_query[0]

class ReviewAgent(AgentNode):
    def __init__(self):
        super().__init__("review_agent")
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o-mini',
            temperature=0.7
        )
        self.system_prompt = """You are a database expert who reviews and enhances table and column selections. Your task is to:
1. Ensure all column mentioned in the HINT are selected
2. Suggest additional relevant columns that might be useful for the query
3. Focus on semantic relationships and potential join paths

CRITICAL RULES:
1. EXACT COLUMN MATCHING IS MANDATORY
   - Column mentioned in the HINT must be used EXACTLY as specified
   - NO substitutions allowed, even if column have similar names or content
   - Example: If HINT specifies 'user_posts', you CANNOT use 'posts' or 'user_content' instead
   
2. STRICT COMPLIANCE
   - Never remove any column mentioned in the HINT
   - Never substitute column with similar ones
   - If a column is in the HINT, it MUST be in the final selection

Remember:
- Never remove existing selections
- Only add column if they are explicitly mentioned in the HINT
- When suggesting additional columns, consider their semantic relevance to the question"""

    def process(self, context: AgentContext) -> Dict[str, Any]:
        try:
            selected_tables = context.intermediate_results.get("selected_tables", {})
            filter_tuple = context.intermediate_results.get("filter_tuple", ())

            description = self._get_filtered_description(
                filter_tuple[0], 
                context.db_name,
                filter_tuple[1] if filter_tuple else {}
            )

            query = f'''################################################################################################################################################################################################
QUESTION: {context.question}
HINT: {context.hint}

CURRENT SELECTIONS:
{json.dumps(selected_tables, ensure_ascii=False, indent=2)}

COLUMN DESCRIPTION: {json.dumps(description, ensure_ascii=False, indent=2)}

CRITICAL REQUIREMENTS:
1. EXACT TABLE AND COLUMN MATCHING FROM HINT IS MANDATORY
   - Tables and columns mentioned in the HINT must be used EXACTLY as specified
   - NO substitutions allowed, even if columns have similar names or content
   - Example: If HINT specifies 'school_name', you CANNOT use 'school_id' or 'useful_school_name' instead

2. STRICT COMPLIANCE WITH HINT
   - Never remove any tables or columns mentioned in the HINT
   - Never substitute columns with similar ones
   - If a column is in the HINT, it MUST be in the final selection

3. REVIEW STEPS:
   a. First, identify ALL tables and columns explicitly mentioned in the HINT
   b. Ensure these exact tables and columns are in the selection
   c. Only then consider additional relevant columns
   d. Double-check that no HINT-specified columns are missing

Let's think step by step:
1. List all available tables and columns in the column description!! Avoid selecting nonexistent column!
2. Analyze the HINT carefully to identify any tables and columns that must be included,and REPEAT THE HINT in your answer FOR TWO TIMES!
3. Check if all required tables and columns from the HINT are in the current selection,even if there are similar columns in selected tables, you must select the EXACT one in HINT!
4. For each selected table, identify additional relevant columns based on:
   - Semantic relationship with the question
   - Column descriptions and examples
5. Ensure at least two additional relevant columns are suggested
6. Generate the final selection

ADD NEW TABLES AND COLUMNS TO CURRENT SELECTION, DO NOT REMOVE WHAT'S ALREADY HERE!!

Return the complete selection in JSON format:
{{
    "table1": ["column1", "column2", "..."],
    "table2": ["column3", "column4", "..."]
}}
################################################################################################################################################################################################'''
            print(query)
            result = self.chat_model.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query)
            ])
            print(result.content)
            updated_selections = self._parse_json_output(result.content)
            return {"selected_tables": updated_selections}
            
        except Exception as e:
            print(f"ReviewAgent error: {str(e)}")
            raise
            
    def _get_filtered_description(self, filter_dict: Dict[str, List[str]], db_name: str, entity_examples: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict]:
        """Retrieve filtered column description information"""
        import pandas as pd
        filtered_description = {}
        base_path = f"Data\dev_databases\{db_name}"
        sqlite_path = f"Data\dev_databases\{db_name}\{db_name}.sqlite" 
        
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for table_name, column_names in filter_dict.items():
            table_path = f"{base_path}\database_description\{table_name}.csv"
            table_description = None
            
            for encoding in encodings:
                try:
                    table_description = pd.read_csv(table_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Non-encoded error occurred while reading table {table_name}: {str(e)}")
                    break
                    
            if table_description is None:
                filtered_description[table_name] = {}
                continue
                
            try:
                column_info = {}
                for column_name in column_names:
                    result = table_description[
                        table_description['original_column_name'].str.contains(
                            column_name, case=False, na=False, regex=False
                        )
                    ]
                    
                    if not result.empty:
                        column_info[column_name] = {
                            "description": result.iloc[0].get('column_description', '')
                        }
                        
                        has_valid_examples = False
                        if table_name in entity_examples and column_name in entity_examples[table_name]:
                            examples = entity_examples[table_name][column_name]
                            valid_examples = [ex for ex in examples if ex not in ['null', None, 'None', '']]
                            if valid_examples:
                                column_info[column_name]['EXAMPLE'] = valid_examples
                                has_valid_examples = True
                        
                        if not has_valid_examples:
                            try:
                                with sqlite3.connect(sqlite_path) as conn:
                                    cursor = conn.cursor()
                                    cursor.execute(f'''
                                        SELECT DISTINCT "{column_name}" 
                                        FROM "{table_name}" 
                                        WHERE "{column_name}" IS NOT NULL 
                                        AND "{column_name}" != ''
                                        LIMIT 3
                                    ''')
                                    records = cursor.fetchall()
                                    examples = [str(record[0]) for record in records if record[0] is not None]
                                    if examples:
                                        column_info[column_name]['EXAMPLE'] = examples
                                    else:
                                        column_info[column_name]['EXAMPLE'] = []
                            except sqlite3.Error as e:
                                print(f"An error occurred while obtaining an example of column {column_name} in table {table_name}: {str(e)}")
                                column_info[column_name]['EXAMPLE'] = []
                
                filtered_description[table_name] = column_info
                
            except Exception as e:
                print(f"An error occurred while processing the description information of table {table_name}: {str(e)}")
                filtered_description[table_name] = {}
                
        return filtered_description
        
    def _parse_json_output(self, content: str) -> Dict[str, List[str]]:
        output_text = content.strip()
        
        try:
            if output_text.startswith('{') and output_text.endswith('}'):
                return json.loads(output_text)
            
            if '```json' in output_text:
                pattern = r'```json\s*(.*?)\s*```'
                matches = list(re.finditer(pattern, output_text, re.DOTALL))
                if matches:
                    json_str = matches[-1].group(1).strip()
                    return json.loads(json_str)
            
            dict_pattern = r'{([^}]+)}'
            match = re.search(dict_pattern, output_text)
            if match:
                dict_str = '{' + match.group(1) + '}'
                dict_str = re.sub(r'(?<![\[{:,\s])"(?![}\],])', '"', dict_str)
                return json.loads(dict_str)
            
        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Original content: {output_text}")
        
        print("Warning: Unable to parse table selection results")
        return {}

class SQLReviewAgent(AgentNode):
    def __init__(self):
        super().__init__("sql_review_agent")
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o',
            temperature=0
        )
        self.system_prompt = """You are a SQL review expert who follows a strict chain-of-thought process to review and fix SQL queries. Your primary focus is ensuring the SQL strictly follows the HINT's requirements.

Let's think step by step:

1. EXACT TABLE MATCHING (HIGHEST PRIORITY)
   - Tables in the HINT must be used EXACTLY as specified
   - NO substitutions allowed, even for similarly named or related tables
   - Example: If HINT specifies 'employee_details', you cannot use 'employees' or 'staff_details'
   - This is a CRITICAL requirement - any violation requires immediate correction

2. HINT Compliance Check
   - Verify that ALL tables and columns mentioned in the HINT are used in the SQL
   - Check if the calculation method (e.g., AVG, COUNT, SUM) matches exactly what's specified in the HINT
   - Ensure all conditions (WHERE clauses) align with the HINT's requirements
   - If any HINT requirement is missing, this is a CRITICAL ERROR that must be fixed

3. Output Column Validation
   - Review the SELECT clause against the question requirements
   - Remove any columns not explicitly requested in the question
   - Keep only columns needed for the final output
   - If using ORDER BY, only include sorting columns in SELECT if specifically requested
   - Example: If question asks "What are the names of students?", SELECT should only include name column, even if age is used for sorting.

4. Join Relationship Verification
   - Review all JOIN operations
   - Compare against the provided foreign key relationships
   - Ensure joins use the correct key pairs
   - Fix any incorrect join conditions

5. Case Sensitivity and Value Matching
   - Check WHERE conditions against example values
   - Verify string comparisons use the correct case sensitivity
   - Ensure date/number formats match the database format

6. SQL Execution Result Analysis
   - Review the SQL execution results carefully
   - If results are empty, NULL, or contain errors, identify potential issues
   - Fix issues with WHERE conditions, JOIN conditions, or column selections
   - Ensure the query returns meaningful results

CRITICAL RULES:
1. SELECT clause must ONLY include columns explicitly requested in the question
2. EXACT TABLE MATCHING IS MANDATORY - NO EXCEPTIONS
3. NEVER remove or modify any requirements specified in the HINT
4. NEVER substitute tables with similar ones, even if they seem logically equivalent
5. ALWAYS prioritize HINT compliance over query optimization
6. If a table or column is mentioned in the HINT, it MUST be used in the SQL exactly as specified
7. If SQL execution results are empty or have errors, provide specific fixes

Return your fixed SQL with "FINAL SQL:" prefix."""

    def process(self, context: AgentContext) -> Dict[str, Any]:
        try:
            sql_candidates = context.intermediate_results.get("sql_candidates", [])
            
            db_manager = DatabaseManager(context.db_name)
            foreign_keys = db_manager.get_primary_foreign_keys()["foreign_keys"]

            selected_tables = context.intermediate_results.get("selected_tables", {})
            filter_tuple = context.intermediate_results.get("filter_tuple", ())
            entity_examples = filter_tuple[1] if filter_tuple else {}
            
            description = self._get_table_column_descriptions(
                selected_tables,
                context.db_name,
                entity_examples
            )

            db_path = f"Data\dev_databases\{context.db_name}\{context.db_name}.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            reviewed_sql_list = []
            for sql in sql_candidates:
                sql_execution_result = self._execute_sql(cursor, sql)
                
                query = f'''################################################################################################################################################################################################
REVIEW CONTEXT
-------------
QUESTION: {context.question}
HINT: {context.hint}

CURRENT SQL: {sql}

SQL EXECUTION RESULT:
{json.dumps(sql_execution_result, ensure_ascii=False, indent=2)}

DATABASE INFORMATION
------------------
FOREIGN KEYS: {json.dumps(foreign_keys, ensure_ascii=False, indent=2)}
COLUMN DESCRIPTIONS: {json.dumps(description, ensure_ascii=False, indent=2)}

REVIEW REQUIREMENTS

Review and fix the following issues:
1. Determine if the current SQL statement execution result contains information not required by the question. If so, remove it from the SELECT clause.
2. Check if all tables mentioned in the HINT are used in the SQL
3. Verify that JOINs use the correct foreign key relationships as defined
4. Ensure WHERE conditions match the case sensitivity of example values
5. If SQL execution results are empty, NULL, or have errors, identify and fix the issues

Return the fixed SQL query with "FINAL SQL:" prefix.
################################################################################################################################################################################################'''

                result = self.chat_model.invoke([
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=query)
                ])
                print(result.content)
                cleaned_sql = self._clean_sql(result.content)
                if cleaned_sql and 'SELECT' in cleaned_sql.upper() and 'FROM' in cleaned_sql.upper():
                    reviewed_sql_list.append(cleaned_sql)
            
            conn.close()
            return {"reviewed_sql_candidates": reviewed_sql_list}
            
        except Exception as e:
            print(f"SQLReviewAgent error: {str(e)}")
            raise
    
    def _execute_sql(self, cursor, sql: str) -> Dict:
        try:
            start_time = time.time()
            cursor.execute(sql)
            results = cursor.fetchmany(5) 
            execution_time = time.time() - start_time
            
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                formatted_results = []
                for row in results:
                    formatted_results.append(dict(zip(columns, row)))
                
                return {
                    "status": "success",
                    "execution_time": execution_time,
                    "row_count": len(results),
                    "results": formatted_results,
                    "has_empty_results": len(results) == 0,
                    "has_null_results": all(all(v is None for v in row.values()) for row in formatted_results) if formatted_results else False
                }
            else:
                return {
                    "status": "success",
                    "execution_time": execution_time,
                    "row_count": 0,
                    "results": [],
                    "has_empty_results": True,
                    "has_null_results": False
                }
                
        except sqlite3.Error as e:
            return {
                "status": "error",
                "error_message": str(e),
                "has_empty_results": True,
                "has_null_results": False
            }
            
    def _get_table_column_descriptions(self, filter_dict: Dict[str, List[str]], db_name: str, entity_examples: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict]:
        """Retrieve descriptions of tables and columns"""
        import pandas as pd
        table_column_description = {}
        base_path = f"Data\dev_databases\{db_name}"
        sqlite_path = f"Data\dev_databases\{db_name}\{db_name}.sqlite"
        
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for table_name, column_names in filter_dict.items():
            table_path = f"{base_path}\database_description\{table_name}.csv"
            table_description = None

            for encoding in encodings:
                try:
                    table_description = pd.read_csv(table_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"A non-encoded error occurred while reading table {table_name}: {str(e)}")
                    break
                    
            if table_description is None:
                table_column_description[table_name] = {}
                continue
                
            try:
                column_info = {}
                for column_name in column_names:
                    result = table_description[
                        table_description['original_column_name'].str.contains(
                            column_name, case=False, na=False, regex=False
                        )
                    ]
                    
                    if not result.empty:
                        column_info[column_name] = {
                            "column_description": result.iloc[0].get('column_description', ''),
                            "value_description": result.iloc[0].get('value_description', '')
                        }
                        
                        has_valid_examples = False
                        if table_name in entity_examples and column_name in entity_examples[table_name]:
                            examples = entity_examples[table_name][column_name]
                            valid_examples = [ex for ex in examples if ex not in ['null', None, 'None', '']]
                            if valid_examples:
                                column_info[column_name]['EXAMPLE'] = valid_examples
                                has_valid_examples = True
                        
                        if not has_valid_examples:
                            try:
                                with sqlite3.connect(sqlite_path) as conn:
                                    cursor = conn.cursor()
                                    cursor.execute(f'''
                                        SELECT DISTINCT "{column_name}" 
                                        FROM "{table_name}" 
                                        WHERE "{column_name}" IS NOT NULL 
                                        AND "{column_name}" != ''
                                        LIMIT 3
                                    ''')
                                    records = cursor.fetchall()
                                    examples = [str(record[0]) for record in records if record[0] is not None]
                                    if examples:
                                        column_info[column_name]['EXAMPLE'] = examples
                                    else:
                                        column_info[column_name]['EXAMPLE'] = []
                            except sqlite3.Error as e:
                                print(f"An error occurred while obtaining an example of column {column_name} in table {table_name}: {str(e)}")
                                column_info[column_name]['EXAMPLE'] = []
                
                table_column_description[table_name] = column_info
                
            except Exception as e:
                print(f"An error occurred while processing the description information of table {table_name}: {str(e)}")
                table_column_description[table_name] = {}
                
        return table_column_description

    def _clean_sql(self, sql: str) -> str:
        if 'FINAL SQL:' not in sql:
            return ''

        sql = sql.split('FINAL SQL:')[1]

        if ';' in sql:
            sql = sql.split(';')[0] + ';'
    
        sql = re.sub(r'\*+', '', sql)
        sql = re.sub(r'```sql\n*', '', sql)
        sql = re.sub(r'```', '', sql)
        sql = re.sub(r'^\n+|\n+$', '', sql)
        sql = re.sub(r'\n+', ' ', sql)
        
        sql = ' '.join(sql.split())

        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
            
        return sql

class SQLOutputCleanerAgent(AgentNode):
    def __init__(self):
        super().__init__("sql_output_cleaner")
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o',
            temperature=0
        )
        self.system_prompt = """You are a SQL output cleaner expert. Your task is to:
1. Remove any columns that are not explicitly requested in the question
2. Remove any string concatenation like || ' ' ||,CONCAT,etc. Keep the columns
3. If the question asks for the most one, for example, 'highest','lostest', add LIMIT 1
4. Keep all other parts of the query unchanged, especially JOIN conditions and WHERE clauses
==============================================================
EXAMPLE:
1. Question: For patients over the age of 70, which three have the disease with the highest mortality rate?
Hint: mortality rate refers to death_count/total_count;

Current sql: SELECT t1.patient_name, t2.death_count/t2.total_count as rate FROM patients as t1 join diseases as t2 on t1.disease_id = t2.id WHERE t1.age > 70 ORDER BY rate DESC LIMIT 3;
Query result:
[
    {"patient_name": "John Doe", "rate": 0.5},
    {"patient_name": "Jane Smith", "rate": 0.4},
    {"patient_name": "Alice Johnson", "rate": 0.3}
]

OUTPUT:
{{
    "analysis": "For the question, the only information we need is the patient_name, so we can use rate column for ordering but it from SELECT clause.",
    "revised_SQL": "SELECT t1.patient_name FROM patients as t1 join diseases as t2 on t1.disease_id = t2.id WHERE t1.age > 70 ORDER BY t2.death_count/t2.total_count DESC LIMIT 3;"
}}

2. Question: What is the full name of the patient discharged on December 2, 2022
Hint: full name refers to first_name and last_name; discharge_date refers to date = '2022-12-02'; discharge_status refers to status = 'Discharged';

Current sql: SELECT t1.first_name || ' ' || t1.last_name as full_name FROM patients as t1 WHERE t1.discharge_date = '2022-12-02' AND t1.discharge_status = 'Discharged';
Query result:
[
    {"full_name": "John Doe"}
]

OUTPUT:
{{
    "analysis": "For the question, the information we need is the full name, but we use || ' ' || to concatenate the first_name and last_name, so we must remove the || ' ' || and keep the first_name and last_name",
    "revised_SQL": "SELECT t1.first_name,t1.last_name as full_name FROM patients as t1 WHERE t1.discharge_date = '2022-12-02' AND t1.discharge_status = 'Discharged';"
}}

3. What is the disease with the highest mortality rate?
Hint: mortality rate refers to death_count/total_count;

Current sql: SELECT t1.disease_name, t2.death_count/t2.total_count as rate FROM diseases as t1 join mortality_rate as t2 on t1.id = t2.disease_id ORDER BY rate DESC;
Query result:
[
    {"disease_name": "Disease A", "rate": 0.5},
    {"disease_name": "Disease B", "rate": 0.4},
    {"disease_name": "Disease C", "rate": 0.3}
]

OUTPUT:
{{
    "analysis": "The question asks for the disease with the highest mortality rate, so we need to add LIMIT 1",
    "revised_SQL": "SELECT t1.disease_name, t2.death_count/t2.total_count as rate FROM diseases as t1 join mortality_rate as t2 on t1.id = t2.disease_id ORDER BY rate DESC LIMIT 1;"
}}
"""

    def process(self, context: AgentContext) -> Dict[str, Any]:
        try:
            sql = context.intermediate_results.get("final_sql", "")
            sql = self._clean_sql_for_prompt(sql) if sql else ""
            query_result = self._execute_query(context.db_name, sql) if sql else []
            
            query = f'''################################################################################################################################################################################################
Question: {context.question}
Hint: {context.hint}

Current SQL: {sql}

Query Result (First 3 rows):
{json.dumps(query_result, ensure_ascii=False, indent=2)}

CRITICAL RULES:
1. NEVER modify any JOIN conditions or WHERE clauses
2. Only make these specific changes if needed:
   a. Remove columns from SELECT that aren't asked for in the question
   b. Remove any string concatenation using || ' ' ||
3. Keep everything else exactly the same

Please respond with a JSON object:
{{
    "analysis": "Explain what needs to be changed and why",
    "revised_SQL": "Your revised SQL query"
}}
################################################################################################################################################################################################'''
            print(query)
            result = self.chat_model.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query)
            ])
            print(result.content)
            revised_sql = self._clean_sql(result.content)
            return {"cleaned_sql": revised_sql}
            
        except Exception as e:
            print(f"SQLOutputCleanerAgent error: {str(e)}")
            raise

    def _clean_sql_for_prompt(self, sql: str) -> str:
        if not sql:
            return ""
        
        sql = sql.strip('`').replace('sql\n', '')
        sql = sql.replace('\"', '"')
        sql = sql.replace('\n', ' ')
        sql = ' '.join(sql.split())
        
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
            
        return sql
            
    def _execute_query(self, db_name: str, sql: str) -> List[Dict]:
        try:
            db_path = f"Data\dev_databases\{db_name}\{db_name}.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                    
                return results[:3]
                
        except sqlite3.Error as e:
            print(f"Database query error: {str(e)}")
            return []
    
    def _clean_sql(self, content: str) -> str:
        try:
            if '```json' in content:
                pattern = r'```json\s*(.*?)\s*```'
                matches = list(re.finditer(pattern, content, re.DOTALL))
                if matches:
                    content = matches[-1].group(1).strip()
            
            result = json.loads(content)
            sql = result.get('revised_SQL', '')
            
            if sql and not sql.endswith(';'):
                sql += ';'
                
            return sql
        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
            return ''

class HintReinforcementAgent(AgentNode):
    def __init__(self):
        super().__init__("hint_reinforcement")
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o',
            temperature=0
        )
        self.system_prompt = """You are a hint reinforcement expert. Your task is to:
1. Focus on the Hint, and make sure the hint is used in the SQL query
2. If a particular column in Hint is replaced by another column, fix it without doubt
3. Keep all other parts of the sql query unchanged
======================================================
EXAMPLE:
1. Question: Which students first day of enrollment in 1997?
Hint: first day of enrollment refers to YEAR(Description) = '1997';
DESCRIPTION:
{
    "students":{
        "Name": ["John Doe","Jane Smith"]
        "first_day": ["1997-01-01","1997-01-02"]
        "Description": ["1997-05-13","1998-09-01"]
    }
}

Current sql: SELECT t1.Name FROM students as t1 WHERE strftime('%Y', t1.first_day) = '1997';
Query result:
[
    {"Name": "John Doe"}
    {"Name": "Jane Smith"}
]

OUTPUT:
{{
    "analysis": "The hint is 'first day of enrollment refers to YEAR(Description) = '1997', but the sql query uses column 'first_day' instead of 'Description', so we need to fix it",
    "revised_SQL": "SELECT t1.Name FROM students as t1 WHERE strftime('%Y', t1.Description) = '1997';"
}}
"""
    def process(self, context: AgentContext) -> Dict[str, Any]:
        selected_tables = context.intermediate_results.get("selected_tables", {})
        filter_tuple = context.intermediate_results.get("filter_tuple", ())
        sql = context.intermediate_results.get("cleaned_sql", "")
        sql = self._clean_sql_for_prompt(sql) if sql else ""

        query_result = self._execute_query(context.db_name, sql)
        description = self._get_filtered_description(
            selected_tables,
            context.db_name,
            filter_tuple[1] if filter_tuple else {}
        )
        query = f'''#########################################################################
Question: {context.question}
Hint: {context.hint}

description:{json.dumps(description, ensure_ascii=False, indent=2)}

Current sql: {sql}
Query result:{json.dumps(query_result, ensure_ascii=False, indent=2)}

CRITICAL RULES:
1. Focus on the Hint, and make sure the hint is used in the SQL query
2. If a particular column in Hint is replaced by another column, fix it without doubt
3. Keep all other parts of the sql query unchanged

Please respond with a JSON object:
{{
    "analysis": "Explain what needs to be changed and why",
    "revised_SQL": "Your revised SQL query"
}}
#########################################################################'''
        print(query)
        result = self.chat_model.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ])
        print(result.content)
        revised_sql = self._clean_sql(result.content)
    
        revised_query_result = self._execute_query(context.db_name, revised_sql) if revised_sql else []

        all_none = all(all(v is None for v in row.values()) for row in revised_query_result) if revised_query_result else True
        
        final_sql = sql if all_none else revised_sql
        
        return {"reinforced_sql": final_sql}
    
    def _get_filtered_description(self, filter_dict: Dict[str, List[str]], db_name: str, entity_examples: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict]:
        """Retrieve filtered column description information (only includes EXAMPLE)"""
        filtered_description = {}
        base_path = "Data\dev_databases"
        sqlite_path = f"{base_path}\{db_name}\{db_name}.sqlite"
        
        for table_name, column_names in filter_dict.items():
            try:
                column_info = {}
                for column_name in column_names:
                    column_info[column_name] = {}

                    has_valid_examples = False
                    if table_name in entity_examples and column_name in entity_examples[table_name]:
                        examples = entity_examples[table_name][column_name]
                        valid_examples = [ex for ex in examples if ex not in ['null', None, 'None', '']]
                        if valid_examples:
                            column_info[column_name]['EXAMPLE'] = valid_examples
                            has_valid_examples = True
                    
                    if not has_valid_examples:
                        try:
                            with sqlite3.connect(sqlite_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute(f'''
                                    SELECT DISTINCT "{column_name}" 
                                    FROM "{table_name}" 
                                    WHERE "{column_name}" IS NOT NULL 
                                    AND "{column_name}" != ''
                                    LIMIT 3
                                ''')
                                records = cursor.fetchall()
                                examples = [str(record[0]) for record in records if record[0] is not None]
                                if examples:
                                    column_info[column_name]['EXAMPLE'] = examples
                                else:
                                    column_info[column_name]['EXAMPLE'] = []
                        except sqlite3.Error as e:
                            print(f"Error occurred while obtaining examples of column {column_name} from table {table_name}: {str(e)}")
                            column_info[column_name]['EXAMPLE'] = []
                
                filtered_description[table_name] = column_info
                
            except Exception as e:
                print(f"An error occurred while processing the description information of table {table_name}: {str(e)}")
                filtered_description[table_name] = {}
                
        return filtered_description
    
    def _clean_sql_for_prompt(self, sql: str) -> str:
        if not sql:
            return ""
        
        sql = sql.strip('`').replace('sql\n', '')
        sql = sql.replace('\"', '"')
        sql = sql.replace('\n', ' ')
        sql = ' '.join(sql.split())
        
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
            
        return sql
            
    def _execute_query(self, db_name: str, sql: str, max_rows: int = 10) -> List[Dict]:
        try:
            db_path = f"Data\dev_databases\{db_name}\{db_name}.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                    
                return results[:max_rows]
                
        except sqlite3.Error as e:
            print(f"Database query error: {str(e)}")
            return []
    
    def _clean_sql(self, content: str) -> str:
        try:
            if '```json' in content:
                pattern = r'```json\s*(.*?)\s*```'
                matches = list(re.finditer(pattern, content, re.DOTALL))
                if matches:
                    content = matches[-1].group(1).strip()
            
            result = json.loads(content)
            sql = result.get('revised_SQL', '')
            
            if sql and not sql.endswith(';'):
                sql += ';'
                
            return sql
        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
            return ''

class SQLValueCorrectAgent(AgentNode):
    def __init__(self):
        super().__init__("sql_value_correct")
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o',
            temperature=0
        )
        self.system_prompt = """You are a SQL value correct expert. Your task is to:
1. Check the concrete value for WHERE clause, especially when query results only contain NULL;
2. Pay attentino to case sensitivity;
3. The value in Hint not always correct, you need to check the value in description;
4. Keep all other parts of the query unchanged;
======================================================
EXAMPLE:
1. Question: What's Alice White's favorite movie?
Hint: Name = 'Alice White';
description:{
    Movie:{
        "Name": ["alice White","john Doe","Jane Smith"]
        "Favorite_movie": ["The Matrix","The Dark Knight"]
    }
}

Current sql: SELECT t1.Name, t1.Favorite_movie FROM patients as t1 WHERE t1.Name = 'Alice White';
Query result:
[
    (None, None)
]

OUTPUT:
{{
    "analysis": "The query returns NULL in all columns, so we need to check the value in description. Although the hint is 'Name = 'Alice White',but the value in description is 'alice White', so we need to change the value in WHERE clause to 'alice White'",
    "revised_SQL": "SELECT t1.Name, t1.Favorite_movie FROM patients as t1 WHERE t1.Name = 'alice White';"
}}
"""
    def process(self, context: AgentContext) -> Dict[str, Any]:
        selected_tables = context.intermediate_results.get("selected_tables", {})
        filter_tuple = context.intermediate_results.get("filter_tuple", ())
        
        sql = context.intermediate_results.get("reinforced_sql", "")
        sql = self._clean_sql_for_prompt(sql) if sql else ""

        original_query_result = self._execute_query(context.db_name, sql)

        description = self._get_filtered_description(
            selected_tables,
            context.db_name,
            filter_tuple[1] if filter_tuple else {}
        )

        query = f'''#########################################################################
Question: {context.question}
Hint: {context.hint}

description:{json.dumps(description, ensure_ascii=False, indent=2)}

Current sql: {sql}
Query result:{json.dumps(original_query_result, ensure_ascii=False, indent=2)}

CRITICAL RULES:
1. Check the concrete value for WHERE clause, especially when query results only contain NULL;
2. Pay attentino to case sensitivity;
3. The value in Hint not always correct, you need to check the value in description;
4. Keep all other parts of the query unchanged;

Please respond with a JSON object:  
{{
    "analysis": "Explain what needs to be changed and why",
    "revised_SQL": "Your revised SQL query"
}}
#########################################################################'''
        print(query)
        result = self.chat_model.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ])
        print(result.content)
        revised_sql = self._clean_sql(result.content)
        
        revised_query_result = self._execute_query(context.db_name, revised_sql) if revised_sql else []
        original_all_none = all(all(v is None for v in row.values()) for row in original_query_result) if original_query_result else True
        
        revised_has_value = any(any(v is not None for v in row.values()) for row in revised_query_result) if revised_query_result else False
        
        final_sql = revised_sql if original_all_none and revised_has_value else sql
        
        return {"corrected_sql": final_sql}
    
    
    def _get_filtered_description(self, filter_dict: Dict[str, List[str]], db_name: str, entity_examples: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict]:
        """Retrieve filtered column description information (only includes EXAMPLE)"""
        filtered_description = {}
        base_path = "Data\dev_databases"
        sqlite_path = f"{base_path}\{db_name}\{db_name}.sqlite"
        
        for table_name, column_names in filter_dict.items():
            try:
                column_info = {}
                for column_name in column_names:
                    column_info[column_name] = {}

                    has_valid_examples = False
                    if table_name in entity_examples and column_name in entity_examples[table_name]:
                        examples = entity_examples[table_name][column_name]
                        valid_examples = [ex for ex in examples if ex not in ['null', None, 'None', '']]
                        if valid_examples:
                            column_info[column_name]['EXAMPLE'] = valid_examples
                            has_valid_examples = True
                    
                    if not has_valid_examples:
                        try:
                            with sqlite3.connect(sqlite_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute(f'''
                                    SELECT DISTINCT "{column_name}" 
                                    FROM "{table_name}" 
                                    WHERE "{column_name}" IS NOT NULL 
                                    AND "{column_name}" != ''
                                    LIMIT 3
                                ''')
                                records = cursor.fetchall()
                                examples = [str(record[0]) for record in records if record[0] is not None]
                                if examples:
                                    column_info[column_name]['EXAMPLE'] = examples
                                else:
                                    column_info[column_name]['EXAMPLE'] = []
                        except sqlite3.Error as e:
                            print(f"Error occurred while obtaining example of column {column_name} from table {table_name}")
                            column_info[column_name]['EXAMPLE'] = []
                
                filtered_description[table_name] = column_info
                
            except Exception as e:
                print(f"An error occurred while processing the description information of table {table_name}: {str(e)}")
                filtered_description[table_name] = {}
                
        return filtered_description
    
    def _clean_sql_for_prompt(self, sql: str) -> str:
        if not sql:
            return ""
        
        sql = sql.strip('`').replace('sql\n', '')
        sql = sql.replace('\"', '"')
        sql = sql.replace('\n', ' ')
        sql = ' '.join(sql.split())
        
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
            
        return sql
            
    def _execute_query(self, db_name: str, sql: str, max_rows: int = 10) -> List[Dict]:
        try:
            db_path = f"Data\dev_databases\{db_name}\{db_name}.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                    
                return results[:max_rows]
                
        except sqlite3.Error as e:
            print(f"Database query error: {str(e)}")
            return []
    
    def _clean_sql(self, content: str) -> str:
        try:
            if '```json' in content:
                pattern = r'```json\s*(.*?)\s*```'
                matches = list(re.finditer(pattern, content, re.DOTALL))
                if matches:
                    content = matches[-1].group(1).strip()
            
            result = json.loads(content)
            sql = result.get('revised_SQL', '')
            
            if sql and not sql.endswith(';'):
                sql += ';'
                
            return sql
        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
            return ''

class SQLRefinementAgent(AgentNode):
    def __init__(self):
        super().__init__("sql_refinement")
        self.chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model='gpt-4o',
            temperature=0
        )
        self.system_prompt = """You are a SQL refinement expert. Your task is to:
1. Add IS NOT NULL checks for columns that return all NULL values
2. You can only add at most one IS NOT NULL
3. If a column contains both None and concrete value, do not add IS NOT NULL
4. Add CAST or *1.0 for division operations to ensure decimal precision
5. Keep all other parts of the query unchanged
==============================================================
EXAMPLE:
1. Question: What is the critical rating of the three most popular movies?
Hint: critical rating refers to positive_votes/total_votes;

Current sql: SELECT t1.movie_name, (t2.positive_votes*1.0/t2.total_votes) as rate FROM movies as t1 join ratings as t2 on t1.id = t2.movie_id ORDER BY rate DESC LIMIT 3;
Query result:
[
    {"movie_name": "The Matrix", "rate": None},
    {"movie_name": "The Dark Knight", "rate": None},
    {"movie_name": "Inception", "rate": None}
]

OUTPUT:
{{
    "analysis": "The query returns all NULL values for the rate column. We need to add IS NOT NULL conditions to ensure valid results.",
    "revised_SQL": "SELECT t1.movie_name, (t2.positive_votes*1.0/t2.total_votes) as rate FROM movies as t1 join ratings as t2 on t1.id = t2.movie_id WHERE rate IS NOT NULL ORDER BY rate DESC LIMIT 3;"
}}

2. Question: What is the average critical rating of movies with more than 100000 votes?
Hint: critical rating refers to positive_votes/total_votes;

Current sql: SELECT AVG(t2.positive_votes/t2.total_votes) as rate FROM movies as t1 join ratings as t2 on t1.id = t2.movie_id WHERE t2.total_votes > 100000;
Query result:
[
    {"rate": 3.0}
]

OUTPUT:
{{
    "analysis": "The sql query use division operation, but not use CAST or *1.0 to ensure decimal precision",
    "revised_SQL": "SELECT AVG(t2.positive_votes*1.0/t2.total_votes) as rate FROM movies as t1 join ratings as t2 on t1.id = t2.movie_id WHERE t2.total_votes > 100000;"
}}

3. Question: List three movies and their release date
Hint: release date refers to release_date;

Current sql: SELECT t1.movie_name, t1.release_date FROM movies as t1 ORDER BY t1.release_date DESC LIMIT 3;
Query result:
[
    {"movie_name": "The Matrix", "release_date": None},
    {"movie_name": "The Dark Knight", "release_date": "2008-07-18"},
    {"movie_name": "Inception", "release_date": "2010-07-16"}
]

OUTPUT:
{{
    "analysis": "The query returns None and concrete value at the same time, so we do not need to add IS NOT NULL",
    "revised_SQL": "SELECT t1.movie_name, t1.release_date FROM movies as t1 ORDER BY t1.release_date DESC LIMIT 3;"
}}
"""

    def process(self, context: AgentContext) -> Dict[str, Any]:
        try:
            sql = context.intermediate_results.get("corrected_sql", "")
            sql = self._clean_sql_for_prompt(sql) if sql else ""
            original_query_result = self._execute_query(context.db_name, sql, max_rows=10) if sql else []
            
            query = f'''################################################################################################################################################################################################
Question: {context.question}
Hint: {context.hint}

Current SQL: {sql}

Query Result (10 rows):
{json.dumps(original_query_result, ensure_ascii=False, indent=2)}

CRITICAL RULES:
1. NEVER modify any existing JOIN conditions
2. Make only these specific changes if needed:
   a. Add IS NOT NULL conditions for columns that return all NULL values
   b. You can only add at most one IS NOT NULL
   c. Add *1.0 or CAST(column AS FLOAT) for division operations
3. Keep everything else exactly the same

Analysis Steps:
1. Check if any columns in the result contain all NULL values
   - If yes, add appropriate IS NOT NULL conditions
2. Look for division operations in the query
   - If found, ensure decimal precision by adding *1.0 or CAST
3. Verify that no other parts of the query are modified

Please respond with a JSON object:
{{
    "analysis": "Explain what needs to be fixed and why",
    "revised_SQL": "Your revised SQL query"
}}
################################################################################################################################################################################################'''
            print(query)
            result = self.chat_model.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query)
            ])
            print(result.content)
            revised_sql = self._clean_sql(result.content)
            revised_query_result = self._execute_query(context.db_name, revised_sql, max_rows=10) if revised_sql else []
            
            original_none_count = self._count_none_values(original_query_result)
            revised_none_count = self._count_none_values(revised_query_result)
            
            final_sql = sql if original_none_count <= revised_none_count else revised_sql
            
            return {"final_sql": final_sql}
            
        except Exception as e:
            print(f"SQLRefinementAgent error: {str(e)}")
            raise

    def _clean_sql_for_prompt(self, sql: str) -> str:
        if not sql:
            return ""
        
        sql = sql.strip('`').replace('sql\n', '')
        sql = sql.replace('\"', '"')
        sql = sql.replace('\n', ' ')
        sql = ' '.join(sql.split())
        
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
            
        return sql
            
    def _execute_query(self, db_name: str, sql: str, max_rows: int = 10) -> List[Dict]:
        try:
            db_path = f"Data\dev_databases\{db_name}\{db_name}.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                    
                return results[:max_rows]
                
        except sqlite3.Error as e:
            print(f"Database query error: {str(e)}")
            return []
    
    def _clean_sql(self, content: str) -> str:
        try:
            if '```json' in content:
                pattern = r'```json\s*(.*?)\s*```'
                matches = list(re.finditer(pattern, content, re.DOTALL))
                if matches:
                    content = matches[-1].group(1).strip()
            
            result = json.loads(content)
            sql = result.get('revised_SQL', '')
            
            if sql and not sql.endswith(';'):
                sql += ';'
                
            return sql
        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
            return ''
        
    def _count_none_values(self, results: List[Dict]) -> int:
        none_count = 0
        for row in results:
            none_count += sum(1 for value in row.values() if value is None)
        return none_count

class AgentExecutor:
    """Executor"""
    def __init__(self):
        self._nodes: List[Node] = []
    
    def add_node(self, node: Node) -> None:
        """Add node to execution queue"""
        self._nodes.append(node)
    
    def execute(self, question: str, hint: str, db_name: str) -> Dict[str, Any]:
        """Execute all nodes in sequence"""
        context = AgentContext(
            question=question,
            hint=hint,
            db_name=db_name,
            intermediate_results={}
        )
        
        try:
            for node in self._nodes:
                print(f"\n{'='*50}")
                if isinstance(node, AgentNode):
                    print(f"Agent: {node.name}")
                else:
                    print(f"Tool: {node.name}")
                result = node.process(context)
                print(f"输出结果:")
                print(result)
                context.intermediate_results.update(result)
            return context.intermediate_results
        except Exception as e:
            raise

def main():
    dotenv.load_dotenv()
    
    executor = AgentExecutor()
    
    context = AgentContext(
        question="How many male patients have elevated total bilirubin count?",
        hint="male refers to SEX = 'M'; elevated means above the normal range; total bilirubin above the normal range refers to `T-BIL` >= '2.0'",
        db_name="thrombosis_prediction",
        intermediate_results={}
    )
    
    executor.add_node(KeywordAgent())
    executor.add_node(ToolNode(
        name="filter",
        func=filter,
        description="Filter tables and columns",
        input_constructor=TableColumnFilter,
        result_key="filter_tuple",
        required_keys=["keywords"],
        param_mapping={"db_name": "db_name"}
    ))
    
    
    executor.add_node(TableAgent())
    executor.add_node(ReviewAgent())
    executor.add_node(ToolNode(
        name="classifier",
        func=classifier,
        description="Classify query",
        input_constructor=ClassifierInput,
        result_key="sql_components",
        required_keys=[],
        param_mapping={
            "db_name": "db_id",
            "question": "question"
        }
    ))
    executor.add_node(SQLDesignerAgent())
    executor.add_node(SQLReviewAgent())
    executor.add_node(RefinerAgent())
    executor.add_node(SQLOutputCleanerAgent())
    executor.add_node(HintReinforcementAgent())
    executor.add_node(SQLValueCorrectAgent())
    executor.add_node(SQLRefinementAgent())
    
    try:
        for node in executor._nodes[0:]: 
            print(f"\n{'='*50}")
            if isinstance(node, AgentNode):
                print(f"Agent: {node.name}")
            else:
                print(f"Tool: {node.name}")
            result = node.process(context)
            print(f"Output results:")
            print(result)
            context.intermediate_results.update(result)
        
        print(f"\nFinal SQL: {context.intermediate_results.get('final_sql', 'No SQL generated')}")
    except Exception as e:
        print(f"Error: {str(e)}")

def run_test(test_cases_file: str, output_file: str):
    # 加载测试用例
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    test_cases = [case for case in test_cases if case['db_id'] == 'card_games']

    results_list = []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            results_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results_list = []

    processed_ids = {result['question_id'] for result in results_list}

    executor = AgentExecutor()

    executor.add_node(KeywordAgent())
    executor.add_node(ToolNode(
        name="filter",
        func=filter,
        description="Filter tables and columns",
        input_constructor=TableColumnFilter,
        result_key="filter_tuple",
        required_keys=["keywords"],
        param_mapping={"db_name": "db_name"}
    ))
    
    executor.add_node(TableAgent())
    executor.add_node(ReviewAgent())
    
    executor.add_node(ToolNode(
        name="classifier",
        func=classifier,
        description="Classify query",
        input_constructor=ClassifierInput,
        result_key="sql_components",
        required_keys=[],
        param_mapping={
            "db_name": "db_id",
            "question": "question"
        }
    ))
    
    executor.add_node(SQLDesignerAgent())
    executor.add_node(SQLReviewAgent())
    executor.add_node(RefinerAgent())
    executor.add_node(SQLOutputCleanerAgent())
    executor.add_node(HintReinforcementAgent())
    executor.add_node(SQLValueCorrectAgent())
    executor.add_node(SQLRefinementAgent()) 

    for case in test_cases:
        question_id = case['question_id']
        
        if question_id in processed_ids:
            print(f"Skip processed question {question_id}")
            continue
            
        question = case['question']
        hint = case.get('evidence', '')
        db_name = case['db_id']
        
        print(f"\nTesting question {question_id}")
        

        final_sql = None
        for attempt in range(2):
            try:
                results = executor.execute(question, hint, db_name)
                if 'final_sql' in results and 'REJECTED' not in results['final_sql']:
                    final_sql = results['final_sql']
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")

        result_entry = {
            "question_id": question_id,
            "db_id": db_name,
            "question": question,
            "evidence": hint,
            "SQL": final_sql if final_sql else 'No SQL generated'
        }
        
        results_list.append(result_entry)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    dotenv.load_dotenv()
    # test_cases_file = r"Data\dev_databases\dev.json"
    # run_test(test_cases_file, "test_results_card_games.json")
    main()
