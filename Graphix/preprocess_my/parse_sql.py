import re
import numpy as np

def analyze_sql_to_vector(sql_query, components_list):

    sql_components = {
        'select': r'\bselect\b',
        'multi-select': r'\bselect\b.*?\bselect\b',
        'from': r'\bfrom\b',
        'join': r'\bjoin\b',
        'multi-join': r'\bjoin\b.*?\bjoin\b',
        'on': r'\bon\b',
        'where': r'\bwhere\b',
        'where-and': r'\bwhere\b.*?\band\b',
        'where-or': r'\bwhere\b.*?\bor\b',
        'where-between': r'\bwhere\b.*?\bbetween\b',
        'where-like': r'\bwhere\b.*?\blike\b',
        'where-in': r'\bwhere\b.*?\bin\b',
        'cast': r'\bcast\b',
        'case': r'\bcase\b',
        'union': r'\bunion\b',
        'exist': r'\bexists?\b',
        'group by': r'\bgroup\s+by\b',
        'order by': r'\border\s+by\b',
        'having': r'\bhaving\b',
        'distinct': r'\bdistinct\b',
        'desc': r'\bdesc\b',
        'asc': r'\basc\b',
        'limit': r'\blimit\b',
        'count': r'\bcount\s*\(',
        'sum': r'\bsum\s*\(',
        'avg': r'\bavg\s*\(',
        'min': r'\bmin\s*\(',
        'max': r'\bmax\s*\('
    }

    analysis_result = {key: False for key in sql_components}

    for component, pattern in sql_components.items():
        if re.search(pattern, sql_query, re.IGNORECASE | re.DOTALL):
            analysis_result[component] = True

    if analysis_result['where-and']:
        where_clause = re.search(r'\bwhere\b(.*)', sql_query, re.IGNORECASE | re.DOTALL)
        if where_clause:
            where_content = where_clause.group(1)
            all_and_matches = re.findall(r'\band\b', where_content, re.IGNORECASE)
            between_matches = re.findall(r'\bbetween\b.*?\band\b.*?', where_content, re.IGNORECASE)
            if len(all_and_matches) == len(between_matches):
                analysis_result['where-and'] = False

    vector = np.zeros(len(components_list))  
    for i, component in enumerate(components_list):
        if analysis_result.get(component, False):
            vector[i] = 1

    return vector

components_list = [
    'select', 'multi-select', 'from', 'join', 'multi-join', 'on', 'where', 'where-and', 'where-or', 
    'where-between', 'where-like', 'where-in', 'cast', 'case', 'union', 'exist', 'group by', 'order by', 
    'having', 'distinct', 'desc', 'asc', 'limit', 'count', 'sum', 'avg', 'min', 'max'
]

sql_query = """
SELECT AVG(T4.amount) FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id 
INNER JOIN account AS T3 ON T2.account_id = T3.account_id 
INNER JOIN loan AS T4 ON T3.account_id = T4.account_id 
WHERE T1.gender = 'M'
"""

vector = analyze_sql_to_vector(sql_query, components_list)
print("SQL 分析独热编码向量：")
print(vector)

