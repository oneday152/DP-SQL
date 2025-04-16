KEYWORD_EXTRACTOR_PROMPT = '''################################################################################################################################################################################################
Objective: Analyze the given question and hint to identify and extract keywords, keyphrases, and named entities. These elements are crucial for understanding the core components of the inquiry and the guidance provided.

Instructions:
1. Read the Question Carefully: Understand the primary focus and specific details of the question. Look for any named entities (such as organizations, locations, etc.), technical terms, and other phrases that encapsulate important aspects of the inquiry.

2. Analyze the Hint: The hint is designed to direct attention toward certain elements relevant to answering the question. Extract any keywords, phrases, or named entities that could provide further clarity or direction in formulating an answer.

3. List Keyphrases and Entities: Combine your findings from both the question and the hint into a single list. This list should contain:
   - Keywords: Single words that capture essential aspects of the question or hint.
   - Keyphrases: Short phrases or named entities that represent specific concepts, locations, organizations, or other significant details.
   Ensure to maintain the original phrasing or terminology used in the question and hint.

4. Output Format: Return the keywords and keyphrases as a Python list. For example:
   ["keyword1", "keyphrase1", "keyword2", "keyphrase2"]

################################################################################################################################################################################################
Here are some examples:

Example 1:
Question: "What is the annual revenue of Acme Corp in the United States for 2022?"
Hint: "Focus on financial reports and U.S. market performance for the fiscal year 2022."
Output: ["annual revenue", "Acme Corp", "United States", "2022", "financial reports", "U.S. market performance", "fiscal year"]

Example 2:
Question: "In the Winter and Summer Olympics of 1988, which game has the most number of competitors?"
Hint: "the most number of competitors refer to MAX(COUNT(person_id));"
Output: ["Winter Olympics", "Summer Olympics", "1988", "number of competitors", "MAX(COUNT(person_id))", "person_id"]

Example 3:
Question: "How many Men's 200 Metres Freestyle events did Ian James Thorpe compete in?"
Hint: "Men's 200 Metres Freestyle events refer to event_name = 'Swimming Men''s 200 metres Freestyle';"
Output: ["Swimming Men's 200 metres Freestyle", "Ian James Thorpe", "event_name"]
'''

TABLE_SELECTOR_PROMPT = '''################################################################################################################################################################################################
### Objective
Identify tables and columns relevant to answering the given question using SQL. Return a dictionary of table names and columns necessary for constructing the SQL query through a systematic step-by-step process.

#### Step-by-Step Guidance

##### 1. Understand Question and Hint
- Analyze key information in the question
- Use hint for additional context
- Identify keywords that may point to specific tables or columns

##### 2. Examine Available Tables and Columns
- Review FILTER TABLE for complete list of tables and columns
- Identify tables with column names matching question or hint keywords
- Look for question keywords that might correspond to columns across multiple tables

##### 3. Use Column Descriptions to Verify Relevance
- Reference COLUMN DESCRIPTION for detailed column information
- Prioritize columns with similar example data when multiple columns match keywords
- Confirm selected tables contain information necessary for answering the question

##### 4. Generate Output
- Compile list of table names required to answer the question
- Ensure list is concise and contains only necessary tables

Output Format:
Return the result as a JSON dictionary containing only the names of relevant tables and columns. For example:

{'table1': ['column1', 'column2'], 'table2': ['column3', 'column4']}
################################################################################################################################################################################################
Example:

Question: "Find the average salary and department distribution of senior engineers working in tech companies in San Francisco"
Hint: "Consider job titles, company location, department information, and salary data"
FILTER TABLE: {
    "employees": ["emp_id", "name", "title", "salary", "dept_id"],
    "departments": ["dept_id", "dept_name", "location"],
    "companies": ["company_id", "name", "city", "industry"]
}
COLUMN DESCRIPTION: {
    "employees": {
        "title": {
            "original_column_name": "title",
            "column_description": "Employee position",
            "data_format": "text",
            "value_description": "Job title",
            "EXAMPLE": ["Senior Software Engineer", "Product Manager", "Data Scientist"]
        },
        "salary": {
            "original_column_name": "salary",
            "column_description": "Annual salary",
            "data_format": "integer",
            "value_description": "Salary amount",
            "EXAMPLE": [150000, 130000, 120000]
        },
        "dept_id": {
            "original_column_name": "dept_id",
            "column_description": "Department ID",
            "data_format": "text",
            "value_description": "Department code",
            "EXAMPLE": ["D001", "D002", "D003"]
        }
    },
    "departments": {
        "company_id": {
            "original_column_name": "company_id",
            "column_description": "the company id of the department belongs to",
            "data_format": "text",
            "value_description": "Company code",
            "EXAMPLE": ["C001", "C002", "C003"]
        },
        "dept_id": {
            "original_column_name": "dept_id",
            "column_description": "Department ID",
            "data_format": "text",
            "value_description": "Department code",
            "EXAMPLE": ["D001", "D002", "D003"]
        },
        "dept_name": {
            "original_column_name": "dept_name",
            "column_description": "Department name",
            "data_format": "text",
            "value_description": "Department name",
            "EXAMPLE": ["Engineering", "Product", "Data Science"]
        }
    },
    "companies": {
        "city": {
            "original_column_name": "city",
            "column_description": "Company location",
            "data_format": "text",
            "value_description": "City name",
            "EXAMPLE": ["San Francisco", "New York", "Seattle"]
        },
        "industry": {
            "original_column_name": "industry",
            "column_description": "Company industry",
            "data_format": "text",
            "value_description": "Industry type",
            "EXAMPLE": ["Technology", "Finance", "Healthcare"]
        },
        "name": {
            "original_column_name": "name",
            "column_description": "Company name",
            "data_format": "text",
            "value_description": "Company name",
            "EXAMPLE": ["Google", "Apple", "Microsoft"]
        },
        "id": {
            "original_column_name": "id",
            "column_description": "Company ID",
            "data_format": "text",
            "value_description": "Company code",
            "EXAMPLE": ["C001", "C002", "C003"]
        }
    },
    "companies_info_additional": {
        "company_id": {
            "original_column_name": "company_id",
            "column_description": "Company ID",
            "data_format": "text",
            "value_description": "Company code",
            "EXAMPLE": ["C001", "C002", "C003"]
        },
        "listed_or_not": {
            "original_column_name": "listed_or_not",
            "column_description": "Whether the company is listed",
            "data_format": "text",
            "value_description": "'0'means not listed,'1'means listed",
            "EXAMPLE": ["0", "1"]   
        },
        "listed_date": {
            "original_column_name": "listed_date",
            "column_description": "The date the company was listed",
            "data_format": "date",
            "value_description": "The date the company was listed",
            "EXAMPLE": ["2024-01-01", "2024-02-01", "2024-03-01"]
        },
        "listed_exchange": {
            "original_column_name": "listed_exchange",
            "column_description": "The exchange the company was listed on",
            "data_format": "text",
            "value_description": "The exchange the company was listed on",
            "EXAMPLE": ["NASDAQ", "NYSE", "AMEX"]
        }
    }
}

Answer:
Let's think step by step:
### **Step 1: Understand the Question and Hint**  
- **Question:** We need to find:  
  1. The senior engineers in tech companies in San Francisco.  
  2. The average salary of senior engineers.  
  3. The department distribution of these engineers.  
- **Hint:** Look for data related to:  
  1. Job titles (to identify "senior engineers").  
  2. Company location (to restrict to "San Francisco").  
  3. Department information (for department distribution).  
  4. Salary data (to calculate the average salary).  

---

### **Step 2: Examine Available Tables and Columns**  
From **FILTER TABLE**, the provided tables and columns are:  
1. **employees:**  
   - `title` (to identify "senior engineers").  
   - `salary` (to calculate the average salary).  
   - `dept_id` (to link employees to departments).  

2. **departments:**  
   - `company_id`: (to link departments to companies).  
   - `dept_id` (to link employees to departments).  
   - `name` (to get the names of departments).  

3. **companies:**  
   - `id` (to link departments to companies).  
   - `city` (to filter companies in "San Francisco").  
   - `industry` (to filter companies in the "Technology" industry).  
   - `name` (for potential company-level filtering, if required).  

There is no need to use the table `companies_info_additional` because the question is about senior engineers in tech companies in San Francisco, not about all companies.

---

### **Step 3: Use Column Descriptions to Verify Relevance**  
Using **COLUMN DESCRIPTION**,
The columns that are related to link tables are important:  

1. **From employees:**  
   - `title`: Includes "Senior Software Engineer," which matches our "senior engineers" filter.  
   - `salary`: Contains salary amounts needed for average calculation.  
   - `dept_id`: Matches table `departments` through `dept_id` to retrieve department information.  

2. **From departments:**  
   - `company_id`: Matches table `companies` through `id` to retrieve company information.  
   - `dept_id`: Matches table `employees` through `dept_id` to retrieve employee information.  
   - `name`: Provides department names for distribution analysis.  

3. **From companies:**  
   - `id`: Matches table `departments` through `company_id` to retrieve department information.  
   - `city`: Filters companies in "San Francisco."  
   - `industry`: Identifies "Technology" companies. According to the EXAMPLE in COLUMN DESCRIPTION, 'tech companies' is a type not a name of company.

---

### **Step 4: Generate Output**  
OUTPUT:
```json
{'employees': ['title', 'salary', 'dept_id'], 'departments': ['company_id', 'dept_id', 'name'], 'companies': ['id', 'city', 'industry']}
```

'''

MULTIVIEW_REPHRASING_PROMPT = '''################################################################################################################################################################################################
You are skilled at rephrasing questions to make them easier to understand and answer. Given a question, generate three different ways to restate it while maintaining the original content. 
################################################################################################################################################################################################
Output Format:
Your output should be a JSON object with two keys: "Question" and "Rephrased". The value of "Question" should be the original question, and the value of "Rephrased" should be an array of three different ways to rephrase the question while maintaining the original content.

Question: original question
Answer:
[{
  "Question": "original question",
  "Rephrased": [
    "rephrased question1",
    "rephrased question2",
    "rephrased question3"
  ]
}]

################################################################################################################################################################################################
Here are some examples:

Example 1:
Question: What are the potential impacts of artificial intelligence on employment?
[{
  "Question": "What are the potential impacts of artificial intelligence on employment?",
  "Rephrased": [
    "How might artificial intelligence affect employment in the future?",
    "In what ways could AI change the employment opportunities?",
    "What changes might we see in employment due to the rise of artificial intelligence?"
  ]
}]

Example 2:
Question: Among the customers with customer ID of 100 and below, how many of them have Thomas as their last name?
[{
  "Question": "Among the customers with customer ID of 100 and below, how many of them have Thomas as their last name?",
  "Rephrased": [
    "How many customers with a customer ID of 100 or less have the last name Thomas?",
    "What is the number of customers whose customer ID is 100 or below and have Thomas as their surname?",
    "Among customers with IDs up to 100, how many have Thomas as their last name?"
  ]
}]
'''

MULTIVIEW_SQL_DESIGN_PROMPT = '''################################################################################################################################################################################################  
Instructions  
### Instructions for Writing SQL Queries

#### 1. **Understand the Question and Hint:**  
   - Carefully analyze the `Question` and `Hint` to fully understand the requirements.  
   - Pay close attention to details such as logic, columns, and conditions.  

---

#### 2. **Use Column Descriptions to Identify Relevant Tables and Columns:**  
   - Cross-reference the `filter table-column` section with the `column description`.  
   - Select only the necessary tables and columns required to answer the question.  

---

#### 3. **Analyze and Expand the SQL Skeleton:**  
   - Use the provided `SQL SKELETON` keywords to construct the query, ensuring alignment with the question's requirements.  
   - If the keywords are insufficient, add appropriate SQL constructs.  

---

#### 4. **Simplification and Optimization SQL query**:  
   - Simplify sql without changing its original meaning
   - When there are too many subqueries and nesting is too complicated, use the JOIN clause to simplify the query.
   - ORDER BY + LIMIT 1 is preferred to MAX()/MIN()

---

#### 5. **Step-by-Step Thinking Process:**  
   - Break down the problem into smaller sub-questions and address them individually.  
   - Use pseudo-SQL to outline your approach for each sub-question.  

---

#### 6. **Provide the Final SQL Query:**  
   - Present the fully assembled SQL query in an optimized form.  
   - Format it as follows:  
     FINAL SQL:
     SELECT ... 
     FROM ... ;
    
################################################################################################################################################################################################

Question  
Question: What is the gender of the youngest client who opened an account in the lowest average salary branch?  
Hint: Later birthdate refers to younger age; A11 refers to average salary. Use the expression A11 / population to determine the lowest average salary branch.  
FILTER TABLE-COLUMN: {  
 "client": ["client_id", "gender", "birth_date", "district id", "email"],  
 "district": ["district id","district_name","population","A11"],  
 "transactions": ["transaction_id","client_id"],  
 "accounts": ["account_id","client_id","account_type"]  
}  
COLUMN DESCRIPTION:  
{  
    "client": {  
      "client id": {  
        "original_column_name": "client id",  
        "column_description": "Unique identifier for each client",  
        "data_format": "integer"  
        "EXAMPLE": [1, 2, 3]
      },  
      "gender": {  
        "original_column_name": "gender",  
        "column_description": "Gender of the client",  
        "data_format": "text"  
        "EXAMPLE": ["male", "female"]
      },  
      "birth_date": {  
        "original_column_name": "birth_date",  
        "column_description": "Birth date of the client, used to determine age",  
        "data_format": "date"  
        "EXAMPLE": ["2000-01-01", "2000-02-01", "2000-03-01"]
      },  
      "district id": {  
        "original_column_name": "district id",  
        "column_description": "Identifier linking the client to their district",  
        "data_format": "integer"  
        "EXAMPLE": [1, 2, 3]
      },  
      "email": {  
        "original_column_name": "email",  
        "column_description": "Email address of the client",  
        "data_format": "text"  
        "EXAMPLE": ["example@example.com", "example2@example.com", "example3@example.com"]
      }  
    },  
    "district": {  
      "district id": {  
        "original_column_name": "district id",  
        "column_description": "Unique identifier for each district",  
        "data_format": "integer"  
        "EXAMPLE": [1, 2, 3]
      },  
      "A11": {  
        "original_column_name": "A11",  
        "column_description": "Average salary in the district",  
        "data_format": "decimal"  
        "EXAMPLE": [10000, 20000, 30000]
      },  
      "district_name": {  
        "original_column_name": "district_name",  
        "column_description": "Name of the district",  
        "data_format": "text"  
        "EXAMPLE": ["District 1", "District 2", "District 3"]
      },  
      "population": {  
        "original_column_name": "population",  
        "column_description": "Population of the district",  
        "data_format": "integer"  
        "EXAMPLE": [10000, 20000, 30000]
      }  
    },  
    "transactions": {  
      "transaction_id": {  
        "original_column_name": "transaction_id",  
        "column_description": "Unique identifier for each transaction",  
        "data_format": "integer"  
        "EXAMPLE": [1, 2, 3]
      },  
      "client_id": {  
        "original_column_name": "client_id",  
        "column_description": "Identifier linking the transaction to a client",  
        "data_format": "integer"  
        "EXAMPLE": [1, 2, 3]
      }  
    },  
    "accounts": {  
      "account_id": {  
        "original_column_name": "account_id",  
        "column_description": "Unique identifier for each account",  
        "data_format": "integer"  
        "EXAMPLE": [1, 2, 3]
      },  
      "client_id": {  
        "original_column_name": "client_id",  
        "column_description": "Identifier linking the account to a client",  
        "data_format": "integer"  
        "EXAMPLE": [1, 2, 3]
      },  
      "account_type": {  
        "original_column_name": "account_type",  
        "column_description": "Type of account (e.g., savings, checking)",  
        "data_format": "text"  
        "EXAMPLE": ["savings", "checking"]
      }  
    }  
}  
SQL SKELETON: ['join','on','where','order_by','desc','limit','cast']  
PRIMARY KEYS: {
    "client": ["client_id"],
    "district": ["district_id"],
    "transactions": ["transaction_id"],
    "accounts": ["account_id"]
}
FOREIGN KEYS: [
    client."district_id"=district."district_id",
    transactions."client_id"=client."client_id",
    accounts."client_id"=client."client_id"
]

################################################################################################################################################################################################
1. Divide and Conquer:

   Main Question:
   What is the gender of the youngest client who opened an account in the lowest average salary branch?
   
   Analysis:
   The question asks about ‘gender’, which appears in the ‘client’ table. We will select this as the output column from the youngest client in the branch with the lowest average salary.
   
   Pseudo SQL:
   SELECT T1.gender 
   FROM client AS T1 
   WHERE youngest client in the lowest average salary branch;
   
   Sub-question 1: 
   Youngest client in the lowest average salary branch
   
   Analysis:
   To find the youngest client, use ‘A11’ from ‘district’ to get salary info, and ‘birth_date’ from ‘client’ to determine age. These tables can be INNER JOINed using "district id".
   
   Pseudo SQL:
   SELECT T1."client id" 
   FROM client AS T1 
   INNER JOIN district AS T2 ON T1."district id" = T2."district id" 
   WHERE lowest average salary branch 
   ORDER BY T1.birth_date DESC NULLS LAST LIMIT 1;
   
   Sub-question 1.1:
   Lowest average salary branch
   
   Analysis:
   Obtain the lowest average salary branch by ordering by ‘A11’ ASC and selecting the top 1. Since ‘A11’ is not NULLABLE, no "IS NOT NULL" filter is needed.
   
   Pseudo SQL:
   SELECT "district id" 
   FROM district 
   ORDER BY A11 ASC LIMIT 1;

2. Assembling SQL:

   Sub-question 1.1 (lowest average salary branch):
   SQL:
   SELECT "district id" 
   FROM district 
   ORDER BY A11 ASC LIMIT 1;

   Sub-question 1 (youngest client in the lowest average salary branch):
   SQL:
   SELECT T1."client id" 
   FROM client AS T1 
   INNER JOIN district AS T2 ON T1."district id" = T2."district id" 
   WHERE T2."district id" IN (SELECT "district id" FROM district ORDER BY A11 ASC LIMIT 1) 
   ORDER BY T1.birth_date DESC NULLS LAST LIMIT 1;

   Main Question (gender of the client):
   SQL:
   SELECT T1.gender 
   FROM client AS T1 
   WHERE T1."client id" = 
   (SELECT T1."client id" 
    FROM client AS T1 
    INNER JOIN district AS T2 ON T1."district id" = T2."district id" 
    WHERE T2."district id" IN (SELECT "district id" FROM district ORDER BY A11 ASC LIMIT 1) 
    ORDER BY T1.birth_date DESC NULLS LAST LIMIT 1);

3. Simplification and Optimization:

   Analysis:
   The nested queries can be combined using a single INNER JOIN to simplify and the filtering can be done within a single ORDER BY clause.
   We utilize ‘JOIN’, ‘ON’, ‘ORDER BY’, ‘DESC’, ‘LIMIT’ from the SQL skeleton to form the final optimized SQL query. Note that ‘WHERE’ is not mandatory in SQL skeleton, and ‘ASC’ and ‘NULLS LAST’ are used to sort clients by birth date in descending order.
   
   Optimized SQL Query:
   SELECT T1.gender 
   FROM client AS T1 
   INNER JOIN district AS T2 ON T1."district id" = T2."district id" 
   ORDER BY T2.A11 ASC, T1.birth_date DESC NULLS LAST LIMIT 1;
   
FINAL SQL:
SELECT T1.gender 
FROM client AS T1 
INNER JOIN district AS T2 ON T1."district id" = T2."district id" 
ORDER BY T2.A11 ASC, T1.birth_date DESC NULLS LAST LIMIT 1;
   
'''

FIX_SQL_PROMPT = '''Given a question with an SQL query and its corresponding error message, fix the SQL query so that it executes correctly. Return only the corrected SQL string.

Format:  
Question: The specific problem or task described in plain language.  
SQL Query: The provided SQL query.  
Error Message: The error message for the SQL query.  

Example:
Question: What is the average salary for each department?
SQL Query: SELECT department, AVG(salary) FROM employees GROUP BY departments;
Error Message: no such column: departments

Corrected SQL: SELECT department, AVG(salary) FROM employees GROUP BY department;
'''

SELECT_SQL_PROMPT = '''You are a SQL expert that selects the best query based on performance and correctness.
'''
