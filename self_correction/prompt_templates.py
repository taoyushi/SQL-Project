# self_correction/prompt_templates.py

from enum import Enum

class PromptStrategy(Enum):
    """定义Prompt构造策略"""
    GENERIC = "generic"  # 通用指令
    GUIDED = "guided"    # 基于反馈的指导性提示

# 默认的Prompt模板
DEFAULT_PROMPT_TEMPLATES = {
    "generic": """You are an expert SQL developer capable of identifying and correcting errors in SQL queries generated from natural language.
Given a natural language question, the database schema, and an initial SQL query, your task is to review the initial SQL and provide a corrected version that answers the question accurately based on the schema.

If the initial SQL is correct, return it as is. If it's incorrect, identify the errors and provide the corrected SQL.

Schema:
{schema_context}

Problem: {nlq}
Initial SQL: {initial_sql}
Hint: {hint}
Correct SQL:""",

    "guided": """You are an expert SQL developer capable of identifying and correcting errors in SQL queries generated from natural language.
Given a natural language question, the database schema, and an initial SQL query, your task is to review the initial SQL and provide a corrected version that answers the question accurately based on the schema.

Please pay close attention to the following hint, which is based on a preliminary validation of the initial SQL.

Schema:
{schema_context}

Problem: {nlq}
Initial SQL: {initial_sql}
Hint: {hint}
Correct SQL:"""
}

# 默认的Few-shot示例
DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "question": "List the names of cities with a population greater than 1 million.",
        "schema_text": """TABLE city (
  city_id INT PRIMARY KEY,
  city_name VARCHAR,
  state VARCHAR,
  country VARCHAR,
  population INT,
  area DECIMAL
);""",
        "initial_sql": "SELECT Name FROM city WHERE Population > 1000000",
        "hint": "The column name 'Name' is incorrect based on the schema. Use 'city_name'.",
        "correct_sql": "SELECT city_name FROM city WHERE population > 1000000"
    },
    {
        "question": "How many employees are there in the 'Sales' department?",
        "schema_text": """TABLE employee (
  emp_id INT PRIMARY KEY,
  emp_name VARCHAR,
  dept_id INT
);
TABLE department (
  dept_id INT PRIMARY KEY,
  dept_name VARCHAR
);
Foreign Keys:
- employee.dept_id references department.dept_id""",
        "initial_sql": "SELECT count(*) FROM employee WHERE dept_name = 'Sales'",
        "hint": "The initial SQL is missing a JOIN with the department table to access `dept_name`.",
        "correct_sql": "SELECT count(T1.emp_id) FROM employee AS T1 JOIN department AS T2 ON T1.dept_id = T2.dept_id WHERE T2.dept_name = 'Sales'"
    }
]