# self_correction/prompt_generator.py

from enum import Enum
from typing import Dict, Any, List, Optional
from .prompt_templates import PromptStrategy, DEFAULT_PROMPT_TEMPLATES, DEFAULT_FEW_SHOT_EXAMPLES

class PromptGenerator:
    """
    负责根据输入和策略构建发送给LLM的Prompt文本。
    """
    def __init__(self, prompt_templates: Optional[Dict[str, str]] = None, 
                 few_shot_examples: Optional[List[Dict[str, Any]]] = None):
        """
        初始化Prompt Generator。

        Args:
            prompt_templates (Optional[Dict[str, str]]): Prompt模板字典，key为策略名称。
            few_shot_examples (Optional[List[Dict[str, Any]]]): Few-shot示例列表。
        """
        self.prompt_templates = prompt_templates or DEFAULT_PROMPT_TEMPLATES.copy()
        self.few_shot_examples = few_shot_examples or DEFAULT_FEW_SHOT_EXAMPLES.copy()

        # Basic validation
        if PromptStrategy.GENERIC.value not in self.prompt_templates:
            raise ValueError(f"Prompt template '{PromptStrategy.GENERIC.value}' is missing.")
        if PromptStrategy.GUIDED.value not in self.prompt_templates:
            raise ValueError(f"Prompt template '{PromptStrategy.GUIDED.value}' is missing.")

    def generate_prompt(self, nlq: str, schema_context: str, initial_sql: str, 
                       strategy: PromptStrategy, hint: str) -> str:
        """
        根据输入信息和指定的策略生成Prompt。

        Args:
            nlq (str): 自然语言问题。
            schema_context (str): 格式化后的数据库Schema文本。
            initial_sql (str): RESDSQL生成的初步SQL。
            strategy (PromptStrategy): 生成Prompt的策略 (GENERIC 或 GUIDED).
            hint (str): 根据策略和验证结果生成的具体提示信息。

        Returns:
            str: 完整的Prompt文本。
        """
        prompt_template_key = strategy.value
        template = self.prompt_templates.get(prompt_template_key)

        if not template:
            # Fallback to generic template
            template = self.prompt_templates[PromptStrategy.GENERIC.value]
            print(f"[WARN] Using generic prompt template as '{prompt_template_key}' not found.")

        # Add few-shot examples if available
        few_shot_section = self._format_few_shot_examples() if self.few_shot_examples else ""

        # Fill the template
        full_prompt = template.format(
            nlq=nlq,
            schema_context=schema_context,
            initial_sql=initial_sql,
            hint=hint
        )

        return few_shot_section + full_prompt

    def _format_few_shot_examples(self) -> str:
        """
        将 Few-shot 示例格式化为 Prompt 的一部分。
        """
        if not self.few_shot_examples:
            return ""
            
        formatted_examples = ""
        example_template = """Problem: {question}
Schema:
{schema_text}
Initial SQL: {initial_sql}
Hint: {hint}
Correct SQL: {correct_sql}

---
"""

        for example in self.few_shot_examples:
            formatted_examples += example_template.format(
                question=example.get("question", ""),
                schema_text=example.get("schema_text", ""),
                initial_sql=example.get("initial_sql", ""),
                hint=example.get("hint", ""),
                correct_sql=example.get("correct_sql", "")
            )

        return formatted_examples + "\n"