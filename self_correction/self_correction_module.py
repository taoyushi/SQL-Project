# self_correction/self_correction_module.py
import re
import os
import logging
import json
import time
import random
from typing import Dict, Any, Tuple, Optional
from .prompt_generator import PromptGenerator, PromptStrategy
from .llm_api import QwenAPIClient, LLMError
from .sql_validator import SQLValidator, ExecutionResult
from .config import SelfCorrectionConfig

# Configure logging
logger = logging.getLogger(__name__)

class SelfCorrectionModule:
    """
    核心自修正模块，集成 Prompt Generation, LLM Interaction, SQL Validation,
    和 Fallback Logic to correct initial SQL queries.
    """

    def __init__(self, config: SelfCorrectionConfig):
        """
        初始化自修正模块。

        Args:
            config (SelfCorrectionConfig): 配置对象，包含 LLM API 设置、Prompt 策略等。
        """
        self.config = config
        
        # 验证配置
        if not config.validate_config():
            raise ValueError("Invalid configuration provided")
            
        self.prompt_generator = PromptGenerator(config.prompt_templates, config.few_shot_examples)
        self.llm_client = QwenAPIClient(
            api_key=config.llm_api_key,
            endpoint=config.llm_api_endpoint,
            model=config.llm_model_name,
            timeout=config.llm_timeout_seconds,
            max_retries=config.llm_max_retries
        )
        self.sql_validator = SQLValidator(timeout_seconds=30)
        self.log_dir = config.log_dir

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        logger.info(f"SelfCorrectionModule initialized with log directory: {self.log_dir}")
        
        # Test API connection
        if not self._test_api_connection():
            logger.warning("API connection test failed. Corrections may fail.")

    def _test_api_connection(self) -> bool:
        """测试API连接"""
        try:
            return self.llm_client.test_connection()
        except Exception as e:
            logger.error(f"API connection test error: {e}")
            return False

    def process(self, nlq: str, schema: Dict[str, Any], initial_sql: str, db_path: str) -> Tuple[str, str]:
        """
        智能处理：使用更智能的策略判断是否需要修正
        """
        db_id = os.path.basename(os.path.dirname(db_path)) if os.path.dirname(db_path) else "unknown"

        correction_log = {
            "question": nlq,
            "db_id": db_id,
            "initial_sql": initial_sql,
            "corrected_sql_llm": None,
            "final_sql_output": initial_sql,
            "initial_sql_executable": False,
            "corrected_sql_executable": False,
            "validation_error": None,
            "correction_status": "correction_failed",
            "prompt_used": None,
            "llm_api_log_id": None,
            "processing_time_ms": 0,
            "correction_reason": None
        }
        start_time = time.time()

        try:
            # 1. 首先验证初始SQL
            initial_validation_result = self._validate_sql(initial_sql, db_path)
            correction_log["initial_sql_executable"] = initial_validation_result.is_executable
            
            # 🎯 修改策略：智能判断是否需要修正
            should_attempt_correction, reason = self._should_attempt_correction(
                initial_sql, initial_validation_result, nlq
            )
            correction_log["correction_reason"] = reason
            
            if not should_attempt_correction:
                logger.info(f"Skipping correction for DB {db_id} - {reason}")
                correction_log["correction_status"] = "no_correction_needed"
                correction_log["final_sql_output"] = initial_sql
                self._save_log(correction_log)
                correction_log["processing_time_ms"] = (time.time() - start_time) * 1000
                return initial_sql, "no_correction_needed"
            
            # 2. 需要修正时，生成适当的Prompt
            logger.info(f"Attempting correction for DB {db_id} - {reason}")
            
            if not initial_validation_result.is_executable:
                hint = self._generate_guided_hint(initial_sql, initial_validation_result.error_message)
                prompt_strategy = PromptStrategy.GUIDED
            else:
                hint = self._generate_improvement_hint(initial_sql, nlq, reason)
                prompt_strategy = PromptStrategy.GUIDED

            # 3. 生成保守的Prompt
            schema_text = self._format_schema_for_prompt(schema)
            prompt = self.prompt_generator.generate_prompt(
                nlq=nlq,
                schema_context=schema_text,
                initial_sql=initial_sql,
                strategy=prompt_strategy,
                hint=hint
            )
            correction_log["prompt_used"] = prompt
            logger.debug(f"Generated prompt for DB {db_id}: {prompt[:500]}...")

            # 4. 调用 LLM API
            try:
                logger.info(f"Calling LLM for DB {db_id}...")
                llm_response_raw = self.llm_client.get_correction(
                    prompt=prompt,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                    stop_sequences=self.config.llm_stop_sequences
                )
                logger.debug(f"LLM response for DB {db_id}: {llm_response_raw[:500]}...")
            except LLMError as e:
                logger.error(f"LLM API call failed for DB {db_id}: {e}")
                correction_log["validation_error"] = f"LLM API Error: {e}"
                correction_log["correction_status"] = "api_error"
                self._save_log(correction_log)
                correction_log["processing_time_ms"] = (time.time() - start_time) * 1000
                return initial_sql, correction_log["correction_status"]

            # 5. 解析 LLM 响应
            corrected_sql = self._parse_llm_response(llm_response_raw)
            correction_log["corrected_sql_llm"] = corrected_sql
            logger.debug(f"Parsed corrected SQL for DB {db_id}: {corrected_sql}")

            corrected_validation_result = ExecutionResult(False, "No corrected SQL to validate")
            if corrected_sql:
                # 6. 验证修正后的 SQL
                corrected_validation_result = self._validate_sql(corrected_sql, db_path)
                correction_log["corrected_sql_executable"] = corrected_validation_result.is_executable
                if not corrected_validation_result.is_executable:
                    correction_log["validation_error"] = corrected_validation_result.error_message
                    logger.warning(f"Corrected SQL is not executable for DB {db_id}: {corrected_validation_result.error_message}")
            else:
                logger.warning(f"LLM returned empty or unparseable SQL for DB {db_id}")
                correction_log["validation_error"] = "LLM returned empty or unparseable SQL"

            # 7. 应用智能回退策略
            final_sql, correction_status = self._apply_intelligent_fallback_strategy(
                initial_sql,
                initial_validation_result,
                corrected_sql if corrected_sql else initial_sql,
                corrected_validation_result
            )
            correction_log["final_sql_output"] = final_sql
            correction_log["correction_status"] = correction_status
            logger.info(f"Final SQL decision for DB {db_id}: {correction_status}")

        except Exception as e:
            logger.error(f"Unexpected error during correction for DB {db_id}: {e}", exc_info=True)
            correction_log["validation_error"] = f"Unexpected internal error: {e}"
            correction_log["correction_status"] = "internal_error"
            final_sql = initial_sql
            correction_status = "internal_error"

        finally:
            correction_log["processing_time_ms"] = (time.time() - start_time) * 1000
            self._save_log(correction_log)
            return correction_log["final_sql_output"], correction_log["correction_status"]

    def _should_attempt_correction(self, sql: str, validation_result: ExecutionResult, nlq: str) -> Tuple[bool, str]:
        """智能判断是否应该尝试修正SQL"""
        
        # 1. 不可执行的SQL必须修正
        if not validation_result.is_executable:
            return True, "SQL不可执行"
        
        # 2. 可执行但返回空结果的SQL可能有问题  
        if hasattr(validation_result, 'result_count') and validation_result.result_count == 0:
            return True, "SQL可执行但返回空结果"
        
        # 3. 检查常见错误模式
        error_patterns = [
            (r'\bstudents\b', 'table_name_plural'),
            (r'\bsingers\b', 'table_name_plural'), 
            (r'\bName\b', 'column_name_case'),
            (r'\bPopulation\b', 'column_name_case'),
            (r'\bAge\b', 'column_name_case'),
            (r'FROM\s+\w+\s+WHERE\s+\w+\.\w+', 'possible_redundant_reference'),
        ]
        
        for pattern, error_type in error_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return True, f"发现可能错误模式: {error_type}"
        
        # 4. 基于问题类型的启发式判断
        nlq_lower = nlq.lower()
        
        if 'how many' in nlq_lower and 'count(' not in sql.lower():
            return True, "问题询问数量但SQL中无count函数"
        
        if 'average' in nlq_lower and 'avg(' not in sql.lower():
            return True, "问题询问平均值但SQL中无avg函数"
            
        if 'maximum' in nlq_lower and 'max(' not in sql.lower():
            return True, "问题询问最大值但SQL中无max函数"
            
        if 'minimum' in nlq_lower and 'min(' not in sql.lower():
            return True, "问题询问最小值但SQL中无min函数"
        
        # 5. 检查SQL复杂度和潜在改进点
        if self._has_potential_improvements(sql, nlq):
            return True, "发现潜在改进点"
        
        # 6. 随机采样修正（用于探索和测试）
        if random.random() < 0.02:  # 10%的概率尝试修正
            return True, "随机选择进行修正（探索模式）"
        
        return False, "SQL判定为无需修正"

    def _has_potential_improvements(self, sql: str, nlq: str) -> bool:
        """检查SQL是否有潜在改进点"""
        
        # 检查是否有不必要的复杂性
        if sql.count('SELECT') > 1 and len(sql) < 100:
            return True  # 短SQL中有多个SELECT可能可以简化
        
        # 检查是否缺少常见的优化
        if 'JOIN' in sql.upper() and 'ON' not in sql.upper():
            return True  # JOIN但没有ON条件
        
        # 检查列名和表名的一致性
        if 'SELECT *' in sql.upper() and any(word in nlq.lower() for word in ['name', 'id', 'count']):
            return True  # 问题可能需要特定列而不是所有列
        
        return False

    def _generate_improvement_hint(self, sql: str, nlq: str, reason: str) -> str:
        """为可执行但可能需要改进的SQL生成提示"""
        
        if "table_name_plural" in reason:
            return "Check if table names should be singular (e.g., 'student' not 'students')"
        elif "column_name_case" in reason:
            return "Check if column names should be lowercase"
        elif "count函数" in reason:
            return "The question asks 'how many', consider using COUNT() function"
        elif "平均值" in reason:
            return "The question asks for average, consider using AVG() function"
        elif "最大值" in reason:
            return "The question asks for maximum, consider using MAX() function"
        elif "最小值" in reason:
            return "The question asks for minimum, consider using MIN() function"
        elif "潜在改进" in reason:
            return "Review the SQL for potential optimizations or simplifications"
        else:
            return "Review and improve the SQL if possible, but be conservative"

    def _apply_intelligent_fallback_strategy(
        self,
        initial_sql: str,
        initial_validation_result: ExecutionResult,
        corrected_sql: str,
        corrected_validation_result: ExecutionResult
    ) -> Tuple[str, str]:
        """智能回退策略：综合考虑多个因素"""
        
        # 情况1: 修正SQL可执行，初始SQL不可执行 -> 使用修正SQL
        if (corrected_sql and 
            corrected_validation_result.is_executable and 
            not initial_validation_result.is_executable):
            logger.info("Using corrected SQL (initial was not executable, corrected is executable).")
            return corrected_sql, "corrected"
        
        # 情况2: 两个都可执行，比较复杂度和合理性
        if (corrected_sql and 
            corrected_validation_result.is_executable and 
            initial_validation_result.is_executable):
            
            # 简单的启发式：更短的SQL可能更好（但不绝对）
            if len(corrected_sql) < len(initial_sql) * 1.2:  # 修正后SQL不应该太长
                logger.info("Using corrected SQL (both executable, corrected seems better).")
                return corrected_sql, "corrected"
            else:
                logger.info("Using initial SQL (corrected SQL seems too complex).")
                return initial_sql, "fallback_corrected_too_complex"
        
        # 情况3: 修正SQL不可执行，但初始SQL可执行 -> 回退到初始SQL
        if (initial_validation_result.is_executable and 
            (not corrected_sql or not corrected_validation_result.is_executable)):
            logger.info("Using initial SQL (corrected SQL not executable or empty).")
            return initial_sql, "fallback_to_initial"
        
        # 情况4: 两个都不可执行 -> 使用初始SQL（至少是原始的）
        logger.warning("Both initial and corrected SQL are not executable.")
        return initial_sql, "both_not_executable"

    def _generate_guided_hint(self, initial_sql: str, validation_error: Optional[str]) -> str:
        """根据执行验证错误信息生成指导性 Hint"""
        if validation_error:
            error_lower = validation_error.lower()
            
            if "syntax error" in error_lower:
                return "Fix the SQL syntax error only. Do not change table or column names unless necessary."
            elif "no such table" in error_lower or "no such column" in error_lower:
                return "Fix the table/column name error only. Check the schema carefully."
            elif "ambiguous column name" in error_lower:
                return "Fix the ambiguous column reference by adding table alias."
            else:
                return f"Fix this specific error: {validation_error[:100]}. Make minimal changes."
        else:
            return "Fix any execution errors, but make minimal changes to the SQL structure."

    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """将结构化的Schema信息转换为详细的文本格式"""
        if not schema:
            return "No schema available"
            
        tables = schema.get('table_names_original', [])
        columns = schema.get('column_names_original', [])
        column_types = schema.get('column_types', [])
        primary_keys = schema.get('primary_keys', [])
        foreign_keys = schema.get('foreign_keys', [])
        
        if not tables or not columns:
            return "Schema incomplete"

        # 构建详细的表结构描述
        schema_text = "Database Schema:\n"
        
        # 按表分组列信息
        table_columns = {}
        for i, (table_id, column_name) in enumerate(columns):
            if table_id == -1:  # 跳过*占位符
                continue
            if table_id < len(tables):
                table_name = tables[table_id]
                if table_name not in table_columns:
                    table_columns[table_name] = []
                
                col_info = column_name
                if i < len(column_types):
                    col_info += f" {column_types[i].upper()}"
                if i in primary_keys:
                    col_info += " PRIMARY KEY"
                    
                table_columns[table_name].append(col_info)
        
        # 生成表定义
        for table_name, cols in table_columns.items():
            schema_text += f"\nTABLE {table_name} (\n"
            for col in cols:
                schema_text += f"  {col},\n"
            schema_text = schema_text.rstrip(',\n') + "\n);\n"
        
        # 添加外键关系
        if foreign_keys:
            schema_text += "\nForeign Keys:\n"
            for fk in foreign_keys:
                if len(fk) == 2 and fk[0] < len(columns) and fk[1] < len(columns):
                    try:
                        from_table = tables[columns[fk[0]][0]]
                        from_col = columns[fk[0]][1]
                        to_table = tables[columns[fk[1]][0]]
                        to_col = columns[fk[1]][1]
                        schema_text += f"- {from_table}.{from_col} references {to_table}.{to_col}\n"
                    except IndexError:
                        continue  # 跳过有问题的外键定义
        
        return schema_text.strip()

    def _parse_llm_response(self, response_text: str) -> Optional[str]:
        """增强的LLM响应解析器"""
        if not response_text:
            return None
        
        # 清理响应文本
        cleaned_text = response_text.strip()
        
        # 方法1: 查找明确的标记
        markers = [
            "Fixed SQL:",
            "Corrected SQL:",
            "Correct SQL:",
            "SQL:",
            "Fixed:"
        ]
        
        for marker in markers:
            if marker in cleaned_text:
                sql_part = cleaned_text.split(marker, 1)[1].strip()
                sql_candidate = self._extract_sql_from_text(sql_part)
                if sql_candidate:
                    return sql_candidate
        
        # 方法2: 查找SQL关键字
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]
        for keyword in sql_keywords:
            if keyword in cleaned_text.upper():
                # 找到关键字的位置
                start_idx = cleaned_text.upper().find(keyword)
                sql_part = cleaned_text[start_idx:]
                sql_candidate = self._extract_sql_from_text(sql_part)
                if sql_candidate:
                    return sql_candidate
        
        # 方法3: 如果包含常见SQL模式，尝试整体解析
        if any(pattern in cleaned_text.upper() for pattern in ["FROM", "WHERE", "JOIN"]):
            sql_candidate = self._extract_sql_from_text(cleaned_text)
            if sql_candidate:
                return sql_candidate
        
        return None

    def _extract_sql_from_text(self, text: str) -> Optional[str]:
        """从文本中提取SQL语句"""
        if not text:
            return None
        
        # 清理代码块标记
        text = text.strip()
        if text.startswith("```sql"):
            text = text[6:].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        
        # 处理停止序列
        for stop_seq in self.config.llm_stop_sequences:
            if stop_seq in text:
                text = text.split(stop_seq)[0].strip()
        
        # 分行处理，取第一个有效的SQL行
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--') and not line.startswith('Note:'):
                # 移除末尾分号
                sql_candidate = line.rstrip(';').strip()
                if len(sql_candidate) > 10 and any(keyword in sql_candidate.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                    return sql_candidate
        
        return None

    def _validate_sql(self, sql: str, db_path: str) -> ExecutionResult:
        """验证 SQL 查询是否能执行成功"""
        if not sql or not db_path:
            return ExecutionResult(False, "SQL or DB path is empty")
        try:
            result = self.sql_validator.execute_sql(sql, db_path)
            return result
        except Exception as e:
            return ExecutionResult(False, f"Validator error: {e}")

    def _save_log(self, log_data: Dict[str, Any]):
        """保存日志"""
        timestamp = int(time.time())
        log_file_path = os.path.join(
            self.log_dir, 
            f"correction_log_{log_data['db_id']}_{timestamp}.json"
        )
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save log: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取修正模块的统计信息"""
        log_files = []
        if os.path.exists(self.log_dir):
            log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.json')]
        
        stats = {
            "total_corrections_attempted": len(log_files),
            "log_directory": self.log_dir,
            "config_summary": {
                "model": self.config.llm_model_name,
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens
            }
        }
        
        return stats

def self_correction_main(predict_sqls, db_ids, questions, config_path, output_path=None, detailed_output_path=None):
    """
    批量自修正入口，供 text2sql.py 调用。
    Args:
        predict_sqls: List[str]，初步生成的 SQL 列表
        db_ids: List[str]，每条 SQL 对应的数据库 ID
        questions: List[str]，每条 SQL 对应的自然语言问题
        config_path: str，自修正配置文件路径
        output_path: str，最终 SQL 输出路径（可选）
        detailed_output_path: str，详细日志输出路径（可选）
    Returns:
        List[str]，修正后的 SQL 列表
    """
    from .config import SelfCorrectionConfig
    import json
    config = SelfCorrectionConfig(config_path)
    module = SelfCorrectionModule(config)
    corrected_sqls = []
    detailed_logs = []
    # 尝试从 config 读取 schema_dict 和 db_path_dict
    schema_dict = getattr(config, 'schema_dict', {})
    db_path_dict = getattr(config, 'db_path_dict', {})
    default_db_path = getattr(config, 'db_path', None)
    for sql, db_id, question in zip(predict_sqls, db_ids, questions):
        schema = schema_dict.get(db_id, {})
        db_path = db_path_dict.get(db_id, default_db_path)
        corrected_sql, status = module.process(question, schema, sql, db_path)
        corrected_sqls.append(corrected_sql)
        detailed_logs.append({
            "question": question,
            "db_id": db_id,
            "initial_sql": sql,
            "corrected_sql": corrected_sql,
            "status": status
        })
    # 保存详细日志
    if detailed_output_path:
        with open(detailed_output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_logs, f, ensure_ascii=False, indent=2)
    # 可选：保存最终 SQL
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sql in corrected_sqls:
                f.write(sql + '\n')
    return corrected_sqls