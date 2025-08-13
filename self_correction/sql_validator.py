# self_correction/sql_validator.py

import sqlite3
import logging
import time
import os
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class ExecutionResult:
    """Holds the result of SQL execution validation."""
    def __init__(self, is_executable: bool, error_message: Optional[str] = None, 
                 result_count: Optional[int] = None, execution_time: Optional[float] = None):
        self.is_executable = is_executable
        self.error_message = error_message
        self.result_count = result_count  # 结果行数，用于进一步分析
        self.execution_time = execution_time  # 执行时间

    def __repr__(self):
        return f"ExecutionResult(executable={self.is_executable}, error='{self.error_message}', count={self.result_count})"

class SQLValidator:
    """
    用于在指定的SQLite数据库文件上验证SQL查询是否可执行。
    """
    def __init__(self, timeout_seconds: int = 30):
        """
        初始化SQL Validator。
        
        Args:
            timeout_seconds (int): SQL执行超时时间（秒）
        """
        self.timeout_seconds = timeout_seconds
        logger.info(f"SQLValidator initialized with timeout: {timeout_seconds}s")

    def execute_sql(self, sql: str, db_path: str, fetch_results: bool = True) -> ExecutionResult:
        """
        尝试连接到指定的SQLite数据库并执行SQL查询。

        Args:
            sql (str): 要执行的SQL查询字符串。
            db_path (str): 目标数据库文件的路径。
            fetch_results (bool): 是否尝试获取查询结果。

        Returns:
            ExecutionResult: 包含执行成功标志和错误信息的对象。
        """
        if not sql or not sql.strip():
            return ExecutionResult(False, "SQL query is empty")
        
        if not db_path or not os.path.exists(db_path):
            return ExecutionResult(False, f"Database file not found: {db_path}")

        conn = None
        start_time = time.time()
        
        try:
            # Connect to the database
            conn = sqlite3.connect(db_path, timeout=self.timeout_seconds)
            conn.execute("PRAGMA foreign_keys = ON;")  # Enable foreign key constraints
            cursor = conn.cursor()

            # Clean and validate SQL
            cleaned_sql = self._clean_sql(sql)
            if not cleaned_sql:
                return ExecutionResult(False, "SQL query is empty after cleaning")

            # Execute the query with timeout handling
            cursor.execute(cleaned_sql)
            
            execution_time = time.time() - start_time
            result_count = None
            
            # For SELECT queries, try to count results
            if fetch_results and cleaned_sql.strip().upper().startswith('SELECT'):
                try:
                    results = cursor.fetchall()
                    result_count = len(results) if results else 0
                    logger.debug(f"SQL executed successfully: {result_count} rows returned in {execution_time:.3f}s")
                except sqlite3.Error:
                    # Some queries might not return fetchable results
                    logger.debug("Query executed but results not fetchable")
            else:
                # For non-SELECT queries, just check if execution succeeded
                logger.debug(f"SQL executed successfully in {execution_time:.3f}s")

            return ExecutionResult(True, None, result_count, execution_time)

        except sqlite3.OperationalError as e:
            error_msg = str(e).lower()
            execution_time = time.time() - start_time
            
            if "timeout" in error_msg or "database is locked" in error_msg:
                logger.debug(f"SQL execution timeout/lock for '{sql[:100]}...' on '{db_path}': {e}")
                return ExecutionResult(False, f"Execution timeout or database locked: {e}", None, execution_time)
            elif "syntax error" in error_msg or "near" in error_msg:
                logger.debug(f"SQL syntax error for '{sql[:100]}...' on '{db_path}': {e}")
                return ExecutionResult(False, f"Syntax error: {e}", None, execution_time)
            elif "no such table" in error_msg or "no such column" in error_msg:
                logger.debug(f"SQL schema error for '{sql[:100]}...' on '{db_path}': {e}")
                return ExecutionResult(False, f"Schema error: {e}", None, execution_time)
            else:
                logger.debug(f"SQL operational error for '{sql[:100]}...' on '{db_path}': {e}")
                return ExecutionResult(False, f"Operational error: {e}", None, execution_time)
                
        except sqlite3.IntegrityError as e:
            execution_time = time.time() - start_time
            logger.debug(f"SQL integrity error for '{sql[:100]}...' on '{db_path}': {e}")
            return ExecutionResult(False, f"Integrity constraint violation: {e}", None, execution_time)
            
        except sqlite3.Error as e:
            execution_time = time.time() - start_time
            logger.debug(f"SQL execution failed for '{sql[:100]}...' on '{db_path}': {e}")
            return ExecutionResult(False, f"SQLite error: {e}", None, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Unexpected error during SQL execution for '{sql[:100]}...' on '{db_path}': {e}", exc_info=True)
            return ExecutionResult(False, f"Unexpected error: {e}", None, execution_time)
            
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def _clean_sql(self, sql: str) -> str:
        """
        清理SQL字符串，移除不必要的字符和格式。
        """
        if not sql:
            return ""
            
        # Remove leading/trailing whitespace
        cleaned = sql.strip()
        
        # Remove trailing semicolon if present
        if cleaned.endswith(';'):
            cleaned = cleaned[:-1].strip()
        
        # Remove common markdown formatting
        if cleaned.startswith('```sql'):
            cleaned = cleaned[6:].strip()
        if cleaned.startswith('```'):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
            
        # Remove extra whitespace and normalize
        cleaned = ' '.join(cleaned.split())
        
        return cleaned

    def validate_multiple(self, sqls: list, db_path: str) -> list:
        """
        批量验证多个SQL查询。
        
        Args:
            sqls (list): SQL查询列表
            db_path (str): 数据库路径
            
        Returns:
            list: ExecutionResult对象列表
        """
        results = []
        for i, sql in enumerate(sqls):
            logger.debug(f"Validating SQL {i+1}/{len(sqls)}")
            result = self.execute_sql(sql, db_path)
            results.append(result)
            
        return results

    def get_database_info(self, db_path: str) -> dict:
        """
        获取数据库的基本信息。
        
        Args:
            db_path (str): 数据库路径
            
        Returns:
            dict: 数据库信息
        """
        if not os.path.exists(db_path):
            return {"error": f"Database file not found: {db_path}"}
            
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get database size
            db_size = os.path.getsize(db_path)
            
            info = {
                "path": db_path,
                "size_bytes": db_size,
                "tables": tables,
                "table_count": len(tables)
            }
            
            return info
            
        except Exception as e:
            return {"error": f"Failed to get database info: {e}"}
        finally:
            if conn:
                conn.close()