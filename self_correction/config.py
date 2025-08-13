# self_correction/config.py

import os
import yaml
from typing import Dict, Any, List, Optional
from .prompt_templates import PromptStrategy, DEFAULT_PROMPT_TEMPLATES, DEFAULT_FEW_SHOT_EXAMPLES

class SelfCorrectionConfig:
    """
    Configuration class for the Self-Correction Module.
    Loads settings from environment variables or a configuration file.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path (Optional[str]): Path to the YAML configuration file.
                                         If None, loads primarily from environment variables.
        """
        # Default values
        self.llm_api_key: str = ""
        self.llm_api_endpoint: str = ""
        self.llm_model_name: str = "qwen3:14b"
        self.llm_temperature: float = 0.1
        self.llm_max_tokens: int = 800
        self.llm_stop_sequences: List[str] = ["#;\n\n", "\n\n---", "---\n"]
        self.llm_timeout_seconds: int = 120
        self.llm_max_retries: int = 5
        self.prompt_templates: Dict[str, str] = {}
        self.few_shot_examples: Optional[List[Dict[str, Any]]] = None
        self.log_dir: str = "./correction_logs_qwen3"

        # 1. Load from config file first if provided
        if config_path and os.path.exists(config_path):
            self._load_from_yaml(config_path)
        else:
            if config_path:
                print(f"[WARN] Config file not found at {config_path}. Loading from environment variables and defaults.")

        # 2. Override with environment variables
        self._load_from_env()

        # 3. Set default prompt templates if not loaded from file
        if not self.prompt_templates:
            self._set_default_prompt_templates()

        # === 自动加载 schema_dict 和 db_path_dict ===
        import json
        tables_path = "data/preprocessed_data/test_tables_for_natsql.json"
        db_root = "database"
        schema_dict = {}
        db_path_dict = {}
        if os.path.exists(tables_path):
            with open(tables_path, 'r', encoding='utf-8') as f:
                tables = json.load(f)
                for t in tables:
                    db_id = t['db_id']
                    schema_dict[db_id] = t
                    db_path_dict[db_id] = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        self.schema_dict = schema_dict
        self.db_path_dict = db_path_dict
        self.db_path = db_root  # 兜底用

        # Basic validation
        if not self.llm_api_key:
            raise ValueError("LLM API Key is not configured. Set QWEN_API_KEY environment variable or configure in YAML.")
        if not self.llm_api_endpoint:
            raise ValueError("LLM API Endpoint is not configured. Set QWEN_API_ENDPOINT environment variable or configure in YAML.")
        if not self.prompt_templates:
            raise ValueError("Prompt templates could not be loaded or defaulted.")

        # Ensure log directory is an absolute path
        self.log_dir = os.path.abspath(self.log_dir)

    def _load_from_yaml(self, config_path: str):
        """Load configuration from a YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                return

            # LLM Settings
            llm_settings = config_data.get('llm_settings', {})
            self.llm_api_key = llm_settings.get('api_key', self.llm_api_key)
            self.llm_api_endpoint = llm_settings.get('api_endpoint', self.llm_api_endpoint)
            self.llm_model_name = llm_settings.get('model_name', self.llm_model_name)
            self.llm_temperature = llm_settings.get('temperature', self.llm_temperature)
            self.llm_max_tokens = llm_settings.get('max_tokens', self.llm_max_tokens)
            self.llm_stop_sequences = llm_settings.get('stop_sequences', self.llm_stop_sequences)
            self.llm_timeout_seconds = llm_settings.get('timeout_seconds', self.llm_timeout_seconds)
            self.llm_max_retries = llm_settings.get('max_retries', self.llm_max_retries)

            # Prompt Settings
            prompt_settings = config_data.get('prompt_settings', {})
            self.prompt_templates = prompt_settings.get('templates', self.prompt_templates)
            self.few_shot_examples = prompt_settings.get('few_shot_examples', self.few_shot_examples)

            # Logging Settings
            log_settings = config_data.get('logging_settings', {})
            self.log_dir = log_settings.get('log_dir', self.log_dir)

            print(f"[INFO] Configuration loaded from {config_path}")

        except yaml.YAMLError as e:
            print(f"[ERROR] Failed to parse YAML config file {config_path}: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred loading config from {config_path}: {e}")
            raise

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # API credentials
        self.llm_api_key = os.environ.get('QWEN_API_KEY', self.llm_api_key)
        self.llm_api_endpoint = os.environ.get('QWEN_API_ENDPOINT', self.llm_api_endpoint)
        self.llm_model_name = os.environ.get('QWEN_MODEL_NAME', self.llm_model_name)
        
        # LLM parameters
        temp_str = os.environ.get('QWEN_TEMPERATURE')
        if temp_str:
            try:
                self.llm_temperature = float(temp_str)
            except ValueError:
                print(f"[WARN] Invalid value for QWEN_TEMPERATURE: {temp_str}. Using default.")

        max_tokens_str = os.environ.get('QWEN_MAX_TOKENS')
        if max_tokens_str:
            try:
                self.llm_max_tokens = int(max_tokens_str)
            except ValueError:
                print(f"[WARN] Invalid value for QWEN_MAX_TOKENS: {max_tokens_str}. Using default.")

        timeout_str = os.environ.get('QWEN_TIMEOUT_SECONDS')
        if timeout_str:
            try:
                self.llm_timeout_seconds = int(timeout_str)
            except ValueError:
                print(f"[WARN] Invalid value for QWEN_TIMEOUT_SECONDS: {timeout_str}. Using default.")

        max_retries_str = os.environ.get('QWEN_MAX_RETRIES')
        if max_retries_str:
            try:
                self.llm_max_retries = int(max_retries_str)
            except ValueError:
                print(f"[WARN] Invalid value for QWEN_MAX_RETRIES: {max_retries_str}. Using default.")

        # Other settings
        log_dir_env = os.environ.get('CORRECTION_LOG_DIR')
        if log_dir_env:
            self.log_dir = log_dir_env

        print("[INFO] Configuration loaded from environment variables.")

    def _set_default_prompt_templates(self):
        """Set default fallback prompt templates if none are loaded."""
        print("[INFO] Setting default prompt templates.")
        self.prompt_templates = DEFAULT_PROMPT_TEMPLATES.copy()
        
        # Set default few-shot examples if not loaded from config
        if self.few_shot_examples is None:
            self.few_shot_examples = DEFAULT_FEW_SHOT_EXAMPLES.copy()

    def validate_config(self) -> bool:
        """验证配置是否完整和有效"""
        errors = []
        
        if not self.llm_api_key:
            errors.append("LLM API Key is missing")
        if not self.llm_api_endpoint:
            errors.append("LLM API Endpoint is missing")
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            errors.append("LLM temperature should be between 0 and 2")
        if self.llm_max_tokens <= 0:
            errors.append("LLM max_tokens should be positive")
        if not self.prompt_templates:
            errors.append("Prompt templates are missing")
        elif not all(key in self.prompt_templates for key in [PromptStrategy.GENERIC.value, PromptStrategy.GUIDED.value]):
            errors.append("Required prompt templates (generic, guided) are missing")
            
        if errors:
            print("[ERROR] Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True

    def __repr__(self):
        return f"SelfCorrectionConfig(log_dir='{self.log_dir}', llm_model='{self.llm_model_name}', api_endpoint='{self.llm_api_endpoint[:50]}...')"