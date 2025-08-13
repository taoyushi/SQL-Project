# self_correction/llm_api.py

import requests
import json
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM API errors."""
    pass

class QwenAPIClient:
    """
    封装与QWEN 2.5turbo API的交互逻辑。
    兼容阿里云DashScope API和其他OpenAI-compatible API。
    """
    def __init__(self, api_key: str, endpoint: str, model: str = "qwen3:14b", 
                 timeout: int = 60, max_retries: int = 3):
        """
        初始化Qwen API客户端。

        Args:
            api_key (str): API密钥。
            endpoint (str): API服务地址。
            model (str): 要使用的模型名称。
            timeout (int): 请求超时时间（秒）。
            max_retries (int): 最大重试次数。
        """
        if not api_key:
            raise ValueError("API Key is not provided.")
        if not endpoint:
            raise ValueError("API Endpoint is not provided.")

        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 根据endpoint判断API类型
        if "dashscope.aliyuncs.com" in endpoint.lower():
            self.api_type = "dashscope"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        elif "chat/completions" in endpoint.lower() or "v1/" in endpoint.lower():
            # OpenAI兼容API (包括自部署的API)
            self.api_type = "openai_compatible"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        else:
            # 默认使用OpenAI兼容格式
            self.api_type = "openai_compatible"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
        logger.info(f"QwenAPIClient initialized for {self.api_type} API: {self.endpoint} using model: {self.model}")

    def get_correction(self, prompt: str, temperature: float, max_tokens: int, 
                      stop_sequences: List[str]) -> str:
        """
        向Qwen API发送请求，获取修正后的SQL。

        Args:
            prompt (str): 发送给LLM的Prompt文本。
            temperature (float): 控制生成随机性。
            max_tokens (int): 最大生成Token数量。
            stop_sequences (List[str]): 停止生成序列列表。

        Returns:
            str: LLM返回的原始文本响应。

        Raises:
            LLMError: 如果API调用失败或返回错误。
        """
        for attempt in range(self.max_retries + 1):
            try:
                if self.api_type == "dashscope":
                    return self._call_dashscope_api(prompt, temperature, max_tokens, stop_sequences)
                else:
                    return self._call_openai_compatible_api(prompt, temperature, max_tokens, stop_sequences)
                    
            except requests.exceptions.Timeout:
                logger.warning(f"API request timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt == self.max_retries:
                    raise LLMError(f"API request timed out after {self.max_retries + 1} attempts")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries + 1})")
                    if attempt == self.max_retries:
                        raise LLMError(f"Rate limit exceeded after {self.max_retries + 1} attempts")
                    time.sleep(5 * (attempt + 1))  # Longer wait for rate limits
                else:
                    logger.error(f"API request failed: {e}")
                    if attempt == self.max_retries:
                        raise LLMError(f"API request failed: {e}")
                    time.sleep(2 ** attempt)

    def _call_dashscope_api(self, prompt: str, temperature: float, max_tokens: int, 
                           stop_sequences: List[str]) -> str:
        """调用阿里云DashScope API"""
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that corrects SQL queries based on natural language questions and database schemas."},
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.8,
                "repetition_penalty": 1.0
            }
        }
        
        # DashScope可能不支持stop_sequences，需要在响应处理中手动处理
        
        logger.debug(f"Sending DashScope API request: {json.dumps(payload, indent=2)[:500]}...")
        response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
        response.raise_for_status()

        response_data = response.json()
        logger.debug(f"Received DashScope API response: {json.dumps(response_data, indent=2)[:500]}...")

        # 修复：DashScope API成功时直接返回结果，没有status_code字段
        if "output" in response_data:
            output = response_data["output"]
            
            # 处理不同的响应格式
            if "text" in output:
                # 标准DashScope格式
                generated_content = output["text"]
            elif "choices" in output and len(output["choices"]) > 0:
                # 某些DashScope变体格式
                generated_content = output["choices"][0].get("message", {}).get("content", "")
            else:
                # 兜底处理
                generated_content = str(output)
                
            # 手动处理stop_sequences
            for stop_seq in stop_sequences:
                if stop_seq in generated_content:
                    generated_content = generated_content.split(stop_seq)[0]
                    
            request_id = response_data.get("request_id", "N/A")
            logger.info(f"DashScope API call successful. Request ID: {request_id}")
            return generated_content.strip()
            
        elif "error" in response_data:
            # 处理错误响应
            error_info = response_data["error"]
            error_msg = error_info.get("message", str(error_info))
            raise LLMError(f"DashScope API error: {error_msg}")
        else:
            # 未知响应格式
            raise LLMError(f"Unexpected DashScope response structure: {response_data}")

    def _call_openai_compatible_api(self, prompt: str, temperature: float, max_tokens: int, 
                                  stop_sequences: List[str]) -> str:
        """调用OpenAI兼容的API（包括自部署API）"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that corrects SQL queries based on natural language questions and database schemas. Always provide clear, executable SQL as your response."},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # 某些自部署API可能不支持stop参数，所以单独处理
        if stop_sequences and len(stop_sequences) > 0:
            # 只使用最重要的停止序列，避免某些API不支持多个停止序列
            payload["stop"] = stop_sequences[:2]  # 限制为前2个

        logger.debug(f"Sending OpenAI-compatible API request: {json.dumps(payload, indent=2)[:500]}...")
        
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 422:
                # 可能是stop参数不支持，重试不带stop参数
                logger.warning("API rejected stop parameter, retrying without it...")
                payload.pop("stop", None)
                response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
            else:
                raise

        response_data = response.json()
        logger.debug(f"Received OpenAI-compatible API response: {json.dumps(response_data, indent=2)[:500]}...")

        if "choices" in response_data and len(response_data["choices"]) > 0:
            choice = response_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                generated_content = choice["message"]["content"]
                
                # 如果API不支持stop参数，手动处理停止序列
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_content:
                            generated_content = generated_content.split(stop_seq)[0]
                
                request_id = response_data.get("id", "N/A")
                logger.info(f"OpenAI-compatible API call successful. Request ID: {request_id}")
                return generated_content.strip()
            else:
                raise LLMError(f"Unexpected OpenAI-compatible response structure: {response_data}")
        else:
            error_msg = response_data.get("error", {}).get("message", "Unknown error")
            raise LLMError(f"OpenAI-compatible API error: {error_msg}")

    def test_connection(self) -> bool:
        """测试API连接是否正常"""
        try:
            # 使用更简单和宽松的测试
            test_response = self.get_correction(
                prompt="Hello, please respond with a short greeting.",
                temperature=0.1,
                max_tokens=50,
                stop_sequences=[]
            )
            
            # 只要能获得响应且不是空字符串就认为连接成功
            if test_response and len(test_response.strip()) > 0:
                logger.info(f"API connection test successful. Response: {test_response[:100]}...")
                return True
            else:
                logger.warning("API connection test returned empty response")
                return False
                
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False



























            # self_correction/llm_api.py

import requests
import json
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM API errors."""
    pass

class QwenAPIClient:
    """
    封装与QWEN 2.5turbo API的交互逻辑。
    兼容阿里云DashScope API和其他OpenAI-compatible API。
    """
    def __init__(self, api_key: str, endpoint: str, model: str = "qwen3:14b", 
                 timeout: int = 60, max_retries: int = 3):
        """
        初始化Qwen API客户端。

        Args:
            api_key (str): API密钥。
            endpoint (str): API服务地址。
            model (str): 要使用的模型名称。
            timeout (int): 请求超时时间（秒）。
            max_retries (int): 最大重试次数。
        """
        if not api_key:
            raise ValueError("API Key is not provided.")
        if not endpoint:
            raise ValueError("API Endpoint is not provided.")

        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 根据endpoint判断API类型
        if "dashscope.aliyuncs.com" in endpoint.lower():
            self.api_type = "dashscope"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        elif "chat/completions" in endpoint.lower() or "v1/" in endpoint.lower():
            # OpenAI兼容API (包括自部署的API)
            self.api_type = "openai_compatible"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        else:
            # 默认使用OpenAI兼容格式
            self.api_type = "openai_compatible"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
        logger.info(f"QwenAPIClient initialized for {self.api_type} API: {self.endpoint} using model: {self.model}")

    def get_correction(self, prompt: str, temperature: float, max_tokens: int, 
                      stop_sequences: List[str]) -> str:
        """
        向Qwen API发送请求，获取修正后的SQL。

        Args:
            prompt (str): 发送给LLM的Prompt文本。
            temperature (float): 控制生成随机性。
            max_tokens (int): 最大生成Token数量。
            stop_sequences (List[str]): 停止生成序列列表。

        Returns:
            str: LLM返回的原始文本响应。

        Raises:
            LLMError: 如果API调用失败或返回错误。
        """
        for attempt in range(self.max_retries + 1):
            try:
                if self.api_type == "dashscope":
                    return self._call_dashscope_api(prompt, temperature, max_tokens, stop_sequences)
                else:
                    return self._call_openai_compatible_api(prompt, temperature, max_tokens, stop_sequences)
                    
            except requests.exceptions.Timeout:
                logger.warning(f"API request timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt == self.max_retries:
                    raise LLMError(f"API request timed out after {self.max_retries + 1} attempts")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries + 1})")
                    if attempt == self.max_retries:
                        raise LLMError(f"Rate limit exceeded after {self.max_retries + 1} attempts")
                    time.sleep(5 * (attempt + 1))  # Longer wait for rate limits
                else:
                    logger.error(f"API request failed: {e}")
                    if attempt == self.max_retries:
                        raise LLMError(f"API request failed: {e}")
                    time.sleep(2 ** attempt)

    def _call_dashscope_api(self, prompt: str, temperature: float, max_tokens: int, 
                           stop_sequences: List[str]) -> str:
        """调用阿里云DashScope API"""
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that corrects SQL queries based on natural language questions and database schemas."},
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.8,
                "repetition_penalty": 1.0
            }
        }
        
        # DashScope可能不支持stop_sequences，需要在响应处理中手动处理
        
        logger.debug(f"Sending DashScope API request: {json.dumps(payload, indent=2)[:500]}...")
        response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
        response.raise_for_status()

        response_data = response.json()
        logger.debug(f"Received DashScope API response: {json.dumps(response_data, indent=2)[:500]}...")

        if response_data.get("status_code") == 200:
            output = response_data.get("output", {})
            if "text" in output:
                generated_content = output["text"]
            elif "choices" in output and len(output["choices"]) > 0:
                generated_content = output["choices"][0].get("message", {}).get("content", "")
            else:
                raise LLMError(f"Unexpected DashScope response structure: {response_data}")
                
            # 手动处理stop_sequences
            for stop_seq in stop_sequences:
                if stop_seq in generated_content:
                    generated_content = generated_content.split(stop_seq)[0]
                    
            request_id = response_data.get("request_id", "N/A")
            logger.info(f"DashScope API call successful. Request ID: {request_id}")
            return generated_content.strip()
        else:
            error_msg = response_data.get("message", "Unknown error")
            raise LLMError(f"DashScope API error: {error_msg}")

    def _call_openai_compatible_api(self, prompt: str, temperature: float, max_tokens: int, 
                                  stop_sequences: List[str]) -> str:
        """调用OpenAI兼容的API（包括自部署API）"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that corrects SQL queries based on natural language questions and database schemas. Always provide clear, executable SQL as your response."},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # 某些自部署API可能不支持stop参数，所以单独处理
        if stop_sequences and len(stop_sequences) > 0:
            # 只使用最重要的停止序列，避免某些API不支持多个停止序列
            payload["stop"] = stop_sequences[:2]  # 限制为前2个

        logger.debug(f"Sending OpenAI-compatible API request: {json.dumps(payload, indent=2)[:500]}...")
        
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 422:
                # 可能是stop参数不支持，重试不带stop参数
                logger.warning("API rejected stop parameter, retrying without it...")
                payload.pop("stop", None)
                response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
            else:
                raise

        response_data = response.json()
        logger.debug(f"Received OpenAI-compatible API response: {json.dumps(response_data, indent=2)[:500]}...")

        if "choices" in response_data and len(response_data["choices"]) > 0:
            choice = response_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                generated_content = choice["message"]["content"]
                
                # 如果API不支持stop参数，手动处理停止序列
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_content:
                            generated_content = generated_content.split(stop_seq)[0]
                
                request_id = response_data.get("id", "N/A")
                logger.info(f"OpenAI-compatible API call successful. Request ID: {request_id}")
                return generated_content.strip()
            else:
                raise LLMError(f"Unexpected OpenAI-compatible response structure: {response_data}")
        else:
            error_msg = response_data.get("error", {}).get("message", "Unknown error")
            raise LLMError(f"OpenAI-compatible API error: {error_msg}")

    def test_connection(self) -> bool:
        """测试API连接是否正常"""
        try:
            test_response = self.get_correction(
                prompt="Please respond with exactly: 'Connection test successful'",
                temperature=0.1,
                max_tokens=50,
                stop_sequences=[]
            )
            success_indicators = ["connection", "successful", "test"]
            return any(indicator in test_response.lower() for indicator in success_indicators)
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False