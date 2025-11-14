"""
API Client module for interacting with OpenAI-compatible endpoints.
Handles HTTP requests with retry logic and error handling.
"""

import time
import requests
from typing import List, Dict
from .config import ModelConfig


class APIClient:
    """Client for OpenAI-compatible chat completion endpoints."""
    
    def __init__(self, model_config: ModelConfig, timeout: float = 60.0, max_retries: int = 5):
        """
        Initialize API client.
        
        Args:
            model_config: Model configuration with endpoint details
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.config = model_config
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = model_config.get_api_key()
        self.verify_ssl = model_config.verify_ssl
        
        if not self.api_key:
            raise ValueError(
                f"API key not found for {model_config.name} (env: {model_config.api_key_env})"
            )
        
        # Disable SSL warnings if verify_ssl is False
        if not self.verify_ssl:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            print(f"⚠️  SSL verification disabled for {model_config.name}")
    
    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 800,
        temperature: float = 0.85,
        top_p: float = 0.95,
    ) -> str:
        """
        Send a chat completion request to the API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If request fails after all retries
        """
        url = self.config.api_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        attempt = 0
        backoff = 2.0
        
        while True:
            attempt += 1
            try:
                resp = requests.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    # Try to extract content from response
                    try:
                        content = data["choices"][0]["message"].get("content", "")
                        
                        # Handle case where content is empty or None
                        if not content or not content.strip():
                            finish_reason = data["choices"][0].get("finish_reason")
                            completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
                            
                            # If empty due to length limit, raise informative error
                            if finish_reason == "length" and completion_tokens == 0:
                                raise RuntimeError(
                                    f"{self.config.name} returned empty response. "
                                    f"Try increasing max_tokens (current: {max_tokens})"
                                )
                            
                            # Otherwise, log and raise
                            import json
                            print(f"⚠️  Empty content from {self.config.name}: {json.dumps(data, indent=2)[:500]}")
                            raise RuntimeError(f"Empty response from {self.config.name}")
                        
                        return content.strip()
                        
                    except (KeyError, IndexError, TypeError) as e:
                        # Log the response for debugging
                        import json
                        print(f"⚠️  Unexpected response format from {self.config.name}: {json.dumps(data, indent=2)[:500]}")
                        raise RuntimeError(f"Failed to parse response from {self.config.name}: {e}")
                
                # Retry on transient errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt <= self.max_retries:
                        time.sleep(backoff)
                        backoff *= 1.6
                        continue
                
                # Non-retryable error
                raise RuntimeError(
                    f"HTTP {resp.status_code} from {self.config.name}: {resp.text[:400]}"
                )
                
            except requests.exceptions.RequestException as e:
                if attempt <= self.max_retries:
                    time.sleep(backoff)
                    backoff *= 1.6
                    continue
                raise RuntimeError(
                    f"Request failed for {self.config.name} after {attempt} attempts: {str(e)}"
                )
    
    def test_connectivity(self) -> tuple[bool, str]:
        """
        Test connection to the API endpoint.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            messages = [{"role": "user", "content": "Say hello"}]
            # Use more tokens for models like Gemini that need space to generate
            response = self.chat_complete(
                messages=messages, max_tokens=30, temperature=0.1, top_p=1.0
            )
            return True, f"✅ Connected (response: {response[:20]})"
        except Exception as e:
            return False, f"❌ {str(e)[:180]}"


def test_all_models(models: List[ModelConfig]) -> Dict[str, str]:
    """
    Test connectivity to all configured models.
    
    Args:
        models: List of model configurations to test
        
    Returns:
        Dictionary mapping model names to status messages
    """
    results = {}
    
    for model_config in models:
        # Check if API key is available
        if not model_config.get_api_key():
            results[model_config.name] = f"❌ Missing API key (env: {model_config.api_key_env})"
            continue
        
        try:
            client = APIClient(model_config, timeout=15.0)
            success, message = client.test_connectivity()
            results[model_config.name] = message
        except Exception as e:
            results[model_config.name] = f"❌ {str(e)[:180]}"
    
    return results
