from typing import Any, ClassVar
import requests
import openai
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from .config import Config

class LLMProvider:
    """
    Base interface for LLM providers.
    """
    def __init__(self, config: dict):
        self.config = config

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

class OllamaProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        url = self.config.get("base_url", "http://localhost:11434") + "/api/generate"
        model = self.config.get("model", "llama3:8b")
        data = {"model": model, "prompt": prompt, "stream": False}
        data.update(kwargs)
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json().get("response", "")

class OpenAIProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        client = openai.OpenAI(api_key=self.config.get("api_key"))
        response = client.chat.completions.create(
            model=self.config.get("model", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

class GeminiProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        # Placeholder: Gemini API integration
        api_key = self.config.get("api_key")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.get('model', 'gemini-pro')}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, headers=headers, params=params, json=data)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

class LlamaIndexLLMAdapter(CustomLLM):
    """
    An adapter to make FastPath's LLMProvider compatible with LlamaIndex's LLM interface.
    This version is compatible with Pydantic v2 used by LlamaIndex.
    """
    llm_provider: LLMProvider
    _model_name: ClassVar[str] = "fastpath_adapted_llm"

    def _get_model_name(self) -> str:
        return self._model_name

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=8192,
            num_output=2048,
            model_name=self._get_model_name(),
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response_text = self.llm_provider.generate(prompt, **kwargs)
        return CompletionResponse(text=response_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # This is a non-streaming stand-in for the streaming interface
        response_text = self.llm_provider.generate(prompt, **kwargs)
        
        def gen() -> CompletionResponseGen:
            yield CompletionResponse(text=response_text, delta=response_text)

        return gen()

def get_llm_provider_from_config(config_path: str = "config.yaml"):
    """
    Factory function to get the configured LLM provider instance.
    """
    config_data = Config(config_path).data
    provider_name = config_data.get("llm_provider", "ollama").lower()
    llm_config = config_data.get("llm_config", {})
    provider_config = llm_config.get(provider_name, {})

    if provider_name == "openai":
        return OpenAIProvider(provider_config)
    elif provider_name == "ollama":
        return OllamaProvider(provider_config)
    elif provider_name == "gemini":
        return GeminiProvider(provider_config)
    else:
        raise ValueError(f"Unknown or unsupported LLM provider: {provider_name}") 