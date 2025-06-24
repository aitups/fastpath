from typing import Any

class LLMProvider:
    """
    Interfaz base para proveedores de LLM (OpenAI, Ollama, llama.cpp, etc).
    """
    def __init__(self, config: dict):
        self.config = config

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError 