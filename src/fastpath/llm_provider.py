from typing import Any, ClassVar
import requests
import openai
import os
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from .config import Config

def remove_think_tags(text: str, keep_prefix: bool = True) -> str:
    """
    Remove <think> tags from the text.
    """
    if "<think>" in text:
        think_start = text.index("<think>") + len("<think>")
        think_end = text.index("</think>")
        if keep_prefix:
            # Keep the text before <think> but remove the think tags
            return text[:think_start] + text[think_end + len("</think>"):]
        # Remove the text before <think> and the think tags
        return text[think_end + len("</think>"):]
    return text

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
        return remove_think_tags(response.json().get("response", ""), keep_prefix=False)

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
        api_key = self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.get('model', 'gemini-pro')}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

class HuggingFaceProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        api_key = self.config.get("api_key") or os.getenv("HUGGINGFACE_API_TOKEN")
        model = self.config.get("model", "HuggingFaceH4/zephyr-7b-beta")
        url = f"https://router.huggingface.co/novita/v3/openai/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"messages": [
            {
                "role": "user",
                "content": prompt
            }
            ],
            "model": "deepseek/deepseek-r1-0528-qwen3-8b"
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        # imprimir la estructura de la respuesta para depuración
        print(f"Respuesta de HuggingFace: {result}")
        generated_text = ""
        # Soporta tanto respuesta string como lista de dicts
        # puede que el contenido venga en el campo generated_text, choices, message o content
        # también puede seguir la estructura message.content
        if isinstance(result, list) and len(result) > 0 and 'text' in result[0]:
            print("Respuesta es una lista de diccionarios con 'text'")
            generated_text = str(result[0]['text'])
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            print("Respuesta es una lista de diccionarios con 'generated_text'")
            generated_text = str(result[0]['generated_text'])
        elif isinstance(result, dict) and 'generated_text' in result:
            # Asume que la respuesta es del tipo HuggingFace con 'generated_text'
            generated_text = str(result['generated_text'])
        elif isinstance(result, dict) and 'choices' in result and len(result['choices']) > 0:
            print("Respuesta es un diccionario con 'choices'")
            # Asume que la respuesta es del tipo OpenAI con 'choices' y 'message'
            if 'message' in result['choices'][0]:
                print("Respuesta contiene 'message' en 'choices'")
                generated_text = str(result['choices'][0]['message']['content'])
                print(f"Mensaje generado: {generated_text}")
            elif 'text' in result['choices'][0]:
                print("Respuesta contiene 'text' en 'choices'")
                generated_text = str(result['choices'][0]['text'])
            # Si no hay 'message', puede ser un formato diferente
            generated_text = str(result['choices'][0])
        elif isinstance(result, dict) and 'message' in result:
            if 'content' in result['message']:
                print("Respuesta contiene 'content' en 'message'")
                generated_text = str(result['message']['content'])
            else:
                # Si no hay 'content', puede ser un formato diferente
                # Asumimos que es un mensaje completo
                if isinstance(result['message'], str):
                    print("Respuesta es un string en 'message'")
                    generated_text = result['message']
                else:
                    # Si es un dict, convertimos a string
                    # Esto es útil si el mensaje es un dict con más información
                    # como role, content, etc.
                    print("Respuesta es un dict en 'message'")
                    generated_text = str(result['message'])
        elif isinstance(result, dict) and 'choices' in result:
            print("Respuesta es un diccionario con 'choices'")
            generated_text = str(result['choices'][0]['message'])
        elif isinstance(result, str):
            print("Respuesta es un string")
            generated_text = result
        else:
            print("Respuesta no coincide con ningún formato esperado")
            generated_text = str(result)
        # Si la respuesta incluye un apartado <think> hay que eliminarlo
        if "<think>" in generated_text:
            think_start = generated_text.index("<think>") + len("<think>")
            think_end = generated_text.index("</think>")
            # Logamos la cadena de pensamiento
            print(f"Cadena de pensamiento: {generated_text[think_start:think_end]}")
            # Eliminamos la cadena de pensamiento del texto generado
            generated_text = generated_text[:think_start] + generated_text[think_end + len("</think>"):]
            print(f"Texto generado sin cadena de pensamiento: {generated_text}")
        return generated_text

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
    provider_name = config_data.get("llm_provider", "huggingface").lower()
    llm_config = config_data.get("llm_config", {})
    provider_config = llm_config.get(provider_name, {})

    if provider_name == "openai":
        return OpenAIProvider(provider_config)
    elif provider_name == "ollama":
        return OllamaProvider(provider_config)
    elif provider_name == "gemini":
        return GeminiProvider(provider_config)
    elif provider_name == "huggingface":
        return HuggingFaceProvider(provider_config)
    else:
        raise ValueError(f"Unknown or unsupported LLM provider: {provider_name}")