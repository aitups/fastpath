from fastpath.llm_provider import LLMProvider
import pytest
from unittest.mock import MagicMock, patch
from fastpath.llm_provider import (
    OllamaProvider,
    OpenAIProvider,
    GeminiProvider,
    LlamaIndexLLMAdapter,
    get_llm_provider_from_config,
)
import tempfile
import yaml
import pydantic

def test_llmprovider_generate_not_implemented():
    provider = LLMProvider(config={})
    with pytest.raises(NotImplementedError):
        provider.generate("prompt de prueba")

@pytest.fixture
def temp_config_file():
    def _create_config(data):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
            yaml.dump(data, tmp)
            return tmp.name
    return _create_config

def test_get_llm_provider_from_config_ollama(temp_config_file):
    config_data = {"llm_provider": "ollama", "llm_config": {"ollama": {"model": "test_model"}}}
    config_path = temp_config_file(config_data)
    provider = get_llm_provider_from_config(config_path)
    assert isinstance(provider, OllamaProvider)
    assert provider.config["model"] == "test_model"

def test_get_llm_provider_from_config_openai(temp_config_file):
    config_data = {"llm_provider": "openai", "llm_config": {"openai": {"api_key": "test_key"}}}
    config_path = temp_config_file(config_data)
    provider = get_llm_provider_from_config(config_path)
    assert isinstance(provider, OpenAIProvider)
    assert provider.config["api_key"] == "test_key"

def test_get_llm_provider_from_config_gemini(temp_config_file):
    config_data = {"llm_provider": "gemini", "llm_config": {"gemini": {"api_key": "test_key"}}}
    config_path = temp_config_file(config_data)
    provider = get_llm_provider_from_config(config_path)
    assert isinstance(provider, GeminiProvider)
    assert provider.config["api_key"] == "test_key"

class MockLLM(pydantic.BaseModel):
    def generate(self, *args, **kwargs):
        return "mock"

def test_llama_index_adapter_fields():
    """Tests if the adapter can be instantiated correctly."""
    mock_provider = MagicMock(spec=LLMProvider)
    adapter = LlamaIndexLLMAdapter(llm_provider=mock_provider)
    assert adapter.llm_provider is mock_provider

def test_llama_index_adapter_complete():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate.return_value = "Mocked LLM response"
    
    adapter = LlamaIndexLLMAdapter(llm_provider=mock_provider)
    response = adapter.complete("test prompt")
    
    mock_provider.generate.assert_called_once_with("test prompt")
    assert response.text == "Mocked LLM response"

def test_llama_index_adapter_stream_complete():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate.return_value = "Mocked stream response"

    adapter = LlamaIndexLLMAdapter(llm_provider=mock_provider)
    
    response_gen = adapter.stream_complete("test stream prompt")
    response_list = list(response_gen)

    mock_provider.generate.assert_called_once_with("test stream prompt")
    assert len(response_list) == 1
    assert response_list[0].text == "Mocked stream response"
    assert response_list[0].delta == "Mocked stream response" 