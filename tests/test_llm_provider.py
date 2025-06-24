from fastpath.llm_provider import LLMProvider
import pytest

def test_llmprovider_generate_not_implemented():
    provider = LLMProvider(config={})
    with pytest.raises(NotImplementedError):
        provider.generate("prompt de prueba") 