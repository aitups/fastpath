import pytest
from unittest.mock import patch, MagicMock
from fastpath.knowledge_base import KnowledgeBase
from fastpath.llm_provider import LLMProvider
import tempfile
import yaml
import os

class DummyLLMProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs):
        return "dummy"

@pytest.fixture
def mock_config_file():
    config_data = {
        "embedding_provider": "huggingface",
        "embedding_model": "test-model",
        "llm_provider": "ollama",
        "llm_config": {
            "ollama": {"model": "test-llm"}
        }
    }
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name
        yaml.dump(config_data, tmp)
    
    yield tmp_path
    
    os.remove(tmp_path)

@patch('fastpath.knowledge_base.get_embedding_from_config')
@patch('fastpath.knowledge_base.get_llm_provider_from_config')
@patch('fastpath.knowledge_base.Settings')
@patch('fastpath.knowledge_base.VectorStoreIndex')
@patch('fastpath.knowledge_base.SimpleDirectoryReader')
def test_kb_init_and_ingest(
    mock_sdr, mock_vsi, mock_settings, 
    mock_get_llm, mock_get_embed, mock_config_file
):
    # Setup Mocks
    mock_embed_model = MagicMock()
    mock_get_embed.return_value = mock_embed_model
    
    mock_llm_provider = DummyLLMProvider(config={})
    mock_get_llm.return_value = mock_llm_provider

    mock_sdr.return_value.load_data.return_value = [MagicMock()]
    
    # Action
    kb = KnowledgeBase(storage_path=":memory:", config_path=mock_config_file)
    kb.ingest("dummy_dir")

    # Assertions
    mock_get_embed.assert_called_once()
    mock_get_llm.assert_called_once_with(mock_config_file)
    assert mock_settings.embed_model is mock_embed_model
    assert mock_settings.llm is not None
    mock_sdr.assert_called_once_with("dummy_dir")
    mock_vsi.from_documents.assert_called_once()

@patch('fastpath.knowledge_base.Settings')
def test_kb_query(mock_settings):
    # Setup Mocks
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_response = MagicMock()
    
    mock_node = MagicMock()
    mock_node.get_text.return_value = "mocked text"
    mock_node.get_score.return_value = 0.99
    mock_response.source_nodes = [mock_node]
    
    mock_query_engine.query.return_value = mock_response
    mock_index.as_query_engine.return_value = mock_query_engine

    # Action
    kb = KnowledgeBase(storage_path=":memory:") # Constructor will run but we mock its effects
    kb.index = mock_index # Manually set the index
    result = kb.query("test query")

    # Assertions
    mock_index.as_query_engine.assert_called_once_with(similarity_top_k=5)
    mock_query_engine.query.assert_called_once_with("test query")
    assert len(result) == 1
    assert result[0]["text"] == "mocked text"
    assert result[0]["score"] == 0.99 