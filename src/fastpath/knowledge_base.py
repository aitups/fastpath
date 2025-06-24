from typing import Any, List
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    from llama_index.embeddings.openvino import OpenVINOEmbedding
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

from .config import Config
from .llm_provider import get_llm_provider_from_config, LlamaIndexLLMAdapter


def get_embedding_from_config(config: dict):
    provider = config.get("embedding_provider", "huggingface").lower()
    model = config.get("embedding_model", "all-MiniLM-L6-v2")
    if provider == "openai":
        # Assumes OPENAI_API_KEY is in the environment or passed in config
        return OpenAIEmbedding(model=model, api_key=config.get("llm_config", {}).get("openai", {}).get("api_key"))
    elif provider == "huggingface":
        return HuggingFaceEmbedding(model_name=model)
    elif provider == "openvino" and HAS_OPENVINO:
        return OpenVINOEmbedding(model_name=model)
    else:
        raise ValueError(f"Unknown or unsupported embedding provider: {provider}")


class KnowledgeBase:
    """
    Shared knowledge base (RAG) using LlamaIndex.
    It automatically configures the LLM and embedding model from config.yaml.
    """
    def __init__(self, storage_path: str, config_path: str = "config.yaml"):
        self.storage_path = storage_path
        self.index = None
        
        config_data = Config(config_path).data
        
        # 1. Configure embedding model from config
        embed_model = get_embedding_from_config(config_data)
        Settings.embed_model = embed_model
        
        # 2. Configure LLM from config and adapt it for LlamaIndex
        llm_provider = get_llm_provider_from_config(config_path)
        llm = LlamaIndexLLMAdapter(llm_provider=llm_provider)
        Settings.llm = llm

    def ingest(self, data_dir: str):
        """Ingests documents from a directory."""
        documents = SimpleDirectoryReader(data_dir).load_data()
        self.index = VectorStoreIndex.from_documents(documents)

    def query(self, query: str, top_k: int = 5) -> List[dict]:
        if not self.index:
            raise ValueError("Knowledge base is empty. Ingest data first.")
        
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(query)
        return [
            {"text": node.get_text(), "score": node.get_score()}
            for node in response.source_nodes
        ] 