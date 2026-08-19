from typing import Any, List
try:
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        Settings,
        StorageContext,
    )
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False
try:
    from llama_index.embeddings.openvino import OpenVINOEmbedding
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

from fastpath.config import Config
from fastpath.llm_provider import get_llm_provider_from_config, LlamaIndexLLMAdapter

# Vector store imports (optional)
try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
try:
    from llama_index.vector_stores.milvus import MilvusVectorStore
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False
try:
    from llama_index.vector_stores.elasticsearch import ElasticsearchStore
    HAS_ELASTIC = True
except ImportError:
    HAS_ELASTIC = False
try:
    from llama_index.vector_stores.postgres import PGVectorStore
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False


def get_vector_store_from_config(config_path: str = "config.yaml"):
    config_data = Config(config_path).data
    provider = config_data.get("vector_store_provider", "in_memory").lower()
    store_config = config_data.get("vector_store_config", {}).get(provider, {})

    if provider == "chroma":
        if not HAS_CHROMA:
            raise ImportError("llama-index-vector-stores-chroma is not installed")
        # Soporta tanto persist_dir local como servidor HTTP remoto (host/port)
        persist_dir = store_config.get("persist_directory") or store_config.get("persist_dir")
        host = store_config.get("host")
        port = store_config.get("port")
        if persist_dir and not host:
            return ChromaVectorStore(
                persist_dir=persist_dir,
                collection_name=store_config.get("collection_name", "fastpath"),
            )
        return ChromaVectorStore(
            host=host or "localhost",
            port=int(port or 8000),
            collection_name=store_config.get("collection_name", "fastpath"),
        )
    elif provider == "milvus":
        if not HAS_MILVUS:
            raise ImportError("llama-index-vector-stores-milvus is not installed")
        return MilvusVectorStore(
            host=store_config.get("host", "localhost"),
            port=store_config.get("port", 19530),
            collection_name=store_config.get("collection_name", "fastpath")
        )
    elif provider == "elasticsearch":
        if not HAS_ELASTIC:
            raise ImportError("llama-index-vector-stores-elasticsearch is not installed")
        return ElasticsearchStore(
            es_url=f"http://{store_config.get('host', 'localhost')}:{store_config.get('port', 9200)}",
            index_name=store_config.get("index_name", "fastpath")
        )
    elif provider == "pgvector":
        if not HAS_PGVECTOR:
            raise ImportError("llama-index-vector-stores-postgres is not installed")
        return PGVectorStore.from_params(
            host=store_config.get("host", "localhost"),
            port=str(store_config.get("port", 5432)),
            user=store_config.get("user", "postgres"),
            password=store_config.get("password", "password"),
            database=store_config.get("database", "fastpath"),
            table_name=store_config.get("table_name", "vectors"),
            embed_dim=store_config.get("embed_dim", 384),
        )
    else:
        # Default: in-memory
        return None


def get_embedding_from_config(config: dict):
    if not HAS_LLAMA_INDEX:
        raise ImportError("KnowledgeBase requires llama-index; install llama-index-core + embeddings")
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
    Supports in-memory and external vector stores (chroma, milvus, elasticsearch, pgvector).
    If the vector store exists, it is reused; if not, it is created.
    On ingest, existing documents are updated.
    """
    def __init__(self, storage_path: str = None, config_path: str = "config.yaml"):
        if not HAS_LLAMA_INDEX:
            raise ImportError("KnowledgeBase requires llama-index; install llama-index-core + embeddings")
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

        # 3. Configure vector store
        vector_store = get_vector_store_from_config(config_path)
        self.vector_store = vector_store

    def ingest(self, data_dir: str):
        """Ingests documents from a directory. Updates if doc exists, adds if not."""
        documents = SimpleDirectoryReader(data_dir).load_data()
        if self.vector_store:
            storage_ctx = StorageContext.from_defaults(vector_store=self.vector_store)
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_ctx)
            else:
                for doc in documents:
                    self.index.insert(doc)
        else:
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(documents)
            else:
                for doc in documents:
                    self.index.insert(doc)

    def query(self, query: str, top_k: int = 5) -> List[dict]:
        if not self.index:
            raise ValueError("Knowledge base is empty. Ingest data first.")
        
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(query)
        return [
            {"text": node.get_text(), "score": node.get_score()}
            for node in response.source_nodes
        ] 