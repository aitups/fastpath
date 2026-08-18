import pytest
import sys
import time
import requests
from testcontainers.chroma import ChromaContainer
from testcontainers.milvus import MilvusContainer
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer
from fastpath.knowledge_base import KnowledgeBase

class CustomElasticsearchContainer(DockerContainer):
    def __init__(self, image="docker.elastic.co/elasticsearch/elasticsearch:8.11.1"):
        super().__init__(image)
        self.with_exposed_ports(9200)
        self.with_env("discovery.type", "single-node")
        self.with_env("xpack.security.enabled", "false")

@pytest.mark.integration
def test_knowledge_base_chroma_integration(tmp_path):
    with ChromaContainer() as chroma:
        config_path = tmp_path / "config.yaml"
        # ChromaContainer levanta un servidor HTTP; usamos host/port del contenedor
        config_path.write_text(f"""
vector_store_provider: chroma
vector_store_config:
  chroma:
    host: "{chroma.get_container_host_ip()}"
    port: {chroma.get_exposed_port(8000)}
    collection_name: "fastpath"
embedding_provider: huggingface
embedding_model: all-MiniLM-L6-v2
""")
        kb = KnowledgeBase(config_path=str(config_path))
        # Simula ingestión de documentos (requiere ./docs o similar)
        # kb.ingest("./docs")
        # assert kb.query("test") == []
        assert kb.vector_store is not None

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="MilvusContainer panics on Windows Docker Desktop (container exits with goroutine panic)",
)
@pytest.mark.integration
def test_knowledge_base_milvus_integration(tmp_path):
    with MilvusContainer() as milvus:
        # Espera a que Milvus esté listo
        time.sleep(10)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"""
vector_store_provider: milvus
vector_store_config:
  milvus:
    host: "{milvus.get_container_host_ip()}"
    port: {milvus.get_exposed_port(19530)}
    collection_name: "fastpath"
embedding_provider: huggingface
embedding_model: all-MiniLM-L6-v2
""")
        kb = KnowledgeBase(config_path=str(config_path))
        assert kb.vector_store is not None

def _wait_elasticsearch_ready(host: str, port: int, timeout: int = 300) -> bool:
    """Espera hasta que el health-check de Elasticsearch responda 200."""
    url = f"http://{host}:{port}/_cluster/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                # Además de responder, debe estar 'green' o 'yellow' (no 'red')
                body = r.json()
                status = body.get("status", "red")
                if status in ("green", "yellow"):
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False

@pytest.mark.integration
def test_knowledge_base_elasticsearch_integration(tmp_path):
    with CustomElasticsearchContainer() as es:
        es.start()
        host = es.get_container_host_ip()
        port = es.get_exposed_port(9200)
        assert _wait_elasticsearch_ready(host, port), "Elasticsearch no alcanzó estado ready a tiempo"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"""
vector_store_provider: elasticsearch
vector_store_config:
  elasticsearch:
    host: "{es.get_container_host_ip()}"
    port: {es.get_exposed_port(9200)}
    index_name: "fastpath"
embedding_provider: huggingface
embedding_model: all-MiniLM-L6-v2
""")
        kb = KnowledgeBase(config_path=str(config_path))
        assert kb.vector_store is not None

@pytest.mark.integration
def test_knowledge_base_pgvector_integration(tmp_path):
    # 'pgvector/pgvector:latest' no existe en Docker Hub; usar el tag pg16
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        time.sleep(10)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"""
vector_store_provider: pgvector
vector_store_config:
  pgvector:
    host: "{pg.get_container_host_ip()}"
    port: {pg.get_exposed_port(5432)}
    user: "postgres"
    password: "postgres"
    database: "test"
    table_name: "vectors"
embedding_provider: huggingface
embedding_model: all-MiniLM-L6-v2
""")
        kb = KnowledgeBase(config_path=str(config_path))
        assert kb.vector_store is not None