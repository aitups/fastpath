from fastpath.memory import LongTermMemory
import pytest
from fastpath.memory import InMemoryMemory, get_memory_from_config
from unittest.mock import patch, MagicMock

class DummyMemory(LongTermMemory):
    def add(self, embedding, metadata):
        self.last_embedding = embedding
        self.last_metadata = metadata
    def query(self, query_embedding, top_k=5):
        return [(self.last_embedding, self.last_metadata)] if hasattr(self, 'last_embedding') else []

def test_memory_add_and_query():
    memory = DummyMemory(config={})
    embedding = [0.1, 0.2, 0.3]
    metadata = {"info": "test"}
    memory.add(embedding, metadata)
    results = memory.query([0.1, 0.2, 0.3])
    assert results[0][0] == embedding
    assert results[0][1] == metadata 

# Test InMemoryMemory

def test_inmemory_add_and_query():
    memory = InMemoryMemory(config={})
    emb1 = [0.1, 0.2, 0.3]
    emb2 = [0.4, 0.5, 0.6]
    memory.add(emb1, {"info": "a"})
    memory.add(emb2, {"info": "b"})
    results = memory.query([0.0, 0.0, 0.0], top_k=2)
    assert results == [(emb1, {"info": "a"}), (emb2, {"info": "b"})]

# Test get_memory_from_config for in_memory

def test_get_memory_from_config_inmemory(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
memory_provider: in_memory
memory_config:
  in_memory: {}
""")
    memory = get_memory_from_config(str(config_path))
    assert isinstance(memory, InMemoryMemory)

# Test initialization for other providers (mocked)

@patch("fastpath.memory.InfluxDBClient")
@patch("fastpath.memory.Point")
def test_influxdb_memory_init(mock_point, mock_client):
    from fastpath.memory import InfluxDBMemory
    config = {"url": "u", "token": "t", "org": "o", "bucket": "b"}
    memory = InfluxDBMemory(config)
    assert hasattr(memory, "client")
    assert memory.bucket == "b"

@patch("fastpath.memory.redis.Redis")
def test_redis_memory_init(mock_redis):
    from fastpath.memory import RedisMemory
    config = {"host": "localhost", "port": 6379, "db": 0}
    memory = RedisMemory(config)
    assert hasattr(memory, "client")
    assert memory.key == "fastpath:memory"

@patch("fastpath.memory.Cluster")
@patch("fastpath.memory.PasswordAuthenticator")
def test_couchbase_memory_init(mock_auth, mock_cluster):
    from fastpath.memory import CouchbaseMemory
    config = {"connection_string": "couchbase://localhost", "username": "a", "password": "b", "bucket": "fastpath"}
    mock_bucket = MagicMock()
    mock_cluster.return_value.bucket.return_value = mock_bucket
    mock_bucket.default_collection.return_value = MagicMock()
    memory = CouchbaseMemory(config)
    assert hasattr(memory, "cluster")
    assert hasattr(memory, "bucket")
    assert hasattr(memory, "collection")