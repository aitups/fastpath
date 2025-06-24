from typing import List, Any, Dict
from abc import ABC, abstractmethod
from fastpath.config import Config

# Optional imports for external backends
try:
    from influxdb_client import InfluxDBClient, Point
    HAS_INFLUX = True
except ImportError:
    HAS_INFLUX = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    from couchbase.cluster import Cluster, ClusterOptions
    from couchbase.auth import PasswordAuthenticator
    HAS_COUCHBASE = True
except ImportError:
    HAS_COUCHBASE = False

class LongTermMemory(ABC):
    """
    Abstract base class for persistent memory providers.
    """
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def add(self, embedding: List[float], metadata: dict):
        pass

    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int = 5) -> list:
        pass

class InMemoryMemory(LongTermMemory):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.data = []  # List of (embedding, metadata)

    def add(self, embedding: List[float], metadata: dict):
        self.data.append((embedding, metadata))

    def query(self, query_embedding: List[float], top_k: int = 5) -> list:
        # Dummy: just return the last k items
        return self.data[-top_k:]

class InfluxDBMemory(LongTermMemory):
    def __init__(self, config: Dict):
        if not HAS_INFLUX:
            raise ImportError("influxdb-client is not installed")
        super().__init__(config)
        self.client = InfluxDBClient(
            url=config["url"],
            token=config["token"],
            org=config["org"]
        )
        self.bucket = config["bucket"]

    def add(self, embedding: List[float], metadata: dict):
        point = Point("embedding")
        for i, v in enumerate(embedding):
            point.field(f"e{i}", v)
        for k, v in metadata.items():
            point.tag(k, str(v))
        self.client.write_api().write(bucket=self.bucket, record=point)

    def query(self, query_embedding: List[float], top_k: int = 5) -> list:
        # Dummy: just return empty (real implementation would require vector search support)
        return []

class RedisMemory(LongTermMemory):
    def __init__(self, config: Dict):
        if not HAS_REDIS:
            raise ImportError("redis is not installed")
        super().__init__(config)
        self.client = redis.Redis(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            db=config.get("db", 0)
        )
        self.key = "fastpath:memory"

    def add(self, embedding: List[float], metadata: dict):
        # Store as a serialized string (for demo)
        import json
        self.client.rpush(self.key, json.dumps({"embedding": embedding, "metadata": metadata}))

    def query(self, query_embedding: List[float], top_k: int = 5) -> list:
        import json
        items = self.client.lrange(self.key, -top_k, -1)
        return [json.loads(item) for item in items]

class CouchbaseMemory(LongTermMemory):
    def __init__(self, config: Dict):
        if not HAS_COUCHBASE:
            raise ImportError("couchbase is not installed")
        super().__init__(config)
        auth = PasswordAuthenticator(config["username"], config["password"])
        self.cluster = Cluster(config["connection_string"], ClusterOptions(auth))
        self.bucket = self.cluster.bucket(config["bucket"])
        self.collection = self.bucket.default_collection()

    def add(self, embedding: List[float], metadata: dict):
        import uuid
        doc_id = str(uuid.uuid4())
        doc = {"embedding": embedding, "metadata": metadata}
        self.collection.upsert(doc_id, doc)

    def query(self, query_embedding: List[float], top_k: int = 5) -> list:
        # Dummy: just return the last k docs (real implementation would require vector search support)
        # For demo, just return empty
        return []

def get_memory_from_config(config_path: str = "config.yaml") -> LongTermMemory:
    config_data = Config(config_path).data
    provider = config_data.get("memory_provider", "in_memory").lower()
    memory_config = config_data.get("memory_config", {})
    provider_config = memory_config.get(provider, {})

    if provider == "in_memory":
        return InMemoryMemory(provider_config)
    elif provider == "influxdb":
        return InfluxDBMemory(provider_config)
    elif provider == "redis":
        return RedisMemory(provider_config)
    elif provider == "couchbase":
        return CouchbaseMemory(provider_config)
    else:
        raise ValueError(f"Unknown or unsupported memory provider: {provider}") 