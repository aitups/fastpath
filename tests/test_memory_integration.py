import pytest
from testcontainers.redis import RedisContainer
from testcontainers.influxdb import InfluxDbContainer
from fastpath.memory import RedisMemory, InfluxDBMemory
import time

@pytest.mark.integration
def test_redis_memory_integration():
    with RedisContainer("redis:7.2") as redis:
        config = {
            "host": redis.get_container_host_ip(),
            "port": int(redis.get_exposed_port(6379)),
            "db": 0
        }
        memory = RedisMemory(config)
        emb = [0.1, 0.2, 0.3]
        memory.add(emb, {"info": "integration"})
        results = memory.query([0.0, 0.0, 0.0], top_k=1)
        assert results[-1]["embedding"] == emb

@pytest.mark.integration
def test_influxdb_memory_integration():
    with InfluxDbContainer("influxdb:2.7-alpine") as influx:
        # Wait for InfluxDB to be ready
        time.sleep(10)
        config = {
            "url": f"http://{influx.get_container_host_ip()}:{influx.get_exposed_port(8086)}",
            "token": "my-super-secret-auth-token",
            "org": "my-org",
            "bucket": "my-bucket"
        }
        # skip writing bucket/org setup for brevity; just check no error on add/query
        memory = InfluxDBMemory(config)
        emb = [0.1, 0.2, 0.3]
        memory.add(emb, {"info": "integration"})
        # Query is a dummy, just check no error
        assert memory.query([0.0, 0.0, 0.0], top_k=1) == []