import pytest
from testcontainers.redis import RedisContainer
from testcontainers.influxdb import InfluxDbContainer
from testcontainers.core.container import DockerContainer
from fastpath.memory import RedisMemory, InfluxDBMemory, CouchbaseMemory
import time
import socket

@pytest.mark.integration
def test_redis_memory_integration():
    with RedisContainer() as redis:
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
    with InfluxDbContainer() as influx:
        # Wait for InfluxDB to be ready
        time.sleep(3)
        config = {
            "url": influx.get_url(),
            "token": "my-token",  # testcontainers sets up a default token
            "org": "test-org",
            "bucket": "test-bucket"
        }
        # Create the bucket and org if needed (skipped for brevity)
        memory = InfluxDBMemory(config)
        emb = [0.1, 0.2, 0.3]
        memory.add(emb, {"info": "integration"})
        # Query is a dummy, just check no error
        assert memory.query([0.0, 0.0, 0.0], top_k=1) == []

@pytest.mark.integration
def test_couchbase_memory_integration():
    with CouchbaseContainer() as cb:
        # Wait for Couchbase to be ready
        time.sleep(10)
        config = {
            "connection_string": cb.get_connection_url(),
            "username": cb.username,
            "password": cb.password,
            "bucket": cb.bucket_name
        }
        memory = CouchbaseMemory(config)
        emb = [0.1, 0.2, 0.3]
        memory.add(emb, {"info": "integration"})
        # Query is a dummy, just check no error
        assert memory.query([0.0, 0.0, 0.0], top_k=1) == []

class CustomCouchbaseContainer(DockerContainer):
    def __init__(self, image="couchbase:community-7.1.0"):
        super().__init__(image)
        self.with_exposed_ports(8091, 8092, 8093, 11210, 11207, 11280, 18091, 18092, 18093)
        self.with_env("CB_USERNAME", "Administrator")
        self.with_env("CB_PASSWORD", "password")
        self.with_env("CB_BUCKET", "fastpath")
        # Forzar el hostname externo para el cluster map
        host_ip = socket.gethostbyname(socket.gethostname())
        self.with_env("CB_SERVER_HOST", host_ip)
        self.with_env("HOSTNAME", host_ip)

    def wait_until_ready(self):
        for _ in range(120):
            try:
                import requests
                url = f"http://{self.get_container_host_ip()}:{self.get_exposed_port(8091)}/ui/index.html"
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("Couchbase did not become ready in time")

@pytest.mark.integration
def test_couchbase_memory_integration_full(tmp_path):
    from couchbase.cluster import Cluster, ClusterOptions
    from couchbase.auth import PasswordAuthenticator
    from couchbase.management.buckets import CreateBucketSettings

    with CustomCouchbaseContainer() as cb:
        cb.start()
        cb.wait_until_ready()
        host = cb.get_container_host_ip()
        port = cb.get_exposed_port(8091)
        conn_str = f"couchbase://{host}"
        username = "Administrator"
        password = "password"
        bucket_name = "fastpath"

        # 1. Inicializa el cluster y crea el bucket si no existe
        cluster = Cluster(conn_str, ClusterOptions(PasswordAuthenticator(username, password)))
        bucket_mgr = cluster.buckets()
        try:
            bucket_mgr.create_bucket(CreateBucketSettings(name=bucket_name, ram_quota_mb=100))
        except Exception:
            pass  # Ya existe

        # Espera a que el bucket esté disponible y operativo
        for _ in range(90):
            try:
                bucket = cluster.bucket(bucket_name)
                bucket.on_connect()
                collection = bucket.default_collection()
                # Intenta una operación real
                collection.upsert("test-doc", {"test": 1})
                break
            except Exception as e:
                print(f"Esperando bucket... {e}")
                time.sleep(3)
        else:
            raise RuntimeError("Bucket not available or not ready")

        # 2. Usa el backend de memoria
        config = {
            "connection_string": conn_str,
            "username": username,
            "password": password,
            "bucket": bucket_name
        }
        memory = CouchbaseMemory(config)
        emb = [0.1, 0.2, 0.3]
        metadata = {"info": "integration"}
        memory.add(emb, metadata)

        # 3. Recupera el embedding (no hay vector search, así que solo verifica que no falle)
        results = memory.query([0.0, 0.0, 0.0], top_k=1)
        assert isinstance(results, list) 