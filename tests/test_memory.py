from fastpath.memory import LongTermMemory

class DummyMemory(LongTermMemory):
    def add(self, embedding, metadata):
        self.last_embedding = embedding
        self.last_metadata = metadata
    def query(self, query_embedding, top_k=5):
        return [(self.last_embedding, self.last_metadata)] if hasattr(self, 'last_embedding') else []

def test_memory_add_and_query():
    memory = DummyMemory(storage_path=":memory:")
    embedding = [0.1, 0.2, 0.3]
    metadata = {"info": "test"}
    memory.add(embedding, metadata)
    results = memory.query([0.1, 0.2, 0.3])
    assert results[0][0] == embedding
    assert results[0][1] == metadata 