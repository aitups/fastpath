from fastpath.knowledge_base import KnowledgeBase

class DummyKB(KnowledgeBase):
    def ingest(self, data):
        self.data = data
    def query(self, query, top_k=5):
        return [self.data] if hasattr(self, 'data') else []

def test_kb_ingest_and_query():
    kb = DummyKB(storage_path=":memory:")
    kb.ingest("dato de prueba")
    results = kb.query("pregunta")
    assert results[0] == "dato de prueba" 