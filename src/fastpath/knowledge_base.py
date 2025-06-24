from typing import Any, List

class KnowledgeBase:
    """
    Base de conocimiento compartida (RAG) entre agentes.
    """
    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def ingest(self, data: Any):
        raise NotImplementedError

    def query(self, query: str, top_k: int = 5) -> List[Any]:
        raise NotImplementedError 