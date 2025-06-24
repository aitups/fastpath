from typing import List

class LongTermMemory:
    """
    Memoria persistente basada en vector store para cada agente.
    """
    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def add(self, embedding: List[float], metadata: dict):
        raise NotImplementedError

    def query(self, query_embedding: List[float], top_k: int = 5) -> list:
        raise NotImplementedError 