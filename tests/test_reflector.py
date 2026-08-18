import pytest
import asyncio
from unittest.mock import MagicMock
from fastpath.reflector import ReflectorAgent
from fastpath.memory import LongTermMemory


class DummyMemory(LongTermMemory):
    def __init__(self, config=None):
        super().__init__(config or {})
        self.added = []

    def add(self, embedding, metadata):
        self.added.append({"embedding": embedding, "metadata": metadata})

    def query(self, query_embedding, top_k=5):
        return []


def stub_embedding(text):
    return [0.1] * 8


class FakeAgent:
    def __init__(self, name, memory):
        self.name = name
        self.memory = memory


def test_reflector_persists_learning():
    llm = MagicMock()
    llm.generate.return_value = "Aprendizaje clave: escalar con estrategias jerárquicas."

    memory = DummyMemory()
    agent = FakeAgent(name="Analista", memory=memory)
    reflector = ReflectorAgent(llm_provider=llm, embedding_function=stub_embedding)

    scratchpad = [
        {"task": "investigar", "selected_tool": "web"},
        {"tool_response": "datos de mercado"},
        {"result": "informe final"},
    ]

    result = asyncio.run(reflector.reflect(agent, "tarea completa", scratchpad))

    assert result is not None
    assert result["summary"] == "Aprendizaje clave: escalar con estrategias jerárquicas."
    assert result["task"] == "tarea completa"
    assert result["agent"] == "Analista"
    # El embedding y metadata se guardaron en el LTM
    assert len(memory.added) == 1
    assert memory.added[0]["metadata"]["agent"] == "Analista"
    assert len(memory.added[0]["embedding"]) == 8


def test_reflector_fallback_to_scratchpad_result():
    """Si el LLM falla (None), se usa el resultado final del scratchpad."""
    llm = MagicMock()
    llm.generate.return_value = None

    memory = DummyMemory()
    agent = FakeAgent(name="Investigador", memory=memory)
    reflector = ReflectorAgent(llm_provider=llm, embedding_function=stub_embedding)

    scratchpad = [
        {"task": "buscar", "selected_tool": "web"},
        {"tool_response": "resultados"},
        {"result": "El mercado crece un 20% anual."},
    ]

    result = asyncio.run(reflector.reflect(agent, "analisis", scratchpad))

    assert result is not None
    assert "El mercado crece un 20% anual." in result["summary"]
    assert len(memory.added) == 1


def test_reflector_skips_empty_scratchpad():
    llm = MagicMock()
    memory = DummyMemory()
    agent = FakeAgent(name="A", memory=memory)
    reflector = ReflectorAgent(llm_provider=llm, embedding_function=stub_embedding)

    result = asyncio.run(reflector.reflect(agent, "tarea", []))
    assert result is None
    assert len(memory.added) == 0


def test_reflector_skips_agent_without_memory():
    llm = MagicMock()
    agent = FakeAgent(name="SinMemoria", memory=None)
    reflector = ReflectorAgent(llm_provider=llm, embedding_function=stub_embedding)

    result = asyncio.run(reflector.reflect(agent, "tarea", [{"result": "x"}]))
    assert result is None
    llm.generate.assert_not_called()


def test_reflector_no_embedding_available():
    llm = MagicMock()
    llm.generate.return_value = "algo"

    memory = DummyMemory()
    agent = FakeAgent(name="A", memory=memory)
    # embedding_function que devuelve None → no se puede persistir
    reflector = ReflectorAgent(llm_provider=llm, embedding_function=lambda t: None)

    result = asyncio.run(reflector.reflect(agent, "tarea", [{"result": "x"}]))
    assert result is None
    assert len(memory.added) == 0