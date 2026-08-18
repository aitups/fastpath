import pytest
import asyncio

from unittest.mock import MagicMock
from fastpath.agent import Agent
from fastpath.orchestrator import Orchestrator, ExecutionResult
from fastpath.tools import Tool
from fastpath.memory import LongTermMemory
from fastpath.llm_provider import LLMProvider

class DummyTool(Tool):
    def __init__(self, name="echo", description="Echo tool"):
        super().__init__(name=name, description=description)
    def run(self, text):
        return text

class DummyMemory(LongTermMemory):
    def add(self, embedding, metadata):
        pass
    def query(self, query_embedding, top_k=5):
        return []

def stub_embedding(text):
    return [0.0] * 8

class MockAgent(Agent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("embedding_function", stub_embedding)
        super().__init__(*args, **kwargs)
    async def act(self, task, context=None, callback=None):
        return f"{self.name} ejecuta {task}"

@pytest.fixture
def mock_llm_provider():
    return MagicMock(spec=LLMProvider)

def test_orchestrator_sequential(mock_llm_provider):
    tool = DummyTool()
    memory = DummyMemory(config={})
    agent1 = MockAgent(name="A1", role="R1", tools=[tool], memory=memory, llm_provider=mock_llm_provider)
    agent2 = MockAgent(name="A2", role="R2", tools=[tool], memory=memory, llm_provider=mock_llm_provider)
    orchestrator = Orchestrator(agents=[agent1, agent2], mode="sequential", llm_provider=mock_llm_provider)
    result = asyncio.run(orchestrator.run("tarea de prueba"))
    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.output == "A2 ejecuta tarea de prueba"

def test_orchestrator_collaborative(mock_llm_provider):
    tool = DummyTool()
    memory = DummyMemory(config={})
    # Cada agente responde con su nombre y el número de mensajes que ve
    class CollabAgent(Agent):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("embedding_function", stub_embedding)
            super().__init__(*args, **kwargs)
        async def act(self, task, context=None, callback=None):
            messages = context if context is not None else []
            return f"{self.name} vio {len(messages)} mensajes y tarea '{task}'"
    agent1 = CollabAgent(name="A1", role="R1", tools=[tool], memory=memory, llm_provider=mock_llm_provider)
    agent2 = CollabAgent(name="A2", role="R2", tools=[tool], memory=memory, llm_provider=mock_llm_provider)
    orchestrator = Orchestrator(agents=[agent1, agent2], mode="collaborative", llm_provider=mock_llm_provider)
    result = asyncio.run(orchestrator.run("tarea colaborativa"))
    # El resultado debe ser la respuesta del último agente
    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.output == f"A2 vio 2 mensajes y tarea 'tarea colaborativa'"