import pytest
from unittest.mock import MagicMock
from fastpath.agent import Agent
from fastpath.orchestrator import Orchestrator
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

class MockAgent(Agent):
    def act(self, task, context=None):
        return f"{self.name} ejecuta {task}"

@pytest.fixture
def mock_llm_provider():
    return MagicMock(spec=LLMProvider)

def test_orchestrator_sequential(mock_llm_provider):
    tool = DummyTool()
    memory = DummyMemory(config={})
    agent1 = MockAgent(name="A1", role="R1", tools=[tool], memory=memory, llm_provider=mock_llm_provider)
    agent2 = MockAgent(name="A2", role="R2", tools=[tool], memory=memory, llm_provider=mock_llm_provider)
    orchestrator = Orchestrator(agents=[agent1, agent2], mode="sequential")
    result = orchestrator.run("tarea de prueba")
    assert result == "A2 ejecuta tarea de prueba"