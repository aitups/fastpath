import pytest
from fastpath.agent import Agent
from fastpath.tools import Tool
from fastpath.memory import LongTermMemory

class DummyTool(Tool):
    def __init__(self):
        super().__init__(name="echo", description="Echo tool")
    def run(self, text):
        return text

class DummyMemory(LongTermMemory):
    def add(self, embedding, metadata):
        pass
    def query(self, query_embedding, top_k=5):
        return []

class DummyAgent(Agent):
    def act(self, task, context=None):
        return f"{self.name} ejecuta {task}"

def test_agent_init():
    tool = DummyTool()
    memory = DummyMemory(storage_path=":memory:")
    agent = DummyAgent(name="TestAgent", role="Tester", tools=[tool], memory=memory)
    assert agent.name == "TestAgent"
    assert agent.role == "Tester"
    assert agent.tools[0].name == "echo"
    assert isinstance(agent.memory, DummyMemory)

def test_agent_act():
    tool = DummyTool()
    memory = DummyMemory(storage_path=":memory:")
    agent = DummyAgent(name="TestAgent", role="Tester", tools=[tool], memory=memory)
    result = agent.act("probar tarea")
    assert "TestAgent ejecuta probar tarea" == result 