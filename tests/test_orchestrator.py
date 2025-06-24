from fastpath.agent import Agent
from fastpath.orchestrator import Orchestrator
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

def test_orchestrator_sequential():
    tool = DummyTool()
    memory = DummyMemory(storage_path=":memory:")
    agent1 = DummyAgent(name="A1", role="R1", tools=[tool], memory=memory)
    agent2 = DummyAgent(name="A2", role="R2", tools=[tool], memory=memory)
    orchestrator = Orchestrator(agents=[agent1, agent2], mode="sequential")
    result = orchestrator.run("tarea de prueba")
    assert result == "A2 ejecuta tarea de prueba" 