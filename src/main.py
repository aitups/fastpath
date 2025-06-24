from fastpath.agent import Agent
from fastpath.memory import LongTermMemory
from fastpath.tools import Tool
from fastpath.orchestrator import Orchestrator

class DummyTool(Tool):
    def __init__(self):
        super().__init__(name="echo", description="Devuelve el input")
    def run(self, text):
        return text

class DummyMemory(LongTermMemory):
    def add(self, embedding, metadata):
        pass
    def query(self, query_embedding, top_k=5):
        return []

class DummyAgent(Agent):
    def act(self, task, context=None):
        print(f"{self.name} ({self.role}) ejecutando tarea: {task}")
        for tool in self.tools:
            print(f"Usando herramienta: {tool.name}")
        return f"Resultado de {self.name} para {task}"

def main():
    tool = DummyTool()
    memory = DummyMemory(storage_path=":memory:")
    agent = DummyAgent(name="Agente1", role="Investigador", tools=[tool], memory=memory)
    orchestrator = Orchestrator(agents=[agent], mode="sequential")
    result = orchestrator.run("Investiga el tema X")
    print("Resultado final:", result)

if __name__ == "__main__":
    main() 