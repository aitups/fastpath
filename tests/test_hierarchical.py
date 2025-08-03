import pytest
from fastpath.agent import Agent
from fastpath.memory import LongTermMemory
from fastpath.tools import LangChainToolAdapter
from fastpath.orchestrator import Orchestrator
from fastpath.strategies import HierarchicalStrategy
from dotenv import load_dotenv

class DummyMemory(LongTermMemory):
    def add(self, embedding, metadata):
        pass
    def query(self, query_embedding, top_k=5):
        return []

def test_hierarchical_strategy():
    load_dotenv()
    from langchain_community.utilities import GoogleSerperAPIWrapper
    web_tool = LangChainToolAdapter(
        GoogleSerperAPIWrapper(),
        name="GoogleSerperSearch",
        description="Búsqueda web en Google vía Serper (requiere SERPER_API_KEY)"
    )
    memory = DummyMemory(config={})
    from fastpath.llm_provider import get_llm_provider_from_config
    llm_provider = get_llm_provider_from_config()
    # Un solo agente investigador y uno escritor
    researcher = Agent(name="Investigador", role="Investigar en internet y sintetizar información", tools=[web_tool], memory=memory, llm_provider=llm_provider)
    writer = Agent(name="Escritor", role="Redactar informes y resúmenes claros", tools=[], memory=memory, llm_provider=llm_provider)
    agents = [researcher, writer]
    orchestrator = Orchestrator(agents=agents, strategy=HierarchicalStrategy())
    # Tarea: preparar un informe sobre "IA para pymes y startups"
    task = "Preparar un informe sobre cómo vender servicios de IA a pymes y startups. Primero, el investigador debe buscar en internet sobre: 1) beneficios de la IA para pymes, 2) casos de éxito de IA en startups, 3) estrategias de venta de IA. Después, el escritor debe usar cada resultado para crear un informe estructurado."
    result = orchestrator.run(task)
    print("Resultado jerárquico:", result)
    assert "summary" in result
    assert len(result["subtasks"]) >= 4  # 3 búsquedas + 1 informe
