import pytest
import asyncio
import time

from fastpath.agent import Agent
from fastpath.memory import LongTermMemory
from fastpath.tools import LangChainToolAdapter
from fastpath.orchestrator import Orchestrator
from fastpath.strategies import HierarchicalStrategy
from fastpath.llm_provider import GeminiProvider
from dotenv import load_dotenv

class DummyMemory(LongTermMemory):
    def add(self, embedding, metadata):
        pass
    def query(self, query_embedding, top_k=5):
        return []

@pytest.mark.integration
def test_hierarchical_strategy():
    """
    Test de integración: requiere SERPER_API_KEY y GEMINI_API_KEY en .env
    (el config.yaml apunta a un llamacpp docker que no está disponible localmente).
    """
    load_dotenv()
    import os
    from langchain_community.utilities import GoogleSerperAPIWrapper
    web_tool = LangChainToolAdapter(
        GoogleSerperAPIWrapper(),
        name="GoogleSerperSearch",
        description="Búsqueda web en Google vía Serper (requiere SERPER_API_KEY)"
    )
    memory = DummyMemory(config={})
    # Gemini es el provider accesible con la API key disponible en .env
    llm_provider = GeminiProvider({
        "model": "gemini-2.5-flash",
        "api_key": os.getenv("GEMINI_API_KEY"),
    })
    # Un solo agente investigador y uno escritor
    researcher = Agent(name="Investigador", role="Investigar en internet y sintetizar información", tools=[web_tool], memory=memory, llm_provider=llm_provider)
    writer = Agent(name="Escritor", role="Redactar informes y resúmenes claros", tools=[], memory=memory, llm_provider=llm_provider)
    agents = [researcher, writer]
    orchestrator = Orchestrator(agents=agents, strategy=HierarchicalStrategy(), llm_provider=llm_provider)
    # Tarea: preparar un informe sobre "IA para pymes y startups"
    task = "Preparar un informe sobre cómo vender servicios de IA a pymes y startups. Primero, el investigador debe buscar en internet sobre: 1) beneficios de la IA para pymes, 2) casos de éxito de IA en startups, 3) estrategias de venta de IA. Después, el escritor debe usar cada resultado para crear un informe estructurado."
    # Los proveedores externos (Gemini) pueden devolver 429 Too Many Requests
    # cuando se ejecutan muchas llamadas seguidas (rate limit del free tier).
    # El backoff progresivo da tiempo a que el límite se restaure; si persiste,
    # se marca como skip (infraestructura externa) en vez de fallo de código.
    result = None
    for attempt in range(3):
        result = asyncio.run(orchestrator.run(task))
        if result.success:
            break
        if result.error and "429" in str(result.error):
            time.sleep(60 * (attempt + 1))  # backoff: 60s, 120s, 180s
        else:
            break

    assert result is not None
    if not result.success:
        err = str(result.error or "")
        if "429" in err or "Too Many Requests" in err:
            pytest.skip(f"Rate limit del proveedor externo impide completar: {err[:120]}")
    assert result.success is True, f"Ejecución falló: {result.error}"
    summary = result.details["summary"]
    print("Resultado jerárquico:", summary)
    assert "summary" in result.details
    assert bool(summary.strip()), "El resumen final está vacío"
    # El LLM decide el número de subtareas; al menos debe haber ejecutado alguna
    assert len(result.details["subtasks"]) >= 1
