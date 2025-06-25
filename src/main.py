from fastpath.agent import Agent
from fastpath.memory import LongTermMemory
from fastpath.tools import Tool, LangChainToolAdapter
from fastpath.orchestrator import Orchestrator
from dotenv import load_dotenv

def main():
    load_dotenv()
    # Importar el wrapper de Serper de LangChain Community
    from langchain_community.utilities import GoogleSerperAPIWrapper
    # Instanciar la herramienta web y adaptarla (asignando nombre y descripción manualmente)
    web_tool = LangChainToolAdapter(
        GoogleSerperAPIWrapper(),
        name="GoogleSerperSearch",
        description="Búsqueda web en Google vía Serper (requiere SERPER_API_KEY)"
    )
    class DummyMemory(LongTermMemory):
        def add(self, embedding, metadata):
            pass
        def query(self, query_embedding, top_k=5):
            return []
    memory = DummyMemory(config={})
    from fastpath.llm_provider import get_llm_provider_from_config
    llm_provider = get_llm_provider_from_config()  # Usará HuggingFace por defecto
    researcher = Agent(name="Investigador", role="Investigar en internet y sintetizar información", tools=[web_tool], memory=memory, llm_provider=llm_provider)
    writer = Agent(name="Escritor", role="Redactar informes y resúmenes claros", tools=[], memory=memory, llm_provider=llm_provider)
    orchestrator = Orchestrator(agents=[researcher, writer], mode="smart")
    result = orchestrator.run("Investiga en detalle el tema 'Como vender servicios de IA a pymes y startups' usando fuentes web y redacta un informe")
    print("Resultado final:", result)

if __name__ == "__main__":
    main()