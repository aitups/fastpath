from fastpath.agent import Agent
from fastpath.memory import LongTermMemory
from fastpath.tools import Tool, LangChainToolAdapter
from langchain_community.utilities import GoogleSerperAPIWrapper
from fastpath.orchestrator import Orchestrator
from dotenv import load_dotenv

def main_smart():
    load_dotenv()
    # Instanciar la herramienta web y adaptarla (asignando nombre y descripción manualmente)
    web_tool = LangChainToolAdapter(
        GoogleSerperAPIWrapper(),
        name="GoogleSerperSearch",
        description="Búsqueda web en Google"
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

def main_hierarchical():
    load_dotenv()
    from fastpath.strategies import HierarchicalStrategy
    from fastpath.llm_provider import get_llm_provider_from_config
    llm_provider = get_llm_provider_from_config()  # Usará HuggingFace por defecto
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
    researcher = Agent(name="Investigador", role="Investigar en internet y sintetizar información", tools=[web_tool], memory=memory, llm_provider=llm_provider)
    writer = Agent(name="Escritor", role="Redactar informes y resúmenes claros", tools=[], memory=memory, llm_provider=llm_provider)
    agents = [researcher, writer]
    orchestrator = Orchestrator(agents=agents, strategy=HierarchicalStrategy())
    task = "Preparar un informe sobre cómo vender servicios de IA a pymes y startups. Primero, el investigador debe buscar en internet sobre: 1) beneficios de la IA para pymes, 2) casos de éxito de IA en startups, 3) estrategias de venta de IA. Después, el escritor debe usar cada resultado para crear un informe estructurado."
    result = orchestrator.run(task)
    print("Resultado jerárquico:", result)

def main():
    # With argument -m or --mode invoke the cooresponding main function
    import argparse
    parser = argparse.ArgumentParser(description="Run the FastPath orchestrator with different strategies.")
    parser.add_argument('-m', '--mode', choices=['smart', 'hierarchical'], default='smart',
                        help="Select the mode of operation: 'smart' for collaborative strategy, 'hierarchical' for hierarchical strategy.")
    args = parser.parse_args()
    if args.mode == 'smart':
        main_smart()
    elif args.mode == 'hierarchical':
        main_hierarchical()
    else:
        print("Modo no reconocido. Usa 'smart' o 'hierarchical'.")
    

if __name__ == "__main__":
    main()