from typing import List, Any
from .agent import Agent
from .collaboration import Message, MessageBus
from .llm_provider import get_llm_provider_from_config, LLMProvider
from .strategies import ExecutionStrategy, STRATEGY_REGISTRY
import logging

# Logger formatted with timestamp and class name in blue color
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("fastpath.Orchestrator")

class Orchestrator:
    """
    Orquestador que gestiona la colaboración entre agentes usando estrategias desacopladas.
    """
    def __init__(self, agents: List[Agent], mode: str = None, strategy: ExecutionStrategy = None, llm_provider: LLMProvider = None):
        self.agents = agents
        # Permitir compatibilidad con el parámetro 'mode' usando el registro dinámico
        if strategy is not None:
            self.strategy = strategy
        elif mode is not None:
            strategy_cls = STRATEGY_REGISTRY.get(mode.lower(), STRATEGY_REGISTRY["sequential"])
            self.strategy = strategy_cls()
        else:
            self.strategy = STRATEGY_REGISTRY["sequential"]()
        self.message_bus = MessageBus() if self.strategy.__class__.__name__ == "CollaborativeStrategy" else None
        self.llm_provider = llm_provider or get_llm_provider_from_config()
        self.logger = logging.getLogger(f"fastpath.{self.__class__.__name__}")

    def run(self, task: str, context: Any = None) -> Any:
        return self.strategy.run(self, task, context)