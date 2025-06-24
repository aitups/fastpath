from typing import List, Any
from .agent import Agent

class Orchestrator:
    """
    Orquestador que gestiona la colaboración entre agentes.
    Modos: secuencial y colaborativo.
    """
    def __init__(self, agents: List[Agent], mode: str = "sequential"):
        self.agents = agents
        self.mode = mode  # 'sequential' o 'collaborative'

    def run(self, task: str, context: Any = None) -> Any:
        if self.mode == "sequential":
            return self._run_sequential(task, context)
        elif self.mode == "collaborative":
            return self._run_collaborative(task, context)
        else:
            raise ValueError(f"Modo de orquestador desconocido: {self.mode}")

    def _run_sequential(self, task: str, context: Any = None) -> Any:
        # Implementación simplificada
        result = context
        for agent in self.agents:
            result = agent.act(task, result)
        return result

    def _run_collaborative(self, task: str, context: Any = None) -> Any:
        # Implementación simplificada (bus de mensajes)
        raise NotImplementedError 