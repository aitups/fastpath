from typing import List, Any
from .agent import Agent
from .collaboration import Message, MessageBus

class Orchestrator:
    """
    Orquestador que gestiona la colaboración entre agentes.
    Modos: secuencial y colaborativo.
    """
    def __init__(self, agents: List[Agent], mode: str = "sequential"):
        self.agents = agents
        self.mode = mode  # 'sequential' o 'collaborative'
        self.message_bus = MessageBus() if mode == "collaborative" else None

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
        """
        Modo colaborativo: los agentes interactúan a través de un bus de mensajes.
        Cada agente puede leer todos los mensajes y decidir si responde o publica uno nuevo.
        El orquestador modera y decide cuándo finalizar (aquí, tras un ciclo por agente).
        """
        self.message_bus.post(Message(sender="orchestrator", content=task))
        for agent in self.agents:
            # Cada agente ve todos los mensajes y decide si responde
            messages = self.message_bus.get_all()
            response = agent.act(task, context=messages)
            self.message_bus.post(Message(sender=agent.name, content=response))
        # El resultado final puede ser el último mensaje publicado
        return self.message_bus.get_all()[-1].content if self.message_bus.get_all() else None