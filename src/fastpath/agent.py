from typing import List, Any
from .memory import LongTermMemory
from .tools import Tool

class Agent:
    """
    Representa un agente autónomo con rol, herramientas y memoria persistente.
    """
    def __init__(self, name: str, role: str, tools: List[Tool], memory: LongTermMemory):
        self.name = name
        self.role = role
        self.tools = tools
        self.memory = memory
        self.scratchpad = []  # Memoria temporal de la tarea actual

    def act(self, task: str, context: Any = None) -> Any:
        """
        Ejecuta la acción principal del agente para una tarea dada.
        """
        raise NotImplementedError 