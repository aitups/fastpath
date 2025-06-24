from typing import Any

class Tool:
    """
    Interfaz base para herramientas que puede usar un agente.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError

# Registro global de herramientas
TOOLS_REGISTRY = {}

def register_tool(tool: Tool):
    TOOLS_REGISTRY[tool.name] = tool 