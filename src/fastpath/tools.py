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

class LangChainToolAdapter(Tool):
    """
    Adaptador para usar herramientas de LangChain como herramientas FastPath.
    """
    def __init__(self, langchain_tool, name=None, description=None):
        # Si la herramienta no tiene name/description, usar los proporcionados o defaults
        tool_name = getattr(langchain_tool, 'name', None) or name or langchain_tool.__class__.__name__
        tool_desc = getattr(langchain_tool, 'description', None) or description or "Herramienta LangChain adaptada"
        super().__init__(name=tool_name, description=tool_desc)
        self.langchain_tool = langchain_tool

    def run(self, text, **kwargs):
        # Soporta invoke, __call__, run y search
        if hasattr(self.langchain_tool, 'invoke'):
            return self.langchain_tool.invoke(text)
        elif hasattr(self.langchain_tool, '__call__'):
            return self.langchain_tool(text)
        elif hasattr(self.langchain_tool, 'run'):
            return self.langchain_tool.run(text)
        elif hasattr(self.langchain_tool, 'search'):
            return self.langchain_tool.search(text)
        else:
            raise NotImplementedError("La herramienta de LangChain no tiene método 'invoke', '__call__', 'run' ni 'search'.")