import os
import requests
from typing import Any, List
from duckduckgo_search import DDGS

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

class DuckDuckGoSearchTool(Tool):
    """
    Herramienta de búsqueda en internet usando DuckDuckGo.
    No requiere API Key.
    """
    def __init__(self):
        super().__init__(
            name="duckduckgo_search", 
            description="Busca información actualizada en internet sobre un tema. Útil para encontrar noticias de última hora o tendencias."
        )

    def run(self, query: str, max_results: int = 5) -> str:
        # Sanitize: LLM may pass JSON or malformed text instead of a plain query
        import json as _json
        clean_query = query.strip()
        if clean_query.startswith("{") or clean_query.startswith("["):
            try:
                parsed = _json.loads(clean_query)
                if isinstance(parsed, dict):
                    # Pick the first string value from the dict
                    clean_query = next((v for v in parsed.values() if isinstance(v, str) and len(v) > 3), query)
                elif isinstance(parsed, list) and parsed:
                    clean_query = str(parsed[0])
            except Exception:
                # Strip JSON-like chars and use as-is
                clean_query = clean_query.strip("{}[]\"'")

        # Final guard: ensure we have a non-empty, reasonable query
        clean_query = clean_query.strip("\"' \n\t")
        if not clean_query or len(clean_query) < 3:
            clean_query = query.strip("{}[]\"' \n\t")

        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(clean_query, max_results=max_results)]
                if not results:
                    return "No se encontraron resultados."
                
                formatted_results = []
                for res in results:
                    formatted_results.append(f"- {res['title']}: {res['body']} ({res['href']})")
                return "\n".join(formatted_results)
        except Exception as e:
            return f"Error al buscar en DuckDuckGo: {str(e)}"

class SerperSearchTool(Tool):
    """
    Herramienta de búsqueda en internet usando Serper.dev (Google Search API).
    Requiere SERPER_API_KEY en el entorno.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        super().__init__(
            name="serper_search", 
            description="Busca información en Google de forma rápida y estructurada. Requiere una API Key de Serper.dev."
        )

    def run(self, query: str) -> str:
        if not self.api_key:
            return "Error: SERPER_API_KEY no configurada."
        
        url = "https://google.serper.dev/search"
        payload = {"q": query}
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            snippets = []
            if "organic" in data:
                for result in data["organic"][:5]:
                    snippets.append(f"- {result['title']}: {result['snippet']} ({result['link']})")
            
            return "\n".join(snippets) if snippets else "No se encontraron resultados orgánicos."
        except Exception as e:
            return f"Error al buscar en Serper: {str(e)}"

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