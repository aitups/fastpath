from typing import List, Any
from .memory import LongTermMemory
from .tools import Tool
from .llm_provider import LLMProvider
import logging

class Agent:
    """
    Represents an autonomous agent with a role, tools, persistent memory, and its own LLM provider.
    """
    def __init__(self, name: str, role: str, tools: List[Tool], memory: LongTermMemory, llm_provider: LLMProvider):
        self.name = name
        self.role = role
        self.tools = tools
        self.memory = memory
        self.llm_provider = llm_provider
        self.scratchpad = []  # Temporary memory for the current task
        self.logger = logging.getLogger(f"fastpath.{self.__class__.__name__}")

    def select_tool(self, task: str) -> Tool:
        """
        Usa el LLM del agente para decidir qué herramienta utilizar según la tarea y las descripciones de las herramientas.
        Si no hay herramientas disponibles, devuelve None.
        Si el LLM no selecciona ninguna herramienta válida, se usa la primera disponible como fallback.
        """
        if not self.tools:
            return None
        tool_descriptions = "\n".join([f"{i+1}. {tool.name}: {tool.description}" for i, tool in enumerate(self.tools)])
        prompt = (
            f"You are an expert agent. Given the following task: '{task}', and these tools:\n"
            f"{tool_descriptions}\n"
            "Which tool (by name) is the most appropriate to use? Respond with only the tool name."
        )
        self.logger.info(f"[{self.__class__.__name__}] Seleccionando herramienta para tarea: '{task}'")
        tool_name = self.llm_provider.generate(prompt).strip()
        self.logger.info(f"[{self.__class__.__name__}] LLM sugiere herramienta: '{tool_name}'")
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                return tool
        # Fallback: return the first tool if available
        return self.tools[0] if self.tools else None

    def act(self, task: str, context: Any = None, callback: Any = None) -> Any:
        def emit(msg: str):
            self.logger.info(msg)
            if callback:
                callback(msg)

        emit(f"[{self.name}] Iniciando tarea: '{task}'")
        
        emit(f"[{self.name}] Analizando herramientas disponibles...")
        tool = self.select_tool(task)
        self.scratchpad.append({"task": task, "selected_tool": tool.name if tool else None})
        
        if tool:
            emit(f"[{self.name}] Herramienta seleccionada: {tool.name}")
            emit(f"[{self.name}] Generando parámetros para {tool.name}...")
            
            prompt = (
                f"You are an autonomous agent. Your role is: {self.role}.\n"
                f"Task: '{task}'.\n"
                f"Tool: {tool.name} - {tool.description}\n"
                "Generate ONLY the search query text to pass to this tool. "
                "Reply with NOTHING but the raw search query - no JSON, no markdown, no explanation."
            )
            tool_params_raw = self.llm_provider.generate(prompt)
            
            if isinstance(tool_params_raw, str):
                # Robustly handle repetition loops by taking only the first non-empty line
                lines = [l.strip() for l in tool_params_raw.split('\n') if l.strip()]
                tool_params = lines[0] if lines else task
                # Basic length safety for search tools
                if len(tool_params) > 300:
                    tool_params = tool_params[:300]
            else:
                tool_params = task
                
            emit(f"[{self.name}] Ejecutando {tool.name} con: {tool_params}")
            tool_response = tool.run(tool_params)
            self.scratchpad.append({"tool_response": tool_response})
            
            if context:
                tool_response = f"{tool_response}\nContext: {context}"
                
            emit(f"[{self.name}] Procesando respuesta de la herramienta...")
            prompt = (
                f"You are an autonomous agent. Your role is: {self.role}.\n"
                f"You have been assigned the following task: '{task}'.\n"
                f"The tool '{tool.name}' returned the following response: '{tool_response}'.\n"
                "Using this information, please provide an answer for the task."
            )
            final_raw = self.llm_provider.generate(prompt)
            emit(f"[{self.name}] Tarea completada.")
            
            if isinstance(final_raw, str):
                return final_raw.strip()
            return tool_response
        else:
            emit(f"[{self.name}] No se requieren herramientas. Generando resolución directa...")
            if context:
                task = f"{task}\nContext: {context}"
            prompt = (
                f"You are an autonomous agent. Your role is: {self.role}.\n"
                f"You have been assigned the following task: '{task}'.\n"
                "There are no tools available. Please use your expertise and role to complete the task as best as possible."
            )
            result = self.llm_provider.generate(prompt)
            return result if isinstance(result, str) else str(result)