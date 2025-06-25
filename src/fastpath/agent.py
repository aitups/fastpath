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

    def act(self, task: str, context: Any = None) -> Any:
        self.logger.info(f"[{self.__class__.__name__}] Ejecutando tarea: '{task}'")
        tool = self.select_tool(task)
        self.scratchpad.append({"task": task, "selected_tool": tool.name if tool else None})
        # Hay que tener en cuenta el contexto como parametro y el de la memoria
        if tool:
            self.logger.info(f"[{self.__class__.__name__}] Usando herramienta: {tool.name} - {tool.description}")
            prompt = (
                f"You are an autonomous agent. Your role is: {self.role}.\n"
                f"You have been assigned the following task: '{task}'.\n"
                f"You have access to the following tool: {tool.name} - {tool.description}\n"
                "Please generate the parameters needed to run the tool\n"
                "and return them as a single string. Do not include any other text."
            )
            tool_params = self.llm_provider.generate(prompt).strip()
            self.logger.info(f"[{self.__class__.__name__}] Parámetros generados para la herramienta: {tool_params}")
            tool_response = tool.run(tool_params)
            self.scratchpad.append({"tool_response": tool_response})
            if context:
                self.logger.info(f"[{self.__class__.__name__}] Contexto proporcionado: {context}")
                tool_response += f"\nContext: {context}"
            prompt = (
                f"You are an autonomous agent. Your role is: {self.role}.\n"
                f"You have been assigned the following task: '{task}'.\n"
                f"The tool '{tool.name}' returned the following response: '{tool_response}'.\n"
                "Using this information, please provide an answer for the task."
            )
            return self.llm_provider.generate(prompt).strip()
        else:
            self.logger.info(f"[{self.__class__.__name__}] Sin herramientas, usando LLM para resolver la tarea.")
            if context:
                self.logger.info(f"[{self.__class__.__name__}] Contexto proporcionado: {context}")
                task += f"\nContext: {context}"
            prompt = (
                f"You are an autonomous agent. Your role is: {self.role}.\n"
                f"You have been assigned the following task: '{task}'.\n"
                "There are no tools available. Please use your expertise and role to complete the task as best as possible."
            )
            return self.llm_provider.generate(prompt)