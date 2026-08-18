from typing import List, Any, Optional, Callable
import asyncio
from .memory import LongTermMemory
from .tools import Tool
from .llm_provider import LLMProvider
import logging

class Agent:
    """
    Represents an autonomous agent with a role, tools, persistent memory, and its own LLM provider.
    """
    def __init__(
        self,
        name: str,
        role: str,
        tools: List[Tool],
        memory: LongTermMemory,
        llm_provider: LLMProvider,
        embedding_function: Optional[Callable[[str], Optional[List[float]]]] = None,
        memory_k: int = 3,
    ):
        self.name = name
        self.role = role
        self.tools = tools
        self.memory = memory
        self.llm_provider = llm_provider
        self.embedding_function = embedding_function
        self.memory_k = memory_k
        self.scratchpad = []  # Temporary memory for the current task
        self.logger = logging.getLogger(f"fastpath.{self.__class__.__name__}")

    async def _retrieve_memory_context(self, task: str) -> str:
        """
        Recuperación proactiva de LTM (REQ-6.3): consulta la memoria persistente
        del agente con el embedding de la tarea y devuelve un contexto textual
        (o string vacío si no hay memoria o no se puede generar el embedding).
        """
        if self.memory is None:
            return ""

        # Fallback a la utilidad global si no se inyectó embedding_function
        embedding_fn = self.embedding_function
        if embedding_fn is None:
            from .embedding_utils import get_text_embedding
            embedding_fn = lambda text: get_text_embedding(text)  # noqa: E731

        query_embedding = embedding_fn(task)
        if not query_embedding:
            return ""

        try:
            # query() es síncrono; puede ser lento para vectores → thread
            results = await asyncio.to_thread(self.memory.query, query_embedding, self.memory_k)
        except Exception as e:
            self.logger.warning(f"[{self.name}] Fallo al consultar LTM: {e}")
            return ""

        if not results:
            return ""

        formatted = []
        for item in results:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            text = metadata.get("summary") or metadata.get("text") or str(item)
            formatted.append(text)

        return "\n".join(f"- {t}" for t in formatted[:self.memory_k])

    async def select_tool(self, task: str) -> Tool:
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
        tool_name = await asyncio.to_thread(self.llm_provider.generate, prompt)
        if not isinstance(tool_name, str):
            tool_name = str(tool_name)
        tool_name = tool_name.strip()
        self.logger.info(f"[{self.__class__.__name__}] LLM sugiere herramienta: '{tool_name}'")
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                return tool
        # Fallback: return the first tool if available
        return self.tools[0] if self.tools else None

    async def act(self, task: str, context: Any = None, callback: Any = None) -> Any:
        def emit(msg: str):
            self.logger.info(msg)
            if callback:
                callback(msg)

        emit(f"[{self.name}] Iniciando tarea: '{task}'")

        # --- Memoria proactiva (REQ-6.3): recuperar contexto del LTM ---
        memory_context = await self._retrieve_memory_context(task)
        if memory_context:
            emit(f"[{self.name}] Contexto recuperado de la memoria a largo plazo:")
            emit(f"[{self.name}] {memory_context[:200]}{'...' if len(memory_context) > 200 else ''}")

        emit(f"[{self.name}] Analizando herramientas disponibles...")
        tool = await self.select_tool(task)
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
            # Añadir contexto de memoria si existe
            if memory_context:
                prompt += f"\n\nRelevant past experience (from long-term memory):\n{memory_context}"

            tool_params_raw = await asyncio.to_thread(self.llm_provider.generate, prompt)

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
            # Añadir memoria como experiencia pasada al generar la respuesta final
            if memory_context:
                prompt += f"\n\nRelevant past experience (from long-term memory):\n{memory_context}"

            final_raw = await asyncio.to_thread(self.llm_provider.generate, prompt)
            emit(f"[{self.name}] Tarea completada.")

            result = final_raw.strip() if isinstance(final_raw, str) else tool_response
            self.scratchpad.append({"result": result})
            return result
        else:
            emit(f"[{self.name}] No se requieren herramientas. Generando resolución directa...")
            if context:
                task = f"{task}\nContext: {context}"
            prompt = (
                f"You are an autonomous agent. Your role is: {self.role}.\n"
                f"You have been assigned the following task: '{task}'.\n"
                "There are no tools available. Please use your expertise and role to complete the task as best as possible."
            )
            # Añadir memoria como experiencia pasada
            if memory_context:
                prompt += f"\n\nRelevant past experience (from long-term memory):\n{memory_context}"

            result = await asyncio.to_thread(self.llm_provider.generate, prompt)
            result = result if isinstance(result, str) else str(result)
            self.scratchpad.append({"result": result})
            return result