from typing import List, Any
from .agent import Agent
from .collaboration import Message, MessageBus
from .llm_provider import get_llm_provider_from_config, LLMProvider
import logging
import json
import re

# Logger formatted with timestamp and class name in blue color
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("fastpath.Orchestrator")

class Orchestrator:
    """
    Orquestador que gestiona la colaboración entre agentes.
    Modos: secuencial, colaborativo y smart.
    """
    def __init__(self, agents: List[Agent], mode: str = "sequential", llm_provider: LLMProvider = None):
        self.agents = agents
        self.mode = mode  # 'sequential', 'collaborative' o 'smart'
        self.message_bus = MessageBus() if mode == "collaborative" else None
        # LLM propio del orquestador para planificación en modo smart
        self.llm_provider = llm_provider or get_llm_provider_from_config()
        self.logger = logging.getLogger(f"fastpath.{self.__class__.__name__}")

    def run(self, task: str, context: Any = None) -> Any:
        self.logger.info(f"[{self.__class__.__name__}] Ejecutando run con modo '{self.mode}' para tarea: '{task}'")
        if self.mode == "sequential":
            return self._run_sequential(task, context)
        elif self.mode == "collaborative":
            return self._run_collaborative(task, context)
        elif self.mode == "smart":
            return self._run_smart(task, context)
        else:
            raise ValueError(f"Modo de orquestador desconocido: {self.mode}")

    def _run_sequential(self, task: str, context: Any = None) -> Any:
        self.logger.info(f"[{self.__class__.__name__}] Ejecutando modo secuencial para tarea: '{task}'")
        result = context
        for agent in self.agents:
            self.logger.info(f"[{self.__class__.__name__}] Invocando agente: {agent.name}")
            result = agent.act(task, result)
        return result

    def _run_collaborative(self, task: str, context: Any = None) -> Any:
        self.logger.info(f"[{self.__class__.__name__}] Ejecutando modo colaborativo para tarea: '{task}'")
        self.message_bus.post(Message(sender="orchestrator", content=task))
        for agent in self.agents:
            messages = self.message_bus.get_all()
            self.logger.info(f"[{self.__class__.__name__}] Invocando agente: {agent.name}")
            response = agent.act(task, context=messages)
            self.message_bus.post(Message(sender=agent.name, content=response))
        return self.message_bus.get_all()[-1].content if self.message_bus.get_all() else None

    def _run_smart(self, task: str, context: Any = None) -> Any:
        self.logger.info(f"[{self.__class__.__name__}] Ejecutando modo smart para tarea: '{task}'")
        if not self.agents:
            return None
        agent_list = "\n".join([f"- {a.name}: {a.role}" for a in self.agents])
        plan_prompt = (
            f"You are a task planner. The main objective is: '{task}'.\n"
            f"Available agents:\n{agent_list}\n"
            "Break down the objective into 2-5 detailed subtasks. For each subtask, assign the most suitable agent by name.\n"
            "It's important to avoid overlapping on the tasks and try be as most specific as can be. For example, if you have a writer and a researcher, break into research tasks and writing tasks, but avoiding assigning any researching task to the writer."
            "The sequential execution of the tasks must output a suitable answer for the main goal or task."
            "Respond in JSON as a list of objects: [{subtask: str, agent: str}]"
        )
        plan_response = self.llm_provider.generate(plan_prompt)
        # Asegurar que el un string
        if not isinstance(plan_response, str):
            self.logger.warning(f"[{self.__class__.__name__}] La respuesta del LLM no es un string, convirtiendo a string.")
            plan_response = str(plan_response)
        # Extraer el primer bloque JSON válido de la respuesta
        json_block = None
        # Buscar bloque entre ```json ... ``` o ``` ... ```
        # Multilinea
        match = re.search(r"```json\s*(\[.*?\])\s*```", plan_response, re.DOTALL)
        if not match:
            match = re.search(r"```\s*(\[.*?\])\s*```", plan_response, re.DOTALL)
        if not match:
            # Buscar el primer array JSON en el texto
            match = re.search(r"(\[\s*{.*?}\s*\])", plan_response, re.DOTALL)
        if match:
            json_block = match.group(1)
        else:
            # Probar cpn index
            if "```json" in plan_response:
                start_index = plan_response.index("```json") + len("```json")
                end_index = plan_response.index("```", start_index)
                json_block = plan_response[start_index:end_index].strip()
            elif "```" in plan_response:
                start_index = plan_response.index("```") + len("```")
                end_index = plan_response.index("```", start_index)
                json_block = plan_response[start_index:end_index].strip()
            else:
                # Si no se encuentra, usar la respuesta completa (puede fallar, pero fallback abajo)
                json_block = plan_response
        try:
            subtasks = json.loads(json_block)
        except Exception as e:
            subtasks = [{"subtask": task, "agent": a.name} for a in self.agents]
            self.logger.warning(f"[{self.__class__.__name__}] Error al parsear el JSON de planificación: {e}")
            self.logger.warning(f"[{self.__class__.__name__}] No se pudo parsear el JSON de planificación {json_block}. Usando fallback: {subtasks}")
        self.logger.info(f"[{self.__class__.__name__}] Planificación LLM: {subtasks}")
        results = []
        for sub in subtasks:
            agent = next((a for a in self.agents if a.name == sub["agent"]), None)
            if agent:
                self.logger.info(f"[{self.__class__.__name__}] Ejecutando subtarea '{sub['subtask']}' con agente: {agent.name}")
                result = agent.act(sub["subtask"], context)
                context = result  # Actualizar el contexto con el resultado de la subtarea
                results.append({"agent": agent.name, "subtask": sub["subtask"], "result": result})
            else:
                fallback = (
                    f"No available agent is compatible with the subtask: '{sub['subtask']}'. "
                    f"Please review the agent roles or add an agent with the required capabilities."
                )
                self.logger.warning(f"[{self.__class__.__name__}] Fallback: {fallback}")
                results.append({"agent": sub["agent"], "subtask": sub["subtask"], "result": fallback})
        summary_prompt = (
            f"You are a project manager. Here are the results of subtasks for the main objective '{task}':\n"
            f"{json.dumps(results, ensure_ascii=False, indent=2)}\n"
            "Write a concise summary of the overall outcome."
        )
        summary = self.llm_provider.generate(summary_prompt)
        self.logger.info(f"[{self.__class__.__name__}] Resumen final generado por LLM.")
        return {"subtasks": results, "summary": summary}