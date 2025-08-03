from typing import Any
import logging
import json
import re
from .collaboration import Message

class ExecutionStrategy:
    def run(self, orchestrator, task: str, context: Any = None) -> Any:
        raise NotImplementedError

class SequentialStrategy(ExecutionStrategy):
    def run(self, orchestrator, task: str, context: Any = None) -> Any:
        orchestrator.logger.info(f"[SequentialStrategy] Ejecutando modo secuencial para tarea: '{task}'")
        result = context
        for agent in orchestrator.agents:
            orchestrator.logger.info(f"[SequentialStrategy] Invocando agente: {agent.name}")
            result = agent.act(task, result)
        return result

class CollaborativeStrategy(ExecutionStrategy):
    def run(self, orchestrator, task: str, context: Any = None) -> Any:
        orchestrator.logger.info(f"[CollaborativeStrategy] Ejecutando modo colaborativo para tarea: '{task}'")
        orchestrator.message_bus.post(Message(sender="orchestrator", content=task))
        for agent in orchestrator.agents:
            messages = orchestrator.message_bus.get_all()
            orchestrator.logger.info(f"[CollaborativeStrategy] Invocando agente: {agent.name}")
            response = agent.act(task, context=messages)
            orchestrator.message_bus.post(Message(sender=agent.name, content=response))
        return orchestrator.message_bus.get_all()[-1].content if orchestrator.message_bus.get_all() else None

class SmartStrategy(ExecutionStrategy):
    def run(self, orchestrator, task: str, context: Any = None) -> Any:
        orchestrator.logger.info(f"[SmartStrategy] Ejecutando modo smart para tarea: '{task}'")
        if not orchestrator.agents:
            return None
        agent_list = "\n".join([f"- {a.name}: {a.role}" for a in orchestrator.agents])
        plan_prompt = (
            f"You are a task planner. The main objective is: '{task}'.\n"
            f"Available agents:\n{agent_list}\n"
            "You will break down the objective into subtasks and assign them to the most suitable agents.\n"
            "To do so, you must first list the ouputs of the main goal. Then break down the subtasks to achieve those outputs.\n"
            "Each subtask should be specific and assigned to an agent based on their role and capabilities.\n"
            "It's important to avoid overlapping on the tasks and try be as most specific as can be. For example, if you have a writer and a researcher, break into research tasks and writing tasks, but avoiding assigning any researching task to the writer."
            "The sequential execution of the tasks must output a suitable answer for the main goal or task."
            "Respond in JSON as a list of objects: [{subtask: str, agent: str}]"
        )
        plan_response = orchestrator.llm_provider.generate(plan_prompt)
        if not isinstance(plan_response, str):
            orchestrator.logger.warning(f"[{orchestrator.__class__.__name__}] La respuesta del LLM no es un string, convirtiendo a string.")
            plan_response = str(plan_response)
        json_block = None
        match = re.search(r"```json\s*(\[.*?\])\s*```", plan_response, re.DOTALL)
        if not match:
            match = re.search(r"``\s*(\[.*?\])\s*``", plan_response, re.DOTALL)
        if not match:
            match = re.search(r"(\[\s*{.*?}\s*\])", plan_response, re.DOTALL)
        if match:
            json_block = match.group(1)
        else:
            if "```json" in plan_response:
                start_index = plan_response.index("```json") + len("```json")
                end_index = plan_response.index("```", start_index)
                json_block = plan_response[start_index:end_index].strip()
            elif "```" in plan_response:
                start_index = plan_response.index("```") + len("```")
                end_index = plan_response.index("```", start_index)
                json_block = plan_response[start_index:end_index].strip()
            else:
                json_block = plan_response
        try:
            subtasks = json.loads(json_block)
        except Exception as e:
            subtasks = [{"subtask": task, "agent": a.name} for a in orchestrator.agents]
            orchestrator.logger.warning(f"[{orchestrator.__class__.__name__}] Error al parsear el JSON de planificación: {e}")
            orchestrator.logger.warning(f"[{orchestrator.__class__.__name__}] No se pudo parsear el JSON de planificación {json_block}. Usando fallback: {subtasks}")
        orchestrator.logger.info(f"[{orchestrator.__class__.__name__}] Planificación LLM: {subtasks}")
        results = []
        for sub in subtasks:
            agent = next((a for a in orchestrator.agents if a.name == sub["agent"]), None)
            if agent:
                orchestrator.logger.info(f"[{orchestrator.__class__.__name__}] Ejecutando subtarea '{sub['subtask']}' con agente: {agent.name}")
                result = agent.act(sub["subtask"], context)
                context = result
                results.append({"agent": agent.name, "subtask": sub["subtask"], "result": result})
            else:
                fallback = (
                    f"No available agent is compatible with the subtask: '{sub['subtask']}'. "
                    f"Please review the agent roles or add an agent with the required capabilities."
                )
                orchestrator.logger.warning(f"[{orchestrator.__class__.__name__}] Fallback: {fallback}")
                results.append({"agent": sub["agent"], "subtask": sub["subtask"], "result": fallback})
        summary_prompt = (
            f"You are a project manager. Here are the results of subtasks for the main objective '{task}':\n"
            f"{json.dumps(results, ensure_ascii=False, indent=2)}\n"
            "Write a concise summary of the overall outcome."
        )
        summary = orchestrator.llm_provider.generate(summary_prompt)
        orchestrator.logger.info(f"[{orchestrator.__class__.__name__}] Resumen final generado por LLM.")
        return {"subtasks": results, "summary": summary}

class HierarchicalStrategy(ExecutionStrategy):
    def _validate_task_graph(self, subtasks):
        # Construir el grafo de dependencias
        graph = {}
        names = [sub.get("subtask") or f"subtask_{i}" for i, sub in enumerate(subtasks)]
        for i, sub in enumerate(subtasks):
            name = sub.get("subtask") or f"subtask_{i}"
            depends = sub.get("depends_on", [])
            graph[name] = set(depends)
        # Detección de ciclos usando DFS
        visited = set()
        rec_stack = set()
        def dfs(node):
            if node not in graph:
                return False
            if node in rec_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph[node]:
                if neighbor == node:
                    continue  # ignorar autodependencia
                if dfs(neighbor):
                    return True
            rec_stack.remove(node)
            return False
        for node in graph:
            if dfs(node):
                return True
        return False

    def run(self, orchestrator, task: str, context: Any = None) -> Any:
        orchestrator.logger.info(f"[HierarchicalStrategy] Ejecutando modo jerárquico para tarea: '{task}'")
        agent_list = "\n".join([f"- {a.name}: {a.role}" for a in orchestrator.agents])
        plan_prompt = (
            f"You are a hierarchical task planner. The main objective is: '{task}'.\n"
            f"Available agents:\n{agent_list}\n"
            "Break down the objective into 2-7 subtasks. For each subtask, specify dependencies (by subtask index or name) and assign the most suitable agent by name.\n"
            "Respond in JSON as a list of objects: [{subtask: str, agent: str, depends_on: List[str]}] where 'depends_on' is a list of subtask names or indices that must be completed first."
        )
        plan_response = orchestrator.llm_provider.generate(plan_prompt)
        if not isinstance(plan_response, str):
            plan_response = str(plan_response)
        match = re.search(r"```json\s*(\[.*?\])\s*```", plan_response, re.DOTALL)
        if not match:
            match = re.search(r"````\s*(\[.*?\])\s*````", plan_response, re.DOTALL)
        if not match:
            match = re.search(r"(\[\s*{.*?}\s*\])", plan_response, re.DOTALL)
        json_block = match.group(1) if match else plan_response
        try:
            subtasks = json.loads(json_block)
        except Exception as e:
            orchestrator.logger.warning(f"[HierarchicalStrategy] Error al parsear el JSON de planificación: {e}")
            subtasks = [{"subtask": task, "agent": a.name, "depends_on": []} for a in orchestrator.agents]
        orchestrator.logger.info(f"[HierarchicalStrategy] Planificación LLM: {subtasks}")
        # Validar grafo de dependencias
        if self._validate_task_graph(subtasks):
            msg = "[HierarchicalStrategy] Error: El grafo de dependencias contiene ciclos. No se puede ejecutar."
            orchestrator.logger.error(msg)
            raise ValueError(msg)
        results = {}
        completed = set()
        subtask_names = [sub.get("subtask") or f"subtask_{i}" for i, sub in enumerate(subtasks)]
        name_to_subtask = {sub.get("subtask") or f"subtask_{i}": sub for i, sub in enumerate(subtasks)}
        while len(completed) < len(subtasks):
            ready_to_run = []
            for i, name in enumerate(subtask_names):
                if name in completed:
                    continue
                depends = name_to_subtask[name].get("depends_on", [])
                if all(dep in completed or dep == name for dep in depends):
                    ready_to_run.append(name)
            if not ready_to_run:
                orchestrator.logger.error("[HierarchicalStrategy] Deadlock detected: no subtasks ready to run, but not all completed.")
                raise RuntimeError("Deadlock detected in hierarchical strategy execution.")
            for name in ready_to_run:
                sub = name_to_subtask[name]
                agent = next((a for a in orchestrator.agents if a.name == sub["agent"]), None)
                depends = sub.get("depends_on", [])
                dep_outputs = [results[dep]["result"] for dep in depends if dep in results]
                context_for_subtask = "\n".join(str(out) for out in dep_outputs) if dep_outputs else context
                if agent:
                    orchestrator.logger.info(f"[HierarchicalStrategy] Ejecutando subtarea '{name}' con agente: {agent.name}")
                    result = agent.act(name, context_for_subtask)
                    results[name] = {"agent": agent.name, "subtask": name, "result": result}
                else:
                    fallback = (
                        f"No available agent is compatible with the subtask: '{name}'. "
                        f"Please review the agent roles or add an agent with the required capabilities."
                    )
                    orchestrator.logger.warning(f"[HierarchicalStrategy] Fallback: {fallback}")
                    results[name] = {"agent": sub["agent"], "subtask": name, "result": fallback}
                completed.add(name)
        summary_prompt = (
            f"You are a project manager. Here are the results of hierarchical subtasks for the main objective '{task}':\n"
            f"{json.dumps(list(results.values()), ensure_ascii=False, indent=2)}\n"
            "Write a concise summary of the overall outcome."
        )
        summary = orchestrator.llm_provider.generate(summary_prompt)
        orchestrator.logger.info(f"[HierarchicalStrategy] Resumen final generado por LLM.")
        return {"subtasks": list(results.values()), "summary": summary}

STRATEGY_REGISTRY = {}

def register_strategy(name, cls, aliases=None):
    STRATEGY_REGISTRY[name.lower()] = cls
    if aliases:
        for alias in aliases:
            STRATEGY_REGISTRY[alias.lower()] = cls

register_strategy("sequential", SequentialStrategy)
register_strategy("collaborative", CollaborativeStrategy, aliases=["collab"])
register_strategy("smart", SmartStrategy)
register_strategy("hierarchical", HierarchicalStrategy, aliases=["hierarquical"]) # typo común

# Ejemplo de estrategia personalizada
class CustomStrategy(ExecutionStrategy):
    def run(self, orchestrator, task: str, context: Any = None) -> Any:
        orchestrator.logger.info(f"[CustomStrategy] Ejecutando estrategia personalizada para tarea: '{task}'")
        # Implementación personalizada aquí
        return f"Resultado personalizado para: {task}"

# Para registrar la estrategia personalizada:
register_strategy("custom", CustomStrategy, aliases=["personalizada"])
