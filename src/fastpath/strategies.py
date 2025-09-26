from typing import Any
import logging
import json
import re
import ast
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
            orchestrator.logger.warning(f"[{orchestrator.__class__.__name__}] Error al parsear el JSON de planificación: {e}")
            subtasks = [{"subtask": task, "agent": a.name} for a in orchestrator.agents]
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
    def _normalize_name(self, sub, index):
        # Prefer explicit id if present, otherwise subtask name, otherwise positional name
        if "id" in sub and sub["id"] is not None:
            return f"id:{sub['id']}"
        return sub.get("subtask") or f"subtask_{index}"

    def _build_graph(self, subtasks):
        # Build node keys and helper maps
        node_keys = []
        index_to_key = {}
        name_to_key = {}
        id_to_key = {}
        for i, sub in enumerate(subtasks):
            key = self._normalize_name(sub, i)
            node_keys.append(key)
            index_to_key[i] = key
            # If key is id:<val>, register id map
            if isinstance(key, str) and key.startswith("id:"):
                try:
                    id_val = int(key.split(":", 1)[1])
                    id_to_key[id_val] = key
                except Exception:
                    pass
            # also map by subtask name if present
            if "subtask" in sub and sub["subtask"]:
                name_to_key[sub["subtask"]] = key

        graph = {k: set() for k in node_keys}

        for i, sub in enumerate(subtasks):
            dst_key = node_keys[i]
            raw_deps = sub.get("depends_on", []) or []
            for dep in raw_deps:
                # Resolve dep: try id match first, then index, then name
                dep_key = None
                if isinstance(dep, int):
                    if dep in id_to_key:
                        dep_key = id_to_key[dep]
                    elif 0 <= dep < len(node_keys):
                        dep_key = index_to_key[dep]
                    else:
                        raise ValueError(f"Dependency index {dep} out of range for subtask '{dst_key}'")
                elif isinstance(dep, str) and dep.isdigit():
                    d = int(dep)
                    if d in id_to_key:
                        dep_key = id_to_key[d]
                    elif 0 <= d < len(node_keys):
                        dep_key = index_to_key[d]
                    else:
                        raise ValueError(f"Dependency index string '{dep}' out of range for subtask '{dst_key}'")
                elif isinstance(dep, str):
                    if dep in id_to_key:
                        dep_key = id_to_key[dep]
                    elif dep in name_to_key:
                        dep_key = name_to_key[dep]
                    else:
                        # last resort: if dep equals a node key directly
                        if dep in graph:
                            dep_key = dep
                        else:
                            raise ValueError(f"Dependency name '{dep}' not found for subtask '{dst_key}'")
                else:
                    raise ValueError(f"Unsupported dependency type {type(dep)} for subtask '{dst_key}'")
                if dep_key == dst_key:
                    raise ValueError(f"Subtask '{dst_key}' cannot depend on itself")
                graph[dep_key].add(dst_key)

        # compute incoming edges
        incoming = {name: 0 for name in node_keys}
        for src, dests in graph.items():
            for dst in dests:
                incoming[dst] += 1
        return node_keys, graph, incoming

    def _detect_cycle_kahn(self, names, graph, incoming):
        # Kahn's algorithm: try to produce a topo order
        queue = [n for n in names if incoming[n] == 0]
        processed = 0
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            processed += 1
            for neigh in list(graph.get(node, [])):
                incoming[neigh] -= 1
                if incoming[neigh] == 0:
                    queue.append(neigh)
        has_cycle = processed != len(names)
        return has_cycle, order

    def run(self, orchestrator, task: str, context: Any = None) -> Any:
        orchestrator.logger.info(f"[HierarchicalStrategy] Ejecutando modo jerárquico para tarea: '{task}'")
        agent_list = "\n".join([f"- {a.name}: {a.role}" for a in orchestrator.agents])
        plan_prompt = (
            f"You are a hierarchical task planner. The main objective is: '{task}'.\n"
            f"Available agents:\n{agent_list}\n"
            "Break down the objective into subtasks. For each subtask, specify dependencies (by subtask index or name) and assign the most suitable agent by name.\n"
            "To do so, you must first list the outputs of the main goal. Then break down the subtasks to achieve those outputs.\n"
            "It's importat that the plan must have only one initial task with no dependencies, and only one final task that produces the final output. Intermediate tasks must have dependencies and be depencies of other subtasks.\n"
            "Respond in JSON as a list of objects: [{id: int, subtask: str, agent: str, depends_on: List[int]}] where 'depends_on' is a list of subtask names or indices that must be completed first."
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
            # Try parsing a Python literal (e.g., str(list) from mocked LLMs)
            try:
                parsed = ast.literal_eval(json_block)
                if isinstance(parsed, list):
                    subtasks = parsed
                else:
                    raise
            except Exception:
                orchestrator.logger.warning(f"[HierarchicalStrategy] Error al parsear el JSON de planificación: {e}")
                subtasks = [{"subtask": task, "agent": a.name, "depends_on": []} for a in orchestrator.agents]

        orchestrator.logger.info(f"[HierarchicalStrategy] Planificación LLM: {subtasks}")
        # Build graph and validate dependencies
        try:
            names, graph, incoming = self._build_graph(subtasks)
        except Exception as e:
            orchestrator.logger.error(f"[HierarchicalStrategy] Invalid dependency specification: {e}")
            raise
        # Detect cycles using Kahn
        has_cycle, topo_order = self._detect_cycle_kahn(names, graph, incoming.copy())
        if has_cycle:
            msg = "[HierarchicalStrategy] Error: El grafo de dependencias contiene ciclos. No se puede ejecutar."
            orchestrator.logger.error(msg)
            raise ValueError(msg)
        orchestrator.logger.info(f"[HierarchicalStrategy] Topological order: {topo_order}")
        # Prepare mapping
        name_to_subtask = {self._normalize_name(sub, i): sub for i, sub in enumerate(subtasks)}
        results = {}
        completed = set()
        # Execute tasks in layers determined by incoming edges: tasks with incoming 0 first, etc.
        # Instead of break/continue, iterate over topo_order and execute when dependencies are met.
        for name in topo_order:
            sub = name_to_subtask[name]
            subtask = sub.get("subtask", name)
            depends_raw = sub.get("depends_on", []) or []
            # Resolve dependency names to normalized names
            depends = []
            for dep in depends_raw:
                if isinstance(dep, int):
                    depends.append(names[dep])
                elif isinstance(dep, str) and dep.isdigit():
                    depends.append(names[int(dep)])
                else:
                    depends.append(dep)
            # Ensure all declared dependencies are present
            missing = [d for d in depends if d not in names and d not in name_to_subtask]
            if missing:
                msg = f"[HierarchicalStrategy] Error: Dependencias no encontradas para '{name}': {missing}"
                orchestrator.logger.error(msg)
                raise ValueError(msg)
            # Gather context only from dependencies
            dep_outputs = [results[d]["result"] for d in depends if d in results]
            context_for_subtask = "\n".join(str(out) for out in dep_outputs) if dep_outputs else context
            agent = next((a for a in orchestrator.agents if a.name == sub["agent"]), None)
            if agent:
                orchestrator.logger.info(f"[HierarchicalStrategy] Ejecutando subtarea '{name} - {subtask}' con agente: {agent.name}")
                result = agent.act(subtask, context_for_subtask)
                results[name] = {"agent": agent.name, "subtask": subtask, "result": result}
                completed.add(name)
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
