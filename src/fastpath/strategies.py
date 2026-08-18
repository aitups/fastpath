from typing import Any
import logging
import json
import re
import asyncio
from .collaboration import Message

class ExecutionStrategy:
    async def run(self, orchestrator, task: str, context: Any = None, callback: Any = None) -> Any:
        raise NotImplementedError

class SequentialStrategy(ExecutionStrategy):
    async def run(self, orchestrator, task: str, context: Any = None, callback: Any = None) -> Any:
        orchestrator.logger.info(f"[SequentialStrategy] Ejecutando modo secuencial para tarea: '{task}'")
        result = context
        for agent in orchestrator.agents:
            orchestrator.logger.info(f"[SequentialStrategy] Invocando agente: {agent.name}")
            result = await agent.act(task, result, callback=callback)
        return result

class CollaborativeStrategy(ExecutionStrategy):
    async def run(self, orchestrator, task: str, context: Any = None, callback: Any = None) -> Any:
        orchestrator.logger.info(f"[CollaborativeStrategy] Ejecutando modo colaborativo para tarea: '{task}'")
        orchestrator.message_bus.post(Message(sender="orchestrator", content=task))
        for agent in orchestrator.agents:
            messages = orchestrator.message_bus.get_all()
            orchestrator.logger.info(f"[CollaborativeStrategy] Invocando agente: {agent.name}")
            response = await agent.act(task, context=messages, callback=callback)
            orchestrator.message_bus.post(Message(sender=agent.name, content=response))
        return orchestrator.message_bus.get_all()[-1].content if orchestrator.message_bus.get_all() else None

class SmartStrategy(ExecutionStrategy):
    async def run(self, orchestrator, task: str, context: Any = None, callback: Any = None) -> Any:
        orchestrator.logger.info(f"[SmartStrategy] Ejecutando modo smart para tarea: '{task}'")
        if callback: callback(f"[Orchestrator] Generando plan inteligente para: {task}")
        if not orchestrator.agents:
            return None
        agent_list = "\n".join([f"- {a.name}: {a.role}" for a in orchestrator.agents])
        plan_prompt = (
            f"You are a task planner. The main objective is: '{task}'.\n"
            f"Available agents:\n{agent_list}\n"
            "You will break down the objective into subtasks and assign them to the most suitable agents.\n"
            "To do so, you must first list the ouputs of the main goal. Then break down the subtasks to achieve those outputs.\n"
            "Each subtask should be specific and assigned to an agent based on their role and capabilities.\n"
            "Respond in JSON as a list of objects: [{subtask: str, agent: str}]"
        )
        plan_response = await asyncio.to_thread(orchestrator.llm_provider.generate, plan_prompt)
        if not isinstance(plan_response, str):
            plan_response = str(plan_response)

        # Robust JSON/list extraction: find the outermost brackets
        match = re.search(r"(\[.*\])", plan_response, re.DOTALL)
        json_block = match.group(1) if match else plan_response

        try:
            subtasks = json.loads(json_block)
        except Exception as e:
            # Try parsing a Python literal (e.g., str(list) from mocked LLMs)
            try:
                import ast
                parsed = ast.literal_eval(json_block)
                if isinstance(parsed, list):
                    subtasks = parsed
                else:
                    raise
            except Exception:
                orchestrator.logger.warning(f"[{orchestrator.__class__.__name__}] Error al parsear el JSON de planificación: {e}")
                subtasks = [{"subtask": task, "agent": a.name} for a in orchestrator.agents]
                orchestrator.logger.warning(f"[{orchestrator.__class__.__name__}] fallback: {subtasks}")
        orchestrator.logger.info(f"[{orchestrator.__class__.__name__}] Planificación LLM: {subtasks}")
        if callback:
            plan_str = "\n".join([f"  - {s['agent']}: {s['subtask']}" for s in subtasks])
            callback(f"[Orchestrator] Plan de ejecución generado:\n{plan_str}")

        results = []
        try:
            for sub in subtasks:
                agent_name = sub.get("agent", "")
                # Flexible lookup: case-insensitive and partial
                agent = next((a for a in orchestrator.agents if a.name.lower() in agent_name.lower() or agent_name.lower() in a.name.lower()), None)

                if agent:
                    msg = f"[Orchestrator] Asignando subtarea '{sub['subtask']}' a {agent.name}..."
                    orchestrator.logger.info(f"[{orchestrator.__class__.__name__}] {msg}")
                    if callback: callback(msg)
                    result = await agent.act(sub["subtask"], context, callback=callback)
                    context = result
                    results.append({"agent": agent.name, "subtask": sub["subtask"], "result": result})
                else:
                    msg = f"No se encontró un agente compatible con '{agent_name}'. Usando primero disponible."
                    if callback: callback(f"[Warn] {msg}")
                    agent = orchestrator.agents[0]
                    result = await agent.act(sub["subtask"], context, callback=callback)
                    results.append({"agent": agent.name, "subtask": sub["subtask"], "result": result})

            if callback: callback("[Orchestrator] Iniciando síntesis final de resultados...")
            summary_prompt = (
                f"You are a project manager. Here are the results of subtasks for the main objective '{task}':\n"
                f"{json.dumps(results, ensure_ascii=False, indent=2)}\n"
                f"Produce the final response for the main objective: '{task}'."
            )
            summary = await asyncio.to_thread(orchestrator.llm_provider.generate, summary_prompt)
            orchestrator.logger.info(f"[{orchestrator.__class__.__name__}] Resumen final generado por LLM.")
            return {"subtasks": results, "summary": summary}
        except Exception as e:
            msg = f"Error crítico en ejecución Smart: {str(e)}"
            orchestrator.logger.error(f"[{orchestrator.__class__.__name__}] {msg}")
            if callback: callback(f"[Error] {msg}")
            return {"subtasks": results, "summary": f"Error: {e}"}

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

    async def _generate_plan(self, orchestrator, task: str, context: Any = None):
        """Generate a hierarchical plan using the orchestrator's LLM."""
        agent_list = "\n".join([f"- {a.name}: {a.role}" for a in orchestrator.agents])
        plan_prompt = (
            f"You are a hierarchical task planner. The main objective is: '{task}'.\n"
            f"Available agents:\n{agent_list}\n"
            "Break down the objective into subtasks. For each subtask, specify dependencies (by subtask index or name) and assign the most suitable agent by name.\n"
            "To do so, you must first list the outputs of the main goal. Then break down the subtasks to achieve those outputs.\n"
            "It's importat that the plan must have only one initial task with no dependencies, and only one final task that produces the final output. Intermediate tasks must have dependencies and be depencies of other subtasks.\n"
            "Respond in JSON as a list of objects: [{id: int, subtask: str, agent: str, depends_on: List[int]}] where 'depends_on' is a list of subtask names or indices that must be completed first."
        )
        plan_response = await asyncio.to_thread(orchestrator.llm_provider.generate, plan_prompt)
        if not isinstance(plan_response, str):
            plan_response = str(plan_response)
        # Robust JSON/list extraction: find the outermost brackets
        match = re.search(r"(\[.*\])", plan_response, re.DOTALL)
        json_block = match.group(1) if match else plan_response

        try:
            subtasks = json.loads(json_block)
        except Exception as e:
            # Try parsing a Python literal (e.g., str(list) from mocked LLMs)
            try:
                import ast
                parsed = ast.literal_eval(json_block)
                if isinstance(parsed, list):
                    subtasks = parsed
                else:
                    raise
            except Exception:
                orchestrator.logger.warning(f"[HierarchicalStrategy] Error al parsear el JSON de planificación: {e}")
                subtasks = [{"subtask": task, "agent": a.name, "depends_on": []} for a in orchestrator.agents]

        orchestrator.logger.info(f"[HierarchicalStrategy] Planificación LLM: {subtasks}")
        return subtasks

    def _normalize_plan(self, subtasks, orchestrator):
        """Ensure every item is a dict with subtask, agent and depends_on."""
        normalized = []
        for i, s in enumerate(subtasks):
            if isinstance(s, dict):
                normalized.append(s)
            elif isinstance(s, str):
                normalized.append({"subtask": s, "agent": orchestrator.agents[i % len(orchestrator.agents)].name, "depends_on": []})
            else:
                normalized.append({"subtask": str(s), "agent": orchestrator.agents[0].name, "depends_on": []})
        return normalized

    def _find_agent(self, orchestrator, agent_name):
        if not agent_name:
            return orchestrator.agents[0] if orchestrator.agents else None
        return next(
            (a for a in orchestrator.agents if a.name.lower() in agent_name.lower() or agent_name.lower() in a.name.lower()),
            None
        ) or (orchestrator.agents[0] if orchestrator.agents else None)

    async def run(self, orchestrator, task: str, context: Any = None, callback: Any = None) -> Any:
        orchestrator.logger.info(f"[HierarchicalStrategy] Ejecutando modo jerárquico para tarea: '{task}'")

        # Support fixed plan injection via context (used by the orchestrator's reflection mechanism)
        fixed_plan = None
        if isinstance(context, dict):
            fixed_plan = context.get("plan")

        if fixed_plan is not None:
            subtasks = fixed_plan
            orchestrator.logger.info("[HierarchicalStrategy] Usando plan fijo inyectado desde contexto (reflexión).")
        else:
            subtasks = await self._generate_plan(orchestrator, task, context)

        subtasks = self._normalize_plan(subtasks, orchestrator)

        if callback:
            plan_str = "\n".join([f"  - {s.get('agent', '?')}: {s.get('subtask', '?')}" for s in subtasks])
            callback(f"[Orchestrator] Plan jerárquico generado:\n{plan_str}")

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

        name_to_subtask = {self._normalize_name(sub, i): sub for i, sub in enumerate(subtasks)}
        results = {}

        # Build BFS levels for parallel execution: all independent subtasks in a level run concurrently
        pending = {n: incoming[n] for n in names}
        remaining = set(names)
        levels = []
        while remaining:
            level = [n for n in remaining if pending[n] == 0]
            if not level:
                # Should not happen after cycle detection, but guard anyway
                raise ValueError("[HierarchicalStrategy] No executable subtasks found (cycle or invalid graph).")
            levels.append(level)
            for n in level:
                remaining.discard(n)
                for dst in graph[n]:
                    pending[dst] -= 1

        orchestrator.logger.info(f"[HierarchicalStrategy] Niveles de ejecución paralela: {levels}")

        async def _run_subtask(name):
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
                msg = f"Dependencias no encontradas para '{name}': {missing}"
                if callback: callback(f"[Error] {msg}")
                raise ValueError(msg)

            # Gather context only from dependencies
            dep_outputs = [results[d]["result"] for d in depends if d in results]
            context_for_subtask = "\n".join(str(out) for out in dep_outputs) if dep_outputs else context

            agent = self._find_agent(orchestrator, sub.get("agent", ""))
            msg = f"[Orchestrator] Iniciando subtarea '{name}' con {agent.name}..."
            orchestrator.logger.info(f"[HierarchicalStrategy] {msg}")
            if callback: callback(msg)

            result = await agent.act(subtask, context_for_subtask, callback=callback)
            results[name] = {"agent": agent.name, "subtask": subtask, "result": result}
            return name

        try:
            for level in levels:
                await asyncio.gather(*[_run_subtask(n) for n in level])

            if callback: callback("[Orchestrator] Consolidando resultados jerárquicos...")
            summary_prompt = (
                f"You are a project manager. Here are the results of hierarchical subtasks for the main objective '{task}':\n"
                f"{json.dumps(list(results.values()), ensure_ascii=False, indent=2)}\n"
                f"Produce the final response for the main objective: '{task}'."
            )
            summary = await asyncio.to_thread(orchestrator.llm_provider.generate, summary_prompt)
            orchestrator.logger.info(f"[HierarchicalStrategy] Resumen final generado por LLM.")
            return {"subtasks": list(results.values()), "summary": summary}
        except Exception as e:
            msg = f"Error crítico en ejecución Hierática: {str(e)}"
            orchestrator.logger.error(f"[HierarchicalStrategy] {msg}")
            if callback: callback(f"[Error] {msg}")
            return {"subtasks": list(results.values()), "summary": f"Error: {e}"}

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
    async def run(self, orchestrator, task: str, context: Any = None) -> Any:
        orchestrator.logger.info(f"[CustomStrategy] Ejecutando estrategia personalizada para tarea: '{task}'")
        # Implementación personalizada aquí
        return f"Resultado personalizado para: {task}"

# Para registrar la estrategia personalizada:
register_strategy("custom", CustomStrategy, aliases=["personalizada"])