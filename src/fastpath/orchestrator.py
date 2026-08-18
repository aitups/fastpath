from typing import List, Any, Optional
from .agent import Agent
from .collaboration import Message, MessageBus
from .llm_provider import get_llm_provider_from_config, LLMProvider
from .strategies import ExecutionStrategy, STRATEGY_REGISTRY
from .system_context_adapter import SystemContextAdapter
from .intent_modeler import IntentModeler
from .reflector import ReflectorAgent
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


class ExecutionResult:
    """
    Resultado estructurado de una ejecución del orquestador.
    Permite al mecanismo de reflexión evaluar éxito/fracaso de forma objetiva.
    """
    def __init__(
        self,
        success: bool,
        output: Any = None,
        details: Any = None,
        reflection_rounds: int = 0,
        attempts: List[dict] = None,
        error: str = None,
    ):
        self.success = success
        self.output = output
        self.details = details
        self.reflection_rounds = reflection_rounds
        self.attempts = attempts or []
        self.error = error

    def __repr__(self):
        return (
            f"ExecutionResult(success={self.success}, "
            f"output={str(self.output)[:200] if self.output else None!r}, "
            f"reflection_rounds={self.reflection_rounds}, "
            f"error={self.error!r})"
        )


class Orchestrator:
    """
    Orquestador que gestiona la colaboración entre agentes usando estrategias desacopladas.
    Por defecto opera en modo jerárquico ('hierarchical') con mecanismo de reflexión
    y autocorrección (múltiples intentos hasta convergencia).
    """
    def __init__(
        self,
        agents: Optional[List[Agent]] = None,
        mode: str = None,
        strategy: ExecutionStrategy = None,
        llm_provider: LLMProvider = None,
        max_attempts: int = 3,
        enable_memory_reflection: bool = True,
        reflector: ReflectorAgent = None,
        embedding_function: Any = None,
        system_context_adapter: SystemContextAdapter = None,
        intent_modeler: IntentModeler = None,
    ):
        self.agents = agents or []
        if strategy is not None:
            self.strategy = strategy
        elif mode is not None:
            strategy_cls = STRATEGY_REGISTRY.get(mode.lower(), STRATEGY_REGISTRY["hierarchical"])
            self.strategy = strategy_cls()
        else:
            # Default: hierarchical (jerárquico-smart)
            self.strategy = STRATEGY_REGISTRY["hierarchical"]()
        self.message_bus = MessageBus() if self.strategy.__class__.__name__ == "CollaborativeStrategy" else None
        self.llm_provider = llm_provider or get_llm_provider_from_config()
        self.max_attempts = max_attempts
        self.enable_memory_reflection = enable_memory_reflection
        if reflector is not None:
            self.reflector = reflector
        elif enable_memory_reflection:
            self.reflector = ReflectorAgent(
                llm_provider=self.llm_provider,
                embedding_function=embedding_function,
            )
        else:
            self.reflector = None
        self.system_context_adapter = system_context_adapter
        self.intent_modeler = intent_modeler
        self.logger = logging.getLogger(f"fastpath.{self.__class__.__name__}")

    async def _enrich_context_with_system_state(self, context: dict):
        """
        (Fase 2) Usa el SystemContextAdapter para enriquecer el contexto de
        ejecución con el estado persistente de sistemas externos (calendar, crm, erp).
        El estado recuperado se inyecta como 'system_state' en el contexto.
        """
        if self.system_context_adapter is None:
            return

        try:
            # Resumen general de todos los dominios configurados
            summary = await self.system_context_adapter.retrieve_state_summary()
            if summary:
                context.setdefault("system_state", {})
                context["system_state"]["summary"] = summary

            # Estado detallado por dominio conocido
            for domain in ("calendar", "crm", "erp"):
                try:
                    state = await self.system_context_adapter.fetch_system_state(domain, "current")
                    if state:
                        context.setdefault("system_state", {})
                        context["system_state"][domain] = state
                except Exception as e:
                    self.logger.warning(
                        f"[Orchestrator] Fallo al recuperar estado de dominio '{domain}': {e}"
                    )

            if "system_state" in context:
                self.logger.info(
                    "[Orchestrator] Contexto enriquecido con estado de sistemas externos."
                )
        except Exception as e:
            self.logger.warning(f"[Orchestrator] Fallo al enriquecer contexto con estado externo: {e}")

    async def _extract_user_intent(self, task: str, context: dict):
        """
        (Fase 2) Usa el IntentModeler para inferir la intención del usuario y
        enriquecer el contexto con 'intent' (tipo, recursos requeridos, urgencia).
        """
        if self.intent_modeler is None:
            return

        try:
            history = context.get("history", [])
            intent = await self.intent_modeler.extract_user_intent(task, history=history or None)
            if intent:
                context["intent"] = intent
                self.logger.info(
                    f"[Orchestrator] Intención del usuario detectada: {intent.get('intent_type')} "
                    f"(urgencia: {intent.get('urgency')})"
                )
        except Exception as e:
            self.logger.warning(f"[Orchestrator] Fallo al extraer intención del usuario: {e}")

    async def _reflect_on_execution(self, task: str, execution_context: dict):
        """
        (REQ-6.2) Tras una ejecución exitosa, reflexiona sobre el scratchpad de cada
        agente y persiste aprendizajes en su Long-Term Memory.
        """
        if not self.enable_memory_reflection or self.reflector is None:
            return

        seen_agents = set()
        for agent in self.agents:
            if id(agent) in seen_agents:
                continue
            seen_agents.add(id(agent))

            scratchpad = getattr(agent, "scratchpad", None)
            if scratchpad:
                try:
                    result = await self.reflector.reflect(
                        agent=agent,
                        task=task,
                        scratchpad=scratchpad,
                    )
                    if result is not None and callable(execution_context.get("callback")):
                        execution_context["callback"](
                            f"[Orchestrator] Memoria actualizada para {agent.name} tras reflexión."
                        )
                except Exception as e:
                    self.logger.warning(
                        f"[Orchestrator] Fallo al reflexionar para {agent.name}: {e}"
                    )

    def _is_result_successful(self, result: Any) -> bool:
        """Evalúa si el resultado de una estrategia es exitoso (con o sin ExecutionResult)."""
        if isinstance(result, ExecutionResult):
            return result.success
        if result is None:
            return False
        if isinstance(result, dict):
            # Las estrategias jerárquica/smart devuelven {'subtasks': [...], 'summary': ...}
            subtasks = result.get("subtasks", [])
            summary = result.get("summary", "")
            # Fallo explícito de la estrategia: summary empieza por "Error:"
            if isinstance(summary, str) and summary.startswith("Error:"):
                return False
            # Se considera éxito si hay subtasks ejecutadas y un resumen no vacío
            if subtasks and summary:
                return True
            return False
        # Resultado plano (string, etc.) se considera éxito
        return bool(result)

    def _parse_reflection_plan(self, reflection_response: str) -> Optional[list]:
        """
        Extrae un plan revisado (lista de subtareas con dependencias) de la respuesta
        del LLM durante la fase de reflexión. Devuelve None si no se puede parsear.
        """
        if not isinstance(reflection_response, str):
            reflection_response = str(reflection_response)

        # Buscar el bloque de lista JSON más externo
        match = re.search(r"(\[.*\])", reflection_response, re.DOTALL)
        json_block = match.group(1) if match else reflection_response

        try:
            parsed = json.loads(json_block)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except Exception:
            pass

        # Intentar como literal de Python (p.ej. str(list) de mocks)
        try:
            import ast
            parsed = ast.literal_eval(json_block)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except Exception:
            pass

        self.logger.warning("[Orchestrator] No se pudo extraer un plan revisado de la reflexión.")
        return None

    def _update_strategy(self, new_plan: list) -> ExecutionStrategy:
        """
        Actualiza la estrategia con el plan revisado generado por la reflexión.
        Inyecta el plan fijo como configuración interna para que la estrategia
        lo use en el siguiente intento (mecanismo de autocorrección).
        """
        # Conservamos la misma clase de estrategia, pero le adjuntamos el plan fijo.
        # El objeto se reutiliza; HierarchicalStrategy lee context.get("plan") en run().
        self._fixed_plan = new_plan
        return self.strategy

    async def run(self, task: str, context: Any = None, callback: Any = None) -> ExecutionResult:
        """
        Ejecuta el plan de manera iterativa con mecanismos de reflexión y autocorrección (SOTA capability).
        Cada intento ejecuta la estrategia actual; si el resultado no es exitoso, se genera un
        plan revisado vía LLM y se reintenta con ese plan fijo, hasta converger o agotar intentos.
        """
        current_context = context if isinstance(context, dict) else {"task": task, "history": []}
        current_context.setdefault("task", task)
        current_context.setdefault("history", [])

        # (Fase 2) Enriquecer contexto con estado externo e intención del usuario
        await self._enrich_context_with_system_state(current_context)
        await self._extract_user_intent(task, current_context)

        attempts: List[dict] = []
        result_obj = None

        for attempt in range(self.max_attempts):
            print(f"--- Starting execution attempt {attempt + 1}/{self.max_attempts} ---")

            # Inyectar plan fijo si la reflexión anterior generó uno
            fixed_plan = getattr(self, "_fixed_plan", None)
            if fixed_plan is not None:
                current_context["plan"] = fixed_plan

            try:
                raw_result = await self.strategy.run(self, task, current_context, callback=callback)
            except ValueError:
                # Errores de validación estructural del plan (ciclos, dependencias
                # inexistentes) no son recuperables mediante reflexión; se propagan.
                raise
            except Exception as e:
                self.logger.error(f"[Orchestrator] Error en intento {attempt + 1}: {e}")
                raw_result = ExecutionResult(success=False, output=None, error=str(e))

            # Normalizar a ExecutionResult si la estrategia devolvió un tipo plano
            if isinstance(raw_result, ExecutionResult):
                result_obj = raw_result
            else:
                success = self._is_result_successful(raw_result)
                if isinstance(raw_result, dict):
                    result_obj = ExecutionResult(
                        success=success,
                        output=raw_result.get("summary", raw_result),
                        details=raw_result,
                    )
                else:
                    result_obj = ExecutionResult(success=success, output=raw_result, details=raw_result)

            result_obj.reflection_rounds = attempt
            attempts.append({
                "attempt": attempt + 1,
                "success": result_obj.success,
                "output": result_obj.output if isinstance(result_obj.output, str) else str(result_obj.output),
            })

            if result_obj.success:
                print("Execution completed successfully.")
                # (REQ-6.2) Reflexión post-ejecución: persistir aprendizajes en LTM
                await self._reflect_on_execution(task, {"callback": callback})
                result_obj.attempts = attempts
                return result_obj

            # (REQ-6.2) Reflexión post-ejecución incluso si falló parcialmente:
            # se guarda lo aprendido del intento fallido.
            await self._reflect_on_execution(task, {"callback": callback})

            # 1. Reflection Phase (Self-Correction Mechanism)
            if attempt == self.max_attempts - 1:
                break  # No more attempts: exit loop

            reflection_prompt = (
                f"El intento {attempt + 1} para el objetivo '{task}' ha fallado o fue subóptimo.\n"
                f"Resultado Detallado: {result_obj}\n"
                f"Contexto actual de la tarea: {current_context['history'][-3:] if current_context['history'] else 'N/A'}\n\n"
                "Como agente experto, analiza el resultado. ¿Por qué falló? ¿Qué pasos faltaron o fueron incorrectos?\n"
                "Genera un plan revisado (JSON) que incluya la corrección necesaria para intentar de nuevo.\n"
                "Responde EXCLUSIVAMENTE con el JSON: una lista de objetos "
                "[{id: int, subtask: str, agent: str, depends_on: List[int]}]\n"
                "Ajusta las tareas, sus dependencias o los agentes asignados para corregir el fallo."
            )
            try:
                import asyncio
                # Los LLMProvider reales exponen generate() (sync); se ejecuta en un thread
                # para no bloquear el event loop. Los mocks de tests pueden devolver un plan.
                reflection_response = await asyncio.to_thread(
                    self.llm_provider.generate, reflection_prompt
                )

                new_plan = self._parse_reflection_plan(reflection_response) if reflection_response else None

                if new_plan is not None:
                    current_context['history'].append({
                        "attempt": attempt + 1,
                        "critique": reflection_response,
                        "suggested_plan": new_plan,
                    })
                    self._update_strategy(new_plan)
                    print(f"Reflection generó plan revisado con {len(new_plan)} subtareas.")
                else:
                    print("Reflection failed to generate a usable plan; proceeding with fallback.")
            except Exception as e:
                print(f"Error durante la reflexión del intento {attempt + 1}: {e}")

        # Si agotamos intentos, devolvemos el último resultado con error final
        result_obj.reflection_rounds = attempts[-1]["attempt"] if attempts else 0
        result_obj.attempts = attempts
        result_obj.error = result_obj.error or "La tarea no pudo completarse después de múltiples intentos y reflexión."
        return result_obj