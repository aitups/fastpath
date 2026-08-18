from typing import Any, List, Optional, Callable
import asyncio
import logging

logger = logging.getLogger("fastpath.Reflector")


class ReflectorAgent:
    """
    (REQ-6.2) Agente "Reflector": tras completar una tarea, resume los aprendizajes
    clave del scratchpad de otro agente y los persiste en su Long-Term Memory (LTM).
    """
    def __init__(
        self,
        llm_provider: Any,
        embedding_function: Optional[Callable[[str], Optional[List[float]]]] = None,
        config_path: str = "config.yaml",
    ):
        self.llm_provider = llm_provider
        self.embedding_function = embedding_function
        self.config_path = config_path

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Genera un embedding usando la función inyectada o la utilidad global."""
        if self.embedding_function is not None:
            return self.embedding_function(text)
        from .embedding_utils import get_text_embedding
        return get_text_embedding(text, config_path=self.config_path)

    async def reflect(self, agent: Any, task: str, scratchpad: List[dict]) -> Optional[dict]:
        """
        Genera un resumen de aprendizajes del scratchpad y lo guarda en el LTM del agente.

        Args:
            agent: el agente cuyo scratchpad se reflexiona (debe tener .memory).
            task: la tarea principal que se ejecutó.
            scratchpad: el scratchpad del agente (lista de dicts con log de acciones).

        Returns:
            El resumen guardado (dict) o None si no se pudo guardar.
        """
        if scratchpad is None or not scratchpad:
            return None

        if agent is None or getattr(agent, "memory", None) is None:
            logger.debug(f"[Reflector] Agente sin memoria; no se persisten aprendizajes.")
            return None

        # 1. Generar resumen de aprendizajes con el LLM
        prompt = (
            f"You are a reflective agent. Review the execution log below for the task '{task}'\n"
            f"and extract the key learnings, decisions, and facts worth remembering for future tasks.\n"
            f"Provide a concise summary (max 200 words).\n\n"
            f"Execution log:\n{scratchpad}\n"
            f"\nSummary:"
        )
        try:
            summary = await asyncio.to_thread(self.llm_provider.generate, prompt)
            if summary is None:
                pass
            elif not isinstance(summary, str):
                summary = str(summary)
                summary = summary.strip()
            else:
                summary = summary.strip()
        except Exception as e:
            logger.warning(f"[Reflector] Fallo al generar resumen con LLM: {e}")
            summary = None

        # 2. Sobre el scratchpad: si el LLM falla o devuelve vacío, usar el resultado final
        if not summary:
            # Buscar el último 'result' en el scratchpad
            last_result = next((s.get("result", "") for s in reversed(scratchpad) if s.get("result")), "")
            summary = last_result[:500] if last_result else f"Ejecutada tarea: {task}"
            logger.info(f"[Reflector] LLM no devolvió resumen; usando resultado final del scratchpad.")

        # 3. Embedding del resumen
        embedding = self._get_embedding(summary)
        if not embedding:
            logger.warning("[Reflector] No se pudo generar embedding; memoría no persistida.")
            return None

        # 4. Persistir en el LTM del agente
        metadata = {
            "summary": summary,
            "task": task,
            "agent": agent.name,
        }
        try:
            await asyncio.to_thread(agent.memory.add, embedding, metadata)
            logger.info(f"[Reflector] Aprendizajes persistidos en LTM de '{agent.name}'.")
            return metadata
        except Exception as e:
            logger.warning(f"[Reflector] Fallo al persistir en LTM de '{agent.name}': {e}")
            return None