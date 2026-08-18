from typing import List, Dict, Any, Optional
import asyncio
import json
import re
import logging

logger = logging.getLogger("fastpath.IntentModeler")


class IntentModeler:
    """
    Modelo avanzado para la inferencia de intención y análisis emocional del usuario (Cognitive Modeling).
    Transforma texto plano en estructuras estructuradas (JSON) que guían mejor el proceso de planificación.
    Usa la API síncrona `generate()` de LLMProvider envuelta en asyncio.to_thread().
    """
    def __init__(self, llm_provider: Any):
        self.llm_provider = llm_provider

    async def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analiza el sentimiento general del texto de entrada (positivo, negativo, neutro).
        Devuelve un dict con claves 'sentiment' (str) y 'score' (float).
        """
        logger.info("Analizando sentimiento...")
        prompt = (
            f"Realiza un análisis de sentimiento profesional y conciso en este texto: '{text}'.\n"
            "Responde EXCLUSIVAMENTE con un JSON estricto con las claves "
            "'sentiment' (uno de: positive, negative, neutral) y 'score' (un float entre -1.0 y 1.0).\n"
            "Ejemplo: {\"sentiment\": \"positive\", \"score\": 0.8}"
        )
        try:
            response = await asyncio.to_thread(self.llm_provider.generate, prompt)
            if not isinstance(response, str):
                response = str(response)
            parsed = self._safe_parse_json(response)
            if parsed is None:
                return {"sentiment": "neutral", "score": 0.5}
            sentiment = parsed.get("sentiment", "neutral")
            score = parsed.get("score", 0.5)
            # Normalizar tipos
            if sentiment not in ("positive", "negative", "neutral"):
                sentiment = "neutral"
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.5
            return {"sentiment": sentiment, "score": max(-1.0, min(1.0, score))}
        except Exception as e:
            logger.warning(f"[IntentModeler] Fallo al analizar sentimiento: {e}")
            return {"sentiment": "neutral", "score": 0.5}

    async def extract_user_intent(
        self,
        task: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Determina la intención subyacente del usuario y las dependencias no declaradas.
        Output es un objeto con 'intent_type', 'required_resources', y 'urgency'.
        """
        full_context = (
            f"Tarea principal: {task}. Historial reciente (si disponible): {history}"
            if history
            else f"Tarea principal: {task}."
        )

        prompt = f"""
        Basándote en el contexto siguiente, determina la intención subyacente del usuario.
        El output debe ser un JSON estricto que contenga las claves 'intent_type'
        (uno de: booking, investigation, management, general, analysis, creation),
        'required_resources' (listado de datos o sistemas necesarios) y 'urgency'
        (uno de: low, medium, high).

        Contexto: {full_context}

        Responde EXCLUSIVAMENTE con el JSON, sin markdown ni explicaciones.
        """
        logger.info("Extrayendo intención del usuario...")
        try:
            response = await asyncio.to_thread(self.llm_provider.generate, prompt)
            if not isinstance(response, str):
                response = str(response)
            parsed = self._safe_parse_json(response)
            if parsed is None:
                return self._default_intent()
            return {
                "intent_type": parsed.get("intent_type", "general"),
                "required_resources": parsed.get("required_resources", []),
                "urgency": parsed.get("urgency", "medium"),
            }
        except Exception as e:
            logger.warning(f"[IntentModeler] Error al extraer intención: {e}")
            return self._default_intent()

    def _safe_parse_json(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extrae y parsea un objeto JSON de la respuesta del LLM de forma robusta:
        1. Intenta json.loads directo.
        2. Busca el primer bloque {...} (llaves equilibradas) en la respuesta.
        3. Intenta ast.literal_eval como fallback.
        """
        if not response:
            return None

        # 1. Intento directo
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 2. Buscar el primer bloque de llaves equilibradas
        start = response.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(response)):
                ch = response[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = response[start:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            pass
                        break

        # 3. ast.literal_eval (para dicts tipo Python de mocks)
        try:
            import ast
            parsed = ast.literal_eval(response.strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        logger.warning("[IntentModeler] No se pudo parsear JSON de la respuesta del LLM.")
        return None

    def _default_intent(self) -> Dict[str, Any]:
        return {
            "intent_type": "general",
            "required_resources": [],
            "urgency": "medium",
        }