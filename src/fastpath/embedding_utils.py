from typing import Any, List, Optional
import logging

logger = logging.getLogger("fastpath.EmbeddingUtils")

# Cache del modelo de embeddings para evitar recargar en cada llamada
_embed_model_cache: dict = {}


def get_text_embedding(
    text: str,
    config_path: str = "config.yaml",
    embedding_function: Any = None,
) -> Optional[List[float]]:
    """
    Genera un embedding numérico para un texto.

    Prioridad:
    1. `embedding_function` explícito (inyectable, útil en tests).
    2. Modelo de embeddings configurado en `config.yaml`
       (usando el mismo provider que la KnowledgeBase).

    Devuelve None si no hay forma de generar el embedding.
    """
    if embedding_function is not None:
        try:
            return embedding_function(text)
        except Exception as e:
            logger.warning(f"[EmbeddingUtils] Fallo al usar embedding_function explícito: {e}")
            return None

    embed_model = _get_embed_model(config_path)
    if embed_model is None:
        return None

    try:
        return embed_model.get_text_embedding(text)
    except Exception as e:
        logger.warning(f"[EmbeddingUtils] Fallo al generar embedding: {e}")
        return None


def _get_embed_model(config_path: str):
    """Carga (y cachea) el modelo de embeddings desde la configuración."""
    global _embed_model_cache

    if config_path in _embed_model_cache:
        return _embed_model_cache[config_path]

    try:
        from .knowledge_base import get_embedding_from_config
        from .config import Config

        config = Config(config_path).data
        embed_model = get_embedding_from_config(config)
        _embed_model_cache[config_path] = embed_model
        return embed_model
    except Exception as e:
        logger.warning(
            f"[EmbeddingUtils] No se pudo cargar el modelo de embeddings desde "
            f"la configuración ({config_path}): {e}"
        )
        _embed_model_cache[config_path] = None
        return None