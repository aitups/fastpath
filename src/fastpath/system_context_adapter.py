from typing import Any, Dict, Optional
import logging
# Import external API wrappers if needed (e.g., from .api_wrappers import CRMClient)

logger = logging.getLogger("fastpath.SystemContextAdapter")

class SystemContextAdapter:
    """
    Adapta la gestión de estado de múltiples dominios externos (Cross-Domain State).
    Permite que el Orchestrator acceda a información persistentemente guardada 
    en sistemas fuera del contexto actual de la conversación o sesión.
    """
    def __init__(self, api_clients: Optional[Dict[str, Any]] = None):
        # Inicializa clientes para diversos sistemas (CRM, Calendar, ERP, etc.)
        self.api_clients = api_clients or {}
        logger.info("SystemContextAdapter inicializado.")

    async def fetch_system_state(self, domain: str, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Busca y recupera el estado actual de un recurso en un dominio externo específico.
        Ejemplo: buscar cita pendiente del usuario en un Calendario (domain='calendar', id='user@email.com').
        """
        if domain not in ["calendar", "crm", "erp"]:
            logger.warning(f"Dominio {domain} no soportado para consulta de estado.")
            return None

        client = self.api_clients.get(domain)
        if client:
            try:
                # Esta es una simulación del llamado real a la API externa
                state_data = await getattr(client, f'fetch_{domain}_state')(identifier)
                logger.info(f"Estado recuperado exitosamente para {domain}/{identifier}.")
                return state_data
            except Exception as e:
                logger.error(f"Error al consultar estado de dominio {domain}: {e}")
                return None
        else:
            logger.warning(f"Cliente API no configurado para el dominio {domain}.")
            return None

    async def update_persistent_state(self, domain: str, data: Dict[str, Any], identifier: str) -> bool:
        """
        Actualiza de forma persistente un dato en un sistema externo. 
        Esto se usa cuando una acción del agente cambia el estado del mundo real.
        """
        client = self.api_clients.get(domain)
        if client:
            try:
                await getattr(client, f'update_{domain}_state')(data, identifier)
                logger.info(f"Estado persistente actualizado en {domain} para el identificador {identifier}.")
                return True
            except Exception as e:
                logger.error(f"Fallo al actualizar estado de dominio {domain}: {e}")
                return False
        return False

    async def retrieve_state_summary(self) -> Dict[str, Any]:
        """Genera un resumen conciso del estado general basado en los dominios configurados."""
        summary = {}
        for domain in self.api_clients:
            # Lógica para consolidar información relevante de múltiples APIs
            summary[domain] = "Estado disponible." 
        return summary