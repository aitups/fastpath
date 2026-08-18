import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from fastpath.orchestrator import Orchestrator
from fastpath.strategies import HierarchicalStrategy
from fastpath.agent import Agent
from fastpath.tools import Tool
from fastpath.memory import LongTermMemory
from fastpath.llm_provider import LLMProvider
from fastpath.system_context_adapter import SystemContextAdapter
from fastpath.intent_modeler import IntentModeler


class DummyTool(Tool):
    def __init__(self, name="echo", description="Echo"):
        super().__init__(name=name, description=description)

    def run(self, text):
        return text


class DummyMemory(LongTermMemory):
    def add(self, embedding, metadata):
        pass

    def query(self, query_embedding, top_k=5):
        return []


def stub_embedding(text):
    return [0.0] * 8


class SimpleAgent(Agent):
    def __init__(self, name, role, *args, **kwargs):
        kwargs.setdefault("embedding_function", stub_embedding)
        super().__init__(name=name, role=role, *args, **kwargs)

    async def act(self, task, context=None, callback=None):
        self.scratchpad.append({"task": task, "result": f"{self.name}-result"})
        return f"{self.name}-result"


def test_orchestrator_enriches_context_with_system_state():
    """El SystemContextAdapter inyecta estado externo en el contexto de ejecución."""
    plan = [
        {"subtask": "t1", "agent": "A", "depends_on": []},
    ]

    llm = MagicMock(spec=LLMProvider)
    llm.generate.side_effect = [str(plan), "resumen final"]

    # SystemContextAdapter simulado
    adapter = MagicMock(spec=SystemContextAdapter)
    adapter.retrieve_state_summary = AsyncMock(return_value={"calendar": "Estado disponible."})
    adapter.fetch_system_state = AsyncMock(return_value={"next_meeting": "2026-08-19 10:00"})

    agent_a = SimpleAgent(name="A", role="rA", tools=[DummyTool()], memory=DummyMemory(config={}), llm_provider=llm)

    # Stub del planificador para capturar el contexto
    captured = {}
    strategy = MagicMock(spec=HierarchicalStrategy)

    async def fake_run(orchestrator, task, context, callback=None):
        captured.update(context)  # copia superficial para inspección
        return {"subtasks": [{"subtask": "t1", "agent": "A", "result": "ok"}], "summary": "resumen"}

    strategy.run = fake_run

    orchestrator = Orchestrator(
        agents=[agent_a],
        strategy=strategy,
        llm_provider=llm,
        system_context_adapter=adapter,
        enable_memory_reflection=False,
    )

    result = asyncio.run(orchestrator.run("tarea principal"))

    assert result.success is True
    # El contexto fue enriquecido con system_state (summary + calendar)
    assert "system_state" in captured
    assert captured["system_state"]["summary"] == {"calendar": "Estado disponible."}
    assert captured["system_state"]["calendar"] == {"next_meeting": "2026-08-19 10:00"}


def test_orchestrator_extracts_intent():
    """El IntentModeler inyecta la intención del usuario en el contexto."""
    plan = [
        {"subtask": "t1", "agent": "A", "depends_on": []},
    ]

    llm = MagicMock(spec=LLMProvider)
    llm.generate.side_effect = [str(plan), "resumen final"]

    intent_modeler = MagicMock(spec=IntentModeler)
    intent_modeler.extract_user_intent = AsyncMock(return_value={
        "intent_type": "investigation",
        "required_resources": ["web"],
        "urgency": "high",
    })

    agent_a = SimpleAgent(name="A", role="rA", tools=[DummyTool()], memory=DummyMemory(config={}), llm_provider=llm)

    captured = {}
    strategy = MagicMock(spec=HierarchicalStrategy)

    async def fake_run(orchestrator, task, context, callback=None):
        captured.update(context)
        return {"subtasks": [{"subtask": "t1", "agent": "A", "result": "ok"}], "summary": "resumen"}

    strategy.run = fake_run

    orchestrator = Orchestrator(
        agents=[agent_a],
        strategy=strategy,
        llm_provider=llm,
        intent_modeler=intent_modeler,
        enable_memory_reflection=False,
    )

    result = asyncio.run(orchestrator.run("Investiga las tendencias de IA"))

    assert result.success is True
    # El contexto fue enriquecido con la intención detectada
    assert captured["intent"] == {
        "intent_type": "investigation",
        "required_resources": ["web"],
        "urgency": "high",
    }
    intent_modeler.extract_user_intent.assert_awaited_once()


def test_orchestrator_skips_enrichment_without_adapters():
    """Sin SystemContextAdapter ni IntentModeler, el contexto no se enriquece."""
    plan = [
        {"subtask": "t1", "agent": "A", "depends_on": []},
    ]

    llm = MagicMock(spec=LLMProvider)
    llm.generate.side_effect = [str(plan), "resumen final"]

    agent_a = SimpleAgent(name="A", role="rA", tools=[DummyTool()], memory=DummyMemory(config={}), llm_provider=llm)

    captured = {}
    strategy = MagicMock(spec=HierarchicalStrategy)

    async def fake_run(orchestrator, task, context, callback=None):
        captured.update(context)
        return {"subtasks": [{"subtask": "t1", "agent": "A", "result": "ok"}], "summary": "resumen"}

    strategy.run = fake_run

    orchestrator = Orchestrator(
        agents=[agent_a],
        strategy=strategy,
        llm_provider=llm,
        enable_memory_reflection=False,
    )

    result = asyncio.run(orchestrator.run("tarea"))

    assert result.success is True
    assert "system_state" not in captured
    assert "intent" not in captured