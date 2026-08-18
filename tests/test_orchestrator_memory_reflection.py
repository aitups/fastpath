import pytest
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock
from fastpath.orchestrator import Orchestrator
from fastpath.strategies import HierarchicalStrategy
from fastpath.agent import Agent
from fastpath.tools import Tool
from fastpath.memory import LongTermMemory
from fastpath.llm_provider import LLMProvider


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


def test_orchestrator_calls_reflector_after_success():
    """
    (REQ-6.2) Tras una ejecución jerárquica exitosa, el orquestador debe invocar
    al Reflector agent para persistir aprendizajes en el LTM de cada agente.
    """
    plan = [
        {"subtask": "t1", "agent": "A", "depends_on": []},
        {"subtask": "t2", "agent": "B", "depends_on": ["t1"]},
    ]

    llm = MagicMock(spec=LLMProvider)
    llm.generate.side_effect = [str(plan), "resumen final"]

    # Reflector mock para aislar la lógica del orquestador
    reflector = MagicMock()
    reflector.reflect = AsyncMock(return_value={"summary": "aprendizaje", "task": "x", "agent": "A"})

    agent_a = SimpleAgent(name="A", role="rA", tools=[DummyTool()], memory=DummyMemory(config={}), llm_provider=llm)
    agent_b = SimpleAgent(name="B", role="rB", tools=[DummyTool()], memory=DummyMemory(config={}), llm_provider=llm)

    orchestrator = Orchestrator(
        agents=[agent_a, agent_b],
        strategy=HierarchicalStrategy(),
        llm_provider=llm,
        reflector=reflector,
    )

    result = asyncio.run(orchestrator.run("tarea principal"))

    assert result.success is True
    # El reflector se invocó una vez por cada agente (A y B)
    assert reflector.reflect.await_count == 2
    # Se pasó la tarea principal y el scratchpad no vacío de cada agente
    for call in reflector.reflect.await_args_list:
        kwargs = call.kwargs
        assert kwargs["task"] == "tarea principal"
        assert len(kwargs["scratchpad"]) >= 1
        assert kwargs["agent"].name in ("A", "B")


def test_orchestrator_skips_reflection_when_disabled():
    """Con enable_memory_reflection=False, el orquestador no invoca al reflector."""
    plan = [
        {"subtask": "t1", "agent": "A", "depends_on": []},
    ]

    llm = MagicMock(spec=LLMProvider)
    llm.generate.side_effect = [str(plan), "resumen final"]

    reflector = MagicMock()
    reflector.reflect = AsyncMock(return_value={"summary": "aprendizaje", "task": "x", "agent": "A"})

    agent_a = SimpleAgent(name="A", role="rA", tools=[DummyTool()], memory=DummyMemory(config={}), llm_provider=llm)

    orchestrator = Orchestrator(
        agents=[agent_a],
        strategy=HierarchicalStrategy(),
        llm_provider=llm,
        reflector=reflector,
        enable_memory_reflection=False,
    )

    result = asyncio.run(orchestrator.run("tarea"))

    assert result.success is True
    reflector.reflect.assert_not_called()