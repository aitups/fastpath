import pytest
import asyncio
from unittest.mock import MagicMock
from fastpath.strategies import HierarchicalStrategy
from fastpath.orchestrator import Orchestrator
from fastpath.agent import Agent
from fastpath.tools import Tool
from fastpath.memory import LongTermMemory
from fastpath.llm_provider import LLMProvider

class DummyTool(Tool):
    def __init__(self, name="echo", description="Echo tool"):
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
    def __init__(self, name, role, llm_provider=None):
        super().__init__(name=name, role=role, tools=[DummyTool(name="echo")], memory=DummyMemory(config={}), llm_provider=llm_provider or MagicMock(spec=LLMProvider), embedding_function=stub_embedding)
    async def act(self, task, context=None, callback=None):
        # return a predictable string to help assert concatenation
        if context:
            return f"{self.name}-out-for-{task}-with-{context}"
        return f"{self.name}-out-for-{task}"

@pytest.fixture
def agents():
    return [SimpleAgent("A", "roleA"), SimpleAgent("B", "roleB"), SimpleAgent("C", "roleC")]

@pytest.fixture
def orchestrator(agents):
    return Orchestrator(agents=agents, strategy=HierarchicalStrategy(), llm_provider=MagicMock(spec=LLMProvider))

def test_hierarchical_basic_execution(orchestrator):
    plan = [
        {"subtask": "t1", "agent": "A", "depends_on": []},
        {"subtask": "t2", "agent": "B", "depends_on": ["t1"]},
        {"subtask": "t3", "agent": "C", "depends_on": ["t2"]}
    ]
    # Patch LLM to return plan directly as JSON
    orchestrator.llm_provider.generate = MagicMock(return_value=str(plan))

    result = asyncio.run(orchestrator.run("main-task"))
    assert result.success is True
    subtasks = result.details["subtasks"]
    assert len(subtasks) == 3
    assert subtasks[0]["subtask"] == "t1"
    assert subtasks[1]["subtask"] == "t2"
    assert subtasks[2]["subtask"] == "t3"

def test_hierarchical_concat_context(orchestrator):
    plan = [
        {"subtask": "r1", "agent": "A", "depends_on": []},
        {"subtask": "r2", "agent": "B", "depends_on": ["r1"]},
        {"subtask": "r3", "agent": "C", "depends_on": ["r1", "r2"]}
    ]
    orchestrator.llm_provider.generate = MagicMock(side_effect=[str(plan), "summary"])
    res = asyncio.run(orchestrator.run("x"))
    assert res.success is True
    # r3 should have context composed of r1 and r2 outputs
    outputs = {s["subtask"]: s["result"] for s in res.details["subtasks"]}
    # Result should be produced by agent C and include dependency outputs from A and B
    assert outputs["r3"].startswith("C-out-for-r3")
    assert "A-out-for-r1" in outputs["r3"]
    assert "B-out-for-r2" in outputs["r3"]

def test_hierarchical_dependency_by_index(orchestrator):
    plan = [
        {"subtask": "s0", "agent": "A", "depends_on": []},
        {"subtask": "s1", "agent": "B", "depends_on": [0]},
    ]
    orchestrator.llm_provider.generate = MagicMock(return_value=str(plan))
    res = asyncio.run(orchestrator.run("idx"))
    assert res.success is True
    assert len(res.details["subtasks"]) == 2

def test_hierarchical_missing_dependency(orchestrator):
    plan = [
        {"subtask": "m1", "agent": "A", "depends_on": ["does_not_exist"]}
    ]
    orchestrator.llm_provider.generate = MagicMock(return_value=str(plan))
    with pytest.raises(ValueError):
        asyncio.run(orchestrator.run("bad"))

def test_hierarchical_detect_cycle(orchestrator):
    plan = [
        {"subtask": "c1", "agent": "A", "depends_on": ["c2"]},
        {"subtask": "c2", "agent": "B", "depends_on": ["c1"]}
    ]
    orchestrator.llm_provider.generate = MagicMock(return_value=str(plan))
    with pytest.raises(ValueError):
        asyncio.run(orchestrator.run("cycle"))