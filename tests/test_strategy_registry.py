import pytest
import asyncio
from fastpath.orchestrator import Orchestrator
from fastpath.strategies import register_strategy, ExecutionStrategy

class DummyStrategy(ExecutionStrategy):
    async def run(self, orchestrator, task, context=None, callback=None):
        return f"dummy:{task}"

def test_register_and_use_custom_strategy():
    register_strategy("dummy", DummyStrategy)
    orc = Orchestrator(mode="dummy", llm_provider=None)
    result = asyncio.run(orc.run("test-task"))
    assert result.success is True
    assert result.output == "dummy:test-task"

def test_mode_alias():
    register_strategy("dummy_alias", DummyStrategy, aliases=["alias1", "alias2"])
    orc = Orchestrator(mode="alias1", llm_provider=None)
    result = asyncio.run(orc.run("alias-task"))
    assert result.success is True
    assert result.output == "dummy:alias-task"