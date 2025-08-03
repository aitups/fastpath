import pytest
from fastpath.orchestrator import Orchestrator
from fastpath.strategies import register_strategy, ExecutionStrategy

class DummyStrategy(ExecutionStrategy):
    def run(self, orchestrator, task, context=None):
        return f"dummy:{task}"


def test_register_and_use_custom_strategy():
    register_strategy("dummy", DummyStrategy)
    orc = Orchestrator(mode="dummy")
    result = orc.run("test-task")
    assert result == "dummy:test-task"


def test_mode_alias():
    register_strategy("dummy_alias", DummyStrategy, aliases=["alias1", "alias2"])
    orc = Orchestrator(mode="alias1")
    result = orc.run("alias-task")
    assert result == "dummy:alias-task"
