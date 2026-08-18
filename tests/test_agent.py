import pytest
import asyncio
from unittest.mock import MagicMock
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

@pytest.fixture
def mock_llm_provider():
    return MagicMock(spec=LLMProvider)

# Embedding stub: devuelve un vector trivial para evitar cargar el modelo real
def stub_embedding(text):
    return [0.0] * 8

def test_agent_init(mock_llm_provider):
    tool = DummyTool()
    memory = DummyMemory(config={})
    agent = Agent(name="TestAgent", role="Tester", tools=[tool], memory=memory, llm_provider=mock_llm_provider)
    assert agent.name == "TestAgent"
    assert agent.role == "Tester"
    assert agent.tools[0].name == "echo"
    assert isinstance(agent.memory, DummyMemory)
    assert agent.llm_provider is mock_llm_provider

def test_agent_select_tool(mock_llm_provider):
    tool1 = DummyTool(name="search", description="Search the web")
    tool2 = DummyTool(name="calculate", description="Calculate a math expression")
    memory = DummyMemory(config={})

    mock_llm_provider.generate.return_value = "calculate"

    agent = Agent(name="TestAgent", role="Tester", tools=[tool1, tool2], memory=memory, llm_provider=mock_llm_provider)

    selected_tool = asyncio.run(agent.select_tool("What is 2+2?"))

    mock_llm_provider.generate.assert_called_once()
    prompt = mock_llm_provider.generate.call_args[0][0]
    assert "What is 2+2?" in prompt
    assert "1. search: Search the web" in prompt
    assert "2. calculate: Calculate a math expression" in prompt
    assert selected_tool.name == "calculate"

def test_agent_select_tool_no_tools(mock_llm_provider):
    memory = DummyMemory(config={})
    agent = Agent(name="TestAgent", role="Tester", tools=[], memory=memory, llm_provider=mock_llm_provider)
    selected = asyncio.run(agent.select_tool("any task"))
    assert selected is None

def test_agent_act(mock_llm_provider):
    tool = DummyTool(name="reverser", description="Reverses text")
    memory = DummyMemory(config={})

    agent = Agent(
        name="TestAgent", role="Tester", tools=[tool], memory=memory,
        llm_provider=mock_llm_provider, embedding_function=stub_embedding
    )
    # Mock select_tool as a callable returning the tool (it is now async; provide an async stub)
    async def fake_select(task):
        return tool
    agent.select_tool = fake_select

    # Mock generate: first call → query, second call → final answer
    mock_llm_provider.generate.side_effect = ["reverse this", "reversed result"]

    result = asyncio.run(agent.act("reverse this"))

    # DummyTool just returns the input; the final LLM generates the answer
    assert result == "reversed result"
    scratch = agent.scratchpad
    assert scratch[0]["task"] == "reverse this"
    assert scratch[0]["selected_tool"] == "reverser"
    assert scratch[1]["tool_response"] == "reverse this"
    assert scratch[2]["result"] == "reversed result"

def test_agent_act_memory_context(mock_llm_provider):
    """Verifica que la memoria recuperada se inyecta en los prompts del agente."""
    tool = DummyTool(name="echo", description="Echo")
    memory = DummyMemory(config={})
    # Memoria que devuelve un item con metadata conteniendo 'summary'
    memory.query = MagicMock(return_value=[{"metadata": {"summary": "Aprendí sobre X en el pasado"}}])

    agent = Agent(
        name="TestAgent", role="Tester", tools=[tool], memory=memory,
        llm_provider=mock_llm_provider, embedding_function=stub_embedding
    )

    async def fake_select(task):
        return tool
    agent.select_tool = fake_select

    mock_llm_provider.generate.side_effect = ["my query", "final answer"]

    asyncio.run(agent.act("pregunta sobre X"))

    # El LTM debe haberse consultado con el embedding de la tarea
    memory.query.assert_called_once()
    # El contexto de memoria debe aparecer en al menos uno de los prompts
    prompts = [c[0][0] for c in mock_llm_provider.generate.call_args_list]
    assert any("Aprendí sobre X en el pasado" in p for p in prompts), (
        "El contexto de memoria no se inyectó en ningún prompt del LLM"
    )