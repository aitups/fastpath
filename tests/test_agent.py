import pytest
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
    
    selected_tool = agent.select_tool("What is 2+2?")
    
    mock_llm_provider.generate.assert_called_once()
    prompt = mock_llm_provider.generate.call_args[0][0]
    assert "What is 2+2?" in prompt
    assert "1. search: Search the web" in prompt
    assert "2. calculate: Calculate a math expression" in prompt
    assert selected_tool.name == "calculate"

def test_agent_act(mock_llm_provider):
    tool = DummyTool(name="reverser", description="Reverses text")
    memory = DummyMemory(config={})
    
    agent = Agent(name="TestAgent", role="Tester", tools=[tool], memory=memory, llm_provider=mock_llm_provider)
    # Mock select_tool to isolate the act method's logic
    agent.select_tool = MagicMock(return_value=tool)
    
    result = agent.act("reverse this")
    
    agent.select_tool.assert_called_once_with("reverse this")
    # DummyTool just returns the input
    assert result == "reverse this"