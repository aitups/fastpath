from typing import List, Any
from .memory import LongTermMemory
from .tools import Tool
from .llm_provider import LLMProvider

class Agent:
    """
    Represents an autonomous agent with a role, tools, persistent memory, and its own LLM provider.
    """
    def __init__(self, name: str, role: str, tools: List[Tool], memory: LongTermMemory, llm_provider: LLMProvider):
        self.name = name
        self.role = role
        self.tools = tools
        self.memory = memory
        self.llm_provider = llm_provider
        self.scratchpad = []  # Temporary memory for the current task

    def select_tool(self, task: str) -> Tool:
        """
        Uses the agent's LLM to decide which tool to use based on the task and tool descriptions.
        """
        tool_descriptions = "\n".join([f"{i+1}. {tool.name}: {tool.description}" for i, tool in enumerate(self.tools)])
        prompt = (
            f"You are an expert agent. Given the following task: '{task}', and these tools:\n"
            f"{tool_descriptions}\n"
            "Which tool (by name) is the most appropriate to use? Respond with only the tool name."
        )
        tool_name = self.llm_provider.generate(prompt).strip()
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                return tool
        # Fallback: return the first tool
        return self.tools[0]

    def act(self, task: str, context: Any = None) -> Any:
        """
        Agent questions itself (using its LLM) about which tool to use, then uses it.
        """
        tool = self.select_tool(task)
        self.scratchpad.append({"task": task, "selected_tool": tool.name})
        # For demonstration, we just pass the task as input to the tool
        return tool.run(task) 