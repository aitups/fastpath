from fastpath.tools import Tool, register_tool, TOOLS_REGISTRY

def test_tool_run_not_implemented():
    tool = Tool(name="dummy", description="desc")
    try:
        tool.run()
    except NotImplementedError:
        assert True
    else:
        assert False, "Tool.run() debería lanzar NotImplementedError"

def test_register_tool():
    class EchoTool(Tool):
        def run(self, x):
            return x
    tool = EchoTool(name="echo", description="Echo tool")
    register_tool(tool)
    assert "echo" in TOOLS_REGISTRY
    assert TOOLS_REGISTRY["echo"].run("hola") == "hola" 