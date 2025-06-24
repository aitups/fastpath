## Product Requirements Document (PRD): The "FastPath" Agentic Framework

**Version:** 1.1 (Revised for Advanced Features)
**Date:** October 26, 2023
**Status:** Draft
**Author:** Product Management

---

### **Executive Summary of Changes (v1.0 -> v1.1)**

This revision significantly expands the scope and ambition of the FastPath framework. Based on feedback and forward-looking requirements, this version moves three key features from "Out of Scope" into the core product definition for the initial release:
1.  **Local LLM Support:** The framework will natively support popular local model servers like Ollama, llama.cpp, and vLLM, enabling privacy-focused, offline, and cost-effective deployments.
2.  **Persistent Long-Term Memory:** Agents will now have the ability to retain and recall information across separate tasks and sessions, moving beyond ephemeral "scratchpad" memory.
3.  **Real-Time Collaboration:** The framework will support a dynamic, real-time collaboration model in addition to the sequential orchestrator, allowing agents to work concurrently in a shared environment.

---

### 1. Introduction & Vision

**Product Name:** FastPath Agentic Framework

**Vision:** To empower developers and organizations to build, deploy, and manage teams of small, specialized, autonomous AI agents that can collaborate to solve complex knowledge-based tasks. FastPath will bridge the gap between disparate data sources and actionable intelligence, powered by a robust, easy-to-use framework that **runs anywhere, from cloud APIs to local machines**, and endows agents with **persistent memory** and **dynamic collaboration** capabilities.

**Elevator Pitch:** FastPath is an open-source framework for creating "agent swarms." Instead of a single AI, you create a team of specialized agents with persistent memory. They can collaborate in real-time, ingest data from anywhere (web, SharePoint, local files), build shared knowledge bases (RAG), and leverage any LLM—from GPT-4 to a local Llama 3 model on your laptop. FastPath automates complex research, analysis, and reporting with unprecedented flexibility and power.

---

### 2. The Problem

(This section remains largely the same, but the problems are now solved more comprehensively)

Knowledge workers spend an inordinate amount of time on low-value, repetitive tasks due to:
1.  **Data Silos:** Critical information is scattered across local directories, internal SharePoint sites, and the public web.
2.  **Cognitive Overhead & Lack of Specialization:** Monolithic agents struggle with complex tasks, losing context and failing to break down problems effectively.
3.  **High Barrier to Entry:** Building robust multi-agent systems with RAG, memory, and flexible LLM backends requires massive bespoke engineering effort.
4.  **Vendor Lock-in and Privacy Concerns:** Reliance on proprietary, cloud-based LLMs creates dependencies, raises costs, and is a non-starter for organizations with sensitive data.

---

### 3. Target Audience & Personas

(Personas remain the same, but their goals are now more ambitious)

1.  **The AI/ML Engineer (Developer):** Now aims to build solutions that can run entirely on-premise for security or cost reasons, and wants to create agents that learn and improve over time.
2.  **The Business Analyst / Knowledge Worker (End-User):** Expects the system to remember previous requests and context, providing a more personalized and intelligent experience.
3.  **The IT Administrator:** Now requires tools to manage local model deployments and the persistent storage associated with agent memories, in addition to cloud credentials.

---

### 4. Core Concepts & Terminology

*   **FastPath Agent:** An autonomous agent with a **Role**, **Tools**, and now, its own persistent **Long-Term Memory Store**.
*   **Tool:** A specific function an agent can execute.
*   **Knowledge Base (KB):** A *shared*, task-specific RAG data store.
*   **Long-Term Memory (LTM):** An *agent-specific*, persistent vector store for retaining key facts, conversations, and learnings across multiple tasks.
*   **Task:** A high-level objective defined by a user.
*   **The Orchestrator:** The "manager" of the framework. It now supports two modes:
    *   **Sequential Mode:** A top-down planner that assigns tasks sequentially.
    *   **Collaborative Mode:** A "moderator" that facilitates real-time, event-driven communication between agents in a shared workspace.
*   **LLM Provider:** A pluggable interface for connecting to different language models, whether via cloud API or a local server.

---

### 5. Features & Requirements

#### **M-1: The FastPath Core Framework**

*   **REQ-1.1: Agent Definition:** Users must be able to define an agent specifying its role, system prompt, tools, and configuration for its Long-Term Memory.
*   **REQ-1.2: Orchestrator Engine:** (Revised) The framework must include a flexible Orchestrator that can operate in different modes (configurable per task).
*   **REQ-1.3: Tool Abstraction:** Provide a simple interface for developers to create new, custom tools.
*   **REQ-1.4: Pluggable LLM Backend (New):**
    *   The framework must abstract the LLM interface.
    *   Users must be able to configure the LLM provider in a central configuration file.
    *   **Must provide out-of-the-box support for:**
        *   OpenAI API-compatible endpoints (including vLLM).
        *   Ollama servers.
        *   Direct loading of GGUF models via `llama-cpp-python`.

#### **M-2: Data Ingestion & RAG Tools (Knowledge Base Creation)**

*   (No changes here. Requirements REQ-2.1 through REQ-2.4 for Directory, Website, and SharePoint readers remain the same.)

#### **M-3: Advanced Agent Collaboration & Cognition** (Revised Section)

*   **REQ-3.1: Knowledge Base Query Tool:** All agents must have a tool to query the shared, task-specific KBs.
*   **REQ-3.2: Dynamic Collaboration Models:** (Revised) The framework must support two primary collaboration patterns:
    *   **Orchestrator-Led (Sequential):** The Orchestrator creates a static plan and delegates tasks sequentially, passing outputs from one agent to the next.
    *   **Conversational Swarm (Real-Time):** Agents are placed in a shared "workspace" (message bus). One agent can post a message (e.g., a question, a piece of data), and other agents can react to it concurrently, with the Orchestrator acting as a moderator to guide the conversation and determine when the task is complete.
*   **REQ-3.3: Logging & Transparency:** Must produce detailed, real-time logs of agent "thoughts," tool usage, and inter-agent communication for both collaboration models.

#### **M-4: User Experience & Interface**

*   (Requirements REQ-4.1 for CLI and REQ-4.2 for a simple Web UI remain the same.)

#### **M-5: Security & Configuration**

*   **REQ-5.1: Secure Credential Management:** Use environment variables or a secure vault system for API keys.
*   **REQ-5.2: Expanded Configuration File:** (Revised) A central `config.yaml` must manage framework settings, now including LLM provider selection.
    *   **Example `config.yaml`:**
        ```yaml
        llm_provider: ollama # 'openai', 'ollama', 'vllm', 'llama_cpp'
        
        llm_config:
          # For Ollama or vLLM
          model: 'llama3:8b'
          base_url: 'http://localhost:11434' 
          
          # For OpenAI
          # model: 'gpt-4o'
          # api_key: ${OPENAI_API_KEY}

          # For llama_cpp
          # model_path: '/path/to/your/model.gguf'
          # n_gpu_layers: -1 # -1 to offload all layers to GPU
        
        embedding_model: 'all-MiniLM-L6-v2'
        vector_store_path: './fastpath_db'
        ```

#### **M-6: Persistent Long-Term Memory (New Section)**

*   **REQ-6.1: Agent-Specific Memory Store:** Each agent instance must be associated with its own persistent vector store for long-term memory.
*   **REQ-6.2: Automated Memory Ingestion:** After a task is completed, the Orchestrator will prompt a "Reflector" agent to summarize the key takeaways, decisions, and learned facts from the task's scratchpad. This summary is then embedded and stored in the LTM of the participating agents.
*   **REQ-6.3: Proactive Memory Retrieval:** At the beginning of a new task, the agent's system prompt will be augmented with relevant information retrieved from its LTM based on the task description. This gives the agent "context" from past experiences.

---

### 6. Out of Scope for v1.1

*   GUI for building/visualizing agent workflows and memory graphs.
*   Support for non-text modalities (image, audio).
*   Complex, role-based access control (RBAC) for Knowledge Bases or Long-Term Memories.
*   Automated tool creation or self-healing code.

---

### 7. Success Metrics

*   **Developer Adoption:** Number of GitHub stars/forks, pip/conda downloads.
*   **Task Completion Rate:** >90% success rate for benchmark tasks.
*   **Time to First "Hello, Local Agent!":** A developer can run a task using an Ollama-powered agent in under 15 minutes.
*   **Demonstrable Memory:** A benchmark task where an agent is asked a follow-up question in a new session and correctly uses information from its LTM to answer.
*   **Real-Time Collaboration Success:** Successful completion of a task requiring at least 3 agents to exchange information in the conversational swarm model to find a solution.
*   **Performance:** Benchmarks for RAG creation time, and task completion time on both cloud and local LLMs (e.g., GPT-4o vs. Llama 3 8B on specific hardware).

---

### 8. Assumptions & Dependencies

*   **LLM Access:** Users have access to cloud LLM APIs **or** have the necessary hardware and software (e.g., Ollama installed) to run local models.
*   **Compute Resources:** Users have sufficient local compute (CPU/RAM/VRAM) and storage for running models, embedding, and storing vector databases for KBs and LTMs.
*   **Python Environment:** The framework will be built in Python.

---

## Implementation Details

### Class Architecture

```mermaid
classDiagram
    class Agent {
        - name: str
        - role: str
        - tools: List~Tool~
        - memory: LongTermMemory
        - scratchpad: list
        + act(task: str, context: Any): Any
    }
    class Tool {
        - name: str
        - description: str
        + run(*args, **kwargs): Any
    }
    class LongTermMemory {
        - storage_path: str
        + add(embedding: List~float~, metadata: dict)
        + query(query_embedding: List~float~, top_k: int): list
    }
    class Orchestrator {
        - agents: List~Agent~
        - mode: str
        + run(task: str, context: Any): Any
    }
    class KnowledgeBase {
        - storage_path: str
        + ingest(data: Any)
        + query(query: str, top_k: int): List~Any~
    }
    class LLMProvider {
        - config: dict
        + generate(prompt: str, **kwargs): str
    }
    class MessageBus {
        - messages: List~Message~
        + post(message: Message)
        + get_all(): List~Message~
    }
    Agent --> "*" Tool
    Agent --> "1" LongTermMemory
    Orchestrator --> "*" Agent
    Agent --> "1" KnowledgeBase
    Orchestrator --> "1" MessageBus
    Agent --> "1" LLMProvider
    MessageBus --> "*" Message
```

### Sequence Diagram (Sequential Execution)

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant Agent1
    participant Agent2
    participant KB as KnowledgeBase
    participant LTM1 as LongTermMemory1
    participant LTM2 as LongTermMemory2
    participant Tool
    User->>Orchestrator: run(task)
    Orchestrator->>Agent1: act(task)
    Agent1->>LTM1: query(context)
    Agent1->>KB: query(task)
    Agent1->>Tool: run(tool)
    Agent1-->>Orchestrator: partial result
    Orchestrator->>Agent2: act(task, context)
    Agent2->>LTM2: query(context)
    Agent2->>KB: query(task)
    Agent2->>Tool: run(tool)
    Agent2-->>Orchestrator: final result
    Orchestrator-->>User: final result
```

### Instructions for Running Tests

1.  Install dependencies (if you haven't already):
    ```bash
    poetry install
    ```
2.  Run tests:
    ```bash
    poetry run pytest
    ```

### Configuring a New Agent Team

1.  Define your custom tools in their own module or in `tools.py`.
2.  Implement your agents by inheriting from `Agent` and overriding the `act` method.
3.  Instantiate the agents and the orchestrator in a script, for example:
    ```python
    from fastpath.agent import Agent
    from fastpath.memory import LongTermMemory
    from fastpath.tools import Tool
    from fastpath.orchestrator import Orchestrator

    class MyTool(Tool):
        def __init__(self):
            super().__init__(name="my_tool", description="My custom tool")
        def run(self, text_to_reverse: str):
            return text_to_reverse[::-1]

    class MyAgent(Agent):
        def act(self, task: str, context=None):
            # For simplicity, this agent just uses its first tool
            tool_result = self.tools[0].run(task)
            return f"{self.name} solved the task. Result: {tool_result}"

    # 1. Create the building blocks
    tool = MyTool()
    memory = LongTermMemory(storage_path=":memory:") # Using an in-memory placeholder

    # 2. Create the agent
    my_agent = MyAgent(name="AgentX", role="Text Reverser", tools=[tool], memory=memory)

    # 3. Create the orchestrator with the agent
    orchestrator = Orchestrator(agents=[my_agent], mode="sequential")

    # 4. Run the task
    final_result = orchestrator.run("reverse this text")
    print(final_result)
    ```

### Running the System

You can run the minimal example included in `src/main.py`:

```bash
poetry run python src/main.py
```

Or run your own script following the example above.