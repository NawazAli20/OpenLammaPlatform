# OpenLammaPlatform 

# Run Gemma 4 Locally with Ollama and LangGraph

This project demonstrates how to:

1. Install **Ollama**.
2. Download and run Google's **Gemma 4** model locally.
3. Connect Gemma 4 to Python through `ChatOllama`.
4. Build a tool-using agent with **LangGraph**.
5. Add short-term conversation memory with a LangGraph checkpointer.

> **Recommended model:** `gemma4:12b` is a practical starting point for a modern computer. Larger variants generally require substantially more RAM or unified memory.

---

## 1. Prerequisites

You will need:

- macOS, Windows, or Linux
- Python 3.11 or newer
- `uv` or `pip`
- Enough free disk space and memory for the selected Gemma 4 model

Verify Python:

```bash
python3 --version
```

---

## 2. Install Ollama

### macOS

Download and install Ollama from:

```text
https://ollama.com/download/mac
```

Alternatively, use the official installation script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Open the Ollama application after installation.

Verify the installation:

```bash
ollama --version
```

### Windows

Run the following command in PowerShell:

```powershell
irm https://ollama.com/install.ps1 | iex
```

You can also download the Windows installer from:

```text
https://ollama.com/download/windows
```

Verify the installation:

```powershell
ollama --version
```

### Linux

Run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama service if it is not already running:

```bash
ollama serve
```

Verify the installation:

```bash
ollama --version
```

---

## 3. Download Gemma 4

Pull the 12-billion-parameter model:

```bash
ollama pull gemma4:12b
```

List the locally installed models:

```bash
ollama list
```

Run Gemma 4 interactively:

```bash
ollama run gemma4:12b
```

Enter a prompt such as:

```text
Explain LangGraph in simple terms.
```

Exit the interactive session with:

```text
/bye
```

### Other Gemma 4 variants

Check the currently available tags at:

```text
https://ollama.com/library/gemma4/tags
```

To use another variant, replace `gemma4:12b` throughout this project with the desired tag.

---

## 4. Create the Python Project

### Using `uv` — recommended

Create and enter a project directory:

```bash
mkdir ollama-gemma4-langgraph
cd ollama-gemma4-langgraph
```

Initialize the project:

```bash
uv init
```

Create a virtual environment and install the dependencies:

```bash
uv venv
uv add langgraph langchain langchain-ollama
```

Activate the virtual environment.

#### macOS or Linux

```bash
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Using `pip`

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
# macOS or Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -U langgraph langchain langchain-ollama
```

---

## 5. Test the Basic ChatOllama Connection

Create a file named `test_model.py`:

```python
from langchain_ollama import ChatOllama


def main() -> None:
    llm = ChatOllama(
        model="gemma4:12b",
        temperature=0.2,
    )

    response = llm.invoke("What is LangGraph?")
    print(response.content)


if __name__ == "__main__":
    main()
```

Run it with `uv`:

```bash
uv run python test_model.py
```

Or run it in an activated virtual environment:

```bash
python test_model.py
```

---

## 6. Build a LangGraph Agent with a Tool

Create a file named `agent.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def get_current_date() -> str:
    """Return the current local date and time."""
    return datetime.now().astimezone().strftime(
        "%A, %B %d, %Y at %I:%M:%S %p %Z"
    )


TOOLS = [get_current_date]

llm = ChatOllama(
    model="gemma4:12b",
    temperature=0.2,
)

# Make the tools available to the model.
llm_with_tools = llm.bind_tools(TOOLS)


def call_model(state: MessagesState) -> dict[str, list[Any]]:
    """Invoke Gemma 4 with the current conversation state."""
    system_message = SystemMessage(
        content=(
            "You are a helpful AI assistant running locally through Ollama. "
            "Use an available tool when it is needed. Give clear and concise "
            "answers, and do not invent tool results."
        )
    )

    response = llm_with_tools.invoke(
        [system_message, *state["messages"]]
    )

    return {"messages": [response]}


# Define the graph.
builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "agent")

# If the model produces a tool call, route to the ToolNode.
# Otherwise, route to END.
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)

# After a tool executes, return its result to the model.
builder.add_edge("tools", "agent")

# In-memory checkpoints provide short-term memory for each thread.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


def print_latest_response(result: dict[str, Any]) -> None:
    """Print the final message returned by the graph."""
    final_message = result["messages"][-1]
    print(f"Assistant: {final_message.content}")


def main() -> None:
    # Reuse the same thread_id to preserve conversation history.
    config = {"configurable": {"thread_id": "gemma4-demo-thread"}}

    print("Gemma 4 LangGraph Agent")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break

            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            print_latest_response(result)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as error:
            print(f"Error: {error}")
            print(
                "Confirm that Ollama is running and that "
                "gemma4:12b has been downloaded.\n"
            )


if __name__ == "__main__":
    main()
```

Run the agent:

```bash
uv run python agent.py
```

Or:

```bash
python agent.py
```

Example conversation:

```text
You: My name is Nawaz.
Assistant: Nice to meet you, Nawaz.

You: What is my name?
Assistant: Your name is Nawaz.

You: What is the current date and time?
Assistant: ...
```

The first two messages demonstrate short-term memory. The last message gives the model an opportunity to call the `get_current_date` tool.

---

## 7. How the Agent Works

```text
User message
     |
     v
LangGraph MessagesState
     |
     v
Gemma 4 through ChatOllama
     |
     +---- no tool call ----> Final answer
     |
     +---- tool call -------> ToolNode
                                |
                                v
                         Tool result message
                                |
                                v
                         Gemma 4 final answer
                                |
                                v
                    InMemorySaver checkpoint
```

### Main components

| Component | Purpose |
|---|---|
| `ChatOllama` | Connects LangChain/LangGraph to the local Ollama server. |
| `MessagesState` | Stores the conversation as LangChain message objects. |
| `StateGraph` | Defines the nodes, edges, and routing logic. |
| `ToolNode` | Executes tool calls requested by the model. |
| `tools_condition` | Routes the graph based on whether the model requested a tool. |
| `InMemorySaver` | Saves state in the current Python process by `thread_id`. |

> `InMemorySaver` is temporary. Its data disappears when the Python process stops. Use a persistent checkpointer for production applications.

---

## 8. Optional: Stream Graph Updates

Replace `graph.invoke(...)` with the following pattern when you want to inspect each graph update:

```python
for chunk in graph.stream(
    {"messages": [HumanMessage(content=user_input)]},
    config=config,
    stream_mode="updates",
):
    print(chunk)
```

For token-by-token user-facing output, consult the current LangGraph streaming documentation because supported stream modes and output structures may evolve.

---

## 9. Change the Model

Define the model name in one place:

```python
MODEL_NAME = "gemma4:12b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
)
```

You can then replace the value with another locally installed model tag:

```python
MODEL_NAME = "gemma4:26b"
```

Before changing the code, pull the selected model:

```bash
ollama pull gemma4:26b
```

---

## 10. Verify the Ollama API

Ollama normally runs locally at:

```text
http://localhost:11434
```

Check the installed models through the local API:

```bash
curl http://localhost:11434/api/tags
```

A connection error usually means that the Ollama application or service is not running.

---

## 11. Troubleshooting

### `ollama: command not found`

Restart the terminal after installation and verify that Ollama is in your `PATH`:

```bash
which ollama
```

On Windows PowerShell:

```powershell
Get-Command ollama
```

### Cannot connect to Ollama

Start Ollama:

```bash
ollama serve
```

On macOS or Windows, opening the Ollama desktop application may start the background service automatically.

### Model not found

Pull the exact model tag used by the Python code:

```bash
ollama pull gemma4:12b
```

Then verify it:

```bash
ollama list
```

### Python package not found

With `uv`:

```bash
uv add langgraph langchain langchain-ollama
```

With `pip`:

```bash
python -m pip install -U langgraph langchain langchain-ollama
```

Ensure VS Code or Jupyter is using the interpreter from the same `.venv` directory.

### Tool is not called

Tool use is model-dependent. Improve the tool docstring and ask a question that clearly requires the tool. Also confirm that the selected Gemma 4 tag supports tool calling.

### The model is too slow or runs out of memory

- Close memory-intensive applications.
- Use a smaller Gemma 4 variant.
- Reduce the conversation length.
- Avoid running multiple copies of the Python agent.
- Check active models:

```bash
ollama ps
```

Stop a loaded model when necessary:

```bash
ollama stop gemma4:12b
```

---

## 12. Suggested Project Structure

```text
ollama-gemma4-langgraph/
├── .venv/
├── .gitignore
├── README.md
├── agent.py
├── pyproject.toml
├── test_model.py
└── uv.lock
```

Suggested `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
.DS_Store
```

---

## 13. Useful Commands

```bash
# Show the Ollama version
ollama --version

# Show installed models
ollama list

# Pull Gemma 4
ollama pull gemma4:12b

# Start an interactive Gemma 4 chat
ollama run gemma4:12b

# Show currently loaded models
ollama ps

# Stop the model
ollama stop gemma4:12b

# Run the Python model test
uv run python test_model.py

# Run the LangGraph agent
uv run python agent.py
```

---

## References

- Ollama: <https://ollama.com/>
- Gemma 4 on Ollama: <https://ollama.com/library/gemma4>
- ChatOllama Python integration: <https://docs.langchain.com/oss/python/integrations/chat/ollama>
- LangGraph repository: <https://github.com/langchain-ai/langgraph>
- LangGraph documentation: <https://docs.langchain.com/oss/python/langgraph/overview>

---

## License

This tutorial code is provided for educational use. Review the applicable licenses and terms for Ollama, LangChain, LangGraph, and the selected Gemma 4 model before production or commercial use.
