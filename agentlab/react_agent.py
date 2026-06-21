"""
Exercise 2.1 — ReAct Agent: Researcher with tools.

Graph topology (2 nodes):

    ┌──────┐   tool_calls?   ┌───────┐
    │agent │ ─── yes ──────► │ tools │
    │      │ ◄───────────────│       │
    └──────┘                 └───────┘
        │
        └── no tool_calls → END

State: messages (append-only via add_messages) + iteration counter.
The iteration counter acts as a safety brake (MAX_ITERATIONS).
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults  # noqa: F401 — deprecation warning is cosmetic
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages:  Annotated[list[BaseMessage], add_messages]
    iteration: int


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

search = DuckDuckGoSearchResults(max_results=4)
tools  = [search]

# ---------------------------------------------------------------------------
# LLM  — ChatOpenAI pointed at OpenRouter
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model=os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    max_tokens=1024,
)
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 6


def agent_node(state: AgentState) -> dict:
    """Call the LLM; it may emit tool_calls or a final answer."""
    iteration = state.get("iteration", 0) + 1
    print(f"\n[agent] iteration {iteration}")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "iteration": iteration}


tool_node = ToolNode(tools)

# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    last     = state["messages"][-1]
    iteration = state.get("iteration", 0)

    if iteration >= MAX_ITERATIONS:
        print(f"  [router] max iterations ({MAX_ITERATIONS}) reached → END")
        return END

    if getattr(last, "tool_calls", None):
        names = [tc["name"] for tc in last.tool_calls]
        print(f"  [router] tool_calls={names} → tools")
        return "tools"

    print("  [router] no tool_calls → END")
    return END

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "agent")

graph = builder.compile()

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_agent(question: str) -> str:
    """Invoke the ReAct graph and return the final answer text."""
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")

    final_state = graph.invoke(
        {"messages": [("user", question)], "iteration": 0}
    )

    answer = final_state["messages"][-1].content
    print(f"\nAnswer:\n{answer}")
    return answer


# ---------------------------------------------------------------------------
# Test — multi-hop question (requires ≥2 searches to answer)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_agent(
        "What programming language was used to build the LangGraph library, "
        "and who created the company behind it?"
    )
