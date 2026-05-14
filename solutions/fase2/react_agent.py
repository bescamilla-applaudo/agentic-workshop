"""
Exercise 2.1 — ReAct Agent: Researcher with tools.

ReAct agent in LangGraph that searches for information and answers
multi-hop questions using Thought → Action → Observation.
"""

import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ─── State ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int


# ─── Tools ────────────────────────────────────────────────────────────

search_tool = DuckDuckGoSearchResults(max_results=3)
tools = [search_tool]

# ─── LLM ──────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model=os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
    max_tokens=1024,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
llm_with_tools = llm.bind_tools(tools)

# ─── Nodes ────────────────────────────────────────────────────────────

MAX_ITERATIONS = 5


def agent_node(state: AgentState) -> dict:
    """Agent node: calls the LLM with bound tools."""
    iteration = state.get("iteration", 0) + 1
    print(f"\n🧠 Agent (iteration {iteration})...")

    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "iteration": iteration}


tool_node = ToolNode(tools)

# ─── Conditional edge ─────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Decide whether to go to tools or finish."""
    last_message = state["messages"][-1]
    iteration = state.get("iteration", 0)

    # Safety: maximum iterations
    if iteration >= MAX_ITERATIONS:
        print(f"   ⚠️ Maximum iterations reached ({MAX_ITERATIONS})")
        return END

    # If there are tool calls → execute tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"   🔧 Tool calls: {[tc['name'] for tc in last_message.tool_calls]}")
        return "tools"

    # No tool calls → the agent has the answer
    print("   ✅ Final response")
    return END


# ─── Graph ────────────────────────────────────────────────────────────

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()


def run_agent(question: str) -> str:
    """Run the ReAct agent with a question."""
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")

    result = graph.invoke(
        {"messages": [("user", question)], "iteration": 0}
    )

    final = result["messages"][-1].content
    print(f"\n📝 Final response:\n{final[:500]}")
    return final


if __name__ == "__main__":
    # Multi-hop question: requires search → reason → search again
    run_agent("Who is the current CEO of Anthropic and when was the company founded?")
