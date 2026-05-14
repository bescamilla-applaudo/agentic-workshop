"""
Exercise 5.1 — Human-in-the-Loop: The agent that asks permission.

LangGraph workflow with HITL: generates plan, pauses for approval,
executes only if the human approves.
"""

import os
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# ─── State ────────────────────────────────────────────────────────────

class HITLState(TypedDict):
    task: str
    plan: str
    approved: bool
    result: str


# ─── LLM ──────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model=os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
    max_tokens=1024,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# ─── Nodes ────────────────────────────────────────────────────────────

def plan_node(state: HITLState) -> dict:
    """Generate an action plan for the task."""
    print(f"\n📋 Generating plan for: {state['task']}")

    response = llm.invoke(
        f"""Generate a detailed action plan for this task:

Task: "{state['task']}"

The plan must list each step with its potential impact.
Mark high-risk steps with ⚠️.
Format: numbered list."""
    )

    plan = response.content
    print(f"   Plan generated:\n{plan[:300]}")
    return {"plan": plan}


def execute_node(state: HITLState) -> dict:
    """Execute the plan (only if approved)."""
    if not state.get("approved", False):
        print("   🚫 Not approved — cancelling execution")
        return {"result": "CANCELLED: Plan not approved by the human."}

    print(f"\n⚡ Executing approved plan...")

    response = llm.invoke(
        f"""Simulate the execution of this plan step by step.
For each step, indicate:
- ✅ if it executed correctly
- ❌ if there was an error (simulated)

Plan:
{state['plan']}

Simulate that everything executes successfully."""
    )

    result = response.content
    print(f"   Result: {result[:200]}")
    return {"result": result}


# ─── Graph ────────────────────────────────────────────────────────────

builder = StateGraph(HITLState)
builder.add_node("plan", plan_node)
builder.add_node("execute", execute_node)

builder.set_entry_point("plan")
# interrupt_before="execute" → pauses before executing
builder.add_edge("plan", "execute")
builder.add_edge("execute", END)

# Checkpointer needed to pause/resume
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["execute"])


def demo_hitl():
    """Complete demo: plan → pause → approve → execute."""
    config = {"configurable": {"thread_id": "demo-1"}}

    # Step 1: Invoke — the graph pauses before "execute"
    print(f"\n{'='*60}")
    print("🚀 Step 1: Generating plan (will pause before executing)...")

    result = graph.invoke(
        {
            "task": "Delete all test users from the production database",
            "plan": "",
            "approved": False,
            "result": "",
        },
        config=config,
    )

    print(f"\n⏸️  PAUSED — Waiting for human approval")
    print(f"   Proposed plan:\n{result['plan'][:300]}")

    # Step 2: Human reviews and approves
    print(f"\n{'='*60}")
    print("👤 Step 2: Human approves the plan")

    graph.update_state(config, {"approved": True})

    # Step 3: Resume execution
    print(f"\n{'='*60}")
    print("▶️  Step 3: Resuming execution...")

    final = graph.invoke(None, config=config)
    print(f"\n✅ Final result:\n{final['result'][:300]}")


def demo_hitl_rejected():
    """Demo: plan rejected."""
    config = {"configurable": {"thread_id": "demo-2"}}

    print(f"\n{'='*60}")
    print("🚀 Generating plan (will be rejected)...")

    result = graph.invoke(
        {
            "task": "Drop the users table and recreate it",
            "plan": "",
            "approved": False,
            "result": "",
        },
        config=config,
    )

    print(f"\n⏸️  PAUSED")
    print("👤 Human REJECTS the plan (doesn't update approved)")

    # Resume without approving
    final = graph.invoke(None, config=config)
    print(f"\n🚫 Result: {final['result']}")


if __name__ == "__main__":
    demo_hitl()
