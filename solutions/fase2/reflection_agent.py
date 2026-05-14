"""
Exercise 2.2 — Reflection Agent: Writer with self-critique.

Generate → Reflect → Refine loop as a state graph.
The agent improves technical documentation iteratively.
"""

import os
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# ─── State ────────────────────────────────────────────────────────────

class ReflectionState(TypedDict):
    task: str
    draft: str
    feedback: str
    iteration: int


# ─── LLM ──────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model=os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
    max_tokens=2048,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

MAX_ITERATIONS = 3

# ─── Nodes ────────────────────────────────────────────────────────────

def generate(state: ReflectionState) -> dict:
    """Generate node: writes or refines the draft based on feedback."""
    iteration = state["iteration"] + 1
    print(f"\n📝 Generate (iteration {iteration})...")

    if state["draft"]:
        # Refine with feedback
        prompt = f"""Improve this technical documentation draft based on the feedback.

Original task: {state['task']}

Current draft:
{state['draft']}

Reviewer feedback:
{state['feedback']}

Generate the complete improved draft (not just the changes)."""
    else:
        # First generation
        prompt = f"""Write technical documentation for: {state['task']}

Include:
- Component description
- How it works internally
- Usage example with code
- Edge cases and considerations
- API reference (if applicable)

Format: Markdown."""

    response = llm.invoke(prompt)
    draft = response.content
    print(f"   ✅ Draft generated ({len(draft)} chars)")

    return {"draft": draft, "iteration": iteration}


def reflect(state: ReflectionState) -> dict:
    """Reflect node: critiques the draft with actionable feedback."""
    print(f"\n🔍 Reflect...")

    prompt = f"""You are a demanding technical documentation reviewer. Evaluate this draft:

Task: {state['task']}

Draft:
{state['draft']}

Evaluate on these criteria:
1. Completeness — does it cover all aspects of the component?
2. Clarity — can an intermediate developer understand it without googling?
3. Examples — are the code examples correct and useful?
4. Structure — is the organization logical?
5. Edge cases — does it mention limitations and gotchas?

If the draft is EXCELLENT on all criteria, reply exactly: "APPROVED"
If not, give SPECIFIC and actionable feedback (not "improve clarity" but "section X doesn't explain Y")."""

    response = llm.invoke(prompt)
    feedback = response.content
    print(f"   📋 Feedback: {feedback[:150]}...")

    return {"feedback": feedback}


# ─── Conditional edge ─────────────────────────────────────────────────

def should_continue(state: ReflectionState) -> str:
    """Decide whether to keep refining or finish."""
    if "APPROVED" in state["feedback"]:
        print("   ✅ Draft APPROVED by the reviewer")
        return END

    if state["iteration"] >= MAX_ITERATIONS:
        print(f"   ⚠️ Maximum iterations reached ({MAX_ITERATIONS})")
        return END

    print(f"   🔄 Refining (iteration {state['iteration']}/{MAX_ITERATIONS})...")
    return "generate"


# ─── Graph ────────────────────────────────────────────────────────────

builder = StateGraph(ReflectionState)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)

builder.set_entry_point("generate")
builder.add_edge("generate", "reflect")
builder.add_conditional_edges("reflect", should_continue, {"generate": "generate", END: END})

graph = builder.compile()


def run_reflection(task: str) -> str:
    """Run the reflection agent."""
    print(f"\n{'='*60}")
    print(f"🎯 Task: {task}")

    result = graph.invoke({
        "task": task,
        "draft": "",
        "feedback": "",
        "iteration": 0,
    })

    print(f"\n{'='*60}")
    print(f"📄 Final draft (iteration {result['iteration']}):")
    print(result["draft"][:500])
    return result["draft"]


if __name__ == "__main__":
    run_reflection("Write documentation for a REST API rate limiter middleware")
