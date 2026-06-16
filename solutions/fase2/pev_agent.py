"""
Exercise 2.3 — PEV Agent: Plan → Execute → Verify.

Agent that plans, executes step by step, and verifies the result.
"""

import json
import os
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# ─── State ────────────────────────────────────────────────────────────

class PEVState(TypedDict):
    goal: str
    plan: list[str]
    current_step: int
    step_results: list[str]
    verified: bool


# ─── LLM ──────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model=os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
    max_tokens=2048,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# ─── Nodes ────────────────────────────────────────────────────────────

def plan_node(state: PEVState) -> dict:
    """Plan node: decomposes the goal into concrete steps."""
    print(f"\n📋 Planning...")

    prompt = f"""Decompose this goal into 3-5 concrete and sequential steps.
Each step must produce a tangible output that the next step can use.

Goal: "{state['goal']}"

Reply ONLY with a JSON array of strings:
["Step 1: ...", "Step 2: ...", "Step 3: ..."]"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    try:
        plan = json.loads(raw)
        if not isinstance(plan, list):
            plan = [raw]
    except json.JSONDecodeError:
        # Fallback: split by lines
        plan = [line.strip() for line in raw.split("\n") if line.strip()]

    for i, step in enumerate(plan):
        print(f"   {i+1}. {step}")

    return {"plan": plan, "current_step": 0, "step_results": []}


def execute_node(state: PEVState) -> dict:
    """Execute node: executes the current step using previous context."""
    step_idx = state["current_step"]
    plan = state["plan"]

    if step_idx >= len(plan):
        return {}

    step = plan[step_idx]
    print(f"\n⚙️  Executing step {step_idx + 1}/{len(plan)}: {step[:60]}...")

    # Context: previous results
    context = ""
    if state["step_results"]:
        context = "\n\nResults from previous steps:\n"
        for i, result in enumerate(state["step_results"]):
            context += f"\n--- Step {i+1} ---\n{result[:300]}\n"

    prompt = f"""Execute this step of the plan:

Overall goal: "{state['goal']}"
Current step: "{step}"
{context}

Generate the complete output for this step. Be specific and detailed."""

    response = llm.invoke(prompt)
    result = response.content

    new_results = list(state["step_results"]) + [result]
    print(f"   ✅ Step {step_idx + 1} completed ({len(result)} chars)")

    return {"step_results": new_results, "current_step": step_idx + 1}


def verify_node(state: PEVState) -> dict:
    """Verify node: verifies that the complete result fulfills the goal."""
    print(f"\n🔍 Verifying...")

    all_results = "\n\n".join(
        f"--- Step {i+1} ---\n{r}" for i, r in enumerate(state["step_results"])
    )

    prompt = f"""Verify if the results fulfill the original goal.

Goal: "{state['goal']}"

Results from all steps:
{all_results}

Evaluate:
1. Was the goal fulfilled?
2. Are there gaps or incomplete steps?
3. Is the result coherent?

Reply ONLY with JSON: {{"verified": true|false, "reason": "..."}}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    try:
        result = json.loads(raw)
        verified = result.get("verified", False)
        reason = result.get("reason", "")
    except json.JSONDecodeError:
        verified = "verified" in raw.lower() and "true" in raw.lower()
        reason = raw

    print(f"   {'✅' if verified else '❌'} Verified: {verified}")
    print(f"   Reason: {reason[:100]}")

    return {"verified": verified}


# ─── Conditional edges ────────────────────────────────────────────────

def after_execute(state: PEVState) -> str:
    """After executing: more steps or verify."""
    if state["current_step"] < len(state["plan"]):
        print(f"   → Next step ({state['current_step'] + 1}/{len(state['plan'])})")
        return "execute"
    print("   → All steps completed, verifying...")
    return "verify"


def after_verify(state: PEVState) -> str:
    """After verifying: finish or retry."""
    if state["verified"]:
        return END
    # If not verified and there are steps, we could retry the last one
    print("   ⚠️ Verification failed, but finishing to avoid infinite loop")
    return END


# ─── Graph ────────────────────────────────────────────────────────────

builder = StateGraph(PEVState)
builder.add_node("plan", plan_node)
builder.add_node("execute", execute_node)
builder.add_node("verify", verify_node)

builder.set_entry_point("plan")
builder.add_edge("plan", "execute")
builder.add_conditional_edges("execute", after_execute, {"execute": "execute", "verify": "verify"})
builder.add_conditional_edges("verify", after_verify, {END: END})

graph = builder.compile()


def run_pev(goal: str) -> dict:
    """Run the PEV agent."""
    print(f"\n{'='*60}")
    print(f"🎯 Goal: {goal}")

    result = graph.invoke({
        "goal": goal,
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "verified": False,
    })

    print(f"\n{'='*60}")
    print(f"📊 Result: {len(result['plan'])} steps, verified={result['verified']}")
    return result


if __name__ == "__main__":
    run_pev("Create a comprehensive comparison of 3 Python web frameworks: FastAPI, Django, and Flask")
