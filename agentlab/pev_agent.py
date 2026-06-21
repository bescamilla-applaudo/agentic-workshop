"""
Exercise 2.3 — PEV Agent: Plan → Execute → Verify.

Graph topology:

    plan ──► execute ──current_step < len(plan)──► execute  (loop)
                  │
                  └──current_step == len(plan)──► verify ──verified──► END
                                                      │
                                                      └──not verified──► execute (retry from step 0)

State fields:
  goal         — the goal (immutable)
  plan         — ordered list of step strings produced by plan_node
  current_step — index of the next step to execute
  step_results — accumulated outputs, one per completed step
  verified     — True once verify_node is satisfied
"""

import json
import os
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openai import RateLimitError

load_dotenv()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PEVState(TypedDict):
    goal:         str
    plan:         list[str]
    current_step: int
    step_results: list[str]
    verified:     bool


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    max_tokens=2048,
)

MAX_RETRIES = 1   # how many times verify failure can send us back to execute


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTEXT_CHARS = 800   # max chars kept per prior step result to control prompt size


def _parse_json(raw: str) -> object:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _invoke(prompt: str) -> str:
    """Call the LLM, sleeping briefly and retrying once on TPM rate-limit."""
    for attempt in range(2):
        try:
            return llm.invoke(prompt).content.strip()
        except RateLimitError as exc:
            if attempt == 0:
                wait = 10
                print(f"  [rate-limit] sleeping {wait}s then retrying...")
                time.sleep(wait)
            else:
                raise exc


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

PLAN_PROMPT = """\
Decompose the goal below into 3-5 concrete, sequential steps.
Each step must produce a tangible output the next step can build on.
Reply with ONLY a JSON array of strings — no markdown, no explanation.

Goal: "{goal}"

Example format:
["Step 1: ...", "Step 2: ...", "Step 3: ..."]
"""

EXECUTE_PROMPT = """\
Execute the current step of the plan below.
Produce the full, detailed output for this step only.

Goal: {goal}
Step {n}/{total}: {step}

{context}
Be thorough — the next step will use your output as its primary input.\
"""

VERIFY_PROMPT = """\
Verify whether the combined step outputs fully satisfy the original goal.

Goal: "{goal}"

Step outputs:
{results}

Evaluate strictly:
  1. Does the combined output address every aspect of the goal?
  2. Are all steps coherent and consistent with each other?
  3. Is the result complete enough to stand on its own?

Reply with ONLY valid JSON (no markdown):
{{"verified": true|false, "reason": "<one concise sentence>", "missing": "<what is still missing, or empty string if nothing>"}}
"""


def plan_node(state: PEVState) -> dict:
    print(f"\n{'=' * 60}")
    print(f"[plan] Goal: {state['goal']}")

    raw = _invoke(PLAN_PROMPT.format(goal=state["goal"]))

    try:
        plan = _parse_json(raw)
        if not isinstance(plan, list):
            raise ValueError("not a list")
        plan = [str(s) for s in plan]
    except (json.JSONDecodeError, ValueError):
        # Fallback: treat non-empty lines as steps
        plan = [l.strip() for l in raw.splitlines() if l.strip()]

    for i, step in enumerate(plan):
        print(f"  {i + 1}. {step}")

    return {"plan": plan, "current_step": 0, "step_results": []}


def execute_node(state: PEVState) -> dict:
    idx   = state["current_step"]
    plan  = state["plan"]
    step  = plan[idx]

    print(f"\n[execute] step {idx + 1}/{len(plan)}: {step[:70]}")

    # Build accumulated context — truncate each prior result to avoid TPM limits.
    if state["step_results"]:
        context_parts = ["Previous step outputs (summarised):"]
        for i, r in enumerate(state["step_results"]):
            snippet = r[:CONTEXT_CHARS] + ("…" if len(r) > CONTEXT_CHARS else "")
            context_parts.append(f"\n--- Step {i + 1} result ---\n{snippet}")
        context = "\n".join(context_parts)
    else:
        context = ""

    result = _invoke(
        EXECUTE_PROMPT.format(
            goal=state["goal"],
            n=idx + 1,
            total=len(plan),
            step=step,
            context=context,
        )
    )

    print(f"  Done ({len(result)} chars)")

    return {
        "step_results": state["step_results"] + [result],
        "current_step": idx + 1,
    }


def verify_node(state: PEVState) -> dict:
    print(f"\n[verify] checking {len(state['step_results'])} step results...")

    results_block = "\n\n".join(
        f"--- Step {i + 1}: {state['plan'][i]} ---\n{r}"
        for i, r in enumerate(state["step_results"])
    )

    raw = _invoke(VERIFY_PROMPT.format(
        goal=state["goal"],
        results=results_block,
    ))

    try:
        data     = _parse_json(raw)
        verified = bool(data.get("verified", False))
        reason   = data.get("reason", "")
        missing  = data.get("missing", "")
    except (json.JSONDecodeError, ValueError):
        verified = "true" in raw.lower()
        reason   = raw
        missing  = ""

    status = "PASS" if verified else "FAIL"
    print(f"  [{status}] {reason}")
    if missing:
        print(f"  Missing: {missing}")

    # Reset execution state so a retry starts from step 0 with a clean slate.
    return {
        "verified":     verified,
        "current_step": 0,
        "step_results": [],
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def after_execute(state: PEVState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "execute"
    return "verify"


_retry_count = 0   # module-level counter; reset per run() call

def after_verify(state: PEVState) -> str:
    global _retry_count
    if state["verified"]:
        print("\n[router] verified → END")
        return END
    if _retry_count < MAX_RETRIES:
        _retry_count += 1
        print(f"\n[router] not verified — retrying all steps (retry {_retry_count}/{MAX_RETRIES})")
        return "execute"
    print(f"\n[router] not verified — max retries reached → END")
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(PEVState)
builder.add_node("plan",    plan_node)
builder.add_node("execute", execute_node)
builder.add_node("verify",  verify_node)

builder.set_entry_point("plan")
builder.add_edge("plan", "execute")
builder.add_conditional_edges(
    "execute",
    after_execute,
    {"execute": "execute", "verify": "verify"},
)
builder.add_conditional_edges(
    "verify",
    after_verify,
    {"execute": "execute", END: END},
)

graph = builder.compile()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(goal: str) -> dict:
    """Run the PEV agent and print the synthesised final output."""
    global _retry_count
    _retry_count = 0

    final = graph.invoke({
        "goal":         goal,
        "plan":         [],
        "current_step": 0,
        "step_results": [],
        "verified":     False,
    })

    print(f"\n{'=' * 60}")
    print(f"Completed  steps={len(final['plan'])}  verified={final['verified']}\n")

    # Concatenate all step results as the final deliverable
    output = "\n\n".join(
        f"## {final['plan'][i]}\n\n{r}"
        for i, r in enumerate(final["step_results"])
    )
    print(output[:3000])
    if len(output) > 3000:
        print(f"\n... ({len(output)} chars total)")

    return final


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run("Create a comprehensive comparison of 3 Python web frameworks")
