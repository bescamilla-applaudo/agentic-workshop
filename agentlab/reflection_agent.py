"""
Exercise 2.2 — Reflection Agent: Writer with self-critique.

Graph topology:

    generate ──► reflect ──APPROVED or iter>=3──► END
        ▲            │
        └────────────┘  (feedback → refine)

State fields:
  task      — the writing goal (immutable)
  draft     — current best draft (updated by generate)
  feedback  — reviewer notes (updated by reflect)
  iteration — how many generate calls have run
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ReflectionState(TypedDict):
    task:      str
    draft:     str
    feedback:  str
    iteration: int

# ---------------------------------------------------------------------------
# LLM — ChatOpenAI pointed at Groq
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    max_tokens=2048,
)

MAX_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

GENERATE_FIRST = """\
Write technical documentation in Markdown for the following task:

{task}

Cover all of these sections:
1. Overview — what the component does and why it exists
2. How it works — internal mechanics (concise, no hand-waving)
3. Configuration — all options with types, defaults, and descriptions
4. Usage example — complete, runnable code snippet
5. Edge cases & gotchas — failure modes, limits, common mistakes
6. API reference — public methods / headers / response codes
"""

GENERATE_REFINE = """\
Revise the technical documentation draft below based on the reviewer's feedback.
Produce the COMPLETE updated draft — do not summarise or skip sections.

Task: {task}

Current draft:
{draft}

Reviewer feedback:
{feedback}
"""

REFLECT_PROMPT = """\
You are a senior technical writer reviewing documentation for correctness and usability.

Task the documentation is supposed to cover:
{task}

Draft under review:
{draft}

Evaluate strictly on these five criteria:
  1. Completeness — all sections present and thorough
  2. Clarity — an intermediate developer understands without googling
  3. Examples — code is correct, self-contained, and illustrates real usage
  4. Structure — sections flow logically; headers and formatting are consistent
  5. Edge cases — limitations, error conditions, and gotchas are documented

If the draft fully satisfies ALL five criteria, reply with exactly the word APPROVED \
on the first line.
Otherwise provide numbered, specific feedback — not "improve clarity" but \
"the Configuration section omits the default value for max_requests". \
One issue per line maximum.
"""


def generate_node(state: ReflectionState) -> dict:
    iteration = state["iteration"] + 1
    is_first  = state["draft"] == ""

    print(f"\n── Generate  (iteration {iteration}/{MAX_ITERATIONS}) {'─'*30}")

    if is_first:
        prompt = GENERATE_FIRST.format(task=state["task"])
    else:
        prompt = GENERATE_REFINE.format(
            task=state["task"],
            draft=state["draft"],
            feedback=state["feedback"],
        )

    draft = llm.invoke(prompt).content.strip()

    # Show the draft inline so progress is visible
    preview_lines = draft.splitlines()[:6]
    print("\n".join(f"  {l}" for l in preview_lines))
    if len(draft.splitlines()) > 6:
        print(f"  ... ({len(draft)} chars total)")

    return {"draft": draft, "iteration": iteration}


def reflect_node(state: ReflectionState) -> dict:
    print(f"\n── Reflect {'─'*42}")

    feedback = llm.invoke(
        REFLECT_PROMPT.format(task=state["task"], draft=state["draft"])
    ).content.strip()

    # Show feedback preview
    for line in feedback.splitlines()[:5]:
        print(f"  {line}")
    if len(feedback.splitlines()) > 5:
        print("  ...")

    return {"feedback": feedback}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(state: ReflectionState) -> str:
    if "APPROVED" in state["feedback"].upper():
        print("\n  → APPROVED — stopping.")
        return END

    if state["iteration"] >= MAX_ITERATIONS:
        print(f"\n  → Max iterations ({MAX_ITERATIONS}) reached — stopping.")
        return END

    print(f"\n  → Score not yet approved — refining (iteration {state['iteration']}/{MAX_ITERATIONS})...")
    return "generate"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(ReflectionState)
builder.add_node("generate", generate_node)
builder.add_node("reflect",  reflect_node)

builder.set_entry_point("generate")
builder.add_edge("generate", "reflect")
builder.add_conditional_edges(
    "reflect",
    should_continue,
    {"generate": "generate", END: END},
)

graph = builder.compile()

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(task: str) -> str:
    """Run the reflection loop and return the final draft."""
    print(f"\n{'=' * 60}")
    print(f"Task: {task}")

    final = graph.invoke({
        "task":      task,
        "draft":     "",
        "feedback":  "",
        "iteration": 0,
    })

    print(f"\n{'=' * 60}")
    print(f"Final draft  (iteration {final['iteration']}):\n")
    print(final["draft"])
    return final["draft"]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run("Write documentation for a REST API rate limiter middleware")
