"""
Exercise 1.5 — Orchestrator-Workers: Dynamic delegation.

The orchestrator receives a goal, decomposes it into an ordered plan (JSON),
runs each subtask through a specialized worker, then synthesizes everything
into a coherent final output.

Workers:
  researcher — gathers facts, data, concrete examples
  writer     — turns raw material into structured, readable content
  critic     — evaluates quality and returns actionable improvement notes

Execution model:
  - Tasks run sequentially; each worker receives the output of its dependency.
  - The synthesizer sees all worker outputs at once to produce the final result.
"""

import json
import os

from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

# Fast model for planning (short output); capable model for workers + synthesis.
FAST_MODEL    = "llama-3.1-8b-instant"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

WORKER_SYSTEM_PROMPTS: dict[str, str] = {
    "researcher": (
        "You are a senior technical researcher. "
        "Your output is factual, specific, and well-organised. "
        "Include concrete examples, real tools, and measurable data where possible. "
        "Do not speculate — flag uncertainty explicitly."
    ),
    "writer": (
        "You are an experienced technical writer. "
        "Transform research notes into clear, engaging prose aimed at software engineers. "
        "Use headers, bullet points, and short code snippets to aid readability. "
        "Prioritise clarity and correctness over style."
    ),
    "critic": (
        "You are a rigorous technical editor. "
        "Evaluate the draft on: accuracy, clarity, structure, completeness, and tone. "
        "Return a numbered list of specific, actionable issues. "
        "For each issue state the problem and the fix. Be direct — no praise padding."
    ),
}

DEFAULT_PLAN = [
    {"worker": "researcher", "task": "Research the topic thoroughly",       "depends_on": None},
    {"worker": "writer",     "task": "Write a draft using the research",    "depends_on": 0},
    {"worker": "critic",     "task": "Critique the draft for improvements", "depends_on": 1},
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Step:
    index: int
    worker: str
    task: str
    depends_on: int | None  # index into the results list; None = no dependency


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chat(model: str, system: str | None, user: str, max_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Phase 1 — Plan
# ---------------------------------------------------------------------------

PLAN_PROMPT = """\
Decompose the goal below into at most 5 ordered subtasks for a 3-worker team:
  researcher — investigates and summarises facts
  writer     — writes the content
  critic     — evaluates quality and flags issues

Rules:
  - Use each worker type at least once.
  - depends_on is the 0-based index of the step this task must wait for (null if independent).
  - Keep task descriptions concrete and specific to the goal.

Goal: "{goal}"

Reply with ONLY valid JSON (no markdown):
{{
  "plan": [
    {{"worker": "researcher|writer|critic", "task": "<specific instruction>", "depends_on": null | <int>}},
    ...
  ]
}}"""


def plan(goal: str) -> list[Step]:
    """Ask the fast model to decompose the goal into an ordered list of Steps."""
    print("\n[Phase 1] Planning...")
    raw = _chat(FAST_MODEL, None, PLAN_PROMPT.format(goal=goal), max_tokens=512)

    try:
        data = _parse_json(raw)
        steps_raw = data["plan"]
    except (json.JSONDecodeError, KeyError, ValueError):
        print("  Plan parse failed — using default plan.")
        steps_raw = DEFAULT_PLAN

    steps = []
    for i, s in enumerate(steps_raw):
        # Normalize worker: "writer|researcher" → "writer"
        raw_worker = s.get("worker", "writer")
        worker = raw_worker.split("|")[0].strip()
        if worker not in WORKER_SYSTEM_PROMPTS:
            worker = "writer"

        # Normalize depends_on: list → max index; anything else → int or None
        dep = s.get("depends_on")
        if isinstance(dep, list):
            dep = max(dep) if dep else None
        elif dep is not None:
            try:
                dep = int(dep)
            except (TypeError, ValueError):
                dep = None

        steps.append(Step(index=i, worker=worker, task=s.get("task", ""), depends_on=dep))

    for s in steps:
        dep = f"after step {s.depends_on}" if s.depends_on is not None else "independent"
        print(f"  {s.index + 1}. [{s.worker}] {s.task[:70]}  ({dep})")

    return steps


# ---------------------------------------------------------------------------
# Phase 2 — Execute
# ---------------------------------------------------------------------------

def execute(steps: list[Step], goal: str) -> list[str]:
    """Run each step sequentially, passing dependency output as context."""
    print("\n[Phase 2] Executing workers...")
    results: list[str] = []

    for step in steps:
        system = WORKER_SYSTEM_PROMPTS.get(step.worker, WORKER_SYSTEM_PROMPTS["writer"])

        user_parts = [f"Goal: {goal}\n\nYour task: {step.task}"]
        if step.depends_on is not None and 0 <= step.depends_on < len(results):
            prior = results[step.depends_on]
            user_parts.append(f"\n\n--- Output from previous step ---\n{prior}")
        user_msg = "\n".join(user_parts)

        print(f"\n  [{step.worker}] {step.task[:65]}...")
        output = _chat(DEFAULT_MODEL, system, user_msg, max_tokens=1024)
        print(f"  Done ({len(output)} chars)")
        results.append(output)

    return results


# ---------------------------------------------------------------------------
# Phase 3 — Synthesize
# ---------------------------------------------------------------------------

SYNTHESIZE_PROMPT = """\
Synthesize the work of the team below into a polished final output.

Original goal: "{goal}"

{worker_outputs}

Instructions:
  - If the critic flagged issues, resolve them in your final output.
  - Produce the deliverable the goal asked for — not a report about the deliverable.
  - Write for a technical audience. Use clear structure."""


def synthesize(goal: str, steps: list[Step], results: list[str]) -> str:
    """Combine all worker outputs into the final deliverable."""
    print("\n[Phase 3] Synthesizing...")

    sections = "\n\n".join(
        f"=== [{s.worker.upper()}] {s.task} ===\n{r}"
        for s, r in zip(steps, results)
    )
    prompt = SYNTHESIZE_PROMPT.format(goal=goal, worker_outputs=sections)
    final = _chat(DEFAULT_MODEL, None, prompt, max_tokens=2048)
    return final


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------

def orchestrate(goal: str) -> str:
    """Full pipeline: plan → execute → synthesize."""
    print(f"\n{'=' * 60}")
    print(f"Goal: {goal}")

    steps   = plan(goal)
    results = execute(steps, goal)
    final   = synthesize(goal, steps, results)

    print(f"\n{'=' * 60}")
    print("Final output:\n")
    print(final)
    return final


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    orchestrate("Write a technical blog post about prompt chaining")
