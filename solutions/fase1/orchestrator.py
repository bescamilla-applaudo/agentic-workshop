"""
Exercise 1.5 — Orchestrator-Workers: Dynamic delegation.

An orchestrator decomposes a goal into subtasks, delegates to specialized
workers, and synthesizes the result.
"""

import json
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

WORKER_PROMPTS = {
    "researcher": "You are an expert researcher. Your job is to search for and summarize relevant information about the assigned topic. Be specific and cite concrete data.",
    "writer": "You are a technical writer. Your job is to take raw information and turn it into well-structured, clear, and engaging content. Use headers, bullet points, and examples.",
    "critic": "You are a demanding critic. Your job is to evaluate content quality: clarity, precision, completeness, structure. Give specific and actionable feedback, not generic.",
}


def orchestrate(goal: str) -> str:
    """The orchestrator: decomposes, delegates, synthesizes."""
    print(f"\n{'='*60}")
    print(f"🎯 Goal: {goal}")

    # Step 1: Decompose into subtasks
    print("\n📋 Step 1: Decomposing into subtasks...")
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"""Decompose this goal into subtasks for a team of 3 workers:
- researcher: investigates and gathers data
- writer: writes content
- critic: evaluates quality

Goal: "{goal}"

Reply ONLY with JSON:
{{
  "plan": [
    {{"worker": "researcher|writer|critic", "task": "specific description", "depends_on": null|0|1}}
  ]
}}

Order tasks by dependency. Maximum 5 subtasks.""",
            }
        ],
    )

    raw = response.choices[0].message.content.strip()
    try:
        plan_data = json.loads(raw)
        plan = plan_data["plan"]
    except (json.JSONDecodeError, KeyError):
        print(f"   ⚠️ Invalid plan, using default plan")
        plan = [
            {"worker": "researcher", "task": f"Research about: {goal}", "depends_on": None},
            {"worker": "writer", "task": "Write an article with the research", "depends_on": 0},
            {"worker": "critic", "task": "Evaluate the article quality", "depends_on": 1},
        ]

    for i, step in enumerate(plan):
        print(f"   {i+1}. [{step['worker']}] {step['task']}")

    # Step 2: Execute each subtask
    print("\n⚙️  Step 2: Executing subtasks...")
    results = []

    for i, step in enumerate(plan):
        worker = step["worker"]
        task = step["task"]
        system_prompt = WORKER_PROMPTS.get(worker, WORKER_PROMPTS["writer"])

        # Context: include previous results if there are dependencies
        context = ""
        if step.get("depends_on") is not None:
            dep_idx = step["depends_on"]
            if 0 <= dep_idx < len(results):
                context = f"\n\nContext from the previous step:\n{results[dep_idx][:500]}"

        print(f"\n   🔧 Worker [{worker}]: {task[:60]}...")

        worker_response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{task}{context}"},
            ],
        )

        result = worker_response.choices[0].message.content
        results.append(result)
        print(f"   ✅ Completed ({len(result)} chars)")

    # Step 3: Synthesize
    print("\n🔗 Step 3: Synthesizing final result...")

    synthesis_context = ""
    for i, (step, result) in enumerate(zip(plan, results)):
        synthesis_context += f"\n\n--- [{step['worker']}] {step['task']} ---\n{result}"

    synthesis = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"""Synthesize the work of all workers into a coherent final result.

Original goal: "{goal}"

Results from each worker:
{synthesis_context}

Generate the final output integrating everything. If the critic identified issues, resolve them in the synthesis.""",
            }
        ],
    )

    final = synthesis.choices[0].message.content
    print(f"\n{'='*60}")
    print("✅ Final result:\n")
    print(final)
    return final


if __name__ == "__main__":
    orchestrate("Write a technical blog post about prompt chaining patterns in LLM applications")
