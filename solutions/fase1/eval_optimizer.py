"""
Exercise 1.6 — Evaluator-Optimizer: Refinement loop.

Generates Python code, evaluates it with 5 criteria, and refines
iteratively until reaching a score >= 0.8 or 3 iterations.
"""

import json
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

CRITERIA = ["correctness", "readability", "typing", "error_handling", "efficiency"]


def generate_code(task: str, feedback: str | None = None) -> str:
    """Generate (or refine) Python code for the given task."""
    prompt = f"Write a Python function for: {task}\n\nReply ONLY with the code, no markdown or explanation."
    if feedback:
        prompt += f"\n\nFeedback from the previous iteration (improve these points):\n{feedback}"

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "You are an expert Python programmer. You generate clean, typed code with error handling."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def evaluate_code(task: str, code: str) -> dict:
    """Evaluate Python code on 5 criteria, each 0.0-1.0."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": "You are a demanding code reviewer. You evaluate Python code with numeric scores and specific feedback."},
            {
                "role": "user",
                "content": f"""Evaluate this Python code that should solve: "{task}"

```python
{code}
```

Evaluate on these 5 criteria (score 0.0-1.0 each):
1. correctness — does it solve the problem correctly?
2. readability — is it easy to read and understand?
3. typing — does it have type hints for parameters and return?
4. error_handling — does it handle edge cases and errors?
5. efficiency — is it efficient for the use case?

Reply ONLY with JSON:
{{
  "scores": {{"correctness": 0.0, "readability": 0.0, "typing": 0.0, "error_handling": 0.0, "efficiency": 0.0}},
  "total": 0.0,
  "feedback": "specific feedback to improve"
}}""",
            }
        ],
    )

    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        # Calculate total as average
        scores = result.get("scores", {})
        if scores:
            result["total"] = round(sum(scores.values()) / len(scores), 3)
        return result
    except json.JSONDecodeError:
        return {
            "scores": {c: 0.5 for c in CRITERIA},
            "total": 0.5,
            "feedback": "Evaluation failed, try again.",
        }


def run_eval_optimizer(task: str, threshold: float = 0.8, max_iterations: int = 3) -> dict:
    """Run the Evaluate-Optimize loop until reaching the threshold or max iterations."""
    print(f"\n{'='*60}")
    print(f"🎯 Task: {task}")
    print(f"📊 Threshold: {threshold} | Max iterations: {max_iterations}")

    feedback = None
    history = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Generate
        print("📝 Generating code...")
        code = generate_code(task, feedback)
        print(f"   {code[:100]}...")

        # Evaluate
        print("🔍 Evaluating...")
        evaluation = evaluate_code(task, code)
        scores = evaluation.get("scores", {})
        total = evaluation.get("total", 0.0)
        feedback = evaluation.get("feedback", "")

        print(f"   Scores: {json.dumps(scores, indent=2)}")
        print(f"   Total: {total}")
        print(f"   Feedback: {feedback[:100]}...")

        history.append({
            "iteration": iteration,
            "code": code,
            "scores": scores,
            "total": total,
            "feedback": feedback,
        })

        if total >= threshold:
            print(f"\n✅ Threshold reached at iteration {iteration}! (score: {total})")
            break
        elif iteration < max_iterations:
            print(f"   ⚡ Score {total} < {threshold}, refining...")

    # Final result
    best = max(history, key=lambda h: h["total"])
    print(f"\n{'='*60}")
    print(f"📊 Summary: {len(history)} iterations")
    for h in history:
        print(f"   Iteration {h['iteration']}: {h['total']}")
    print(f"\n🏆 Best result (iteration {best['iteration']}, score {best['total']}):")
    print(best["code"])

    return {
        "task": task,
        "iterations": len(history),
        "best_score": best["total"],
        "best_code": best["code"],
        "history": history,
    }


if __name__ == "__main__":
    run_eval_optimizer("Write a function to merge two sorted lists into one sorted list")
