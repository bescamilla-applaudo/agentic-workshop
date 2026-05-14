"""
Exercise 5.3 — LLM-as-a-Judge: Automated evaluation.

Evaluates LLM outputs against defined criteria.
Returns individual scores, reasoning, and a pass/fail verdict.
"""

import json
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

DEFAULT_CRITERIA = ["correctness", "readability", "typing", "error_handling", "efficiency"]


def llm_judge(
    task: str,
    response: str,
    criteria: list[str] | None = None,
    threshold: float = 4.0,
) -> dict:
    """
    Evaluate a response against criteria using LLM-as-a-Judge.

    Args:
        task: The original task the response attempts to solve.
        response: The output to evaluate.
        criteria: List of criteria (default: 5 code criteria).
        threshold: Minimum average score to pass (1-5).

    Returns:
        Dict with scores, total, reasoning, passed.
    """
    if criteria is None:
        criteria = DEFAULT_CRITERIA

    criteria_list = "\n".join(f"- {c}" for c in criteria)

    prompt = f"""You are an expert evaluator. Evaluate this response against the given criteria.

## Original task
{task}

## Response to evaluate
{response}

## Criteria (score 1-5 each)
{criteria_list}

Scoring:
- 1: Very poor, doesn't meet the criterion
- 2: Poor, partially meets
- 3: Acceptable, meets the minimum
- 4: Good, meets well
- 5: Excellent, exceptional

Reply ONLY with JSON:
{{
  "scores": {{
    "criterion1": {{"score": N, "reason": "..."}},
    "criterion2": {{"score": N, "reason": "..."}}
  }},
  "total": N.N,
  "overall_reasoning": "general evaluation in 2-3 sentences",
  "passed": true|false
}}

The total is the AVERAGE of all scores. passed = true if total >= {threshold}."""

    llm_response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = llm_response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if JSON is invalid
        result = {
            "scores": {c: {"score": 3, "reason": "Evaluation failed"} for c in criteria},
            "total": 3.0,
            "overall_reasoning": "The evaluator could not generate valid JSON.",
            "passed": False,
        }

    # Recalculate total for safety
    scores = result.get("scores", {})
    if scores:
        numeric_scores = [
            v["score"] if isinstance(v, dict) else v
            for v in scores.values()
        ]
        result["total"] = round(sum(numeric_scores) / len(numeric_scores), 2)
        result["passed"] = result["total"] >= threshold

    return result


def print_evaluation(task: str, response: str, evaluation: dict) -> None:
    """Print the evaluation in a readable format."""
    print(f"\n{'='*60}")
    print(f"📋 Task: {task[:60]}...")
    print(f"📝 Response: {response[:100]}...")
    print(f"\n📊 Scores:")

    for criterion, data in evaluation.get("scores", {}).items():
        if isinstance(data, dict):
            score = data["score"]
            reason = data["reason"]
        else:
            score = data
            reason = "N/A"
        bar = "█" * score + "░" * (5 - score)
        print(f"  {criterion:20s} [{bar}] {score}/5 — {reason[:60]}")

    total = evaluation.get("total", 0)
    passed = evaluation.get("passed", False)
    reasoning = evaluation.get("overall_reasoning", "")

    print(f"\n  {'─'*40}")
    print(f"  Total: {total}/5.0")
    print(f"  Verdict: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"  Reasoning: {reasoning[:200]}")


if __name__ == "__main__":
    # Example: evaluate generated code
    task = "Write a Python function to merge two sorted lists into one sorted list"

    code_response = """
def merge_sorted(a: list[int], b: list[int]) -> list[int]:
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
"""

    evaluation = llm_judge(task, code_response, threshold=4.0)
    print_evaluation(task, code_response, evaluation)
