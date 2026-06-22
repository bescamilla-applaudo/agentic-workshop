"""
Exercise 5.3 — LLM-as-a-Judge: Automated evaluation.

llm_judge(task, response, criteria, threshold) evaluates any LLM response
against a list of named criteria, each scored 1-5.

Return schema:
{
  "scores": {
    "<criterion>": {"score": <1-5>, "reason": "<one sentence>"},
    ...
  },
  "average": <float>,          # arithmetic mean of all scores
  "passed": <bool>,            # average >= threshold
  "overall_reasoning": "<str>" # 2-3 sentence synthesis
}

Scoring rubric (embedded in the prompt so the judge is consistent):
  1 — Does not meet the criterion at all
  2 — Partially meets; significant gaps
  3 — Meets the minimum; average quality
  4 — Meets well; minor issues only
  5 — Exemplary; no meaningful improvement possible

The judge recalculates `average` and `passed` locally after parsing
to ensure they match the raw scores even if the model drifts.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Use the capable model as judge — a weak judge produces unreliable verdicts.
JUDGE_MODEL = os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

CODE_CRITERIA = ["correctness", "readability", "typing", "error_handling", "efficiency"]

THRESHOLD = 4.0   # average score required to pass


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are a strict, expert evaluator. Score the response below against each criterion.

## Task the response is supposed to solve
{task}

## Response under evaluation
{response}

## Criteria and scoring rubric (score 1-5 for each)
{criteria_block}

Rubric:
  5 — Exemplary; no meaningful improvement possible
  4 — Meets the criterion well; only minor issues
  3 — Meets minimum requirements; average quality
  2 — Partially meets; significant gaps present
  1 — Does not meet the criterion at all

Be strict: a score of 5 means there is genuinely nothing to improve.
Score each criterion independently; do not let one strong area inflate others.

Reply with ONLY valid JSON (no markdown fences):
{{
  "scores": {{
    {score_template}
  }},
  "overall_reasoning": "<2-3 sentence synthesis of the evaluation>"
}}

Do NOT include "average" or "passed" — those are computed externally.\
"""


def _build_criteria_block(criteria: list[str]) -> str:
    return "\n".join(f"  - {c}" for c in criteria)


def _build_score_template(criteria: list[str]) -> str:
    entries = [f'"{c}": {{"score": <1-5>, "reason": "<one actionable sentence>"}}' for c in criteria]
    return ",\n    ".join(entries)


def _parse_json(raw: str) -> dict:
    """
    Extract and parse the first JSON object from `raw`.

    Handles three common model output patterns:
      1. Clean JSON                       → parse directly
      2. Markdown code fence (```json…```)→ strip fence, parse
      3. Reasoning/prose then JSON        → find first '{' … last '}' block
    """
    text = raw.strip()

    # Pattern 2: markdown fence
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.lstrip("json").strip()
            if candidate.startswith("{"):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Pattern 1: clean JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Pattern 3: JSON embedded in reasoning prose — extract outermost { … }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())

    raise json.JSONDecodeError("No JSON object found", text, 0)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def llm_judge(
    task:      str,
    response:  str,
    criteria:  list[str] | None = None,
    threshold: float = THRESHOLD,
    model:     str | None = None,
) -> dict:
    """
    Evaluate `response` against `criteria` using an LLM judge.

    Args:
        task:      The original goal the response attempts to solve.
        response:  The output to evaluate.
        criteria:  Named criteria to score. Defaults to CODE_CRITERIA.
        threshold: Minimum average score (1-5) to mark as passed.
        model:     Override the judge model (defaults to JUDGE_MODEL).

    Returns:
        dict with keys: scores, average, passed, overall_reasoning.
    """
    criteria = criteria or CODE_CRITERIA

    prompt = JUDGE_PROMPT.format(
        task=task,
        response=response,
        criteria_block=_build_criteria_block(criteria),
        score_template=_build_score_template(criteria),
    )

    raw = client.chat.completions.create(
        model=model or JUDGE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content.strip()

    # Parse — fall back to neutral scores on failure
    try:
        data = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        # Print raw output to help diagnose unexpected model responses
        preview = raw[:300].replace("\n", " ")
        print(f"  [judge] parse error — raw: {preview!r}")
        data = {
            "scores": {c: {"score": 3, "reason": "parse error — judge returned invalid JSON"} for c in criteria},
            "overall_reasoning": "Evaluation could not be parsed.",
        }

    scores_raw = data.get("scores", {})

    # Normalise: accept both {"score": N, "reason": "..."} and plain int/float
    scores: dict[str, dict] = {}
    for c in criteria:
        entry = scores_raw.get(c, {"score": 3, "reason": "missing"})
        if isinstance(entry, dict):
            raw_score = entry.get("score", 3)
            reason    = entry.get("reason", "")
        else:
            raw_score = entry
            reason    = ""
        # Clamp to [1, 5]
        scores[c] = {"score": max(1, min(5, int(round(raw_score)))), "reason": reason}

    # Recompute average and passed locally — never trust the model's arithmetic
    avg    = round(sum(s["score"] for s in scores.values()) / len(scores), 2)
    passed = avg >= threshold

    return {
        "scores":             scores,
        "average":            avg,
        "passed":             passed,
        "overall_reasoning":  data.get("overall_reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def print_verdict(task: str, evaluation: dict, threshold: float = THRESHOLD) -> None:
    """Render an evaluation result to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Task     : {task[:80]}")
    print(f"Threshold: {threshold}/5.0\n")
    print("Scores:")

    for criterion, data in evaluation["scores"].items():
        score  = data["score"]
        reason = data["reason"]
        bar    = "█" * score + "░" * (5 - score)
        print(f"  {criterion:<16} [{bar}] {score}/5  {reason[:55]}")

    avg    = evaluation["average"]
    passed = evaluation["passed"]
    print(f"\n  {'─' * 46}")
    print(f"  Average  : {avg}/5.0")
    print(f"  Verdict  : {'PASSED' if passed else 'FAILED'}")
    print(f"\n  Reasoning: {evaluation['overall_reasoning'][:300]}")


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Example 1: evaluate generated code ──────────────────────────────
    code_task = "Write a Python function to merge two sorted lists into one sorted list"

    code_response = """\
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

    evaluation = llm_judge(
        task=code_task,
        response=code_response,
        criteria=CODE_CRITERIA,
        threshold=4.0,
    )
    print_verdict(code_task, evaluation, threshold=4.0)

    time.sleep(5)   # avoid back-to-back rate-limit on free tier

    # ── Example 2: evaluate a prose explanation ──────────────────────────
    explanation_task = "Explain what the GIL is and when it matters for Python developers"

    explanation_response = """\
The Global Interpreter Lock (GIL) is a mutex in CPython that allows only one
thread to execute Python bytecode at a time. It simplifies memory management
but limits true multi-core parallelism for CPU-bound Python threads.

For I/O-bound work (network calls, file reads) the GIL is released during the
wait, so threading works fine. For CPU-bound work (number crunching, image
processing) you should use multiprocessing or an alternative runtime like Jython
that has no GIL.
"""

    prose_criteria = ["accuracy", "clarity", "completeness", "conciseness"]

    evaluation2 = llm_judge(
        task=explanation_task,
        response=explanation_response,
        criteria=prose_criteria,
        threshold=4.0,
    )
    print_verdict(explanation_task, evaluation2, threshold=4.0)
