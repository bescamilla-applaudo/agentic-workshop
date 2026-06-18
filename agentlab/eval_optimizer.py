"""
Exercise 1.6 — Evaluator-Optimizer: Refinement loop.

Loop:
  1. Generator produces Python code for the task (seeded with evaluator feedback on retry).
  2. Evaluator scores it on 5 criteria (0.0-1.0 each); total = average.
  3. If total >= THRESHOLD or iteration == MAX_ITER → stop.
  4. Otherwise, pass per-criterion feedback back to the generator and repeat.

The best-scoring iteration is returned regardless of which loop it came from.
"""

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

MODEL      = "llama-3.3-70b-versatile"
CRITERIA   = ("correctness", "readability", "typing", "error_handling", "efficiency")
THRESHOLD  = 0.8
MAX_ITER   = 3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Evaluation:
    scores:   dict[str, float]
    total:    float
    feedback: dict[str, str]   # per-criterion actionable note

    def score_bar(self, width: int = 20) -> str:
        filled = round(self.total * width)
        return f"[{'█' * filled}{'░' * (width - filled)}] {self.total:.2f}"


@dataclass
class Iteration:
    number:     int
    code:       str
    evaluation: Evaluation


@dataclass
class Result:
    task:       str
    iterations: list[Iteration] = field(default_factory=list)

    @property
    def best(self) -> Iteration:
        return max(self.iterations, key=lambda it: it.evaluation.total)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat(system: str, user: str, max_tokens: int = 1024) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _strip_fences(code: str) -> str:
    """Remove markdown code fences if the generator wraps the output in them."""
    if code.startswith("```"):
        lines = code.splitlines()
        # drop opening fence line
        lines = lines[1:]
        # drop closing fence if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return code


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GENERATE_SYSTEM = (
    "You are an expert Python programmer. "
    "Write clean, idiomatic Python with full type hints and proper error handling. "
    "Reply with ONLY the code — no explanation, no markdown fences."
)

GENERATE_PROMPT = "Write a Python function for the following task:\n\n{task}"

REFINE_PROMPT = (
    "Write an improved Python function for the following task:\n\n{task}\n\n"
    "The previous attempt scored {total:.2f}/1.0. "
    "Apply these specific improvements:\n{feedback_lines}"
)


def generate(task: str, prev: Evaluation | None = None) -> str:
    if prev is None:
        prompt = GENERATE_PROMPT.format(task=task)
    else:
        feedback_lines = "\n".join(
            f"  • [{criterion}] {note}"
            for criterion, note in prev.feedback.items()
            if note
        )
        prompt = REFINE_PROMPT.format(
            task=task, total=prev.total, feedback_lines=feedback_lines
        )
    return _strip_fences(_chat(GENERATE_SYSTEM, prompt, max_tokens=1024))


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

EVALUATE_SYSTEM = (
    "You are a demanding Python code reviewer. "
    "Score honestly — a score of 1.0 means the criterion is fully satisfied with no room for improvement."
)

EVALUATE_PROMPT = """\
Evaluate the Python code below for the task: "{task}"

```python
{code}
```

Score each criterion from 0.0 to 1.0 and provide one specific, actionable improvement note.
Criteria:
  correctness    — solves the problem completely and correctly
  readability    — naming, structure, and clarity
  typing         — full type hints on parameters and return value
  error_handling — handles edge cases (empty inputs, wrong types, etc.)
  efficiency     — algorithmically sound for the problem

Reply with ONLY valid JSON (no markdown):
{{
  "scores": {{
    "correctness":    <0.0-1.0>,
    "readability":    <0.0-1.0>,
    "typing":         <0.0-1.0>,
    "error_handling": <0.0-1.0>,
    "efficiency":     <0.0-1.0>
  }},
  "feedback": {{
    "correctness":    "<one actionable sentence or empty string if perfect>",
    "readability":    "<one actionable sentence or empty string if perfect>",
    "typing":         "<one actionable sentence or empty string if perfect>",
    "error_handling": "<one actionable sentence or empty string if perfect>",
    "efficiency":     "<one actionable sentence or empty string if perfect>"
  }}
}}"""


def evaluate(task: str, code: str) -> Evaluation:
    raw = _chat(EVALUATE_SYSTEM, EVALUATE_PROMPT.format(task=task, code=code), max_tokens=512)
    try:
        data     = _parse_json(raw)
        scores   = {c: float(data["scores"].get(c, 0.5)) for c in CRITERIA}
        feedback = {c: data.get("feedback", {}).get(c, "") for c in CRITERIA}
    except (json.JSONDecodeError, KeyError, ValueError):
        scores   = {c: 0.5 for c in CRITERIA}
        feedback = {c: "evaluation parse error" for c in CRITERIA}

    total = round(sum(scores.values()) / len(CRITERIA), 3)
    return Evaluation(scores=scores, total=total, feedback=feedback)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

def run(task: str, threshold: float = THRESHOLD, max_iter: int = MAX_ITER) -> Result:
    print(f"\n{'=' * 60}")
    print(f"Task      : {task}")
    print(f"Threshold : {threshold}   Max iterations: {max_iter}")

    result = Result(task=task)
    prev_eval: Evaluation | None = None

    for n in range(1, max_iter + 1):
        print(f"\n── Iteration {n}/{max_iter} {'─' * 40}")

        # Generate
        print("  [generator] producing code...")
        code = generate(task, prev_eval)
        print(f"  {code.splitlines()[0][:72]}...")   # show signature line

        # Evaluate
        print("  [evaluator] scoring...")
        ev = evaluate(task, code)
        result.iterations.append(Iteration(number=n, code=code, evaluation=ev))

        # Print scores
        print(f"\n  Score  {ev.score_bar()}")
        for criterion in CRITERIA:
            note = ev.feedback.get(criterion, "")
            note_str = f"  ← {note}" if note else ""
            print(f"    {criterion:<16} {ev.scores[criterion]:.2f}{note_str}")

        prev_eval = ev

        if ev.total >= threshold:
            print(f"\n  ✓ Threshold {threshold} reached — stopping early.")
            break
        elif n < max_iter:
            print(f"\n  ✗ Score {ev.total:.2f} < {threshold} — refining...")

    # Summary
    best = result.best
    print(f"\n{'=' * 60}")
    print("Summary:")
    for it in result.iterations:
        marker = " ← best" if it.number == best.number else ""
        print(f"  Iteration {it.number}: {it.evaluation.score_bar()}{marker}")

    print(f"\nBest code (iteration {best.number}, score {best.evaluation.total:.2f}):\n")
    print(best.code)
    return result


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run("Write a function to merge two sorted lists")
