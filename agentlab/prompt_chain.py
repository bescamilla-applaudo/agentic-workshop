"""
Exercise 1.2 — Prompt Chaining: Processing pipeline.

3-step pipeline with validation gates between each step.
Step 1 (fast model): Extract entities → Gate: valid JSON with required fields.
Step 2 (fast model): Enrich with category and risk → Gate: category field present.
Step 3 (capable model): Format executive summary.
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Simple steps use a fast/cheap model; the final human-facing output uses a stronger one.
FAST_MODEL = os.environ.get("FAST_MODEL", "google/gemma-4-31b-it:free")
CAPABLE_MODEL = os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

def _chat(model: str, messages: list[dict], max_tokens: int = 512) -> str:
    """Call the model with one retry on rate-limit, falling back to CAPABLE_MODEL."""
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            if attempt == 0 and model != CAPABLE_MODEL:
                print(f"  Rate-limited on {model}, retrying with {CAPABLE_MODEL}...")
                model = CAPABLE_MODEL
                time.sleep(2)
            else:
                raise


REQUIRED_FIELDS = {"name", "date", "amount", "description"}

# Amount thresholds for category and risk classification
CATEGORY_THRESHOLDS = [
    (50, "micro"),
    (500, "regular"),
    (5_000, "significant"),
    (float("inf"), "major"),
]
RISK_THRESHOLDS = [
    (1_000, "low"),
    (5_000, "medium"),
    (float("inf"), "high"),
]


def _classify(value: float, thresholds: list[tuple]) -> str:
    for limit, label in thresholds:
        if value < limit:
            return label
    return thresholds[-1][1]


# ---------------------------------------------------------------------------
# Step 1 — Extract
# ---------------------------------------------------------------------------

def step_extract(text: str) -> dict:
    """Extract name, date, amount, description from free text as JSON."""
    print("\n[Step 1] Extracting entities...")

    raw = _chat(
        FAST_MODEL,
        [
            {
                "role": "user",
                "content": (
                    "Extract the transaction entities from the text below as JSON.\n"
                    "Required fields:\n"
                    "  - name (string): person or entity involved\n"
                    "  - date (string, YYYY-MM-DD): transaction date; infer year if missing\n"
                    "  - amount (number): monetary amount as a numeric value\n"
                    "  - description (string): brief description of what the transaction was for\n\n"
                    f"Text: {text}\n\n"
                    "Reply with ONLY the JSON object, no markdown fences or explanation."
                ),
            }
        ],
        max_tokens=512,
    )
    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    print(f"  Raw output: {raw[:120]}")

    # Gate 1a: must be valid JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gate 1 FAILED — invalid JSON: {exc}\nRaw: {raw}") from exc

    # Unwrap envelope patterns like {"transactions": [{...}]} or {"transaction": {...}}
    if isinstance(data, dict) and len(data) == 1:
        inner = next(iter(data.values()))
        if isinstance(inner, list) and inner:
            data = inner[0]
        elif isinstance(inner, dict):
            data = inner

    # Gate 1b: all required fields must be present
    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"Gate 1 FAILED — missing fields: {sorted(missing)}")

    # Gate 1c: amount must be numeric
    if not isinstance(data["amount"], (int, float)):
        raise ValueError(f"Gate 1 FAILED — 'amount' is not a number: {data['amount']!r}")

    print(f"  Gate 1 passed — fields: {list(data.keys())}")
    return data


# ---------------------------------------------------------------------------
# Step 2 — Enrich
# ---------------------------------------------------------------------------

def step_enrich(data: dict) -> dict:
    """Add category and risk fields based on amount."""
    print("\n[Step 2] Enriching with category and risk...")

    raw = _chat(
        FAST_MODEL,
        [
            {
                "role": "user",
                "content": (
                    "Given the transaction record below, add two fields:\n"
                    "  - category: one of micro (<50), regular (50-500), "
                    "significant (500-5000), major (>5000) — based on amount\n"
                    "  - risk: one of low (<1000), medium (1000-5000), high (>5000) — based on amount\n\n"
                    f"Record:\n{json.dumps(data, ensure_ascii=False)}\n\n"
                    "Reply with ONLY the complete JSON (original fields + new fields), "
                    "no markdown fences or explanation."
                ),
            }
        ],
        max_tokens=384,
    )
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # Gate 2a: must still be valid JSON
    try:
        enriched = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gate 2 FAILED — invalid JSON: {exc}") from exc

    # Gate 2b: category field must exist
    if "category" not in enriched:
        raise ValueError("Gate 2 FAILED — 'category' field not present in response")

    # Fallback: if the LLM omitted risk, compute it locally
    if "risk" not in enriched:
        enriched["risk"] = _classify(float(data["amount"]), RISK_THRESHOLDS)

    print(f"  Gate 2 passed — category={enriched['category']!r}, risk={enriched['risk']!r}")
    return enriched


# ---------------------------------------------------------------------------
# Step 3 — Format
# ---------------------------------------------------------------------------

def step_format(data: dict) -> str:
    """Generate a readable executive summary using the more capable model."""
    print("\n[Step 3] Formatting summary...")

    summary = _chat(
        CAPABLE_MODEL,
        [
            {
                "role": "system",
                "content": (
                    "You are a financial analyst assistant. Write clear, concise executive "
                    "summaries for transaction records. For medium or high risk transactions, "
                    "include a brief recommendation."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write an executive summary for the following transaction record. "
                    "Use plain prose with clear structure — do not output JSON.\n\n"
                    f"{json.dumps(data, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        max_tokens=512,
    )
    print(f"  Summary generated ({len(summary)} chars)")
    return summary


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(text: str) -> str:
    """Run the 3-step pipeline. Raises ValueError if any gate fails."""
    print(f"\n{'=' * 60}")
    print(f"Input: {text}")

    extracted = step_extract(text)
    enriched = step_enrich(extracted)
    summary = step_format(enriched)

    print(f"\n--- Final Summary ---\n{summary}\n")
    return summary


if __name__ == "__main__":
    # Test 1: well-formed transaction with clear numeric amount
    run_pipeline(
        "On April 3, 2025, Sarah Chen transferred $8,750 to Apex Cloud Solutions "
        "as payment for annual infrastructure services."
    )

    # Test 2: ambiguous text with a small informal purchase
    run_pipeline(
        "Yesterday, Miguel paid about twenty-two dollars for lunch at the downtown deli."
    )
