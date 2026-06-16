"""
Exercise 1.2 — Prompt Chaining: Processing pipeline.

3-step pipeline with validation gates between each step.
Step 1: Extract entities → Step 2: Enrich → Step 3: Format.
"""

import json
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

FAST_MODEL = os.environ.get("FAST_MODEL", "google/gemma-4-31b-it:free")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")


def step_extract(text: str) -> dict:
    """Step 1: Extract entities from free text as JSON."""
    print("\n📋 Step 1: Extracting entities...")

    response = client.chat.completions.create(
        model=FAST_MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"""Extract the entities from this transaction as JSON.
Required fields: name (string), date (string YYYY-MM-DD), amount (number), description (string).
If any field is unclear, infer it from context.

Text: {text}

Reply ONLY with the JSON, no markdown or explanation.""",
            }
        ],
    )

    raw = response.choices[0].message.content.strip()
    print(f"   Raw: {raw[:100]}")

    # Gate: validate that it's valid JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gate 1 FAILED: Invalid JSON — {e}\nRaw: {raw}")

    required_fields = ["name", "date", "amount", "description"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Gate 1 FAILED: Missing fields — {missing}")

    print(f"   ✅ Gate 1 passed: Valid JSON with all fields")
    return data


def step_enrich(data: dict) -> dict:
    """Step 2: Enrich with category based on amount."""
    print("\n🏷️  Step 2: Enriching with category...")

    response = client.chat.completions.create(
        model=FAST_MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": f"""Given this transaction record:
{json.dumps(data, ensure_ascii=False)}

Add a "category" field based on the amount:
- amount < 50: "micro"
- amount 50-500: "regular"
- amount 500-5000: "significant"
- amount > 5000: "major"

Also add a "risk" field: "low" if < 1000, "medium" if 1000-5000, "high" if > 5000.

Reply ONLY with the complete JSON (original + new fields), no markdown.""",
            }
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Gate: validate that the field exists
    try:
        enriched = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gate 2 FAILED: Invalid JSON — {e}")

    if "category" not in enriched:
        raise ValueError(f"Gate 2 FAILED: 'category' field not found")

    print(f"   ✅ Gate 2 passed: category={enriched['category']}, risk={enriched.get('risk', 'N/A')}")
    return enriched


def step_format(data: dict) -> str:
    """Step 3: Format as readable summary using the more capable model."""
    print("\n📝 Step 3: Formatting summary...")

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"""Generate an executive summary of this transaction for a financial report.
Include relevant emojis, clear formatting, and a recommendation if the risk is medium or high.

Data:
{json.dumps(data, ensure_ascii=False, indent=2)}

Format: human-readable text (not JSON).""",
            }
        ],
    )

    summary = response.choices[0].message.content.strip()
    print(f"   ✅ Summary generated ({len(summary)} chars)")
    return summary


def run_pipeline(text: str) -> str:
    """Run the complete pipeline with gates."""
    print(f"\n{'='*60}")
    print(f"📥 Input: {text[:80]}...")

    try:
        # Step 1: Extract
        extracted = step_extract(text)

        # Step 2: Enrich
        enriched = step_enrich(extracted)

        # Step 3: Format
        summary = step_format(enriched)

        print(f"\n{'='*60}")
        print(f"✅ Pipeline completed successfully\n")
        print(summary)
        return summary

    except ValueError as e:
        print(f"\n❌ Pipeline stopped: {e}")
        return f"ERROR: {e}"


if __name__ == "__main__":
    # Test 1: Clear transaction
    run_pipeline(
        "On March 15, 2025, Juan Perez paid $3,200 for AI consulting services "
        "for his startup."
    )

    # Test 2: Transaction with ambiguous data
    run_pipeline(
        "Maria bought a coffee and a sandwich yesterday for twelve dollars at the corner cafe."
    )
