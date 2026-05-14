"""
Exercise 1.4 — Parallelization: Multi-perspective analysis.

Task A: Sectioning — analyzes text in 3 dimensions in parallel.
Task B: Voting — verifies a claim with N instances, decides by majority.
"""

import asyncio
import json
import os

from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = os.environ.get("FAST_MODEL", "google/gemma-4-31b-it:free")


# ─── Task A: Sectioning ─────────────────────────────────────────────

async def analyze_summary(text: str) -> str:
    """Generate a summary of the text."""
    response = await client.chat.completions.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": f"Summarize this text in 2-3 sentences:\n\n{text}"}],
    )
    return response.choices[0].message.content


async def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of the text."""
    response = await client.chat.completions.create(
        model=MODEL,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": f'Analyze the sentiment of this text. Reply ONLY with JSON: {{"sentiment": "positive|negative|neutral|mixed", "score": -1.0 to 1.0, "reason": "..."}}\n\nText: {text}',
            }
        ],
    )
    return response.choices[0].message.content


async def analyze_keywords(text: str) -> str:
    """Extract keywords from the text."""
    response = await client.chat.completions.create(
        model=MODEL,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": f'Extract the 5 most relevant keywords from this text. Reply ONLY with JSON: {{"keywords": ["k1", "k2", ...]}}\n\nText: {text}',
            }
        ],
    )
    return response.choices[0].message.content


async def analyze_parallel(text: str) -> dict:
    """Analyze a text in 3 simultaneous dimensions using asyncio.gather."""
    print(f"\n{'='*60}")
    print(f"📊 Parallel analysis of: {text[:60]}...")

    import time
    start = time.time()

    summary, sentiment, keywords = await asyncio.gather(
        analyze_summary(text),
        analyze_sentiment(text),
        analyze_keywords(text),
    )

    elapsed = time.time() - start
    print(f"⏱️  Completed in {elapsed:.2f}s (3 calls in parallel)")

    result = {
        "summary": summary,
        "sentiment": sentiment,
        "keywords": keywords,
    }

    for key, value in result.items():
        print(f"\n  📌 {key}: {value[:100]}")

    return result


# ─── Task B: Voting ─────────────────────────────────────────────────

async def single_fact_check(claim: str, instance_id: int) -> str:
    """One instance of the fact checker."""
    response = await client.chat.completions.create(
        model=MODEL,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": f"""Verify if this claim is true or false.
Reply ONLY with JSON: {{"verdict": "TRUE|FALSE|UNCERTAIN", "reason": "..."}}

Claim: "{claim}" """,
            }
        ],
    )
    raw = response.choices[0].message.content.strip()
    print(f"   🗳️  Instance {instance_id}: {raw[:80]}")
    return raw


async def fact_check_vote(claim: str, n: int = 3) -> dict:
    """Verify a claim with N independent instances. Decide by majority."""
    print(f"\n{'='*60}")
    print(f"🗳️  Fact-checking with {n} instances: '{claim}'")

    tasks = [single_fact_check(claim, i + 1) for i in range(n)]
    results = await asyncio.gather(*tasks)

    # Count votes
    votes = {"TRUE": 0, "FALSE": 0, "UNCERTAIN": 0}
    for raw in results:
        try:
            parsed = json.loads(raw)
            verdict = parsed.get("verdict", "UNCERTAIN").upper()
            if verdict in votes:
                votes[verdict] += 1
            else:
                votes["UNCERTAIN"] += 1
        except json.JSONDecodeError:
            votes["UNCERTAIN"] += 1

    # Majority
    total = sum(votes.values())
    winner = max(votes, key=votes.get)
    confidence = votes[winner] / total

    result = {
        "claim": claim,
        "verdict": winner,
        "confidence": round(confidence, 2),
        "votes": votes,
    }

    print(f"\n   📊 Result: {winner} (confidence: {confidence:.0%})")
    print(f"   Votes: {votes}")
    return result


async def main():
    # Task A: Sectioning
    text = (
        "OpenAI launched GPT-5 in 2025, marking a significant leap in reasoning. "
        "However, the open source community responded quickly with models like "
        "Llama 4 from Meta and Mistral Large, which offer competitive performance at a "
        "fraction of the cost. The race for AGI intensifies while European regulators "
        "propose new restrictions."
    )
    await analyze_parallel(text)

    # Task B: Voting
    await fact_check_vote("Python was created by Guido van Rossum in 1991")
    await fact_check_vote("The Moon has greater gravity than the Earth")


if __name__ == "__main__":
    asyncio.run(main())
