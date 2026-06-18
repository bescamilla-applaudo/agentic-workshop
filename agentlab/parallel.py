"""
Exercise 1.4 — Parallelization: Multi-perspective analysis.

Task A (Sectioning): analyze_parallel(text)
  — runs summary, sentiment, and keyword extraction simultaneously with asyncio.gather.
  — wall-clock ≈ slowest single call, not the sum of all three.

Task B (Voting): fact_check_vote(claim, n=3)
  — spawns N independent fact-check instances in parallel.
  — final verdict decided by majority; confidence = winning_votes / n.
"""

import asyncio
import json
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

FAST_MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

async def _chat(prompt: str, max_tokens: int = 256) -> str:
    resp = await client.chat.completions.create(
        model=FAST_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
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


# ---------------------------------------------------------------------------
# Task A — Sectioning (3 independent dimensions in parallel)
# ---------------------------------------------------------------------------

async def _summarize(text: str) -> str:
    return await _chat(
        f"Summarize the following text in 2-3 sentences. Reply with plain prose only.\n\n{text}",
        max_tokens=256,
    )


async def _sentiment(text: str) -> dict:
    raw = await _chat(
        'Analyze the sentiment of the text below.\n'
        'Reply with ONLY a JSON object (no markdown):\n'
        '{"sentiment": "positive|negative|neutral|mixed", "score": <-1.0 to 1.0>, "reason": "<one sentence>"}\n\n'
        f"Text: {text}",
        max_tokens=128,
    )
    try:
        return _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {"sentiment": "unknown", "score": 0.0, "reason": raw}


async def _keywords(text: str) -> list[str]:
    raw = await _chat(
        'Extract the 5 most relevant keywords from the text below.\n'
        'Reply with ONLY a JSON object (no markdown):\n'
        '{"keywords": ["k1", "k2", "k3", "k4", "k5"]}\n\n'
        f"Text: {text}",
        max_tokens=128,
    )
    try:
        return _parse_json(raw).get("keywords", [])
    except (json.JSONDecodeError, ValueError):
        return []


async def analyze_parallel(text: str) -> dict:
    """Analyze text across 3 dimensions simultaneously. Returns a structured dict."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {text[:80]}...")

    t0 = time.perf_counter()
    summary, sentiment, keywords = await asyncio.gather(
        _summarize(text),
        _sentiment(text),
        _keywords(text),
    )
    elapsed = time.perf_counter() - t0

    result = {"summary": summary, "sentiment": sentiment, "keywords": keywords}

    print(f"Completed in {elapsed:.2f}s (3 calls in parallel)")
    print(f"  summary   : {summary[:100]}")
    print(f"  sentiment : {sentiment}")
    print(f"  keywords  : {keywords}")
    return result


# ---------------------------------------------------------------------------
# Task B — Voting (N independent fact-check instances)
# ---------------------------------------------------------------------------

VERDICTS = frozenset({"TRUE", "FALSE", "UNCERTAIN"})


async def _single_check(claim: str, idx: int) -> str:
    """One independent fact-check instance. Returns the raw verdict string."""
    raw = await _chat(
        f'Verify whether the claim below is true or false based on your knowledge.\n'
        f'Reply with ONLY a JSON object (no markdown):\n'
        f'{{"verdict": "TRUE|FALSE|UNCERTAIN", "reason": "<one sentence>"}}\n\n'
        f'Claim: "{claim}"',
        max_tokens=128,
    )
    try:
        parsed = _parse_json(raw)
        verdict = parsed.get("verdict", "UNCERTAIN").upper()
        reason  = parsed.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        verdict, reason = "UNCERTAIN", raw

    if verdict not in VERDICTS:
        verdict = "UNCERTAIN"

    print(f"  voter {idx + 1}: {verdict:9s} — {reason[:80]}")
    return verdict


async def fact_check_vote(claim: str, n: int = 3) -> dict:
    """Verify a claim with N parallel instances. Majority wins."""
    print(f"\n{'=' * 60}")
    print(f"Claim ({n} voters): \"{claim}\"")

    verdicts = await asyncio.gather(*[_single_check(claim, i) for i in range(n)])

    tally: dict[str, int] = {"TRUE": 0, "FALSE": 0, "UNCERTAIN": 0}
    for v in verdicts:
        tally[v] += 1

    winner = max(tally, key=tally.__getitem__)
    confidence = round(tally[winner] / n, 2)

    result = {
        "claim":      claim,
        "verdict":    winner,
        "confidence": confidence,
        "tally":      tally,
    }
    print(f"  → {winner} (confidence {confidence:.0%})  tally={tally}")
    return result


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

async def main() -> None:
    # ── Task A: Sectioning ──────────────────────────────────────────
    await analyze_parallel(
        "Anthropic released Claude 3.5 Sonnet in mid-2024, setting a new benchmark "
        "for coding and reasoning tasks. Shortly after, OpenAI responded with o1, a "
        "model that uses chain-of-thought reasoning at inference time. The open-source "
        "community kept pace with Meta's Llama 3.1 405B, offering near-frontier "
        "performance without API costs. Enterprise adoption of LLMs accelerated "
        "significantly, though concerns about hallucinations and data privacy persisted."
    )

    # ── Task B: Voting ───────────────────────────────────────────────
    # True claim — expect unanimous or strong TRUE
    await fact_check_vote("Python was created by Guido van Rossum", n=3)

    # False claim — expect strong FALSE
    await fact_check_vote("The Great Wall of China is visible from the Moon with the naked eye", n=3)

    # Ambiguous claim — expect UNCERTAIN or split vote
    await fact_check_vote("Drinking coffee extends human lifespan", n=5)


if __name__ == "__main__":
    asyncio.run(main())
