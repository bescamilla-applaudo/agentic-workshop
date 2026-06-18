"""
Exercise 1.1 — Augmented LLM: Assistant with tools.

An LLM with access to 2 tools (search_docs + calculate) that decides
when to use each one vs. answer directly.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Hardcoded FAQ knowledge base
FAQ = {
    "api_rate_limits": (
        "The API rate limit is 100 requests per minute per user. "
        "Contact the platform team to request an increase."
    ),
    "deployment": (
        "We deploy with GitHub Actions. The pipeline runs tests, builds the image, "
        "and deploys to AWS ECS. The main branch maps to production."
    ),
    "database": (
        "We use PostgreSQL 16 on RDS. The connection pool is capped at 20 per instance. "
        "Always use async sessions in service layer code."
    ),
    "auth": (
        "Authentication uses JWT. Access tokens expire in 1 hour, refresh tokens in 30 days. "
        "OAuth2 providers: Google and GitHub."
    ),
    "monitoring": (
        "Metrics live in Datadog. Alerts route through PagerDuty. "
        "Primary dashboard: https://app.datadoghq.com/dashboard/main"
    ),
    "on_call": (
        "On-call rotation is weekly, Monday to Monday. Escalation path: "
        "engineer → team lead → engineering manager. SLA is 15 min for P0."
    ),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search internal team documentation. Use this when the user asks about "
                "internal processes, infrastructure, policies, or team-specific configurations. "
                "Do NOT use for general programming or factual questions you already know."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword or short phrase to look up in the docs.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a numeric math expression and return the result. "
                "Use this whenever the user requests a calculation. "
                "Supports: +, -, *, /, ** (power), % (modulo), and parentheses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '(100 * 8) * 60'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def search_docs(query: str) -> str:
    """Search the FAQ dict for entries matching any word in the query."""
    words = query.lower().split()
    results = [
        f"[{key}]: {value}"
        for key, value in FAQ.items()
        if any(w in key or w in value.lower() for w in words)
    ]
    return "\n\n".join(results) if results else f"No documentation found for '{query}'."


def calculate(expression: str) -> str:
    """Safely evaluate a math expression using an allowlist of characters."""
    allowed = set("0123456789+-*/.() %")
    sanitized = expression.replace("**", "")
    if not all(c in allowed for c in sanitized):
        return f"Error: expression contains disallowed characters: '{expression}'"
    try:
        result = eval(expression)  # noqa: S307 — sanitized above
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


def _dispatch(tool_name: str, args: dict) -> str:
    if tool_name == "search_docs":
        return search_docs(args["query"])
    if tool_name == "calculate":
        return calculate(args["expression"])
    return f"Unknown tool: {tool_name}"


def run(user_message: str) -> str:
    """Run the augmented LLM tool-use loop and return the final answer."""
    print(f"\n{'=' * 60}")
    print(f"User: {user_message}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical assistant for a software engineering team. "
                "Use the available tools when the user asks about internal docs or needs a calculation. "
                "Answer directly for general questions."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        tools=TOOLS,
        messages=messages,
        max_tokens=1024,
    )

    # Agentic loop: keep executing tools until the model stops requesting them
    while response.choices[0].finish_reason == "tool_calls":
        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)

        for call in assistant_msg.tool_calls:
            args = json.loads(call.function.arguments)
            print(f"  Tool: {call.function.name}({json.dumps(args)})")
            result = _dispatch(call.function.name, args)
            print(f"  Result: {result[:120]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )

        response = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS,
            messages=messages,
            max_tokens=1024,
        )

    answer = response.choices[0].message.content or ""
    print(f"Assistant: {answer}")
    return answer


if __name__ == "__main__":
    # Test 1: Simple greeting — LLM answers directly, no tools
    run("Hello! What can you help me with?")

    # Test 2: Internal docs question — LLM uses search_docs
    run("What are our API rate limits and how do I request an increase?")

    # Test 3: Math calculation — LLM uses calculate
    run(
        "We have 8 instances each handling 100 requests per minute. "
        "How many total requests can we handle per hour?"
    )
