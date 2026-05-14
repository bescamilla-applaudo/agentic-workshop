"""
Exercise 1.1 — Augmented LLM: Assistant with tools.

An LLM with access to 2 tools (search_docs + calculate) that decides
when to use each one vs. answer directly.
"""

import json
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

# Local knowledge base (simulated)
KNOWLEDGE_BASE = {
    "api_rate_limits": "The API rate limit is 100 requests per minute per user. To increase it, contact the platform team.",
    "deployment": "We deploy with GitHub Actions. The pipeline runs tests, build, and deploy to AWS ECS. Branch main = production.",
    "database": "We use PostgreSQL 16 on RDS. The connection pool is configured at 20 max per instance.",
    "auth": "Authentication with JWT. Tokens expire in 1 hour. Refresh tokens in 30 days. OAuth2 with Google and GitHub.",
    "monitoring": "We use Datadog for metrics and PagerDuty for alerts. Main dashboard: https://app.datadoghq.com/dashboard/main",
}

# Tool definitions for OpenAI format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the team's internal documentation. Use this tool when "
                "the user asks about internal processes, configuration, or team policies. "
                "DO NOT use it for general programming questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term. Can be a keyword or short phrase.",
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
                "Evaluates a math expression and returns the result. Use this tool "
                "when the user requests numeric calculations. Supports basic operations: +, -, *, /, **, %."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g. '2 + 3 * 4').",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def search_docs(query: str) -> str:
    """Search the local knowledge base."""
    query_lower = query.lower()
    results = []
    for key, value in KNOWLEDGE_BASE.items():
        if any(word in key or word in value.lower() for word in query_lower.split()):
            results.append(f"[{key}]: {value}")
    if results:
        return "\n\n".join(results)
    return f"No results found for '{query}'."


def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() %")
    if not all(c in allowed for c in expression.replace("**", "")):
        return f"Error: Invalid expression '{expression}'. Only numeric operations are allowed."
    try:
        result = eval(expression)  # noqa: S307 — input sanitized above
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Dispatch the call to the correct tool."""
    if tool_name == "search_docs":
        return search_docs(tool_input["query"])
    elif tool_name == "calculate":
        return calculate(tool_input["expression"])
    return f"Unknown tool: {tool_name}"


def run_augmented_llm(user_message: str) -> str:
    """Run the Augmented LLM with tool-use loop."""
    print(f"\n{'='*60}")
    print(f"👤 User: {user_message}")

    messages = [
        {"role": "system", "content": "You are a technical assistant for an engineering team. Use the tools when necessary."},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )

    # Tool-use loop: the LLM can call tools multiple times
    while response.choices[0].finish_reason == "tool_calls":
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_input = json.loads(tool_call.function.arguments)
            print(f"🔧 Tool: {tool_call.function.name}({json.dumps(tool_input)})")
            result = process_tool_call(tool_call.function.name, tool_input)
            print(f"   → {result[:100]}...")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

    # Extract final text
    final_text = response.choices[0].message.content or ""
    print(f"🤖 Response: {final_text}")
    return final_text


if __name__ == "__main__":
    # Test 1: Simple question (no tools)
    run_augmented_llm("Hi, how are you?")

    # Test 2: Question that requires search_docs
    run_augmented_llm("What is our API rate limit?")

    # Test 3: Question that requires calculate
    run_augmented_llm("If we have 100 requests per minute and 8 instances, how many total requests per hour do we support?")
