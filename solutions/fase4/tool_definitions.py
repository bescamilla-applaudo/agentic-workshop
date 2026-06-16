"""
Exercise 4.2 — Tool Definitions: The art of describing tools.

3 tool definitions optimized so an LLM uses them correctly.
Each description explains when to use AND when NOT to use the tool.
"""

# ─── Tool 1: Knowledge base search ──────────────────────────────────
# WHY: The description explicitly says when to search vs. answer from memory.
# This prevents the LLM from searching things it already knows (wasting tokens and latency).

search_knowledge_base = {
    "name": "search_knowledge_base",
    "description": (
        "Search the team's internal documentation (runbooks, ADRs, postmortems, "
        "onboarding guides). USE IT when the user asks about internal processes, project-specific "
        "configuration, or past architecture decisions. DO NOT USE IT for general programming "
        "questions, CS concepts, or public information you already know — answer "
        "directly in those cases."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search term. Use domain-specific keywords, not full sentences. "
                    "E.g.: 'rate limit config' instead of 'how is rate limiting configured'."
                ),
            },
            "category": {
                "type": "string",
                "enum": ["runbook", "adr", "postmortem", "guide", "all"],
                "description": "Filter by document category. Use 'all' if unsure.",
            },
        },
        "required": ["query"],
    },
}

# ─── Tool 2: File writing ───────────────────────────────────────────
# WHY: The description specifies expected format and when to prefer this tool
# vs. simply showing the code in chat.

write_file = {
    "name": "write_file",
    "description": (
        "Write content to a workspace file. USE IT when the user asks to create or modify "
        "a specific file ('create X.py', 'update the README'). The content must be the COMPLETE "
        "file, not a diff or partial snippet. DO NOT USE IT to show code as an example or "
        "explanation — in those cases, show the code directly in chat."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path relative to workspace root. Must include extension. "
                    "E.g.: 'src/utils/helpers.py'. Don't use absolute paths."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "COMPLETE file content. Include imports, docstrings, and all code. "
                    "Don't use placeholders like '# ... rest of code'. If the file exists, "
                    "this content replaces it completely."
                ),
            },
        },
        "required": ["path", "content"],
    },
}

# ─── Tool 3: Record deletion (DANGEROUS) ────────────────────────────
# WHY: The description has explicit warnings, requires confirmation,
# and explains the consequences. This makes the LLM ask for confirmation
# before using it, even without explicit HITL.

delete_records = {
    "name": "delete_records",
    "description": (
        "⚠️ DESTRUCTIVE OPERATION — Deletes records from the database. This action is "
        "IRREVERSIBLE. ALWAYS ask for user confirmation before executing, showing "
        "exactly which records will be deleted and how many. USE ONLY when the "
        "user explicitly asks to delete data ('delete test users', "
        "'remove completed tasks'). NEVER use it as part of automatic cleanup "
        "or refactoring without the user requesting it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "table": {
                "type": "string",
                "description": "Table name. Only known tables: 'users', 'tasks', 'projects'.",
            },
            "filter": {
                "type": "object",
                "description": (
                    "Filter conditions as key-value pairs. E.g.: "
                    '{"status": "completed", "created_before": "2024-01-01"}. '
                    "At least one filter is REQUIRED — bulk deletion without filters is not allowed."
                ),
            },
            "confirm_count": {
                "type": "integer",
                "description": (
                    "Expected number of records to delete. The tool verifies this before executing. "
                    "If the actual count differs, the operation is aborted."
                ),
            },
        },
        "required": ["table", "filter", "confirm_count"],
    },
}


# ─── Summary ──────────────────────────────────────────────────────────

ALL_TOOLS = [search_knowledge_base, write_file, delete_records]

if __name__ == "__main__":
    import json
    for tool in ALL_TOOLS:
        print(f"\n{'='*60}")
        print(f"🔧 {tool['name']}")
        print(f"   {tool['description'][:100]}...")
        print(f"   Schema: {json.dumps(tool['input_schema'], indent=2)[:200]}")
