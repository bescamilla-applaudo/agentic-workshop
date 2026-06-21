"""
Exercise 4.2 — Tool Definitions: The art of describing tools.

The description field is the primary signal the LLM uses to decide:
  (a) whether to call this tool at all
  (b) what arguments to pass
  (c) when NOT to call it

Bad descriptions produce tools that get called at the wrong time, with wrong
arguments, or get ignored entirely. Each definition below includes inline
comments explaining the deliberate choices.
"""

# ---------------------------------------------------------------------------
# Tool 1 — search_knowledge_base
# ---------------------------------------------------------------------------
#
# WHY the description is written this way:
#
# Problem: without explicit "when NOT to use" guidance, models default to
# searching for everything — including well-known CS concepts they could answer
# instantly. This wastes tokens, adds latency, and degrades UX.
#
# Solution: the description draws a hard line between internal knowledge
# (search) and public/general knowledge (answer directly). It also tells the
# model *how* to form queries — short keyword phrases, not full sentences —
# because that determines result quality more than the query content.
#
# The `scope` enum prevents free-text scope fields like "the architecture docs"
# (ambiguous) and forces the model to pick a known category, making routing
# on the backend predictable.

search_knowledge_base = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the team's INTERNAL documentation: runbooks, Architecture Decision "
            "Records (ADRs), incident postmortems, onboarding guides, and API contracts. "
            "USE THIS TOOL when the user asks about internal processes, team-specific "
            "configuration, past incidents, or project decisions that would not appear "
            "in public documentation. "
            # Explicit negative case — prevents the tool from being called for things
            # the model already knows, which is the most common over-use pattern.
            "DO NOT USE for general programming questions, well-known library APIs, "
            "CS concepts, or anything that public documentation covers — answer "
            "those directly from your own knowledge without calling this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    # Short keyword phrases return better results than natural language
                    # sentences in keyword/vector search. Spelling this out prevents the
                    # model from sending verbose queries like "how do I configure X in Y".
                    "description": (
                        "Keywords to search for. Use 2-5 domain-specific terms, NOT full "
                        "sentences. Good: 'postgres connection pool limit'. "
                        "Bad: 'how is the database connection pool configured'."
                    ),
                },
                "scope": {
                    "type": "string",
                    "enum": ["runbook", "adr", "postmortem", "guide", "api-contract", "all"],
                    # Enum over free-text so the backend can route to the right index.
                    # "all" is the safe default when the user hasn't specified a doc type.
                    "description": (
                        "Document category to restrict the search. Use a specific scope when "
                        "the user names a document type ('the runbook', 'past incidents'). "
                        "Default to 'all' when unsure."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    # Without a bound, models sometimes request absurd result counts.
                    "description": "Maximum number of results to return. Default 3; raise only if the user asks for a broad survey.",
                },
            },
            "required": ["query"],
            # scope and max_results are optional — the backend has sensible defaults.
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 2 — write_file
# ---------------------------------------------------------------------------
#
# WHY the description is written this way:
#
# Problem: models frequently show code in the chat response AND call write_file
# simultaneously, producing duplicate output. The other failure mode is calling
# write_file with partial/snippet content and leaving the file broken.
#
# Solution:
#   1. "USE when the user explicitly asks to create or update a file" — prevents
#      proactive writes that the user didn't request.
#   2. "DO NOT use to show code as an example" — draws the line between displaying
#      code (chat) and persisting code (tool).
#   3. "COMPLETE file content … no placeholders" — the most important constraint.
#      If the model writes "# ... existing code ..." the file is unusable.
#   4. Relative path with explicit format example prevents "/home/user/project/..."
#      absolute paths that break portability.
#   5. "If the file already exists, the entire content is replaced" — this warning
#      makes the model pause before overwriting something it shouldn't.

write_file = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write or overwrite a file in the project workspace. "
            "USE THIS TOOL when the user explicitly asks to create, update, or save "
            "a specific file ('create src/utils.py', 'update the README', "
            "'save this as a script'). "
            # Prevents the model from both displaying code AND writing it.
            "DO NOT USE to show code as an example or explanation — in that case "
            "include the code in your chat response instead. "
            # Prevents partial writes — the most common cause of broken files.
            "The content argument MUST be the complete, final file content. "
            "Never use placeholders like '# ... rest of code here ...' or "
            "'# existing implementation unchanged'. "
            # Overwrite warning prompts the model to verify intent before acting.
            "If the file already exists, its entire content is replaced with no backup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    # Relative path convention prevents absolute paths that break across
                    # machines. The extension requirement catches "create utils" typos.
                    "description": (
                        "Path relative to the project root, including file extension. "
                        "Examples: 'src/api/routes/tasks.py', 'tests/conftest.py', 'README.md'. "
                        "Do not use absolute paths or leading slashes."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The complete content to write. Must include every line of the file: "
                        "imports, docstrings, all functions, and closing newline. "
                        "No truncation, no ellipsis, no placeholders."
                    ),
                },
                "create_dirs": {
                    "type": "boolean",
                    # Optional quality-of-life flag — prevents 'directory not found' errors
                    # when writing to a new sub-path without requiring a separate tool call.
                    "description": (
                        "If true, create any missing parent directories automatically. "
                        "Default false. Set true when writing to a new sub-path that may not exist yet."
                    ),
                },
            },
            "required": ["path", "content"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 3 — delete_records
# ---------------------------------------------------------------------------
#
# WHY the description is written this way:
#
# Problem: deletion is irreversible. Two failure modes are catastrophic:
#   (a) the model deletes without asking — user loses data silently.
#   (b) the model deletes the wrong records because the filter was too broad.
#
# Solution — layered defence-in-depth:
#   1. The word IRREVERSIBLE in caps at the very start. Models treat prominent
#      words as strong signals. This triggers the "pause and confirm" behaviour
#      trained into most safety-tuned models.
#   2. "ALWAYS ask for explicit confirmation" — makes confirmation mandatory,
#      not optional. The model cannot claim it "assumed approval".
#   3. "NEVER infer permission" — closes the gap between "the user mentioned
#      cleaning up" (casual) and "the user told me to delete" (explicit).
#   4. `filter` is required (no empty filter allowed at schema level) — prevents
#      accidental full-table wipes from a missing where-clause.
#   5. `expected_count` is a pre-flight check: the backend runs a COUNT query
#      first and aborts if actual != expected. Even a correct filter + wrong
#      expectation catches accidents.
#   6. `dry_run` flag lets the model show the user what WOULD be deleted before
#      doing it — a built-in preview without needing a separate tool.
#   7. `reason` creates an audit trail and forces the model to articulate why
#      it is deleting, which surfaces mistakes before they happen.

delete_records = {
    "type": "function",
    "function": {
        "name": "delete_records",
        "description": (
            "⚠️  IRREVERSIBLE — permanently deletes database records. "
            "ALWAYS ask for explicit user confirmation before calling this tool, "
            "showing the exact filter conditions and the expected number of rows affected. "
            "USE THIS TOOL only when the user has directly and unambiguously requested "
            "a deletion ('delete all test users', 'remove tasks completed before January'). "
            # Prevents the model from acting on implied or inferred intent.
            "NEVER infer permission to delete from phrases like 'clean up', 'tidy', "
            "'remove old stuff', or similar — those require clarification first. "
            # Prevents using this tool as part of larger automated flows.
            "Do not call this tool as a step inside a larger autonomous workflow unless "
            "the user has approved the full plan including this step explicitly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    # Enum over free-text — prevents SQL injection via table name and
                    # constrains the model to known, safe targets.
                    "enum": ["users", "tasks", "projects", "labels", "assignments"],
                    "description": "The table to delete from. Only the listed tables are permitted.",
                },
                "filter": {
                    "type": "object",
                    # Required at schema level — the backend will reject calls without it,
                    # and the schema-level requirement surfaces this to the model before the call.
                    "description": (
                        "WHERE conditions as key-value pairs. ALL conditions are ANDed together. "
                        "At least one condition is required — an empty filter object {} is rejected. "
                        "Supported operators via key suffix: "
                        "'status' (exact match), 'created_before' (ISO date, <), "
                        "'created_after' (ISO date, >), 'id_in' (array of IDs). "
                        "Example: {\"status\": \"archived\", \"created_before\": \"2024-01-01\"}"
                    ),
                    "minProperties": 1,
                },
                "expected_count": {
                    "type": "integer",
                    "minimum": 1,
                    # The backend runs COUNT(*) with the filter first. If actual != expected,
                    # the delete is aborted and an error is returned. This is the last safety net.
                    "description": (
                        "The exact number of rows you expect this filter to delete. "
                        "The backend verifies this with a COUNT query before executing. "
                        "If the actual count differs, the operation is aborted. "
                        "Obtain this number by asking the user or calling search first."
                    ),
                },
                "dry_run": {
                    "type": "boolean",
                    # Dry-run lets the model show the user what would be deleted before committing.
                    # Surfacing this in the schema encourages its use as a first step.
                    "description": (
                        "If true, return the records that WOULD be deleted without actually deleting them. "
                        "Use this first to show the user what will be affected, then call again "
                        "with dry_run=false after confirmation. Default false."
                    ),
                },
                "reason": {
                    "type": "string",
                    # A required audit field. Forcing the model to articulate the reason
                    # surfaces intent — if the reason sounds wrong, the model may reconsider.
                    "description": (
                        "One sentence explaining why these records are being deleted. "
                        "Written to the audit log. Example: 'Removing test accounts created "
                        "during load testing on 2024-03-15 per request from @alice'."
                    ),
                },
            },
            "required": ["table", "filter", "expected_count", "reason"],
        },
    },
}


# ---------------------------------------------------------------------------
# Exported list — pass directly to the `tools` parameter of any OpenAI call
# ---------------------------------------------------------------------------

ALL_TOOLS = [search_knowledge_base, write_file, delete_records]


if __name__ == "__main__":
    import json

    for tool in ALL_TOOLS:
        fn = tool["function"]
        print(f"\n{'=' * 60}")
        print(f"Tool    : {fn['name']}")
        print(f"Desc    : {fn['description'][:120].rstrip()}...")
        required = fn["parameters"].get("required", [])
        optional = [k for k in fn["parameters"]["properties"] if k not in required]
        print(f"Required: {required}")
        print(f"Optional: {optional}")
