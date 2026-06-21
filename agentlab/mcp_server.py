"""
Exercise 3.1 — MCP Server: Your first tool server.

Primitives exposed:
  Tools     — search_notes(query), create_note(title, content)
  Resource  — notes://all  (all notes as JSON)
  Prompt    — summarize_notes(topic)  (ready-made user message)

Storage: data/notes.json — persists across restarts.
Seeded on first run from data/sample_notes.json if notes.json is absent.

Run:
  python agentlab/mcp_server.py
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT       = Path(__file__).parent.parent
DATA_FILE   = _ROOT / "data" / "notes.json"
SEED_FILE   = _ROOT / "data" / "sample_notes.json"

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load() -> list[dict]:
    """Load notes from disk; seed from sample_notes.json on first run."""
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    if SEED_FILE.exists():
        notes = json.loads(SEED_FILE.read_text(encoding="utf-8"))
        _save(notes)
        return notes
    return []


def _save(notes: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _next_id(notes: list[dict]) -> int:
    return max((n.get("id", 0) for n in notes), default=0) + 1


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

server = Server("notes-server")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_notes",
            description=(
                "Search notes by keyword. Returns all notes whose title or content "
                "contains the query (case-insensitive). Use this when the user asks "
                "to find, look up, or retrieve notes about a topic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword or phrase to search for in titles and content.",
                    }
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="create_note",
            description=(
                "Create a new note and persist it to disk. Use when the user asks to "
                "save, add, write, or create a note. Each note gets a unique id and "
                "an ISO-8601 timestamp."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short descriptive title for the note.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content of the note.",
                    },
                },
                "required": ["title", "content"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_notes":
        query = arguments["query"].lower()
        notes = _load()
        hits  = [
            n for n in notes
            if query in n.get("title", "").lower()
            or query in n.get("content", "").lower()
        ]
        if hits:
            body = json.dumps(hits, ensure_ascii=False, indent=2)
            return [TextContent(type="text", text=f"Found {len(hits)} note(s):\n{body}")]
        return [TextContent(type="text", text=f"No notes found matching '{arguments['query']}'.")]

    if name == "create_note":
        notes = _load()
        note  = {
            "id":         _next_id(notes),
            "title":      arguments["title"],
            "content":    arguments["content"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        notes.append(note)
        _save(notes)
        return [TextContent(
            type="text",
            text=f"Note created — id={note['id']}, title='{note['title']}'",
        )]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="notes://all",           # type: ignore[arg-type]
            name="All notes",
            description="Complete list of all stored notes as JSON.",
            mimeType="application/json",
        )
    ]


@server.read_resource()
async def read_resource(uri) -> str:
    if str(uri) == "notes://all":
        return json.dumps(_load(), ensure_ascii=False, indent=2)
    raise ValueError(f"Unknown resource URI: {uri}")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

@server.list_prompts()
async def list_prompts():
    return [
        {
            "name":        "summarize_notes",
            "description": "Generate a focused summary of the notes around a specific topic.",
            "arguments": [
                {
                    "name":        "topic",
                    "description": "The topic or theme to focus the summary on.",
                    "required":    True,
                }
            ],
        }
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None) -> GetPromptResult:
    if name == "summarize_notes":
        topic = (arguments or {}).get("topic", "general")
        notes = _load()
        notes_json = json.dumps(notes, ensure_ascii=False, indent=2)

        return GetPromptResult(
            description=f"Summary of notes on topic: {topic}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"You are a helpful assistant. "
                            f"Summarise the notes below, focusing on the topic '{topic}'. "
                            f"Be concise and highlight the most relevant insights.\n\n"
                            f"Notes:\n{notes_json}"
                        ),
                    ),
                )
            ],
        )

    raise ValueError(f"Unknown prompt: {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
