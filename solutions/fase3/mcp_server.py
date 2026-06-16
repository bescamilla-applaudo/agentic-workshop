"""
Exercise 3.1 — MCP Server: Your first tool server.

MCP server with tools (search_notes, create_note), resource (notes://all),
and prompt (summarize_notes). Persists data in data/notes.json.
"""

import json
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    PromptArgument,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)

# ─── Data ─────────────────────────────────────────────────────────────

DATA_FILE = Path(__file__).parent.parent / "data" / "notes.json"


def load_notes() -> list[dict]:
    """Load notes from the JSON file."""
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return []


def save_notes(notes: list[dict]) -> None:
    """Save notes to the JSON file."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── Server ───────────────────────────────────────────────────────────

server = Server("notes-server")


# ─── Tools ────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_notes",
            description="Search notes that contain the search term in title or content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term (searches in title and content)",
                    }
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="create_note",
            description="Create a new note with title and content. Persisted to disk.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Note title",
                    },
                    "content": {
                        "type": "string",
                        "description": "Note content",
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
        notes = load_notes()
        matches = [
            n for n in notes
            if query in n["title"].lower() or query in n["content"].lower()
        ]
        if matches:
            result = json.dumps(matches, ensure_ascii=False, indent=2)
            return [TextContent(type="text", text=f"Found {len(matches)} notes:\n{result}")]
        return [TextContent(type="text", text=f"No notes found matching '{arguments['query']}'.")]

    elif name == "create_note":
        notes = load_notes()
        new_id = max((n["id"] for n in notes), default=0) + 1
        new_note = {
            "id": new_id,
            "title": arguments["title"],
            "content": arguments["content"],
        }
        notes.append(new_note)
        save_notes(notes)
        return [TextContent(type="text", text=f"Note created with ID {new_id}: '{arguments['title']}'")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ─── Resources ────────────────────────────────────────────────────────

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="notes://all",
            name="All notes",
            description="Complete list of all stored notes.",
            mimeType="application/json",
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    if str(uri) == "notes://all":
        notes = load_notes()
        return json.dumps(notes, ensure_ascii=False, indent=2)
    raise ValueError(f"Unknown resource: {uri}")


# ─── Prompts ──────────────────────────────────────────────────────────

@server.list_prompts()
async def list_prompts():
    return [
        {
            "name": "summarize_notes",
            "description": "Generate a summary of notes about a specific topic.",
            "arguments": [
                {
                    "name": "topic",
                    "description": "Topic to summarize notes about",
                    "required": True,
                }
            ],
        }
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None) -> GetPromptResult:
    if name == "summarize_notes":
        topic = (arguments or {}).get("topic", "general")
        notes = load_notes()
        notes_text = json.dumps(notes, ensure_ascii=False, indent=2)

        return GetPromptResult(
            description=f"Summary of notes about: {topic}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Summarize the following notes focusing on the topic '{topic}':\n\n{notes_text}",
                    ),
                )
            ],
        )

    raise ValueError(f"Unknown prompt: {name}")


# ─── Main ─────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
