"""
Exercise 2.1 — ReAct Agent: Researcher with tools.

Graph topology (2 nodes):

    ┌──────┐   tool_calls?   ┌───────┐
    │agent │ ─── yes ──────► │ tools │
    │      │ ◄───────────────│       │
    └──────┘                 └───────┘
        │
        └── no tool_calls → END

State: messages (append-only via add_messages) + iteration counter.
The iteration counter acts as a safety brake (MAX_ITERATIONS).
"""

import concurrent.futures
import logging
import os
import re
import time
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_community.tools import DuckDuckGoSearchResults  # noqa: F401 — deprecation warning is cosmetic
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# Patch 8: avoid duplicate API calls for identical prompts
set_llm_cache(InMemoryCache())

# Patch 5: structured logging with timestamps
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("react_agent")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages:  Annotated[list[BaseMessage], add_messages]
    iteration: int


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Patch 1: sanitize inputs before they reach the LLM
def _sanitize(text: str, max_len: int = 500) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text[:max_len]


_raw_search = DuckDuckGoSearchResults(max_results=4)


# Patch 2: 10-second timeout on every search call
@tool
def search(query: str) -> str:
    """Search the web for current information. Use concise keyword queries."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_raw_search.run, query)
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            logger.warning("search timed out query=%r", query[:80])
            return "Search timed out after 10 seconds. Try a narrower query."


tools  = [search]

# ---------------------------------------------------------------------------
# LLM  — ChatOpenAI pointed at OpenRouter
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model=os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    max_tokens=1024,
)
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 6


def agent_node(state: AgentState) -> dict:
    """Call the LLM; may emit tool_calls or a final answer. Retries up to 3x."""
    iteration = state.get("iteration", 0) + 1
    logger.info("agent iteration=%d", iteration)

    # Patch 3: retry with exponential backoff
    for attempt in range(3):
        try:
            response = llm_with_tools.invoke(state["messages"])
            break
        except Exception as exc:
            if attempt == 2:
                raise
            wait = 2 ** attempt
            logger.warning("LLM failed attempt=%d %s retrying in %ds", attempt + 1, exc, wait)
            time.sleep(wait)

    # Patch 7: log token usage when the model reports it
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        logger.info("tokens input=%s output=%s", um.get("input_tokens", "?"), um.get("output_tokens", "?"))

    return {"messages": [response], "iteration": iteration}


tool_node = ToolNode(tools)

# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    last     = state["messages"][-1]
    iteration = state.get("iteration", 0)

    if iteration >= MAX_ITERATIONS:
        logger.info("router max_iterations=%d → END", MAX_ITERATIONS)
        return END

    if getattr(last, "tool_calls", None):
        names = [tc["name"] for tc in last.tool_calls]
        logger.info("router tool_calls=%s → tools", names)
        return "tools"

    logger.info("router no tool_calls → END")
    return END

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "agent")

# Patch 4: MemorySaver checkpoints state so mid-run crashes don't lose progress
memory = MemorySaver()
graph  = builder.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_agent(question: str, thread_id: str | None = None) -> str:
    """Invoke the ReAct graph and return the final answer text."""
    # Patches 1, 5, 6: sanitize input; assign run_id for traceability; time the run
    question  = _sanitize(question)
    run_id    = str(uuid.uuid4())[:8]
    thread_id = thread_id or f"thread-{run_id}"
    config    = {"configurable": {"thread_id": thread_id}}

    logger.info("run_start run_id=%s question=%r", run_id, question[:100])
    t0 = time.perf_counter()

    final_state = graph.invoke(
        {"messages": [("user", question)], "iteration": 0},
        config=config,
    )

    elapsed = time.perf_counter() - t0
    logger.info("run_end run_id=%s elapsed=%.2fs iterations=%d", run_id, elapsed, final_state["iteration"])

    answer = final_state["messages"][-1].content
    print(f"\nAnswer:\n{answer}")
    return answer


# ---------------------------------------------------------------------------
# Test — multi-hop question (requires ≥2 searches to answer)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_agent(
        "What programming language was used to build the LangGraph library, "
        "and who created the company behind it?"
    )
