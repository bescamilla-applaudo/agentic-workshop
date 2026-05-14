# Agentic AI Workshop

AI-First workshop for software engineers who want to evolve into AI Engineering.

## What is this?

A hands-on workshop where you build agents, tools, and agentic flows. This is not a programming course — it's an **AI fluency** course: learning to design systems where AI is the main engine.

## Who is it for?

Software engineers from any stack who want to:
- Understand the foundational patterns of agentic systems
- Build tools (MCP) that any LLM can use
- Design agents that go beyond chatbots
- Prepare for the AI Engineer role

## What will you build?

| Phase | What you build |
|---|---|
| **1. Patterns** | 6 scripts: augmented LLM, prompt chain, router, parallelizer, orchestrator, eval loop |
| **2. Architectures** | 3 LangGraph agents: ReAct researcher, Reflection writer, PEV verifier |
| **3. MCP** | 1 tool server that works in Claude Desktop, VS Code Copilot, or Cursor |
| **4. Context Engineering** | Context files + tool definitions + permission models |
| **5. Production** | HITL, dry-run, LLM-as-Judge, production checklist |

## Prerequisites

- Know how to code (any language — exercises use Python as the vehicle)
- Python 3.11+ installed
- OpenRouter API key (free tier available — works with multiple model providers)
- VS Code with GitHub Copilot (recommended, not required)

## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API key
```

## Repo structure

```
├── INSTRUCTIONS.md          ← The workshop (start here)
├── AGENTIC.md               ← Full technical reference
├── PROGRESS.md              ← Your progress checklist
├── agentlab/                ← Your workspace (write code here)
└── solutions/               ← Reference solutions
    ├── fase1/               ← 6 implemented patterns
    ├── fase2/               ← 3 LangGraph agents
    ├── fase3/               ← MCP server
    ├── fase4/               ← Context engineering
    └── fase5/               ← Production patterns
```

## How to use this workshop

1. Open `INSTRUCTIONS.md` and follow the exercises in order
2. For each exercise, **ask Copilot** (or your preferred AI) to generate the code
3. Run, understand, iterate
4. Compare with `solutions/` when you want to validate
5. Track your progress in `PROGRESS.md`

## AI-First Methodology

> AI already knows how to code. Your job as an AI Engineer is to know **what to ask** and **how to evaluate it**.

Each exercise includes a suggested prompt for Copilot. Use it as a starting point, not a recipe. The learning is in understanding what was generated and why.
