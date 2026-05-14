# Agentic AI Workshop — From Software Engineer to AI Engineer

> **Format:** AI-First workshop — ask AI to generate, run it, understand it, iterate.
> **Estimated duration:** 3-5 days (at your own pace).
> **Prerequisites:** Know how to code (any language). Python 3.11+ installed. OpenRouter API key (free tier available).
> **Technical reference:** All code details are in `AGENTIC.md` — this file is the exercise guide.

---

## What you'll learn

| Phase | Topic | What you produce |
|---|---|---|
| 1 | The 6 Foundational Patterns | 6 independent scripts, each solving a different problem |
| 2 | Agentic Architectures in LangGraph | 3 functional mini-agents with state graphs |
| 3 | MCP: Build Your Own Tools | 1 MCP server with tools + resources that works in any client |
| 4 | Context Engineering | Context files so any agent understands your project |
| 5 | Production Patterns | HITL, Dry-Run, LLM-as-Judge, production checklist |

---

## Initial Setup

### 1. Clone the repo and create the environment

```bash
cd agentic-workshop
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env with your key:
# OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Verify it works

```bash
python -c "from openai import OpenAI; print('✅ OpenAI SDK ready')"
```

> **Recommended setup for the workshop:** OpenRouter with free models like `nvidia/nemotron-3-super-120b-a12b:free` or `google/gemma-4-31b-it:free`. All solutions use the OpenAI SDK pointing to OpenRouter's API.

---

## Phase 1 — The 6 Foundational Patterns
**Goal:** Recognize which pattern to apply for each problem. Implement each one as an independent script.

> **Golden rule:** Start with the simplest approach. Add complexity *only when* there's evidence it improves measurable results.

📖 **Reference:** `AGENTIC.md → Phase 1` has the complete code for each pattern.

---

### Exercise 1.1 — Augmented LLM: Assistant with tools

**Concept:** The base LLM + retrieval + tools + memory = building block of everything else.

**Task:** Create a script with an LLM that has access to 2 tools:
- `search_docs(query)` — searches a local dictionary of "documentation" (hardcoded)
- `calculate(expression)` — evaluates a math expression

The LLM must decide when to use each tool vs. answer directly.

**Ask Copilot:**
```
Create a script in agentlab/augmented_llm.py that implements an Augmented LLM
with the OpenAI SDK (pointing to OpenRouter). It must have 2 tools: search_docs
(searches a hardcoded FAQ dict) and calculate (evaluates math). The LLM decides
whether to use tools or answer directly. Use nvidia/nemotron-3-super-120b-a12b:free.
Include 3 test prompts at the end.
Reference: AGENTIC.md section 1.1
```

**Validation:**
- [ ] The LLM answers simple questions directly ("hello")
- [ ] The LLM uses `search_docs` when you ask domain questions
- [ ] The LLM uses `calculate` when you ask for a calculation
- [ ] The script runs end-to-end without errors

**Output:** `agentlab/augmented_llm.py` → compare with `solutions/fase1/augmented_llm.py`

---

### Exercise 1.2 — Prompt Chaining: Processing pipeline

**Concept:** Break a complex task into sequential steps. Each LLM processes the previous one's output. Validation between steps (gates).

**Task:** Create a 3-step pipeline that processes free text:
1. **Extract** → Extract entities (name, date, amount) from text. Gate: validate that JSON is valid.
2. **Enrich** → Add category based on amount. Gate: verify the field exists.
3. **Format** → Generate a human-readable summary.

**Ask Copilot:**
```
Create agentlab/prompt_chain.py — a 3-step pipeline with OpenAI SDK (OpenRouter).
Input: free text with transaction data. Step 1: extract entities as JSON
(with gate that validates JSON). Step 2: enrich with category. Step 3: format
summary. Use a fast model for simple steps and a more capable one for final output.
Include 2 test inputs. Reference: AGENTIC.md section 1.2
```

**Validation:**
- [ ] The pipeline processes real text and produces structured JSON
- [ ] If Step 1 fails the gate (invalid JSON), the pipeline stops with a clear error
- [ ] Uses cheap models for simple steps and a more capable one for final output

**Output:** `agentlab/prompt_chain.py` → compare with `solutions/fase1/prompt_chain.py`

---

### Exercise 1.3 — Routing: Classify and direct

**Concept:** The LLM classifies the input and sends it to the most appropriate handler. Optimizes cost and quality.

**Task:** Create a router that classifies user queries into 4 categories and uses the right model/handler:
- `simple` → cheap model, direct response
- `complex` → mid-tier model, detailed response
- `critical` → powerful model, exhaustive response
- `unsafe` → direct rejection, no tokens spent

**Ask Copilot:**
```
Create agentlab/router.py — a routing system that classifies user inputs as
simple/complex/critical/unsafe and sends them to the appropriate model. Classification
with a fast model (cheap), handlers with fast/default/powerful depending on category.
Include 5 test inputs covering all 4 categories. Reference: AGENTIC.md section 1.3
```

**Validation:**
- [ ] "what time is it" → simple (fast model)
- [ ] "explain microservices architecture" → complex (default model)
- [ ] "design a high-availability payment system" → critical (powerful model)
- [ ] An adversarial prompt → unsafe (rejected)
- [ ] Total cost is optimized: doesn't use powerful model for trivial tasks

**Output:** `agentlab/router.py` → compare with `solutions/fase1/router.py`

---

### Exercise 1.4 — Parallelization: Multi-perspective analysis

**Concept:** Run independent subtasks in parallel (Sectioning) or the same task N times for consensus (Voting).

**Task A (Sectioning):** Analyze a text in 3 simultaneous dimensions: summary, sentiment, keywords. All in parallel with `asyncio`.

**Task B (Voting):** Verify a claim with 3 instances of the same model. Decide by majority.

**Ask Copilot:**
```
Create agentlab/parallel.py with 2 async functions:
1. analyze_parallel(text) — analyzes summary + sentiment + keywords in parallel
   using AsyncOpenAI (OpenRouter) and asyncio.gather
2. fact_check_vote(claim, n=3) — verifies a claim with N instances, decides by
   majority (TRUE/FALSE/UNCERTAIN with confidence %)
Use a fast model for all calls (speed). Include examples of both.
Reference: AGENTIC.md section 1.4
```

**Validation:**
- [ ] `analyze_parallel` returns all 3 dimensions in less time than sequential
- [ ] `fact_check_vote` returns consistent verdicts with confidence score
- [ ] You use `asyncio.gather` correctly

**Output:** `agentlab/parallel.py` → compare with `solutions/fase1/parallel.py`

---

### Exercise 1.5 — Orchestrator-Workers: Dynamic delegation

**Concept:** An LLM "director" breaks the task into parts, delegates to specialized workers, synthesizes the result.

**Task:** Create an orchestrator with 3 workers:
- `researcher` — searches and summarizes information
- `writer` — writes content
- `critic` — evaluates quality

The orchestrator dynamically decides which workers it needs and in what order.

**Ask Copilot:**
```
Create agentlab/orchestrator.py — an orchestrator-workers system with OpenAI SDK
(OpenRouter). The orchestrator receives a goal in natural language, decomposes it
into subtasks (JSON), assigns each to a specialized worker (researcher/writer/critic),
executes, and synthesizes the final result. Workers have different system prompts.
Include a test goal: "Write a technical blog post about prompt chaining".
Reference: AGENTIC.md section 1.5
```

**Validation:**
- [ ] The orchestrator generates a dynamic plan (not hardcoded)
- [ ] Each worker receives specific instructions from the orchestrator
- [ ] The final result integrates the work of all workers
- [ ] You can change the goal and the plan changes

**Output:** `agentlab/orchestrator.py` → compare with `solutions/fase1/orchestrator.py`

---

### Exercise 1.6 — Evaluator-Optimizer: Refinement loop

**Concept:** One agent generates, another evaluates with a score, the first one refines. Iterates until quality is reached.

**Task:** Create a Generate → Evaluate → Refine loop that:
1. Generates Python code for a given task
2. Evaluates the code on 5 criteria (score 0-1 each)
3. If total score < 0.8, refines with the feedback
4. Maximum 3 iterations

**Ask Copilot:**
```
Create agentlab/eval_optimizer.py — an evaluator-optimizer loop with OpenAI SDK
(OpenRouter). Generates Python code, evaluates it with 5 criteria (correctness,
readability, typing, error_handling, efficiency) each 0.0-1.0. If total score < 0.8,
the generator refines using the evaluator's feedback. Maximum 3 iterations. Show
progress for each iteration. Test input: "Write a function to merge two sorted lists".
Reference: AGENTIC.md section 1.6
```

**Validation:**
- [ ] The first generation has a score
- [ ] If the score is low, the generator refines with specific feedback
- [ ] The score improves between iterations (not always, but the trend)
- [ ] The loop ends when it reaches 0.8 or hits 3 iterations

**Output:** `agentlab/eval_optimizer.py` → compare with `solutions/fase1/eval_optimizer.py`

---

## Phase 2 — Agentic Architectures in LangGraph
**Goal:** Implement 3 advanced architectures as state graphs. Each produces a different functional agent.

📖 **Reference:** `AGENTIC.md → Phase 2` has the table of 17 architectures and code for ReAct, Reflection, and PEV.

> **Why LangGraph?** It's the current standard for stateful agent orchestration. It models the flow as a graph: nodes = functions, edges = transitions, state = shared context.

---

### Exercise 2.1 — ReAct Agent: Researcher with tools

**Concept:** The agent alternates Thought → Action → Observation until it solves the task. It's the most used pattern in production agents.

**Task:** Create a ReAct agent in LangGraph that can search for information on the internet (Tavily or DuckDuckGo) and answer multi-hop questions.

**Ask Copilot:**
```
Create agentlab/react_agent.py — a ReAct agent using LangGraph and LangChain.
State with messages + iteration count. Agent node (calls the LLM with tools),
tools node (executes the tools). Conditional edge: if there are tool_calls
→ tools, otherwise → END. Use ChatOpenAI with OpenRouter base_url and
DuckDuckGoSearchResults. Include a multi-hop test question.
Reference: AGENTIC.md section 2.1
```

**Validation:**
- [ ] The agent searches for info, reasons about results, and can search again
- [ ] It answers questions that require multiple searches
- [ ] The graph has exactly 2 nodes (agent, tools) + conditional edge
- [ ] It stops automatically when it has the answer

**Output:** `agentlab/react_agent.py` → compare with `solutions/fase2/react_agent.py`

---

### Exercise 2.2 — Reflection Agent: Writer with self-critique

**Concept:** Generate → Reflect → Refine loop as a state graph. The agent improves its own output until the reviewer approves it or iterations run out.

**Task:** Create an agent that writes and improves technical documentation:
1. `generate` node — writes or refines the draft
2. `reflect` node — critiques the draft with specific feedback
3. Conditional edge — if feedback says "APPROVED" or iteration ≥ 3, finish

**Ask Copilot:**
```
Create agentlab/reflection_agent.py — a reflection agent with LangGraph.
State: task, draft, feedback, iteration. Generate node (writes/refines),
reflect node (critiques with actionable feedback). If feedback contains "APPROVED"
or iteration >= 3 → END, otherwise → generate. Use ChatOpenAI with OpenRouter.
Test input: "Write documentation for a REST API rate limiter middleware".
Show the draft from each iteration.
Reference: AGENTIC.md section 2.2
```

**Validation:**
- [ ] The agent generates an initial draft
- [ ] The reflector gives specific feedback (not generic)
- [ ] The draft visibly improves between iterations
- [ ] Ends with "APPROVED" or upon reaching 3 iterations

**Output:** `agentlab/reflection_agent.py` → compare with `solutions/fase2/reflection_agent.py`

---

### Exercise 2.3 — PEV Agent: Plan → Execute → Verify

**Concept:** The agent first plans, then executes step by step, then verifies that the result meets the goal. If not, it can retry.

**Task:** Create a PEV agent that:
1. `plan` node — decomposes the goal into concrete steps
2. `execute` node — executes one step at a time using previous context
3. `verify` node — verifies that the whole result meets the goal
4. If verification fails → can go back to execute

**Ask Copilot:**
```
Create agentlab/pev_agent.py — a Plan-Execute-Verify agent with LangGraph.
State: goal, plan (list), current_step (int), step_results (list), verified (bool).
Plan node (generates list of steps as JSON), execute node (executes current step),
verify node (verifies complete result against goal). If current_step < len(plan)
→ execute, otherwise → verify. If verified → END, otherwise → execute.
Test input: "Create a comprehensive comparison of 3 Python web frameworks".
Reference: AGENTIC.md section 2.3
```

**Validation:**
- [ ] The agent generates a plan with 3-5 clear steps
- [ ] It executes each step sequentially, accumulating context
- [ ] The verifier evaluates the complete result
- [ ] If verification fails, it retries steps

**Output:** `agentlab/pev_agent.py` → compare with `solutions/fase2/pev_agent.py`

---

## Phase 3 — MCP: Build Your Own Tools
**Goal:** Understand the MCP standard, build a functional server, connect it to a real client.

📖 **Reference:** `AGENTIC.md → Phase 3` has the complete MCP server code and client configuration.

> **MCP = "USB-C for AI."** You build tools once, connect them to any client (Claude Desktop, VS Code Copilot, Cursor, OpenCode).

---

### Exercise 3.1 — MCP Server: Your first tool server

**Concept:** An MCP server exposes 3 primitives: **Tools** (executable actions), **Resources** (readable data), **Prompts** (reusable templates).

**Task:** Create a complete MCP server with:
- **Tool** `search_notes` — searches a local JSON file of notes
- **Tool** `create_note` — creates a new note (persists to JSON)
- **Resource** `notes://all` — exposes all notes as a readable resource
- **Prompt** `summarize_notes` — template that accepts `topic` and generates a summary

**Ask Copilot:**
```
Create agentlab/mcp_server.py — an MCP server using the official SDK (pip install mcp).
Use Server from mcp.server and stdio_server from mcp.server.stdio.
Tools: search_notes(query) searches in data/notes.json, create_note(title, content) adds
to the JSON. Resource: notes://all returns all notes. Prompt: summarize_notes with
topic argument. The JSON persists to disk.
Reference: AGENTIC.md section 3.2
```

**Validation:**
- [ ] The server starts without errors: `python agentlab/mcp_server.py`
- [ ] It has at least 2 tools, 1 resource, 1 prompt
- [ ] `create_note` persists data to disk (survives restart)
- [ ] `search_notes` searches existing notes

**Output:** `agentlab/mcp_server.py` → compare with `solutions/fase3/mcp_server.py`

---

### Exercise 3.2 — Connect to a real client

**Concept:** The MCP server is backend-agnostic. A single server works with any compatible client.

**Task:** Connect the MCP server from the previous exercise to at least one of these clients:

**Option A — VS Code Copilot** (already configured in `.vscode/mcp.json`):
```json
{
  "servers": {
    "my-notes": {
      "type": "stdio",
      "command": "python",
      "args": ["agentlab/mcp_server.py"]
    }
  }
}
```

**Option B — Claude Desktop** (`~/.config/claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "my-notes": {
      "command": "python",
      "args": ["/absolute/path/agentlab/mcp_server.py"]
    }
  }
}
```

**Validation:**
- [ ] The client detects the server's tools
- [ ] You can ask the LLM "search my notes about X" and it uses `search_notes`
- [ ] You can tell it "create a note about Y" and it uses `create_note`
- [ ] Notes persist between sessions

**Output:** Screenshot or log showing the integration working.

---

## Phase 4 — Context Engineering
**Goal:** Understand the universal principles that make an agent understand your project. It's not about a tool — it's about context design.

> **Key insight:** `.cursorrules`, `copilot-instructions.md`, `AGENTS.md`, `CLAUDE.md` are different implementations of the same principle: giving the agent the context it needs to work well on your project.

📖 **Reference:** `AGENTIC.md → Phases 4-5` has examples of each format.

---

### Exercise 4.1 — Context File: Your project in one file

**Concept:** A context file tells the agent: what the project is, what stack it uses, what conventions to follow, what not to do. It's the equivalent of an onboarding for a new developer.

**Task:** Create a context file for a real or fictional project. It must cover these 6 blocks:

```markdown
1. PROJECT — What it is and what it does (2-3 sentences)
2. STACK — Technologies, versions, tools
3. STRUCTURE — Where everything lives (key folders)
4. CONVENTIONS — Patterns to follow (naming, style, architecture decisions)
5. COMMANDS — How to run, test, deploy
6. DO NOT — What to never do (the "negative rules" are the most useful)
```

**Ask Copilot:**
```
Generate a sample context file in agentlab/context_example.md for a fictional project:
a task management API with Python (FastAPI + SQLAlchemy + pytest). It must cover all
6 blocks: PROJECT, STACK, STRUCTURE, CONVENTIONS, COMMANDS, DO NOT. Make it concrete
and actionable, not generic. An agent should be able to generate correct code just by
reading this file.
```

**Validation:**
- [ ] Anyone (or an agent) reading only that file can understand how to contribute to the project
- [ ] The conventions are specific (not "write good code" but "use AsyncSession, not Session")
- [ ] The DO NOT section has at least 5 concrete prohibitions
- [ ] The commands are copy-paste-ready

**Output:** `agentlab/context_example.md` → compare with `solutions/fase4/context_example.md`

---

### Exercise 4.2 — Tool Definitions: The art of describing tools

**Concept:** The quality of a tool definition determines whether the LLM uses it correctly. A bad `description` = the agent doesn't understand when to use it.

**Task:** Write 3 tool definitions (JSON Schema) for the following tools, optimizing `name`, `description`, and `input_schema` so an LLM uses them correctly:

1. **A search tool** — the LLM must know when to search vs. answer from memory
2. **A writing tool** — the LLM must know what format to use
3. **A dangerous tool** — the LLM must know when to ask for confirmation

**Ask Copilot:**
```
Create agentlab/tool_definitions.py with 3 tool definitions as Python dicts
(OpenAI compatible format). Each with name, description, and input_schema.
1. search_knowledge_base — for internal docs searches
2. write_file — for writing files (with path and content)
3. delete_records — for deleting DB records (dangerous, requires confirmation)
The descriptions must be clear enough that the LLM knows exactly when to use them and
when NOT to. Include comments explaining why each description is written that way.
```

**Validation:**
- [ ] Each description says when to use AND when not to use the tool
- [ ] The input_schemas have descriptions for each field
- [ ] The dangerous tool has clear warnings in its description
- [ ] An LLM reading only the definitions understands the context of each one

**Output:** `agentlab/tool_definitions.py` → compare with `solutions/fase4/tool_definitions.py`

---

### Exercise 4.3 — Permission Models: Autonomy levels

**Concept:** Every agentic system needs to decide how much the agent can do without asking. The 3 universal levels: `allow` (without asking), `ask` (asks for confirmation), `deny` (forbidden).

**Task:** Design a permission model for an agent that works on a production codebase. Define permissions for:
- Reading files
- Editing files
- Executing commands
- Making HTTP requests
- Calling subagents

**Ask Copilot:**
```
Create agentlab/permissions.py with a permission model for a production agent.
Use a dataclass PermissionModel with fields: read, edit, bash, web, delegate.
Each can be "allow", "ask", or "deny". Include 3 presets:
1. development — maximum autonomy (read: allow, edit: allow, bash: allow)
2. staging — medium autonomy (read: allow, edit: ask, bash: ask)
3. production — minimum autonomy (read: allow, edit: deny, bash: deny)
Include a check(action, context) method that decides whether to allow, ask, or deny.
```

**Validation:**
- [ ] The 3 presets represent real scenarios
- [ ] The `check()` method works correctly
- [ ] The production preset doesn't allow anything destructive without approval
- [ ] The model is extensible (you can add new permissions)

**Output:** `agentlab/permissions.py` → compare with `solutions/fase4/permissions.py`

---

## Phase 5 — Production Patterns
**Goal:** The patterns that separate a hobby agent from a production one. Security, reliability, observability, costs.

📖 **Reference:** `AGENTIC.md → Phase 9` has HITL, Dry-Run, LLM-as-Judge, and the complete checklist.

---

### Exercise 5.1 — Human-in-the-Loop: The agent that asks permission

**Concept:** For high-risk actions, the agent must pause and wait for human approval before executing.

**Task:** Create a LangGraph workflow with HITL:
1. The agent receives a task ("Delete all test users from production")
2. Generates an action plan
3. **Pauses** — shows the plan and waits for approval
4. If approved → executes. If rejected → cancels.

**Ask Copilot:**
```
Create agentlab/hitl_agent.py — a LangGraph workflow with Human-in-the-Loop.
State: task, plan, approved (bool), result. Plan node (generates action plan),
await_approval node (pauses with interrupt_before), execute node (executes if approved).
Use MemorySaver as checkpointer to pause/resume.
Show how to invoke, pause, approve, and resume.
Reference: AGENTIC.md section 9.2
```

**Validation:**
- [ ] The graph pauses before executing destructive actions
- [ ] You can inspect the plan before approving
- [ ] `graph.update_state()` with `approved: True` resumes execution
- [ ] Without approval, the action is not executed

**Output:** `agentlab/hitl_agent.py` → compare with `solutions/fase5/hitl_agent.py`

---

### Exercise 5.2 — Dry-Run Harness: Simulate before acting

**Concept:** In dry-run mode, the agent plans and simulates all actions without executing them. After human review, it executes for real.

**Task:** Create a harness that intercepts all agent actions and simulates them:

**Ask Copilot:**
```
Create agentlab/dry_run.py — a DryRunHarness that intercepts agent actions.
In dry_run=True mode, logs all actions without executing them. Generates a summary
of "what it would do". approve_and_run() method executes the real actions after
approval. Include an example agent that tries to: create a file, make an
HTTP request, and execute a command. Show dry-run vs real execution.
Reference: AGENTIC.md section 9.3
```

**Validation:**
- [ ] In dry-run mode, no action is actually executed
- [ ] The summary shows exactly what the agent would do
- [ ] `approve_and_run()` executes everything that was logged
- [ ] You can review the log before approving

**Output:** `agentlab/dry_run.py` → compare with `solutions/fase5/dry_run.py`

---

### Exercise 5.3 — LLM-as-a-Judge: Automated evaluation

**Concept:** Use an LLM to evaluate the quality of another LLM's outputs. Fundamental pattern for agent CI/CD.

**Task:** Create an evaluator that:
1. Receives a task and a response (generated by any LLM)
2. Evaluates on N criteria (each 1-5)
3. Returns scores, reasoning, and pass/fail verdict

**Ask Copilot:**
```
Create agentlab/llm_judge.py — an LLM-as-a-Judge with OpenAI SDK (OpenRouter).
Function llm_judge(task, response, criteria) that evaluates a response against criteria.
Each criterion score 1-5. Returns JSON with individual scores, total, reasoning, passed.
Include an example: evaluate generated code with criteria [correctness, readability,
typing, error_handling, efficiency]. Threshold: passed if total >= 4.0 average.
Reference: AGENTIC.md section 9.5
```

**Validation:**
- [ ] Each criterion has an individual score with reasoning
- [ ] The pass/fail verdict works with the threshold
- [ ] You can change criteria to evaluate anything (not just code)
- [ ] The judge is strict and specific in its feedback

**Output:** `agentlab/llm_judge.py` → compare with `solutions/fase5/llm_judge.py`

---

### Exercise 5.4 — Production Checklist: Audit your agent

**Concept:** Before deploying an agent, there's a security, reliability, observability, and cost checklist it must meet.

**Task:** Take any of the agents you built in previous phases and audit it against the production checklist:

```markdown
## Security
- [ ] Tools with minimum permissions (least privilege)
- [ ] Inputs sanitized before passing to tools
- [ ] Rate limiting on write tools
- [ ] Dry-Run for destructive actions
- [ ] HITL for critical changes

## Reliability
- [ ] Timeouts per tool
- [ ] Retry with exponential backoff
- [ ] Maximum iterations configured
- [ ] Persistent state (checkpointing)

## Observability
- [ ] Structured logs with correlation IDs
- [ ] Metrics: latency, error rate, tokens/request
- [ ] Traceability for each invocation

## Costs
- [ ] Right model per task (not powerful where cheap suffices)
- [ ] Budget per session/user
- [ ] Caching of repeated responses
```

**Ask Copilot:**
```
Analyze the agent in agentlab/react_agent.py (or whichever I choose) against the
production checklist from AGENTIC.md section 9.4. For each item it doesn't meet,
suggest the minimum necessary change. Don't rewrite everything — just patch what's missing.
```

**Validation:**
- [ ] You identified at least 5 gaps in the agent
- [ ] Each gap has a concrete solution (not "add error handling" but the actual code)
- [ ] The agent passes more checklist items after the patches

**Output:** A diff of the agent with the patches applied.

---

## Quick Reference: Which pattern for which problem?

| Problem | Pattern | Phase |
|---|---|---|
| Sequential task with validation | Prompt Chaining | 1 |
| Different inputs → different handlers | Routing | 1 |
| Fast multi-dimensional analysis | Parallelization (Sectioning) | 1 |
| High-confidence decision | Parallelization (Voting) | 1 |
| Complex task with dynamic delegation | Orchestrator-Workers | 1 |
| Generation with quality control | Evaluator-Optimizer | 1 |
| Research + iterative actions | ReAct | 2 |
| Generation with self-critique | Reflection | 2 |
| Planning → execution → verification | PEV | 2 |
| Standard tools for LLMs | MCP | 3 |
| Project context for agents | Context Files | 4 |
| Agent with production access | HITL + Dry-Run | 5 |
| Evaluate output quality | LLM-as-Judge | 5 |

---

## What's next after the workshop?

This workshop put you on the path. Next steps for an AI Engineer:

1. **Build a real agent** — take a problem from your work and apply the patterns
2. **Publish an MCP server** — on the MCP registry (`modelcontextprotocol.io`)
3. **Explore the 17 architectures** — the table in `AGENTIC.md → Phase 2` has the 11 we didn't cover
4. **Evaluate agents** — LangSmith, Langfuse, or your own LLM-as-Judge in CI/CD
5. **Read the papers** — "Building Effective Agents" (Anthropic), "ReAct" (Yao et al.), "Tree of Thoughts" (Yao et al.)
