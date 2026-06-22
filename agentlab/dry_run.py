"""
Exercise 5.2 — Dry-Run Harness: Simulate before acting.

DryRunHarness intercepts every agent action call:
  • dry_run=True  → logs the intent, returns a stub result, nothing touches disk/network
  • dry_run=False → dispatches to the registered real handler

Workflow:
  1. Instantiate with dry_run=True
  2. Run the agent — all actions are logged, nothing executes
  3. Call harness.summary() to show the human what WOULD happen
  4. Call harness.approve_and_run() to execute the same actions for real
     (or just discard the harness to cancel everything)
"""

from __future__ import annotations

import shlex
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

ActionStatus = Literal["pending", "success", "error", "skipped"]


@dataclass
class Action:
    """One intercepted agent action."""
    name:        str
    description: str
    kwargs:      dict[str, Any]
    status:      ActionStatus = "pending"
    result:      str          = ""

    def mark(self, status: ActionStatus, result: str) -> None:
        self.status = status
        self.result = result


# ---------------------------------------------------------------------------
# DryRunHarness
# ---------------------------------------------------------------------------

@dataclass
class DryRunHarness:
    """
    Intercept, log, and optionally execute agent actions.

    Usage:
        harness = DryRunHarness(dry_run=True)
        harness.register("create_file", real_create_file)

        # agent does its work, calling harness.execute(...)
        harness.execute("create_file", "Write config", path="x.yml", content="...")

        print(harness.summary())          # shows what would happen
        results = harness.approve_and_run()  # actually executes
    """

    dry_run: bool = True
    _log:      list[Action]            = field(default_factory=list, repr=False)
    _handlers: dict[str, Callable]     = field(default_factory=dict, repr=False)

    # ── Registration ───────────────────────────────────────────────────

    def register(self, name: str, handler: Callable) -> None:
        """Bind a real implementation to an action name."""
        self._handlers[name] = handler

    # ── Main intercept point ───────────────────────────────────────────

    def execute(self, name: str, description: str, **kwargs: Any) -> str:
        """
        Intercept an action.

        dry_run=True  → record the Action, print a stub line, return placeholder.
        dry_run=False → call the registered handler and record the real result.
        """
        action = Action(name=name, description=description, kwargs=kwargs)
        self._log.append(action)

        if self.dry_run:
            stub = f"[DRY-RUN] {name}: would {description.lower()}"
            action.mark("skipped", stub)
            print(f"  [DRY-RUN] {name:14s} — {description}")
            return stub

        handler = self._handlers.get(name)
        if handler is None:
            msg = f"[ERROR] no handler registered for '{name}'"
            action.mark("error", msg)
            print(f"  [ERROR]   {name:14s} — {msg}")
            return msg

        print(f"  [REAL]    {name:14s} — {description}")
        try:
            result = handler(**kwargs)
            action.mark("success", result)
            print(f"             ↳ {result}")
            return result
        except Exception as exc:
            msg = f"[ERROR] {exc}"
            action.mark("error", msg)
            print(f"             ↳ {msg}")
            return msg

    # ── Inspection ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of every action logged so far."""
        if not self._log:
            return "No actions logged."

        lines = [f"Dry-Run Summary — {len(self._log)} action(s) pending:"]
        lines.append("─" * 56)
        for i, a in enumerate(self._log, 1):
            lines.append(f"  {i}. [{a.name}] {a.description}")
            for k, v in a.kwargs.items():
                display = str(v)
                if len(display) > 80:
                    display = display[:77] + "..."
                lines.append(f"       {k}: {display}")
        lines.append("─" * 56)
        lines.append("Review the above. Call approve_and_run() to execute for real.")
        return "\n".join(lines)

    def report(self) -> str:
        """Post-execution report showing outcome of each action."""
        if not self._log:
            return "No actions recorded."

        status_icon = {"pending": "?", "success": "✓", "error": "✗", "skipped": "~"}
        lines = [f"Execution Report — {len(self._log)} action(s):"]
        lines.append("─" * 56)
        for i, a in enumerate(self._log, 1):
            icon = status_icon[a.status]
            lines.append(f"  {i}. [{icon}] [{a.name}] {a.description}")
            if a.result:
                lines.append(f"         {a.result[:100]}")
        return "\n".join(lines)

    # ── Execution ──────────────────────────────────────────────────────

    def approve_and_run(self) -> list[str]:
        """
        Execute all pending (skipped) actions for real, in logged order.
        Actions stay in the log with updated status/result so report() works
        immediately after. Call reset() to start a new dry-run cycle.
        """
        pending = [a for a in self._log if a.status == "skipped"]
        if not pending:
            print("Nothing to execute (log is empty or already run).")
            return []

        print(f"\nExecuting {len(pending)} approved action(s)...")
        results: list[str] = []

        for action in pending:
            handler = self._handlers.get(action.name)
            if handler is None:
                msg = f"[ERROR] no handler registered for '{action.name}'"
                action.mark("error", msg)
                print(f"  [ERROR]   {action.name:14s} — {msg}")
                results.append(msg)
                continue

            print(f"  [REAL]    {action.name:14s} — {action.description}")
            try:
                result = handler(**action.kwargs)
                action.mark("success", result)
                print(f"             ↳ {result}")
                results.append(result)
            except Exception as exc:
                msg = f"[ERROR] {exc}"
                action.mark("error", msg)
                print(f"             ↳ {msg}")
                results.append(msg)

        return results

    def reset(self) -> None:
        """Clear the action log to start a new dry-run cycle."""
        self._log.clear()


# ---------------------------------------------------------------------------
# Real action handlers
# ---------------------------------------------------------------------------

def real_create_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Created {path} ({len(content)} bytes)"


def real_http_request(method: str, url: str, body: str = "", timeout: int = 8) -> str:
    data = body.encode() if body else None
    req  = urllib.request.Request(url, data=data, method=method.upper())
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return f"HTTP {resp.status} {resp.reason} ← {method.upper()} {url}"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} {exc.reason} ← {method.upper()} {url}"
    except urllib.error.URLError as exc:
        return f"URL error: {exc.reason} ← {method.upper()} {url}"


def real_run_command(command: str, cwd: str | None = None) -> str:
    proc = subprocess.run(
        shlex.split(command),
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=30,
    )
    out = (proc.stdout or proc.stderr or "").strip()[:200]
    return f"exit={proc.returncode}  output: {out}" if out else f"exit={proc.returncode}"


# ---------------------------------------------------------------------------
# Example agent
# ---------------------------------------------------------------------------

def example_agent(harness: DryRunHarness) -> None:
    """
    A fictional deployment agent that does three things:
      1. Write a config file
      2. Ping a health-check URL
      3. Run a shell command
    The harness transparently intercepts all three.
    """
    harness.execute(
        "create_file",
        "Write production config to disk",
        path="/tmp/dry_run_demo/prod.yaml",
        content=(
            "database:\n"
            "  host: db.prod.internal\n"
            "  pool: 20\n"
            "cache:\n"
            "  host: redis.prod.internal\n"
            "  ttl: 300\n"
        ),
    )

    harness.execute(
        "http_request",
        "Ping public health-check endpoint",
        method="GET",
        url="https://httpbin.org/get",
    )

    harness.execute(
        "run_command",
        "List files in /tmp to verify the write",
        command="ls /tmp/dry_run_demo",
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Register real handlers once ────────────────────────────────────
    harness = DryRunHarness(dry_run=True)
    harness.register("create_file",  real_create_file)
    harness.register("http_request", real_http_request)
    harness.register("run_command",  real_run_command)

    # ── Phase 1: dry-run ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Phase 1 — DRY-RUN (nothing executes)")
    print("=" * 60)
    example_agent(harness)

    print(f"\n{harness.summary()}")

    # ── Phase 2: human approves ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Phase 2 — APPROVED: executing for real")
    print("=" * 60)
    harness.approve_and_run()

    # ── Phase 3: report ────────────────────────────────────────────────
    # Actions stay in the log with updated status — report() reads them directly.
    print(f"\n{harness.report()}")


if __name__ == "__main__":
    main()
