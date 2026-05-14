"""
Exercise 5.2 — Dry-Run Harness: Simulate before acting.

Intercepts agent actions, logs them without executing,
and allows execution after approval.
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Action:
    """Represents an action the agent wants to execute."""
    name: str
    args: dict
    description: str


@dataclass
class DryRunHarness:
    """
    Intercepts and logs actions without executing them.
    After review, can execute them for real.
    """

    dry_run: bool = True
    actions_log: list[Action] = field(default_factory=list)
    _handlers: dict[str, Callable] = field(default_factory=dict)

    def register(self, name: str, handler: Callable) -> None:
        """Register a real handler for an action."""
        self._handlers[name] = handler

    def execute(self, name: str, description: str, **kwargs: Any) -> str:
        """
        Execute an action. In dry_run mode, only logs it.
        In real mode, executes it with the registered handler.
        """
        action = Action(name=name, args=kwargs, description=description)

        if self.dry_run:
            self.actions_log.append(action)
            print(f"  📝 [DRY-RUN] {name}: {description}")
            return f"[DRY-RUN] {name} logged (not executed)"

        handler = self._handlers.get(name)
        if not handler:
            return f"[ERROR] No handler registered for '{name}'"

        print(f"  ⚡ [REAL] {name}: {description}")
        return handler(**kwargs)

    def summary(self) -> str:
        """Summary of all actions logged in dry-run."""
        if not self.actions_log:
            return "No actions logged."

        lines = [f"📋 Dry-Run Summary: {len(self.actions_log)} actions"]
        for i, action in enumerate(self.actions_log, 1):
            lines.append(f"  {i}. [{action.name}] {action.description}")
            for key, val in action.args.items():
                lines.append(f"     {key}: {str(val)[:80]}")
        return "\n".join(lines)

    def approve_and_run(self) -> list[str]:
        """Execute all logged actions for real."""
        if not self.actions_log:
            print("   ⚠️ No actions to execute.")
            return []

        print(f"\n▶️  Executing {len(self.actions_log)} approved actions...")
        self.dry_run = False
        results = []

        for action in self.actions_log:
            result = self.execute(action.name, action.description, **action.args)
            results.append(result)

        self.dry_run = True
        self.actions_log.clear()
        return results


# ─── Real handlers (simulated for the workshop) ─────────────────────

def create_file_handler(path: str, content: str) -> str:
    print(f"     → File created: {path} ({len(content)} bytes)")
    return f"File created: {path}"


def http_request_handler(method: str, url: str, body: str = "") -> str:
    print(f"     → {method} {url}")
    return f"HTTP {method} {url} → 200 OK"


def run_command_handler(command: str) -> str:
    print(f"     → Executing: {command}")
    return f"Command executed: {command}"


# ─── Demo ─────────────────────────────────────────────────────────────

def demo():
    print(f"\n{'='*60}")
    print("🔧 Demo: DryRunHarness")

    harness = DryRunHarness(dry_run=True)

    # Register real handlers
    harness.register("create_file", create_file_handler)
    harness.register("http_request", http_request_handler)
    harness.register("run_command", run_command_handler)

    # Simulate an agent that wants to do 3 things
    print("\n📝 DRY-RUN mode:")
    harness.execute(
        "create_file",
        "Create configuration file",
        path="config/prod.yaml",
        content="database:\n  host: db.prod.internal\n  pool: 20",
    )
    harness.execute(
        "http_request",
        "Deploy to production",
        method="POST",
        url="https://deploy.internal/api/deploy",
        body='{"version": "2.1.0"}',
    )
    harness.execute(
        "run_command",
        "Run migrations",
        command="alembic upgrade head",
    )

    # Show summary
    print(f"\n{harness.summary()}")

    # Approve and execute
    print(f"\n{'='*60}")
    print("👤 Human reviews and approves...")
    results = harness.approve_and_run()

    print(f"\n✅ Results:")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    demo()
