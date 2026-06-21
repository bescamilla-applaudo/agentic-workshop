"""
Exercise 4.3 — Permission Models: Autonomy levels.

Three presets (development / staging / production) that map each action type
to one of three verdicts:

  allow — execute without asking
  ask   — pause and request human confirmation
  deny  — reject immediately, no execution

The check() method returns a (verdict, reason) pair.
The gate() method enforces the verdict — raises PermissionError on deny,
returns a confirmation prompt string on ask, returns None on allow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Verdict = Literal["allow", "ask", "deny"]

# Actions that are destructive regardless of the configured permission level.
# A "bash: allow" preset still escalates these to "ask".
_DESTRUCTIVE_BASH_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\brm\b.*-[rf]",       # rm -rf
        r"\bdrop\b.*table",     # DROP TABLE
        r"\btruncate\b",        # TRUNCATE
        r"\bdelete\b.*where\b", # DELETE … WHERE (mass delete)
        r"\bformat\b",          # format disk
        r"\bdd\b.*if=",         # dd if=/dev/zero …
        r"\bgit\b.*--force",    # git push --force
        r">\s*/dev/sd",         # overwrite block device
    ]
]

# Web destinations that always require confirmation regardless of preset.
_SENSITIVE_WEB_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"localhost",           # internal services — confirm intent
        r"\.internal\b",       # private DNS zones
        r"/delete",            # destructive HTTP endpoints
        r"/purge",
        r"/drop",
    ]
]


def _is_destructive_bash(context: str) -> bool:
    return any(p.search(context) for p in _DESTRUCTIVE_BASH_PATTERNS)


def _is_sensitive_web(context: str) -> bool:
    return any(p.search(context) for p in _SENSITIVE_WEB_PATTERNS)


# ---------------------------------------------------------------------------
# PermissionModel
# ---------------------------------------------------------------------------

@dataclass
class PermissionModel:
    """
    Defines what a production agent may do at each autonomy level.

    Fields map to action categories:
      read     — read files, list directories, grep
      edit     — write, create, or delete files
      bash     — run shell commands
      web      — make outbound HTTP requests
      delegate — spawn sub-agents or call external LLM APIs
    """

    name:     str
    read:     Verdict
    edit:     Verdict
    bash:     Verdict
    web:      Verdict
    delegate: Verdict

    # Actions the model never permits regardless of field settings.
    # Add patterns here to lock down specific commands across all presets.
    always_deny: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------------------
    # Internal routing
    # ---------------------------------------------------------------------------

    # Maps user-facing action keywords to the five fields above.
    _ACTION_MAP: dict[str, str] = field(default_factory=lambda: {
        # read
        "read": "read", "ls": "read", "grep": "read", "find": "read", "cat": "read",
        # edit
        "edit": "edit", "write": "edit", "create": "edit", "delete": "edit",
        "save": "edit", "overwrite": "edit",
        # bash
        "bash": "bash", "run": "bash", "execute": "bash", "exec": "bash",
        "shell": "bash", "terminal": "bash",
        # web
        "web": "web", "http": "web", "fetch": "web", "request": "web",
        "curl": "web", "post": "web", "get": "web",
        # delegate
        "delegate": "delegate", "subagent": "delegate", "spawn": "delegate",
        "agent": "delegate",
    })

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def check(self, action: str, context: str = "") -> tuple[Verdict, str]:
        """
        Determine the verdict for (action, context).

        Context-aware escalation rules applied on top of the base field value:
          • A bash command matching a destructive pattern is escalated to "ask"
            even when bash=allow, and escalated to "deny" when bash=deny.
          • A web request to an internal/destructive URL is escalated to "ask"
            even when web=allow.
          • Any action in always_deny is rejected immediately.

        Returns:
            (verdict, human-readable reason)
        """
        action_key = action.lower().strip()

        # always_deny list — hard block regardless of anything else
        for blocked in self.always_deny:
            if blocked.lower() in action_key or blocked.lower() in context.lower():
                return (
                    "deny",
                    f"[{self.name}] '{blocked}' is on the always-deny list for this preset.",
                )

        # Resolve to a field name
        field_name = self._ACTION_MAP.get(action_key)
        if field_name is None:
            # Unknown action — deny by default (fail-safe)
            return (
                "deny",
                f"[{self.name}] Unknown action '{action}'. Denied by default (fail-safe).",
            )

        base: Verdict = getattr(self, field_name)

        # Context-aware escalation for bash
        if field_name == "bash" and context and _is_destructive_bash(context):
            if base == "allow":
                return (
                    "ask",
                    f"[{self.name}] bash=allow but '{context[:60]}' matches a destructive pattern — escalating to ask.",
                )
            if base == "deny":
                return (
                    "deny",
                    f"[{self.name}] bash=deny and '{context[:60]}' is also destructive — denied.",
                )

        # Context-aware escalation for web
        if field_name == "web" and context and _is_sensitive_web(context):
            if base == "allow":
                return (
                    "ask",
                    f"[{self.name}] web=allow but '{context[:60]}' targets a sensitive endpoint — escalating to ask.",
                )

        # Base verdict
        reasons: dict[Verdict, str] = {
            "allow": f"[{self.name}] {action} → allow",
            "ask":   f"[{self.name}] {action} requires confirmation. Context: {context or '(none)'}",
            "deny":  f"[{self.name}] {action} → deny (not permitted in this environment)",
        }
        return base, reasons[base]

    def gate(self, action: str, context: str = "") -> str | None:
        """
        Enforce the verdict:
          allow → return None (caller proceeds)
          ask   → return a confirmation prompt string (caller must display and wait)
          deny  → raise PermissionError immediately

        Usage:
            prompt = model.gate("bash", "rm -rf ./cache")
            if prompt:
                confirmed = await ask_human(prompt)
                if not confirmed:
                    return
            # proceed with action
        """
        verdict, reason = self.check(action, context)
        if verdict == "deny":
            raise PermissionError(reason)
        if verdict == "ask":
            return (
                f"Confirmation required\n"
                f"  Action  : {action}\n"
                f"  Context : {context or '(none)'}\n"
                f"  Reason  : {reason}\n"
                f"  Proceed? [y/N]"
            )
        return None  # allow

    def summary(self) -> str:
        """Single-glance table of all permission levels for this preset."""
        _icons: dict[Verdict, str] = {"allow": "✓ allow", "ask": "? ask  ", "deny": "✗ deny "}
        lines = [f"Preset: {self.name}"]
        lines.append("  Action    Verdict")
        lines.append("  --------- -------")
        for f in ("read", "edit", "bash", "web", "delegate"):
            v: Verdict = getattr(self, f)
            lines.append(f"  {f:<9} {_icons[v]}")
        if self.always_deny:
            lines.append(f"  always-deny: {self.always_deny}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

DEVELOPMENT = PermissionModel(
    name="development",
    # Full autonomy — agent works locally, mistakes are cheap to reverse.
    read=     "allow",
    edit=     "allow",
    bash=     "allow",
    web=      "allow",
    delegate= "allow",
)

STAGING = PermissionModel(
    name="staging",
    # Read freely; anything that mutates state or leaves the machine needs a nod.
    # Delegates are cheap to spawn but can incur costs — ask first.
    read=     "allow",
    edit=     "ask",
    bash=     "ask",
    web=      "ask",
    delegate= "ask",
)

PRODUCTION = PermissionModel(
    name="production",
    # Read-only autonomous access. Everything else is forbidden or requires
    # explicit human approval. Web is ask (not deny) because read-only GETs
    # to external APIs are common and low-risk.
    read=     "allow",
    edit=     "deny",
    bash=     "deny",
    web=      "ask",
    delegate= "deny",
    # Extra hard-blocks that apply even if someone accidentally sets bash=allow.
    always_deny=["rm -rf", "DROP TABLE", "git push --force"],
)

PRESETS: dict[str, PermissionModel] = {
    "development": DEVELOPMENT,
    "staging":     STAGING,
    "production":  PRODUCTION,
}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases: list[tuple[str, str]] = [
        ("read",    "src/config.py"),
        ("edit",    "src/api/routes/tasks.py"),
        ("bash",    "pytest tests/ -v"),
        ("bash",    "rm -rf ./dist"),            # destructive — should escalate
        ("web",     "https://api.github.com/repos/foo"),
        ("web",     "http://localhost:8080/delete"),  # sensitive — should escalate
        ("delegate","code-review sub-agent"),
        ("unknown", ""),                         # unknown action — should deny
    ]

    for preset_name, model in PRESETS.items():
        print(f"\n{'=' * 55}")
        print(model.summary())
        print()
        for action, ctx in test_cases:
            verdict, reason = model.check(action, ctx)
            label = {"allow": "ALLOW", "ask": "ASK  ", "deny": "DENY "}[verdict]
            print(f"  [{label}] {action:<10} {ctx[:45]}")
