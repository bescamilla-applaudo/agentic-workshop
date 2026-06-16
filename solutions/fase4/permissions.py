"""
Exercise 4.3 — Permission Models: Autonomy levels.

Permission model with 3 presets (development, staging, production)
and check() method to decide whether to allow, ask, or deny.
"""

from dataclasses import dataclass
from enum import Enum


class Permission(Enum):
    ALLOW = "allow"   # Execute without asking
    ASK = "ask"       # Ask human for confirmation
    DENY = "deny"     # Forbidden, always reject


@dataclass
class PermissionModel:
    """Defines what an agent can do without asking."""

    name: str
    read: Permission    # Read workspace files
    edit: Permission    # Edit/create files
    bash: Permission    # Execute terminal commands
    web: Permission     # Make HTTP requests
    delegate: Permission  # Invoke sub-agents

    def check(self, action: str, context: str = "") -> tuple[Permission, str]:
        """
        Check if an action is allowed.

        Returns:
            (Permission, explanatory message)
        """
        action_lower = action.lower()

        # Map action to permission
        permission_map = {
            "read": self.read,
            "edit": self.edit,
            "write": self.edit,
            "create": self.edit,
            "bash": self.bash,
            "execute": self.bash,
            "run": self.bash,
            "web": self.web,
            "http": self.web,
            "fetch": self.web,
            "delegate": self.delegate,
            "subagent": self.delegate,
        }

        perm = permission_map.get(action_lower)
        if perm is None:
            return Permission.DENY, f"Unknown action: '{action}'. Denied by default."

        if perm == Permission.ALLOW:
            return perm, f"✅ {action} allowed in '{self.name}' mode."
        elif perm == Permission.ASK:
            return perm, f"⚠️ {action} requires confirmation in '{self.name}' mode. Context: {context}"
        else:
            return perm, f"🚫 {action} denied in '{self.name}' mode."

    def summary(self) -> str:
        """Human-readable summary of the permission model."""
        lines = [f"📋 Permissions: {self.name}"]
        for field in ["read", "edit", "bash", "web", "delegate"]:
            perm = getattr(self, field)
            icon = {"allow": "✅", "ask": "⚠️", "deny": "🚫"}[perm.value]
            lines.append(f"  {icon} {field}: {perm.value}")
        return "\n".join(lines)


# ─── Presets ──────────────────────────────────────────────────────────

DEVELOPMENT = PermissionModel(
    name="development",
    read=Permission.ALLOW,
    edit=Permission.ALLOW,
    bash=Permission.ALLOW,
    web=Permission.ALLOW,
    delegate=Permission.ALLOW,
)

STAGING = PermissionModel(
    name="staging",
    read=Permission.ALLOW,
    edit=Permission.ASK,
    bash=Permission.ASK,
    web=Permission.ASK,
    delegate=Permission.ASK,
)

PRODUCTION = PermissionModel(
    name="production",
    read=Permission.ALLOW,
    edit=Permission.DENY,
    bash=Permission.DENY,
    web=Permission.ASK,
    delegate=Permission.DENY,
)

PRESETS = {"development": DEVELOPMENT, "staging": STAGING, "production": PRODUCTION}


if __name__ == "__main__":
    # Demo: test all 3 presets with the same actions
    actions = [
        ("read", "src/config.py"),
        ("edit", "src/main.py — fixing bug #123"),
        ("bash", "npm run test"),
        ("web", "POST /api/deploy"),
        ("delegate", "code-review agent"),
    ]

    for preset_name, model in PRESETS.items():
        print(f"\n{'='*50}")
        print(model.summary())
        print()
        for action, context in actions:
            perm, msg = model.check(action, context)
            print(f"  {msg}")
