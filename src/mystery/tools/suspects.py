"""Suspects tool: list every interrogable character. Free action (no turn cost).

The player needs this on turn 1 — they otherwise have no way to know which
suspect ids the ``ask`` and ``show`` verbs accept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mystery.models import CaseBible


def apply_suspects(bible: CaseBible) -> dict[str, Any]:
    lines = [f"  [{s.id}] {s.name} — {s.archetype}" for s in bible.suspects]
    text = "People in this house:\n" + "\n".join(lines)
    return {"last_output": text}
