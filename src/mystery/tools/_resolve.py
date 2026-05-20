"""Lenient suspect-reference resolution.

Players naturally reach for first names ("Eleanor"), last names ("Vance"),
or archetype nouns ("servant", "butler") instead of the snake_case ids we
use internally. The strict equality lookup that the tools used to do made
that frustrating — especially when the displayed archetype ("The Nervous
Servant") diverges from the id ("housekeeper"). This helper centralises a
single resolution policy so every suspect-taking tool resolves the same
way and the same error surface points the player back at the canonical
roster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mystery.models import CaseBible, Suspect


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "mr",
        "mrs",
        "ms",
        "miss",
        "dr",
        "sir",
        "lord",
        "lady",
    }
)


def _tokens(text: str) -> set[str]:
    """Lowercase tokens, with stopwords and 1-char noise dropped.

    Splits on whitespace AND on hyphens/underscores so archetypes like
    ``"estranged-niece"`` and ids like ``"old_groundskeeper"`` decompose
    into the words a player is likely to type.
    """
    cleaned = text.lower()
    for sep in (".", ",", "-", "_", "/"):
        cleaned = cleaned.replace(sep, " ")
    return {t for t in cleaned.split() if len(t) > 1 and t not in _STOPWORDS}


def resolve_suspect(bible: CaseBible, ref: str) -> Suspect | None:
    """Match a player-supplied suspect reference against the roster.

    Resolution order:
        1. Exact id (case-insensitive).
        2. Exact full ``name`` (case-insensitive).
        3. Token overlap with id, name, or archetype — but only if exactly
           one suspect matches. Ambiguous matches return ``None`` so the
           caller can surface a "did you mean…" listing.
    """
    ref_clean = ref.strip().lower()
    if not ref_clean:
        return None
    for s in bible.suspects:
        if s.id.lower() == ref_clean:
            return s
    for s in bible.suspects:
        if s.name.lower() == ref_clean:
            return s
    ref_tokens = _tokens(ref)
    if not ref_tokens:
        return None
    matches: list[Suspect] = []
    for s in bible.suspects:
        haystack = _tokens(s.name) | _tokens(s.archetype) | {s.id.lower()}
        if ref_tokens & haystack:
            matches.append(s)
    return matches[0] if len(matches) == 1 else None


def format_suspect_roster(bible: CaseBible) -> str:
    """Render the canonical roster line-by-line for error messages."""
    return "\n".join(f"  [{s.id}] {s.name} — {s.archetype}" for s in bible.suspects)
