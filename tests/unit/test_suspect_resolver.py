"""Tests for the lenient suspect-reference resolver.

The resolver is what unblocks players who type 'Hodges', 'butler', or
'servant' instead of the snake_case id we use internally. The exact
matching contract is asserted here so changes in policy show up as
test diffs rather than as silent regressions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mystery.tools._resolve import format_suspect_roster, resolve_suspect

if TYPE_CHECKING:
    from mystery.models import CaseBible


def test_resolves_by_exact_id(valid_bible: CaseBible) -> None:
    assert resolve_suspect(valid_bible, "butler") is not None
    s = resolve_suspect(valid_bible, "butler")
    assert s is not None
    assert s.id == "butler"


def test_resolves_by_exact_id_case_insensitive(valid_bible: CaseBible) -> None:
    s = resolve_suspect(valid_bible, "Butler")
    assert s is not None and s.id == "butler"


def test_resolves_by_full_name(valid_bible: CaseBible) -> None:
    s = resolve_suspect(valid_bible, "Mr. Hodges")
    assert s is not None and s.id == "butler"


def test_resolves_by_last_name_token(valid_bible: CaseBible) -> None:
    s = resolve_suspect(valid_bible, "Hodges")
    assert s is not None and s.id == "butler"


def test_resolves_by_first_name_token(valid_bible: CaseBible) -> None:
    s = resolve_suspect(valid_bible, "Eleanor")
    assert s is not None and s.id == "niece"


def test_resolves_by_archetype_token(valid_bible: CaseBible) -> None:
    """The point: id=housekeeper / archetype='The Nervous Servant' style
    mismatch is what triggered this work. The fixture has archetype='cook'
    matching id='cook', but the resolver should also accept a player who
    types just the archetype noun."""
    s = resolve_suspect(valid_bible, "cook")
    assert s is not None and s.id == "cook"


def test_resolves_archetype_split_on_hyphen(valid_bible: CaseBible) -> None:
    """archetype='estranged-niece' should be matched by either word."""
    s = resolve_suspect(valid_bible, "estranged")
    assert s is not None and s.id == "niece"


def test_strips_title_words(valid_bible: CaseBible) -> None:
    s = resolve_suspect(valid_bible, "Mrs. Pell")
    assert s is not None and s.id == "cook"


def test_returns_none_for_unknown(valid_bible: CaseBible) -> None:
    assert resolve_suspect(valid_bible, "phantom") is None


def test_returns_none_for_empty(valid_bible: CaseBible) -> None:
    assert resolve_suspect(valid_bible, "") is None
    assert resolve_suspect(valid_bible, "   ") is None


def test_returns_none_for_stopwords_only(valid_bible: CaseBible) -> None:
    """'the' alone should not match — would otherwise hit anything with a
    multi-word name on a different policy."""
    assert resolve_suspect(valid_bible, "the") is None
    assert resolve_suspect(valid_bible, "Mr") is None


def test_format_roster_lists_every_suspect(valid_bible: CaseBible) -> None:
    roster = format_suspect_roster(valid_bible)
    for s in valid_bible.suspects:
        assert s.id in roster
        assert s.name in roster
        assert s.archetype in roster
