"""Post-generation semantic invariants for a CaseBible.

Pydantic guards the *shape*; this module guards the *meaning*: that ids
cross-reference correctly, that the killer's alibi is a lie, that the case
is at least nominally solvable. Run after a generator produces a bible;
any violation triggers a retry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mystery.models import CaseBible


class BibleInvariantError(ValueError):
    """Raised when a generated CaseBible violates a semantic invariant."""


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    dups: list[str] = []
    for v in values:
        if v in seen:
            dups.append(v)
        seen.add(v)
    return dups


def _check_unique_ids(bible: CaseBible) -> None:
    for kind, ids in (
        ("suspect", [s.id for s in bible.suspects]),
        ("location", [loc.id for loc in bible.locations]),
        ("clue", [c.id for c in bible.clues]),
    ):
        dups = _duplicates(ids)
        if dups:
            raise BibleInvariantError(f"duplicate {kind} ids: {dups}")


def _check_killer_is_a_suspect(bible: CaseBible) -> None:
    suspect_ids = {s.id for s in bible.suspects}
    if bible.killer_id not in suspect_ids:
        raise BibleInvariantError(
            f"killer_id {bible.killer_id!r} is not in suspects {sorted(suspect_ids)}",
        )


def _check_location_refs(bible: CaseBible) -> None:
    location_ids = {loc.id for loc in bible.locations}

    def _require(loc_id: str, where: str) -> None:
        if loc_id not in location_ids:
            raise BibleInvariantError(f"{where} references unknown location {loc_id!r}")

    _require(bible.victim.location_of_death_id, "victim.location_of_death_id")
    for loc in bible.locations:
        for adj in loc.connected_location_ids:
            _require(adj, f"location {loc.id}.connected_location_ids")
    for s in bible.suspects:
        for a in s.alibis:
            _require(a.location_id, f"suspect {s.id} alibi")
    for c in bible.clues:
        _require(c.location_id, f"clue {c.id}")
    for e in bible.canonical_timeline:
        _require(e.location_id, f"timeline event @t={e.time}")


def _check_suspect_refs(bible: CaseBible) -> None:
    suspect_ids = {s.id for s in bible.suspects}

    def _require(sus_id: str, where: str) -> None:
        if sus_id not in suspect_ids:
            raise BibleInvariantError(f"{where} references unknown suspect {sus_id!r}")

    for s in bible.suspects:
        for a in s.alibis:
            if a.corroborating_witness_id is not None:
                _require(a.corroborating_witness_id, f"suspect {s.id} alibi witness")
    for c in bible.clues:
        for sus_id in c.incriminates_suspect_ids:
            _require(sus_id, f"clue {c.id}.incriminates_suspect_ids")


def _check_killer_alibi_is_a_lie(bible: CaseBible) -> None:
    killer = next(s for s in bible.suspects if s.id == bible.killer_id)
    tod = bible.victim.time_of_death
    covering = [a for a in killer.alibis if a.time_window[0] <= tod <= a.time_window[1]]
    if not covering:
        raise BibleInvariantError(
            f"killer {killer.id!r} has no alibi covering time_of_death={tod}",
        )
    if not any(not a.is_true for a in covering):
        raise BibleInvariantError(
            f"killer {killer.id!r} alibis covering time_of_death are all true",
        )


def _check_killer_is_incriminated(bible: CaseBible) -> None:
    if not any(bible.killer_id in c.incriminates_suspect_ids for c in bible.clues):
        raise BibleInvariantError(
            f"no clue incriminates the killer {bible.killer_id!r} — case is unsolvable",
        )


def _check_location_edges_are_symmetric(bible: CaseBible) -> None:
    """A door from A to B must imply a door from B to A.

    The optimal-player DFS walks `move next` then `move current` to backtrack;
    a one-way edge silently strands the player and inflates parse errors. Even
    for the LLM player, asymmetric doors are surprising enough to count as a
    generator bug.
    """
    adj = {loc.id: set(loc.connected_location_ids) for loc in bible.locations}
    missing: list[str] = []
    for a, neighbours in adj.items():
        for b in neighbours:
            if b in adj and a not in adj[b]:
                missing.append(f"{a}->{b}")
    if missing:
        raise BibleInvariantError(
            f"asymmetric location edges (each must be bidirectional): {missing}",
        )


def _check_time_windows_well_formed(bible: CaseBible) -> None:
    for s in bible.suspects:
        for a in s.alibis:
            start, end = a.time_window
            if end < start:
                raise BibleInvariantError(
                    f"suspect {s.id} has alibi with end<start: {a.time_window}",
                )


def validate_bible(bible: CaseBible) -> None:
    """Raise BibleInvariantError on the first failing invariant.

    Order matters only insofar as later checks may assume earlier ones (e.g.
    ``_check_killer_alibi_is_a_lie`` assumes the killer is a suspect).
    """
    _check_unique_ids(bible)
    _check_killer_is_a_suspect(bible)
    _check_location_refs(bible)
    _check_location_edges_are_symmetric(bible)
    _check_suspect_refs(bible)
    _check_time_windows_well_formed(bible)
    _check_killer_alibi_is_a_lie(bible)
    _check_killer_is_incriminated(bible)
