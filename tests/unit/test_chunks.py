from __future__ import annotations

from typing import TYPE_CHECKING

from mystery.rag.chunks import build_chunks

if TYPE_CHECKING:
    from mystery.models import CaseBible


def test_chunks_have_unique_ids(valid_bible: CaseBible) -> None:
    chunks = build_chunks(valid_bible)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_world_chunks_have_no_character_id(valid_bible: CaseBible) -> None:
    chunks = build_chunks(valid_bible)
    for c in chunks:
        if c.chunk_type in {"location", "victim"}:
            assert c.character_id is None
            assert c.scope == "world"


def test_private_chunks_are_attributed(valid_bible: CaseBible) -> None:
    chunks = build_chunks(valid_bible)
    for c in chunks:
        if c.chunk_type in {"knowledge", "alibi"}:
            assert c.character_id is not None
            assert c.scope == "private"


def test_timeline_is_not_chunked(valid_bible: CaseBible) -> None:
    """The canonical timeline is the author's view and must never enter RAG."""
    chunks = build_chunks(valid_bible)
    timeline_text = valid_bible.canonical_timeline[1].description  # the poisoning
    assert not any(timeline_text in c.text for c in chunks)


def test_deception_policy_is_not_chunked(valid_bible: CaseBible) -> None:
    chunks = build_chunks(valid_bible)
    for s in valid_bible.suspects:
        assert not any(s.deception_policy in c.text for c in chunks)


def test_clues_are_not_chunked(valid_bible: CaseBible) -> None:
    """Clues are surfaced via the examine tool, not suspect-facing retrieval."""
    chunks = build_chunks(valid_bible)
    for clue in valid_bible.clues:
        assert not any(clue.description in c.text for c in chunks)


def test_metadata_omits_none_fields(valid_bible: CaseBible) -> None:
    """Chroma rejects None-valued metadata; verify we never emit any."""
    for chunk in build_chunks(valid_bible):
        meta = chunk.metadata()
        assert all(v is not None for v in meta.values())
