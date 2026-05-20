"""Unit tests for the streaming UX helper.

The wrapper exists to make slow first-token latency tolerable: the player
sees ``<name> is thinking…`` immediately, and the indicator is wiped the
moment a real chunk arrives. The exact ANSI bytes are an implementation
detail, but the lifecycle and visible result are part of the contract.
"""

from __future__ import annotations

from mystery.tools._streaming import StreamingPrefix


def test_writes_thinking_indicator_on_construction() -> None:
    chunks: list[str] = []
    StreamingPrefix(chunks.append, "Mr. Hodges")
    assert chunks == ["Mr. Hodges is thinking…"]


def test_first_chunk_replaces_indicator_with_prefix() -> None:
    chunks: list[str] = []
    sp = StreamingPrefix(chunks.append, "Mr. Hodges")
    sp("Hello.")
    joined = "".join(chunks)
    assert "Mr. Hodges is thinking" in joined
    # The visible prefix appears before the content, regardless of ANSI bytes.
    assert joined.index("Mr. Hodges: ") < joined.index("Hello.")


def test_empty_chunks_do_not_paint_prefix_or_advance() -> None:
    """Some streaming backends emit empty leading chunks. They shouldn't
    flip the wrapper's 'opened' state, otherwise the prefix is painted
    before any real content arrives."""
    chunks: list[str] = []
    sp = StreamingPrefix(chunks.append, "Mr. Hodges")
    before = list(chunks)
    sp("")
    assert chunks == before  # no-op


def test_finalize_writes_trailing_newline_when_chunks_arrived() -> None:
    chunks: list[str] = []
    sp = StreamingPrefix(chunks.append, "Mr. Hodges")
    sp("A reply.")
    sp.finalize()
    assert "".join(chunks).endswith("\n")


def test_finalize_handles_silent_stream_with_no_reply_placeholder() -> None:
    """If the LLM produces nothing, finalize must still leave the terminal on
    a clean line and tell the player so they aren't staring at the indicator."""
    chunks: list[str] = []
    sp = StreamingPrefix(chunks.append, "Mr. Hodges")
    sp.finalize()
    joined = "".join(chunks)
    assert "no reply" in joined
    assert joined.endswith("\n")
