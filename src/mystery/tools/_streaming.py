"""Streaming UX helpers shared by the interrogate / show tools.

The first token from a locally-hosted LLM can take 10-30 seconds on a 4 GB
GPU. Without visible feedback, players assume the game is stuck (and have
killed sessions thinking exactly that). This wrapper prints a 'thinking…'
indicator immediately, then replaces it with the speaker prefix the moment
the first chunk arrives — so even if the wait is long, the player always
sees that something is happening.

The wrapper is a one-shot callable: it owns the indicator state across the
chunk stream and exposes ``finalize()`` for the trailing newline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


# ANSI: carriage return + erase-to-end-of-line. Used to wipe the
# "thinking…" line when the first chunk arrives. Terminals that don't
# support ANSI will just see the wipe sequence as garbled punctuation,
# which is no worse than the original behaviour.
_ERASE_LINE = "\r\033[2K"


class StreamingPrefix:
    """Wrap a raw stream callback with a speaker prefix and thinking indicator.

    Lifecycle:
        ``StreamingPrefix`` is instantiated, immediately writes the
        thinking indicator, then is passed as the stream callback to
        the LLM. The first call with non-empty content replaces the
        indicator with ``{speaker}: `` before forwarding the chunk.
        ``finalize()`` writes the trailing newline regardless.
    """

    def __init__(self, raw_callback: Callable[[str], None], speaker_name: str) -> None:
        self._raw = raw_callback
        self._speaker = speaker_name
        self._opened = False
        self._raw(f"{speaker_name} is thinking…")

    def __call__(self, chunk: str) -> None:
        if not chunk:
            return
        if not self._opened:
            self._raw(f"{_ERASE_LINE}{self._speaker}: ")
            self._opened = True
        self._raw(chunk)

    def finalize(self) -> None:
        """Always called after the stream completes. Writes the trailing
        newline. If no chunk ever arrived we still wipe the indicator so the
        next REPL prompt starts on a clean line."""
        if not self._opened:
            self._raw(f"{_ERASE_LINE}{self._speaker}: (no reply)")
        self._raw("\n")
