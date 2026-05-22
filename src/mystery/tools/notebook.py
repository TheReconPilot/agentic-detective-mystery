"""Notebook tool: display the player's accumulated notes. Free action (no turn cost).

The notebook is both the rolling log (clue lines, victim exam, forensics) AND
a derived suspect-summary section that surfaces what the player should be
*comparing*: who has corroboration, who doesn't, which revealed clues point at
which suspect. The summary is computed at render time rather than persisted, so
it always reflects current state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mystery.graph.state import GameState
    from mystery.models import CaseBible, Suspect


def _format_time_window(window: tuple[int, int] | None) -> str:
    if window is None:
        return "?"
    return f"{window[0]}-{window[1]}"


def _suspect_name_lookup(bible: CaseBible) -> dict[str, str]:
    return {s.id: s.name for s in bible.suspects}


def _render_suspect_summary(state: GameState, bible: CaseBible) -> str:
    """Per-suspect rollup: latest claim, corroboration, incriminating clues seen.

    Three signals the player should be cross-checking but tends to miss
    when they live as scattered LLM dialogue lines:
        ✓ corroborated  — another suspect named this one as a witness
        ⚠ uncorroborated — no one else has vouched for the claim
        ⚠ clue           — a *revealed* clue points at this suspect
    Nothing is decided for the player; the marks just say which threads
    need pulling.
    """
    commitments = state["suspect_commitments"]
    if not commitments:
        return ""

    name_by_id = _suspect_name_lookup(bible)
    # Build the reverse-corroboration map: who has been named as a witness by whom.
    named_by: dict[str, set[str]] = {}
    for speaker_id, speaker_commits in commitments.items():
        for c in speaker_commits:
            for witness_id in c.named_witness_ids:
                named_by.setdefault(witness_id, set()).add(speaker_id)

    revealed_clues = [c for c in bible.clues if c.id in state["revealed_clue_ids"]]

    def _lines_for(suspect: Suspect) -> list[str]:
        suspect_commits = commitments.get(suspect.id, [])
        if not suspect_commits:
            return []
        latest = suspect_commits[-1]
        header = f"  {suspect.id} ({suspect.name}) — {latest.summary}"
        sub: list[str] = []
        if latest.claimed_location_id or latest.claimed_time_window:
            sub.append(
                "    claimed: "
                f"{latest.claimed_location_id or '?'} "
                f"[{_format_time_window(latest.claimed_time_window)}]"
            )
        # Corroboration: did anyone else name them as a witness?
        vouchers = sorted(named_by.get(suspect.id, set()))
        if vouchers:
            voucher_names = ", ".join(name_by_id.get(v, v) for v in vouchers)
            sub.append(f"    ✓ corroborated by {voucher_names}")
        elif latest.claimed_location_id is not None:
            sub.append("    ⚠ no corroborator named")
        # Revealed clues that point at this suspect.
        hits = [c for c in revealed_clues if suspect.id in c.incriminates_suspect_ids]
        for c in hits:
            sub.append(f"    ⚠ clue [{c.id}] points here")
        return [header, *sub]

    lines: list[str] = []
    for s in bible.suspects:
        lines.extend(_lines_for(s))
    if not lines:
        return ""
    return "Suspect summary:\n" + "\n".join(lines)


def apply_notebook(state: GameState, bible: CaseBible) -> dict[str, Any]:
    if not state["notebook"] and not state["suspect_commitments"]:
        return {"last_output": "Your notebook is empty."}
    log_lines = "\n".join(f"  {line}" for line in state["notebook"])
    log_block = f"Your notebook:\n{log_lines}" if state["notebook"] else "Your notebook is empty."
    summary = _render_suspect_summary(state, bible)
    text = f"{log_block}\n\n{summary}" if summary else log_block
    return {"last_output": text}
