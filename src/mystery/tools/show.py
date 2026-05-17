"""Show tool: confront a suspect with a specific revealed clue.

The M10 payoff for M9: when a clue contradicts a prior commitment, this is
where the deception policy is supposed to crack. The suspect agent receives
the clue's description in the system prompt on top of the persona / voice /
commitments scaffolding, then the same commitment extractor runs on the
reaction so the confrontation feeds the next-turn carry-through.

A clue must already be in ``state["revealed_clue_ids"]`` to be shown — the
player can't bluff their way through by naming clues they haven't earned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mystery.agents.commitments import NullCommitmentExtractor
from mystery.agents.suspect import respond_as_suspect
from mystery.rag.retriever import suspect_retriever

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel

    from mystery.agents.commitments import CommitmentExtractor
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_show(
    state: GameState,
    bible: CaseBible,
    vectorstore: Chroma,
    chat_model: BaseChatModel,
    suspect_id: str,
    clue_id: str,
    commitment_extractor: CommitmentExtractor | None = None,
) -> dict[str, Any]:
    suspect = next((s for s in bible.suspects if s.id == suspect_id), None)
    if suspect is None:
        ids = ", ".join(s.id for s in bible.suspects)
        return {"last_output": f"There is no suspect {suspect_id!r}. Known: {ids}."}

    clue = next((c for c in bible.clues if c.id == clue_id), None)
    if clue is None:
        return {
            "last_output": (
                f"You have no clue called {clue_id!r}. "
                f"Check your notebook for the clues you have found."
            ),
        }
    if clue_id not in state["revealed_clue_ids"]:
        return {
            "last_output": (
                f"You haven't found {clue_id!r} yet. "
                f"You can only confront a suspect with clues you have actually examined."
            ),
        }

    retriever = suspect_retriever(vectorstore, suspect_id=suspect_id)
    prior = state["suspect_commitments"].get(suspect_id, [])
    # The "question" we pass is the framing the suspect sees: the system
    # prompt's clue block already handles the confrontation, so the user
    # message just narrates the moment in the detective's voice.
    framing_question = "I'm showing you this. What do you have to say about it?"
    reply = respond_as_suspect(
        suspect,
        retriever,
        chat_model,
        question=framing_question,
        prior_commitments=prior,
        confronting_clue=clue,
    )

    extractor: CommitmentExtractor = commitment_extractor or NullCommitmentExtractor()
    new_commitment = extractor.extract(suspect, framing_question, reply)
    update: dict[str, Any] = {
        "turn_count": state["turn_count"] + 1,
        "last_output": f"You show {suspect.name} the {clue.description}\n{suspect.name}: {reply}",
    }
    if new_commitment is not None:
        updated_commitments = {k: list(v) for k, v in state["suspect_commitments"].items()}
        updated_commitments.setdefault(suspect_id, []).append(new_commitment)
        update["suspect_commitments"] = updated_commitments
    return update
