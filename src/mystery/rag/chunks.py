"""Convert a CaseBible into RAG chunks with scoping metadata.

Two scopes exist:

* **private** — knowledge or alibis belonging to a single suspect. Tagged with
  ``character_id``. Only that suspect's retriever may surface these.
* **world**  — location descriptions and the victim's public information.
  Shared across all suspect retrievers.

Deliberately excluded from RAG entirely:

* ``canonical_timeline`` — the omniscient author's view; no agent should ever
  retrieve it.
* ``deception_policy`` — controls *how* a suspect lies, lives in the suspect
  agent's prompt, never in retrieval.
* ``clues`` — physical evidence reached through the ``examine`` tool against
  the bible directly (M5), not via suspect-facing similarity search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from mystery.models import Alibi, CaseBible

ChunkType = Literal["knowledge", "alibi", "location", "victim"]


class Chunk(BaseModel):
    """A single retrievable unit, ready to be turned into a langchain Document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    text: str
    chunk_type: ChunkType
    character_id: str | None = None
    location_id: str | None = None

    @property
    def scope(self) -> Literal["private", "world"]:
        return "private" if self.character_id is not None else "world"

    def metadata(self) -> dict[str, str]:
        """Chroma rejects None-valued metadata, so emit only set fields."""
        meta: dict[str, str] = {"chunk_type": self.chunk_type, "scope": self.scope}
        if self.character_id is not None:
            meta["character_id"] = self.character_id
        if self.location_id is not None:
            meta["location_id"] = self.location_id
        return meta


def _render_alibi(alibi: Alibi) -> str:
    start, end = alibi.time_window
    witness = (
        f" Witness: {alibi.corroborating_witness_id}."
        if alibi.corroborating_witness_id is not None
        else ""
    )
    return f"Claims to have been at {alibi.location_id} from t={start} to t={end}.{witness}"


def build_chunks(bible: CaseBible) -> list[Chunk]:
    """Flatten a bible into a list of retrievable chunks. Order is stable."""
    chunks: list[Chunk] = []

    chunks.append(
        Chunk(
            id="victim",
            text=f"The victim, {bible.victim.name} ({bible.victim.role}), was found in the "
            f"{bible.victim.location_of_death_id} at t={bible.victim.time_of_death}.",
            chunk_type="victim",
        ),
    )

    for loc in bible.locations:
        chunks.append(
            Chunk(
                id=f"location:{loc.id}",
                text=f"{loc.name}. {loc.description}",
                chunk_type="location",
                location_id=loc.id,
            ),
        )

    for s in bible.suspects:
        for i, fact in enumerate(s.knowledge):
            chunks.append(
                Chunk(
                    id=f"knowledge:{s.id}:{i}",
                    text=fact,
                    chunk_type="knowledge",
                    character_id=s.id,
                ),
            )
        for i, alibi in enumerate(s.alibis):
            chunks.append(
                Chunk(
                    id=f"alibi:{s.id}:{i}",
                    text=_render_alibi(alibi),
                    chunk_type="alibi",
                    character_id=s.id,
                    location_id=alibi.location_id,
                ),
            )

    return chunks
