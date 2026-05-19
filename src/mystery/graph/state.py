"""Game state and player-action types.

``GameState`` is a TypedDict because LangGraph merges partial dict updates
returned from each node. Action types are a discriminated Pydantic union so
the router can return one typed object instead of a tagged tuple.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

# Imported eagerly (not under TYPE_CHECKING) because LangGraph resolves
# TypedDict forward references at runtime via typing.get_type_hints when
# the graph is compiled. A TYPE_CHECKING-only import would NameError there.
from mystery.models import Commitment  # noqa: TC001

if TYPE_CHECKING:
    from mystery.models import CaseBible

_StrictAction = ConfigDict(extra="forbid", frozen=True)


class MoveAction(BaseModel):
    model_config = _StrictAction
    kind: Literal["move"] = "move"
    location_id: str


class InterrogateAction(BaseModel):
    model_config = _StrictAction
    kind: Literal["interrogate"] = "interrogate"
    suspect_id: str
    question: str


class ExamineAction(BaseModel):
    model_config = _StrictAction
    kind: Literal["examine"] = "examine"


class NotebookAction(BaseModel):
    model_config = _StrictAction
    kind: Literal["notebook"] = "notebook"


class AccuseAction(BaseModel):
    model_config = _StrictAction
    kind: Literal["accuse"] = "accuse"
    suspect_id: str


class ShowAction(BaseModel):
    """Confront a suspect with a specific revealed clue."""

    model_config = _StrictAction
    kind: Literal["show"] = "show"
    suspect_id: str
    clue_id: str


class HelpAction(BaseModel):
    model_config = _StrictAction
    kind: Literal["help"] = "help"


class SuspectsAction(BaseModel):
    """Free action: list everyone the player can interrogate."""

    model_config = _StrictAction
    kind: Literal["suspects"] = "suspects"


class LocationsAction(BaseModel):
    """Free action: list current room + connected exits."""

    model_config = _StrictAction
    kind: Literal["locations"] = "locations"


Action = Annotated[
    MoveAction
    | InterrogateAction
    | ExamineAction
    | NotebookAction
    | AccuseAction
    | ShowAction
    | HelpAction
    | SuspectsAction
    | LocationsAction,
    Field(discriminator="kind"),
]


class AccusationResult(BaseModel):
    """Outcome of an accuse action. Frozen — the verdict cannot be revised."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accused_id: str
    correct: bool
    actual_killer_id: str


class GameState(TypedDict):
    """All session-level state. Bible/vectorstore/chat live outside (immutable deps)."""

    current_location_id: str
    revealed_clue_ids: list[str]
    visited_location_ids: list[str]
    examined_location_ids: list[str]
    notebook: list[str]
    turn_count: int
    accusation: AccusationResult | None
    done: bool

    pending_action: Action | None
    last_output: str

    suspect_commitments: dict[str, list[Commitment]]


def initial_state(bible: CaseBible) -> GameState:
    """Build the starting GameState: the player walks in on the crime scene."""
    here = bible.victim.location_of_death_id
    return GameState(
        current_location_id=here,
        revealed_clue_ids=[],
        visited_location_ids=[here],
        examined_location_ids=[],
        notebook=[
            f"VICTIM: {bible.victim.name} ({bible.victim.role}) — found in "
            f"{here} at t={bible.victim.time_of_death}.",
        ],
        turn_count=0,
        accusation=None,
        done=False,
        pending_action=None,
        last_output="",
        suspect_commitments={},
    )
