"""Pydantic schemas for the case bible and downstream game state.

The CaseBible is the single source of truth for a generated mystery. Suspects
may lie within the bounds of their ``deception_policy``, but never beyond what
the bible records. Anything that contradicts the bible is, by definition, a
bug — guarded by the consistency eval.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_StrictModel = ConfigDict(extra="forbid", frozen=False, str_strip_whitespace=True)


class Victim(BaseModel):
    """Who died, where, and when."""

    model_config = _StrictModel

    name: str
    role: str = Field(description="Short description, e.g. 'host of the dinner party'.")
    location_of_death_id: str
    time_of_death: int = Field(
        description="Minutes from case t0 (negative allowed for pre-events).",
    )


class Location(BaseModel):
    """A room or area the player can visit."""

    model_config = _StrictModel

    id: str = Field(description="Slug, e.g. 'library'. Lowercase, snake_case.")
    name: str
    description: str
    connected_location_ids: list[str] = Field(
        default_factory=list,
        description="Adjacent locations the player can move to directly.",
    )


class TimelineEvent(BaseModel):
    """A factual event in the canonical timeline. Visible only to the case author."""

    model_config = _StrictModel

    time: int = Field(description="Minutes from case t0.")
    actor_id: str = Field(description="Suspect id, or the victim's name as a sentinel.")
    location_id: str
    description: str


class Alibi(BaseModel):
    """A claim a suspect makes about their whereabouts.

    ``is_true`` is the bible's verdict — the suspect may or may not tell the truth
    about it, governed by their ``deception_policy``.
    """

    model_config = _StrictModel

    location_id: str
    time_window: tuple[int, int] = Field(description="(start, end) minutes from case t0.")
    is_true: bool
    corroborating_witness_id: str | None = Field(
        default=None,
        description="Another suspect who can vouch for this alibi, if any.",
    )


class Suspect(BaseModel):
    """A suspect — exactly one of whom is the killer, recorded in CaseBible.killer_id."""

    model_config = _StrictModel

    id: str = Field(description="Slug, e.g. 'butler'. Lowercase, snake_case.")
    name: str
    archetype: str = Field(
        description="One- or two-word archetype, e.g. 'butler', 'estranged-sibling'.",
    )
    motive: str | None = Field(
        default=None,
        description="Reason this suspect might have killed the victim. None for clearly-innocent.",
    )
    alibis: list[Alibi]
    knowledge: list[str] = Field(
        description="Discrete facts this character truthfully knows (one sentence each).",
    )
    deception_policy: str = Field(
        description=(
            "Natural-language rules for how this character lies. "
            "E.g. 'Lies about being in the garden at 21:00 but is truthful about everything else.'"
        ),
    )
    voice: str = Field(
        default="",
        description=(
            "Two or three sentences on how this character talks: speech rhythm, "
            "a verbal tic or favourite expression, and what they steer the "
            "conversation away from. Optional but strongly recommended — without "
            "it, suspects sound like generic NPCs and the game feels lifeless. "
            "Default empty for backward-compat with bibles generated before "
            "this field existed."
        ),
    )


class Clue(BaseModel):
    """Physical evidence the player can discover by examining a location."""

    model_config = _StrictModel

    id: str
    location_id: str
    description: str
    incriminates_suspect_ids: list[str] = Field(
        description="Suspects this clue points to. May include red herrings.",
    )


class CaseBible(BaseModel):
    """The complete, private specification of a generated mystery."""

    model_config = _StrictModel

    seed: int
    victim: Victim
    suspects: list[Suspect]
    locations: list[Location]
    clues: list[Clue]
    killer_id: str
    canonical_timeline: list[TimelineEvent]


class Commitment(BaseModel):
    """A structured summary of what a suspect just claimed in one interrogation turn.

    The raw transcript is deliberately NOT persisted into the next suspect prompt;
    if it were, the LLM would treat its own past lies as ground truth and the
    bible-as-canon discipline would erode. A Commitment is the small, surgical
    slice we feed back: claimed location, claimed time window, named witnesses,
    explicitly denied facts. The deception policy then has something concrete to
    stay consistent with across turns.
    """

    model_config = _StrictModel

    claimed_location_id: str | None = Field(
        default=None,
        description="Location the suspect claimed to be at, if any.",
    )
    claimed_time_window: tuple[int, int] | None = Field(
        default=None,
        description="(start, end) minutes from case t0, if the suspect committed to a time.",
    )
    named_witness_ids: list[str] = Field(
        default_factory=list,
        description="Other suspect ids the speaker named as witnesses to their account.",
    )
    denied_facts: list[str] = Field(
        default_factory=list,
        description="Statements the suspect explicitly denied (one short sentence each).",
    )
    summary: str = Field(
        description=(
            "One short sentence paraphrasing the claim, used verbatim in the "
            "next-turn 'you previously told this detective' block."
        ),
    )
