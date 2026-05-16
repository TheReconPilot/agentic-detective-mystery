from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from mystery.models import Alibi, CaseBible, Suspect


def test_valid_bible_round_trips_through_dict(valid_bible: CaseBible) -> None:
    dumped = valid_bible.model_dump()
    rebuilt = CaseBible.model_validate(dumped)
    assert rebuilt == valid_bible


def test_extras_are_forbidden(valid_bible_dict: dict[str, Any]) -> None:
    valid_bible_dict["spurious_field"] = "nope"
    with pytest.raises(ValidationError):
        CaseBible.model_validate(valid_bible_dict)


def test_alibi_time_window_is_a_two_tuple() -> None:
    with pytest.raises(ValidationError):
        Alibi(location_id="x", time_window=(1, 2, 3), is_true=True)  # type: ignore[arg-type]


def test_suspect_motive_is_optional() -> None:
    s = Suspect(
        id="x",
        name="X",
        archetype="x",
        motive=None,
        alibis=[],
        knowledge=[],
        deception_policy="truthful",
    )
    assert s.motive is None
