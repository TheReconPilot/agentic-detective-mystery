from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from mystery.graph.state import (
    AccusationResult,
    AccuseAction,
    Action,
    ExamineAction,
    InterrogateAction,
    MoveAction,
)

_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def test_action_dispatches_on_kind_field() -> None:
    move = _ACTION_ADAPTER.validate_python({"kind": "move", "location_id": "library"})
    interrogate = _ACTION_ADAPTER.validate_python(
        {"kind": "interrogate", "suspect_id": "butler", "question": "where?"},
    )
    assert isinstance(move, MoveAction)
    assert isinstance(interrogate, InterrogateAction)


def test_action_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        _ACTION_ADAPTER.validate_python({"kind": "teleport", "location_id": "library"})


def test_action_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        MoveAction(location_id="library", surprise="boo")  # type: ignore[call-arg]


def test_examine_action_needs_no_args() -> None:
    ExamineAction()  # should not raise


def test_accusation_result_is_frozen() -> None:
    result = AccusationResult(accused_id="butler", correct=True, actual_killer_id="butler")
    with pytest.raises(ValidationError):
        result.accused_id = "niece"


def test_accuse_action_carries_suspect_id() -> None:
    a = AccuseAction(suspect_id="butler")
    assert a.suspect_id == "butler"
    assert a.kind == "accuse"
