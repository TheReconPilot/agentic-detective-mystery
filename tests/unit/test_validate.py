from __future__ import annotations

import pytest

from mystery.case_gen.validate import BibleInvariantError, validate_bible
from mystery.models import Alibi, CaseBible


def test_valid_bible_passes(valid_bible: CaseBible) -> None:
    validate_bible(valid_bible)  # should not raise


def test_rejects_killer_not_in_suspects(valid_bible: CaseBible) -> None:
    bad = valid_bible.model_copy(update={"killer_id": "ghost"})
    with pytest.raises(BibleInvariantError, match="killer_id"):
        validate_bible(bad)


def test_rejects_alibi_to_unknown_location(valid_bible: CaseBible) -> None:
    butler = valid_bible.suspects[0].model_copy(
        update={
            "alibis": [
                Alibi(location_id="atlantis", time_window=(45, 75), is_true=False),
            ],
        },
    )
    bad = valid_bible.model_copy(update={"suspects": [butler, *valid_bible.suspects[1:]]})
    with pytest.raises(BibleInvariantError, match="unknown location"):
        validate_bible(bad)


def test_rejects_unknown_witness(valid_bible: CaseBible) -> None:
    niece = valid_bible.suspects[1].model_copy(
        update={
            "alibis": [
                Alibi(
                    location_id="hallway",
                    time_window=(50, 70),
                    is_true=True,
                    corroborating_witness_id="phantom",
                ),
            ],
        },
    )
    bad = valid_bible.model_copy(
        update={"suspects": [valid_bible.suspects[0], niece, valid_bible.suspects[2]]},
    )
    with pytest.raises(BibleInvariantError, match="unknown suspect"):
        validate_bible(bad)


def test_rejects_killer_whose_alibi_is_true(valid_bible: CaseBible) -> None:
    butler = valid_bible.suspects[0].model_copy(
        update={
            "alibis": [
                Alibi(location_id="garden", time_window=(45, 75), is_true=True),
            ],
        },
    )
    bad = valid_bible.model_copy(update={"suspects": [butler, *valid_bible.suspects[1:]]})
    with pytest.raises(BibleInvariantError, match="all true"):
        validate_bible(bad)


def test_rejects_killer_with_no_covering_alibi(valid_bible: CaseBible) -> None:
    butler = valid_bible.suspects[0].model_copy(
        update={
            "alibis": [
                Alibi(location_id="garden", time_window=(0, 10), is_true=False),
            ],
        },
    )
    bad = valid_bible.model_copy(update={"suspects": [butler, *valid_bible.suspects[1:]]})
    with pytest.raises(BibleInvariantError, match="no alibi covering"):
        validate_bible(bad)


def test_rejects_unsolvable_case(valid_bible: CaseBible) -> None:
    clean_clues = [
        c.model_copy(update={"incriminates_suspect_ids": ["niece"]}) for c in valid_bible.clues
    ]
    bad = valid_bible.model_copy(update={"clues": clean_clues})
    with pytest.raises(BibleInvariantError, match="unsolvable"):
        validate_bible(bad)


def test_rejects_duplicate_suspect_ids(valid_bible: CaseBible) -> None:
    dup = valid_bible.suspects[1].model_copy(update={"id": "butler"})
    bad = valid_bible.model_copy(
        update={"suspects": [valid_bible.suspects[0], dup, valid_bible.suspects[2]]},
    )
    with pytest.raises(BibleInvariantError, match="duplicate suspect"):
        validate_bible(bad)


def test_rejects_inverted_time_window(valid_bible: CaseBible) -> None:
    butler = valid_bible.suspects[0].model_copy(
        update={
            "alibis": [
                Alibi(location_id="garden", time_window=(75, 45), is_true=False),
            ],
        },
    )
    bad = valid_bible.model_copy(update={"suspects": [butler, *valid_bible.suspects[1:]]})
    with pytest.raises(BibleInvariantError, match="end<start"):
        validate_bible(bad)
