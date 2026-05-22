from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from mystery.agents.suspect import build_suspect_messages

if TYPE_CHECKING:
    from mystery.models import CaseBible, Suspect


def _by_id(bible: CaseBible, suspect_id: str) -> Suspect:
    return next(s for s in bible.suspects if s.id == suspect_id)


def test_messages_have_system_then_user(valid_bible: CaseBible) -> None:
    msgs = build_suspect_messages(_by_id(valid_bible, "butler"), [], question="where were you?")
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage)


def test_system_message_contains_persona(valid_bible: CaseBible) -> None:
    butler = _by_id(valid_bible, "butler")
    msgs = build_suspect_messages(butler, [], question="?")
    system = str(msgs[0].content)
    assert butler.name in system
    assert butler.archetype in system
    assert butler.deception_policy in system
    assert butler.motive is not None  # narrows for the next assertion
    assert butler.motive in system


def test_system_message_handles_innocent_with_no_motive(valid_bible: CaseBible) -> None:
    cook = _by_id(valid_bible, "cook")
    msgs = build_suspect_messages(cook, [], question="?")
    system = str(msgs[0].content)
    assert "no obvious motive" in system


def test_user_message_includes_retrieved_facts_and_question(valid_bible: CaseBible) -> None:
    docs = [
        Document(page_content="The library door sticks when humid."),
        Document(page_content="Dinner was at half past eight."),
    ]
    msgs = build_suspect_messages(_by_id(valid_bible, "butler"), docs, question="alibi?")
    user = str(msgs[1].content)
    assert "The library door sticks when humid." in user
    assert "Dinner was at half past eight." in user
    assert "alibi?" in user


def test_user_message_handles_empty_retrieval(valid_bible: CaseBible) -> None:
    msgs = build_suspect_messages(_by_id(valid_bible, "butler"), [], question="who are you?")
    user = str(msgs[1].content)
    assert "no specific facts" in user
    assert "who are you?" in user


def test_system_message_renders_voice_when_set(valid_bible: CaseBible) -> None:
    """A suspect with a voice gets a 'How you talk:' line in the system prompt."""
    butler = _by_id(valid_bible, "butler")
    assert butler.voice  # fixture must have one or this test is meaningless
    msgs = build_suspect_messages(butler, [], question="?")
    system = str(msgs[0].content)
    assert "How you talk:" in system
    assert butler.voice in system


def test_system_message_omits_voice_line_when_empty(valid_bible: CaseBible) -> None:
    """Bibles generated before the voice field load with voice='' — no broken line."""
    butler = _by_id(valid_bible, "butler").model_copy(update={"voice": ""})
    msgs = build_suspect_messages(butler, [], question="?")
    system = str(msgs[0].content)
    assert "How you talk:" not in system


def test_confronting_clue_that_incriminates_suspect_demands_a_tell(
    valid_bible: CaseBible,
) -> None:
    butler = _by_id(valid_bible, "butler")
    muddy_boots = next(c for c in valid_bible.clues if c.id == "muddy_boots")
    assert butler.id in muddy_boots.incriminates_suspect_ids  # sanity check on fixture

    msgs = build_suspect_messages(butler, [], question="?", confronting_clue=muddy_boots)
    system = str(msgs[0].content)
    assert "real hit" in system
    assert "physical tell" in system


def test_confronting_clue_that_does_not_incriminate_omits_tell_instruction(
    valid_bible: CaseBible,
) -> None:
    cook = _by_id(valid_bible, "cook")
    muddy_boots = next(c for c in valid_bible.clues if c.id == "muddy_boots")
    assert cook.id not in muddy_boots.incriminates_suspect_ids

    msgs = build_suspect_messages(cook, [], question="?", confronting_clue=muddy_boots)
    system = str(msgs[0].content)
    assert "real hit" not in system
    # The base confrontation framing must still appear so cook reacts to the item.
    assert "holding up a piece of evidence" in system
