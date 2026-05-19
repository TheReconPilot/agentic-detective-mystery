"""Suspect agent: retrieve → reason → respond.

A suspect agent is a thin chain over three building blocks:

* a character-scoped retriever (built in :mod:`mystery.rag.retriever`),
* a chat model (anything implementing langchain's ``BaseChatModel``),
* and a prompt builder that renders the suspect's persona plus the
  retrieved facts.

The chain is intentionally not a LangGraph sub-graph: every interrogation
turn is strictly linear, so a graph would add ceremony without buying any
branching. If a suspect ever needs internal control flow (e.g. "if the
question is about the alibi, fetch alibi-specific chunks first"), promote
this to a sub-graph then — not before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.retrievers import BaseRetriever

    from mystery.models import Clue, Commitment, Suspect

    StreamCallback = Callable[[str], None]


def _render_commitments(commitments: list[Commitment]) -> str:
    """Render prior commitments as a 'you previously told this detective' block.

    Crucially, we only render the structured Commitment summaries — never the
    raw transcript. If the model saw its own prior wording verbatim it would
    treat its lies as ground truth and freely elaborate on them; the
    bible-as-canon discipline depends on filtering that loop.
    """
    if not commitments:
        return ""
    lines = "\n".join(f"  - {c.summary}" for c in commitments)
    return (
        "You previously told this detective:\n"
        f"{lines}\n"
        "Stay consistent with those prior statements — your deception policy "
        "applies across turns, not just this one.\n"
    )


def _render_confronting_clue(clue: Clue | None) -> str:
    """Render the 'detective is holding up X' block when a clue is being shown.

    This is the M10 hinge: a clue that contradicts a prior commitment is
    the moment the deception policy is supposed to crack. The block
    deliberately frames the moment as physical confrontation, not abstract
    question — the suspect can SEE the evidence.
    """
    if clue is None:
        return ""
    return (
        f"The detective is now holding up a piece of evidence: {clue.description}\n"
        "You must react to seeing it in front of you. If it contradicts something "
        "you previously told this detective, your deception policy says how you "
        "should cope — bluster, deny, redirect, or break — but DO react. Generic "
        "denial without engaging with the specific item is forbidden.\n"
    )


def _render_system(
    suspect: Suspect,
    prior_commitments: list[Commitment] | None = None,
    confronting_clue: Clue | None = None,
) -> str:
    motive_line = (
        f"Your motive, if any: {suspect.motive}."
        if suspect.motive is not None
        else "You have no obvious motive to kill the victim."
    )
    voice_line = f"How you talk: {suspect.voice}\n" if suspect.voice else ""
    commitments_block = _render_commitments(prior_commitments or [])
    clue_block = _render_confronting_clue(confronting_clue)
    return (
        f"You are {suspect.name}, a {suspect.archetype} caught up in a murder mystery.\n"
        f"{motive_line}\n"
        f"Deception policy: {suspect.deception_policy}\n"
        f"{voice_line}"
        f"{commitments_block}"
        f"{clue_block}"
        "\n"
        "A detective is questioning you. Answer in character in 1-3 sentences. "
        "You may lie within your deception policy, but never invent facts that contradict "
        "what you actually know. If you are not supposed to know something, say so plainly. "
        "Stay in your distinctive voice — generic 'as a suspect, I…' phrasing is forbidden."
    )


def _render_user(retrieved_docs: list[Document], question: str) -> str:
    if retrieved_docs:
        facts = "\n".join(f"- {doc.page_content}" for doc in retrieved_docs)
        facts_block = f"What you know and can draw on:\n{facts}\n\n"
    else:
        facts_block = "You have no specific facts to draw on for this question.\n\n"
    return f"{facts_block}The detective asks: {question}"


def build_suspect_messages(
    suspect: Suspect,
    retrieved_docs: list[Document],
    question: str,
    prior_commitments: list[Commitment] | None = None,
    confronting_clue: Clue | None = None,
) -> list[BaseMessage]:
    """Render the system + user messages for a single interrogation turn.

    ``prior_commitments`` is the list of structured summaries from earlier
    turns with this same suspect — see :class:`mystery.models.Commitment`.

    ``confronting_clue`` is the M10 hook: when the detective uses ``show``,
    pass the Clue object so the suspect agent renders a "detective is
    holding up X" block. The persona + voice + commitments scaffolding is
    fully shared with the plain ``ask`` path — the suspect inherits all of
    M8/M9 even when the action is a confrontation.
    """
    return [
        SystemMessage(content=_render_system(suspect, prior_commitments, confronting_clue)),
        HumanMessage(content=_render_user(retrieved_docs, question)),
    ]


def respond_as_suspect(
    suspect: Suspect,
    retriever: BaseRetriever,
    chat_model: BaseChatModel,
    question: str,
    prior_commitments: list[Commitment] | None = None,
    confronting_clue: Clue | None = None,
    stream_callback: StreamCallback | None = None,
) -> str:
    """Run one interrogation turn end-to-end.

    When ``stream_callback`` is provided the chat model is consumed via
    ``.stream()`` and each non-empty content chunk is forwarded to the
    callback as it arrives. The full concatenated reply is still returned so
    upstream bookkeeping (commitment extraction, ``last_output``) is
    unchanged. Without a callback we keep the original ``.invoke()`` path so
    tests and evals (where ``FakeListChatModel`` is used) remain
    deterministic.
    """
    docs = retriever.invoke(question)
    messages = build_suspect_messages(
        suspect,
        docs,
        question,
        prior_commitments,
        confronting_clue,
    )
    if stream_callback is None:
        response = chat_model.invoke(messages)
        return str(response.content)

    parts: list[str] = []
    for chunk in chat_model.stream(messages):
        content = str(chunk.content)
        if content:
            stream_callback(content)
            parts.append(content)
    return "".join(parts)
