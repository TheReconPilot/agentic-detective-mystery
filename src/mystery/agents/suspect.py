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
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.retrievers import BaseRetriever

    from mystery.models import Suspect


def _render_system(suspect: Suspect) -> str:
    motive_line = (
        f"Your motive, if any: {suspect.motive}."
        if suspect.motive is not None
        else "You have no obvious motive to kill the victim."
    )
    return (
        f"You are {suspect.name}, a {suspect.archetype} caught up in a murder mystery.\n"
        f"{motive_line}\n"
        f"Deception policy: {suspect.deception_policy}\n"
        "\n"
        "A detective is questioning you. Answer in character in 1-3 sentences. "
        "You may lie within your deception policy, but never invent facts that contradict "
        "what you actually know. If you are not supposed to know something, say so plainly."
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
) -> list[BaseMessage]:
    """Render the system + user messages for a single interrogation turn."""
    return [
        SystemMessage(content=_render_system(suspect)),
        HumanMessage(content=_render_user(retrieved_docs, question)),
    ]


def respond_as_suspect(
    suspect: Suspect,
    retriever: BaseRetriever,
    chat_model: BaseChatModel,
    question: str,
) -> str:
    """Run one interrogation turn end-to-end."""
    docs = retriever.invoke(question)
    messages = build_suspect_messages(suspect, docs, question)
    response = chat_model.invoke(messages)
    return str(response.content)
