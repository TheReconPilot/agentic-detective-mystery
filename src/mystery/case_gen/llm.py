"""Concrete BibleLLM backed by Ollama via langchain-ollama.

Kept separate from the generator so tests can avoid importing langchain.
"""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from mystery.models import CaseBible


class OllamaBibleLLM:
    """Wraps ChatOllama with structured output bound to CaseBible."""

    def __init__(self, model: str, seed: int, base_url: str | None = None) -> None:
        chat = ChatOllama(
            model=model,
            temperature=0.7,
            seed=seed,
            base_url=base_url,
        )
        self._structured = chat.with_structured_output(CaseBible)

    def generate_bible(self, system: str, user: str) -> CaseBible:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        result = self._structured.invoke(messages)
        return cast("CaseBible", result)
