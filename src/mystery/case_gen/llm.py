"""Concrete BibleLLM backed by Ollama via langchain-ollama.

Kept separate from the generator so tests can avoid importing langchain.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from mystery.models import CaseBible

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def _strip_json_fence(text: str) -> str:
    """Remove optional ```json ... ``` fences some models wrap around JSON."""
    m = _JSON_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace all JSON Schema $ref pointers with their inline definitions.

    Ollama's grammar-based output constraint resolves only the top-level schema
    structure when $ref is used for nested objects; inner objects are left
    unconstrained and the model invents its own field names.  Inlining removes
    $defs and $ref entirely so every nested type is constrained correctly.
    """
    defs: dict[str, Any] = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:  # noqa: ANN401
        if isinstance(node, dict):
            if "$ref" in node:
                ref_key = node["$ref"].rsplit("/", 1)[-1]
                return _resolve(dict(defs[ref_key]))
            return {k: _resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    result: dict[str, Any] = _resolve(schema)
    return result


class OllamaBibleLLM:
    """Wraps ChatOllama with structured output bound to CaseBible."""

    def __init__(self, model: str, seed: int, base_url: str | None = None) -> None:
        chat = ChatOllama(
            model=model,
            temperature=0.7,
            seed=seed,
            base_url=base_url,
            # Disable thinking/reasoning mode: models like qwen3 output <think> tags
            # in main content when reasoning=None, which breaks JSON parsing.
            reasoning=False,
        )
        # Inline $defs so Ollama's grammar constraint covers nested schemas too.
        inline_schema = _inline_refs(CaseBible.model_json_schema())
        self._chat = chat.bind(format=inline_schema)

    def generate_bible(self, system: str, user: str) -> CaseBible:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        try:
            response = self._chat.invoke(messages)
            content = _strip_json_fence(str(response.content))
            return CaseBible.model_validate_json(content)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"LLM output parsing failed: {e}") from e
