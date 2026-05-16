# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A local-first agentic detective mystery game. Player interrogates LLM-driven suspects to solve a procedurally generated murder. The architectural keystone is a private **case bible** generated up-front; every suspect agent answers via RAG over a character-scoped slice of it, so the agents cannot drift away from canonical truth.

Implementation roadmap, domain model, eval strategy, and milestones live in [PLAN.md](PLAN.md) — read it before doing non-trivial work.

## Stack

- Python 3.13, managed with `uv`
- LangGraph (state machine) + LangChain (`langchain-ollama` for chat & embeddings)
- Ollama for local LLMs — `qwen2.5:3b-instruct-q4_K_M` on 4 GB GPUs, `qwen2.5:14b-instruct-q4_K_M` or `llama3.1:8b` on 16 GB
- `nomic-embed-text` (Ollama) + Chroma for RAG
- Pydantic v2 for the case bible and all tool I/O
- Typer + Rich for the CLI

Models are selected via env vars (`MYSTERY_LLM_MODEL`, `MYSTERY_EMBED_MODEL`) so the same code runs on either hardware tier.

## Commands

```bash
# One-time: install Ollama, then pull models
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull nomic-embed-text

# Env setup
uv sync                          # install runtime + dev deps from pyproject.toml

# Run
uv run mystery new --seed 42     # generate a case bible
uv run mystery play              # play the most recent case
uv run mystery eval              # run the full eval suite (slow, real LLM)

# Quality gates (run before committing)
uv run ruff format .
uv run ruff check . --fix
uv run mypy src
uv run pytest                    # unit + integration, no real-LLM evals
uv run pytest -m eval            # opt-in: real-LLM evals (slow)
uv run pytest tests/unit/test_case_bible.py::test_killer_is_a_suspect   # single test
```

## Architecture (orientation)

```
src/mystery/
  models.py        ← CaseBible, Suspect, Clue, Location, GameState (pydantic)
  case_gen/        ← LLM-driven generator + post-gen invariant validation
  rag/             ← character-scoped Chroma retriever (this is the anti-drift mechanism)
  agents/          ← per-suspect sub-graph (retrieve → reason → apply deception policy)
  graph/           ← top-level LangGraph: router → action → update_state → check_win
  tools/           ← move / examine / notebook / accuse (LangChain tools)
  cli.py           ← typer entry point
```

Two ideas to keep in mind when editing:

1. **The case bible is canon.** Suspects can lie, but only according to their `deception_policy`. They cannot invent facts. If a suspect response could contradict the bible, it's a bug — covered by the consistency eval.
2. **RAG scope is per-character.** When implementing or modifying the retriever, never let suspect A retrieve suspect B's private chunks. Integration tests guard this; don't relax them.

## Conventions

- `ruff` (line length 100) and `mypy --strict` are CI gates — keep them green.
- New tools and agent I/O go through Pydantic models, not raw dicts.
- Real-LLM tests are marked `@pytest.mark.eval` and skipped by default. Default `pytest` must stay fast and offline (fake LLM via `langchain_core.language_models.fake`).
- Frozen eval cases live in `evals/cases/` as JSON — don't regenerate them casually; they are regression fixtures.
