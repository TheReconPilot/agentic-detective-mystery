# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A local-first agentic detective mystery game. Player interrogates LLM-driven suspects to solve a procedurally generated murder. The architectural keystone is a private **case bible** generated up-front; every suspect agent answers via RAG over a character-scoped slice of it, so the agents cannot drift away from canonical truth.

Implementation roadmap, domain model, eval strategy, and milestones live in [PLAN.md](PLAN.md) — read it before doing non-trivial work.

**Progress so far:** M1–M6 done (skeleton, case-bible generator, character-scoped RAG, suspect agent, game loop, eval harness). Post-v0.1: M8 (per-suspect voice) and M9 (commitment-based suspect memory) are done. M7 (polish: architecture diagram, demo recording, badges, v0.1.0 tag) is what's left.

## Stack

- Python 3.13, managed with `uv`
- LangGraph (state machine, used from M5 on) + LangChain (`langchain-ollama` for chat & embeddings, `langchain-chroma` for the vector store)
- Ollama for local LLMs — `qwen2.5:3b-instruct-q4_K_M` on 4 GB GPUs, `qwen2.5:14b-instruct-q4_K_M` or `llama3.1:8b` on 16 GB
- `nomic-embed-text` (Ollama) + Chroma for RAG
- Pydantic v2 for the case bible and all tool I/O
- Typer + Rich for the CLI

Models are selected via env vars (`MYSTERY_LLM_MODEL`, `MYSTERY_EMBED_MODEL`, `MYSTERY_OLLAMA_BASE_URL`, `MYSTERY_CASES_DIR`, `MYSTERY_MAX_GEN_ATTEMPTS`) so the same code runs on either hardware tier.

## Commands

```bash
# One-time: install Ollama, then pull models
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull nomic-embed-text

# Env setup
uv sync                          # install runtime + dev deps from pyproject.toml

# Run
uv run mystery new --seed 42                                       # generate cases/42.json
uv run mystery interrogate --seed 42 --suspect butler "Where were you at nine?"
uv run mystery play --seed 42    # play the full REPL
uv run mystery eval --cases-dir evals/cases                # solvability across the cases dir
uv run mystery eval --cases-dir evals/cases --consistency  # adds the LLM-judge pass (slow)
uv run mystery playtest --seed 42 --transcript             # LLM-vs-LLM blind playtest

# Quality gates (run before committing)
uv run ruff format .
uv run ruff check . --fix
uv run mypy src tests
uv run pytest                    # unit + integration, no real-LLM evals
uv run pytest -m eval            # opt-in: real-LLM evals (slow)
uv run pytest tests/unit/test_validate.py::test_rejects_killer_not_in_suspects   # single test
```

Pre-commit hooks (ruff check + ruff format) are installed via `uv run pre-commit install` and fire on every commit.

## Architecture (orientation)

```
src/mystery/
  models.py        ← CaseBible, Suspect, Clue, Location (pydantic v2, extras forbidden)
  config.py        ← pydantic-settings, env-prefixed MYSTERY_*
  case_gen/
    generator.py   ← BibleLLM Protocol + retry loop over validate_bible()
    validate.py    ← eight semantic invariants (killer is a suspect, alibis resolve, etc.)
    prompts.py     ← system + user prompts for case generation
    llm.py         ← OllamaBibleLLM (ChatOllama.with_structured_output(CaseBible))
  rag/
    chunks.py      ← bible → typed Chunks with scope/character_id metadata
    indexer.py     ← chunks → Chroma store; get_or_build_index() for persistence
    retriever.py   ← suspect_retriever() with character-scope metadata filter
  agents/
    suspect.py     ← respond_as_suspect: retrieve → render persona prompt → chat → string
  graph/
    state.py       ← GameState TypedDict, Action discriminated union, initial_state()
    router.py      ← parse raw input → typed Action | ParseError
    game.py        ← build_game_graph(): LangGraph dispatcher, one node per Action kind
  tools/           ← apply_move / apply_examine / apply_notebook / apply_accuse / apply_interrogate
                     pure functions returning state-update dicts (no LangGraph dep)
  evals/
    optimal_player.py  ← DFS-based player that always wins; produces SolvabilityReport
    solvability.py     ← aggregator across many bibles
    consistency.py     ← ConsistencyJudge Protocol + LLMConsistencyJudge + run_consistency_eval
    llm_player.py      ← blind LLM detective: render observation, parse one command, loop
  cli.py           ← typer entry point (new, interrogate, play, eval, playtest, version)
tests/
  unit/            ← pure functions, schema validation, CLI with stubbed factories
  integration/     ← real Chroma + DeterministicFakeEmbedding + FakeListChatModel
```

Six ideas to keep in mind when editing:

1. **The case bible is canon.** Suspects can lie, but only according to their `deception_policy`. They cannot invent facts. If a suspect response could contradict the bible, it's a bug — covered by the consistency eval (M6).
2. **RAG scope is per-character.** `suspect_retriever` filters on `{"$or": [{"character_id": id}, {"scope": "world"}]}`. Tests in [tests/integration/test_rag_scope_isolation.py](tests/integration/test_rag_scope_isolation.py) prove that A cannot retrieve B's private chunks. Don't relax those tests.
3. **Three things are deliberately not in RAG:** the `canonical_timeline` (author's omniscient view), `deception_policy` (lives in the suspect prompt, not retrieval), and `clues` (the `examine` tool reads them from the bible directly in M5). [tests/unit/test_chunks.py](tests/unit/test_chunks.py) guards these exclusions.
4. **External services are injected via module-level factories.** `cli._default_llm_factory`, `cli._default_chat_model_factory`, `cli._default_embeddings_factory` — tests `monkeypatch.setattr` them to inject stubs. Keep this pattern when adding new services; don't import langchain at the top of `cli.py` for things only used inside a single command.
5. **Real-LLM evals are pytest-marked `eval`** and skipped by default. Default `pytest` must stay fast and offline. Integration tests use `DeterministicFakeEmbedding` + `FakeListChatModel` from `langchain_core` — no network.
6. **Game tools are pure functions, not LangGraph nodes.** Each `apply_*` in [src/mystery/tools/](src/mystery/tools/) takes state + bible (+ optional deps) and returns a state-update dict. LangGraph's contribution is only the dispatch topology in [graph/game.py](src/mystery/graph/game.py). If you need to test new game logic, do it against the `apply_*` function directly — don't reach for the graph unless you're testing routing.

## Conventions

- `ruff` (line length 100) and `mypy --strict` are gates — keep them green.
- New tools and agent I/O go through Pydantic models, not raw dicts.
- All persisted data (case bibles, future Chroma indexes) goes under directories named in `Settings` — never hard-code paths.
- Frozen eval cases will live in `evals/cases/` as JSON — don't regenerate them casually; they are regression fixtures.
