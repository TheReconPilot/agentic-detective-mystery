# agentic-detective-mystery

A local-first text adventure where you interrogate LLM-driven suspects to solve a procedurally generated murder. The architectural keystone is a private **case bible** — victim, suspects, real killer, motives, true and false alibis, physical clues, timeline — generated up-front and never shown to the player. Every suspect agent answers via RAG over a *character-scoped* slice of that bible, so long conversations cannot drift away from canonical truth. Suspects may lie within an explicit `deception_policy`, but they cannot invent facts.

Built on **LangGraph + LangChain + Chroma**, running entirely on **local Ollama models** (3B on a 4 GB laptop GPU, 8B–14B on a 16 GB workstation). Designed as a portfolio project: every claim it makes is backed by a test or an eval that produces a number.

> **Status:** Milestones M1–M6 are complete — schemas, generator, RAG, suspect agent, the full playable game loop, and the eval harness (solvability + consistency). M7 (polish & v0.1.0) remains. See [PLAN.md](PLAN.md) for the full roadmap.

## Why this design

LLM narrative agents drift. They forget, contradict themselves, and confabulate. The usual workarounds (longer context, summarisation) are band-aids. This project tries a different one: **ground every response in a retrieval over a frozen, structured truth document**. The agent's job becomes "retrieve what your character knows, then phrase it according to your deception policy" — not "imagine a character from scratch every turn".

The case bible enforces three properties:

1. **Consistency** — a suspect can't contradict the bible, because their retrievable knowledge *is* the bible.
2. **Solvability** — generated invariants (`validate_bible`) guarantee at least one clue incriminates the killer and the killer's alibi is provably false.
3. **Privacy** — `suspect_retriever` filters on metadata so suspect A can never retrieve suspect B's private chunks. The integration test probes this adversarially.

## Quickstart

```bash
# Prerequisites: Python 3.13, uv, Ollama
ollama pull qwen2.5:3b-instruct-q4_K_M     # or qwen2.5:14b-instruct-q4_K_M on a 16 GB GPU
ollama pull nomic-embed-text

# Install
uv sync

# Generate a case (writes cases/42.json)
uv run mystery new --seed 42

# Interrogate one suspect (smoke surface for the agent)
uv run mystery interrogate --seed 42 --suspect butler "Where were you at nine o'clock?"

# Play the full game (REPL)
uv run mystery play --seed 42
```

In the REPL, type `help` to see the command list. The first run for a given seed embeds the case bible into a persistent Chroma index at `cases/{seed}.chroma`; subsequent plays load it instantly.

### Running the eval suite

Drop generated case bibles into `evals/cases/` (e.g. `for s in $(seq 1 20); do uv run mystery new --seed $s; cp cases/$s.json evals/cases/; done`), then:

```bash
uv run mystery eval                              # solvability: does an optimal player win?
uv run mystery eval --consistency                # also: do suspects contradict the bible?
```

**Solvability** uses an omniscient optimal player ([src/mystery/evals/optimal_player.py](src/mystery/evals/optimal_player.py)) that DFS-walks every location, examines each, then accuses the bible's killer. A failure means the generator produced an unsolvable case.

**Consistency** interrogates every suspect with a standard question set and hands each response to an LLM judge that sees the full bible. The judge classifies as `consistent`, `contradicts`, or `refused`. Aggregate "contradicts" rate is the headline metric this project exists to drive toward zero.

### Switching hardware tiers

Models are env-vared, so the same code runs on either:

```bash
# 4 GB laptop GPU
export MYSTERY_LLM_MODEL=qwen2.5:3b-instruct-q4_K_M

# 16 GB workstation
export MYSTERY_LLM_MODEL=qwen2.5:14b-instruct-q4_K_M
```

## Development

```bash
uv run ruff check . && uv run ruff format .
uv run mypy src tests
uv run pytest                   # unit + integration, fully offline
uv run pytest -m eval           # opt-in: real-LLM evals (M6)
```

Pre-commit hooks (`ruff-check`, `ruff-format`) run on every commit. Tests use `DeterministicFakeEmbedding` and `FakeListChatModel` from `langchain-core` so the default suite needs no network or GPU.

## Layout

See [CLAUDE.md](CLAUDE.md) for an annotated tree and the key invariants the codebase enforces. See [PLAN.md](PLAN.md) for the implementation roadmap, the eval strategy, and milestone status.

## License

TBD.
