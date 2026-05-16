# Project Plan — Agentic Detective Mystery

A text-based detective game where the player interrogates LLM-driven suspects to solve a procedurally generated murder. The key architectural idea: a **private "case bible"** is generated up-front and stored as a grounded document. Every suspect agent answers via RAG over a character-scoped slice of that bible, so long conversations cannot drift away from canonical truth.

This document is the implementation roadmap. Operational commands live in [CLAUDE.md](CLAUDE.md).

## Goals

- **Portfolio-quality**: a small, well-tested codebase that demonstrates LangChain + LangGraph + RAG + tool use end-to-end.
- **Runs locally** on consumer GPUs (4 GB laptop minimum, 16 GB workstation comfortably) using open-source models via Ollama.
- **Honest agents**: suspects can lie, but only in ways the case bible sanctions; they cannot hallucinate facts that contradict the bible.
- **Measurable**: every claim in the README ("agents don't drift", "cases are solvable") is backed by an eval that produces a number.

## Non-goals

- Multiplayer, web UI, voice, image generation. CLI only, at least for v1.
- Cloud LLMs. The whole point is local-first.
- Open-ended sandbox: each case has a finite suspect list, location list, and clue set generated up front.

## Tech stack

| Concern              | Choice                                                                                                   |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| Language             | Python 3.13                                                                                              |
| Package / env        | `uv`                                                                                                     |
| LLM runtime          | [Ollama](https://ollama.com/) (local HTTP API)                                                           |
| LLM (4 GB GPU)       | `qwen2.5:3b-instruct-q4_K_M` or `llama3.2:3b-instruct-q4_K_M` (~2 GB VRAM)                               |
| LLM (16 GB GPU)      | `qwen2.5:14b-instruct-q4_K_M` or `llama3.1:8b-instruct-q4_K_M`                                           |
| Orchestration        | `langgraph` (state machine) + `langchain-core` (messages, prompts) + `langchain-ollama` (chat/embedding) |
| Embeddings           | `nomic-embed-text` via Ollama (CPU-friendly, 768-dim)                                                    |
| Vector store         | `chromadb` (local, persistent, no server)                                                                |
| Schemas / validation | `pydantic` v2                                                                                            |
| CLI                  | `typer` + `rich` for pretty output                                                                       |
| Tests                | `pytest`, `pytest-cov`, `pytest-asyncio`                                                                 |
| Lint + format        | `ruff` (replaces black, isort, flake8)                                                                   |
| Types                | `mypy --strict` (or `pyright` — pick one and stick with it)                                              |
| Pre-commit           | `pre-commit` with ruff + mypy hooks                                                                      |
| CI                   | GitHub Actions: lint, type-check, test, eval-smoke on every PR                                           |

Model swapping is via env var (`MYSTERY_LLM_MODEL`, `MYSTERY_EMBED_MODEL`) so the same code runs on both hardware tiers.

## Repository layout

```
src/mystery/
  __init__.py
  config.py              # pydantic-settings: model names, paths, temperatures
  models.py              # Pydantic schemas: CaseBible, Suspect, Clue, Location, Action, GameState
  case_gen/
    generator.py         # builds a CaseBible from a seed
    prompts.py           # generator prompts (themes, archetypes, deception policies)
    validate.py          # post-gen sanity: exactly one killer, motive resolves, timeline coherent
  rag/
    indexer.py           # chunks the bible per-character + per-location, embeds into Chroma
    retriever.py         # character-scoped retriever wrappers
  agents/
    suspect.py           # per-suspect sub-graph: retrieve → reason → apply deception policy → respond
    narrator.py          # describes locations, examined evidence (no deception)
  graph/
    game.py              # top-level LangGraph: route → act → update_state → check_win
    state.py             # GameState TypedDict / Pydantic
    router.py            # classifies player input → action enum
  tools/
    notebook.py          # player-facing clue notebook (append, list)
    move.py              # location graph navigation
    examine.py           # surfaces clue text for the current location
    accuse.py            # terminal: resolves accusation against the bible
  cli.py                 # typer entry point: `mystery new`, `mystery play`, `mystery eval`
tests/
  unit/                  # pure functions, schema validation, state transitions
  integration/           # fake-LLM end-to-end runs (langchain_core.language_models.fake)
  eval/                  # real-LLM evals, marked @pytest.mark.eval (skipped by default)
evals/
  cases/                 # frozen seeded CaseBibles for regression
  judge_prompts/
  optimal_player.py      # heuristic agent that asks the right questions
  consistency.py         # LLM-judge: did any suspect contradict the bible?
  solvability.py         # can optimal_player accuse correctly within N turns?
```

## Domain model (the heart of the project)

The `CaseBible` is the single source of truth. Everything else is derived from it.

```python
class Alibi(BaseModel):
    location_id: str
    time_window: tuple[int, int]   # minutes from case t0
    is_true: bool                  # the bible knows; the suspect may not tell
    corroborating_witness_id: str | None

class Suspect(BaseModel):
    id: str
    name: str
    archetype: str                 # "butler", "estranged-sibling", ...
    motive: str | None             # None for innocents who still look guilty
    alibis: list[Alibi]
    knowledge: list[str]           # what this character truthfully knows
    deception_policy: str          # natural-language rules: "lies about being in the garden"

class Clue(BaseModel):
    id: str
    location_id: str
    description: str
    incriminates_suspect_ids: list[str]   # may include red herrings
    revealed: bool = False                # mutated by GameState, not bible

class CaseBible(BaseModel):
    seed: int
    victim: Victim
    suspects: list[Suspect]
    locations: list[Location]
    clues: list[Clue]
    killer_id: str
    canonical_timeline: list[TimelineEvent]
```

`validate.py` enforces invariants post-generation: killer is one of the suspects, every alibi references a real location, the killer's alibi `is_true=False`, every clue location exists, etc. Generation retries up to N times on failure.

## LangGraph topology

Top-level loop is a small state machine. Each suspect interrogation is its own sub-graph compiled once per suspect.

```
                     ┌─────────────┐
   player input ───▶│   router    │
                     └──┬──┬──┬──┬─┘
                        │  │  │  │
              ┌─────────┘  │  │  └──────────┐
              ▼            ▼  ▼             ▼
        interrogate     examine  move   accuse (terminal)
              │            │       │        │
              ▼            ▼       ▼        ▼
        suspect_subgraph  tool   tool    resolve → END
              │
              └──▶ update_state ──▶ check_win ──▶ next turn
```

The suspect sub-graph: `retrieve(character_scope) → reason(persona + retrieved facts + deception_policy) → response`. The retriever is filtered to that character's chunks plus shared world knowledge; this is what prevents cross-contamination between suspects.

## Evaluation strategy

Three named evals, each producing a number, runnable via `uv run pytest -m eval` or `uv run mystery eval`:

1. **Consistency** — for each frozen case, run a scripted interrogation that asks every suspect about every clue and alibi. An LLM-judge with the full bible visible scores each suspect response as `consistent | contradicts-bible | refuses`. Target: ≥ 95% consistent on the 8B model, ≥ 90% on the 3B model.
2. **Solvability** — an "optimal player" heuristic agent visits every location, examines every clue, interrogates every suspect, and accuses. Run on 20 generated cases. Target: ≥ 90% correct accusations. Cases that fail are saved as regression fixtures.
3. **Difficulty** — mean turns-to-solve by the optimal player. Tracked over time to detect generator regressions (cases getting trivially easy or unsolvable).

Cheaper smoke evals run in CI (3 cases, fake LLM where possible). The full eval suite runs locally on the workstation.

## Dev tooling configuration

Add to `pyproject.toml`:

```toml
[dependency-groups]
dev = ["ruff", "mypy", "pytest", "pytest-cov", "pytest-asyncio", "pre-commit"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "RUF", "ANN", "TCH"]
ignore = ["ANN101", "ANN102"]

[tool.mypy]
strict = true
python_version = "3.13"

[tool.pytest.ini_options]
addopts = "-ra --strict-markers --cov=src/mystery"
markers = ["eval: real-LLM evals, slow, skipped by default"]
testpaths = ["tests"]
```

CI matrix runs lint → type-check → unit+integration tests. The `eval` marker is opt-in (`-m eval`) so the default run stays fast.

## Milestones

Each milestone ends with green tests and a runnable demo.

- **M1 — Skeleton** Project layout, `uv sync`, ruff + mypy + pytest wired, CI green on an empty test. Typer CLI prints help.
- **M2 — Case bible** `CaseBible` schemas + validators, deterministic generator behind a seed (real LLM, but stubbable). Unit tests cover invariants. `mystery new --seed 42` writes a bible JSON.
- **M3 — RAG layer** Chunking strategy (per-suspect + per-location), Chroma persistence, character-scoped retriever. Integration test: querying as suspect A never returns suspect B's private knowledge chunks.
- **M4 — Suspect agent** Single suspect sub-graph end-to-end with a fake LLM. Then with Ollama. Manual smoke: interrogate one suspect about their alibi.
- **M5 — Game loop** Router, tools (move/examine/notebook/accuse), full LangGraph, win condition. Playable end-to-end in the CLI.
- **M6 — Evals** Optimal-player agent, consistency judge, solvability harness, 20 frozen cases. README publishes the numbers.
- **M7 — Polish** README with architecture diagram, recorded demo (asciinema), badges (CI, coverage), tagged v0.1.0 release.

## Open questions

- **Streaming vs. batch suspect responses?** Streaming feels better in the CLI but complicates the LLM-judge. Probably batch in v1, stream later.
- **Memory across turns within one interrogation?** Suspects should remember what they've already told the player this session, or they'll repeat themselves. Likely a short rolling buffer per suspect, stored in `GameState`, kept out of the RAG retrieval scope.
- **3B model quality floor.** May need few-shot examples in the suspect prompt to keep the small model in character. Decide after M4 smoke tests.
