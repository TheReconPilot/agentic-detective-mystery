from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer
from rich.console import Console

from mystery import __version__
from mystery.agents.suspect import respond_as_suspect
from mystery.case_gen.generator import generate_bible
from mystery.config import Settings
from mystery.evals.consistency import run_consistency_eval
from mystery.evals.llm_player import play_with_llm
from mystery.evals.solvability import run_solvability_eval
from mystery.graph.game import build_game_graph
from mystery.graph.router import ParseError, parse_action
from mystery.graph.state import GameState, initial_state
from mystery.models import CaseBible
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index, get_or_build_index
from mystery.rag.retriever import suspect_retriever

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

    from mystery.agents.commitments import CommitmentExtractor
    from mystery.case_gen.generator import BibleLLM
    from mystery.evals.consistency import ConsistencyJudge

app = typer.Typer(
    name="mystery",
    help="Agentic detective mystery game — interrogate suspects, examine clues, solve the case.",
    no_args_is_help=True,
)
console = Console()


def _default_llm_factory(seed: int, settings: Settings) -> BibleLLM:
    """Construct the production case-gen LLM. Tests override this attribute."""
    from mystery.case_gen.llm import OllamaBibleLLM

    return OllamaBibleLLM(
        model=settings.llm_model,
        seed=seed,
        base_url=settings.ollama_base_url,
    )


def _default_chat_model_factory(settings: Settings) -> BaseChatModel:
    """Construct the production chat model used by agents. Tests override."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        temperature=0.7,
        base_url=settings.ollama_base_url,
    )


def _default_embeddings_factory(settings: Settings) -> Embeddings:
    """Construct the production embeddings model. Tests override."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )


def _default_commitment_extractor_factory(settings: Settings) -> CommitmentExtractor:
    """Production extractor: 0-temp chat for stable structured-output extraction."""
    from langchain_ollama import ChatOllama

    from mystery.agents.commitments import LLMCommitmentExtractor

    extractor_chat = ChatOllama(
        model=settings.llm_model,
        temperature=0.0,
        base_url=settings.ollama_base_url,
    )
    return LLMCommitmentExtractor(extractor_chat)


def _default_judge_factory(settings: Settings) -> ConsistencyJudge:
    """Build the production consistency judge — uses a 0-temp chat for stable verdicts."""
    from langchain_ollama import ChatOllama

    from mystery.evals.consistency import LLMConsistencyJudge

    judge_chat = ChatOllama(
        model=settings.llm_model,
        temperature=0.0,
        base_url=settings.ollama_base_url,
    )
    return LLMConsistencyJudge(judge_chat)


def _load_bible(settings: Settings, seed: int) -> CaseBible:
    path = settings.cases_dir / f"{seed}.json"
    if not path.exists():
        raise typer.BadParameter(
            f"No case found at {path}. Generate one with: mystery new --seed {seed}",
            param_hint="--seed",
        )
    return CaseBible.model_validate_json(path.read_text(encoding="utf-8"))


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"mystery [bold cyan]{__version__}[/]")


@app.command()
def new(
    seed: int = typer.Option(..., help="Seed for case generation."),
) -> None:
    """Generate a new case bible and write it to the cases directory."""
    settings = Settings()
    llm = _default_llm_factory(seed, settings)

    console.print(f"generating case [cyan]seed={seed}[/] with [cyan]{settings.llm_model}[/]...")
    bible = generate_bible(seed, llm, max_attempts=settings.max_gen_attempts)

    out_path = settings.cases_dir / f"{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(bible.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"wrote [green]{out_path}[/]")


@app.command()
def interrogate(
    seed: int = typer.Option(..., help="Seed of the case to load."),
    suspect: str = typer.Option(..., help="Suspect id to interrogate."),
    question: str = typer.Argument(..., help="Your question to the suspect."),
) -> None:
    """Ask one question of one suspect (smoke surface for the suspect agent)."""
    settings = Settings()
    bible = _load_bible(settings, seed)

    suspect_obj = next((s for s in bible.suspects if s.id == suspect), None)
    if suspect_obj is None:
        ids = ", ".join(s.id for s in bible.suspects)
        raise typer.BadParameter(
            f"Unknown suspect {suspect!r}. Known: {ids}",
            param_hint="--suspect",
        )

    embeddings = _default_embeddings_factory(settings)
    index = build_index(build_chunks(bible), embeddings)
    retriever = suspect_retriever(index, suspect_id=suspect)
    chat = _default_chat_model_factory(settings)

    reply = respond_as_suspect(suspect_obj, retriever, chat, question=question)
    console.print(f"[bold]{suspect_obj.name}[/]: {reply}")


def _index_dir_for(settings: Settings, seed: int) -> Path:
    return settings.cases_dir / f"{seed}.chroma"


def _opening_blurb(bible: CaseBible) -> str:
    death_loc = next(loc for loc in bible.locations if loc.id == bible.victim.location_of_death_id)
    return (
        f"[bold]The case of {bible.victim.name}.[/] "
        f"The {bible.victim.role.lower()} was found dead in the {death_loc.name}. "
        f"You arrive to investigate. Type 'help' for commands.\n"
    )


def _play_turn(state: GameState, user_input: str) -> tuple[str, bool]:
    """Parse one input line. Return (error_message, should_dispatch). For tests."""
    parsed = parse_action(user_input)
    if isinstance(parsed, ParseError):
        return parsed.message, False
    state["pending_action"] = parsed
    return "", True


@app.command()
def play(
    seed: int = typer.Option(..., help="Seed of the case to play."),
) -> None:
    """Play the case end-to-end in a REPL."""
    settings = Settings()
    bible = _load_bible(settings, seed)
    embeddings = _default_embeddings_factory(settings)
    chat = _default_chat_model_factory(settings)
    vectorstore = get_or_build_index(bible, embeddings, _index_dir_for(settings, seed))
    commitment_extractor = _default_commitment_extractor_factory(settings)
    graph = build_game_graph(bible, vectorstore, chat, commitment_extractor)

    state = initial_state(bible)
    console.print(_opening_blurb(bible))

    while not state["done"]:
        console.print(
            f"[dim]({state['current_location_id']}, turn {state['turn_count']})[/]",
        )
        try:
            raw = input("> ")
        except EOFError:
            console.print("\nYou abandon the case.")
            return

        message, should_dispatch = _play_turn(state, raw.strip())
        if not should_dispatch:
            if message:
                console.print(f"[yellow]{message}[/]")
            continue

        state = cast("GameState", graph.invoke(state))
        if state["last_output"]:
            console.print(state["last_output"])

    console.print(f"\n[bold]Turns used: {state['turn_count']}.[/]")


def _load_bibles_from_dir(cases_dir: Path) -> list[CaseBible]:
    if not cases_dir.exists():
        return []
    bibles: list[CaseBible] = []
    for path in sorted(cases_dir.glob("*.json")):
        bibles.append(CaseBible.model_validate_json(path.read_text(encoding="utf-8")))
    return bibles


_DEFAULT_CASES_DIR = Path("evals/cases")


@app.command(name="eval")
def eval_cmd(
    cases_dir: Path = typer.Option(
        _DEFAULT_CASES_DIR,
        help="Directory containing case-bible JSON files.",
    ),
    consistency: bool = typer.Option(
        False,
        "--consistency",
        help="Additionally run the (slow) consistency eval with an LLM judge.",
    ),
) -> None:
    """Run the eval suite: solvability (always) and optionally consistency."""
    bibles = _load_bibles_from_dir(cases_dir)
    if not bibles:
        console.print(f"[red]No case JSON files found in {cases_dir}.[/]")
        raise typer.Exit(1)

    settings = Settings()
    console.print(f"Running solvability eval on [cyan]{len(bibles)}[/] cases...")

    solv = run_solvability_eval(
        bibles,
        embeddings_factory=lambda: _default_embeddings_factory(settings),
        chat_factory=lambda: _default_chat_model_factory(settings),
    )

    console.print(
        f"  successes: [bold]{solv.successes}/{solv.cases_run}[/] "
        f"([bold]{solv.success_rate:.0%}[/])",
    )
    if solv.mean_turns_on_success is not None:
        console.print(f"  mean turns on success: [bold]{solv.mean_turns_on_success:.1f}[/]")

    for r in solv.per_case:
        mark = "[green]✓[/]" if r.success else "[red]✗[/]"
        console.print(
            f"  {mark} seed={r.seed} turns={r.turns} "
            f"accused={r.accused or '(none)'} actual={r.actual_killer}",
        )

    if consistency:
        console.print("\nRunning consistency eval (this calls the LLM many times)...")
        judge = _default_judge_factory(settings)
        chat = _default_chat_model_factory(settings)

        total = 0
        contradicts = 0
        refused = 0
        for bible in bibles:
            embeddings = _default_embeddings_factory(settings)
            vectorstore = build_index(build_chunks(bible), embeddings)
            report = run_consistency_eval(bible, vectorstore, chat, judge)
            total += report.total
            contradicts += report.contradicts
            refused += report.refused
            console.print(
                f"  seed={bible.seed}: {report.consistent}/{report.total} consistent "
                f"({report.consistency_rate:.0%}), "
                f"{report.contradicts} contradicts, {report.refused} refused",
            )

        consistent_total = total - contradicts - refused
        rate = consistent_total / total if total else 0.0
        console.print(
            f"  [bold]overall: {consistent_total}/{total} consistent ({rate:.0%}), "
            f"{contradicts} contradicts, {refused} refused[/]",
        )


@app.command()
def playtest(
    seed: int = typer.Option(..., help="Seed of the case to playtest."),
    max_turns: int = typer.Option(60, help="Hard cap on detective turns."),
    transcript: bool = typer.Option(
        False,
        "--transcript",
        help="Print the per-turn transcript after the run.",
    ),
) -> None:
    """LLM-vs-LLM playtest: a blind detective LLM plays the case end-to-end."""
    settings = Settings()
    bible = _load_bible(settings, seed)
    embeddings = _default_embeddings_factory(settings)
    suspect_chat = _default_chat_model_factory(settings)
    detective_chat = _default_chat_model_factory(settings)
    vectorstore = get_or_build_index(bible, embeddings, _index_dir_for(settings, seed))

    console.print(f"playtesting [cyan]seed={seed}[/] for up to {max_turns} turns...")
    report = play_with_llm(
        bible,
        vectorstore,
        suspect_chat,
        detective_chat,
        max_turns=max_turns,
    )
    mark = "[green]✓[/]" if report.success else "[red]✗[/]"
    console.print(
        f"  {mark} accused={report.accused or '(none)'} actual={report.actual_killer} "
        f"turns={report.turns_used} parse_errors={report.parse_errors} "
        f"repeats={report.repeated_actions}",
    )
    if transcript:
        for step in report.steps:
            console.print(
                f"  t{step.turn} [bold]{step.parsed_kind}[/] {step.raw_command!r}",
            )
            if step.output:
                console.print(f"    [dim]{step.output.splitlines()[0][:140]}[/]")


if __name__ == "__main__":
    app()
