from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console

from mystery import __version__
from mystery.agents.suspect import respond_as_suspect
from mystery.case_gen.generator import generate_bible
from mystery.config import Settings
from mystery.models import CaseBible
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index
from mystery.rag.retriever import suspect_retriever

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

    from mystery.case_gen.generator import BibleLLM

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


@app.command()
def play() -> None:
    """Play the most recent case. (Stub — implemented in M5.)"""
    console.print("[yellow]TODO[/] play loop")


@app.command(name="eval")
def eval_cmd() -> None:
    """Run the eval suite. (Stub — implemented in M6.)"""
    console.print("[yellow]TODO[/] eval suite")


if __name__ == "__main__":
    app()
