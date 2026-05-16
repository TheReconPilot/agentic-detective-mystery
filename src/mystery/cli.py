from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console

from mystery import __version__
from mystery.case_gen.generator import generate_bible
from mystery.config import Settings

if TYPE_CHECKING:
    from mystery.case_gen.generator import BibleLLM

app = typer.Typer(
    name="mystery",
    help="Agentic detective mystery game — interrogate suspects, examine clues, solve the case.",
    no_args_is_help=True,
)
console = Console()


def _default_llm_factory(seed: int, settings: Settings) -> BibleLLM:
    """Construct the production LLM. Imported lazily to keep CLI startup fast.

    Tests override this attribute on the module to inject a stub.
    """
    from mystery.case_gen.llm import OllamaBibleLLM

    return OllamaBibleLLM(
        model=settings.llm_model,
        seed=seed,
        base_url=settings.ollama_base_url,
    )


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
def play() -> None:
    """Play the most recent case. (Stub — implemented in M5.)"""
    console.print("[yellow]TODO[/] play loop")


@app.command(name="eval")
def eval_cmd() -> None:
    """Run the eval suite. (Stub — implemented in M6.)"""
    console.print("[yellow]TODO[/] eval suite")


if __name__ == "__main__":
    app()
