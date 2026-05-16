from __future__ import annotations

import typer
from rich.console import Console

from mystery import __version__

app = typer.Typer(
    name="mystery",
    help="Agentic detective mystery game — interrogate suspects, examine clues, solve the case.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"mystery [bold cyan]{__version__}[/]")


@app.command()
def new(seed: int = typer.Option(..., help="Seed for case generation.")) -> None:
    """Generate a new case bible. (Stub — implemented in M2.)"""
    console.print(f"[yellow]TODO[/] generate case with seed={seed}")


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
