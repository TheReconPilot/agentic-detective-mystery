"""Runtime configuration sourced from environment variables and an optional .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All knobs are overridable via env vars prefixed ``MYSTERY_``.

    The two models switch the same code between the 4 GB laptop tier and the
    16 GB workstation tier without touching code.
    """

    model_config = SettingsConfigDict(env_prefix="MYSTERY_", env_file=".env", extra="ignore")

    llm_model: str = "qwen2.5:3b-instruct-q4_K_M"
    embed_model: str = "nomic-embed-text"
    ollama_base_url: str | None = None
    cases_dir: Path = Field(default=Path("cases"))
    max_gen_attempts: int = 5
